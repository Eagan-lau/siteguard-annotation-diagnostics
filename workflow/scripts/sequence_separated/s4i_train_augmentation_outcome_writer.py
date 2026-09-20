"""Stream TRAIN recorded-concordance labels onto frozen S4H augmentation."""
import argparse
import hashlib
import json
import traceback
from collections import Counter, defaultdict
from pathlib import Path


ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")
PROTEIN_COLUMNS = ["protein_id", "node_id", "component_id", "role"]
LIBRARY_COLUMNS = ["reference_node", "reference_component_id", "reference_protein_id", "reference_activity_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier", "reference_role", "activity_provenance"]
AUGMENTATION_COLUMNS = ["query_protein_id", "query_node", "query_component_id", "query_role", "reference_node", "reference_component_id", "reference_protein_id", "reference_activity_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier", "candidate_origin", "augmentation_mechanisms", "maximum_mechanism_inclusion_probability", "augmentation_probability_is_population_weight", "query_truth_provenance", "outcome_semantics"]
AUGMENTATION_STATUS_COLUMNS = ["query_protein_id", "query_node", "query_component_id", "query_role", "status", "documented_train_activities", "selected_pairs", "selected_mechanism_occurrences"]
OUTCOME_COLUMNS = ["query_documented_activity_count", "query_documented_ec_l3_count", "query_documented_ec_l4_count", "query_documented_rhea_count", "observed_same_ec_l3", "observed_same_ec_l4", "exact_rhea_outcome_evaluable", "observed_same_exact_rhea", "deepest_shared_recorded_ec_level", "outcome_truth_provenance", "ground_truth_only"]
LABELED_COLUMNS = AUGMENTATION_COLUMNS + OUTCOME_COLUMNS
STATUS_COLUMNS = AUGMENTATION_STATUS_COLUMNS + ["labeled_pairs", "outcome_join_status"]


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def emit(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")


def schemas():
    import pyarrow as pa
    strings = {"query_protein_id", "query_role", "reference_protein_id", "reference_activity_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier", "candidate_origin", "augmentation_mechanisms", "query_truth_provenance", "outcome_semantics", "outcome_truth_provenance"}
    ints = {"query_node", "query_component_id", "reference_node", "reference_component_id", "query_documented_activity_count", "query_documented_ec_l3_count", "query_documented_ec_l4_count", "query_documented_rhea_count"}
    bools = {"augmentation_probability_is_population_weight", "observed_same_ec_l3", "observed_same_ec_l4", "exact_rhea_outcome_evaluable", "observed_same_exact_rhea", "ground_truth_only"}
    types = {name: pa.string() for name in strings}; types.update({name: pa.int32() for name in ints}); types.update({name: pa.bool_() for name in bools})
    types["maximum_mechanism_inclusion_probability"] = pa.float64(); types["deepest_shared_recorded_ec_level"] = pa.int8()
    labeled = pa.schema([(name, types[name]) for name in LABELED_COLUMNS])
    status = pa.schema([("query_protein_id", pa.string()), ("query_node", pa.int32()), ("query_component_id", pa.int32()), ("query_role", pa.string()), ("status", pa.string()), ("documented_train_activities", pa.int32()), ("selected_pairs", pa.int32()), ("selected_mechanism_occurrences", pa.int32()), ("labeled_pairs", pa.int32()), ("outcome_join_status", pa.string())])
    return labeled, status


class Sink:
    def __init__(self, path, schema, batch_rows):
        import pyarrow.parquet as pq
        require(type(batch_rows) is int and not isinstance(batch_rows, bool) and batch_rows > 0, "batch rows")
        self.schema, self.limit, self.rows, self.count = schema, batch_rows, [], 0; self.writer = pq.ParquetWriter(path, schema, compression="zstd")
    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.limit: self.flush()
    def flush(self):
        if self.rows:
            import pyarrow as pa
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema)); self.count += len(self.rows); self.rows = []
    def close(self): self.flush(); self.writer.close()


def load_proteins(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path); require(table.column_names == PROTEIN_COLUMNS, "protein ledger schema")
    proteins = {}
    for row in table.to_pylist():
        require(row["protein_id"] not in proteins and row["role"] in ROLES, "protein identity"); proteins[row["protein_id"]] = row
    require(proteins and set(row["role"] for row in proteins.values()) == set(ROLES), "protein role universe")
    return proteins


def load_truth(path, proteins):
    import pyarrow.parquet as pq
    table = pq.read_table(path); require(table.column_names == LIBRARY_COLUMNS, "TRAIN activity library schema")
    truth = defaultdict(lambda: {"activities": set(), "ec_l3": set(), "ec_l4": set(), "rhea": set()}); activities = {}; previous = None
    for row in table.to_pylist():
        key = row["reference_node"], row["reference_protein_id"], row["reference_activity_id"]
        require(previous is None or key > previous, "TRAIN activity library order"); previous = key
        protein, activity = row["reference_protein_id"], row["reference_activity_id"]; meta = proteins.get(protein)
        require(meta is not None and meta["role"] == "TRAIN" and activity not in activities, "TRAIN activity identity")
        require(row["reference_node"] == meta["node_id"] and row["reference_component_id"] == meta["component_id"], "TRAIN activity mapping")
        require(row["reference_role"] == "TRAIN" and row["activity_provenance"] == "TRAIN_REFERENCE_SOURCE_ACTIVITY_PRESERVED", "TRAIN activity provenance")
        canonical = row["canonical_ec"]; parts = canonical.split(".") if isinstance(canonical, str) else []
        require(len(parts) == 4 and row["ec_l1"] == parts[0] and row["ec_l2"] == ".".join(parts[:2]) and row["ec_l3"] == ".".join(parts[:3]) and row["ec_l4"] == canonical, "EC hierarchy")
        require(row["canonical_rhea"] is None or (isinstance(row["canonical_rhea"], str) and row["canonical_rhea"].startswith("RHEA:")), "canonical Rhea")
        activities[activity] = row; target = truth[protein]; target["activities"].add(activity); target["ec_l3"].add(row["ec_l3"]); target["ec_l4"].add(row["ec_l4"])
        if row["canonical_rhea"] is not None: target["rhea"].add(row["canonical_rhea"])
    require(activities, "empty TRAIN activity library")
    return truth, activities


def label(row, truth):
    same_l3, same_l4 = row["ec_l3"] in truth["ec_l3"], row["ec_l4"] in truth["ec_l4"]
    evaluable = row["canonical_rhea"] is not None and bool(truth["rhea"]); same_rhea = evaluable and row["canonical_rhea"] in truth["rhea"]
    return {**row, "query_documented_activity_count": len(truth["activities"]), "query_documented_ec_l3_count": len(truth["ec_l3"]), "query_documented_ec_l4_count": len(truth["ec_l4"]), "query_documented_rhea_count": len(truth["rhea"]), "observed_same_ec_l3": same_l3, "observed_same_ec_l4": same_l4, "exact_rhea_outcome_evaluable": evaluable, "observed_same_exact_rhea": same_rhea if evaluable else None, "deepest_shared_recorded_ec_level": 4 if same_l4 else (3 if same_l3 else 0), "outcome_truth_provenance": "TRAIN_QUERY_OUTCOME_ONLY_JOINED_AFTER_AUGMENTATION_FREEZE", "ground_truth_only": True}


def execute(protein_ledger, augmentation_pairs, augmentation_status, train_activity_library, output, batch_rows=100000):
    paths = {"protein_ledger": Path(protein_ledger).resolve(), "augmentation_pairs": Path(augmentation_pairs).resolve(), "augmentation_status": Path(augmentation_status).resolve(), "train_activity_library": Path(train_activity_library).resolve()}; output = Path(output).resolve()
    require(all(path.is_file() for path in paths.values()), "S4I inputs"); require(output.parent.is_dir() and not output.exists(), "exclusive output"); output.mkdir(exist_ok=False)
    inputs = {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}; emit(output / "reservation.json", {"status": "ONE_S4I_TRAIN_AUGMENTATION_OUTCOME_ATTEMPT_RESERVED", "inputs": inputs, "automatic_retry": False})
    pair_sink = status_sink = None; state, error = "FAIL_CLOSED", None
    try:
        import pyarrow.parquet as pq
        proteins = load_proteins(paths["protein_ledger"]); truth, activities = load_truth(paths["train_activity_library"], proteins)
        pair_schema, status_schema = schemas(); pair_sink = Sink(output / "train_labeled_augmentation_pairs.parquet", pair_schema, batch_rows); status_sink = Sink(output / "train_augmentation_outcome_status.parquet", status_schema, batch_rows)
        parquet = pq.ParquetFile(paths["augmentation_pairs"]); require(parquet.schema_arrow.names == AUGMENTATION_COLUMNS, "augmentation pair schema")
        labeled_counts, positives, previous = Counter(), Counter(), None
        for batch in parquet.iter_batches(batch_size=batch_rows):
            for row in batch.to_pylist():
                key = row["query_protein_id"], row["reference_protein_id"], row["reference_activity_id"]
                require(previous is None or key > previous, "augmentation pair order"); previous = key
                query = proteins.get(row["query_protein_id"]); reference = activities.get(row["reference_activity_id"])
                require(query is not None and query["role"] == "TRAIN" and row["query_role"] == "TRAIN" and row["query_node"] == query["node_id"] and row["query_component_id"] == query["component_id"], "TRAIN query identity")
                require(reference is not None and all(row[name] == reference[name] for name in ("reference_node", "reference_component_id", "reference_protein_id", "reference_activity_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier")), "augmentation/reference activity identity")
                require(row["reference_node"] != row["query_node"] and row["candidate_origin"] == "SUPERVISED_TRAIN_AUGMENTATION_NOT_DEPLOYMENT_CANDIDATE", "augmentation provenance")
                require(row["query_truth_provenance"] == "TRAIN_SAMPLING_ONLY_NOT_MODEL_INPUT" and row["outcome_semantics"] == "DOCUMENTED_CONCORDANCE_NOT_BIOCHEMICAL_NEGATIVE" and not row["augmentation_probability_is_population_weight"], "augmentation semantics")
                qtruth = truth.get(row["query_protein_id"]); require(qtruth and qtruth["activities"], "TRAIN query truth")
                labeled = label(row, qtruth); pair_sink.add(labeled); labeled_counts[row["query_protein_id"]] += 1
                positives["ec_l3"] += int(labeled["observed_same_ec_l3"]); positives["ec_l4"] += int(labeled["observed_same_ec_l4"]); positives["rhea_evaluable"] += int(labeled["exact_rhea_outcome_evaluable"]); positives["exact_rhea"] += int(labeled["observed_same_exact_rhea"] is True)
        status_table = pq.read_table(paths["augmentation_status"]); require(status_table.column_names == AUGMENTATION_STATUS_COLUMNS, "augmentation status schema")
        seen_status, status_counts = set(), Counter()
        for row in status_table.to_pylist():
            query = proteins.get(row["query_protein_id"]); require(query is not None and query["role"] == "TRAIN" and row["query_protein_id"] not in seen_status, "augmentation status identity"); seen_status.add(row["query_protein_id"])
            require(row["query_node"] == query["node_id"] and row["query_component_id"] == query["component_id"] and row["query_role"] == "TRAIN", "augmentation status mapping")
            require(row["selected_pairs"] == labeled_counts[row["query_protein_id"]], "augmentation/status pair census")
            join_status = "TRAIN_AUGMENTATION_OUTCOMES_JOINED" if row["selected_pairs"] else "NO_AUGMENTATION_PAIR_TO_LABEL"
            status_counts[join_status] += 1; status_sink.add({**row, "labeled_pairs": labeled_counts[row["query_protein_id"]], "outcome_join_status": join_status})
        train = {protein for protein, row in proteins.items() if row["role"] == "TRAIN"}; require(seen_status == train, "augmentation status TRAIN universe")
        pair_sink.close(); status_sink.close(); require(pair_sink.count == sum(labeled_counts.values()) and status_sink.count == len(train), "S4I writer census")
        summary = {"status": "PASS_S4I_TRAIN_AUGMENTATION_OUTCOME_PENDING_INDEPENDENT_AUDIT", "inputs": inputs, "train_proteins": len(train), "train_activity_library_rows": len(activities), "augmentation_input_rows": pair_sink.count, "labeled_pair_rows": pair_sink.count, "status_counts": {name: status_counts[name] for name in ("TRAIN_AUGMENTATION_OUTCOMES_JOINED", "NO_AUGMENTATION_PAIR_TO_LABEL")}, "positive_counts": {name: positives[name] for name in ("ec_l3", "ec_l4", "rhea_evaluable", "exact_rhea")}, "outcome_truth_scope": "TRAIN_ONLY_AFTER_AUGMENTATION_FREEZE", "rhea_is_independent_target_not_ec_level": True, "different_rhea_is_biochemical_negative": False, "candidate_membership_changed": False, "nontrain_truth_read": False, "feature_matrix_created": False, "training_started": False}
        emit(output / "producer_summary.json", summary); state = summary["status"]
    except BaseException as exc:
        for sink in (pair_sink, status_sink):
            if sink is not None:
                try: sink.close()
                except BaseException: pass
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    emit(output / "terminal.json", {"status": state, "error": error, "automatic_retry": False})
    if error: raise RuntimeError(error["message"])
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protein-ledger", type=Path, required=True); parser.add_argument("--augmentation-pairs", type=Path, required=True); parser.add_argument("--augmentation-status", type=Path, required=True); parser.add_argument("--train-activity-library", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--batch-rows", type=int, default=100000); args = parser.parse_args(); execute(args.protein_ledger, args.augmentation_pairs, args.augmentation_status, args.train_activity_library, args.output, args.batch_rows)


if __name__ == "__main__": main()
