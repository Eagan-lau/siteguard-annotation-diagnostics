"""Stream frozen TRAIN candidates into recorded-concordance outcome pairs."""
import argparse
import hashlib
import json
import traceback
from collections import Counter, defaultdict
from pathlib import Path


ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")
RAW_METRICS = (
    "fident", "alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen",
    "qcov", "tcov", "evalue", "bits",
)
PROTEIN_COLUMNS = ["protein_id", "node_id", "component_id", "role"]
UNION_COLUMNS = [
    "query_node", "reference_node", "query_role", "reference_role",
    "reference_component_id", "mmseqs_rank", "foldseek_rank", "mmseqs_raw_rank",
    "foldseek_raw_rank",
] + [f"{modality}_{name}" for modality in ("mmseqs", "foldseek") for name in RAW_METRICS] + [
    "source_class", "rrf60_score", "rrf_constant", "candidate_union_provenance",
    "query_truth_read",
]
ACTIVITY_FIELDS = [
    "reference_protein_id", "reference_activity_id", "canonical_ec", "ec_l1",
    "ec_l2", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier",
]
EXPANDED_COLUMNS = UNION_COLUMNS + ACTIVITY_FIELDS + ["activity_provenance"]
LIBRARY_COLUMNS = [
    "reference_node", "reference_component_id", "reference_protein_id",
    "reference_activity_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3",
    "ec_l4", "canonical_rhea", "evidence_tier", "reference_role",
    "activity_provenance",
]
OUTCOME_FIELDS = [
    "query_documented_activity_count", "query_documented_ec_l3_count",
    "query_documented_ec_l4_count", "query_documented_rhea_count",
    "observed_same_ec_l3", "observed_same_ec_l4",
    "exact_rhea_outcome_evaluable", "observed_same_exact_rhea",
    "deepest_shared_recorded_ec_level", "query_truth_provenance",
    "outcome_semantics", "ground_truth_only",
]
PAIR_COLUMNS = ["query_protein_id", "query_component_id"] + EXPANDED_COLUMNS + OUTCOME_FIELDS
STATUS_COLUMNS = [
    "query_protein_id", "query_node", "query_component_id", "query_role", "status",
    "retrieved_activity_candidates", "eligible_query_activities", "labeled_pairs",
]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def schemas():
    import pyarrow as pa

    types = {
        "query_protein_id": pa.string(), "query_component_id": pa.int32(),
        "query_node": pa.int32(), "reference_node": pa.int32(),
        "query_role": pa.string(), "reference_role": pa.string(),
        "reference_component_id": pa.int32(), "mmseqs_rank": pa.int16(),
        "foldseek_rank": pa.int16(), "mmseqs_raw_rank": pa.int32(),
        "foldseek_raw_rank": pa.int32(), "source_class": pa.string(),
        "rrf60_score": pa.float64(), "rrf_constant": pa.int16(),
        "candidate_union_provenance": pa.string(), "query_truth_read": pa.bool_(),
        "reference_protein_id": pa.string(), "reference_activity_id": pa.string(),
        "canonical_ec": pa.string(), "ec_l1": pa.string(), "ec_l2": pa.string(),
        "ec_l3": pa.string(), "ec_l4": pa.string(), "canonical_rhea": pa.string(),
        "evidence_tier": pa.string(), "activity_provenance": pa.string(),
        "query_documented_activity_count": pa.int32(),
        "query_documented_ec_l3_count": pa.int32(),
        "query_documented_ec_l4_count": pa.int32(),
        "query_documented_rhea_count": pa.int32(),
        "observed_same_ec_l3": pa.bool_(), "observed_same_ec_l4": pa.bool_(),
        "exact_rhea_outcome_evaluable": pa.bool_(), "observed_same_exact_rhea": pa.bool_(),
        "deepest_shared_recorded_ec_level": pa.int8(),
        "query_truth_provenance": pa.string(), "outcome_semantics": pa.string(),
        "ground_truth_only": pa.bool_(),
    }
    integer_metrics = {"alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen"}
    for modality in ("mmseqs", "foldseek"):
        for name in RAW_METRICS:
            types[f"{modality}_{name}"] = pa.int32() if name in integer_metrics else pa.float64()
    pair_schema = pa.schema([(name, types[name]) for name in PAIR_COLUMNS])
    status_schema = pa.schema(
        [
            ("query_protein_id", pa.string()), ("query_node", pa.int32()),
            ("query_component_id", pa.int32()), ("query_role", pa.string()),
            ("status", pa.string()), ("retrieved_activity_candidates", pa.int32()),
            ("eligible_query_activities", pa.int32()), ("labeled_pairs", pa.int32()),
        ]
    )
    return pair_schema, status_schema


class BufferedSink:
    def __init__(self, path, schema, batch_rows):
        import pyarrow.parquet as pq

        require(type(batch_rows) is int and not isinstance(batch_rows, bool) and batch_rows > 0, "batch rows")
        self.schema, self.batch_rows, self.rows, self.count = schema, batch_rows, [], 0
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.batch_rows:
            self.flush()

    def flush(self):
        if self.rows:
            import pyarrow as pa

            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.count += len(self.rows)
            self.rows = []

    def close(self):
        self.flush()
        self.writer.close()


def load_proteins(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    require(table.column_names == PROTEIN_COLUMNS, "protein ledger schema")
    by_id, by_node, component_roles = {}, defaultdict(list), {}
    for row in table.to_pylist():
        protein, node, component, role = row["protein_id"], row["node_id"], row["component_id"], row["role"]
        require(isinstance(protein, str) and protein and protein not in by_id, "protein identity")
        require(type(node) is int and type(component) is int and role in ROLES, "protein ledger row")
        require(component not in component_roles or component_roles[component] == role, "component split across roles")
        by_id[protein], component_roles[component] = row, role
        by_node[node].append(row)
    require(by_id and set(row["role"] for row in by_id.values()) == set(ROLES), "protein role universe")
    for rows in by_node.values():
        require(len({(row["component_id"], row["role"]) for row in rows}) == 1, "node role/component")
        rows.sort(key=lambda row: row["protein_id"])
    return by_id, by_node


def load_train_labels(path, proteins):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    require(table.column_names == LIBRARY_COLUMNS, "TRAIN activity library schema")
    labels = defaultdict(lambda: {"activities": set(), "ec_l3": set(), "ec_l4": set(), "rhea": set()})
    seen, previous, activities = set(), None, {}
    for row in table.to_pylist():
        key = row["reference_node"], row["reference_protein_id"], row["reference_activity_id"]
        require(previous is None or key > previous, "TRAIN activity library order")
        previous = key
        activity, protein = row["reference_activity_id"], row["reference_protein_id"]
        require(activity not in seen, "duplicate TRAIN activity")
        seen.add(activity)
        meta = proteins.get(protein)
        require(meta is not None and meta["role"] == "TRAIN" and meta["node_id"] == row["reference_node"] and meta["component_id"] == row["reference_component_id"], "TRAIN activity protein identity")
        require(row["reference_role"] == "TRAIN" and row["activity_provenance"] == "TRAIN_REFERENCE_SOURCE_ACTIVITY_PRESERVED", "TRAIN activity provenance")
        canonical = row["canonical_ec"]
        parts = canonical.split(".") if isinstance(canonical, str) else []
        require(len(parts) == 4 and row["ec_l1"] == parts[0] and row["ec_l2"] == ".".join(parts[:2]) and row["ec_l3"] == ".".join(parts[:3]) and row["ec_l4"] == canonical, "EC hierarchy")
        require(row["canonical_rhea"] is None or (isinstance(row["canonical_rhea"], str) and row["canonical_rhea"].startswith("RHEA:")), "canonical Rhea")
        activities[activity] = row
        target = labels[protein]
        target["activities"].add(activity)
        target["ec_l3"].add(row["ec_l3"])
        target["ec_l4"].add(row["ec_l4"])
        if row["canonical_rhea"] is not None:
            target["rhea"].add(row["canonical_rhea"])
    return labels, activities


def iter_candidate_groups(path, proteins, by_node, activities, batch_rows):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    require(parquet.schema_arrow.names == EXPANDED_COLUMNS, "TRAIN candidate schema")
    current, rows, previous = None, [], None
    for batch in parquet.iter_batches(batch_size=batch_rows):
        for row in batch.to_pylist():
            key = row["query_node"], row["reference_node"], row["reference_protein_id"], row["reference_activity_id"]
            require(previous is None or key > previous, "TRAIN candidate order")
            previous = key
            require(row["query_role"] == "TRAIN" and row["reference_role"] == "TRAIN", "TRAIN candidate roles")
            require(row["query_node"] in by_node and by_node[row["query_node"]][0]["role"] == "TRAIN", "TRAIN query node")
            require(row["query_node"] != row["reference_node"], "self-node candidate")
            reference = proteins.get(row["reference_protein_id"])
            require(reference is not None and reference["role"] == "TRAIN" and reference["node_id"] == row["reference_node"] and reference["component_id"] == row["reference_component_id"], "candidate reference identity")
            require(not row["query_truth_read"] and row["candidate_union_provenance"] == "TRUTH_FREE_AUDITED_MODALITY_UNION_NO_POLICY_SELECTION", "candidate truth/provenance")
            require(row["activity_provenance"] == "TRAIN_REFERENCE_ACTIVITY_EXPANDED_AFTER_TRUTH_FREE_RETRIEVAL", "candidate activity provenance")
            source = activities.get(row["reference_activity_id"])
            require(source is not None, "candidate activity absent from TRAIN library")
            require(
                row["reference_node"] == source["reference_node"]
                and row["reference_component_id"] == source["reference_component_id"]
                and row["reference_protein_id"] == source["reference_protein_id"]
                and all(row[name] == source[name] for name in (
                    "canonical_ec", "ec_l1", "ec_l2", "ec_l3", "ec_l4",
                    "canonical_rhea", "evidence_tier",
                )),
                "candidate/reference activity identity",
            )
            if current is None:
                current = row["query_node"]
            if row["query_node"] != current:
                yield current, rows
                current, rows = row["query_node"], []
            rows.append(row)
    if current is not None:
        yield current, rows


def label_row(query, candidate, truth):
    same_l3 = candidate["ec_l3"] in truth["ec_l3"]
    same_l4 = candidate["ec_l4"] in truth["ec_l4"]
    rhea_evaluable = candidate["canonical_rhea"] is not None and bool(truth["rhea"])
    same_rhea = rhea_evaluable and candidate["canonical_rhea"] in truth["rhea"]
    return {
        "query_protein_id": query["protein_id"], "query_component_id": query["component_id"],
        **candidate,
        "query_documented_activity_count": len(truth["activities"]),
        "query_documented_ec_l3_count": len(truth["ec_l3"]),
        "query_documented_ec_l4_count": len(truth["ec_l4"]),
        "query_documented_rhea_count": len(truth["rhea"]),
        "observed_same_ec_l3": same_l3, "observed_same_ec_l4": same_l4,
        "exact_rhea_outcome_evaluable": rhea_evaluable,
        "observed_same_exact_rhea": same_rhea if rhea_evaluable else None,
        "deepest_shared_recorded_ec_level": 4 if same_l4 else (3 if same_l3 else 0),
        "query_truth_provenance": "TRAIN_QUERY_OUTCOME_ONLY_JOINED_AFTER_CANDIDATE_FREEZE",
        "outcome_semantics": "DOCUMENTED_CONCORDANCE_NOT_BIOCHEMICAL_NEGATIVE",
        "ground_truth_only": True,
    }


def execute(protein_ledger, train_candidates, train_activity_library, output, batch_rows=100000):
    protein_ledger, train_candidates = Path(protein_ledger).resolve(), Path(train_candidates).resolve()
    train_activity_library, output = Path(train_activity_library).resolve(), Path(output).resolve()
    require(protein_ledger.is_file() and train_candidates.is_file() and train_activity_library.is_file(), "outcome inputs")
    require(output.parent.is_dir() and not output.exists(), "exclusive output")
    output.mkdir(exist_ok=False)
    inputs = {
        "protein_ledger": {"path": str(protein_ledger), "sha256": sha(protein_ledger)},
        "train_candidates": {"path": str(train_candidates), "sha256": sha(train_candidates)},
        "train_activity_library": {"path": str(train_activity_library), "sha256": sha(train_activity_library)},
    }
    emit(output / "reservation.json", {"status": "ONE_S4F_TRAIN_OUTCOME_ATTEMPT_RESERVED", "inputs": inputs, "automatic_retry": False})
    pair_sink = status_sink = None
    state, error = "FAIL_CLOSED", None
    try:
        proteins, by_node = load_proteins(protein_ledger)
        labels, activities = load_train_labels(train_activity_library, proteins)
        library_rows = len(activities)
        pair_schema, status_schema = schemas()
        pair_sink = BufferedSink(output / "train_labeled_candidate_pairs.parquet", pair_schema, batch_rows)
        status_sink = BufferedSink(output / "train_query_outcome_status.parquet", status_schema, batch_rows)
        candidate_counts, labeled_counts = Counter(), Counter()
        positives = Counter()
        candidate_rows = candidate_query_nodes = 0
        for node, candidates in iter_candidate_groups(train_candidates, proteins, by_node, activities, batch_rows):
            candidate_query_nodes += 1
            candidate_rows += len(candidates)
            for query in by_node[node]:
                candidate_counts[query["protein_id"]] = len(candidates)
                truth = labels.get(query["protein_id"])
                if truth and truth["activities"]:
                    for candidate in candidates:
                        row = label_row(query, candidate, truth)
                        pair_sink.add(row)
                        labeled_counts[query["protein_id"]] += 1
                        positives["ec_l3"] += int(row["observed_same_ec_l3"])
                        positives["ec_l4"] += int(row["observed_same_ec_l4"])
                        positives["rhea_evaluable"] += int(row["exact_rhea_outcome_evaluable"])
                        positives["exact_rhea"] += int(row["observed_same_exact_rhea"] is True)
        status_counts = Counter()
        train_proteins = sorted((row for row in proteins.values() if row["role"] == "TRAIN"), key=lambda row: row["protein_id"])
        for query in train_proteins:
            candidates = candidate_counts[query["protein_id"]]
            truth = labels.get(query["protein_id"])
            eligible = len(truth["activities"]) if truth else 0
            if not candidates:
                status = "NO_RETRIEVED_ACTIVITY_CANDIDATE"
            elif not eligible:
                status = "NO_ELIGIBLE_TRAIN_QUERY_ACTIVITY"
            else:
                status = "TRAIN_OUTCOMES_JOINED"
            status_counts[status] += 1
            status_sink.add(
                {
                    "query_protein_id": query["protein_id"], "query_node": query["node_id"],
                    "query_component_id": query["component_id"], "query_role": "TRAIN",
                    "status": status, "retrieved_activity_candidates": candidates,
                    "eligible_query_activities": eligible,
                    "labeled_pairs": labeled_counts[query["protein_id"]],
                }
            )
        pair_sink.close()
        status_sink.close()
        require(status_sink.count == len(train_proteins) and pair_sink.count == sum(labeled_counts.values()), "writer census")
        summary = {
            "status": "PASS_S4F_TRAIN_OUTCOME_STREAM_PENDING_INDEPENDENT_AUDIT",
            "inputs": inputs, "train_proteins": len(train_proteins),
            "train_activity_library_rows": library_rows, "candidate_query_nodes": candidate_query_nodes,
            "train_candidate_rows": candidate_rows, "labeled_pair_rows": pair_sink.count,
            "status_counts": {name: status_counts[name] for name in (
                "TRAIN_OUTCOMES_JOINED", "NO_ELIGIBLE_TRAIN_QUERY_ACTIVITY", "NO_RETRIEVED_ACTIVITY_CANDIDATE"
            )},
            "positive_counts": {name: positives[name] for name in ("ec_l3", "ec_l4", "rhea_evaluable", "exact_rhea")},
            "query_truth_scope": "TRAIN_OUTCOME_ONLY_AFTER_CANDIDATE_FREEZE",
            "outcome_semantics": "DOCUMENTED_CONCORDANCE_NOT_BIOCHEMICAL_NEGATIVE",
            "rhea_is_independent_target_not_ec_level": True,
            "dev_truth_read": False, "cal_fit_truth_read": False,
            "cal_rule_truth_read": False, "retest_truth_read": False,
            "candidate_membership_or_rank_changed": False,
            "balanced_sampling_started": False, "pair_features_started": False,
            "preprocessing_started": False, "training_started": False,
        }
        emit(output / "producer_summary.json", summary)
        state = summary["status"]
    except BaseException as exc:
        for sink in (pair_sink, status_sink):
            if sink is not None:
                try:
                    sink.close()
                except BaseException:
                    pass
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    emit(output / "terminal.json", {"status": state, "error": error, "automatic_retry": False})
    if error:
        raise RuntimeError(error["message"])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein-ledger", type=Path, required=True)
    parser.add_argument("--train-candidates", type=Path, required=True)
    parser.add_argument("--train-activity-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=100000)
    args = parser.parse_args()
    execute(args.protein_ledger, args.train_candidates, args.train_activity_library, args.output, args.batch_rows)


if __name__ == "__main__":
    main()
