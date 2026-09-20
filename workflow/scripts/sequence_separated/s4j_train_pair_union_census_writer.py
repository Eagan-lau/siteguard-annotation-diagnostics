"""Build the lossless, unsampled TRAIN pair union and frequency census."""
import argparse
import hashlib
import json
import traceback
from collections import Counter
from pathlib import Path


ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")
RAW_METRICS = ("fident", "alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen", "qcov", "tcov", "evalue", "bits")
PROTEIN_COLUMNS = ["protein_id", "node_id", "component_id", "role"]
COVARIATE_PROJECTION = ["protein_id", "node_id", "component_id", "role", "primary_pfam"]
RETRIEVAL_UNION_COLUMNS = [
    "query_node", "reference_node", "query_role", "reference_role", "reference_component_id",
    "mmseqs_rank", "foldseek_rank", "mmseqs_raw_rank", "foldseek_raw_rank",
] + [f"{modality}_{name}" for modality in ("mmseqs", "foldseek") for name in RAW_METRICS] + [
    "source_class", "rrf60_score", "rrf_constant", "candidate_union_provenance", "query_truth_read",
]
ACTIVITY_FIELDS = ["reference_protein_id", "reference_activity_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier"]
RETRIEVAL_OUTCOME_FIELDS = [
    "query_documented_activity_count", "query_documented_ec_l3_count", "query_documented_ec_l4_count",
    "query_documented_rhea_count", "observed_same_ec_l3", "observed_same_ec_l4",
    "exact_rhea_outcome_evaluable", "observed_same_exact_rhea", "deepest_shared_recorded_ec_level",
    "query_truth_provenance", "outcome_semantics", "ground_truth_only",
]
RETRIEVAL_COLUMNS = ["query_protein_id", "query_component_id"] + RETRIEVAL_UNION_COLUMNS + ACTIVITY_FIELDS + ["activity_provenance"] + RETRIEVAL_OUTCOME_FIELDS
AUGMENTATION_COLUMNS = [
    "query_protein_id", "query_node", "query_component_id", "query_role", "reference_node",
    "reference_component_id", "reference_protein_id", "reference_activity_id", "canonical_ec",
    "ec_l1", "ec_l2", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier", "candidate_origin",
    "augmentation_mechanisms", "maximum_mechanism_inclusion_probability",
    "augmentation_probability_is_population_weight", "query_truth_provenance", "outcome_semantics",
    "query_documented_activity_count", "query_documented_ec_l3_count", "query_documented_ec_l4_count",
    "query_documented_rhea_count", "observed_same_ec_l3", "observed_same_ec_l4",
    "exact_rhea_outcome_evaluable", "observed_same_exact_rhea", "deepest_shared_recorded_ec_level",
    "outcome_truth_provenance", "ground_truth_only",
]
COMMON_FIELDS = [
    "query_protein_id", "query_node", "query_component_id", "query_role", "reference_node",
    "reference_component_id", "reference_protein_id", "reference_activity_id", "canonical_ec", "ec_l1",
    "ec_l2", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier",
    "query_documented_activity_count", "query_documented_ec_l3_count", "query_documented_ec_l4_count",
    "query_documented_rhea_count", "observed_same_ec_l3", "observed_same_ec_l4",
    "exact_rhea_outcome_evaluable", "observed_same_exact_rhea", "deepest_shared_recorded_ec_level",
    "outcome_semantics", "ground_truth_only",
]
SORT_FIELDS = ("query_protein_id", "reference_protein_id", "reference_activity_id")
RETRIEVAL_FEATURE_FIELDS = [
    "mmseqs_rank", "foldseek_rank", "mmseqs_raw_rank", "foldseek_raw_rank",
] + [f"{modality}_{name}" for modality in ("mmseqs", "foldseek") for name in RAW_METRICS] + [
    "source_class", "rrf60_score", "rrf_constant",
]
OUTPUT_COLUMNS = COMMON_FIELDS + [
    "reference_role", "query_primary_pfam", "reference_primary_pfam", "pair_origin",
    "retrieval_present",
] + RETRIEVAL_FEATURE_FIELDS + [
    "retrieval_candidate_union_provenance", "retrieval_query_truth_read",
    "retrieval_activity_provenance", "retrieval_outcome_truth_provenance",
    "augmentation_present", "augmentation_mechanisms",
    "augmentation_design_inclusion_probability", "augmentation_probability_is_population_weight",
    "augmentation_candidate_origin", "augmentation_sampling_truth_provenance",
    "augmentation_outcome_truth_provenance", "hard_case_flags_available_before_features",
    "reference_activity_frequency", "query_family_frequency", "reference_component_frequency",
    "sampling_applied", "sample_weight",
]


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


def output_schema():
    import pyarrow as pa
    strings = {
        "query_protein_id", "query_role", "reference_protein_id", "reference_activity_id", "canonical_ec",
        "ec_l1", "ec_l2", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier", "outcome_semantics",
        "reference_role", "query_primary_pfam", "reference_primary_pfam", "pair_origin", "source_class",
        "retrieval_candidate_union_provenance", "retrieval_activity_provenance", "retrieval_outcome_truth_provenance",
        "augmentation_mechanisms", "augmentation_candidate_origin", "augmentation_sampling_truth_provenance",
        "augmentation_outcome_truth_provenance", "hard_case_flags_available_before_features",
    }
    int32 = {
        "query_node", "query_component_id", "reference_node", "reference_component_id",
        "query_documented_activity_count", "query_documented_ec_l3_count", "query_documented_ec_l4_count",
        "query_documented_rhea_count", "mmseqs_raw_rank", "foldseek_raw_rank", "reference_activity_frequency",
        "query_family_frequency", "reference_component_frequency",
    }
    int16 = {"mmseqs_rank", "foldseek_rank", "rrf_constant"}
    bools = {
        "observed_same_ec_l3", "observed_same_ec_l4", "exact_rhea_outcome_evaluable",
        "observed_same_exact_rhea", "ground_truth_only", "retrieval_present", "retrieval_query_truth_read",
        "augmentation_present", "augmentation_probability_is_population_weight", "sampling_applied",
    }
    types = {name: pa.string() for name in strings}; types.update({name: pa.int32() for name in int32}); types.update({name: pa.int16() for name in int16}); types.update({name: pa.bool_() for name in bools})
    types["deepest_shared_recorded_ec_level"] = pa.int8(); types["augmentation_design_inclusion_probability"] = pa.float64(); types["sample_weight"] = pa.float64(); types["rrf60_score"] = pa.float64()
    integer_metrics = {"alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen"}
    for modality in ("mmseqs", "foldseek"):
        for name in RAW_METRICS: types[f"{modality}_{name}"] = pa.int32() if name in integer_metrics else pa.float64()
    return pa.schema([(name, types[name]) for name in OUTPUT_COLUMNS])


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


def read_covariates(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=COVARIATE_PROJECTION); require(table.column_names == COVARIATE_PROJECTION, "covariate projection")
    rows = {}
    for row in table.to_pylist():
        protein = row["protein_id"]; require(isinstance(protein, str) and protein and protein not in rows and row["role"] in ROLES, "covariate identity"); rows[protein] = row
    require(rows and set(row["role"] for row in rows.values()) == set(ROLES), "covariate role universe"); return rows


def prepare_sorted(source, columns, target, label):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(source); require(parquet.schema_arrow.names == columns, label + " input schema")
    table = pq.read_table(source, columns=columns).sort_by([(name, "ascending") for name in SORT_FIELDS])
    pq.write_table(table, target, compression="zstd"); return table.num_rows


def iter_rows(path, columns, batch_rows):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path); require(parquet.schema_arrow.names == columns, "sorted staging schema")
    previous = None
    for batch in parquet.iter_batches(batch_size=batch_rows):
        for row in batch.to_pylist():
            key = tuple(row[name] for name in SORT_FIELDS); require(previous is None or key > previous, "strict pair order"); previous = key; yield key, row


def validate_common(row, covariates):
    query, reference = covariates.get(row["query_protein_id"]), covariates.get(row["reference_protein_id"])
    require(query is not None and reference is not None and query["role"] == reference["role"] == row["query_role"] == "TRAIN", "TRAIN pair scope")
    require(row["query_node"] == query["node_id"] and row["query_component_id"] == query["component_id"], "query identity")
    require(row["reference_node"] == reference["node_id"] and row["reference_component_id"] == reference["component_id"], "reference identity")
    require(row["query_node"] != row["reference_node"], "same exact-sequence node")
    for name in ("query_documented_activity_count", "query_documented_ec_l3_count", "query_documented_ec_l4_count"):
        require(type(row[name]) is int and row[name] > 0, "query outcome counts")
    require(type(row["query_documented_rhea_count"]) is int and row["query_documented_rhea_count"] >= 0, "query Rhea count")
    require(type(row["observed_same_ec_l3"]) is bool and type(row["observed_same_ec_l4"]) is bool and type(row["exact_rhea_outcome_evaluable"]) is bool, "outcome types")
    require(not row["observed_same_ec_l4"] or row["observed_same_ec_l3"], "EC hierarchy outcome")
    require(row["deepest_shared_recorded_ec_level"] == (4 if row["observed_same_ec_l4"] else (3 if row["observed_same_ec_l3"] else 0)), "deepest EC outcome")
    require(row["ground_truth_only"] is True and row["outcome_semantics"] == "DOCUMENTED_CONCORDANCE_NOT_BIOCHEMICAL_NEGATIVE", "outcome provenance")
    if row["exact_rhea_outcome_evaluable"]:
        require(type(row["observed_same_exact_rhea"]) is bool and row["canonical_rhea"] is not None and row["query_documented_rhea_count"] > 0, "evaluable Rhea outcome")
    else: require(row["observed_same_exact_rhea"] is None, "nullable Rhea outcome")
    canonical = row["canonical_ec"]; parts = canonical.split(".") if isinstance(canonical, str) else []
    require(len(parts) == 4 and row["ec_l1"] == parts[0] and row["ec_l2"] == ".".join(parts[:2]) and row["ec_l3"] == ".".join(parts[:3]) and row["ec_l4"] == canonical, "reference EC hierarchy")
    return query, reference


def validate_retrieval(row, covariates):
    query, reference = validate_common(row, covariates)
    require(row["reference_role"] == "TRAIN" and not row["query_truth_read"], "retrieval truth/role scope")
    require(row["candidate_union_provenance"] == "TRUTH_FREE_AUDITED_MODALITY_UNION_NO_POLICY_SELECTION", "retrieval provenance")
    require(row["activity_provenance"] == "TRAIN_REFERENCE_ACTIVITY_EXPANDED_AFTER_TRUTH_FREE_RETRIEVAL", "retrieval activity provenance")
    require(row["query_truth_provenance"] == "TRAIN_QUERY_OUTCOME_ONLY_JOINED_AFTER_CANDIDATE_FREEZE", "retrieval outcome provenance")
    return query, reference


def validate_augmentation(row, covariates):
    query, reference = validate_common(row, covariates)
    require(row["candidate_origin"] == "SUPERVISED_TRAIN_AUGMENTATION_NOT_DEPLOYMENT_CANDIDATE", "augmentation provenance")
    probability = row["maximum_mechanism_inclusion_probability"]
    require(not row["augmentation_probability_is_population_weight"] and isinstance(probability, float) and 0 < probability <= 1, "augmentation probability semantics")
    require(row["query_truth_provenance"] == "TRAIN_SAMPLING_ONLY_NOT_MODEL_INPUT" and row["outcome_truth_provenance"] == "TRAIN_QUERY_OUTCOME_ONLY_JOINED_AFTER_AUGMENTATION_FREEZE", "augmentation truth provenance")
    return query, reference


def merge_rows(retrieval_path, augmentation_path, covariates, batch_rows):
    retrieval = iter(iter_rows(retrieval_path, RETRIEVAL_COLUMNS, batch_rows)); augmentation = iter(iter_rows(augmentation_path, AUGMENTATION_COLUMNS, batch_rows))
    left = next(retrieval, None); right = next(augmentation, None)
    while left is not None or right is not None:
        if right is None or (left is not None and left[0] < right[0]):
            key, rrow, arow = left[0], left[1], None; left = next(retrieval, None)
        elif left is None or right[0] < left[0]:
            key, rrow, arow = right[0], None, right[1]; right = next(augmentation, None)
        else:
            key, rrow, arow = left[0], left[1], right[1]; left = next(retrieval, None); right = next(augmentation, None)
        if rrow is not None: query, reference = validate_retrieval(rrow, covariates)
        if arow is not None:
            aquery, areference = validate_augmentation(arow, covariates)
            if rrow is None: query, reference = aquery, areference
        if rrow is not None and arow is not None:
            require({name: rrow[name] for name in COMMON_FIELDS} == {name: arow[name] for name in COMMON_FIELDS}, "cross-origin identity/outcome disagreement")
        yield key, rrow, arow, query, reference


def hard_flags(row, query_pfam, reference_pfam):
    flags = []
    if row["observed_same_ec_l3"] and not row["observed_same_ec_l4"]: flags.append("H1_SAME_EC3_DIFFERENT_EC4")
    if row["observed_same_ec_l4"] and row["exact_rhea_outcome_evaluable"] and not row["observed_same_exact_rhea"]: flags.append("H2_SAME_EC4_DIFFERENT_RHEA")
    divergent = not row["observed_same_ec_l4"] or (row["exact_rhea_outcome_evaluable"] and not row["observed_same_exact_rhea"])
    if query_pfam is not None and query_pfam == reference_pfam and divergent: flags.append("H5_SAME_PFAM_FINE_DIVERGENCE")
    return flags


def output_row(rrow, arow, query, reference, activity_frequency, family_frequency, component_frequency):
    source = rrow if rrow is not None else arow; origin = "BOTH" if rrow is not None and arow is not None else ("RETRIEVAL" if rrow is not None else "AUGMENTATION")
    result = {name: source[name] for name in COMMON_FIELDS}
    result.update({"reference_role": "TRAIN", "query_primary_pfam": query["primary_pfam"], "reference_primary_pfam": reference["primary_pfam"], "pair_origin": origin, "retrieval_present": rrow is not None})
    for name in RETRIEVAL_FEATURE_FIELDS: result[name] = rrow[name] if rrow is not None else None
    result.update({
        "retrieval_candidate_union_provenance": rrow["candidate_union_provenance"] if rrow is not None else None,
        "retrieval_query_truth_read": rrow["query_truth_read"] if rrow is not None else None,
        "retrieval_activity_provenance": rrow["activity_provenance"] if rrow is not None else None,
        "retrieval_outcome_truth_provenance": rrow["query_truth_provenance"] if rrow is not None else None,
        "augmentation_present": arow is not None,
        "augmentation_mechanisms": arow["augmentation_mechanisms"] if arow is not None else None,
        "augmentation_design_inclusion_probability": arow["maximum_mechanism_inclusion_probability"] if arow is not None else None,
        "augmentation_probability_is_population_weight": False,
        "augmentation_candidate_origin": arow["candidate_origin"] if arow is not None else None,
        "augmentation_sampling_truth_provenance": arow["query_truth_provenance"] if arow is not None else None,
        "augmentation_outcome_truth_provenance": arow["outcome_truth_provenance"] if arow is not None else None,
    })
    flags = hard_flags(source, query["primary_pfam"], reference["primary_pfam"])
    result.update({
        "hard_case_flags_available_before_features": "+".join(flags) if flags else "NONE_AT_PAIR_UNION_STAGE",
        "reference_activity_frequency": activity_frequency[source["reference_activity_id"]],
        "query_family_frequency": family_frequency[query["primary_pfam"]],
        "reference_component_frequency": component_frequency[source["reference_component_id"]],
        "sampling_applied": False, "sample_weight": None,
    })
    return result


def execute(retrieval_pairs, augmentation_pairs, covariate_ledger, output, batch_rows=100000):
    paths = {"retrieval_pairs": Path(retrieval_pairs).resolve(), "augmentation_pairs": Path(augmentation_pairs).resolve(), "covariate_ledger": Path(covariate_ledger).resolve()}; output = Path(output).resolve()
    require(all(path.is_file() for path in paths.values()), "S4J inputs"); require(output.parent.is_dir() and not output.exists(), "exclusive output"); output.mkdir(exist_ok=False)
    inputs = {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}; emit(output / "reservation.json", {"status": "ONE_S4J_TRAIN_PAIR_UNION_ATTEMPT_RESERVED", "inputs": inputs, "automatic_retry": False})
    sink = None; state, error = "FAIL_CLOSED", None
    try:
        covariates = read_covariates(paths["covariate_ledger"])
        retrieval_sorted = output / "staging_retrieval_pairs_sorted.parquet"; augmentation_sorted = output / "staging_augmentation_pairs_sorted.parquet"
        retrieval_rows = prepare_sorted(paths["retrieval_pairs"], RETRIEVAL_COLUMNS, retrieval_sorted, "retrieval")
        augmentation_rows = prepare_sorted(paths["augmentation_pairs"], AUGMENTATION_COLUMNS, augmentation_sorted, "augmentation")
        activity_frequency, family_frequency, component_frequency = Counter(), Counter(), Counter(); origin_counts, hard_counts = Counter(), Counter(); union_rows = 0
        for key, rrow, arow, query, reference in merge_rows(retrieval_sorted, augmentation_sorted, covariates, batch_rows):
            source = rrow if rrow is not None else arow; union_rows += 1
            activity_frequency[source["reference_activity_id"]] += 1; family_frequency[query["primary_pfam"]] += 1; component_frequency[source["reference_component_id"]] += 1
            origin_counts["BOTH" if rrow is not None and arow is not None else ("RETRIEVAL" if rrow is not None else "AUGMENTATION")] += 1
            hard_counts.update(hard_flags(source, query["primary_pfam"], reference["primary_pfam"]))
        require(union_rows and union_rows == retrieval_rows + augmentation_rows - origin_counts["BOTH"], "union census")
        sink = Sink(output / "train_pair_union_census.parquet", output_schema(), batch_rows)
        for key, rrow, arow, query, reference in merge_rows(retrieval_sorted, augmentation_sorted, covariates, batch_rows):
            sink.add(output_row(rrow, arow, query, reference, activity_frequency, family_frequency, component_frequency))
        sink.close(); require(sink.count == union_rows, "union output rows")
        summary = {
            "status": "PASS_S4J_TRAIN_PAIR_UNION_CENSUS_PENDING_INDEPENDENT_AUDIT", "inputs": inputs,
            "retrieval_input_rows": retrieval_rows, "augmentation_input_rows": augmentation_rows,
            "union_rows": union_rows, "origin_counts": {name: origin_counts[name] for name in ("RETRIEVAL", "AUGMENTATION", "BOTH")},
            "hard_case_counts": {name: hard_counts[name] for name in ("H1_SAME_EC3_DIFFERENT_EC4", "H2_SAME_EC4_DIFFERENT_RHEA", "H5_SAME_PFAM_FINE_DIVERGENCE")},
            "maximum_reference_activity_frequency": max(activity_frequency.values()), "maximum_query_family_frequency": max(family_frequency.values()), "maximum_reference_component_frequency": max(component_frequency.values()),
            "full_new_retrieval_fields_preserved": True, "two_pass_sorted_merge": True, "sorted_staging_retained": True,
            "cross_origin_outcomes_required_to_agree": True, "nontrain_truth_read": False,
            "augmentation_probability_is_population_weight": False, "sampling_applied": False,
            "sample_weights_created": False, "feature_matrix_created": False, "training_started": False,
        }
        emit(output / "producer_summary.json", summary); state = summary["status"]
    except BaseException as exc:
        if sink is not None:
            try: sink.close()
            except BaseException: pass
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    emit(output / "terminal.json", {"status": state, "error": error, "automatic_retry": False})
    if error: raise RuntimeError(error["message"])
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--retrieval-pairs", type=Path, required=True); parser.add_argument("--augmentation-pairs", type=Path, required=True); parser.add_argument("--covariate-ledger", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--batch-rows", type=int, default=100000); args = parser.parse_args(); execute(args.retrieval_pairs, args.augmentation_pairs, args.covariate_ledger, args.output, args.batch_rows)


if __name__ == "__main__": main()
