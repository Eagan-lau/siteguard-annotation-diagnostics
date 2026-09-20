"""External-sort, memory-bounded S4J lossless TRAIN pair union and census."""
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import traceback
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("s4j_core_writer", HERE / "s4j_train_pair_union_census_core.py")
core = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(core)
OUTPUT_COLUMNS = list(core.IDENTITY_FIELDS + core.OUTCOME_FIELDS) + [
    "query_primary_pfam", "reference_primary_pfam", "pair_origin",
    "retrieval_present", "rrf60_score", "mmseqs_rank", "foldseek_rank",
    "augmentation_present", "augmentation_mechanisms",
    "augmentation_design_inclusion_probability", "augmentation_probability_is_population_weight",
    "hard_case_flags_available_before_features", "reference_activity_frequency",
    "query_family_frequency", "reference_component_frequency", "sampling_applied", "sample_weight",
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


def schema():
    import pyarrow as pa
    string_fields = {"query_protein_id", "query_role", "reference_protein_id", "reference_activity_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier", "query_primary_pfam", "reference_primary_pfam", "pair_origin", "augmentation_mechanisms", "hard_case_flags_available_before_features"}
    int32_fields = {"query_node", "query_component_id", "reference_node", "reference_component_id", "query_documented_activity_count", "query_documented_ec_l3_count", "query_documented_ec_l4_count", "query_documented_rhea_count", "mmseqs_rank", "foldseek_rank"}
    int64_fields = {"reference_activity_frequency", "query_family_frequency", "reference_component_frequency"}
    int8_fields = {"deepest_shared_recorded_ec_level"}
    bool_fields = {"observed_same_ec_l3", "observed_same_ec_l4", "exact_rhea_outcome_evaluable", "observed_same_exact_rhea", "ground_truth_only", "retrieval_present", "augmentation_present", "augmentation_probability_is_population_weight", "sampling_applied"}
    float_fields = {"rrf60_score", "augmentation_design_inclusion_probability", "sample_weight"}
    types = {**{x: pa.string() for x in string_fields}, **{x: pa.int32() for x in int32_fields}, **{x: pa.int64() for x in int64_fields}, **{x: pa.int8() for x in int8_fields}, **{x: pa.bool_() for x in bool_fields}, **{x: pa.float64() for x in float_fields}}
    require(set(types) == set(OUTPUT_COLUMNS), "output type coverage")
    return pa.schema([(name, types[name]) for name in OUTPUT_COLUMNS])


class Sink:
    def __init__(self, path, arrow_schema, batch_rows):
        import pyarrow.parquet as pq
        self.path, self.schema, self.batch_rows, self.rows, self.count = path, arrow_schema, batch_rows, [], 0
        self.writer = pq.ParquetWriter(path, arrow_schema, compression="zstd")
    def add(self, row):
        self.rows.append(row); self.count += 1
        if len(self.rows) >= self.batch_rows: self.flush()
    def flush(self):
        if self.rows:
            import pyarrow as pa
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema)); self.rows = []
    def close(self): self.flush(); self.writer.close()


def load_covariates(path):
    import pyarrow.parquet as pq
    required = ["protein_id", "node_id", "component_id", "role", "primary_pfam"]
    parquet = pq.ParquetFile(path); require(set(required) <= set(parquet.schema_arrow.names), "covariate schema")
    rows = pq.read_table(path, columns=required).to_pylist()
    return core.validate_covariates(rows)


def safe_key(row):
    values = (row["query_protein_id"], row["reference_protein_id"], row["reference_activity_id"])
    require(all(isinstance(value, str) and value and "\t" not in value and "\n" not in value for value in values), "sortable pair identity")
    return values


def spool_origin(path, origin, covariates, handle, counts):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    required = set(core.IDENTITY_FIELDS + core.OUTCOME_FIELDS) | (core.RETRIEVAL_FIELDS if origin == "RETRIEVAL" else core.AUGMENTATION_FIELDS)
    require(required <= set(parquet.schema_arrow.names), origin + " pair schema")
    for batch in parquet.iter_batches(batch_size=100000):
        for row in batch.to_pylist():
            query, reference = core.validate_common(row, covariates); key = safe_key(row)
            if origin == "RETRIEVAL":
                require(core.RETRIEVAL_FIELDS <= set(row) and row["candidate_union_provenance"] == "TRUTH_FREE_AUDITED_MODALITY_UNION_NO_POLICY_SELECTION", "retrieval provenance")
                payload = {"shared": core.shared_payload(row), "query_primary_pfam": query["primary_pfam"], "reference_primary_pfam": reference["primary_pfam"], "retrieval": {"rrf60_score": row["rrf60_score"], "mmseqs_rank": row["mmseqs_rank"], "foldseek_rank": row["foldseek_rank"]}}
            else:
                require(core.AUGMENTATION_FIELDS <= set(row) and row["candidate_origin"] == "SUPERVISED_TRAIN_AUGMENTATION_NOT_DEPLOYMENT_CANDIDATE", "augmentation provenance")
                require(row["augmentation_probability_is_population_weight"] is False and 0 < row["maximum_mechanism_inclusion_probability"] <= 1, "augmentation probability semantics")
                payload = {"shared": core.shared_payload(row), "query_primary_pfam": query["primary_pfam"], "reference_primary_pfam": reference["primary_pfam"], "augmentation": {"augmentation_mechanisms": row["augmentation_mechanisms"], "augmentation_design_inclusion_probability": row["maximum_mechanism_inclusion_probability"]}}
            handle.write("\t".join((*key, origin, json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))) + "\n")
            counts[origin.lower() + "_input_rows"] += 1


def parse_spool(line):
    fields = line.rstrip("\n").split("\t", 4); require(len(fields) == 5, "sorted spool row")
    return tuple(fields[:3]), fields[3], json.loads(fields[4])


def base_union(group):
    items = {}
    for key, origin, payload in group:
        require(origin in ("RETRIEVAL", "AUGMENTATION") and origin not in items, "duplicate within origin")
        items[origin] = payload
    require(items, "empty key group")
    payloads = list(items.values()); shared = payloads[0]["shared"]
    require(all(item["shared"] == shared and item["query_primary_pfam"] == payloads[0]["query_primary_pfam"] and item["reference_primary_pfam"] == payloads[0]["reference_primary_pfam"] for item in payloads), "cross-origin identity/outcome disagreement")
    retrieval, augmentation = items.get("RETRIEVAL"), items.get("AUGMENTATION")
    return {
        **shared, "query_primary_pfam": payloads[0]["query_primary_pfam"], "reference_primary_pfam": payloads[0]["reference_primary_pfam"],
        "pair_origin": "BOTH" if retrieval and augmentation else ("RETRIEVAL" if retrieval else "AUGMENTATION"),
        "retrieval_present": retrieval is not None,
        "rrf60_score": retrieval["retrieval"]["rrf60_score"] if retrieval else None,
        "mmseqs_rank": retrieval["retrieval"]["mmseqs_rank"] if retrieval else None,
        "foldseek_rank": retrieval["retrieval"]["foldseek_rank"] if retrieval else None,
        "augmentation_present": augmentation is not None,
        "augmentation_mechanisms": augmentation["augmentation"]["augmentation_mechanisms"] if augmentation else None,
        "augmentation_design_inclusion_probability": augmentation["augmentation"]["augmentation_design_inclusion_probability"] if augmentation else None,
        "augmentation_probability_is_population_weight": False,
    }


def hard_flags(row):
    flags = []
    if row["observed_same_ec_l3"] and not row["observed_same_ec_l4"]: flags.append("H1_SAME_EC3_DIFFERENT_EC4")
    if row["observed_same_ec_l4"] and row["exact_rhea_outcome_evaluable"] and not row["observed_same_exact_rhea"]: flags.append("H2_SAME_EC4_DIFFERENT_RHEA")
    divergent = not row["observed_same_ec_l4"] or (row["exact_rhea_outcome_evaluable"] and not row["observed_same_exact_rhea"])
    if row["query_primary_pfam"] is not None and row["query_primary_pfam"] == row["reference_primary_pfam"] and divergent: flags.append("H5_SAME_PFAM_FINE_DIVERGENCE")
    return "+".join(flags) if flags else "NONE_AT_PAIR_UNION_STAGE"


def execute(retrieval_pairs, augmentation_pairs, covariate_ledger, output, scratch_root, sort_binary=Path("/usr/bin/sort"), batch_rows=100000):
    retrieval_pairs, augmentation_pairs, covariate_ledger, output, scratch_root, sort_binary = map(lambda p: Path(p).resolve(), (retrieval_pairs, augmentation_pairs, covariate_ledger, output, scratch_root, sort_binary))
    require(all(path.is_file() for path in (retrieval_pairs, augmentation_pairs, covariate_ledger, sort_binary)), "S4J inputs")
    require(output.parent.is_dir() and not output.exists() and scratch_root.is_dir(), "S4J output/scratch")
    output.mkdir(exist_ok=False); emit(output / "reservation.json", {"status": "ONE_S4J_TRAIN_PAIR_UNION_RESERVED", "automatic_retry": False})
    state, error, sink = "FAIL_CLOSED", None, None
    try:
        covariates = load_covariates(covariate_ledger)
        scratch = scratch_root / ("siteguard_s4j_" + os.environ.get("SLURM_JOB_ID", "local")); require(not scratch.exists(), "scratch identity"); scratch.mkdir(exist_ok=False)
        raw_spool, sorted_spool, base_spool = scratch / "origin.tsv", scratch / "origin.sorted.tsv", scratch / "union.jsonl"
        counts = Counter()
        with raw_spool.open("x", encoding="utf-8", newline="\n") as handle:
            spool_origin(retrieval_pairs, "RETRIEVAL", covariates, handle, counts); spool_origin(augmentation_pairs, "AUGMENTATION", covariates, handle, counts)
        environment = dict(os.environ); environment.update({"LC_ALL": "C", "TMPDIR": str(scratch)})
        result = subprocess.run([str(sort_binary), "--stable", "--field-separator=\t", "--key=1,1", "--key=2,2", "--key=3,3", "--output=" + str(sorted_spool), str(raw_spool)], capture_output=True, text=True, env=environment)
        require(result.returncode == 0 and not result.stderr, "external pair sort")
        activity_frequency, family_frequency, component_frequency = Counter(), Counter(), Counter()
        previous = None
        with sorted_spool.open("r", encoding="utf-8", newline="") as reader, base_spool.open("x", encoding="utf-8", newline="\n") as base_handle:
            parsed = (parse_spool(line) for line in reader)
            for key, rows in itertools.groupby(parsed, key=lambda item: item[0]):
                require(previous is None or key > previous, "union key order"); previous = key
                base = base_union(rows); require(key == safe_key(base), "union key identity")
                base_handle.write(json.dumps(base, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                counts["union_rows"] += 1; counts[base["pair_origin"].lower() + "_rows"] += 1
                activity_frequency[base["reference_activity_id"]] += 1; family_frequency[base["query_primary_pfam"]] += 1; component_frequency[base["reference_component_id"]] += 1
        require(counts["union_rows"] > 0 and counts["retrieval_input_rows"] == counts["retrieval_rows"] + counts["both_rows"] and counts["augmentation_input_rows"] == counts["augmentation_rows"] + counts["both_rows"], "lossless origin census")
        sink = Sink(output / "train_pair_union_census.parquet", schema(), batch_rows); row_digest = hashlib.sha256()
        with base_spool.open("r", encoding="utf-8", newline="") as handle:
            for line in handle:
                row = json.loads(line)
                row.update({"hard_case_flags_available_before_features": hard_flags(row), "reference_activity_frequency": activity_frequency[row["reference_activity_id"]], "query_family_frequency": family_frequency[row["query_primary_pfam"]], "reference_component_frequency": component_frequency[row["reference_component_id"]], "sampling_applied": False, "sample_weight": None})
                canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"; row_digest.update(canonical); sink.add(row)
        sink.close(); require(sink.count == counts["union_rows"], "output census")
        inputs = {name: {"path": str(path), "sha256": sha(path)} for name, path in {"retrieval_pairs": retrieval_pairs, "augmentation_pairs": augmentation_pairs, "covariate_ledger": covariate_ledger}.items()}
        summary = {"status": "PASS_S4J_TRAIN_PAIR_UNION_CENSUS_PENDING_INDEPENDENT_AUDIT", "inputs": inputs, **dict(counts), "output_sha256": sha(output / "train_pair_union_census.parquet"), "ordered_row_sha256": row_digest.hexdigest(), "external_sort": str(sort_binary), "sampling_applied": False, "sample_weight_created": False, "nontrain_truth_read": False, "feature_matrix_created": False, "training_started": False}
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
    parser = argparse.ArgumentParser(); parser.add_argument("--retrieval-pairs", type=Path, required=True); parser.add_argument("--augmentation-pairs", type=Path, required=True); parser.add_argument("--covariate-ledger", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--scratch-root", type=Path, required=True); parser.add_argument("--sort-binary", type=Path, default=Path("/usr/bin/sort")); parser.add_argument("--batch-rows", type=int, default=100000)
    args = parser.parse_args(); execute(args.retrieval_pairs, args.augmentation_pairs, args.covariate_ledger, args.output, args.scratch_root, args.sort_binary, args.batch_rows)


if __name__ == "__main__": main()
