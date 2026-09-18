#!/usr/bin/env python3
"""Join documented query outcomes only after a role's candidate/feature table is frozen."""
import argparse
import hashlib
import json
import traceback
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROLES = ("DEV", "CAL_FIT", "CAL_RULE", "RETEST")
KEY = ["query_protein_id", "reference_protein_id", "reference_activity_id"]
ACTIVITY_COLUMNS = ["protein_id", "ec_l3", "ec_l4", "canonical_rhea"]


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def emit(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle: json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")


def load_truth(activity_source, ledger, role):
    import pyarrow.dataset as ds
    role_proteins = set()
    for batch in pq.ParquetFile(ledger).iter_batches(columns=["protein_id", "role"]):
        for row in batch.to_pylist():
            if row["role"] == role:
                role_proteins.add(row["protein_id"])
    require(role_proteins, "empty role protein universe")
    values = {}
    source = ds.dataset(activity_source, format="parquet")
    predicate = ds.field("protein_id").isin(sorted(role_proteins))
    for batch in source.to_batches(batch_size=100000, columns=ACTIVITY_COLUMNS, filter=predicate):
        for row in batch.to_pylist():
            require(row["protein_id"] in role_proteins, "role-filtered activity scan")
            item = values.setdefault(row["protein_id"], {"ec3": set(), "ec4": set(), "rhea": set(), "rows": 0})
            if row["ec_l3"]: item["ec3"].add(row["ec_l3"])
            if row["ec_l4"]: item["ec4"].add(row["ec_l4"])
            if row["canonical_rhea"]: item["rhea"].add(row["canonical_rhea"])
            item["rows"] += 1
    require(values, "empty role truth")
    return values


def schema():
    return pa.schema([(name, pa.string()) for name in KEY] + [
        ("query_documented_activity_count", pa.int32()),
        ("query_documented_ec_l3_count", pa.int32()),
        ("query_documented_ec_l4_count", pa.int32()),
        ("query_documented_rhea_count", pa.int32()),
        ("observed_same_ec_l3", pa.bool_()), ("observed_same_ec_l4", pa.bool_()),
        ("exact_rhea_outcome_evaluable", pa.bool_()), ("observed_same_exact_rhea", pa.bool_()),
        ("deepest_shared_recorded_ec_level", pa.int8()),
        ("outcome_semantics", pa.string()),
    ])


def execute(root, role, producer, output):
    require(role in ROLES and producer.is_dir() and output.parent.is_dir() and not output.exists(), "role/output")
    feature_audit = producer.parent / "audit" / "S4N_ROLE_FEATURES_PASS.json"
    audit_json = producer.parent / "audit" / "independent_audit.json"
    checkpoint = json.loads(feature_audit.read_text(encoding="utf-8"))
    require(checkpoint["status"] == "PASS_S4N_ROLE_FEATURES" and checkpoint["role"] == role and checkpoint["audit_sha256"] == sha(audit_json), "feature freeze")
    output.mkdir(exist_ok=False); emit(output / "reservation.json", {"status": "ONE_S4O_ROLE_OUTCOME_JOIN_RESERVED", "role": role, "automatic_retry": False})
    state, error, writer = "FAIL_CLOSED", None, None
    try:
        activity = root / "data/processed/activity_table_canonical.parquet"
        ledger = root / "revisions/s4_20260910/truth_free_ledger_attempt_01/producer/protein_role_ledger.parquet"
        truth = load_truth(activity, ledger, role)
        metadata = producer / f"metadata_{role}.parquet"
        destination = output / f"labels_{role}.parquet"
        writer = pq.ParquetWriter(destination, schema(), compression="zstd")
        count = exact_evaluable = exact_positive = 0
        for batch in pq.ParquetFile(metadata).iter_batches(batch_size=100000):
            out = []
            for row in batch.to_pylist():
                query = truth.get(row["query_protein_id"])
                require(query is not None and query["rows"] > 0, "query truth coverage")
                same3 = bool(row["ec_l3"] and row["ec_l3"] in query["ec3"])
                same4 = bool(row["ec_l4"] and row["ec_l4"] in query["ec4"])
                evaluable = bool(row["canonical_rhea"] and query["rhea"])
                same_rhea = bool(row["canonical_rhea"] in query["rhea"]) if evaluable else None
                depth = 4 if same4 else (3 if same3 else 0)
                out.append({
                    **{name: row[name] for name in KEY},
                    "query_documented_activity_count": query["rows"],
                    "query_documented_ec_l3_count": len(query["ec3"]),
                    "query_documented_ec_l4_count": len(query["ec4"]),
                    "query_documented_rhea_count": len(query["rhea"]),
                    "observed_same_ec_l3": same3, "observed_same_ec_l4": same4,
                    "exact_rhea_outcome_evaluable": evaluable,
                    "observed_same_exact_rhea": same_rhea,
                    "deepest_shared_recorded_ec_level": depth,
                    "outcome_semantics": "DOCUMENTED_CONCORDANCE_NOT_BIOCHEMICAL_NEGATIVE",
                })
                count += 1; exact_evaluable += int(evaluable); exact_positive += int(same_rhea is True)
            writer.write_table(pa.Table.from_pylist(out, schema=schema()))
        writer.close(); writer = None
        require(count == pq.ParquetFile(metadata).metadata.num_rows > 0, "outcome row reconciliation")
        summary = {
            "status": "PASS_S4O_ROLE_OUTCOME_JOIN", "role": role, "rows": count,
            "exact_rhea_evaluable_rows": exact_evaluable, "exact_rhea_positive_rows": exact_positive,
            "feature_freeze_audit_sha256": sha(audit_json), "activity_source_sha256": sha(activity),
            "outcome_semantics": "DOCUMENTED_CONCORDANCE_NOT_BIOCHEMICAL_NEGATIVE",
            "other_role_truth_materialized": False, "output_sha256": sha(destination),
        }
        emit(output / "summary.json", summary); emit(output / "S4O_ROLE_OUTCOME_JOIN_PASS.json", {"status": "PASS_S4O_ROLE_OUTCOME_JOIN", "role": role, "summary_sha256": sha(output / "summary.json")}); state = summary["status"]
    except BaseException as exc:
        if writer is not None:
            try: writer.close()
            except BaseException: pass
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    emit(output / "terminal.json", {"status": state, "role": role, "error": error, "automatic_retry": False})
    if error: raise RuntimeError(error["message"])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--role", choices=ROLES, required=True); parser.add_argument("--producer", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); execute(args.root.resolve(), args.role, args.producer.resolve(), args.output.resolve())


if __name__ == "__main__": main()
