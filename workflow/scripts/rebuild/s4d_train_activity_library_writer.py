"""Materialize the TRAIN-only reference activity library for S4D."""
import argparse
import hashlib
import importlib.util
import json
import traceback
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("s4d_activity_expansion_core_for_writer", HERE / "s4d_activity_expansion_core.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


PROTEIN_COLUMNS = ["protein_id", "node_id", "component_id", "role"]
ACTIVITY_COLUMNS = [
    "activity_id", "protein_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3",
    "ec_l4", "canonical_rhea", "evidence_tier",
]
LIBRARY_COLUMNS = [
    "reference_node", "reference_component_id", "reference_protein_id",
    "reference_activity_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3",
    "ec_l4", "canonical_rhea", "evidence_tier", "reference_role",
    "activity_provenance",
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


def library_schema():
    import pyarrow as pa

    return pa.schema(
        [("reference_node", pa.int32()), ("reference_component_id", pa.int32())]
        + [(name, pa.string()) for name in (
            "reference_protein_id", "reference_activity_id", "canonical_ec", "ec_l1",
            "ec_l2", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier",
            "reference_role", "activity_provenance",
        )]
    )


def execute(protein_ledger, activity_source, output):
    import pyarrow as pa
    import pyarrow.parquet as pq

    protein_ledger, activity_source, output = Path(protein_ledger).resolve(), Path(activity_source).resolve(), Path(output).resolve()
    require(protein_ledger.is_file() and activity_source.is_file(), "library inputs")
    require(output.parent.is_dir() and not output.exists(), "exclusive output")
    output.mkdir(exist_ok=False)
    inputs = {
        "protein_ledger": {"path": str(protein_ledger), "sha256": sha(protein_ledger)},
        "activity_source": {"path": str(activity_source), "sha256": sha(activity_source)},
    }
    emit(output / "reservation.json", {"status": "ONE_S4D_TRAIN_LIBRARY_ATTEMPT_RESERVED", "inputs": inputs, "automatic_retry": False})
    state, error = "FAIL_CLOSED", None
    try:
        proteins_table = pq.read_table(protein_ledger)
        require(proteins_table.column_names == PROTEIN_COLUMNS, "protein ledger schema")
        protein_rows = proteins_table.to_pylist()
        protein_map = {}
        for row in protein_rows:
            require(row["protein_id"] not in protein_map, "duplicate protein ledger")
            protein_map[row["protein_id"]] = row
        activity_table = pq.read_table(activity_source, columns=ACTIVITY_COLUMNS)
        require(activity_table.column_names == ACTIVITY_COLUMNS, "activity projection schema")
        selected, activity_ids, counts = [], set(), Counter()
        for row in activity_table.to_pylist():
            counts["source_rows"] += 1
            activity = row["activity_id"]
            require(isinstance(activity, str) and activity and activity not in activity_ids, "source activity identity")
            activity_ids.add(activity)
            protein = protein_map.get(row["protein_id"])
            if protein is None:
                counts["excluded_outside_benchmark"] += 1
            elif protein["role"] != "TRAIN":
                counts["excluded_nontrain"] += 1
            elif row["evidence_tier"] not in core.EVIDENCE_TIERS:
                counts["excluded_evidence_tier"] += 1
            elif not isinstance(row["canonical_ec"], str) or not row["canonical_ec"]:
                counts["excluded_missing_canonical_ec"] += 1
            else:
                selected.append(row)
                counts["selected_train_activities"] += 1
        library = core.build_train_library(selected, protein_rows)
        require(len(library) == counts["selected_train_activities"], "selected library census")
        pq.write_table(pa.Table.from_pylist(library, schema=library_schema()), output / "train_activity_reference_library.parquet", compression="zstd")
        summary = {
            "status": "PASS_S4D_TRAIN_ACTIVITY_LIBRARY_PENDING_INDEPENDENT_AUDIT",
            "inputs": inputs,
            "projection": ACTIVITY_COLUMNS,
            "counts": {name: counts[name] for name in (
                "source_rows", "selected_train_activities", "excluded_outside_benchmark",
                "excluded_nontrain", "excluded_evidence_tier", "excluded_missing_canonical_ec",
            )},
            "reference_nodes_with_activities": len({row["reference_node"] for row in library}),
            "reference_proteins_with_activities": len({row["reference_protein_id"] for row in library}),
            "all_activity_projection_scanned_before_role_filter": True,
            "only_train_rows_materialized": True,
            "query_truth_table_read": False,
            "retest_evaluation_labels_read": False,
            "candidate_retrieval_or_ranking_influenced": False,
            "pair_generation_started": False,
            "training_started": False,
        }
        emit(output / "producer_summary.json", summary)
        state = summary["status"]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    emit(output / "terminal.json", {"status": state, "error": error, "automatic_retry": False})
    if error:
        raise RuntimeError(error["message"])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein-ledger", type=Path, required=True)
    parser.add_argument("--activity-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.protein_ledger, args.activity_source, args.output)


if __name__ == "__main__":
    main()

