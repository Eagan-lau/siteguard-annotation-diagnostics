"""Prepare a label-free execution plan for the S3 direct cross-role search."""
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s3_cross_role_core", HERE / "s3_cross_role_core.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


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


def main():
    contract_path = Path(sys.argv[1])
    contract = json.loads(contract_path.read_text())
    require(
        contract["status"] == "ARMED_S3_CROSS_ROLE_SEARCH"
        and contract["authorization"]["prepare"],
        "unarmed S3 cross-role preparation",
    )
    require(
        not contract["authorization"]["functional_labels"]
        and not contract["authorization"]["pair_generation"]
        and not contract["authorization"]["training"],
        "scope",
    )
    require(list(sys.version_info[:3]) == contract["python_version"], "Python identity")
    contract_sha = sha(contract_path)
    root = Path(contract["root"])
    require(str(root.resolve()) == contract["physical_root"], "root identity")
    for name, expected in contract["scripts"].items():
        require(sha(HERE / name) == expected, "script identity " + name)
    for evidence in contract["evidence"]:
        path = Path(evidence["path"])
        require(path.is_file() and sha(path) == evidence["sha256"], "evidence identity")

    output = root / contract["setup_output"]
    require(output.parent.is_dir() and not output.exists(), "setup output identity")
    output.mkdir(exist_ok=False)
    emit(output / "reservation.json", {"status": "ONE_S3_CROSS_ROLE_SETUP_ATTEMPT_RESERVED", "contract_sha256": contract_sha})
    status = "FAIL_CLOSED"
    error = None
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        node_split = pq.read_table(root / contract["inputs"]["node_split"], columns=["node_id", "role"]).to_pylist()
        node_manifest = pq.read_table(
            root / contract["inputs"]["node_manifest"],
            columns=["node_id", "sequence_sha256", "sequence_length"],
        ).to_pylist()
        require(len(node_split) == len(node_manifest) == contract["expected"]["nodes"], "node census")
        manifest_ids = [row["node_id"] for row in node_manifest]
        require(sorted(manifest_ids) == list(range(len(manifest_ids))) and len(set(manifest_ids)) == len(manifest_ids), "node manifest")
        require({row["node_id"] for row in node_split} == set(manifest_ids), "split node universe")

        workers, tasks, summary = core.worker_plan(
            node_split,
            query_shard_size=contract["query_shard_size"],
            target_shard_size=contract["target_shard_size"],
        )
        require(summary["ordered_role_pairs"] == 20 and summary["search_passes"] == 2, "search coverage")
        require(summary["workers"] == len(workers) and summary["task_pass_units"] == len(tasks), "plan census")
        require(summary == contract["plan"], "locked execution plan")

        worker_schema = pa.schema(
            [
                ("worker_index", pa.int32()),
                ("query_role", pa.string()),
                ("query_shard", pa.int16()),
                ("query_count", pa.int32()),
                ("task_count", pa.int32()),
            ]
        )
        task_schema = pa.schema(
            [
                ("worker_index", pa.int32()),
                ("query_role", pa.string()),
                ("query_shard", pa.int16()),
                ("query_count", pa.int32()),
                ("target_role", pa.string()),
                ("target_shard", pa.int16()),
                ("target_count", pa.int32()),
                ("search_pass", pa.string()),
            ]
        )
        pq.write_table(pa.Table.from_pylist(workers, schema=worker_schema), output / "worker_plan.parquet", compression="zstd")
        pq.write_table(pa.Table.from_pylist(tasks, schema=task_schema), output / "task_plan.parquet", compression="zstd")
        (root / contract["worker_output"]).mkdir(exist_ok=False)
        result = {
            "status": "PASS_S3_CROSS_ROLE_SETUP",
            "contract_sha256": contract_sha,
            **summary,
            "identity_estimator": "literal_identical_non_gap_characters_over_gapped_alignment_columns",
            "functional_labels_read": False,
            "pair_generation_started": False,
            "training_started": False,
        }
        emit(output / "plan_summary.json", result)
        emit(output / "S3_CROSS_ROLE_SETUP_PASS.json", result)
        status = result["status"]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(error["traceback"], file=sys.stderr)
    emit(output / "terminal.json", {"status": status, "error": error, "contract_sha256": contract_sha, "automatic_retry": False})
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
