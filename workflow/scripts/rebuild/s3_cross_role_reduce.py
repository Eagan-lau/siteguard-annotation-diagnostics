"""Fail-closed reducer for complete S3 cross-role sequence-search receipts."""
import hashlib
import importlib.util
import json
import sys
import traceback
from collections import Counter
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
        and contract["authorization"]["reduce"],
        "unarmed S3 cross-role reduction",
    )
    require(list(sys.version_info[:3]) == contract["python_version"], "Python identity")
    contract_sha = sha(contract_path)
    root = Path(contract["root"])
    require(str(root.resolve()) == contract["physical_root"], "root identity")
    output = root / contract["reduction_output"]
    require(not output.exists(), "reduction output exists")
    output.mkdir(exist_ok=False)
    emit(output / "reservation.json", {"status": "ONE_S3_CROSS_ROLE_REDUCTION_ATTEMPT_RESERVED", "contract_sha256": contract_sha})
    status = "FAIL_CLOSED"
    error = None
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for name, expected in contract["scripts"].items():
            require(sha(HERE / name) == expected, "script identity " + name)
        for evidence in contract["evidence"]:
            path = Path(evidence["path"])
            require(path.is_file() and sha(path) == evidence["sha256"], "evidence identity")
        setup = root / contract["setup_output"]
        setup_pass = json.loads((setup / "S3_CROSS_ROLE_SETUP_PASS.json").read_text())
        setup_terminal = json.loads((setup / "terminal.json").read_text())
        require(setup_pass["status"] == setup_terminal["status"] == "PASS_S3_CROSS_ROLE_SETUP", "setup state")
        require(setup_pass["contract_sha256"] == setup_terminal["contract_sha256"] == contract_sha, "setup contract")
        tasks = pq.read_table(setup / "task_plan.parquet").to_pylist()
        workers = pq.read_table(setup / "worker_plan.parquet").to_pylist()
        node_split = pq.read_table(root / contract["inputs"]["node_split"], columns=["node_id", "role"]).to_pylist()
        require(len(node_split) == contract["expected"]["nodes"], "node census")
        role_by_node = {row["node_id"]: row["role"] for row in node_split}
        require(len(role_by_node) == len(node_split), "duplicate split node")

        all_receipts = []
        query_status_seen = set()
        query_aggregates = Counter()
        edge_rows = []
        workers_root = root / contract["worker_output"]
        diagnostic_names = (
            "ambiguity_rows",
            "engine_literal_disagreements",
            "engine_literal_threshold_flips",
            "serialized_nident_disagreements",
        )
        for worker in workers:
            index = worker["worker_index"]
            folder = workers_root / f"worker_{index:03d}"
            terminal = json.loads((folder / "terminal.json").read_text())
            checkpoint = json.loads((folder / "worker_checkpoint.json").read_text())
            require(terminal["status"] == checkpoint["status"] == "PASS_S3_CROSS_ROLE_WORKER", "worker state")
            require(
                terminal["worker_index"] == checkpoint["worker_index"] == index
                and terminal["contract_sha256"] == checkpoint["contract_sha256"] == contract_sha,
                "worker identity",
            )
            require(checkpoint["python_version"] == contract["python_version"], "worker Python identity")
            require(checkpoint["mmseqs_sha256"] == contract["mmseqs_sha256"], "worker MMseqs identity")
            require(checkpoint["identity_estimator"] == "literal_identical_non_gap_characters_over_gapped_alignment_columns", "identity estimator")
            require(not checkpoint["functional_labels_read"] and not checkpoint["pair_generation_started"] and not checkpoint["training_started"], "worker scope")
            receipts = pq.read_table(folder / "task_receipts.parquet").to_pylist()
            require(len(receipts) == worker["task_count"] == checkpoint["tasks"], "worker task census")
            require(sum(row["qualifying_cross_role_edges"] for row in receipts) == checkpoint["edge_rows"], "worker edge census")
            require(
                all(sum(row[name] for row in receipts) == checkpoint["identity_diagnostics"][name] for name in diagnostic_names),
                "worker identity diagnostics",
            )
            all_receipts.extend(receipts)
            query_status = pq.read_table(folder / "query_status.parquet").to_pylist()
            require(len(query_status) == worker["query_count"] * 2, "query status census")
            for row in query_status:
                key = row["query_node"], row["search_pass"]
                require(key not in query_status_seen, "duplicate query status")
                require(role_by_node[row["query_node"]] == row["query_role"] == worker["query_role"], "query status role")
                require(row["search_pass"] in core.PASSES, "query status pass")
                require(
                    0 <= row["qualifying_cross_role_edges"] <= row["alignments_returned"] <= row["prefilter_pairs"],
                    "query status counts",
                )
                query_status_seen.add(key)
                for field in ("prefilter_pairs", "alignments_returned", "qualifying_cross_role_edges"):
                    query_aggregates[(index, row["search_pass"], field)] += row[field]
            local_edges = pq.read_table(folder / "cross_role_edges.parquet").to_pylist()
            require(len(local_edges) == checkpoint["edge_rows"], "worker edge rows")
            for row in local_edges:
                require(row["worker_index"] == index, "edge worker")
                require(role_by_node[row["query_node"]] == row["query_role"], "edge query role")
                require(role_by_node[row["target_node"]] == row["target_role"], "edge target role")
                require(row["query_role"] != row["target_role"], "within-role edge")
                require(row["nident"] * 10 >= row["alnlen"] * 3, "edge identity")
                require(row["qspan"] * 10 >= row["qlen"] * 7 and row["tspan"] * 10 >= row["tlen"] * 7, "edge coverage")
            edge_rows.extend(local_edges)

        audit = core.audit_receipts(tasks, all_receipts)
        require(audit["all_twenty_ordered_role_pairs"] and audit["both_search_passes"], "receipt coverage")
        require(query_status_seen == {(row["node_id"], search_pass) for row in node_split for search_pass in core.PASSES}, "query status universe")
        for worker in workers:
            related = [row for row in all_receipts if row["worker_index"] == worker["worker_index"]]
            for search_pass in core.PASSES:
                selected = [row for row in related if row["search_pass"] == search_pass]
                for field in ("prefilter_pairs", "alignments_returned", "qualifying_cross_role_edges"):
                    require(
                        query_aggregates[(worker["worker_index"], search_pass, field)] == sum(row[field] for row in selected),
                        "query/task aggregate",
                    )
        require(len(edge_rows) == audit["qualifying_cross_role_edges"], "global edge census")
        edge_schema = pa.schema(
            [(name, pa.int32()) for name in ("query_node", "target_node", "nident", "alnlen", "qspan", "qlen", "tspan", "tlen")]
            + [
                ("worker_index", pa.int32()),
                ("query_role", pa.string()),
                ("query_shard", pa.int16()),
                ("target_role", pa.string()),
                ("target_shard", pa.int16()),
                ("search_pass", pa.string()),
            ]
        )
        pq.write_table(pa.Table.from_pylist(edge_rows, schema=edge_schema), output / "cross_role_edges.parquet", compression="zstd")
        status = (
            "PASS_S3_CROSS_ROLE_REDUCTION_PENDING_INDEPENDENT_AUDIT"
            if audit["qualifying_cross_role_edges"] == 0
            else "FAIL_S3_CROSS_ROLE_LEAKAGE_DETECTED"
        )
        summary = {
            "status": status,
            "contract_sha256": contract_sha,
            "workers": len(workers),
            "task_pass_units": len(all_receipts),
            "query_status_rows": len(query_status_seen),
            "qualifying_cross_role_edges": len(edge_rows),
            "all_twenty_ordered_role_pairs": audit["all_twenty_ordered_role_pairs"],
            "both_search_passes": audit["both_search_passes"],
            "identity_estimator": "literal_identical_non_gap_characters_over_gapped_alignment_columns",
            "identity_diagnostics": {name: sum(row[name] for row in all_receipts) for name in diagnostic_names},
            "functional_labels_read": False,
            "pair_generation_authorized": False,
            "training_authorized": False,
        }
        emit(output / "reduction_summary.json", summary)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(error["traceback"], file=sys.stderr)
    emit(output / "terminal.json", {"status": status, "error": error, "contract_sha256": contract_sha, "automatic_retry": False})
    if error or status != "PASS_S3_CROSS_ROLE_REDUCTION_PENDING_INDEPENDENT_AUDIT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
