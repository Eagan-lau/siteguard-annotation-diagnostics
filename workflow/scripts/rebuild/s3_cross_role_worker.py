"""Run one complete S3 query-role shard against every other role."""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


alignment = load("sc2_alignment", "sc2_alignment.py")
monitor = load("sc2_monitor", "sc2_monitor.py")
core = load("s3_cross_role_core", "s3_cross_role_core.py")


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


def fasta(path, sequences):
    with path.open("x", encoding="ascii", newline="\n") as handle:
        for key in sorted(sequences, key=lambda value: int(value[1:])):
            handle.write(">" + key + "\n" + sequences[key] + "\n")


def execute(argv, logdir, tag, scratch, deadline, commands):
    started = time.monotonic()
    stdout = logdir / (tag + ".stdout")
    stderr = logdir / (tag + ".stderr")
    with stdout.open("xb") as out_handle, stderr.open("xb") as err_handle:
        process = subprocess.Popen([str(value) for value in argv], stdout=out_handle, stderr=err_handle, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() > min(started + 600, deadline):
                    raise RuntimeError("worker/command time budget")
                size, count = monitor.size_tree(scratch)
                if size > 20 * 1024**3 or count > 15000:
                    raise RuntimeError("worker scratch budget")
                time.sleep(1)
        except BaseException:
            import signal

            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    stdout_text = stdout.read_text(errors="replace")
    stderr_text = stderr.read_text(errors="replace")
    commands.append(
        {
            "argv": [str(value) for value in argv],
            "returncode": process.returncode,
            "wall_seconds": time.monotonic() - started,
            "stdout_sha256": sha(stdout),
            "stderr_sha256": sha(stderr),
            "stdout_tail": stdout_text[-2000:],
            "stderr_tail": stderr_text[-2000:],
        }
    )
    require(process.returncode == 0, "command failed " + tag)
    return stdout_text + stderr_text


def db_keys(prefix):
    rows = [line.split("\t") for line in Path(str(prefix) + ".index").read_text().splitlines()]
    require(all(len(row) == 3 for row in rows), "DB index")
    keys = [int(row[0]) for row in rows]
    require(len(keys) == len(set(keys)), "duplicate DB key")
    return set(keys)


def create_db(mmseqs, sequences, folder, name, threads, scratch, deadline, commands):
    source = folder / (name + ".fasta")
    fasta(source, sequences)
    database = folder / name
    execute(
        [mmseqs, "createdb", source, database, "--dbtype", "1", "--shuffle", "0", "--compressed", "0", "-v", "3"],
        folder,
        name + "_createdb",
        scratch,
        deadline,
        commands,
    )
    require(len(db_keys(database)) == len(sequences), "createdb lost sequences")
    return database


def prefilter_args(mmseqs, query_db, target_db, output, cap, mask, threads):
    return [
        mmseqs,
        "prefilter",
        query_db,
        target_db,
        output,
        "-s",
        "7.5",
        "-k",
        "0",
        "--max-seqs",
        str(cap),
        "--mask",
        str(mask),
        "--comp-bias-corr",
        str(mask),
        "--split",
        "1",
        "--split-memory-limit",
        "24G",
        "-c",
        "0.69",
        "--cov-mode",
        "0",
        "--threads",
        str(threads),
        "--compressed",
        "0",
        "-v",
        "3",
    ]


def align_args(mmseqs, query_db, target_db, prefilter, output, cap, mask, threads):
    return [
        mmseqs,
        "align",
        query_db,
        target_db,
        prefilter,
        output,
        "--alignment-mode",
        "3",
        "--seq-id-mode",
        "0",
        "-a",
        "1",
        "--min-seq-id",
        "0.29",
        "-c",
        "0.69",
        "--cov-mode",
        "0",
        "-e",
        "10",
        "--max-accept",
        str(cap),
        "--max-rejected",
        str(cap),
        "--comp-bias-corr",
        str(mask),
        "--alt-ali",
        "0",
        "--realign",
        "0",
        "--threads",
        str(threads),
        "--compressed",
        "0",
        "-v",
        "3",
    ]


def search_task(mmseqs, query_db, target_db, query_sequences, target_sequences, folder, mask, threads, scratch, deadline, commands):
    folder.mkdir(exist_ok=False)
    cap = len(target_sequences) + 1
    prefilter = folder / "pref"
    execute(prefilter_args(mmseqs, query_db, target_db, prefilter, cap, mask, threads), folder, "prefilter", scratch, deadline, commands)
    require(db_keys(prefilter) == db_keys(query_db), "prefilter missing zero-hit query record")
    prefilter_tsv = folder / "prefilter.tsv"
    execute(
        [mmseqs, "createtsv", query_db, target_db, prefilter, prefilter_tsv, "--threads", str(threads), "-v", "3"],
        folder,
        "prefilter_export",
        scratch,
        deadline,
        commands,
    )
    prefilter_counts, prefilter_pairs = alignment.prefilter_counts(
        prefilter_tsv, set(query_sequences), set(target_sequences), cap
    )
    aligned = folder / "aln"
    logs = execute(
        align_args(mmseqs, query_db, target_db, prefilter, aligned, cap, mask, threads),
        folder,
        "align",
        scratch,
        deadline,
        commands,
    )
    calculated = re.findall(r"(\d+) alignments calculated", logs)
    require(len(calculated) == 1 and int(calculated[0]) == len(prefilter_pairs), "not all prefilter pairs aligned")
    alignment_tsv = folder / "alignments.tsv"
    execute(
        [
            mmseqs,
            "convertalis",
            query_db,
            target_db,
            aligned,
            alignment_tsv,
            "--format-output",
            ",".join(alignment.FIELDS),
            "--threads",
            str(threads),
            "-v",
            "3",
        ],
        folder,
        "alignment_export",
        scratch,
        deadline,
        commands,
    )
    all_sequences = {**query_sequences, **target_sequences}
    returned = set()
    alignment_counts = Counter()
    edge_counts = Counter()
    diagnostics = Counter()
    edges = []
    with alignment_tsv.open(encoding="ascii") as handle:
        for line in handle:
            row = alignment.parse_alignment(line, all_sequences)
            pair = row["query"], row["target"]
            require(pair in prefilter_pairs and pair not in returned, "alignment pair provenance")
            returned.add(pair)
            alignment_counts[pair[0]] += 1
            diagnostics["ambiguity_rows"] += row["ambiguity_columns"] > 0
            diagnostics["engine_literal_disagreements"] += row["engine_literal_disagreement"]
            diagnostics["engine_literal_threshold_flips"] += row["engine_literal_threshold_flip"]
            diagnostics["serialized_nident_disagreements"] += row["nident"] != row["nident_literal"]
            if row["qualifies"]:
                query_node = int(pair[0][1:])
                target_node = int(pair[1][1:])
                require(query_node != target_node, "cross-role self edge")
                edge_counts[pair[0]] += 1
                edges.append(
                    {
                        "query_node": query_node,
                        "target_node": target_node,
                        "nident": row["nident_literal"],
                        "alnlen": row["alnlen"],
                        "qspan": row["qspan"],
                        "qlen": row["qlen"],
                        "tspan": row["tspan"],
                        "tlen": row["tlen"],
                    }
                )
    counts = {
        query: (prefilter_counts[query], alignment_counts[query], edge_counts[query])
        for query in query_sequences
    }
    totals = {
        "cap": cap,
        "prefilter_pairs": len(prefilter_pairs),
        "alignments_calculated": int(calculated[0]),
        "alignments_returned": len(returned),
        "qualifying_cross_role_edges": len(edges),
        **{
            name: diagnostics[name]
            for name in (
                "ambiguity_rows",
                "engine_literal_disagreements",
                "engine_literal_threshold_flips",
                "serialized_nident_disagreements",
            )
        },
    }
    return edges, counts, totals


def load_sequences(source, node_manifest):
    import pyarrow.parquet as pq

    table = pq.read_table(source, columns=["protein_id", "sequence"])
    groups, _ = alignment.group_sequences(zip(table["protein_id"].to_pylist(), table["sequence"].to_pylist()))
    hashes = sorted(groups)
    require(len(hashes) == len(node_manifest), "sequence node census")
    by_node = {row["node_id"]: row for row in node_manifest}
    require(set(by_node) == set(range(len(hashes))), "node manifest universe")
    for node, sequence_hash in enumerate(hashes):
        row = by_node[node]
        require(row["sequence_sha256"] == sequence_hash, "sequence hash ordering")
        require(row["sequence_length"] == len(groups[sequence_hash]["sequence"]), "sequence length")
    return {node: groups[sequence_hash]["sequence"] for node, sequence_hash in enumerate(hashes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--worker-index", required=True, type=int)
    args = parser.parse_args()
    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text())
    require(
        contract["status"] == "ARMED_S3_CROSS_ROLE_SEARCH"
        and contract["authorization"]["full_search"],
        "unarmed S3 cross-role search",
    )
    require(
        not contract["authorization"]["functional_labels"]
        and not contract["authorization"]["pair_generation"]
        and not contract["authorization"]["training"],
        "scope",
    )
    require(list(sys.version_info[:3]) == contract["python_version"], "Python identity")
    contract_sha = sha(contract_path)
    mmseqs = Path(contract["mmseqs"])
    require(mmseqs.is_file() and sha(mmseqs) == contract["mmseqs_sha256"], "MMseqs identity")
    root = Path(contract["root"])
    require(str(root.resolve()) == contract["physical_root"], "root identity")
    source = root / contract["source"]["path"]
    require(source.stat().st_size == contract["source"]["bytes"] and sha(source) == contract["source"]["sha256"], "source identity")
    for name, expected in contract["scripts"].items():
        require(sha(HERE / name) == expected, "script identity " + name)
    for evidence in contract["evidence"]:
        path = Path(evidence["path"])
        require(path.is_file() and sha(path) == evidence["sha256"], "evidence identity")

    import pyarrow as pa
    import pyarrow.parquet as pq

    setup = root / contract["setup_output"]
    setup_pass = json.loads((setup / "S3_CROSS_ROLE_SETUP_PASS.json").read_text())
    setup_terminal = json.loads((setup / "terminal.json").read_text())
    require(setup_pass["status"] == setup_terminal["status"] == "PASS_S3_CROSS_ROLE_SETUP", "setup state")
    require(setup_pass["contract_sha256"] == setup_terminal["contract_sha256"] == contract_sha, "setup contract")
    worker_plan = pq.read_table(setup / "worker_plan.parquet").to_pylist()
    require(0 <= args.worker_index < len(worker_plan), "worker index")
    worker = worker_plan[args.worker_index]
    require(worker["worker_index"] == args.worker_index, "worker plan ordering")

    node_split = pq.read_table(root / contract["inputs"]["node_split"], columns=["node_id", "role"]).to_pylist()
    node_manifest = pq.read_table(
        root / contract["inputs"]["node_manifest"], columns=["node_id", "sequence_sha256", "sequence_length"]
    ).to_pylist()
    shards = core.role_shards(node_split, contract["query_shard_size"])
    target_shards = core.role_shards(node_split, contract["target_shard_size"])
    query_nodes = shards[worker["query_role"]][worker["query_shard"]]
    require(len(query_nodes) == worker["query_count"], "query plan")
    sequences = load_sequences(source, node_manifest)
    labels = {node: "n" + str(node) for node in sequences}

    output = root / contract["worker_output"] / f"worker_{args.worker_index:03d}"
    require(not output.exists(), "worker output exists")
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    deadline = started + contract["worker_seconds"]
    commands = []
    status = "FAIL_CLOSED"
    error = None
    emit(
        output / "reservation.json",
        {
            "status": "ONE_S3_CROSS_ROLE_WORKER_ATTEMPT_RESERVED",
            "worker_index": args.worker_index,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "contract_sha256": contract_sha,
        },
    )
    try:
        scratch_base = Path(os.environ["LOCALSCRATCH"])
        scratch = scratch_base / f"sg_s3_cross_{os.environ['SLURM_JOB_ID']}_{args.worker_index:03d}"
        require(scratch_base.is_dir() and not scratch.exists(), "scratch identity")
        scratch.mkdir(exist_ok=False)
        query_sequences = {labels[node]: sequences[node] for node in query_nodes}
        query_db = create_db(mmseqs, query_sequences, scratch, "queries", contract["threads"], scratch, deadline, commands)
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
        writer = pq.ParquetWriter(output / "cross_role_edges.parquet", edge_schema, compression="zstd")
        receipts = []
        query_aggregate = {(node, search_pass): [0, 0, 0] for node in query_nodes for search_pass in core.PASSES}
        try:
            for target_role in core.ROLES:
                if target_role == worker["query_role"]:
                    continue
                for target_shard, target_nodes in enumerate(target_shards[target_role]):
                    target_folder = scratch / f"target_{target_role}_{target_shard:03d}"
                    target_folder.mkdir(exist_ok=False)
                    target_sequences = {labels[node]: sequences[node] for node in target_nodes}
                    target_db = create_db(
                        mmseqs, target_sequences, target_folder, "targets", contract["threads"], scratch, deadline, commands
                    )
                    for search_pass, mask in (("masked", 1), ("unmasked", 0)):
                        edges, counts, totals = search_task(
                            mmseqs,
                            query_db,
                            target_db,
                            query_sequences,
                            target_sequences,
                            target_folder / search_pass,
                            mask,
                            contract["threads"],
                            scratch,
                            deadline,
                            commands,
                        )
                        for row in edges:
                            row.update(
                                worker_index=args.worker_index,
                                query_role=worker["query_role"],
                                query_shard=worker["query_shard"],
                                target_role=target_role,
                                target_shard=target_shard,
                                search_pass=search_pass,
                            )
                        if edges:
                            writer.write_table(pa.Table.from_pylist(edges, schema=edge_schema))
                        for query, values in counts.items():
                            aggregate = query_aggregate[(int(query[1:]), search_pass)]
                            for index in range(3):
                                aggregate[index] += values[index]
                        receipt = {
                            "worker_index": args.worker_index,
                            "query_role": worker["query_role"],
                            "query_shard": worker["query_shard"],
                            "query_count": len(query_nodes),
                            "target_role": target_role,
                            "target_shard": target_shard,
                            "target_count": len(target_nodes),
                            "search_pass": search_pass,
                            "status": "PASS",
                            **totals,
                            "prefilter_complete": True,
                            "unknown_truncation": False,
                        }
                        receipts.append(receipt)
                        print(
                            json.dumps(
                                {
                                    "worker_index": args.worker_index,
                                    "query_role": worker["query_role"],
                                    "target_role": target_role,
                                    "target_shard": target_shard,
                                    "search_pass": search_pass,
                                    "edges": len(edges),
                                }
                            ),
                            flush=True,
                        )
        finally:
            writer.close()

        receipt_schema = pa.schema(
            [
                ("worker_index", pa.int32()),
                ("query_role", pa.string()),
                ("query_shard", pa.int16()),
                ("query_count", pa.int32()),
                ("target_role", pa.string()),
                ("target_shard", pa.int16()),
                ("target_count", pa.int32()),
                ("search_pass", pa.string()),
                ("status", pa.string()),
                ("cap", pa.int32()),
                ("prefilter_pairs", pa.int64()),
                ("alignments_calculated", pa.int64()),
                ("alignments_returned", pa.int64()),
                ("qualifying_cross_role_edges", pa.int64()),
                ("ambiguity_rows", pa.int64()),
                ("engine_literal_disagreements", pa.int64()),
                ("engine_literal_threshold_flips", pa.int64()),
                ("serialized_nident_disagreements", pa.int64()),
                ("prefilter_complete", pa.bool_()),
                ("unknown_truncation", pa.bool_()),
            ]
        )
        pq.write_table(pa.Table.from_pylist(receipts, schema=receipt_schema), output / "task_receipts.parquet", compression="zstd")
        statuses = [
            {
                "query_node": node,
                "query_role": worker["query_role"],
                "search_pass": search_pass,
                "prefilter_pairs": values[0],
                "alignments_returned": values[1],
                "qualifying_cross_role_edges": values[2],
            }
            for (node, search_pass), values in sorted(query_aggregate.items())
        ]
        pq.write_table(pa.Table.from_pylist(statuses), output / "query_status.parquet", compression="zstd")
        emit(output / "command_manifest.json", commands)
        diagnostic_names = (
            "ambiguity_rows",
            "engine_literal_disagreements",
            "engine_literal_threshold_flips",
            "serialized_nident_disagreements",
        )
        status = "PASS_S3_CROSS_ROLE_WORKER"
        checkpoint = {
            "status": status,
            "worker_index": args.worker_index,
            "query_role": worker["query_role"],
            "query_shard": worker["query_shard"],
            "query_nodes": len(query_nodes),
            "tasks": len(receipts),
            "edge_rows": sum(row["qualifying_cross_role_edges"] for row in receipts),
            "wall_seconds": time.monotonic() - started,
            "identity_diagnostics": {name: sum(row[name] for row in receipts) for name in diagnostic_names},
            "identity_estimator": "literal_identical_non_gap_characters_over_gapped_alignment_columns",
            "contract_sha256": contract_sha,
            "python_version": list(sys.version_info[:3]),
            "mmseqs_sha256": contract["mmseqs_sha256"],
            "functional_labels_read": False,
            "pair_generation_started": False,
            "training_started": False,
        }
        emit(output / "worker_checkpoint.json", checkpoint)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(error["traceback"], file=sys.stderr)
        if not (output / "command_manifest.json").exists():
            emit(output / "command_manifest.json", commands)
    emit(
        output / "terminal.json",
        {
            "status": status,
            "error": error,
            "worker_index": args.worker_index,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "contract_sha256": contract_sha,
            "automatic_retry": False,
        },
    )
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
