"""Production worker for one frozen S4M direct-alignment shard."""
import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("s4m_uniform_direct_alignment_core_worker", HERE / "s4m_uniform_direct_alignment_core.py")
core = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(core)

FOLDSEEK_COLUMNS = ["node_id", "component_id", "role", "protein_id", "db_key", "structure_name"]


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
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")


def file_record(path):
    require(path.is_file() and not path.is_symlink(), "required regular file " + path.name)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def read_mmseqs_mapping(path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, fields in enumerate(csv.reader(handle, delimiter="\t"), 1):
            require(len(fields) == 3 and all(fields), f"MMseqs lookup fields line {line_number}")
            try:
                key, node, part = int(fields[0]), int(fields[1]), int(fields[2])
            except ValueError as exc:
                raise RuntimeError(f"MMseqs lookup integer line {line_number}") from exc
            require(min(key, node, part) >= 0, f"MMseqs lookup range line {line_number}")
            rows.append({"db_key": key, "node_id": node, "output_name": fields[1]})
    return rows


def read_foldseek_mapping(path):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    require(parquet.schema_arrow.names == FOLDSEEK_COLUMNS, "Foldseek selection schema")
    rows = []
    for batch in parquet.iter_batches(columns=FOLDSEEK_COLUMNS):
        for row in batch.to_pylist():
            require(row["role"] in {"TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST"}, "Foldseek role")
            if row["role"] == "TRAIN":
                rows.append({"db_key": row["db_key"], "node_id": row["node_id"], "output_name": row["structure_name"]})
    return rows


def result_schema(modality):
    import pyarrow as pa
    fields = [
        ("modality", pa.string()), ("shard", pa.int16()),
        ("query_node", pa.int32()), ("reference_node", pa.int32()),
        ("query_db_key", pa.int64()), ("reference_db_key", pa.int64()),
        ("fident", pa.float64()), ("alnlen", pa.int64()),
        ("qstart", pa.int64()), ("qend", pa.int64()), ("qlen", pa.int64()),
        ("tstart", pa.int64()), ("tend", pa.int64()), ("tlen", pa.int64()),
        ("qcov", pa.float64()), ("tcov", pa.float64()),
        ("evalue", pa.float64()), ("bits", pa.float64()),
    ]
    if modality == "foldseek":
        fields.extend((name, pa.float64()) for name in core.FOLDSEEK_COLUMNS)
    return pa.schema(fields)


def write_result(path, modality, shard, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = result_schema(modality)
    output_rows, digest = [], hashlib.sha256()
    for source in rows:
        row = {"modality": modality, "shard": shard, **source}
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big")); digest.update(encoded)
        output_rows.append(row)
    pq.write_table(pa.Table.from_pylist(output_rows, schema=schema), path, compression="zstd")
    return digest.hexdigest()


def execute(modality, shard, prefilter, mapping, database, tool, output, scratch_root, threads=8, runner=subprocess.run):
    require(modality in {"mmseqs", "foldseek"}, "modality")
    require(type(shard) is int and not isinstance(shard, bool) and 0 <= shard < 64, "shard")
    require(type(threads) is int and not isinstance(threads, bool) and threads > 0, "threads")
    prefilter, mapping, database, tool, output, scratch_root = (
        Path(prefilter).resolve(), Path(mapping).resolve(), Path(database).resolve(), Path(tool).resolve(),
        Path(output).resolve(), Path(scratch_root).resolve(),
    )
    require(output.parent.is_dir() and not output.exists(), "exclusive output")
    require(scratch_root.is_dir(), "scratch root")
    dbtype = Path(str(database) + ".dbtype")
    inputs = {"prefilter": file_record(prefilter), "mapping": file_record(mapping), "database_dbtype": file_record(dbtype), "tool": file_record(tool)}
    output.mkdir(exist_ok=False)
    emit(output / "reservation.json", {
        "status": "ONE_S4M_DIRECT_ALIGNMENT_WORKER_ATTEMPT_RESERVED",
        "unit": {"modality": modality, "shard": shard}, "inputs": inputs,
        "threads": threads, "automatic_retry": False,
    })
    state, error, summary, receipts = "FAIL_CLOSED", None, None, []
    try:
        mapping_rows = read_mmseqs_mapping(mapping) if modality == "mmseqs" else read_foldseek_mapping(mapping)
        prefilter_lines = prefilter.read_text(encoding="ascii").splitlines(keepends=True)
        expected, _ = core.expected_requests(prefilter_lines, mapping_rows)
        scratch = scratch_root / f"s4m_{modality}_{shard:03d}"
        require(not scratch.exists(), "exclusive worker scratch")
        scratch.mkdir(exist_ok=False)
        scratch_prefilter = scratch / "prefilter.tsv"
        shutil.copyfile(prefilter, scratch_prefilter)
        raw_scratch = scratch / "raw.tsv"
        if expected:
            commands = core.command_plan(str(tool), modality, str(database), str(scratch / "prefilter"), str(scratch / "alignment"), str(raw_scratch), threads)
            for index, command in enumerate(commands):
                completed = runner(command, capture_output=True, text=True)
                receipt = {"index": index, "argv": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
                receipts.append(receipt)
                require(completed.returncode == 0, f"direct alignment command {index}")
            require(raw_scratch.is_file() and not raw_scratch.is_symlink(), "raw alignment output")
        else:
            with raw_scratch.open("x", encoding="ascii", newline="\n"):
                pass
        raw_output = output / "raw_alignments.tsv"
        shutil.copyfile(raw_scratch, raw_output)
        raw_lines = raw_output.read_text(encoding="ascii").splitlines(keepends=True)
        emit(output / "command_receipt.json", {
            "status": "DIRECT_ALIGNMENT_COMMANDS_COMPLETE" if expected else "EMPTY_SHARD_NO_COMMANDS_REQUIRED",
            "unit": {"modality": modality, "shard": shard}, "commands": receipts,
        })
        if modality == "foldseek" and expected:
            for receipt in receipts[1:]:
                require("--exact-tmscore" in receipt["argv"] and receipt["argv"][receipt["argv"].index("--exact-tmscore") + 1] == "1", "explicit exact TM flag")
                lines = (receipt["stdout"] + "\n" + receipt["stderr"]).splitlines()
                flags = [line.split()[-1] for line in lines if line.strip().startswith("Exact TMscore")]
                require(flags == ["1"], "actual exact TM mode")
        normalized = core.normalize(raw_lines, modality, expected)
        result = output / "normalized_alignments.parquet"
        row_digest = write_result(result, modality, shard, normalized)
        require(all(sha(path) == item["sha256"] for path, item in (
            (prefilter, inputs["prefilter"]), (mapping, inputs["mapping"]),
            (dbtype, inputs["database_dbtype"]), (tool, inputs["tool"]),
        )), "worker input stability")
        summary = {
            "status": "PASS_S4M_DIRECT_ALIGNMENT_WORKER_PENDING_INDEPENDENT_AUDIT",
            "unit": {"modality": modality, "shard": shard}, "inputs": inputs,
            "requested_node_pairs": len(expected), "normalized_alignment_rows": len(normalized),
            "normalized_row_sha256": row_digest, "raw_file_sha256": sha(raw_output),
            "normalized_file_sha256": sha(result), "empty_shard": not expected,
            "commands_executed": len(receipts), "direct_known_pair_measurement": True,
            "candidate_search_performed": False, "functional_labels_read": False,
            "query_truth_read": False, "sampling_probabilities_read": False,
            "training_weights_read": False, "retrieval_metrics_read": False,
            "feature_matrix_created": False, "preprocessing_started": False,
            "training_started": False, "evaluation_started": False,
        }
        emit(output / "worker_summary.json", summary); state = summary["status"]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    if not (output / "command_receipt.json").exists():
        emit(output / "command_receipt.json", {
            "status": "FAIL_CLOSED_BEFORE_COMMAND_COMPLETION",
            "unit": {"modality": modality, "shard": shard}, "commands": receipts,
        })
    emit(output / "terminal.json", {"status": state, "error": error, "automatic_retry": False})
    if error:
        raise RuntimeError(error["message"])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=("mmseqs", "foldseek"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--prefilter", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    execute(args.modality, args.shard, args.prefilter, args.mapping, args.database, args.tool, args.output, args.scratch_root, args.threads)


if __name__ == "__main__":
    main()

