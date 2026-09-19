"""Stream native retrieval TSV into bounded-memory S4B Parquet outputs."""
import argparse
import hashlib
import importlib.util
import json
import math
import re
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("s4b_stream_retention_writer", HERE / "s4b_stream_retention.py")
retention = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retention)

RAW_NAMES = (
    "query_node", "reference_node", "fident", "alnlen", "qstart", "qend", "qlen",
    "tstart", "tend", "tlen", "qcov", "tcov", "evalue", "bits",
)
INTEGER_NAMES = {"query_node", "reference_node", "alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen"}
NODE_COLUMNS = [
    "node_id", "component_id", "role", "sequence_sha256", "sequence_length",
    "representative_protein_id", "protein_id_count",
]
CANDIDATE_COLUMNS = list(RAW_NAMES) + [
    "query_role", "reference_role", "reference_component_id", "raw_rank",
    "modality_rank", "modality", "candidate_provenance",
]
STATUS_COLUMNS = [
    "query_node", "query_role", "modality", "status", "raw_hits", "self_hits_removed",
    "component_duplicate_or_beyond_top_k", "retained_candidates", "query_truth_read",
    "modality_input_available", "unavailability_reason",
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


def iter_raw_tsv(path):
    with path.open("r", encoding="ascii", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").rstrip("\r").split("\t")
            require(len(fields) == len(RAW_NAMES) and all(fields), f"raw TSV fields line {line_number}")
            row = {}
            for name, value in zip(RAW_NAMES, fields):
                if name in INTEGER_NAMES:
                    require(re.fullmatch(r"\d+", value) is not None, f"raw integer {name} line {line_number}")
                    row[name] = int(value)
                else:
                    parsed = float(value)
                    require(math.isfinite(parsed), f"raw finite {name} line {line_number}")
                    row[name] = parsed
            yield row


class BufferedParquetSink:
    def __init__(self, path, schema, batch_rows):
        import pyarrow.parquet as pq

        require(type(batch_rows) is int and batch_rows > 0, "batch rows")
        self.schema = schema
        self.batch_rows = batch_rows
        self.rows = []
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.count = 0

    def __call__(self, row):
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


def schemas():
    import pyarrow as pa

    candidate = pa.schema(
        [("query_node", pa.int32()), ("reference_node", pa.int32()), ("fident", pa.float64())]
        + [(name, pa.int32()) for name in ("alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen")]
        + [(name, pa.float64()) for name in ("qcov", "tcov", "evalue", "bits")]
        + [
            ("query_role", pa.string()), ("reference_role", pa.string()),
            ("reference_component_id", pa.int32()), ("raw_rank", pa.int32()),
            ("modality_rank", pa.int16()), ("modality", pa.string()),
            ("candidate_provenance", pa.string()),
        ]
    )
    status = pa.schema(
        [
            ("query_node", pa.int32()), ("query_role", pa.string()), ("modality", pa.string()),
            ("status", pa.string()), ("raw_hits", pa.int64()), ("self_hits_removed", pa.int64()),
            ("component_duplicate_or_beyond_top_k", pa.int64()), ("retained_candidates", pa.int32()),
            ("query_truth_read", pa.bool_()), ("modality_input_available", pa.bool_()),
            ("unavailability_reason", pa.string()),
        ]
    )
    return candidate, status


def execute(node_ledger, raw_hits, query_role, modality, top_k, output, unavailable_path=None, batch_rows=100000):
    import pyarrow.parquet as pq

    node_ledger = Path(node_ledger).resolve()
    raw_hits = Path(raw_hits).resolve()
    output = Path(output).resolve()
    unavailable_path = Path(unavailable_path).resolve() if unavailable_path is not None else None
    require(node_ledger.is_file() and raw_hits.is_file(), "producer inputs")
    require(output.parent.is_dir() and not output.exists(), "exclusive output")
    output.mkdir(exist_ok=False)
    inputs = {"node_ledger": {"path": str(node_ledger), "sha256": sha(node_ledger)}, "raw_hits": {"path": str(raw_hits), "sha256": sha(raw_hits)}}
    if unavailable_path is not None:
        require(unavailable_path.is_file(), "unavailability input")
        inputs["unavailability"] = {"path": str(unavailable_path), "sha256": sha(unavailable_path)}
    emit(output / "reservation.json", {"status": "ONE_S4B_STREAM_WRITER_ATTEMPT_RESERVED", "inputs": inputs, "automatic_retry": False})
    state, error, candidate_sink, status_sink = "FAIL_CLOSED", None, None, None
    try:
        nodes_table = pq.read_table(node_ledger)
        require(nodes_table.column_names == NODE_COLUMNS, "node ledger schema")
        node_rows = nodes_table.to_pylist()
        unavailable = {}
        if unavailable_path is not None:
            table = pq.read_table(unavailable_path)
            require(table.column_names == ["query_node", "reason"], "unavailability schema")
            for row in table.to_pylist():
                require(row["query_node"] not in unavailable, "duplicate unavailable query")
                unavailable[row["query_node"]] = row["reason"]
        candidate_schema, status_schema = schemas()
        candidate_sink = BufferedParquetSink(output / "candidates.parquet", candidate_schema, batch_rows)
        status_sink = BufferedParquetSink(output / "query_status.parquet", status_schema, batch_rows)
        totals = retention.stream_retain(
            iter_raw_tsv(raw_hits), node_rows, query_role, modality, top_k,
            candidate_sink, status_sink, unavailable,
        )
        candidate_sink.close()
        status_sink.close()
        require(candidate_sink.count == totals["retained_candidates"] and status_sink.count == totals["queries"], "writer census")
        summary = {
            "status": "PASS_S4B_STREAM_WRITER_PENDING_INDEPENDENT_AUDIT",
            "query_role": query_role,
            "modality": modality,
            "top_k": top_k,
            "batch_rows": batch_rows,
            "inputs": inputs,
            "totals": totals,
            "functional_labels_read": False,
            "query_truth_read": False,
            "activity_expansion_started": False,
            "pair_generation_started": False,
            "training_started": False,
        }
        emit(output / "producer_summary.json", summary)
        state = summary["status"]
    except BaseException as exc:
        for sink in (candidate_sink, status_sink):
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
    parser.add_argument("--node-ledger", type=Path, required=True)
    parser.add_argument("--raw-hits", type=Path, required=True)
    parser.add_argument("--query-role", choices=retention.core.ROLES, required=True)
    parser.add_argument("--modality", choices=retention.core.MODALITIES, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unavailable", type=Path)
    parser.add_argument("--batch-rows", type=int, default=100000)
    args = parser.parse_args()
    execute(args.node_ledger, args.raw_hits, args.query_role, args.modality, args.top_k, args.output, args.unavailable, args.batch_rows)


if __name__ == "__main__":
    main()

