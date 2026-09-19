"""Bounded-memory writer for the truth-free S4C cross-modality union."""
import argparse
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("s4c_union_core_writer", HERE / "s4c_candidate_union_core.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)

CANDIDATE_COLUMNS = [
    "query_node", "reference_node", "fident", "alnlen", "qstart", "qend", "qlen",
    "tstart", "tend", "tlen", "qcov", "tcov", "evalue", "bits", "query_role",
    "reference_role", "reference_component_id", "raw_rank", "modality_rank",
    "modality", "candidate_provenance",
]
INTEGER_METRICS = {"alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen"}


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


def output_schema():
    import pyarrow as pa

    fields = [
        ("query_node", pa.int32()), ("reference_node", pa.int32()),
        ("query_role", pa.string()), ("reference_role", pa.string()),
        ("reference_component_id", pa.int32()), ("mmseqs_rank", pa.int16()),
        ("foldseek_rank", pa.int16()), ("mmseqs_raw_rank", pa.int32()),
        ("foldseek_raw_rank", pa.int32()),
    ]
    for modality in ("mmseqs", "foldseek"):
        for name in core.RAW_METRICS:
            fields.append((f"{modality}_{name}", pa.int32() if name in INTEGER_METRICS else pa.float64()))
    fields.extend(
        [
            ("source_class", pa.string()), ("rrf60_score", pa.float64()),
            ("rrf_constant", pa.int16()), ("candidate_union_provenance", pa.string()),
            ("query_truth_read", pa.bool_()),
        ]
    )
    return pa.schema(fields)


class BufferedParquetSink:
    def __init__(self, path, schema, batch_rows):
        import pyarrow.parquet as pq

        require(type(batch_rows) is int and not isinstance(batch_rows, bool) and batch_rows > 0, "batch rows")
        self.schema = schema
        self.batch_rows = batch_rows
        self.rows = []
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.count = 0

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


def iter_query_groups(path, expected_modality, query_role, batch_rows):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    require(parquet.schema_arrow.names == CANDIDATE_COLUMNS, f"{expected_modality} candidate schema")
    current_query, current_rows, previous_query = None, [], None
    for batch in parquet.iter_batches(batch_size=batch_rows):
        for row in batch.to_pylist():
            query = row["query_node"]
            require(type(query) is int and not isinstance(query, bool) and query >= 0, f"{expected_modality} query identity")
            require(row["modality"] == expected_modality and row["query_role"] == query_role, f"{expected_modality} unit identity")
            if previous_query is not None:
                require(query >= previous_query, f"{expected_modality} query order")
            previous_query = query
            if current_query is None:
                current_query = query
            if query != current_query:
                yield current_query, current_rows
                current_query, current_rows = query, []
            current_rows.append(row)
    if current_query is not None:
        yield current_query, current_rows


def merged_query_groups(mmseqs_path, foldseek_path, query_role, batch_rows):
    mm_iter = iter(iter_query_groups(mmseqs_path, "mmseqs", query_role, batch_rows))
    fs_iter = iter(iter_query_groups(foldseek_path, "foldseek", query_role, batch_rows))
    mm = next(mm_iter, None)
    fs = next(fs_iter, None)
    while mm is not None or fs is not None:
        if fs is None or (mm is not None and mm[0] < fs[0]):
            yield mm[0], mm[1], []
            mm = next(mm_iter, None)
        elif mm is None or fs[0] < mm[0]:
            yield fs[0], [], fs[1]
            fs = next(fs_iter, None)
        else:
            yield mm[0], mm[1], fs[1]
            mm = next(mm_iter, None)
            fs = next(fs_iter, None)


def execute(mmseqs_candidates, foldseek_candidates, query_role, output, rrf_constant=60, batch_rows=100000):
    mmseqs_candidates = Path(mmseqs_candidates).resolve()
    foldseek_candidates = Path(foldseek_candidates).resolve()
    output = Path(output).resolve()
    require(query_role in core.ROLES, "query role")
    require(type(rrf_constant) is int and not isinstance(rrf_constant, bool) and rrf_constant > 0, "RRF constant")
    require(mmseqs_candidates.is_file() and foldseek_candidates.is_file(), "candidate inputs")
    require(output.parent.is_dir() and not output.exists(), "exclusive output")
    output.mkdir(exist_ok=False)
    inputs = {
        "mmseqs_candidates": {"path": str(mmseqs_candidates), "sha256": sha(mmseqs_candidates)},
        "foldseek_candidates": {"path": str(foldseek_candidates), "sha256": sha(foldseek_candidates)},
    }
    emit(
        output / "reservation.json",
        {
            "status": "ONE_S4C_STREAMING_UNION_ATTEMPT_RESERVED", "query_role": query_role,
            "rrf_constant": rrf_constant, "inputs": inputs, "automatic_retry": False,
        },
    )
    sink, state, error = None, "FAIL_CLOSED", None
    try:
        sink = BufferedParquetSink(output / "union_candidates.parquet", output_schema(), batch_rows)
        totals = {
            "query_groups": 0, "mmseqs_input_rows": 0, "foldseek_input_rows": 0,
            "union_rows": 0, "mmseqs_only_rows": 0, "foldseek_only_rows": 0,
            "both_rows": 0,
        }
        for query, mm_rows, fs_rows in merged_query_groups(
            mmseqs_candidates, foldseek_candidates, query_role, batch_rows
        ):
            require(mm_rows or fs_rows, "empty query group")
            require(all(row["query_node"] == query for row in mm_rows + fs_rows), "query group identity")
            rows = core.candidate_union(mm_rows, fs_rows, query_role, rrf_constant)
            totals["query_groups"] += 1
            totals["mmseqs_input_rows"] += len(mm_rows)
            totals["foldseek_input_rows"] += len(fs_rows)
            totals["union_rows"] += len(rows)
            for row in rows:
                totals[row["source_class"] + "_rows"] += 1
                sink.add(row)
        sink.close()
        require(sink.count == totals["union_rows"], "writer census")
        require(
            totals["union_rows"] == totals["mmseqs_only_rows"] + totals["foldseek_only_rows"] + totals["both_rows"],
            "source-class census",
        )
        require(
            totals["union_rows"] == totals["mmseqs_input_rows"] + totals["foldseek_input_rows"] - totals["both_rows"],
            "union cardinality",
        )
        summary = {
            "status": "PASS_S4C_STREAMING_UNION_PENDING_INDEPENDENT_AUDIT",
            "query_role": query_role, "rrf_constant": rrf_constant,
            "batch_rows": batch_rows, "inputs": inputs, "totals": totals,
            "bounded_live_state": "ONE_QUERY_TWO_MODALITY_TOP_K_GROUPS",
            "functional_labels_read": False, "query_truth_read": False,
            "activity_expansion_started": False, "policy_selection_started": False,
            "pair_generation_started": False, "training_started": False,
        }
        emit(output / "producer_summary.json", summary)
        state = summary["status"]
    except BaseException as exc:
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
    parser.add_argument("--mmseqs-candidates", type=Path, required=True)
    parser.add_argument("--foldseek-candidates", type=Path, required=True)
    parser.add_argument("--query-role", choices=core.ROLES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--batch-rows", type=int, default=100000)
    args = parser.parse_args()
    execute(args.mmseqs_candidates, args.foldseek_candidates, args.query_role, args.output, args.rrf_constant, args.batch_rows)


if __name__ == "__main__":
    main()

