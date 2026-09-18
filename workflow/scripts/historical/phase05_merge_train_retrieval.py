#!/usr/bin/env python3
"""Merge and audit Foldseek train-to-train candidate chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=6)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if not (root / "checkpoints" / "CHECKPOINT_04_PASS").is_file():
        raise RuntimeError("CHECKPOINT_04_PASS is required")
    work = root / "data" / "interim" / "phase05"
    sources = [work / f"foldseek_train_candidates_chunk_{index}.parquet" for index in range(args.chunks)]
    for source in sources:
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError(f"Missing candidate chunk: {source}")
    output = work / "foldseek_train_candidates.parquet"
    temporary = output.with_suffix(".parquet.tmp")
    writer: pq.ParquetWriter | None = None
    row_count = 0
    queries: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    maximum_rank = 0
    try:
        for source in sources:
            parquet = pq.ParquetFile(source)
            if writer is None:
                writer = pq.ParquetWriter(temporary, parquet.schema_arrow, compression="zstd")
            elif parquet.schema_arrow != writer.schema:
                raise RuntimeError(f"Schema mismatch in {source}")
            for batch in parquet.iter_batches(batch_size=100_000):
                table = pa.Table.from_batches([batch])
                query_values = table["query_protein_id"].to_pylist()
                reference_values = table["reference_protein_id"].to_pylist()
                rank_values = table["modality_rank"].to_pylist()
                for query, reference, rank in zip(query_values, reference_values, rank_values):
                    pair = (query, reference)
                    if query == reference or pair in pairs:
                        raise RuntimeError(f"Self or duplicate Foldseek train pair: {pair}")
                    pairs.add(pair)
                    queries.add(query)
                    maximum_rank = max(maximum_rank, int(rank))
                row_count += table.num_rows
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output)
    summary = {
        "status": "PASS",
        "chunks": args.chunks,
        "rows": row_count,
        "queries_with_candidates": len(queries),
        "unique_pairs": len(pairs),
        "maximum_rank": maximum_rank,
        "self_pairs": 0,
        "duplicate_pairs": 0,
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
