#!/usr/bin/env python3
"""Merge disjoint Foldseek candidate chunks without loading them all into memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=3)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data" / "interim" / "phase04"
    output = root / "data" / "processed" / "foldseek_candidates.parquet"
    temporary = output.with_suffix(".parquet.tmp")
    writer = None
    summaries = []
    seen_queries: set[str] = set()
    total_rows = 0
    try:
        for index in range(args.chunks):
            path = work / f"foldseek_candidates_chunk_{index}.parquet"
            summary_path = path.with_suffix(".summary.json")
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            parquet = pq.ParquetFile(path)
            if writer is None:
                writer = pq.ParquetWriter(temporary, parquet.schema_arrow, compression="zstd")
            for batch in parquet.iter_batches(batch_size=100000):
                writer.write_batch(batch)
                total_rows += batch.num_rows
            chunk_queries = set(pq.read_table(path, columns=["query_protein_id"]).column(0).to_pylist())
            if seen_queries & chunk_queries:
                raise RuntimeError(f"Query overlap between Foldseek chunks: {len(seen_queries & chunk_queries)}")
            seen_queries.update(chunk_queries)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output)
    result = {
        "modality": "foldseek",
        "raw_rows": sum(int(item["raw_rows"]) for item in summaries),
        "deduplicated_top_k_rows": total_rows,
        "queries_with_candidates": len(seen_queries),
        "queries_with_100_candidates": sum(int(item["queries_with_100_candidates"]) for item in summaries),
        "median_candidates_by_chunk": [item["median_candidates"] for item in summaries],
        "maximum_rank": max(int(item["maximum_rank"]) for item in summaries),
        "reference_partitions": ["train"],
        "query_partitions": ["test", "validation"],
        "chunks": args.chunks,
        "streaming_merge": True,
    }
    output.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
