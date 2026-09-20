#!/usr/bin/env python3
"""Merge and validate exact gapped Phase 7 mapping alignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


SHARDS = 4
COLUMNS = [
    "query_protein_id", "reference_protein_id", "sequence_identity", "alignment_length",
    "query_start", "query_end", "query_length", "reference_start", "reference_end",
    "reference_length", "query_coverage", "reference_coverage", "evalue", "bitscore",
    "query_alignment", "reference_alignment", "cigar",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data/interim/phase07"
    frames = []
    for name in ["train", "eval"]:
        for shard in range(SHARDS):
            path = work / f"site_{name}_chunk_{shard}_alignments.tsv"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing Phase 7 alignment shard: {path}")
            frames.append(pl.read_csv(path, separator="\t", has_header=False, new_columns=COLUMNS))
    alignments = pl.concat(frames, how="vertical_relaxed")
    unique = alignments.select(pl.struct(["query_protein_id", "reference_protein_id"]).n_unique()).item()
    expected = pl.read_parquet(work / "site_protein_pairs.parquet").height
    bad_backtrace = alignments.filter(
        pl.col("query_alignment").is_null() | pl.col("reference_alignment").is_null() |
        (pl.col("query_alignment").str.len_chars() != pl.col("reference_alignment").str.len_chars())
    ).height
    if alignments.height != expected or unique != expected or bad_backtrace:
        raise RuntimeError(
            f"Phase 7 alignment QC failed: rows={alignments.height}/{expected}, unique={unique}, bad={bad_backtrace}"
        )
    alignments.write_parquet(work / "site_pair_alignments.parquet", compression="zstd")
    summary = {
        "status": "PASS", "alignments": alignments.height, "unique_protein_pairs": unique,
        "gapped_backtraces_valid": True, "query_ground_truth_used": False,
        "mapping_method": "MMSEQS2_DIRECT_KNOWN_PAIR_SMITH_WATERMAN_WITH_BACKTRACE",
    }
    (work / "site_alignment_merge_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
