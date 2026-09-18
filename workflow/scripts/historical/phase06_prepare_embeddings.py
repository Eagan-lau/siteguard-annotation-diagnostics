#!/usr/bin/env python3
"""Validate reusable ESM2 rows by exact sequence and prepare the missing V4 panel."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    source = args.source_root.resolve()
    if not (root / "checkpoints/CHECKPOINT_05_PASS").is_file():
        raise RuntimeError("CHECKPOINT_05_PASS is required before Phase 6")
    work = root / "data/interim/phase06"
    work.mkdir(parents=True, exist_ok=True)
    benchmark = pd.read_parquet(
        root / "data/interim/phase03/benchmark_proteins.parquet", columns=["protein_id", "sequence"]
    ).reset_index(drop=True)
    if not benchmark["protein_id"].is_unique:
        raise RuntimeError("V4 benchmark protein IDs are not unique")
    old_sequences = pd.read_csv(
        source / "05_features/esm2_required_sequences.tsv.gz", sep="\t", compression="gzip",
        header=None, names=["protein_id", "sequence"], dtype=str,
    )
    old_index = pd.read_csv(source / "05_features/esm2_t33_index.tsv", sep="\t", dtype={"protein_id": str})
    old_matrix = np.load(source / "05_features/esm2_t33_embeddings.npy", mmap_mode="r")
    if len(old_index) != old_matrix.shape[0] or old_matrix.shape[1] != 1280:
        raise RuntimeError("Reusable ESM2 index/matrix shape mismatch")
    if old_index["row_index"].tolist() != list(range(len(old_index))):
        raise RuntimeError("Reusable ESM2 row indices are not contiguous")
    old = old_sequences.merge(old_index[["protein_id", "row_index"]], on="protein_id", validate="one_to_one")
    merged = benchmark.merge(old, on="protein_id", how="left", suffixes=("_v4", "_old"), validate="one_to_one")
    reusable = merged["row_index"].notna()
    mismatched = reusable & (merged["sequence_v4"] != merged["sequence_old"])
    if mismatched.any():
        raise RuntimeError(f"Sequence mismatch in purported reusable ESM2 rows: {int(mismatched.sum())}")
    mapping = pd.DataFrame({
        "protein_id": merged["protein_id"],
        "v4_row_index": np.arange(len(merged), dtype=np.int64),
        "reused_source_row_index": merged["row_index"].astype("Int64"),
        "sequence_sha256": merged["sequence_v4"].map(sha256_text),
        "embedding_status": np.where(reusable, "REUSED_EXACT_SEQUENCE_MATCH", "MISSING_TO_COMPUTE"),
    })
    mapping.to_parquet(work / "esm2_t33_v4_row_mapping.parquet", index=False, compression="zstd")
    missing = merged.loc[~reusable, ["protein_id", "sequence_v4"]]
    with gzip.open(work / "esm2_t33_missing_sequences.tsv.gz", "wt", encoding="utf-8") as handle:
        for row in missing.itertuples(index=False):
            handle.write(f"{row.protein_id}\t{row.sequence_v4}\n")
    summary = {
        "status": "PASS", "benchmark_proteins": len(benchmark),
        "reused_exact_sequence_match": int(reusable.sum()), "missing_to_compute": len(missing),
        "sequence_mismatches": int(mismatched.sum()), "hidden_size": int(old_matrix.shape[1]),
        "source_embedding_summary": str(source / "05_features/esm2_t33_summary.json"),
        "reuse_rule": "protein_id AND exact amino-acid sequence equality",
    }
    (work / "esm2_t33_reuse_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
