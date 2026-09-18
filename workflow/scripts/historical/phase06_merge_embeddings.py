#!/usr/bin/env python3
"""Merge exact-sequence reusable and newly computed ESM2 rows into the V4 order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    source = args.source_root.resolve()
    work = root / "data/interim/phase06"
    mapping = pd.read_parquet(work / "esm2_t33_v4_row_mapping.parquet")
    old = np.load(source / "05_features/esm2_t33_embeddings.npy", mmap_mode="r")
    output = work / "esm2_t33_v4_embeddings.npy"
    matrix = np.lib.format.open_memmap(output, mode="w+", dtype=np.float16, shape=(len(mapping), old.shape[1]))
    reused = mapping["reused_source_row_index"].notna().to_numpy()
    matrix[reused] = old[mapping.loc[reused, "reused_source_row_index"].astype(int).to_numpy()]
    missing_positions = np.flatnonzero(~reused)
    if len(missing_positions):
        new_index = pd.read_csv(work / "esm2_t33_missing_index.tsv", sep="\t", dtype={"protein_id": str})
        new = np.load(work / "esm2_t33_missing_embeddings.npy", mmap_mode="r")
        if len(new_index) != new.shape[0] or new.shape[1] != old.shape[1]:
            raise RuntimeError("New ESM2 matrix/index mismatch")
        new_rows = new_index.set_index("protein_id")["row_index"].to_dict()
        source_rows = [new_rows.get(protein) for protein in mapping.loc[~reused, "protein_id"]]
        if any(value is None for value in source_rows):
            raise RuntimeError("New ESM2 output does not cover every missing V4 protein")
        matrix[missing_positions] = new[np.asarray(source_rows, dtype=np.int64)]
    matrix.flush()
    norms = np.linalg.norm(np.asarray(matrix, dtype=np.float32), axis=1)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise RuntimeError("Merged V4 ESM2 matrix contains invalid rows")
    index = mapping[["protein_id", "v4_row_index", "sequence_sha256", "embedding_status"]].rename(
        columns={"v4_row_index": "row_index"}
    )
    index.to_csv(work / "esm2_t33_v4_index.tsv", sep="\t", index=False)
    summary = {
        "status": "PASS", "proteins": len(mapping), "hidden_size": int(matrix.shape[1]),
        "dtype": str(matrix.dtype), "reused_rows": int(reused.sum()), "new_rows": int((~reused).sum()),
        "minimum_l2_norm": float(norms.min()), "maximum_l2_norm": float(norms.max()),
        "provenance": "ESM2-t33 650M; reused rows require exact amino-acid sequence equality",
    }
    (work / "esm2_t33_v4_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
