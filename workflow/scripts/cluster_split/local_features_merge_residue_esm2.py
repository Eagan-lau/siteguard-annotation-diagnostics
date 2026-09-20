#!/usr/bin/env python3
"""Attach residue-centred ESM2 similarity to Local-2 and site mapping outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROW_KEY = ["pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id"]
SITE_KEY = ROW_KEY + ["reference_site_id"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data/interim/phase07"
    processed = root / "data/processed"
    reports = root / "reports"
    contexts = pd.read_parquet(work / "residue_esm2_contexts.parquet")
    embeddings = np.load(work / "residue_esm2_t12_embeddings.npy", mmap_mode="r")
    if len(contexts) != embeddings.shape[0] or embeddings.shape[1] != 480:
        raise RuntimeError(f"Residue ESM2 context/matrix mismatch: {len(contexts)} vs {embeddings.shape}")
    lookup = contexts.set_index(["protein_id", "residue_position"])["embedding_row"].to_dict()
    mapping = pd.read_parquet(processed / "site_mapping.parquet")
    successful = mapping["mapping_status"].eq("PASS")
    query_rows = np.array([
        lookup.get((protein, int(position)), -1)
        for protein, position in zip(
            mapping.loc[successful, "query_protein_id"],
            mapping.loc[successful, "mapped_query_position"], strict=True,
        )
    ], dtype=np.int64)
    reference_rows = np.array([
        lookup.get((protein, int(position)), -1)
        for protein, position in zip(
            mapping.loc[successful, "reference_protein_id"],
            mapping.loc[successful, "reference_site_position"], strict=True,
        )
    ], dtype=np.int64)
    if (query_rows < 0).any() or (reference_rows < 0).any():
        raise RuntimeError("Successful site mappings are absent from residue ESM2 contexts")
    values = np.full(len(mapping), np.nan, dtype=np.float32)
    indices = np.flatnonzero(successful.to_numpy())
    chunk_size = 20_000
    for start in range(0, len(indices), chunk_size):
        end = min(start + chunk_size, len(indices))
        query = np.asarray(embeddings[query_rows[start:end]], dtype=np.float32)
        reference = np.asarray(embeddings[reference_rows[start:end]], dtype=np.float32)
        denominator = np.linalg.norm(query, axis=1) * np.linalg.norm(reference, axis=1)
        values[indices[start:end]] = np.divide(
            np.einsum("ij,ij->i", query, reference), denominator,
            out=np.zeros(end - start, dtype=np.float32), where=denominator > 0,
        )
    mapping["local2_residue_esm2_t12_cosine"] = values
    mapping.to_parquet(processed / "site_mapping.parquet", index=False, compression="zstd")
    pair_values = (
        mapping.loc[mapping["mapping_status"].eq("PASS")]
        .groupby(ROW_KEY, sort=False)["local2_residue_esm2_t12_cosine"].mean().reset_index()
    )
    l2 = pd.read_parquet(processed / "local_features_L2.parquet")
    l2 = l2.drop(columns=["local2_residue_esm2_t12_cosine"], errors="ignore")
    l2 = l2.merge(pair_values, on=ROW_KEY, how="left", validate="one_to_one")
    l2.to_parquet(processed / "local_features_L2.parquet", index=False, compression="zstd")
    summary = {
        "phase": 7, "stage": "residue_esm2_Local2", "status": "PASS",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "residue_contexts": len(contexts), "successful_site_rows": int(successful.sum()),
        "site_rows_with_residue_esm2": int(np.isfinite(values).sum()),
        "pair_rows_with_residue_esm2": int(l2["local2_residue_esm2_t12_cosine"].notna().sum()),
        "model": "ESM2-t12 35M frozen central-residue last-hidden-state",
        "query_ground_truth_used": False,
        "checkpoint_07_status": "PENDING_AF_PDB_ROBUSTNESS_AND_FINAL_QC",
    }
    (reports / "phase07_residue_esm2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
