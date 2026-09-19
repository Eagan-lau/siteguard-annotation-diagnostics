#!/usr/bin/env python3
"""Prepare label-blind external pair evidence and frozen LightGBM scores."""

from __future__ import annotations

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from siteguard.fasta import read_fasta
from siteguard.predictor import _build_pairs, _model_matrix, _query_pfam, _run_mmseqs


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
WORK = ROOT / "data/interim/phase26_inference"
RESULTS = ROOT / "results/phase26"
CHECKPOINTS = ROOT / "checkpoints"
LEVELS = ("EC_L3", "EC_L4", "EXACT_RHEA")


def main() -> None:
    for required in [
        CHECKPOINTS / "CHECKPOINT_26A_EXTERNAL_COHORT_LOCKED",
        CHECKPOINTS / "CHECKPOINT_26A3_EXTERNAL_ESM2_PASS",
        CHECKPOINTS / "CHECKPOINT_26A4_EXTERNAL_PFAM_PASS",
    ]:
        if not required.is_file():
            raise FileNotFoundError(required)
    if (RESULTS / "external_predictions.parquet").exists():
        raise RuntimeError("External outcomes have already been opened")
    fasta = ROOT / "data/interim/phase26_independence/external_independent_survivors.fasta"
    embeddings_path = WORK / "external_esm2_t33.npy"
    index_path = WORK / "external_esm2_t33_index.tsv"
    pfam_path = WORK / "external_query_pfam.tsv"
    records = read_fasta(fasta)
    embeddings = np.load(embeddings_path, mmap_mode="r")
    index = pd.read_csv(index_path, sep="\t")
    embedding_rows = index.set_index("protein_id")["row_index"].astype(int).to_dict()
    retrieval_work = WORK / "external_pair_retrieval"
    retrieval_work.mkdir(parents=True, exist_ok=True)
    hits = _run_mmseqs(
        fasta, ROOT, retrieval_work, "mmseqs",
        int(os.environ.get("SLURM_CPUS_PER_TASK", "12")),
    )
    pairs = _build_pairs(
        ROOT, records, embeddings, embedding_rows, hits, _query_pfam(pfam_path)
    )
    if pairs.empty:
        raise RuntimeError("No train-only reference activities retrieved for external cohort")
    values = _model_matrix(ROOT, pairs)
    tree = np.column_stack([
        lgb.Booster(model_file=str(ROOT / "models/phase10" / f"lightgbm_global_{level}.txt")).predict(values)
        for level in LEVELS
    ]).astype(np.float32)
    for index_level, level in enumerate(LEVELS):
        pairs[f"score_lightgbm_global_{level}"] = tree[:, index_level]
    pairs["score_sequence_identity"] = pairs["sequence_identity"].astype(np.float32)
    pairs["score_esm2_t33_cosine"] = pairs["esm2_t33_cosine"].astype(np.float32)
    pairs["score_pfam_jaccard"] = pairs["pfam_jaccard"].astype(np.float32)
    pairs["pair_set"] = "SABIO_EXTERNAL_BLIND"
    pairs["evidence_policy"] = "TRAIN_ONLY_REFERENCE;NO_EXTERNAL_TRUTH_FEATURES"
    pairs.to_parquet(RESULTS / "external_pair_evidence_blind.parquet", index=False, compression="zstd")
    np.save(WORK / "external_model_matrix.npy", values.astype(np.float32))
    summary = {
        "phase": "26A5A",
        "status": "PASS",
        "queries": len(records),
        "mmseqs_hits": len(hits),
        "pair_activity_rows": len(pairs),
        "model_matrix_shape": list(values.shape),
        "reference_partition": "train",
        "external_truth_columns_read": False,
        "direct_query_structure_evidence": False,
    }
    (ROOT / "reports/phase26_external_pair_evidence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (CHECKPOINTS / "CHECKPOINT_26A5A_EXTERNAL_PAIR_EVIDENCE_PASS").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
