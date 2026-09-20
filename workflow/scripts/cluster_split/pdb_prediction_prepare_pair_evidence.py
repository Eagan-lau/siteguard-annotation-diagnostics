#!/usr/bin/env python3
"""Build Phase 28 label-blind union-retrieval pair evidence and tree scores."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from siteguard.fasta import read_fasta
from siteguard.predictor import _build_pairs, _model_matrix, _query_pfam, _run_mmseqs, json_ids


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
PHASE_ID = int(os.environ.get("RCSB_EXTERNAL_PHASE", "28"))
if PHASE_ID not in {28, 30, 31, 32}:
    raise RuntimeError(f"Unsupported structural external phase: {PHASE_ID}")
WORK = ROOT / f"data/interim/phase{PHASE_ID}_inference"
RESULTS = ROOT / f"results/phase{PHASE_ID}"
CHECKPOINTS = ROOT / "checkpoints"
REPORTS = ROOT / f"reports/phase{PHASE_ID}_external_blind"
LEVELS = ("EC_L3", "EC_L4", "EXACT_RHEA")
AF_NAME = re.compile(r"AF-([A-Z0-9]+)-F\d+-model_v\d+", re.IGNORECASE)


def foldseek_hits() -> pd.DataFrame:
    names = [
        "query_structure_name", "target_structure_name", "fident", "alnlen", "qstart", "qend", "qlen",
        "tstart", "tend", "tlen", "qcov", "tcov", "evalue", "bits",
    ]
    hits = pd.read_csv(WORK / "external_foldseek_hits.tsv", sep="\t", names=names)
    mapping = pd.read_parquet(WORK / "foldseek_query.mapping.parquet")[["structure_name", "query_protein_id"]]
    hits = hits.merge(mapping, left_on="query_structure_name", right_on="structure_name", how="inner", validate="many_to_one")
    hits["reference_protein_id"] = hits["target_structure_name"].map(
        lambda value: (match.group(1).upper() if (match := AF_NAME.search(str(value))) else None)
    )
    hits = hits.dropna(subset=["reference_protein_id"]).copy()
    for column in ["fident", "qcov", "tcov", "evalue", "bits"]:
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    for column in ["fident", "qcov", "tcov"]:
        if hits[column].max() > 1:
            hits[column] /= 100.0
    hits.sort_values(
        ["query_protein_id", "bits", "reference_protein_id"],
        ascending=[True, False, True], kind="mergesort", inplace=True,
    )
    hits = hits.drop_duplicates(["query_protein_id", "reference_protein_id"])
    hits["foldseek_rank"] = hits.groupby("query_protein_id", observed=True).cumcount() + 1
    return hits.loc[hits["foldseek_rank"].le(100)].reset_index(drop=True)


def jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def main() -> None:
    for required in [
        CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}A_EXTERNAL_COHORT_LOCKED",
        CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}A3_EXTERNAL_ESM2_PASS",
        CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}A4_EXTERNAL_STRUCTURE_DOMAIN_PASS",
    ]:
        if not required.is_file():
            raise FileNotFoundError(required)
    if (RESULTS / "external_blind_predictions.parquet").exists() or (RESULTS / "external_predictions.parquet").exists():
        raise RuntimeError("Phase 28 prediction stage has started")

    fasta = ROOT / f"data/interim/phase{PHASE_ID}_independence/rcsb_independent_survivors.fasta"
    records = read_fasta(fasta)
    embeddings = np.load(WORK / "external_esm2_t33.npy", mmap_mode="r")
    embedding_index = pd.read_csv(WORK / "external_esm2_t33_index.tsv", sep="\t")
    embedding_rows = embedding_index.set_index("protein_id")["row_index"].astype(int).to_dict()
    retrieval_work = WORK / "external_pair_retrieval"
    retrieval_work.mkdir(parents=True, exist_ok=True)
    sequence = _run_mmseqs(fasta, ROOT, retrieval_work, "mmseqs", int(os.environ.get("SLURM_CPUS_PER_TASK", "12")))
    structure = foldseek_hits()

    union_keys = pd.concat([
        sequence[["query_protein_id", "reference_protein_id"]],
        structure[["query_protein_id", "reference_protein_id"]],
    ]).drop_duplicates()
    union = union_keys.merge(sequence, on=["query_protein_id", "reference_protein_id"], how="left", validate="one_to_one")
    fallback_rank = union.merge(
        structure[["query_protein_id", "reference_protein_id", "foldseek_rank"]],
        on=["query_protein_id", "reference_protein_id"], how="left", validate="one_to_one",
    )["foldseek_rank"]
    union["retrieval_rank"] = union["retrieval_rank"].fillna(fallback_rank).fillna(999).astype(int)
    pairs = _build_pairs(
        ROOT, records, embeddings, embedding_rows, union,
        _query_pfam(WORK / "external_query_pfam.tsv"),
    )
    if pairs.empty:
        raise RuntimeError("No train-only reference activities retrieved for Phase 28")
    sequence_missing = pairs["sequence_identity"].isna()
    pairs.loc[sequence_missing, ["sequence_bitscore_log1p", "sequence_alignment_fraction"]] = np.nan

    fold = structure[[
        "query_protein_id", "reference_protein_id", "fident", "qcov", "tcov", "bits", "foldseek_rank",
    ]].rename(columns={
        "fident": "phase28_foldseek_identity", "qcov": "phase28_foldseek_query_coverage",
        "tcov": "phase28_foldseek_reference_coverage", "bits": "phase28_foldseek_bits",
    })
    pairs = pairs.drop(columns=[
        "foldseek_identity", "foldseek_query_coverage", "foldseek_reference_coverage",
        "foldseek_bitscore_log1p", "foldseek_alignment_fraction",
    ]).merge(fold, on=["query_protein_id", "reference_protein_id"], how="left", validate="many_to_one")
    pairs["foldseek_identity"] = pairs["phase28_foldseek_identity"]
    pairs["foldseek_query_coverage"] = pairs["phase28_foldseek_query_coverage"]
    pairs["foldseek_reference_coverage"] = pairs["phase28_foldseek_reference_coverage"]
    pairs["foldseek_bitscore_log1p"] = np.log1p(pairs["phase28_foldseek_bits"].clip(lower=0))
    pairs["foldseek_alignment_fraction"] = pairs[["foldseek_query_coverage", "foldseek_reference_coverage"]].min(axis=1)
    pairs["query_structure_available"] = True
    pairs["both_structures_available"] = pairs["reference_structure_available"].fillna(False).astype(bool)

    query_meta = pd.read_parquet(RESULTS / "external_query_metadata.parquet").set_index("query_protein_id")
    reference_ids = set(pairs["reference_protein_id"].astype(str))
    family = pd.read_parquet(
        ROOT / "data/splits/split_family.parquet", columns=["protein_id", "primary_pfam_clan"],
    ).loc[lambda d: d["protein_id"].isin(reference_ids)].set_index("protein_id")
    cath = pd.read_parquet(
        ROOT / "data/splits/split_structure.parquet",
        columns=["protein_id", "cath_superfamilies_json", "primary_cath_superfamily"],
    ).loc[lambda d: d["protein_id"].isin(reference_ids)].set_index("protein_id")
    query_cath = query_meta["cath_superfamilies_json"].map(json_ids).to_dict()
    reference_cath = cath["cath_superfamilies_json"].map(json_ids).to_dict()
    pairs["primary_pfam_clan_match"] = [
        str(query_meta.loc[q, "primary_pfam_clan"]) == str(family.loc[r, "primary_pfam_clan"])
        for q, r in pairs[["query_protein_id", "reference_protein_id"]].itertuples(index=False)
    ]
    pairs["cath_jaccard"] = [
        jaccard(query_cath.get(q, frozenset()), reference_cath.get(r, frozenset()))
        for q, r in pairs[["query_protein_id", "reference_protein_id"]].itertuples(index=False)
    ]
    pairs["primary_cath_match"] = [
        str(query_meta.loc[q, "primary_cath_superfamily"]) == str(cath.loc[r, "primary_cath_superfamily"])
        and str(query_meta.loc[q, "primary_cath_superfamily"]) != "__MISSING__"
        for q, r in pairs[["query_protein_id", "reference_protein_id"]].itertuples(index=False)
    ]

    values = _model_matrix(ROOT, pairs)
    tree = np.column_stack([
        lgb.Booster(model_file=str(ROOT / "models/phase10" / f"lightgbm_global_{level}.txt")).predict(values)
        for level in LEVELS
    ]).astype(np.float32)
    for level_index, level in enumerate(LEVELS):
        pairs[f"score_lightgbm_global_{level}"] = tree[:, level_index]
    pairs["score_sequence_identity"] = pairs["sequence_identity"].astype(np.float32)
    pairs["score_esm2_t33_cosine"] = pairs["esm2_t33_cosine"].astype(np.float32)
    pairs["score_foldseek_identity"] = pairs["foldseek_identity"].astype(np.float32)
    pairs["score_pfam_jaccard"] = pairs["pfam_jaccard"].astype(np.float32)
    pairs["score_cath_jaccard"] = pairs["cath_jaccard"].astype(np.float32)
    missing_pfam = set(query_meta.index[query_meta["pfam_domain_count"].eq(0)])
    missing_cath = set(query_meta.index[query_meta["cath_domain_count"].eq(0)])
    pairs.loc[pairs["query_protein_id"].isin(missing_pfam), "score_pfam_jaccard"] = np.nan
    pairs.loc[pairs["query_protein_id"].isin(missing_cath), "score_cath_jaccard"] = np.nan
    pairs["pair_set"] = f"RCSB_PHASE{PHASE_ID}_EXTERNAL_BLIND"
    pairs["evidence_policy"] = f"TRAIN_ONLY_REFERENCE;MMSEQS_FOLDSEEK_UNION;NO_PHASE{PHASE_ID}_TRUTH_FEATURES"
    pairs.to_parquet(RESULTS / "external_pair_evidence_blind.parquet", index=False, compression="zstd")
    np.save(WORK / "external_model_matrix.npy", values.astype(np.float32))
    summary = {
        "phase": f"{PHASE_ID}A5A", "status": "PASS", "queries": len(records),
        "mmseqs_top50_hits": len(sequence), "foldseek_top100_hits": len(structure),
        "union_protein_pairs": len(union_keys), "pair_activity_rows": len(pairs),
        "model_matrix_shape": list(values.shape), "reference_partition": "train",
        "queries_with_foldseek": int(structure["query_protein_id"].nunique()),
        "external_truth_columns_read": False,
    }
    (REPORTS / f"phase{PHASE_ID}_pair_evidence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}A5A_EXTERNAL_PAIR_EVIDENCE_PASS").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
