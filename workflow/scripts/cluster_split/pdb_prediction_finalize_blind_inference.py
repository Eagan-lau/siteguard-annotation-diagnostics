#!/usr/bin/env python3
"""Run the frozen EvidenceJudge on Phase 28 without reading RCSB EC labels."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sabio_finalize_blind_inference as base
from siteguard.model import blend_scores


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
R24 = ROOT / "results/phase24"
R25 = ROOT / "results/phase25"
R28 = ROOT / "results/phase28"
WORK = ROOT / "data/interim/phase28_inference"
CHECKPOINTS = ROOT / "checkpoints"
ERROR_MODELS = ROOT / "models/phase25/evidencejudge/errorjudge"
LEVELS = ("EC_L3", "EC_L4", "EXACT_RHEA")
SELECTED_ERROR_MODEL = "ERROR_LGBM_L15_M100_N2"
SEEDS = (20260819, 20260820, 20260821)
AVAILABLE_METHOD_SCORE = {
    "SEQUENCE_IDENTITY": "score_sequence_identity",
    "ESM2_SIMILARITY": "score_esm2_t33_cosine",
    "FOLDSEEK_IDENTITY": "score_foldseek_identity",
    "PFAM_JACCARD": "score_pfam_jaccard",
    "CATH_JACCARD": "score_cath_jaccard",
    "LIGHTGBM_GLOBAL": "score_lightgbm_global_{level}",
    "DEEP_GLOBAL": "score_deep_global_{level}",
    "SITEGUARD": "score_siteguard_{level}",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for required in [
        CHECKPOINTS / "CHECKPOINT_28A5A_EXTERNAL_PAIR_EVIDENCE_PASS",
        CHECKPOINTS / "CHECKPOINT_28A5B_EXTERNAL_DEEP_SCORE_PASS",
    ]:
        if not required.is_file():
            raise FileNotFoundError(required)
    if (R28 / "external_predictions.parquet").exists() or (R28 / "external_blind_predictions.parquet").exists():
        raise RuntimeError("Phase 28 inference/evaluation output already exists")
    schema = json.loads((R24 / "evidencejudge_feature_schema.json").read_text())
    error_lock_path = ERROR_MODELS / "errorjudge_operating_point_lock.json"
    robust_lock_path = ERROR_MODELS.parent / "robust_screen_operating_point_lock.json"
    error_lock = json.loads(error_lock_path.read_text())
    robust_lock = json.loads(robust_lock_path.read_text())

    cohort_columns = ["query_id", "external_cluster_id_30"]
    locked = pd.read_parquet(R28 / "rcsb_strict_blind_cohort.parquet", columns=cohort_columns)
    query_metadata = pd.read_parquet(R28 / "external_query_metadata.parquet")
    cohort = locked.merge(
        query_metadata[["query_protein_id", "taxonomy_id", "sequence_length_observed"]],
        left_on="query_id", right_on="query_protein_id", how="left", validate="one_to_one",
    ).rename(columns={"query_id": "uniprot_accession"}).drop(columns="query_protein_id")
    inference_metadata = query_metadata.drop(columns=["taxonomy_id", "sequence_length_observed"])

    pairs = pd.read_parquet(R28 / "external_pair_evidence_blind.parquet")
    deep = np.load(WORK / "external_deep_global_scores.npy")
    if len(pairs) != len(deep):
        raise RuntimeError(f"Pair/deep score mismatch: pairs={len(pairs)} deep={deep.shape}")
    for index, level in enumerate(LEVELS):
        pairs[f"score_deep_global_{level}"] = deep[:, index].astype(np.float32)
    config = json.loads((ROOT / "models/phase11/siteguard_model_config.json").read_text())
    tree = pairs[[f"score_lightgbm_global_{level}" for level in LEVELS]].to_numpy(np.float32)
    blended = blend_scores(deep.astype(np.float32), tree, config["deep_blend_weights"], LEVELS)
    for index, level in enumerate(LEVELS):
        pairs[f"score_siteguard_{level}"] = blended[:, index]
    pairs.to_parquet(R28 / "external_pair_scores_blind.parquet", index=False, compression="zstd")

    base.AVAILABLE_METHOD_SCORE = AVAILABLE_METHOD_SCORE
    top = base.make_top_predictions(pairs, query_metadata)
    top.to_parquet(R28 / "external_tool_top_predictions_blind.parquet", index=False, compression="zstd")
    matrix = base.build_candidate_matrix(top, cohort, schema)
    matrix["pair_set"] = "RCSB_PHASE28_EXTERNAL_BLIND"
    matrix.to_parquet(R28 / "external_candidate_matrix_blind.parquet", index=False, compression="zstd")
    meta = base.build_metadata(cohort, inference_metadata)
    meta["pdb_availability"] = True

    context_columns = [
        "pair_set", "development_partition", "annotation_level", "query_protein_id", "query_cluster_id_30",
        "candidate_label", "candidate_row_id", "query_weight", *schema["feature_columns"],
    ]
    proposals = top.merge(
        matrix[context_columns], on=["annotation_level", "query_protein_id", "candidate_label"],
        how="inner", validate="many_to_one",
    )
    proposals.rename(columns={
        "method": "proposal_method", "evidence_channel": "proposal_channel",
        "raw_score": "proposal_raw_score", "top1_margin": "proposal_margin",
    }, inplace=True)
    proposals = proposals.merge(meta, on="query_protein_id", how="left", validate="many_to_one")
    rhea_map = base.reference_rhea_ec1()
    proposals["reference_ec_l1"] = [
        base.candidate_ec1(level, label, rhea_map)
        for level, label in zip(proposals["annotation_level"], proposals["candidate_label"], strict=True)
    ]
    proposals.to_parquet(R28 / "external_errorjudge_proposals_blind.parquet", index=False, compression="zstd")

    numeric = list(schema["feature_columns"]) + [
        "proposal_raw_score", "proposal_margin", "log1p_length",
        "log1p_fit_pfam_frequency", "log1p_fit_pfam_clan_frequency", "log1p_fit_cath_frequency",
        "pfam_seen_in_fit", "pfam_clan_seen_in_fit", "cath_seen_in_fit",
    ]
    categorical = [
        "proposal_method", "proposal_channel", "taxonomy_supergroup", "reference_ec_l1",
        "alphafold_availability", "pdb_availability",
    ]
    score_frames = []
    for level in LEVELS:
        frame = proposals.loc[proposals["annotation_level"] == level].copy()
        preprocessor = joblib.load(ERROR_MODELS / f"preprocessor__{level.lower()}.joblib")
        transformed = preprocessor.transform(frame[numeric + categorical])
        seed_scores = []
        for seed in SEEDS:
            model = joblib.load(ERROR_MODELS / f"{level.lower()}__{SELECTED_ERROR_MODEL.lower()}__seed{seed}.joblib")
            seed_scores.append(model.predict_proba(transformed)[:, 1].astype(np.float32))
        stack = np.vstack(seed_scores)
        exported = frame[[
            "annotation_level", "query_protein_id", "query_cluster_id_30", "candidate_label", "candidate_row_id",
            "proposal_method", "candidate_set_size", "agreement_entropy", "supporting_tools", "supporting_channels",
            "tool_vote_fraction", "channel_vote_fraction", "second_candidate_votes", "vote_lead",
            "support__LIGHTGBM_GLOBAL", "support__DEEP_GLOBAL", "support__SITEGUARD",
            "support__SEQUENCE_IDENTITY", "support__ESM2_SIMILARITY", "support__FOLDSEEK_IDENTITY",
            "support__PFAM_JACCARD", "support__CATH_JACCARD",
            "pfam_seen_in_fit", "pfam_clan_seen_in_fit", "cath_seen_in_fit",
        ]].copy()
        exported["proposal_probability"] = stack.mean(axis=0).astype(np.float32)
        exported["seed_probability_min"] = stack.min(axis=0).astype(np.float32)
        exported["seed_probability_sd"] = stack.std(axis=0).astype(np.float32)
        score_frames.append(exported)
    scored = pd.concat(score_frames, ignore_index=True)
    scored.to_parquet(R28 / "external_errorjudge_proposal_scores_blind.parquet", index=False, compression="zstd")

    candidate_keys = ["annotation_level", "query_protein_id", "query_cluster_id_30", "candidate_label"]
    context = scored.groupby(candidate_keys, observed=True).first().reset_index()
    probability = scored.groupby(candidate_keys, observed=True)["proposal_probability"].max().rename("raw_probability").reset_index()
    candidates = context.merge(probability, on=candidate_keys, how="inner", validate="one_to_one")
    candidates["system_model"] = SELECTED_ERROR_MODEL + "__PROPOSAL_MAX"
    winners = base.winner_rows(candidates)

    ood_features = error_lock["gate_state"]["ood_feature_columns"]
    query_ood = matrix[["annotation_level", "query_protein_id", *ood_features]].drop_duplicates([
        "annotation_level", "query_protein_id"
    ])
    scored_ood = []
    for level in LEVELS:
        frame = query_ood.loc[query_ood["annotation_level"] == level].copy()
        pipeline = joblib.load(ERROR_MODELS.parent / f"ood_query_evidence__{level.lower()}.joblib")
        frame["ood_score"] = -pipeline.decision_function(frame[ood_features])
        scored_ood.append(frame[["annotation_level", "query_protein_id", "ood_score"]])
    winners = winners.merge(pd.concat(scored_ood), on=["annotation_level", "query_protein_id"], how="left", validate="one_to_one")
    winners["selected_gate"] = "NONE"
    winners["frozen_threshold"] = np.inf
    winners["passes_gate"] = False
    winners["accepted"] = False
    for level in LEVELS:
        info = error_lock["selected_systems"][level]
        if info["system_model"] == "NO_ERRORJUDGE_QUALIFIED":
            continue
        if str(info["system_model"]) != SELECTED_ERROR_MODEL + "__PROPOSAL_MAX":
            raise RuntimeError(f"Unexpected selected external model for {level}: {info}")
        threshold = float(info["threshold"])
        level_mask = winners["annotation_level"].eq(level)
        gate = str(info["gate"])
        if gate == "NONE":
            gate_mask = level_mask
        elif gate == "PFAM_SEEN_AND_OOD_Q95":
            q95 = float(error_lock["gate_state"]["ood_quantiles"][level]["Q95"])
            gate_mask = level_mask & winners["pfam_seen_in_fit"].astype(bool) & winners["ood_score"].le(q95)
        else:
            raise RuntimeError(f"Unsupported frozen external gate: {gate}")
        winners.loc[level_mask, "selected_gate"] = gate
        winners.loc[level_mask, "frozen_threshold"] = threshold
        winners.loc[level_mask, "passes_gate"] = gate_mask[level_mask]
        winners.loc[level_mask, "accepted"] = gate_mask[level_mask] & winners.loc[level_mask, "raw_probability"].ge(threshold)
    winners["interpretation"] = "EXTERNAL_BLIND_PREDICTION_TRUTH_NOT_OPENED"
    winners.to_parquet(R28 / "external_blind_predictions.parquet", index=False, compression="zstd")

    baseline_rows = []
    for level in LEVELS:
        info = robust_lock["best_single_tools"][level]
        method = str(info["method"])
        threshold = float(info["threshold"])
        if method.startswith("NO_SINGLE") or not math.isfinite(threshold):
            continue
        frame = top.loc[(top["annotation_level"] == level) & (top["method"] == method)].copy()
        frame["frozen_threshold"] = threshold
        frame["accepted"] = frame["raw_score"].ge(threshold)
        baseline_rows.append(frame)
    baseline = pd.concat(baseline_rows, ignore_index=True) if baseline_rows else pd.DataFrame()
    baseline.to_parquet(R28 / "external_blind_single_tool_predictions.parquet", index=False, compression="zstd")

    lock = {
        "phase": "28B", "status": "PASS", "truth_opened": False,
        "strict_cohort_queries": len(cohort), "available_methods": sorted(top["method"].unique().tolist()),
        "explicitly_missing_methods": sorted(set(base.METHOD_CHANNEL) - set(top["method"])),
        "pair_rows": len(pairs), "candidate_rows": len(matrix), "blind_winner_rows": len(winners),
        "accepted_before_truth": winners.groupby("annotation_level")["accepted"].sum().astype(int).to_dict(),
        "errorjudge_lock_sha256": sha256(error_lock_path), "robust_lock_sha256": sha256(robust_lock_path),
        "cohort_lock_sha256": sha256(CHECKPOINTS / "CHECKPOINT_28A_EXTERNAL_COHORT_LOCKED"),
        "truth_columns_read": [], "cohort_columns_read": cohort_columns,
        "missing_evidence_policy": "Per-query absent Pfam/CATH/Foldseek remains missing/abstain; no truth imputation",
    }
    lock_path = ROOT / "models/phase28_external_inference_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    checks = [
        ("truth_output_absent_before_inference", not (R28 / "external_predictions.parquet").exists(), "blind stage"),
        ("truth_columns_not_read", not lock["truth_columns_read"], cohort_columns),
        ("frozen_feature_schema_complete", set(schema["feature_columns"]).issubset(matrix.columns), len(schema["feature_columns"])),
        ("candidate_label_not_model_feature", "candidate_label" not in schema["feature_columns"], "grouping/output only"),
        ("experimental_structure_channel_present", "FOLDSEEK_IDENTITY" in set(top["method"]), lock["available_methods"]),
        ("ec4_frozen_system_unchanged", error_lock["selected_systems"]["EC_L4"]["gate"] == "PFAM_SEEN_AND_OOD_Q95", error_lock["selected_systems"]["EC_L4"]),
        ("blind_prediction_grain_unique", not winners.duplicated(["annotation_level", "query_protein_id"]).any(), len(winners)),
        ("inference_lock_written", lock_path.is_file(), str(lock_path)),
    ]
    qc = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    qc.to_csv(ROOT / "reports/phase28_external_blind/phase28_blind_inference_qc.tsv", sep="\t", index=False)
    failures = qc.loc[~qc["passed"].astype(bool), "check"].tolist()
    if failures:
        lock["status"] = "FAIL"; lock["failures"] = failures
        lock_path.write_text(json.dumps(lock, indent=2) + "\n")
        raise RuntimeError(f"Phase 28 blind inference failed: {failures}")
    (CHECKPOINTS / "CHECKPOINT_28B_BLIND_PREDICTIONS_LOCKED").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
