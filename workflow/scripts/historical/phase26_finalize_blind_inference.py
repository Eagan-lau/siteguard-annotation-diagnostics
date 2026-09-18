#!/usr/bin/env python3
"""Create external EvidenceJudge decisions without reading SABIO truth labels."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from siteguard.model import blend_scores


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
R24 = ROOT / "results/phase24"
R25 = ROOT / "results/phase25"
R26 = ROOT / "results/phase26"
WORK = ROOT / "data/interim/phase26_inference"
CHECKPOINTS = ROOT / "checkpoints"
ERROR_MODELS = ROOT / "models/phase25/evidencejudge/errorjudge"
LEVELS = ("EC_L3", "EC_L4", "EXACT_RHEA")
METHOD_CHANNEL = {
    "SEQUENCE_IDENTITY": "sequence",
    "ESM2_SIMILARITY": "sequence_embedding",
    "FOLDSEEK_IDENTITY": "structure",
    "PFAM_JACCARD": "domain",
    "CATH_JACCARD": "domain_structure",
    "LIGHTGBM_GLOBAL": "learned_global",
    "DEEP_GLOBAL": "learned_global",
    "SITEGUARD": "learned_global",
}
AVAILABLE_METHOD_SCORE = {
    "SEQUENCE_IDENTITY": "score_sequence_identity",
    "ESM2_SIMILARITY": "score_esm2_t33_cosine",
    "PFAM_JACCARD": "score_pfam_jaccard",
    "LIGHTGBM_GLOBAL": "score_lightgbm_global_{level}",
    "DEEP_GLOBAL": "score_deep_global_{level}",
    "SITEGUARD": "score_siteguard_{level}",
}
SELECTED_ERROR_MODEL = "ERROR_LGBM_L15_M100_N2"
SEEDS = (20260819, 20260820, 20260821)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_entropy(values: pd.Series) -> float:
    counts = values.value_counts().to_numpy(dtype=float)
    if len(counts) <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(counts)))


def parse_supergroup(value: object) -> str:
    text = "" if value is None else str(value)
    for token in ("Bacteria", "Eukaryota", "Archaea", "Viruses"):
        if token.lower() in text.lower():
            return token.upper()
    return "OTHER"


def normalize_rhea(value: object) -> str:
    match = re.search(r"(\d+)", str(value).upper())
    return f"RHEA:{int(match.group(1))}" if match else str(value).upper()


def reference_rhea_ec1() -> dict[str, str]:
    activity = pd.read_parquet(
        ROOT / "data/reference/activity_reference_library.parquet", columns=["canonical_rhea", "ec_l1"]
    ).dropna(subset=["canonical_rhea"])
    activity["canonical_rhea"] = activity["canonical_rhea"].map(normalize_rhea)
    mapping: dict[str, str] = {}
    for rhea, group in activity.groupby("canonical_rhea", observed=True):
        values = sorted({str(value) for value in group["ec_l1"].dropna() if str(value) not in {"", "nan", "None"}})
        mapping[str(rhea)] = values[0] if len(values) == 1 else ("MULTI" if values else "UNKNOWN")
    return mapping


def candidate_ec1(level: str, label: object, rhea_map: dict[str, str]) -> str:
    if level == "EXACT_RHEA":
        return rhea_map.get(normalize_rhea(label), "UNKNOWN")
    match = re.match(r"\s*(\d+)", str(label))
    return match.group(1) if match else "UNKNOWN"


def make_top_predictions(pairs: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    frames = []
    pfam_available = set(metadata.loc[metadata["pfam_domain_count"].gt(0), "query_protein_id"].astype(str))
    for level in LEVELS:
        label_column = {"EC_L3": "ec_l3", "EC_L4": "ec_l4", "EXACT_RHEA": "canonical_rhea"}[level]
        for method, template in AVAILABLE_METHOD_SCORE.items():
            score_column = template.format(level=level)
            work = pairs.loc[pairs[score_column].notna() & pairs[label_column].notna(), [
                "query_protein_id", "reference_protein_id", "reference_activity_id", label_column, score_column,
            ]].copy()
            work = work.loc[work[label_column].astype(str).ne("")]
            if method == "PFAM_JACCARD":
                work = work.loc[work["query_protein_id"].astype(str).isin(pfam_available)]
            work[label_column] = work[label_column].astype(str)
            work.sort_values(
                ["query_protein_id", score_column, "reference_activity_id", "reference_protein_id"],
                ascending=[True, False, True, True], kind="mergesort", inplace=True,
            )
            rank = work.groupby("query_protein_id", observed=True).cumcount()
            top = work.loc[rank.eq(0)].copy()
            second = work.loc[rank.eq(1), ["query_protein_id", score_column]].rename(columns={score_column: "second_score"})
            top = top.merge(second, on="query_protein_id", how="left", validate="one_to_one")
            top["top1_margin"] = top[score_column] - top["second_score"]
            top.rename(columns={score_column: "raw_score", label_column: "candidate_label"}, inplace=True)
            top["method"] = method
            top["evidence_channel"] = METHOD_CHANNEL[method]
            top["annotation_level"] = level
            frames.append(top[[
                "query_protein_id", "method", "evidence_channel", "annotation_level",
                "reference_protein_id", "reference_activity_id", "candidate_label", "raw_score", "top1_margin",
            ]])
    return pd.concat(frames, ignore_index=True)


def build_candidate_matrix(top: pd.DataFrame, cohort: pd.DataFrame, schema: dict[str, object]) -> pd.DataFrame:
    query_keys = ["pair_set", "annotation_level", "query_protein_id", "query_cluster_id_30"]
    top = top.merge(
        cohort.rename(columns={"uniprot_accession": "query_protein_id", "external_cluster_id_30": "query_cluster_id_30"})[
            ["query_protein_id", "query_cluster_id_30"]
        ], on="query_protein_id", how="left", validate="many_to_one",
    )
    top["pair_set"] = "SABIO_EXTERNAL_BLIND"
    top["development_partition"] = "EXTERNAL_BLIND"
    query_index = top[query_keys + ["development_partition"]].drop_duplicates()
    query_wide = query_index.copy()
    for method in METHOD_CHANNEL:
        own = top.loc[top["method"].eq(method), query_keys + ["raw_score", "top1_margin"]].rename(columns={
            "raw_score": f"query_score__{method}", "top1_margin": f"query_margin__{method}",
        })
        query_wide = query_wide.merge(own, on=query_keys, how="left", validate="one_to_one")

    candidate_keys = query_keys + ["candidate_label"]
    candidates = top[candidate_keys].drop_duplicates().sort_values(candidate_keys, kind="mergesort")
    candidates = candidates.merge(query_wide, on=query_keys, how="left", validate="many_to_one")
    for method in METHOD_CHANNEL:
        own = top.loc[top["method"].eq(method), candidate_keys + ["raw_score", "top1_margin"]].rename(columns={
            "raw_score": f"support_score__{method}", "top1_margin": f"support_margin__{method}",
        })
        own[f"support__{method}"] = 1.0
        candidates = candidates.merge(own, on=candidate_keys, how="left", validate="one_to_one")
        candidates[f"support__{method}"] = candidates[f"support__{method}"].fillna(0.0)
    support_columns = [f"support__{method}" for method in METHOD_CHANNEL]
    candidates["supporting_tools"] = candidates[support_columns].sum(axis=1).astype(np.int16)
    channel_columns = []
    for channel in sorted(set(METHOD_CHANNEL.values())):
        methods = [method for method, mapped in METHOD_CHANNEL.items() if mapped == channel]
        column = f"support_channel__{channel}"
        candidates[column] = candidates[[f"support__{method}" for method in methods]].max(axis=1)
        channel_columns.append(column)
    candidates["supporting_channels"] = candidates[channel_columns].sum(axis=1).astype(np.int16)
    candidates["available_tools"] = candidates[[f"query_score__{method}" for method in METHOD_CHANNEL]].notna().sum(axis=1).astype(np.int16)
    availability = top.groupby(query_keys, observed=True).agg(
        available_channels=("evidence_channel", "nunique")
    ).reset_index()
    candidates = candidates.merge(availability, on=query_keys, how="left", validate="many_to_one")
    candidates["tool_vote_fraction"] = candidates["supporting_tools"] / candidates["available_tools"].clip(lower=1)
    candidates["channel_vote_fraction"] = candidates["supporting_channels"] / candidates["available_channels"].clip(lower=1)
    query_stats = top.groupby(query_keys, observed=True).agg(
        candidate_set_size=("candidate_label", "nunique"),
        agreement_entropy=("candidate_label", normalized_entropy),
    ).reset_index()
    votes = top.groupby(candidate_keys, observed=True)["method"].nunique().rename("candidate_votes").reset_index()
    votes.sort_values(query_keys + ["candidate_votes", "candidate_label"], ascending=[True, True, True, True, False, True], kind="mergesort", inplace=True)
    votes["vote_rank"] = votes.groupby(query_keys, observed=True)["candidate_votes"].rank(method="dense", ascending=False)
    second = votes.loc[votes["vote_rank"].eq(2)].groupby(query_keys, observed=True)["candidate_votes"].max().rename("second_candidate_votes").reset_index()
    candidates = candidates.merge(query_stats, on=query_keys, how="left", validate="many_to_one")
    candidates = candidates.merge(second, on=query_keys, how="left", validate="many_to_one")
    candidates["second_candidate_votes"] = candidates["second_candidate_votes"].fillna(0)
    candidates["vote_lead"] = candidates["supporting_tools"] - candidates["second_candidate_votes"]
    candidates.sort_values(candidate_keys, kind="mergesort", inplace=True)
    candidates.reset_index(drop=True, inplace=True)
    candidates.insert(0, "candidate_row_id", np.arange(len(candidates), dtype=np.int64))
    candidates["query_weight"] = 1.0 / candidates["candidate_set_size"].clip(lower=1)
    missing_features = sorted(set(schema["feature_columns"]) - set(candidates.columns))
    if missing_features:
        raise RuntimeError(f"External candidate matrix missing frozen features: {missing_features}")
    return candidates


def build_metadata(cohort: pd.DataFrame, query_metadata: pd.DataFrame) -> pd.DataFrame:
    protein = pd.read_parquet(ROOT / "data/processed/protein_table.parquet", columns=["taxonomy_id", "lineage_json"])
    protein["taxonomy_id"] = protein["taxonomy_id"].astype(str)
    taxonomy = protein.dropna(subset=["taxonomy_id"]).drop_duplicates("taxonomy_id").set_index("taxonomy_id")["lineage_json"].to_dict()
    meta = cohort.rename(columns={"uniprot_accession": "query_protein_id"})[[
        "query_protein_id", "taxonomy_id", "sequence_length_observed",
    ]].merge(query_metadata, on="query_protein_id", how="left", validate="one_to_one")
    meta["taxonomy_supergroup"] = meta["taxonomy_id"].astype(str).map(taxonomy).map(parse_supergroup)
    meta["length"] = pd.to_numeric(meta["sequence_length_observed"], errors="coerce")
    meta["log1p_length"] = np.log1p(meta["length"])
    meta["alphafold_availability"] = False
    meta["pdb_availability"] = False

    dev = pd.read_parquet(R25 / "errorjudge_development_proposals.parquet", columns=[
        "development_partition", "query_protein_id", "primary_pfam", "primary_pfam_clan", "primary_cath_superfamily",
    ])
    fit = dev.loc[dev["development_partition"] == "FIT"].drop_duplicates("query_protein_id")
    for column, output in [
        ("primary_pfam", "fit_pfam_frequency"),
        ("primary_pfam_clan", "fit_pfam_clan_frequency"),
        ("primary_cath_superfamily", "fit_cath_frequency"),
    ]:
        counts = fit[column].value_counts()
        meta[output] = meta[column].map(counts).fillna(0).astype(float)
        meta[f"log1p_{output}"] = np.log1p(meta[output])
    return meta


def winner_rows(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["annotation_level", "query_protein_id", "raw_probability", "supporting_channels", "supporting_tools", "vote_lead", "candidate_label"],
        ascending=[True, True, False, False, False, False, True], kind="mergesort",
    )
    return ordered.drop_duplicates(["annotation_level", "query_protein_id"], keep="first").copy()


def main() -> None:
    for required in [
        CHECKPOINTS / "CHECKPOINT_26A5A_EXTERNAL_PAIR_EVIDENCE_PASS",
        CHECKPOINTS / "CHECKPOINT_26A5B_EXTERNAL_DEEP_SCORE_PASS",
    ]:
        if not required.is_file():
            raise FileNotFoundError(required)
    if (R26 / "external_predictions.parquet").exists():
        raise RuntimeError("External truth evaluation already exists")
    schema = json.loads((R24 / "evidencejudge_feature_schema.json").read_text())
    error_lock_path = ERROR_MODELS / "errorjudge_operating_point_lock.json"
    robust_lock_path = ERROR_MODELS.parent / "robust_screen_operating_point_lock.json"
    error_lock = json.loads(error_lock_path.read_text())
    robust_lock = json.loads(robust_lock_path.read_text())

    # Explicitly omit ec_l3/ec_l4 and all SABIO truth/evidence columns.
    cohort_columns = [
        "uniprot_accession", "external_cluster_id_30", "taxonomy_id", "sequence_length_observed",
    ]
    cohort = pd.read_parquet(R26 / "sabio_strict_blind_cohort.parquet", columns=cohort_columns)
    query_metadata = pd.read_parquet(R26 / "external_query_metadata.parquet")
    pairs = pd.read_parquet(R26 / "external_pair_evidence_blind.parquet")
    deep = np.load(WORK / "external_deep_global_scores.npy")
    if len(pairs) != len(deep):
        raise RuntimeError(f"Pair/deep score mismatch: pairs={len(pairs)};deep={deep.shape}")
    for index, level in enumerate(LEVELS):
        pairs[f"score_deep_global_{level}"] = deep[:, index].astype(np.float32)
    config = json.loads((ROOT / "models/phase11/siteguard_model_config.json").read_text())
    tree = pairs[[f"score_lightgbm_global_{level}" for level in LEVELS]].to_numpy(np.float32)
    blended = blend_scores(deep.astype(np.float32), tree, config["deep_blend_weights"], LEVELS)
    for index, level in enumerate(LEVELS):
        pairs[f"score_siteguard_{level}"] = blended[:, index]
    pairs.to_parquet(R26 / "external_pair_scores_blind.parquet", index=False, compression="zstd")

    top = make_top_predictions(pairs, query_metadata)
    top.to_parquet(R26 / "external_tool_top_predictions_blind.parquet", index=False, compression="zstd")
    matrix = build_candidate_matrix(top, cohort, schema)
    matrix.to_parquet(R26 / "external_candidate_matrix_blind.parquet", index=False, compression="zstd")
    meta = build_metadata(cohort, query_metadata)

    context_columns = [
        "pair_set", "development_partition", "annotation_level", "query_protein_id", "query_cluster_id_30", "candidate_label",
        "candidate_row_id", "query_weight", *schema["feature_columns"],
    ]
    proposals = top.merge(
        matrix[context_columns],
        on=["annotation_level", "query_protein_id", "candidate_label"], how="inner", validate="many_to_one",
    )
    proposals.rename(columns={
        "method": "proposal_method", "evidence_channel": "proposal_channel",
        "raw_score": "proposal_raw_score", "top1_margin": "proposal_margin",
    }, inplace=True)
    proposals = proposals.merge(meta, on="query_protein_id", how="left", validate="many_to_one")
    rhea_map = reference_rhea_ec1()
    proposals["reference_ec_l1"] = [
        candidate_ec1(level, label, rhea_map)
        for level, label in zip(proposals["annotation_level"], proposals["candidate_label"], strict=True)
    ]
    proposals.to_parquet(R26 / "external_errorjudge_proposals_blind.parquet", index=False, compression="zstd")

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
            "support__SEQUENCE_IDENTITY", "support__ESM2_SIMILARITY",
            "pfam_seen_in_fit", "pfam_clan_seen_in_fit", "cath_seen_in_fit",
        ]].copy()
        exported["proposal_probability"] = stack.mean(axis=0).astype(np.float32)
        exported["seed_probability_min"] = stack.min(axis=0).astype(np.float32)
        exported["seed_probability_sd"] = stack.std(axis=0).astype(np.float32)
        score_frames.append(exported)
    scored = pd.concat(score_frames, ignore_index=True)
    scored.to_parquet(R26 / "external_errorjudge_proposal_scores_blind.parquet", index=False, compression="zstd")

    candidate_keys = ["annotation_level", "query_protein_id", "query_cluster_id_30", "candidate_label"]
    context = scored.groupby(candidate_keys, observed=True).first().reset_index()
    probability = scored.groupby(candidate_keys, observed=True)["proposal_probability"].max().rename("raw_probability").reset_index()
    candidates = context.merge(probability, on=candidate_keys, how="inner", validate="one_to_one")
    candidates["system_model"] = SELECTED_ERROR_MODEL + "__PROPOSAL_MAX"
    winners = winner_rows(candidates)

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
    winners.to_parquet(R26 / "external_blind_predictions.parquet", index=False, compression="zstd")

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
    baseline.to_parquet(R26 / "external_blind_single_tool_predictions.parquet", index=False, compression="zstd")

    inference_lock = {
        "phase": "26A5",
        "status": "PASS",
        "truth_opened": False,
        "strict_cohort_queries": len(cohort),
        "available_methods": sorted(top["method"].unique().tolist()),
        "explicitly_missing_methods": sorted(set(METHOD_CHANNEL) - set(top["method"])),
        "pair_rows": len(pairs),
        "candidate_rows": len(matrix),
        "blind_winner_rows": len(winners),
        "accepted_before_truth": winners.groupby("annotation_level")["accepted"].sum().astype(int).to_dict(),
        "errorjudge_lock_sha256": sha256(error_lock_path),
        "robust_lock_sha256": sha256(robust_lock_path),
        "cohort_lock_sha256": sha256(CHECKPOINTS / "CHECKPOINT_26A_EXTERNAL_COHORT_LOCKED"),
        "embedding_checkpoint_sha256": sha256(CHECKPOINTS / "CHECKPOINT_26A3_EXTERNAL_ESM2_PASS"),
        "pfam_checkpoint_sha256": sha256(CHECKPOINTS / "CHECKPOINT_26A4_EXTERNAL_PFAM_PASS"),
        "missing_evidence_policy": "Foldseek/CATH/structure are missing; no zero-score proposal or truth-derived imputation",
        "cohort_columns_read": cohort_columns,
        "truth_columns_read": [],
    }
    lock_path = ROOT / "models/phase26_external_inference_lock.json"
    lock_path.write_text(json.dumps(inference_lock, indent=2) + "\n")
    checks = [
        ("truth_output_absent_before_inference", not (R26 / "external_predictions.parquet").exists(), "blind stage"),
        ("truth_columns_not_read", not inference_lock["truth_columns_read"], cohort_columns),
        ("frozen_feature_schema_complete", set(schema["feature_columns"]).issubset(matrix.columns), len(schema["feature_columns"])),
        ("candidate_label_not_model_feature", "candidate_label" not in schema["feature_columns"], "grouping/output only"),
        ("structure_evidence_missing_not_imputed", set(inference_lock["explicitly_missing_methods"]) == {"FOLDSEEK_IDENTITY", "CATH_JACCARD"}, inference_lock["explicitly_missing_methods"]),
        ("ec4_frozen_system_unchanged", error_lock["selected_systems"]["EC_L4"]["gate"] == "PFAM_SEEN_AND_OOD_Q95", error_lock["selected_systems"]["EC_L4"]),
        ("blind_prediction_grain_unique", not winners.duplicated(["annotation_level", "query_protein_id"]).any(), len(winners)),
        ("inference_lock_written", lock_path.is_file(), str(lock_path)),
    ]
    qc = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    qc.to_csv(ROOT / "reports/phase26_external_blind_inference_qc.tsv", sep="\t", index=False)
    failures = qc.loc[~qc["passed"].astype(bool), "check"].tolist()
    if failures:
        inference_lock["status"] = "FAIL"
        inference_lock["failures"] = failures
        lock_path.write_text(json.dumps(inference_lock, indent=2) + "\n")
        raise RuntimeError(f"External blind inference failed: {failures}")
    (CHECKPOINTS / "CHECKPOINT_26A5_EXTERNAL_BLIND_PREDICTIONS_LOCKED").write_text(json.dumps(inference_lock, indent=2) + "\n")
    print(json.dumps(inference_lock, indent=2))


if __name__ == "__main__":
    main()
