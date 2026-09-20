#!/usr/bin/env python3
"""Apply the frozen augmented EvidenceJudge to Phase30 without reading EC truth."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sabio_finalize_blind_inference as base
from siteguard.model import blend_scores


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
R30 = ROOT / "results/phase30"
WORK = ROOT / "data/interim/phase30_inference"
CHECKPOINTS = ROOT / "checkpoints"
REPORTS = ROOT / "reports/phase30_external_blind"
MODEL_DIR = ROOT / "results/phase29_augmented"
LEVELS = ("EC_L3", "EC_L4", "EXACT_RHEA")
FIXED_OLD_METHODS = {
    "SEQUENCE_IDENTITY", "ESM2_SIMILARITY", "FOLDSEEK_IDENTITY", "PFAM_JACCARD",
    "LIGHTGBM_GLOBAL", "DEEP_GLOBAL", "SITEGUARD",
}
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flag(value: object) -> int:
    return int(pd.notna(value) and bool(value))


def augmented_candidates(
    locked: pd.DataFrame, metadata: pd.DataFrame, old: pd.DataFrame,
    hit: pd.DataFrame, clean: pd.DataFrame,
) -> pd.DataFrame:
    hit = hit.rename(columns={
        "query_id": "query_protein_id", "sequence_length": "hit_sequence_length",
        "truncated_residue_count": "hit_truncated_residue_count",
    })
    clean = clean.rename(columns={
        "query_id": "query_protein_id", "sequence_length": "clean_sequence_length",
        "truncated_residue_count": "clean_truncated_residue_count",
    })
    query = locked.rename(columns={"query_id": "query_protein_id"}).merge(
        metadata, on="query_protein_id", how="left", validate="one_to_one", suffixes=("", "_metadata")
    ).merge(hit, on="query_protein_id", validate="one_to_one").merge(
        clean, on="query_protein_id", validate="one_to_one"
    )
    query["clean_ec3_top1"] = query["clean_top1_ec4"].str.rsplit(".", n=1).str[0]
    records = []
    for level, number in [("EC_L3", 3), ("EC_L4", 4)]:
        old_level = old.loc[old["annotation_level"].eq(level) & old["method"].isin(FIXED_OLD_METHODS)].copy()
        old_level.sort_values(["query_protein_id", "method", "candidate_label"], kind="mergesort", inplace=True)
        if old_level.duplicated(["query_protein_id", "method"]).any():
            raise RuntimeError(f"{level}: duplicate fixed-tool prediction")
        method_channel = old_level[["method", "evidence_channel"]].drop_duplicates().set_index("method")["evidence_channel"].to_dict()
        old_by_query = {key: group for key, group in old_level.groupby("query_protein_id", observed=True)}
        for _, row in query.iterrows():
            query_id = str(row["query_protein_id"])
            rows = old_by_query.get(query_id, pd.DataFrame(columns=old_level.columns))
            old_predictions = {
                str(item.method): {
                    "label": str(item.candidate_label),
                    "score": float(item.raw_score) if pd.notna(item.raw_score) else np.nan,
                    "margin": float(item.top1_margin) if pd.notna(item.top1_margin) else np.nan,
                    "channel": str(item.evidence_channel),
                }
                for item in rows.itertuples(index=False)
            }
            hit_label = str(row[f"ec{number}_top1"])
            clean_label = str(row["clean_ec3_top1"] if number == 3 else row["clean_top1_ec4"])
            labels = sorted({item["label"] for item in old_predictions.values()} | {hit_label, clean_label})
            counts = Counter([item["label"] for item in old_predictions.values()] + [hit_label, clean_label])
            total_votes = len(old_predictions) + 2
            sorted_counts = sorted(counts.values(), reverse=True)
            second_votes = sorted_counts[1] if len(sorted_counts) > 1 else 0
            proportions = np.asarray(list(counts.values()), float) / total_votes
            entropy = float(-(proportions * np.log(proportions)).sum() / max(math.log(len(proportions)), 1e-12))
            for candidate in labels:
                record: dict[str, object] = {
                    "annotation_level": level,
                    "query_protein_id": query_id,
                    "query_cluster_id_30": str(row["external_cluster_id_30"]),
                    "candidate_label": candidate,
                    "available_old_tools": len(old_predictions),
                    "available_tools_augmented": total_votes,
                    "candidate_set_size": len(labels),
                    "agreement_entropy_augmented": entropy,
                    "candidate_votes_augmented": counts[candidate],
                    "candidate_vote_fraction_augmented": counts[candidate] / total_votes,
                    "second_candidate_votes_augmented": second_votes,
                    "candidate_vote_lead_augmented": counts[candidate] - second_votes,
                    "candidate_is_plurality": int(counts[candidate] == sorted_counts[0]),
                    "support__HIT_EC": int(candidate == hit_label),
                    "support__CLEAN": int(candidate == clean_label),
                    "support_score__HIT_EC": float(row[f"ec{number}_top1_softmax"]) if candidate == hit_label else np.nan,
                    "support_score__CLEAN": float(row["clean_top1_gmm_confidence"]) if candidate == clean_label else np.nan,
                    "hit_old_agreement_count": sum(item["label"] == hit_label for item in old_predictions.values()),
                    "clean_old_agreement_count": sum(item["label"] == clean_label for item in old_predictions.values()),
                    "hit_clean_agree": int(hit_label == clean_label),
                    "hit_top1_softmax": float(row[f"ec{number}_top1_softmax"]),
                    "hit_top1_logit": float(row[f"ec{number}_top1_logit"]),
                    "hit_softmax_margin12": float(row[f"ec{number}_softmax_margin12"]),
                    "hit_ec4_sigmoid": float(row["ec4_top1_sigmoid"]),
                    "hit_ec4_sigmoid_margin12": float(row["ec4_sigmoid_margin12"]),
                    "hit_ec3_ec4_consistent": int(row["ec3_ec4_hierarchy_consistent"]),
                    "clean_top1_distance": float(row["clean_top1_distance"]),
                    "clean_top1_gmm_confidence": float(row["clean_top1_gmm_confidence"]),
                    "clean_distance_margin12": float(row["clean_distance_margin12"]),
                    "clean_distance_ratio12": float(row["clean_distance_ratio12"]),
                    "clean_maxsep_n": int(row["clean_maxsep_n"]),
                    "log1p_sequence_length": math.log1p(float(row["hit_sequence_length"])),
                    "log1p_hit_truncation": math.log1p(float(row["hit_truncated_residue_count"])),
                    "nearest_frozen_identity_fraction": float(row.get("nearest_frozen_identity_fraction", np.nan)),
                    "pfam_seen_in_fit": flag(row.get("pfam_seen_in_fit")),
                    "pfam_clan_seen_in_fit": flag(row.get("pfam_clan_seen_in_fit")),
                    "cath_seen_in_fit": flag(row.get("cath_seen_in_fit")),
                    "query_structure_available": flag(row.get("query_structure_available")),
                }
                supporting_channels = set()
                for method, item in old_predictions.items():
                    support = int(candidate == item["label"])
                    record[f"support__{method}"] = support
                    record[f"query_score__{method}"] = item["score"]
                    record[f"query_margin__{method}"] = item["margin"]
                    record[f"support_score__{method}"] = item["score"] if support else np.nan
                    record[f"support_margin__{method}"] = item["margin"] if support else np.nan
                    if support:
                        supporting_channels.add(method_channel[method])
                if candidate == hit_label:
                    supporting_channels.add("hierarchical_transformer")
                if candidate == clean_label:
                    supporting_channels.add("contrastive_embedding")
                record["supporting_channels_augmented"] = len(supporting_channels)
                records.append(record)
    return pd.DataFrame(records)


def single_tool_predictions(
    locked: pd.DataFrame, old: pd.DataFrame, hit: pd.DataFrame, clean: pd.DataFrame, policies: pd.DataFrame,
) -> pd.DataFrame:
    cluster = locked.set_index("query_id")["external_cluster_id_30"].to_dict()
    rows = []
    for policy in policies.itertuples(index=False):
        level = str(policy.annotation_level)
        number = 3 if level == "EC_L3" else 4
        method = str(policy.method)
        if method in FIXED_OLD_METHODS:
            predictions = old.loc[old["annotation_level"].eq(level) & old["method"].eq(method)]
            for item in predictions.itertuples(index=False):
                score = getattr(item, str(policy.score_variant)) if str(policy.score_variant) != "NONE" else np.nan
                rows.append({
                    "annotation_level": level, "query_protein_id": item.query_protein_id,
                    "query_cluster_id_30": cluster[str(item.query_protein_id)], "method": method,
                    "candidate_label": str(item.candidate_label), "selection_score": score,
                    "policy_status": policy.policy_status,
                })
        elif method == "HIT_EC":
            for item in hit.itertuples(index=False):
                rows.append({
                    "annotation_level": level, "query_protein_id": item.query_id,
                    "query_cluster_id_30": cluster[str(item.query_id)], "method": method,
                    "candidate_label": str(getattr(item, f"ec{number}_top1")),
                    "selection_score": getattr(item, str(policy.score_variant)) if str(policy.score_variant) != "NONE" else np.nan,
                    "policy_status": policy.policy_status,
                })
        elif method == "CLEAN":
            for item in clean.itertuples(index=False):
                label = str(item.clean_top1_ec4).rsplit(".", 1)[0] if number == 3 else str(item.clean_top1_ec4)
                rows.append({
                    "annotation_level": level, "query_protein_id": item.query_id,
                    "query_cluster_id_30": cluster[str(item.query_id)], "method": method,
                    "candidate_label": label,
                    "selection_score": getattr(item, str(policy.score_variant)) if str(policy.score_variant) != "NONE" else np.nan,
                    "policy_status": policy.policy_status,
                })
        start = len(rows)
        # Acceptance is applied after all rows for this policy are appended below.
        del start
    output = pd.DataFrame(rows).merge(
        policies[["annotation_level", "method", "direction", "threshold"]],
        on=["annotation_level", "method"], how="left", validate="many_to_one",
    )
    output["accepted"] = False
    high = output["direction"].eq("HIGH")
    low = output["direction"].eq("LOW")
    output.loc[high, "accepted"] = output.loc[high, "selection_score"].ge(output.loc[high, "threshold"])
    output.loc[low, "accepted"] = output.loc[low, "selection_score"].le(output.loc[low, "threshold"])
    return output


def main() -> None:
    required = [
        CHECKPOINTS / "CHECKPOINT_30A5A_EXTERNAL_PAIR_EVIDENCE_PASS",
        CHECKPOINTS / "CHECKPOINT_30A5B_EXTERNAL_DEEP_SCORE_PASS",
        ROOT / "data/interim/phase30_hit_ec/phase30_hit_ec_predictions.tsv",
        ROOT / "data/interim/phase30_clean/phase30_clean_predictions.tsv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if (R30 / "external_blind_predictions.parquet").exists() or (R30 / "external_predictions.parquet").exists():
        raise RuntimeError("Phase30 inference/evaluation output already exists")
    REPORTS.mkdir(parents=True, exist_ok=True)

    cohort_columns = ["query_id", "external_cluster_id_30", "nearest_frozen_identity_fraction"]
    locked = pd.read_parquet(R30 / "rcsb_strict_blind_cohort.parquet", columns=cohort_columns)
    metadata = pd.read_parquet(R30 / "external_query_metadata.parquet")
    hit = pd.read_csv(required[2], sep="\t")
    clean = pd.read_csv(required[3], sep="\t")

    pairs = pd.read_parquet(R30 / "external_pair_evidence_blind.parquet")
    deep = np.load(WORK / "external_deep_global_scores.npy")
    if len(pairs) != len(deep):
        raise RuntimeError(f"pair/deep mismatch: {len(pairs)} vs {deep.shape}")
    for index, level in enumerate(LEVELS):
        pairs[f"score_deep_global_{level}"] = deep[:, index].astype(np.float32)
    config = json.loads((ROOT / "models/phase11/siteguard_model_config.json").read_text())
    tree = pairs[[f"score_lightgbm_global_{level}" for level in LEVELS]].to_numpy(np.float32)
    blended = blend_scores(deep.astype(np.float32), tree, config["deep_blend_weights"], LEVELS)
    for index, level in enumerate(LEVELS):
        pairs[f"score_siteguard_{level}"] = blended[:, index]
    pairs.to_parquet(R30 / "external_pair_scores_blind.parquet", index=False, compression="zstd")

    base.AVAILABLE_METHOD_SCORE = AVAILABLE_METHOD_SCORE
    top_all = base.make_top_predictions(pairs, metadata)
    top_all.to_parquet(R30 / "external_tool_top_predictions_all_blind.parquet", index=False, compression="zstd")
    top = top_all.loc[top_all["method"].isin(FIXED_OLD_METHODS)].copy()
    top.to_parquet(R30 / "external_tool_top_predictions_blind.parquet", index=False, compression="zstd")

    candidates = augmented_candidates(locked, metadata, top, hit, clean)
    freeze_path = MODEL_DIR / "augmented_evidencejudge_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    features = list(freeze["feature_columns"])
    missing_features = sorted(set(features) - set(candidates.columns))
    for feature in missing_features:
        candidates[feature] = np.nan
    candidates[features] = candidates[features].apply(pd.to_numeric, errors="coerce")

    info = freeze["selected"]["EC_L3"]
    model_path = Path(info["model_path"])
    if sha256(model_path) != info["model_sha256"]:
        raise RuntimeError("frozen EC3 model hash mismatch")
    model = joblib.load(model_path)
    ec3 = candidates.loc[candidates["annotation_level"].eq("EC_L3")].copy()
    ec3["selection_score"] = model.predict_proba(ec3[features])[:, 1]
    ec3.sort_values(
        ["query_protein_id", "selection_score", "candidate_label"],
        ascending=[True, False, True], kind="mergesort", inplace=True,
    )
    winners = ec3.drop_duplicates("query_protein_id")[[
        "annotation_level", "query_protein_id", "query_cluster_id_30", "candidate_label", "selection_score",
    ]].copy()
    winners["system_model"] = str(info["model"])
    winners["frozen_threshold"] = float(info["threshold"])
    winners["accepted"] = winners["selection_score"].ge(winners["frozen_threshold"])
    winners["interpretation"] = "PHASE30_BLIND_PREDICTION_TRUTH_NOT_OPENED"
    winners.to_parquet(R30 / "external_blind_predictions.parquet", index=False, compression="zstd")
    candidates.to_parquet(R30 / "external_augmented_candidate_matrix_blind.parquet", index=False, compression="zstd")

    hit_rows = candidates.loc[candidates["support__HIT_EC"].eq(1)].copy()
    if hit_rows.duplicated(["annotation_level", "query_protein_id"]).any():
        raise RuntimeError("HIT candidate grain is not unique")
    hit_rows["accepted"] = False
    ec3_mask = hit_rows["annotation_level"].eq("EC_L3")
    ec4_mask = hit_rows["annotation_level"].eq("EC_L4")
    hit_rows.loc[ec3_mask, "accepted"] = hit_rows.loc[ec3_mask, "hit_old_agreement_count"].ge(5) | (
        hit_rows.loc[ec3_mask, "hit_old_agreement_count"].eq(4)
        & hit_rows.loc[ec3_mask, "clean_distance_margin12"].ge(1.5050787925720217)
    )
    hit_rows.loc[ec4_mask, "accepted"] = hit_rows.loc[ec4_mask, "hit_old_agreement_count"].ge(5) | (
        hit_rows.loc[ec4_mask, "hit_old_agreement_count"].eq(4)
        & hit_rows.loc[ec4_mask, "clean_distance_margin12"].ge(1.712409257888794)
    )
    hit_rows["system_model"] = np.where(ec3_mask, "TRANSPARENT_EC3_RULE", "TRANSPARENT_EC4_RULE")
    hit_rows[[
        "annotation_level", "query_protein_id", "query_cluster_id_30", "candidate_label",
        "hit_old_agreement_count", "clean_distance_margin12", "accepted", "system_model",
    ]].to_parquet(R30 / "external_blind_rule_predictions.parquet", index=False, compression="zstd")

    policies = pd.read_csv(MODEL_DIR / "single_tool_policy_freeze.tsv", sep="\t")
    baselines = single_tool_predictions(locked, top, hit, clean, policies)
    baselines.to_parquet(R30 / "external_blind_single_tool_predictions.parquet", index=False, compression="zstd")

    lock = {
        "phase": "30B", "status": "PASS", "truth_opened": False,
        "strict_cohort_queries": len(locked),
        "fixed_old_methods": sorted(FIXED_OLD_METHODS),
        "available_old_methods": sorted(set(top["method"])),
        "candidate_rows": len(candidates), "blind_winner_rows": len(winners),
        "accepted_before_truth": int(winners["accepted"].sum()),
        "transparent_rule_accepted_before_truth": hit_rows.groupby("annotation_level")["accepted"].sum().astype(int).to_dict(),
        "feature_count": len(features), "features_missing_and_preserved_as_nan": missing_features,
        "model_freeze_sha256": sha256(freeze_path), "model_sha256": sha256(model_path),
        "single_tool_policy_sha256": sha256(MODEL_DIR / "single_tool_policy_freeze.json"),
        "cohort_lock_sha256": sha256(CHECKPOINTS / "CHECKPOINT_30A_EXTERNAL_COHORT_LOCKED"),
        "inference_script_sha256": sha256(Path(__file__)),
        "hit_manifest_truth_free": json.loads(Path(str(required[2]) + ".manifest.json").read_text())["truth_inputs_used"] is False,
        "clean_manifest_truth_free": json.loads(Path(str(ROOT / "data/interim/phase30_clean/phase30_clean_predictions_raw.tsv") + ".manifest.json").read_text())["truth_inputs_used"] is False,
        "truth_columns_read": [], "cohort_columns_read": cohort_columns,
    }
    lock_path = ROOT / "models/phase30_external_inference_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checks = {
        "truth_columns_not_read": not lock["truth_columns_read"],
        "fixed_feature_schema_complete": not missing_features,
        "candidate_label_not_a_feature": "candidate_label" not in features,
        "one_winner_per_query": len(winners) == len(locked) and not winners["query_protein_id"].duplicated().any(),
        "both_new_tool_manifests_truth_free": lock["hit_manifest_truth_free"] and lock["clean_manifest_truth_free"],
        "frozen_model_hash_matches": sha256(model_path) == info["model_sha256"],
    }
    (REPORTS / "phase30_blind_inference_qc.json").write_text(json.dumps(checks, indent=2) + "\n")
    failures = [key for key, value in checks.items() if not value]
    if failures:
        lock["status"] = "FAIL"; lock["failures"] = failures
        lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError(f"Phase30 blind inference QC failed: {failures}")
    (CHECKPOINTS / "CHECKPOINT_30B_BLIND_PREDICTIONS_LOCKED").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(lock, indent=2, sort_keys=True))
    print("CHECKPOINT_30B_BLIND_PREDICTIONS_LOCKED")


if __name__ == "__main__":
    main()
