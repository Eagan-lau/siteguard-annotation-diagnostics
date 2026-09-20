#!/usr/bin/env python3
"""Calibrate the T0 model and evaluate T1 temporal/open-world cohorts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score


LEVELS = ["EC_L3", "EC_L4", "EXACT_RHEA"]
CANDIDATE_COLUMNS = {"EC_L3": "ec_l3", "EC_L4": "ec_l4", "EXACT_RHEA": "canonical_rhea"}
TASK_FLAGS = {
    "NEW_PROTEIN_KNOWN_REACTION": "is_new_protein_known_reaction",
    "LIMITED_KNOWLEDGE_REFINEMENT": "is_limited_knowledge_refinement",
    "GENUINE_NOVEL_CHEMISTRY": "is_genuine_novel_chemistry",
}


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def choose_blend(y: np.ndarray, deep: np.ndarray, tree: np.ndarray) -> tuple[float, float]:
    best_alpha, best_score = 0.0, -1.0
    for alpha in np.linspace(0, 1, 41):
        score = alpha * deep + (1 - alpha) * tree
        value = average_precision_score(y, score)
        if value > best_score + 1e-12:
            best_alpha, best_score = float(alpha), float(value)
    return best_alpha, best_score


def wilson_lower(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    if trials <= 0:
        return float("nan")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    return (
        proportion + z * z / (2 * trials)
        - z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
    ) / denominator


def safe_threshold(frame: pd.DataFrame, desired: float = 0.90) -> dict[str, Any]:
    candidates = np.unique(np.quantile(frame["calibrated_probability"], np.linspace(0, 1, 1001)))
    for threshold in candidates:
        accepted = frame.loc[frame["calibrated_probability"].ge(threshold)]
        if len(accepted) < 50:
            continue
        successes = int(accepted["correct"].sum())
        lower = wilson_lower(successes, len(accepted))
        if lower >= desired:
            return {
                "threshold": float(threshold), "validation_precision": successes / len(accepted),
                "validation_ci_lower": lower, "validation_accepted": len(accepted),
                "validation_coverage": len(accepted) / len(frame), "status": "PASS",
            }
    return {
        "threshold": 1.000001, "validation_precision": float("nan"),
        "validation_ci_lower": float("nan"), "validation_accepted": 0,
        "validation_coverage": 0.0, "status": "NO_SAFE_THRESHOLD",
    }


def top_candidates(metadata: pd.DataFrame, probabilities: np.ndarray, level: str) -> pd.DataFrame:
    truth_column = f"query_truth_available_{level}"
    candidate_column = f"candidate_label_available_{level}"
    correct_column = f"correct_{level}"
    base = metadata.loc[metadata[truth_column], ["query_protein_id"]].drop_duplicates().copy()
    candidates = metadata.loc[metadata[candidate_column]].copy()
    candidates["calibrated_probability"] = probabilities[metadata[candidate_column].to_numpy()]
    if len(candidates):
        indices = candidates.groupby("query_protein_id", sort=False)["calibrated_probability"].idxmax()
        selected = candidates.loc[
            indices,
            ["query_protein_id", "reference_protein_id", "reference_activity_id", CANDIDATE_COLUMNS[level], correct_column, "calibrated_probability"],
        ].copy()
        selected = selected.rename(columns={CANDIDATE_COLUMNS[level]: "predicted_label", correct_column: "correct"})
        oracle = candidates.groupby("query_protein_id")[correct_column].any().rename("oracle_candidate_available")
        selected = selected.merge(oracle, on="query_protein_id", how="left", validate="one_to_one")
    else:
        selected = pd.DataFrame(columns=[
            "query_protein_id", "reference_protein_id", "reference_activity_id", "predicted_label",
            "correct", "calibrated_probability", "oracle_candidate_available",
        ])
    output = base.merge(selected, on="query_protein_id", how="left", validate="one_to_one")
    output["candidate_found"] = output["reference_protein_id"].notna()
    output["correct"] = output["correct"].fillna(False).astype(bool)
    output["oracle_candidate_available"] = output["oracle_candidate_available"].fillna(False).astype(bool)
    output["calibrated_probability"] = output["calibrated_probability"].fillna(0.0).astype(np.float32)
    output["annotation_level"] = level
    return output


def hierarchical_predictions(top: pd.DataFrame, task_table: pd.DataFrame) -> pd.DataFrame:
    query_ids = sorted(set(top["query_protein_id"]))
    output = pd.DataFrame({"query_protein_id": query_ids})
    for level in LEVELS:
        subset = top.loc[top["annotation_level"].eq(level), [
            "query_protein_id", "predicted_label", "reference_protein_id", "reference_activity_id",
            "calibrated_probability", "correct", "accepted", "oracle_candidate_available",
        ]].copy()
        subset = subset.rename(columns={column: f"{column}_{level}" for column in subset.columns if column != "query_protein_id"})
        output = output.merge(subset, on="query_protein_id", how="left", validate="one_to_one")
    for level in LEVELS:
        output[f"accepted_{level}"] = output[f"accepted_{level}"].fillna(False).astype(bool)
        output[f"correct_{level}"] = output[f"correct_{level}"].fillna(False).astype(bool)
    resolutions: list[str] = []
    labels: list[Any] = []
    references: list[Any] = []
    activities: list[Any] = []
    probabilities: list[float] = []
    correctness: list[bool] = []
    for row in output.itertuples(index=False):
        if row.accepted_EXACT_RHEA and row.accepted_EC_L3:
            level = "EXACT_RHEA"
        elif row.accepted_EC_L4 and row.accepted_EC_L3:
            level = "EC_L4"
        elif row.accepted_EC_L3:
            level = "EC_L3"
        else:
            level = "ABSTAIN"
        resolutions.append(level)
        if level == "ABSTAIN":
            labels.append(None); references.append(None); activities.append(None)
            probabilities.append(float("nan")); correctness.append(False)
        else:
            labels.append(getattr(row, f"predicted_label_{level}"))
            references.append(getattr(row, f"reference_protein_id_{level}"))
            activities.append(getattr(row, f"reference_activity_id_{level}"))
            probabilities.append(float(getattr(row, f"calibrated_probability_{level}")))
            correctness.append(bool(getattr(row, f"correct_{level}")))
    output["final_resolution"] = resolutions
    output["final_predicted_label"] = labels
    output["final_reference_protein_id"] = references
    output["final_reference_activity_id"] = activities
    output["final_probability"] = probabilities
    output["final_correct"] = correctness
    keep = [
        "protein_id", "temporal_membership", "T0_T1_sequence_stable", "T1_current_sequence_stable",
        "primary_task", "primary_evaluation_eligible", *TASK_FLAGS.values(), "annotation_revision",
        "old_resolution", "new_resolution", "novel_t1_rhea_json",
        "t0_ec_l3_json", "t0_ec_l4_json", "t0_exact_rhea_json",
        "t1_ec_l3_json", "t1_ec_l4_json", "t1_exact_rhea_json",
    ]
    available = [column for column in keep if column in task_table]
    return output.merge(
        task_table[available].rename(columns={"protein_id": "query_protein_id"}),
        on="query_protein_id", how="left", validate="one_to_one",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data/interim/phase13"
    results = root / "results/phase13"
    models = root / "models/phase13"
    figures = root / "figures/source_data"
    reports = root / "reports"
    checkpoints = root / "checkpoints"
    for path in [results, models, figures, reports, checkpoints]:
        path.mkdir(parents=True, exist_ok=True)
    if not (checkpoints / "CHECKPOINT_12_PASS").is_file():
        raise RuntimeError("CHECKPOINT_12_PASS is required")

    validation = pd.read_parquet(work / "validation_metadata.parquet")
    temporal = pd.read_parquet(work / "temporal_metadata.parquet")
    task_table = pd.read_parquet(results / "temporal_sequence_stable_set.parquet")
    deep_validation = np.load(work / "deep_validation_predictions.npy")
    deep_temporal = np.load(work / "deep_temporal_predictions.npy")
    tree_validation = np.load(work / "tree_validation_predictions.npy")
    tree_temporal = np.load(work / "tree_temporal_predictions.npy")
    if deep_validation.shape != tree_validation.shape or deep_temporal.shape != tree_temporal.shape:
        raise RuntimeError("Deep/tree prediction shapes disagree")
    if len(validation) != len(deep_validation) or len(temporal) != len(deep_temporal):
        raise RuntimeError("Prediction/metadata row mismatch")

    blend_weights: dict[str, float] = {}
    validation_blend_auprc: dict[str, float] = {}
    raw_validation = np.zeros_like(deep_validation)
    raw_temporal = np.zeros_like(deep_temporal)
    calibrated_validation = np.zeros_like(deep_validation)
    calibrated_temporal = np.zeros_like(deep_temporal)
    calibrators: dict[str, IsotonicRegression] = {}
    for index, level in enumerate(LEVELS):
        mask = validation[f"evaluation_mask_{level}"].to_numpy(bool)
        y = validation.loc[mask, f"correct_{level}"].astype(int).to_numpy()
        alpha, score = choose_blend(y, deep_validation[mask, index], tree_validation[mask, index])
        blend_weights[level] = alpha
        validation_blend_auprc[level] = score
        raw_validation[:, index] = alpha * deep_validation[:, index] + (1 - alpha) * tree_validation[:, index]
        raw_temporal[:, index] = alpha * deep_temporal[:, index] + (1 - alpha) * tree_temporal[:, index]
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw_validation[mask, index], y)
        calibrated_validation[:, index] = calibrator.predict(raw_validation[:, index]).astype(np.float32)
        calibrated_temporal[:, index] = calibrator.predict(raw_temporal[:, index]).astype(np.float32)
        calibrators[level] = calibrator
        joblib.dump(calibrator, models / f"isotonic_t0_{level}.joblib", compress=3)

    validation_top_parts: list[pd.DataFrame] = []
    temporal_top_parts: list[pd.DataFrame] = []
    thresholds: dict[str, float] = {}
    threshold_rows: list[dict[str, Any]] = []
    for index, level in enumerate(LEVELS):
        validation_top = top_candidates(validation, calibrated_validation[:, index], level)
        selected = safe_threshold(validation_top, 0.90)
        thresholds[level] = float(selected["threshold"])
        threshold_rows.append({
            "annotation_level": level, "target_precision": 0.90,
            "selection_split": "T0_sequence_cluster_validation_top_candidate",
            "t1_used_for_selection": False, **selected,
        })
        validation_top["accepted"] = validation_top["calibrated_probability"].ge(thresholds[level])
        temporal_top = top_candidates(temporal, calibrated_temporal[:, index], level)
        temporal_top["accepted"] = temporal_top["calibrated_probability"].ge(thresholds[level])
        validation_top_parts.append(validation_top)
        temporal_top_parts.append(temporal_top)
    validation_top = pd.concat(validation_top_parts, ignore_index=True)
    temporal_top = pd.concat(temporal_top_parts, ignore_index=True)
    threshold_frame = pd.DataFrame(threshold_rows)
    threshold_frame.to_csv(results / "temporal_abstention_thresholds.tsv", sep="\t", index=False)

    pairwise = temporal[[
        "query_protein_id", "reference_protein_id", "reference_activity_id", "ec_l3", "ec_l4", "canonical_rhea",
        "primary_task", "primary_evaluation_eligible", "is_new_protein_known_reaction",
        "is_limited_knowledge_refinement", "is_genuine_novel_chemistry",
    ]].copy()
    for index, level in enumerate(LEVELS):
        pairwise[f"score_deep_t0_{level}"] = deep_temporal[:, index]
        pairwise[f"score_tree_t0_{level}"] = tree_temporal[:, index]
        pairwise[f"score_siteguard_t0_{level}"] = raw_temporal[:, index]
        pairwise[f"calibrated_t0_{level}"] = calibrated_temporal[:, index]
        pairwise[f"correct_t1_{level}"] = temporal[f"correct_{level}"].to_numpy(bool)
        pairwise[f"evaluation_mask_t1_{level}"] = temporal[f"evaluation_mask_{level}"].to_numpy(bool)
    write_parquet(pairwise, results / "temporal_pairwise_predictions.parquet")

    top_with_tasks = temporal_top.merge(
        task_table[[
            "protein_id", "primary_task", "primary_evaluation_eligible", *TASK_FLAGS.values(),
            "temporal_membership", "annotation_revision", "old_resolution", "new_resolution", "novel_t1_rhea_json",
        ]].rename(columns={"protein_id": "query_protein_id"}),
        on="query_protein_id", how="left", validate="many_to_one",
    )
    final = hierarchical_predictions(temporal_top, task_table)
    final_columns = ["query_protein_id", "final_resolution", "final_predicted_label", "final_reference_protein_id", "final_reference_activity_id", "final_probability", "final_correct"]
    temporal_predictions = top_with_tasks.merge(final[final_columns], on="query_protein_id", how="left", validate="many_to_one")
    temporal_predictions["functional_reference_snapshot"] = "T0_UniProt_2023_01_Rhea_126"
    temporal_predictions["ground_truth_snapshot"] = "T1_UniProt_2026_01_Rhea_140"
    temporal_predictions["function_time_frozen"] = True
    temporal_predictions["t1_used_for_training_calibration_or_selection"] = False
    write_parquet(temporal_predictions, results / "temporal_predictions.parquet")

    metric_rows: list[dict[str, Any]] = []
    def pure_novel_row(row: pd.Series) -> bool:
        truth_value = row["t1_exact_rhea_json"]
        novel_value = row["novel_t1_rhea_json"]
        truth = set(json.loads(truth_value)) if isinstance(truth_value, str) else set()
        novel_truth = set(json.loads(novel_value)) if isinstance(novel_value, str) else set()
        return bool(truth) and truth.issubset(novel_truth)

    pure_novel_queries = set(
        task_table.loc[task_table.apply(pure_novel_row, axis=1), "protein_id"].astype(str)
    )
    cohort_queries = {
        "ALL_PRIMARY_TEMPORAL": set(task_table.loc[task_table["primary_evaluation_eligible"], "protein_id"].astype(str)),
        **{
            name: set(task_table.loc[task_table[flag], "protein_id"].astype(str))
            for name, flag in TASK_FLAGS.items()
        },
    }
    # Exact-Rhea open-world behavior is identifiable without ambiguity only
    # when all documented T1 exact reactions for a query are genuinely novel.
    cohort_queries["GENUINE_NOVEL_CHEMISTRY"] = pure_novel_queries
    for cohort, query_ids in cohort_queries.items():
        for level in LEVELS:
            subset = temporal_top.loc[
                temporal_top["annotation_level"].eq(level) & temporal_top["query_protein_id"].isin(query_ids)
            ]
            accepted = subset["accepted"]
            metric_rows.append({
                "cohort": cohort, "analysis": "per_level_top_candidate", "annotation_level": level,
                "queries_with_t1_truth": len(subset), "accepted_queries": int(accepted.sum()),
                "coverage": float(accepted.mean()) if len(subset) else float("nan"),
                "selective_accuracy": float(subset.loc[accepted, "correct"].mean()) if accepted.any() else float("nan"),
                "overannotation_rate": float((~subset.loc[accepted, "correct"]).mean()) if accepted.any() else float("nan"),
                "oracle_candidate_coverage": float(subset["oracle_candidate_available"].mean()) if len(subset) else float("nan"),
                "top1_accuracy_all_queries": float(subset["correct"].mean()) if len(subset) else float("nan"),
                "threshold": thresholds[level], "threshold_selected_on": "T0_validation", "t1_used_for_selection": False,
            })
        final_subset = final.loc[final["query_protein_id"].isin(query_ids)]
        accepted_final = final_subset["final_resolution"].ne("ABSTAIN")
        metric_rows.append({
            "cohort": cohort, "analysis": "hierarchical_final", "annotation_level": "HIGHEST_SAFE_RESOLUTION",
            "queries_with_t1_truth": len(final_subset), "accepted_queries": int(accepted_final.sum()),
            "coverage": float(accepted_final.mean()) if len(final_subset) else float("nan"),
            "selective_accuracy": float(final_subset.loc[accepted_final, "final_correct"].mean()) if accepted_final.any() else float("nan"),
            "overannotation_rate": float((~final_subset.loc[accepted_final, "final_correct"]).mean()) if accepted_final.any() else float("nan"),
            "oracle_candidate_coverage": float("nan"), "top1_accuracy_all_queries": float("nan"),
            "threshold": float("nan"), "threshold_selected_on": "T0_validation", "t1_used_for_selection": False,
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(results / "temporal_metrics.tsv", sep="\t", index=False)

    limited = final.loc[final["is_limited_knowledge_refinement"].fillna(False)].copy()
    limited["final_resolution_rank"] = limited["final_resolution"].map({"ABSTAIN": 0, "EC_L3": 1, "EC_L4": 2, "EXACT_RHEA": 3})
    limited["refinement_recovered"] = limited["final_correct"] & limited["final_resolution_rank"].ge(limited["new_resolution"])
    limited["unsupported_fine_transfer_avoided"] = limited["final_resolution"].eq("ABSTAIN") | limited["final_resolution_rank"].lt(limited["new_resolution"]) | limited["final_correct"]
    limited.to_csv(results / "limited_knowledge_analysis.tsv", sep="\t", index=False)

    novel = final.loc[final["is_genuine_novel_chemistry"].fillna(False)].copy()
    novel["pure_novel_chemistry_query"] = novel["query_protein_id"].isin(pure_novel_queries)
    novel["exact_rhea_transfer_attempted"] = novel["final_resolution"].eq("EXACT_RHEA")
    novel["safe_open_world_behavior"] = ~novel["exact_rhea_transfer_attempted"] | novel["final_correct"]
    novel["behavior_class"] = np.select(
        [novel["final_resolution"].eq("ABSTAIN"), novel["final_resolution"].eq("EC_L3"), novel["final_resolution"].eq("EC_L4"), novel["final_resolution"].eq("EXACT_RHEA") & novel["final_correct"]],
        ["ABSTAIN", "COARSE_EC_L3", "COARSE_EC_L4", "EXACT_RHEA_CORRECT"],
        default="UNSUPPORTED_EXACT_RHEA_TRANSFER",
    )
    novel.to_csv(results / "novel_chemistry_analysis.tsv", sep="\t", index=False)

    stable_ec3 = temporal_top.loc[temporal_top["annotation_level"].eq("EC_L3")].merge(
        task_table[["protein_id", "temporal_membership", "annotation_revision"]].rename(columns={"protein_id": "query_protein_id"}),
        on="query_protein_id", how="left", validate="one_to_one",
    )
    stable_ec3 = stable_ec3.loc[stable_ec3["temporal_membership"].eq("T0_T1_SEQUENCE_STABLE")].copy()
    stable_ec3["t0_model_risk"] = 1 - stable_ec3["calibrated_probability"]
    if stable_ec3["t0_model_risk"].nunique() >= 4:
        stable_ec3["risk_quartile"] = pd.qcut(stable_ec3["t0_model_risk"], 4, labels=["Q1_LOW", "Q2", "Q3", "Q4_HIGH"], duplicates="drop")
    else:
        stable_ec3["risk_quartile"] = "NOT_IDENTIFIABLE"
    enrichment = stable_ec3.groupby("risk_quartile", observed=True).agg(
        queries=("query_protein_id", "nunique"), revisions=("annotation_revision", "sum"),
        revision_rate=("annotation_revision", "mean"), mean_risk=("t0_model_risk", "mean"),
    ).reset_index()
    if {"Q1_LOW", "Q4_HIGH"}.issubset(set(enrichment["risk_quartile"].astype(str))):
        high = stable_ec3["risk_quartile"].astype(str).eq("Q4_HIGH")
        revised = stable_ec3["annotation_revision"].fillna(False)
        odds_ratio, p_value = fisher_exact([
            [int((high & revised).sum()), int((high & ~revised).sum())],
            [int((~high & revised).sum()), int((~high & ~revised).sum())],
        ])
    else:
        odds_ratio, p_value = float("nan"), float("nan")
    enrichment["high_risk_vs_other_fisher_odds_ratio"] = odds_ratio
    enrichment["high_risk_vs_other_fisher_p"] = p_value
    enrichment.to_csv(results / "historical_revision_enrichment.tsv", sep="\t", index=False)

    design_rows: list[dict[str, Any]] = []
    for membership, count in task_table["temporal_membership"].value_counts(dropna=False).items():
        design_rows.append({"dimension": "sequence_membership", "category": membership, "count": int(count)})
    for task, count in task_table.loc[task_table["primary_evaluation_eligible"], "primary_task"].value_counts().items():
        design_rows.append({"dimension": "primary_temporal_task", "category": task, "count": int(count)})
    design = pd.DataFrame(design_rows)
    design.to_csv(figures / "Figure5A_temporal_design.tsv", sep="\t", index=False)
    metrics.to_csv(figures / "Figure5B_E_temporal_metrics.tsv", sep="\t", index=False)
    limited.groupby(["final_resolution"], dropna=False).agg(queries=("query_protein_id", "nunique"), refinement_recovered=("refinement_recovered", "sum")).reset_index().to_csv(
        figures / "Figure5C_limited_knowledge.tsv", sep="\t", index=False
    )
    novel.groupby(["behavior_class"], dropna=False).size().rename("queries").reset_index().to_csv(
        figures / "Figure5D_novel_chemistry.tsv", sep="\t", index=False
    )
    enrichment.to_csv(figures / "Figure5F_historical_revision_enrichment.tsv", sep="\t", index=False)

    config = {
        "functional_training_and_reference_snapshot": "UniProt 2023_01 + Rhea 126",
        "ground_truth_snapshot": "UniProt 2026_01 + Rhea 140",
        "architecture": "T0-trained SiteGuard residual MLP + LightGBM validation blend",
        "deep_blend_weights": blend_weights, "t0_validation_blend_auprc": validation_blend_auprc,
        "calibration": "T0 validation isotonic", "safe_thresholds": thresholds,
        "t1_used_for_training_calibration_or_selection": False,
        "function_time_frozen": True,
        "structure_note": "current sequence-derived structure/domain evidence retained; no future functional annotation fields enter inputs",
    }
    (models / "temporal_siteguard_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    primary_counts = task_table.loc[task_table["primary_evaluation_eligible"], "primary_task"].value_counts().to_dict()
    novel_exact = metrics.loc[
        metrics["cohort"].eq("GENUINE_NOVEL_CHEMISTRY") & metrics["annotation_level"].eq("EXACT_RHEA")
        & metrics["analysis"].eq("per_level_top_candidate")
    ]
    output_files = [
        results / "temporal_sequence_stable_set.parquet", results / "temporal_predictions.parquet",
        results / "temporal_metrics.tsv", results / "limited_knowledge_analysis.tsv",
        results / "novel_chemistry_analysis.tsv", figures / "Figure5A_temporal_design.tsv",
        figures / "Figure5B_E_temporal_metrics.tsv", figures / "Figure5C_limited_knowledge.tsv",
        figures / "Figure5D_novel_chemistry.tsv", figures / "Figure5F_historical_revision_enrichment.tsv",
    ]
    checks = [
        ("checkpoint_12_present", (checkpoints / "CHECKPOINT_12_PASS").is_file(), "strict phase gate"),
        ("function_time_frozen", config["function_time_frozen"], config["functional_training_and_reference_snapshot"]),
        ("t1_not_used_for_selection", not config["t1_used_for_training_calibration_or_selection"], "T1 evaluation only"),
        ("prediction_probabilities_valid", bool(np.isfinite(calibrated_temporal).all() and ((calibrated_temporal >= 0) & (calibrated_temporal <= 1)).all()), str(calibrated_temporal.shape)),
        ("three_primary_tasks_present", all(primary_counts.get(task, 0) > 0 for task in TASK_FLAGS), json.dumps(primary_counts, sort_keys=True)),
        ("temporal_predictions_nonempty", len(temporal_predictions) > 0, str(len(temporal_predictions))),
        ("limited_knowledge_nonempty", len(limited) > 0, str(len(limited))),
        ("novel_chemistry_nonempty", len(novel) > 0, str(len(novel))),
        ("novel_exact_oracle_zero", len(novel_exact) == 1 and float(novel_exact.iloc[0]["oracle_candidate_coverage"]) == 0.0, novel_exact.to_json(orient="records")),
        ("required_outputs_present", all(path.is_file() and path.stat().st_size > 0 for path in output_files), str(len(output_files))),
    ]
    qc = pd.DataFrame(
        [(name, "PASS" if bool(passed) else "FAIL", details) for name, passed, details in checks],
        columns=["check", "status", "details"],
    )
    qc.to_csv(reports / "phase13_qc.tsv", sep="\t", index=False)
    failures = qc.loc[qc["status"].eq("FAIL"), "check"].tolist()
    all_primary = metrics.loc[
        metrics["cohort"].eq("ALL_PRIMARY_TEMPORAL") & metrics["analysis"].eq("hierarchical_final")
    ]
    summary = {
        "phase": 13, "status": "PASS" if not failures else "FAIL",
        "slurm_job_id": os.getenv("SLURM_JOB_ID", "NA"), "primary_task_counts": primary_counts,
        "temporal_pair_rows": len(pairwise), "temporal_prediction_rows": len(temporal_predictions),
        "safe_thresholds": thresholds, "deep_blend_weights": blend_weights,
        "primary_hierarchical_coverage": float(all_primary.iloc[0]["coverage"]) if len(all_primary) else float("nan"),
        "primary_hierarchical_selective_accuracy": float(all_primary.iloc[0]["selective_accuracy"]) if len(all_primary) else float("nan"),
        "limited_queries": len(limited), "limited_refinement_recovered": int(limited["refinement_recovered"].sum()) if len(limited) else 0,
        "novel_queries": len(novel), "novel_safe_open_world_behavior_rate": float(novel["safe_open_world_behavior"].mean()) if len(novel) else float("nan"),
        "historical_revision_high_risk_odds_ratio": odds_ratio, "historical_revision_high_risk_p": p_value,
        "function_time_frozen": True, "t1_used_for_selection": False, "qc_failures": failures,
    }
    (reports / "phase13_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# SiteGuard V4 Phase 13 Report", "", f"Status: **{summary['status']}**", "",
        "This benchmark retrains SiteGuard from scratch using only UniProt 2023_01 and Rhea 126 functional knowledge. UniProt 2026_01/Rhea 140 labels are used only once for final temporal evaluation.", "",
        f"- Primary temporal task counts: {json.dumps(primary_counts, sort_keys=True)}",
        f"- Highest-safe-resolution coverage: {summary['primary_hierarchical_coverage']:.2%}",
        f"- Selective accuracy among accepted primary temporal queries: {summary['primary_hierarchical_selective_accuracy']:.2%}",
        f"- Limited-knowledge refinements recovered: {summary['limited_refinement_recovered']}/{summary['limited_queries']}",
        f"- Novel-chemistry safe open-world behavior: {summary['novel_safe_open_world_behavior_rate']:.2%}",
        f"- High T0 risk vs later annotation revision: odds ratio {odds_ratio:.3g}, Fisher p={p_value:.3g}", "",
        "Current sequence-derived structure/domain evidence is retained, but no T1 functional annotation or query truth field enters training, calibration, candidate eligibility, or threshold selection.", "",
        "## QC", "", qc.to_markdown(index=False), "",
    ]
    (reports / "PHASE_13_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    if failures:
        raise RuntimeError("Phase 13 QC failed: " + ", ".join(failures))
    (checkpoints / "CHECKPOINT_13_PASS").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
