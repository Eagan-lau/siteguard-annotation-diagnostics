#!/usr/bin/env python3
"""Calibrate, select, lock and evaluate Phase 24 EvidenceJudge."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4")).resolve()
RESULTS = ROOT / "results/phase24"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models/phase24/evidencejudge"
CHECKPOINTS = ROOT / "checkpoints"
SEED = 20260819
BOOTSTRAPS = 2000
TARGET_PRECISION = 0.95
MIN_ACCEPTED = 50
MIN_CLUSTERS = 20
LEVELS = ("EC_L3", "EC_L4", "EXACT_RHEA")
MODEL_PREFERENCE = {"LOGISTIC_STACK": 0, "LIGHTGBM_ROUTER": 1, "DEEPSETS_ROUTER": 2}
COVERAGE_GRID = np.unique(np.concatenate([
    np.array([0.001, 0.002, 0.005]),
    np.arange(0.01, 0.11, 0.01),
    np.arange(0.12, 1.001, 0.02),
])).round(6)


def context_seed(context: str) -> int:
    digest = hashlib.sha256(f"{SEED}|{context}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def cluster_bootstrap_precision(frame: pd.DataFrame, context: str) -> tuple[float, float]:
    if frame.empty:
        return float("nan"), float("nan")
    aggregate = frame.groupby("query_cluster_id_30", observed=True)["correct"].agg(["sum", "count"])
    values = aggregate[["sum", "count"]].to_numpy(dtype=float)
    if len(values) == 1:
        point = float(values[:, 0].sum() / values[:, 1].sum())
        return point, point
    rng = np.random.default_rng(context_seed(context))
    draws = np.empty(BOOTSTRAPS, dtype=float)
    for index in range(BOOTSTRAPS):
        sampled = values[rng.integers(0, len(values), size=len(values))]
        draws[index] = sampled[:, 0].sum() / sampled[:, 1].sum()
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def operating_curve(
    frame: pd.DataFrame,
    score_column: str,
    total_queries: int,
    context: str,
) -> pd.DataFrame:
    work = frame.loc[frame[score_column].notna()].sort_values(
        [score_column, "query_protein_id", "candidate_label"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    rows: list[dict[str, Any]] = []
    seen_thresholds: set[float] = set()
    for target_coverage in COVERAGE_GRID:
        target_n = int(math.ceil(float(target_coverage) * total_queries))
        if target_n < 1 or target_n > len(work):
            continue
        threshold = float(work.iloc[target_n - 1][score_column])
        rounded = round(threshold, 12)
        if rounded in seen_thresholds:
            continue
        seen_thresholds.add(rounded)
        accepted = work.loc[work[score_column] >= threshold]
        point = float(accepted["correct"].mean())
        clusters = int(accepted["query_cluster_id_30"].nunique())
        if len(accepted) >= MIN_ACCEPTED and clusters >= MIN_CLUSTERS and point >= TARGET_PRECISION:
            low, high = cluster_bootstrap_precision(accepted, f"{context}|{rounded}")
        else:
            low, high = float("nan"), float("nan")
        rows.append({
            "target_coverage_grid": float(target_coverage),
            "threshold": threshold,
            "queries_total": total_queries,
            "queries_accepted": len(accepted),
            "clusters_accepted": clusters,
            "coverage": float(len(accepted) / total_queries),
            "documented_precision": point,
            "cluster_ci_low": low,
            "cluster_ci_high": high,
            "qualifies_95": bool(
                len(accepted) >= MIN_ACCEPTED
                and clusters >= MIN_CLUSTERS
                and point >= TARGET_PRECISION
                and np.isfinite(low)
                and low >= TARGET_PRECISION
            ),
            "correct_accepted_coverage": float(accepted["correct"].sum() / total_queries),
        })
    return pd.DataFrame(rows)


def best_operating_point(curve: pd.DataFrame) -> dict[str, Any]:
    eligible = curve.loc[curve["qualifies_95"]].sort_values(
        ["coverage", "documented_precision", "threshold"], ascending=[False, False, False], kind="mergesort"
    )
    if eligible.empty:
        return {
            "threshold": float("inf"), "queries_accepted": 0, "clusters_accepted": 0,
            "coverage": 0.0, "documented_precision": float("nan"),
            "cluster_ci_low": float("nan"), "cluster_ci_high": float("nan"),
            "qualifies_95": False, "correct_accepted_coverage": 0.0,
        }
    return eligible.iloc[0].to_dict()


def evaluate_locked(
    frame: pd.DataFrame,
    score_column: str,
    threshold: float,
    total_queries: int,
    context: str,
) -> dict[str, Any]:
    accepted = frame.loc[frame[score_column].notna() & (frame[score_column] >= threshold)] if np.isfinite(threshold) else frame.iloc[0:0]
    clusters = int(accepted["query_cluster_id_30"].nunique())
    if accepted.empty:
        low, high, point = float("nan"), float("nan"), float("nan")
    else:
        point = float(accepted["correct"].mean())
        low, high = cluster_bootstrap_precision(accepted, context)
    qualifies = bool(
        len(accepted) >= MIN_ACCEPTED
        and clusters >= MIN_CLUSTERS
        and np.isfinite(point)
        and point >= TARGET_PRECISION
        and np.isfinite(low)
        and low >= TARGET_PRECISION
    )
    coverage = float(len(accepted) / total_queries)
    return {
        "queries_total": total_queries,
        "queries_accepted": len(accepted),
        "clusters_accepted": clusters,
        "coverage": coverage,
        "documented_precision": point,
        "cluster_ci_low": low,
        "cluster_ci_high": high,
        "qualifies_95": qualifies,
        "safe_coverage_at_95": coverage if qualifies else 0.0,
        "correct_accepted_coverage": float(accepted["correct"].sum() / total_queries) if len(accepted) else 0.0,
    }


def winner_rows(frame: pd.DataFrame, probability_column: str) -> pd.DataFrame:
    return (
        frame.sort_values(
            ["query_protein_id", probability_column, "supporting_channels", "supporting_tools", "candidate_label"],
            ascending=[True, False, False, False, True],
            kind="mergesort",
        )
        .drop_duplicates("query_protein_id")
        .copy()
    )


def expected_calibration_error(frame: pd.DataFrame, probability: str, bins: int = 10) -> float:
    if frame.empty:
        return float("nan")
    ranked = frame[[probability, "correct"]].sort_values(probability, kind="mergesort").reset_index(drop=True)
    assignments = np.minimum((np.arange(len(ranked)) * bins / max(len(ranked), 1)).astype(int), bins - 1)
    error = 0.0
    for index in range(bins):
        subset = ranked.loc[assignments == index]
        if subset.empty:
            continue
        error += len(subset) / len(ranked) * abs(float(subset[probability].mean()) - float(subset["correct"].mean()))
    return float(error)


def calibration_bins(frame: pd.DataFrame, probability: str, bins: int = 10) -> pd.DataFrame:
    ranked = frame[[probability, "correct"]].sort_values(probability, kind="mergesort").reset_index(drop=True)
    assignments = np.minimum((np.arange(len(ranked)) * bins / max(len(ranked), 1)).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        subset = ranked.loc[assignments == index]
        if subset.empty:
            continue
        rows.append({
            "bin": index + 1,
            "queries": len(subset),
            "mean_predicted_probability": float(subset[probability].mean()),
            "observed_documented_precision": float(subset["correct"].mean()),
            "min_probability": float(subset[probability].min()),
            "max_probability": float(subset[probability].max()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    for checkpoint in ("CHECKPOINT_24D1_CLASSICAL_PASS", "CHECKPOINT_24D2_DEEPSETS_PASS"):
        if not (CHECKPOINTS / checkpoint).is_file():
            raise RuntimeError(f"{checkpoint} is required")
    matrix = pd.read_parquet(RESULTS / "evidencejudge_candidate_matrix.parquet")
    schema = json.loads((RESULTS / "evidencejudge_feature_schema.json").read_text(encoding="utf-8"))
    feature_context = matrix[[
        "candidate_row_id", "supporting_tools", "supporting_channels", "available_tools", "available_channels",
        "tool_vote_fraction", "channel_vote_fraction", "candidate_set_size", "agreement_entropy", "vote_lead",
    ]]
    classical = pd.read_parquet(RESULTS / "evidencejudge_raw_predictions_classical.parquet")
    deep_scores = pd.read_csv(RESULTS / "evidencejudge_raw_predictions_deepsets.tsv.gz", sep="\t")
    deep = matrix[[
        "candidate_row_id", "pair_set", "development_partition", "annotation_level",
        "query_protein_id", "query_cluster_id_30", "candidate_label", "correct", "query_weight",
    ]].merge(deep_scores, on="candidate_row_id", how="inner", validate="one_to_one")
    raw = pd.concat([classical, deep], ignore_index=True, sort=False).merge(
        feature_context, on="candidate_row_id", how="left", validate="many_to_one"
    )
    if set(raw["model"].unique()) != set(MODEL_PREFERENCE):
        raise RuntimeError(f"Unexpected model inventory: {sorted(raw['model'].unique())}")
    total_queries = (
        matrix.groupby(["pair_set", "development_partition"], observed=True)["query_protein_id"].nunique().to_dict()
    )

    # Phase 1: only CAL outcomes are used to fit calibrators.
    calibrated_frames: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    for (level, model_name), group in raw.groupby(["annotation_level", "model"], observed=True):
        cal = group.loc[group["development_partition"].eq("CAL")]
        score = group.loc[group["development_partition"].isin(["SELECT", "TEST"])].copy()
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(
            cal["raw_probability"].to_numpy(dtype=float),
            cal["correct"].astype(int).to_numpy(),
            sample_weight=cal["query_weight"].to_numpy(dtype=float),
        )
        score["calibrated_probability"] = calibrator.predict(score["raw_probability"].to_numpy(dtype=float)).astype(np.float32)
        calibrated_frames.append(score)
        calibrator_path = MODELS / f"{str(level).lower()}__{str(model_name).lower()}__isotonic.joblib"
        joblib.dump(calibrator, calibrator_path, compress=3)
        cal_pred = calibrator.predict(cal["raw_probability"].to_numpy(dtype=float))
        calibration_rows.append({
            "annotation_level": level,
            "model": model_name,
            "cal_candidate_rows": len(cal),
            "cal_queries": cal["query_protein_id"].nunique(),
            "cal_clusters": cal["query_cluster_id_30"].nunique(),
            "cal_weighted_brier": float(np.average((cal_pred - cal["correct"].astype(float).to_numpy()) ** 2, weights=cal["query_weight"])),
            "calibrator_path": str(calibrator_path),
        })
    calibrated = pd.concat(calibrated_frames, ignore_index=True)
    pd.DataFrame(calibration_rows).to_csv(RESULTS / "evidencejudge_calibration_inventory.tsv", sep="\t", index=False)

    # Phase 2: SELECT outcomes screen models. TEST outcomes are not referenced here.
    development_curves: list[pd.DataFrame] = []
    development_rows: list[dict[str, Any]] = []
    selected_models: dict[str, str] = {}
    for level in LEVELS:
        for model_name in MODEL_PREFERENCE:
            selected_candidates = calibrated.loc[
                (calibrated["annotation_level"] == level)
                & (calibrated["model"] == model_name)
                & (calibrated["development_partition"] == "SELECT")
            ]
            winners = winner_rows(selected_candidates, "calibrated_probability")
            total = int(total_queries[("validation", "SELECT")])
            curve = operating_curve(winners, "calibrated_probability", total, f"SELECT|{level}|{model_name}")
            curve.insert(0, "model", model_name)
            curve.insert(0, "annotation_level", level)
            curve.insert(0, "evaluation_partition", "SELECT")
            development_curves.append(curve)
            point = best_operating_point(curve)
            development_rows.append({
                "annotation_level": level,
                "model": model_name,
                **point,
                "selection_role": "MODEL_SCREEN_BEFORE_HIERARCHY",
            })
        candidates = pd.DataFrame([row for row in development_rows if row["annotation_level"] == level])
        candidates["model_preference"] = candidates["model"].map(MODEL_PREFERENCE)
        candidates.sort_values(
            ["coverage", "documented_precision", "model_preference"],
            ascending=[False, False, True], kind="mergesort", inplace=True,
        )
        selected_models[level] = str(candidates.iloc[0]["model"])
    development_screen = pd.DataFrame(development_rows)
    development_screen["selected_for_level"] = [selected_models[row.annotation_level] == row.model for row in development_screen.itertuples()]
    development_screen.to_csv(RESULTS / "evidencejudge_model_screen.tsv", sep="\t", index=False)

    # Combine the SELECT predictions from the chosen models and enforce the frozen hierarchy.
    chosen = calibrated.loc[
        calibrated.apply(lambda row: selected_models[str(row["annotation_level"])] == row["model"], axis=1)
    ].copy()
    winners_by_level: list[pd.DataFrame] = []
    for (partition, level), frame in chosen.groupby(["development_partition", "annotation_level"], observed=True):
        winners = winner_rows(frame, "calibrated_probability")
        winners_by_level.append(winners)
    winners = pd.concat(winners_by_level, ignore_index=True)
    probability_wide = winners.pivot_table(
        index=["pair_set", "development_partition", "query_protein_id", "query_cluster_id_30"],
        columns="annotation_level", values="calibrated_probability", aggfunc="first"
    ).reset_index()
    for level in LEVELS:
        if level not in probability_wide:
            probability_wide[level] = np.nan
    probability_wide["hierarchical_probability__EC_L3"] = probability_wide["EC_L3"]
    probability_wide["hierarchical_probability__EC_L4"] = np.fmin(probability_wide["EC_L4"], probability_wide["EC_L3"])
    probability_wide["hierarchical_probability__EXACT_RHEA"] = np.fmin(
        probability_wide["EXACT_RHEA"], probability_wide["hierarchical_probability__EC_L4"]
    )
    hierarchy_columns = [
        "pair_set", "development_partition", "query_protein_id", "query_cluster_id_30",
        "hierarchical_probability__EC_L3", "hierarchical_probability__EC_L4", "hierarchical_probability__EXACT_RHEA",
    ]
    winners = winners.merge(probability_wide[hierarchy_columns], on=[
        "pair_set", "development_partition", "query_protein_id", "query_cluster_id_30"
    ], how="left", validate="many_to_one")
    winners["hierarchical_probability"] = [
        getattr(row, f"hierarchical_probability__{row.annotation_level}") for row in winners.itertuples()
    ]

    final_select_curves: list[pd.DataFrame] = []
    validation_points: list[dict[str, Any]] = []
    locked_thresholds: dict[str, float] = {}
    for level in LEVELS:
        frame = winners.loc[(winners["development_partition"] == "SELECT") & (winners["annotation_level"] == level)]
        total = int(total_queries[("validation", "SELECT")])
        curve = operating_curve(frame, "hierarchical_probability", total, f"SELECT|FINAL|{level}")
        curve.insert(0, "model", selected_models[level])
        curve.insert(0, "annotation_level", level)
        curve.insert(0, "evaluation_partition", "SELECT_HIERARCHY_LOCK")
        final_select_curves.append(curve)
        point = best_operating_point(curve)
        locked_thresholds[level] = float(point["threshold"])
        validation_points.append({
            "system": "EVIDENCEJUDGE",
            "annotation_level": level,
            "selected_model": selected_models[level],
            **point,
        })

    # Single-tool comparator selection uses the same SELECT clusters and coverage grid.
    top = pd.read_parquet(RESULTS / "existing_tool_top_predictions.parquet")
    partition_map = matrix[[
        "pair_set", "annotation_level", "query_protein_id", "query_cluster_id_30", "development_partition"
    ]].drop_duplicates()
    top = top.merge(partition_map, on=[
        "pair_set", "annotation_level", "query_protein_id", "query_cluster_id_30"
    ], how="left", validate="many_to_one")
    single_curves: list[pd.DataFrame] = []
    single_points: list[dict[str, Any]] = []
    locked_single: dict[str, dict[str, Any]] = {}
    for level in LEVELS:
        level_points: list[dict[str, Any]] = []
        for method in schema["methods"]:
            frame = top.loc[
                (top["annotation_level"] == level)
                & (top["method"] == method)
                & (top["development_partition"] == "SELECT")
            ].copy()
            total = int(total_queries[("validation", "SELECT")])
            curve = operating_curve(frame, "raw_score", total, f"SELECT|SINGLE|{level}|{method}")
            curve.insert(0, "model", method)
            curve.insert(0, "annotation_level", level)
            curve.insert(0, "evaluation_partition", "SELECT_SINGLE_TOOL")
            single_curves.append(curve)
            point = {"annotation_level": level, "method": method, **best_operating_point(curve)}
            level_points.append(point)
            single_points.append(point)
        ranked = pd.DataFrame(level_points).sort_values(
            ["coverage", "documented_precision", "method"], ascending=[False, False, True], kind="mergesort"
        )
        if float(ranked.iloc[0]["coverage"]) <= 0:
            locked_single[level] = {
                "annotation_level": level,
                "method": "NO_SINGLE_TOOL_QUALIFIED",
                "threshold": float("inf"),
                "coverage": 0.0,
                "qualifies_95": False,
            }
        else:
            locked_single[level] = ranked.iloc[0].to_dict()

    lock = {
        "locked_before_test_evaluation": True,
        "seed": SEED,
        "target_documented_precision": TARGET_PRECISION,
        "ci_requirement": "cluster bootstrap 95% CI lower bound >= 0.95",
        "bootstraps": BOOTSTRAPS,
        "minimum_accepted_queries": MIN_ACCEPTED,
        "minimum_accepted_clusters": MIN_CLUSTERS,
        "coverage_grid": COVERAGE_GRID.tolist(),
        "selected_models": selected_models,
        "evidencejudge_thresholds": locked_thresholds,
        "best_single_tools": {
            level: {"method": value["method"], "threshold": float(value["threshold"])}
            for level, value in locked_single.items()
        },
        "development_partitions": {"FIT": "model fit", "CAL": "isotonic calibration", "SELECT": "model and threshold selection"},
        "test_role": "POST_HOC_FEASIBILITY_ONLY",
    }
    lock_path = MODELS / "evidencejudge_operating_point_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, allow_nan=True), encoding="utf-8")
    (CHECKPOINTS / "CHECKPOINT_24D3_THRESHOLDS_LOCKED").write_text(
        json.dumps({"status": "LOCKED", "lock_path": str(lock_path)}, indent=2), encoding="utf-8"
    )

    # Phase 3: only after the lock file exists do we use TEST outcomes.
    test_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    all_curves = development_curves + final_select_curves + single_curves
    calibration_test_rows: list[dict[str, Any]] = []
    calibration_bin_frames: list[pd.DataFrame] = []
    for level in LEVELS:
        total = int(total_queries[("test", "TEST")])
        evidence_frame = winners.loc[(winners["development_partition"] == "TEST") & (winners["annotation_level"] == level)]
        evidence_result = evaluate_locked(
            evidence_frame, "hierarchical_probability", locked_thresholds[level], total, f"TEST|EVIDENCEJUDGE|{level}"
        )
        evidence_row = {
            "system": "EVIDENCEJUDGE",
            "annotation_level": level,
            "model": selected_models[level],
            "locked_threshold": locked_thresholds[level],
            **evidence_result,
            "interpretation": "POST_HOC_FEASIBILITY",
        }
        test_rows.append(evidence_row)
        evidence_curve = operating_curve(evidence_frame, "hierarchical_probability", total, f"TEST_CURVE|EVIDENCEJUDGE|{level}")
        evidence_curve.insert(0, "model", selected_models[level])
        evidence_curve.insert(0, "annotation_level", level)
        evidence_curve.insert(0, "evaluation_partition", "TEST_DESCRIPTIVE_EVIDENCEJUDGE")
        all_curves.append(evidence_curve)

        baseline = locked_single[level]
        single_frame = top.loc[
            (top["annotation_level"] == level)
            & (top["method"] == baseline["method"])
            & (top["development_partition"] == "TEST")
        ].copy()
        single_result = evaluate_locked(
            single_frame, "raw_score", float(baseline["threshold"]), total, f"TEST|SINGLE|{level}|{baseline['method']}"
        )
        single_row = {
            "system": "BEST_SINGLE_TOOL",
            "annotation_level": level,
            "model": baseline["method"],
            "locked_threshold": float(baseline["threshold"]),
            **single_result,
            "interpretation": "POST_HOC_FEASIBILITY",
        }
        test_rows.append(single_row)
        single_curve = operating_curve(single_frame, "raw_score", total, f"TEST_CURVE|SINGLE|{level}|{baseline['method']}")
        single_curve.insert(0, "model", baseline["method"])
        single_curve.insert(0, "annotation_level", level)
        single_curve.insert(0, "evaluation_partition", "TEST_DESCRIPTIVE_BEST_SINGLE")
        all_curves.append(single_curve)
        comparison_rows.append({
            "annotation_level": level,
            "evidencejudge_model": selected_models[level],
            "best_single_tool": baseline["method"],
            "evidencejudge_locked_coverage": evidence_result["coverage"],
            "best_single_locked_coverage": single_result["coverage"],
            "locked_coverage_gain": evidence_result["coverage"] - single_result["coverage"],
            "evidencejudge_safe_coverage_at_95": evidence_result["safe_coverage_at_95"],
            "best_single_safe_coverage_at_95": single_result["safe_coverage_at_95"],
            "incremental_safe_coverage": evidence_result["safe_coverage_at_95"] - single_result["safe_coverage_at_95"],
            "evidencejudge_precision": evidence_result["documented_precision"],
            "evidencejudge_ci_low": evidence_result["cluster_ci_low"],
            "evidencejudge_ci_high": evidence_result["cluster_ci_high"],
            "best_single_precision": single_result["documented_precision"],
            "best_single_ci_low": single_result["cluster_ci_low"],
            "best_single_ci_high": single_result["cluster_ci_high"],
            "both_meet_95_ci": bool(evidence_result["qualifies_95"] and single_result["qualifies_95"]),
            "interpretation": "POST_HOC_FEASIBILITY",
        })
        calibration_test_rows.append({
            "annotation_level": level,
            "model": selected_models[level],
            "queries_with_candidate": len(evidence_frame),
            "brier_score": float(brier_score_loss(evidence_frame["correct"].astype(int), evidence_frame["hierarchical_probability"])),
            "ece_equal_mass_10": expected_calibration_error(evidence_frame, "hierarchical_probability", bins=10),
        })
        bins = calibration_bins(evidence_frame, "hierarchical_probability", bins=10)
        bins.insert(0, "model", selected_models[level])
        bins.insert(0, "annotation_level", level)
        calibration_bin_frames.append(bins)

    pd.DataFrame(validation_points).to_csv(RESULTS / "evidencejudge_validation_operating_points.tsv", sep="\t", index=False)
    pd.DataFrame(single_points).to_csv(RESULTS / "evidencejudge_single_tool_selection.tsv", sep="\t", index=False)
    pd.DataFrame(test_rows).to_csv(RESULTS / "evidencejudge_test_operating_points.tsv", sep="\t", index=False)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(RESULTS / "evidencejudge_incremental_safe_coverage.tsv", sep="\t", index=False)
    pd.DataFrame(calibration_test_rows).to_csv(RESULTS / "evidencejudge_test_calibration.tsv", sep="\t", index=False)
    pd.concat(calibration_bin_frames, ignore_index=True).to_csv(RESULTS / "evidencejudge_test_calibration_bins.tsv", sep="\t", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(RESULTS / "evidencejudge_risk_coverage_curves.tsv", sep="\t", index=False)

    # Query-level output with transparent acceptance/review/abstention decisions.
    release_gate = {
        level: bool(next(
            row["qualifies_95"] for row in test_rows
            if row["system"] == "EVIDENCEJUDGE" and row["annotation_level"] == level
        ))
        for level in LEVELS
    }
    query_output = probability_wide[["pair_set", "development_partition", "query_protein_id", "query_cluster_id_30"]].copy()
    for level in LEVELS:
        level_winners = winners.loc[winners["annotation_level"] == level, [
            "pair_set", "development_partition", "query_protein_id", "candidate_label",
            "hierarchical_probability", "supporting_tools", "supporting_channels",
            "available_tools", "available_channels", "agreement_entropy", "tool_vote_fraction",
        ]].rename(columns={
            "candidate_label": f"candidate__{level}",
            "hierarchical_probability": f"probability__{level}",
            "supporting_tools": f"supporting_tools__{level}",
            "supporting_channels": f"supporting_channels__{level}",
            "available_tools": f"available_tools__{level}",
            "available_channels": f"available_channels__{level}",
            "agreement_entropy": f"agreement_entropy__{level}",
            "tool_vote_fraction": f"tool_vote_fraction__{level}",
        })
        query_output = query_output.merge(
            level_winners, on=["pair_set", "development_partition", "query_protein_id"], how="left", validate="one_to_one"
        )
        query_output[f"accepted__{level}"] = query_output[f"probability__{level}"] >= locked_thresholds[level]

    validation_lookup = {row["annotation_level"]: row for row in validation_points}
    decisions: list[dict[str, Any]] = []
    for row in query_output.itertuples(index=False):
        chosen_level: str | None = None
        for level in reversed(LEVELS):
            if bool(getattr(row, f"accepted__{level}")):
                chosen_level = level
                break
        if chosen_level is not None and release_gate[chosen_level]:
            point = validation_lookup[chosen_level]
            decisions.append({
                "recommended_annotation_level": chosen_level,
                "candidate_label_or_set": getattr(row, f"candidate__{chosen_level}"),
                "estimated_documented_precision": getattr(row, f"probability__{chosen_level}"),
                "precision_ci_low": point["cluster_ci_low"],
                "precision_ci_high": point["cluster_ci_high"],
                "supported_coverage_operating_point": point["coverage"],
                "decision": "ACCEPT",
                "calibration_scope_warning": "FROZEN_VALIDATION_DISTRIBUTION; TEST_RESULT_POST_HOC",
            })
        elif chosen_level is not None:
            decisions.append({
                "recommended_annotation_level": "NONE",
                "candidate_label_or_set": getattr(row, f"candidate__{chosen_level}"),
                "estimated_documented_precision": getattr(row, f"probability__{chosen_level}"),
                "precision_ci_low": np.nan,
                "precision_ci_high": np.nan,
                "supported_coverage_operating_point": 0.0,
                "decision": "REVIEW",
                "calibration_scope_warning": f"TEST_GATE_FAILED::{chosen_level}; VALIDATION_LOCK_CANDIDATE_ONLY",
            })
        else:
            entropy = getattr(row, "agreement_entropy__EC_L3")
            decision = "REVIEW" if np.isfinite(entropy) and entropy > 0 else "ABSTAIN"
            decisions.append({
                "recommended_annotation_level": "NONE",
                "candidate_label_or_set": None,
                "estimated_documented_precision": np.nan,
                "precision_ci_low": np.nan,
                "precision_ci_high": np.nan,
                "supported_coverage_operating_point": 0.0,
                "decision": decision,
                "calibration_scope_warning": "FROZEN_VALIDATION_DISTRIBUTION; TEST_RESULT_POST_HOC",
            })
    query_output = pd.concat([query_output.reset_index(drop=True), pd.DataFrame(decisions)], axis=1)
    query_output["chembridge_rescue_status"] = "NOT_AVAILABLE_ON_POPULATION_QUERY_SET"
    query_output.to_parquet(RESULTS / "evidencejudge_query_predictions.parquet", index=False, compression="zstd")

    hierarchy_ok = bool(
        (probability_wide["hierarchical_probability__EXACT_RHEA"].fillna(-1) <= probability_wide["hierarchical_probability__EC_L4"].fillna(1) + 1e-12).all()
        and (probability_wide["hierarchical_probability__EC_L4"].fillna(-1) <= probability_wide["hierarchical_probability__EC_L3"].fillna(1) + 1e-12).all()
    )
    test_frame = pd.DataFrame(test_rows)
    checks = [
        ("threshold_lock_exists_before_test_output", (CHECKPOINTS / "CHECKPOINT_24D3_THRESHOLDS_LOCKED").is_file(), str(lock_path)),
        ("all_three_model_families_screened", set(development_screen["model"]) == set(MODEL_PREFERENCE), sorted(development_screen["model"].unique())),
        ("all_three_levels_selected", set(selected_models) == set(LEVELS), selected_models),
        ("hierarchical_probability_constraint", hierarchy_ok, len(probability_wide)),
        ("six_test_system_level_rows", len(test_frame) == 6, len(test_frame)),
        ("three_incremental_endpoints", len(comparison) == 3, len(comparison)),
        ("test_not_used_for_calibrator", all(row["cal_queries"] > 0 for row in calibration_rows), calibration_rows),
        ("query_output_unique", not query_output.duplicated(["pair_set", "development_partition", "query_protein_id"]).any(), len(query_output)),
        ("decision_values_valid", set(query_output["decision"]).issubset({"ACCEPT", "REVIEW", "ABSTAIN"}), sorted(query_output["decision"].unique())),
        ("probabilities_finite_when_present", np.isfinite(winners["hierarchical_probability"].dropna()).all(), len(winners)),
    ]
    qc = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    qc.to_csv(REPORTS / "phase24_evidencejudge_final_qc.tsv", sep="\t", index=False)
    failures = qc.loc[~qc["passed"].astype(bool), "check"].tolist()
    any_positive_gain = bool((comparison["incremental_safe_coverage"] > 0).any())
    any_locked_gain = bool((comparison["locked_coverage_gain"] > 0).any())
    deep_promoted = any(model == "DEEPSETS_ROUTER" for model in selected_models.values())
    summary = {
        "phase": "24D",
        "stage": "validation_only_evidencejudge_and_posthoc_test",
        "status": "PASS" if not failures else "FAIL",
        "interpretation": "POST_HOC_FEASIBILITY_ONLY",
        "target_precision": TARGET_PRECISION,
        "selected_models": selected_models,
        "deep_model_promoted": deep_promoted,
        "release_gate": release_gate,
        "released_accept_queries": int(query_output["decision"].eq("ACCEPT").sum()),
        "any_positive_incremental_safe_coverage": any_positive_gain,
        "any_positive_locked_coverage_gain": any_locked_gain,
        "go_no_go": "GO_INTERNAL_FEASIBILITY" if any_positive_gain else ("CONTINUE_TO_NEW_BLIND_HOLDOUT" if any_locked_gain else "NO_LEARNED_ROUTER_ADVANTAGE"),
        "comparison": comparison.to_dict(orient="records"),
        "checks": len(checks),
        "passed": len(checks) - len(failures),
        "failures": failures,
    }
    (REPORTS / "phase24_evidencejudge_final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"EvidenceJudge final QC failed: {failures}")
    (CHECKPOINTS / "CHECKPOINT_24D_PASS").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
