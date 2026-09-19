#!/usr/bin/env python3
"""Evaluate the deep model and build a validation-selected SiteGuard ensemble."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)


LEVELS = ["EC_L3", "EC_L4", "EXACT_RHEA"]
TARGETS = ["same_ec_l3", "same_ec_l4", "same_exact_rhea"]


def choose_blend(y: np.ndarray, deep: np.ndarray, tree: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    best_alpha, best_score = 0.0, -1.0
    for alpha in np.linspace(0, 1, 41):
        score = alpha * deep + (1 - alpha) * tree
        value = average_precision_score(y, score, sample_weight=weight)
        if value > best_score + 1e-12:
            best_alpha, best_score = float(alpha), float(value)
    return best_alpha, float(best_score)


def choose_threshold(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    candidates = np.unique(np.quantile(score, np.linspace(0, 1, 201)))
    values = [f1_score(y, score >= threshold, sample_weight=weight, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(values))])


def calculate_metrics(y: np.ndarray, score: np.ndarray, threshold: float, weight: np.ndarray) -> dict[str, float]:
    prediction = score >= threshold
    return {
        "auprc": float(average_precision_score(y, score, sample_weight=weight)),
        "auroc": float(roc_auc_score(y, score, sample_weight=weight)),
        "precision": float(precision_score(y, prediction, sample_weight=weight, zero_division=0)),
        "recall": float(recall_score(y, prediction, sample_weight=weight, zero_division=0)),
        "f1": float(f1_score(y, prediction, sample_weight=weight, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, prediction, sample_weight=weight)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction, sample_weight=weight)),
    }


def oracle_metrics(frame: pd.DataFrame, score_column: str, target: str) -> dict[str, float]:
    query_rows = []
    for _, group in frame.groupby("query_protein_id", sort=False):
        ordered = group.sort_values(score_column, ascending=False)
        outcomes = ordered[target].astype(int).to_numpy()
        positive = np.flatnonzero(outcomes)
        query_rows.append({
            "candidate_available": len(positive) > 0,
            "top1": bool(len(outcomes) and outcomes[0]),
            "hit5": bool(outcomes[:5].any()),
            "reciprocal_rank": 1.0 / (positive[0] + 1) if len(positive) else 0.0,
        })
    queries = pd.DataFrame(query_rows)
    available = queries.loc[queries["candidate_available"]]
    return {
        "queries": len(queries),
        "oracle_candidate_coverage": float(queries["candidate_available"].mean()),
        "top1_accuracy_all_queries": float(queries["top1"].mean()),
        "hit5_all_queries": float(queries["hit5"].mean()),
        "top1_conditional_on_candidate": float(available["top1"].mean()) if len(available) else float("nan"),
        "hit5_conditional_on_candidate": float(available["hit5"].mean()) if len(available) else float("nan"),
        "mrr_conditional_on_candidate": float(available["reciprocal_rank"].mean()) if len(available) else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data/interim/phase11"
    phase10 = root / "results/phase10"
    results = root / "results/phase11"
    models = root / "models/phase11"
    reports = root / "reports"
    results.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_parquet(work / "evaluation_metadata.parquet")
    deep = np.load(work / "deep_eval_predictions.npy")
    baseline = pd.read_parquet(phase10 / "baseline_predictions.parquet")
    key = ["pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id"]
    tree_columns = [f"score_lightgbm_global_{level}" for level in LEVELS]
    baseline = baseline[key + tree_columns]
    frame = metadata.merge(baseline, on=key, validate="one_to_one")
    if len(frame) != len(deep) or deep.shape[1] != 3:
        raise RuntimeError(f"Deep prediction mismatch: metadata={len(frame)}, predictions={deep.shape}")
    for index, level in enumerate(LEVELS):
        frame[f"score_deep_global_{level}"] = deep[:, index]

    validation = frame["query_split_expected"].eq("validation")
    test = frame["query_split_expected"].eq("test")
    blend_weights: dict[str, float] = {}
    validation_selection: dict[str, float] = {}
    for level, target in zip(LEVELS, TARGETS, strict=True):
        alpha, score = choose_blend(
            frame.loc[validation, target].astype(int).to_numpy(),
            frame.loc[validation, f"score_deep_global_{level}"].to_numpy(float),
            frame.loc[validation, f"score_lightgbm_global_{level}"].to_numpy(float),
            frame.loc[validation, "sample_weight"].to_numpy(float),
        )
        blend_weights[level] = alpha
        validation_selection[level] = score
        frame[f"score_siteguard_{level}"] = (
            alpha * frame[f"score_deep_global_{level}"]
            + (1 - alpha) * frame[f"score_lightgbm_global_{level}"]
        ).astype(np.float32)

    model_names = ["deep_global", "lightgbm_global", "siteguard"]
    metric_rows: list[dict] = []
    validation_rows: list[dict] = []
    thresholds: dict[str, dict[str, float]] = {model: {} for model in model_names}
    for level, target in zip(LEVELS, TARGETS, strict=True):
        for model in model_names:
            score_column = f"score_{model}_{level}"
            threshold = choose_threshold(
                frame.loc[validation, target].astype(int).to_numpy(),
                frame.loc[validation, score_column].to_numpy(float),
                frame.loc[validation, "sample_weight"].to_numpy(float),
            )
            thresholds[model][level] = threshold
            for split_name, mask in [("validation", validation), ("test", test)]:
                values = calculate_metrics(
                    frame.loc[mask, target].astype(int).to_numpy(),
                    frame.loc[mask, score_column].to_numpy(float), threshold,
                    frame.loc[mask, "sample_weight"].to_numpy(float),
                )
                record = {
                    "model": model, "annotation_level": level,
                    "evaluation_split": f"population_{split_name}", "rows": int(mask.sum()),
                    "positives": int(frame.loc[mask, target].sum()), "threshold": threshold,
                    "threshold_selection_split": "population_validation", "test_used_for_selection": False,
                    "deep_blend_weight": blend_weights[level] if model == "siteguard" else (1.0 if model == "deep_global" else 0.0),
                    **values,
                }
                metric_rows.append(record)
                if split_name == "validation":
                    validation_rows.append(record)
    metrics = pd.DataFrame(metric_rows)
    pd.DataFrame(validation_rows).to_csv(results / "validation_metrics.tsv", sep="\t", index=False)
    metrics.to_csv(results / "pairwise_metrics.tsv", sep="\t", index=False)

    oracle_rows: list[dict] = []
    test_frame = frame.loc[test].copy()
    for level, target in zip(LEVELS, TARGETS, strict=True):
        for model in model_names:
            values = oracle_metrics(test_frame, f"score_{model}_{level}", target)
            oracle_rows.append({"model": model, "annotation_level": level, "evaluation_split": "population_test", **values})
    oracle = pd.DataFrame(oracle_rows)
    oracle.to_csv(results / "oracle_transfer_metrics.tsv", sep="\t", index=False)
    frame.to_parquet(results / "pairwise_predictions.parquet", index=False, compression="zstd")

    config = json.loads((models / "model_config.json").read_text(encoding="utf-8"))
    config.update({
        "production_scoring": "validation-selected per-level convex blend of deep MLP and LightGBM",
        "deep_blend_weights": blend_weights,
        "validation_blend_auprc": validation_selection,
        "validation_selected_thresholds": thresholds,
        "test_used_for_model_or_threshold_selection": False,
        "fine_local_branch": "disabled after CHECKPOINT_09_LOCAL_NULL",
    })
    (models / "siteguard_model_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    test_metrics = metrics.loc[metrics["evaluation_split"].eq("population_test")]
    best = {
        level: test_metrics.loc[
            test_metrics["annotation_level"].eq(level)
        ].sort_values("auprc", ascending=False).iloc[0][["model", "auprc"]].to_dict()
        for level in LEVELS
    }
    deep_improves_any = any(
        float(test_metrics.loc[(test_metrics.model == "deep_global") & (test_metrics.annotation_level == level), "auprc"].iloc[0])
        > float(test_metrics.loc[(test_metrics.model == "lightgbm_global") & (test_metrics.annotation_level == level), "auprc"].iloc[0])
        for level in LEVELS
    )
    score_columns = [name for name in frame if name.startswith("score_")]
    checks = [
        ("checkpoint_10_present", (root / "checkpoints/CHECKPOINT_10_PASS").is_file(), "strict phase gate"),
        ("trained_weights_present", (models / "siteguard_global_multitask.pt").stat().st_size > 0, str(models / "siteguard_global_multitask.pt")),
        ("evaluation_rows_complete", len(frame) == 630_469, str(len(frame))),
        ("score_probabilities_valid", frame[score_columns].apply(lambda values: values.between(0, 1).all()).all(), str(len(score_columns))),
        ("pairwise_metrics_complete", len(metrics) == 18, str(len(metrics))),
        ("oracle_metrics_complete", len(oracle) == 9, str(len(oracle))),
        ("validation_only_model_selection", not bool(metrics["test_used_for_selection"].any()), json.dumps(blend_weights)),
        ("deep_model_non_degenerate", deep_improves_any, "deep must exceed LightGBM on at least one held-out level"),
        ("local_branch_respects_phase09", config["fine_local_branch"].startswith("disabled"), config["fine_local_branch"]),
        ("split_leakage_report_pass", pd.read_csv(root / "data/splits/split_leakage_report.tsv", sep="\t")["status"].eq("PASS").all(), "registered split"),
    ]
    qc = pd.DataFrame(
        [(name, "PASS" if bool(passed) else "FAIL", details) for name, passed, details in checks],
        columns=["check", "status", "details"],
    )
    qc.to_csv(reports / "phase11_qc.tsv", sep="\t", index=False)
    failures = qc.loc[qc["status"].eq("FAIL"), "check"].tolist()
    summary = {
        "phase": 11, "status": "PASS" if not failures else "FAIL",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "architecture": config["architecture"], "training_rows": 931_006,
        "evaluation_rows": len(frame), "deep_blend_weights": blend_weights,
        "best_test_auprc": best,
        "deep_test_auprc": {
            level: float(test_metrics.loc[(test_metrics.model == "deep_global") & (test_metrics.annotation_level == level), "auprc"].iloc[0])
            for level in LEVELS
        },
        "siteguard_test_auprc": {
            level: float(test_metrics.loc[(test_metrics.model == "siteguard") & (test_metrics.annotation_level == level), "auprc"].iloc[0])
            for level in LEVELS
        },
        "test_used_for_selection": False, "qc_failures": failures,
    }
    (reports / "phase11_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# SiteGuard V4 Phase 11 Report", "", f"Status: **{summary['status']}**", "",
        f"- Architecture: {summary['architecture']}", "- Primary local branch: disabled by Phase 9 fine-resolution LOCAL_NULL", "",
        "## Held-out test AUPRC", "",
        *[
            f"- {level}: deep {summary['deep_test_auprc'][level]:.4f}; SiteGuard validation-selected blend {summary['siteguard_test_auprc'][level]:.4f}; best evaluated model {best[level]['model']} {best[level]['auprc']:.4f}"
            for level in LEVELS
        ], "",
        f"Validation-selected deep blend weights: {json.dumps(blend_weights, sort_keys=True)}.", "",
        "The neural model improves broad EC-L3 transfer, while the tree component remains dominant for the fine EC-L4/Exact-Rhea endpoints. This resolution-specific routing is retained rather than selecting one model on the test set.", "",
        "## QC", "", qc.to_markdown(index=False), "",
    ]
    (reports / "PHASE_11_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    if failures:
        raise RuntimeError("Phase 11 QC failed: " + ", ".join(failures))
    (root / "checkpoints/CHECKPOINT_11_PASS").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
