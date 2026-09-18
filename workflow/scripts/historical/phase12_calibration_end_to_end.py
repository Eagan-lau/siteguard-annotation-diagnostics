#!/usr/bin/env python3
"""Calibrate SiteGuard, derive abstention rules, and evaluate end-to-end transfer."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


LEVELS = ["EC_L3", "EC_L4", "EXACT_RHEA"]
TARGETS = {"EC_L3": "same_ec_l3", "EC_L4": "same_ec_l4", "EXACT_RHEA": "same_exact_rhea"}


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            value += mask.mean() * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return float(value)


def wilson_lower(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    if trials <= 0:
        return float("nan")
    p = successes / trials
    denominator = 1 + z * z / trials
    return (p + z * z / (2 * trials) - z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))) / denominator


def safe_threshold(frame: pd.DataFrame, probability: str, target: str, desired: float) -> dict:
    candidates = np.unique(np.quantile(frame[probability], np.linspace(0, 1, 1001)))
    selected = None
    for threshold in candidates:
        accepted = frame.loc[frame[probability].ge(threshold)]
        if len(accepted) < 100:
            continue
        successes = int(accepted[target].sum())
        lower = wilson_lower(successes, len(accepted))
        if lower >= desired:
            selected = {
                "threshold": float(threshold), "precision": successes / len(accepted),
                "ci_lower": lower, "accepted": len(accepted), "coverage": len(accepted) / len(frame),
            }
            break
    return selected or {
        "threshold": 1.000001, "precision": float("nan"), "ci_lower": float("nan"),
        "accepted": 0, "coverage": 0.0,
    }


def top_candidates(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    score = f"calibrated_{level}"
    indices = frame.groupby("query_protein_id", sort=False)[score].idxmax()
    columns = [
        "query_protein_id", "reference_protein_id", "reference_activity_id", score, TARGETS[level],
    ]
    output = frame.loc[indices, columns].copy()
    return output.rename(columns={
        "reference_protein_id": f"top_reference_protein_{level}",
        "reference_activity_id": f"top_reference_activity_{level}",
        score: f"top_probability_{level}", TARGETS[level]: f"top_correct_{level}",
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    phase11 = root / "results/phase11"
    results = root / "results/phase12"
    figures = root / "figures/source_data"
    models = root / "models/phase12"
    reports = root / "reports"
    results.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(phase11 / "pairwise_predictions.parquet")
    validation = frame["query_split_expected"].eq("validation")
    test = frame["query_split_expected"].eq("test")
    calibration_rows: list[dict] = []
    calibrators: dict[str, IsotonicRegression] = {}
    for level in LEVELS:
        target = TARGETS[level]
        raw = f"score_siteguard_{level}"
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(
            frame.loc[validation, raw].to_numpy(float), frame.loc[validation, target].astype(int).to_numpy(),
            sample_weight=frame.loc[validation, "sample_weight"].to_numpy(float),
        )
        calibrated = f"calibrated_{level}"
        frame[calibrated] = calibrator.predict(frame[raw].to_numpy(float)).astype(np.float32)
        calibrators[level] = calibrator
        joblib.dump(calibrator, models / f"isotonic_{level}.joblib", compress=3)
        for split_name, mask in [("validation", validation), ("test", test)]:
            y = frame.loc[mask, target].astype(int).to_numpy()
            for state, column in [("raw", raw), ("isotonic", calibrated)]:
                probability = np.clip(frame.loc[mask, column].to_numpy(float), 1e-7, 1 - 1e-7)
                calibration_rows.append({
                    "annotation_level": level, "calibration": state,
                    "evaluation_split": f"population_{split_name}", "rows": len(y),
                    "brier_score": float(brier_score_loss(y, probability)),
                    "log_loss": float(log_loss(y, probability)),
                    "ece_15": expected_calibration_error(y, probability, 15),
                    "calibrator_fit_split": "population_validation", "test_used_for_calibration": False,
                })
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(results / "calibration_metrics.tsv", sep="\t", index=False)

    validation_top = {level: top_candidates(frame.loc[validation], level) for level in LEVELS}
    test_top = {level: top_candidates(frame.loc[test], level) for level in LEVELS}
    threshold_records: list[dict] = []
    thresholds: dict[str, float] = {}
    for level in LEVELS:
        probability, target = f"top_probability_{level}", f"top_correct_{level}"
        for desired in [0.90, 0.95]:
            selected = safe_threshold(validation_top[level], probability, target, desired)
            threshold_records.append({
                "annotation_level": level, "target_precision": desired,
                "selection_split": "population_validation_top_candidate", "test_used_for_selection": False,
                "status": "PASS" if selected["accepted"] else "NO_SAFE_THRESHOLD",
                **selected,
            })
            if desired == 0.90:
                thresholds[level] = selected["threshold"]
    threshold_frame = pd.DataFrame(threshold_records)
    threshold_frame.to_csv(results / "abstention_thresholds.tsv", sep="\t", index=False)

    end_to_end = test_top["EC_L3"]
    for level in ["EC_L4", "EXACT_RHEA"]:
        end_to_end = end_to_end.merge(test_top[level], on="query_protein_id", validate="one_to_one")
    for level in LEVELS:
        query_has = frame.loc[test].groupby("query_protein_id")[TARGETS[level]].any()
        end_to_end[f"oracle_candidate_available_{level}"] = end_to_end["query_protein_id"].map(query_has).fillna(False)
        end_to_end[f"accepted_{level}"] = end_to_end[f"top_probability_{level}"].ge(thresholds[level])
    final_resolution: list[str] = []
    final_reference: list[str | None] = []
    final_activity: list[str | None] = []
    final_probability: list[float] = []
    final_correct: list[bool] = []
    for row in end_to_end.itertuples(index=False):
        if row.accepted_EXACT_RHEA and row.accepted_EC_L3:
            level = "EXACT_RHEA"
        elif row.accepted_EC_L4 and row.accepted_EC_L3:
            level = "EC_L4"
        elif row.accepted_EC_L3:
            level = "EC_L3"
        else:
            level = "ABSTAIN"
        final_resolution.append(level)
        if level == "ABSTAIN":
            final_reference.append(None); final_activity.append(None); final_probability.append(float("nan")); final_correct.append(False)
        else:
            final_reference.append(getattr(row, f"top_reference_protein_{level}"))
            final_activity.append(getattr(row, f"top_reference_activity_{level}"))
            final_probability.append(float(getattr(row, f"top_probability_{level}")))
            final_correct.append(bool(getattr(row, f"top_correct_{level}")))
    end_to_end["final_resolution"] = final_resolution
    end_to_end["final_reference_protein_id"] = final_reference
    end_to_end["final_reference_activity_id"] = final_activity
    end_to_end["final_probability"] = final_probability
    end_to_end["final_correct"] = final_correct
    end_to_end.to_parquet(results / "end_to_end_predictions.parquet", index=False, compression="zstd")

    metric_rows: list[dict] = []
    for level in LEVELS:
        accepted = end_to_end[f"accepted_{level}"]
        metric_rows.append({
            "analysis": "per_level_top_candidate", "annotation_level": level,
            "queries": len(end_to_end), "accepted_queries": int(accepted.sum()),
            "coverage": float(accepted.mean()),
            "precision_among_accepted": float(end_to_end.loc[accepted, f"top_correct_{level}"].mean()) if accepted.any() else float("nan"),
            "oracle_candidate_coverage": float(end_to_end[f"oracle_candidate_available_{level}"].mean()),
            "threshold": thresholds[level], "threshold_selection_split": "population_validation",
        })
    accepted_final = end_to_end["final_resolution"].ne("ABSTAIN")
    metric_rows.append({
        "analysis": "hierarchical_final", "annotation_level": "HIGHEST_SAFE_RESOLUTION",
        "queries": len(end_to_end), "accepted_queries": int(accepted_final.sum()),
        "coverage": float(accepted_final.mean()),
        "precision_among_accepted": float(end_to_end.loc[accepted_final, "final_correct"].mean()) if accepted_final.any() else float("nan"),
        "oracle_candidate_coverage": float("nan"), "threshold": float("nan"),
        "threshold_selection_split": "population_validation",
    })
    end_metrics = pd.DataFrame(metric_rows)
    end_metrics.to_csv(results / "end_to_end_metrics.tsv", sep="\t", index=False)

    risk_rows: list[dict] = []
    for level in LEVELS:
        top = test_top[level].sort_values(f"top_probability_{level}", ascending=False).reset_index(drop=True)
        for coverage in np.linspace(0.01, 1.0, 100):
            count = max(1, int(round(coverage * len(top))))
            selected = top.iloc[:count]
            precision = float(selected[f"top_correct_{level}"].mean())
            risk_rows.append({
                "annotation_level": level, "coverage": count / len(top), "accepted_queries": count,
                "precision": precision, "risk": 1 - precision,
                "minimum_probability": float(selected[f"top_probability_{level}"].iloc[-1]),
                "evaluation_split": "population_test", "selection_rule": "ranked confidence; descriptive test curve",
            })
    risk = pd.DataFrame(risk_rows)
    risk.to_csv(results / "risk_coverage.tsv", sep="\t", index=False)

    error_rows: list[dict] = []
    for level in LEVELS:
        oracle = end_to_end[f"oracle_candidate_available_{level}"]
        correct = end_to_end[f"top_correct_{level}"].astype(bool)
        accepted = end_to_end[f"accepted_{level}"].astype(bool)
        categories = {
            "NO_POSITIVE_CANDIDATE_RETRIEVAL_LIMIT": ~oracle,
            "POSITIVE_PRESENT_BUT_TOP1_WRONG_RANKING": oracle & ~correct,
            "TOP1_CORRECT_BUT_ABSTAINED": correct & ~accepted,
            "ACCEPTED_WRONG_OVERANNOTATION": accepted & ~correct,
            "ACCEPTED_CORRECT": accepted & correct,
        }
        for category, mask in categories.items():
            error_rows.append({
                "annotation_level": level, "error_component": category,
                "queries": int(mask.sum()), "fraction_of_queries": float(mask.mean()),
                "evaluation_split": "population_test",
            })
    error = pd.DataFrame(error_rows)
    error.to_csv(results / "error_decomposition.tsv", sep="\t", index=False)

    calibration.to_csv(figures / "Figure4_calibration.tsv", sep="\t", index=False)
    risk.to_csv(figures / "Figure4_risk_coverage.tsv", sep="\t", index=False)
    end_metrics.to_csv(figures / "Figure4_end_to_end_metrics.tsv", sep="\t", index=False)
    error.to_csv(figures / "Figure4_error_decomposition.tsv", sep="\t", index=False)
    config = {
        "calibration": "per-level isotonic regression", "fit_split": "population_validation",
        "top_candidate_precision_target": 0.90, "thresholds": thresholds,
        "hierarchy": "Exact Rhea or EC-L4 requires EC-L3 acceptance; otherwise EC-L3 or abstain",
        "test_used_for_calibration_or_threshold_selection": False,
    }
    (models / "calibration_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    calibrated_test = calibration.loc[(calibration["evaluation_split"] == "population_test") & (calibration["calibration"] == "isotonic")]
    checks = [
        ("checkpoint_11_present", (root / "checkpoints/CHECKPOINT_11_PASS").is_file(), "strict phase gate"),
        ("calibration_metrics_complete", len(calibration) == 12, str(len(calibration))),
        ("calibrated_probabilities_valid", frame[[f"calibrated_{level}" for level in LEVELS]].apply(lambda x: x.between(0, 1).all()).all(), "[0,1]"),
        ("test_calibration_metrics_finite", calibrated_test[["brier_score", "log_loss", "ece_15"]].notna().all().all(), str(len(calibrated_test))),
        ("validation_only_calibration", not bool(calibration["test_used_for_calibration"].any()) and not bool(threshold_frame["test_used_for_selection"].any()), "test untouched"),
        ("end_to_end_query_unique", end_to_end["query_protein_id"].is_unique, str(len(end_to_end))),
        ("end_to_end_test_query_count", len(end_to_end) == 27_639, str(len(end_to_end))),
        ("risk_coverage_complete", len(risk) == 300, str(len(risk))),
        ("error_decomposition_complete", len(error) == 15, str(len(error))),
        ("figure4_source_data_present", all((figures / name).stat().st_size > 0 for name in ["Figure4_calibration.tsv", "Figure4_risk_coverage.tsv", "Figure4_end_to_end_metrics.tsv", "Figure4_error_decomposition.tsv"]), "4 tables"),
    ]
    qc = pd.DataFrame(
        [(name, "PASS" if bool(passed) else "FAIL", details) for name, passed, details in checks],
        columns=["check", "status", "details"],
    )
    qc.to_csv(reports / "phase12_qc.tsv", sep="\t", index=False)
    failures = qc.loc[qc["status"].eq("FAIL"), "check"].tolist()
    final_row = end_metrics.loc[end_metrics["analysis"].eq("hierarchical_final")].iloc[0]
    summary = {
        "phase": 12, "status": "PASS" if not failures else "FAIL",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "test_queries": len(end_to_end), "safe_thresholds": thresholds,
        "hierarchical_coverage": float(final_row["coverage"]),
        "hierarchical_precision": float(final_row["precision_among_accepted"]),
        "final_resolution_counts": end_to_end["final_resolution"].value_counts().to_dict(),
        "test_used_for_selection": False, "qc_failures": failures,
    }
    (reports / "phase12_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# SiteGuard V4 Phase 12 Report", "", f"Status: **{summary['status']}**", "",
        f"- Test queries: {len(end_to_end):,}",
        f"- Highest-safe-resolution coverage: {summary['hierarchical_coverage']:.2%}",
        f"- Precision among annotated queries: {summary['hierarchical_precision']:.2%}",
        f"- Final resolution counts: {json.dumps(summary['final_resolution_counts'], sort_keys=True)}", "",
        "Calibration and abstention thresholds were fitted on population validation only. Test data were used only for final metrics and descriptive risk–coverage curves.", "",
        "## QC", "", qc.to_markdown(index=False), "",
    ]
    (reports / "PHASE_12_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    if failures:
        raise RuntimeError("Phase 12 QC failed: " + ", ".join(failures))
    (root / "checkpoints/CHECKPOINT_12_PASS").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
