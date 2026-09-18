#!/usr/bin/env python3
"""Train and evaluate compatible homology and classical-ML baselines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260820
ROW_KEY = ["pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id"]
TARGETS = {"EC_L3": "same_ec_l3", "EC_L4": "same_ec_l4", "EXACT_RHEA": "same_exact_rhea"}
RAW_BASELINES = {
    "sequence_identity": "sequence_identity",
    "esm2_t33_cosine": "esm2_t33_cosine",
    "foldseek_identity": "foldseek_identity",
    "pfam_jaccard": "pfam_jaccard",
    "cath_jaccard": "cath_jaccard",
}
FORBIDDEN = {
    "query_ec_l3_ground_truth", "query_ec_l4_ground_truth", "query_rhea_ground_truth",
    "same_ec_l3", "same_ec_l4", "same_exact_rhea", "observed_concordance_depth",
    "is_difficult_case", "candidate_origin", "augmentation_source", "sampling_probability",
    "sample_weight",
}


def optimal_f1_threshold(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    candidates = np.unique(np.quantile(score, np.linspace(0, 1, 201)))
    best_threshold, best_value = 0.5, -1.0
    for threshold in candidates:
        prediction = score >= threshold
        value = f1_score(y, prediction, sample_weight=weight, zero_division=0)
        if value > best_value:
            best_threshold, best_value = float(threshold), float(value)
    return best_threshold


def metrics(y: np.ndarray, score: np.ndarray, threshold: float, weight: np.ndarray) -> dict[str, float]:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    processed = root / "data/processed"
    results = root / "results/phase10"
    reports = root / "reports"
    models = root / "models/phase10"
    results.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    feature_file = processed / "global_features.parquet"
    feature_columns = pq.ParquetFile(feature_file).schema_arrow.names
    input_columns = [name for name in feature_columns if name not in ROW_KEY + ["query_split"]]
    forbidden_found = sorted(set(input_columns) & FORBIDDEN | {name for name in input_columns if "ground_truth" in name.lower()})
    if forbidden_found:
        raise RuntimeError(f"Forbidden training inputs: {forbidden_found}")
    feature_frame = pd.read_parquet(feature_file, columns=ROW_KEY + input_columns)

    label_columns = [
        "query_protein_id", "reference_protein_id", "reference_activity_id", "query_split_expected",
        "query_cluster_id_30", "same_ec_l3", "same_ec_l4", "same_exact_rhea", "sample_weight",
    ]
    training_labels = pd.read_parquet(processed / "training_pairs.parquet", columns=label_columns).assign(pair_set="training")
    population_labels = pd.read_parquet(processed / "population_pairs.parquet", columns=label_columns).assign(pair_set="population")
    training = training_labels.merge(feature_frame, on=ROW_KEY, validate="one_to_one")
    evaluation = population_labels.loc[population_labels["query_split_expected"].isin(["validation", "test"])].merge(
        feature_frame, on=ROW_KEY, validate="one_to_one"
    )
    training = training.loc[training["query_split_expected"].eq("train")].reset_index(drop=True)
    evaluation = evaluation.reset_index(drop=True)
    if len(training) < 500_000 or len(evaluation) < 500_000:
        raise RuntimeError(f"Unexpected Phase 10 cohorts: train={len(training)}, evaluation={len(evaluation)}")

    categorical = [name for name in ["reference_ec_l1", "reference_cofactor_class"] if name in input_columns]
    numeric = [name for name in input_columns if name not in categorical]
    train_categories = pd.get_dummies(training[categorical].fillna("MISSING").astype(str), prefix=categorical, dtype=np.float32)
    eval_categories = pd.get_dummies(evaluation[categorical].fillna("MISSING").astype(str), prefix=categorical, dtype=np.float32)
    eval_categories = eval_categories.reindex(columns=train_categories.columns, fill_value=0.0)
    train_numeric = training[numeric].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    eval_numeric = evaluation[numeric].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    model_columns = numeric + train_categories.columns.tolist()
    x_train = pd.concat([train_numeric.reset_index(drop=True), train_categories.reset_index(drop=True)], axis=1)
    x_eval = pd.concat([eval_numeric.reset_index(drop=True), eval_categories.reset_index(drop=True)], axis=1)

    predictions = evaluation[ROW_KEY + ["query_split_expected", "query_cluster_id_30", *TARGETS.values(), "sample_weight"]].copy()
    for baseline, column in RAW_BASELINES.items():
        predictions[f"score_{baseline}"] = pd.to_numeric(evaluation[column], errors="coerce").fillna(0.0).clip(0, 1).astype(np.float32)

    normalized_weight = training["sample_weight"].to_numpy(float).copy()
    normalized_weight /= normalized_weight.mean()
    trained_models: dict[str, dict[str, object]] = {}
    for level, target in TARGETS.items():
        y_train = training[target].astype(int).to_numpy()
        logistic = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=300, random_state=SEED, solver="lbfgs")),
        ])
        logistic.fit(x_train, y_train, model__sample_weight=normalized_weight)
        predictions[f"score_logistic_global_{level}"] = logistic.predict_proba(x_eval)[:, 1].astype(np.float32)
        joblib.dump({"model": logistic, "feature_columns": model_columns, "target": target}, models / f"logistic_global_{level}.joblib", compress=3)

        lightgbm = lgb.LGBMClassifier(
            objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
            max_depth=-1, min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=SEED, n_jobs=8, verbosity=-1,
        )
        lightgbm.fit(x_train, y_train, sample_weight=normalized_weight)
        predictions[f"score_lightgbm_global_{level}"] = lightgbm.predict_proba(x_eval)[:, 1].astype(np.float32)
        lightgbm.booster_.save_model(str(models / f"lightgbm_global_{level}.txt"))
        trained_models[level] = {"logistic": logistic, "lightgbm": lightgbm}

    baseline_names = list(RAW_BASELINES) + ["logistic_global", "lightgbm_global"]
    metric_rows: list[dict] = []
    threshold_rows: list[dict] = []
    validation_mask = predictions["query_split_expected"].eq("validation").to_numpy()
    for level, target in TARGETS.items():
        for baseline in baseline_names:
            score_column = f"score_{baseline}" if baseline in RAW_BASELINES else f"score_{baseline}_{level}"
            y_validation = predictions.loc[validation_mask, target].astype(int).to_numpy()
            score_validation = predictions.loc[validation_mask, score_column].to_numpy(float)
            weight_validation = predictions.loc[validation_mask, "sample_weight"].to_numpy(float)
            threshold = optimal_f1_threshold(y_validation, score_validation, weight_validation)
            threshold_rows.append({
                "baseline": baseline, "annotation_level": level, "selected_threshold": threshold,
                "selection_split": "population_validation", "test_used_for_selection": False,
            })
            for split in ["validation", "test"]:
                mask = predictions["query_split_expected"].eq(split).to_numpy()
                y = predictions.loc[mask, target].astype(int).to_numpy()
                score = predictions.loc[mask, score_column].to_numpy(float)
                weight = predictions.loc[mask, "sample_weight"].to_numpy(float)
                values = metrics(y, score, threshold, weight)
                metric_rows.append({
                    "baseline": baseline, "annotation_level": level, "evaluation_split": f"population_{split}",
                    "rows": int(mask.sum()), "positives": int(y.sum()), "threshold": threshold,
                    "threshold_selection_split": "population_validation", "test_used_for_threshold_selection": False,
                    **values,
                })

    prediction_path = results / "baseline_predictions.parquet"
    predictions.to_parquet(prediction_path, index=False, compression="zstd")
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(results / "baseline_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(threshold_rows).to_csv(results / "baseline_thresholds.tsv", sep="\t", index=False)
    config = {
        "seed": SEED, "training_rows": len(training), "evaluation_rows": len(evaluation),
        "feature_columns": model_columns, "forbidden_fields": forbidden_found,
        "sklearn_version": sklearn.__version__, "lightgbm_version": lgb.__version__,
        "training_design": "registered balanced training pairs, train queries only, normalized inverse sampling weights",
        "evaluation_design": "population atlas validation/test",
    }
    (models / "baseline_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    compatibility = [
        "# SiteGuard V4 Baseline Compatibility Report", "",
        "| Method | Task compatibility | Included | Implementation |",
        "|---|---|---:|---|",
        "| MMseqs2/identity transfer | Directly scores reference-conditioned transfer pairs | Yes | Frozen direct MMseqs2 sequence identity |",
        "| ESM2 global similarity | Directly scores the same protein pair | Yes | Frozen ESM2-t33 cosine |",
        "| Foldseek transfer | Directly scores pairs with structures; absence retained as no structural evidence | Yes | Frozen direct Foldseek identity |",
        "| Pfam family transfer | Pairwise domain-family compatibility | Yes | Pfam Jaccard |",
        "| CATH transfer | Pairwise structural-domain compatibility | Yes | CATH Jaccard |",
        "| Logistic regression | Classical supervised reference-conditioned baseline | Yes | scikit-learn global features |",
        "| LightGBM | Strong nonlinear classical baseline on identical global features | Yes | LightGBM global features |",
        "| Standalone query-only EC classifiers | Predict query EC, not whether a named reference activity transfers | No | Not numerically compared across incompatible tasks |",
        "| Reaction-retrieval-only methods | Retrieve reactions but do not score reference-to-query transferability | No | Reserved for end-to-end compatible evaluation |",
        "| Optional GNN local model | Phase 9 fine-resolution Go/No-Go was LOCAL_NULL | No | Not promoted as a baseline claim; may remain exploratory |", "",
        "All included methods use the same frozen population validation/test rows. Thresholds are selected on validation only.", "",
    ]
    (results / "baseline_compatibility_report.md").write_text("\n".join(compatibility), encoding="utf-8")

    test_metrics = metric_frame.loc[metric_frame["evaluation_split"].eq("population_test")]
    probability_columns = [name for name in predictions if name.startswith("score_")]
    checks = [
        ("checkpoint_09_present", (root / "checkpoints/CHECKPOINT_09_LOCAL_NULL").is_file() or (root / "checkpoints/CHECKPOINT_09_PASS").is_file(), "strict phase gate"),
        ("forbidden_model_inputs_absent", not forbidden_found, str(forbidden_found)),
        ("registered_training_cohort", len(training) == 931_006, str(len(training))),
        ("population_evaluation_rows", len(evaluation) == 630_469, str(len(evaluation))),
        ("prediction_probabilities_valid", predictions[probability_columns].apply(lambda values: values.between(0, 1).all()).all(), str(len(probability_columns))),
        ("metrics_complete", len(metric_frame) == len(baseline_names) * len(TARGETS) * 2, str(len(metric_frame))),
        ("metrics_finite", test_metrics[["auprc", "auroc", "precision", "recall", "f1", "mcc", "balanced_accuracy"]].notna().all().all(), "population test"),
        ("test_not_used_for_thresholds", not bool(metric_frame["test_used_for_threshold_selection"].any()), "validation only"),
        ("compatibility_report_present", (results / "baseline_compatibility_report.md").stat().st_size > 0, "task comparability documented"),
        ("split_leakage_report_pass", pd.read_csv(root / "data/splits/split_leakage_report.tsv", sep="\t")["status"].eq("PASS").all(), "registered split"),
    ]
    qc = pd.DataFrame(
        [(name, "PASS" if bool(passed) else "FAIL", details) for name, passed, details in checks],
        columns=["check", "status", "details"],
    )
    qc.to_csv(reports / "phase10_qc.tsv", sep="\t", index=False)
    failures = qc.loc[qc["status"].eq("FAIL"), "check"].tolist()
    summary = {
        "phase": 10, "status": "PASS" if not failures else "FAIL",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "training_rows": len(training), "evaluation_rows": len(evaluation),
        "baselines": baseline_names, "metric_rows": len(metric_frame),
        "model_features": len(model_columns), "forbidden_fields": forbidden_found,
        "best_test_auprc": {
            level: test_metrics.loc[test_metrics["annotation_level"].eq(level)].sort_values("auprc", ascending=False).iloc[0][["baseline", "auprc"]].to_dict()
            for level in TARGETS
        },
        "qc_failures": failures,
    }
    (reports / "phase10_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# SiteGuard V4 Phase 10 Report", "", f"Status: **{summary['status']}**", "",
        f"- Training rows: {len(training):,}", f"- Population validation/test rows: {len(evaluation):,}",
        f"- Compatible baselines: {len(baseline_names)}", f"- Model features: {len(model_columns)}", "",
        "## Best held-out AUPRC", "",
        *[f"- {level}: {values['baseline']} = {values['auprc']:.4f}" for level, values in summary["best_test_auprc"].items()], "",
        "## QC", "", qc.to_markdown(index=False), "",
    ]
    (reports / "PHASE_10_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    if failures:
        raise RuntimeError("Phase 10 QC failed: " + ", ".join(failures))
    (root / "checkpoints/CHECKPOINT_10_PASS").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
