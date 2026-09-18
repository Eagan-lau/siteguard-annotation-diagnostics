#!/usr/bin/env python3
"""Fit frozen Logistic and LightGBM EvidenceJudge candidates on FIT only."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4")).resolve()
RESULTS = ROOT / "results/phase24"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models/phase24/evidencejudge"
CHECKPOINTS = ROOT / "checkpoints"
SEED = 20260819


def main() -> None:
    if not (CHECKPOINTS / "CHECKPOINT_24D0_MATRIX_PASS").is_file():
        raise RuntimeError("CHECKPOINT_24D0_MATRIX_PASS is required")
    MODELS.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_parquet(RESULTS / "evidencejudge_candidate_matrix.parquet")
    schema = json.loads((RESULTS / "evidencejudge_feature_schema.json").read_text(encoding="utf-8"))
    features = schema["feature_columns"]
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    training_rows: list[dict[str, object]] = []

    for level in schema["levels"]:
        fit = matrix.loc[(matrix["annotation_level"] == level) & (matrix["development_partition"] == "FIT")]
        score = matrix.loc[(matrix["annotation_level"] == level) & matrix["development_partition"].isin(["CAL", "SELECT", "TEST"])]
        x_fit = fit[features]
        y_fit = fit["correct"].astype(int).to_numpy()
        weights = fit["query_weight"].to_numpy(dtype=float)
        if y_fit.min() == y_fit.max():
            raise RuntimeError(f"Degenerate FIT target for {level}")

        models = {
            "LOGISTIC_STACK": Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED)),
            ]),
            "LIGHTGBM_ROUTER": LGBMClassifier(
                objective="binary",
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=-1,
                min_child_samples=100,
                subsample=0.80,
                colsample_bytree=0.80,
                reg_alpha=0.10,
                reg_lambda=1.0,
                random_state=SEED,
                n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
                verbosity=-1,
            ),
        }
        for model_name, model in models.items():
            if model_name == "LOGISTIC_STACK":
                model.fit(x_fit, y_fit, classifier__sample_weight=weights)
            else:
                model.fit(x_fit, y_fit, sample_weight=weights)
            probabilities = model.predict_proba(score[features])[:, 1]
            exported = score[[
                "candidate_row_id", "pair_set", "development_partition", "annotation_level",
                "query_protein_id", "query_cluster_id_30", "candidate_label", "correct", "query_weight",
            ]].copy()
            exported["model"] = model_name
            exported["raw_probability"] = probabilities.astype(np.float32)
            prediction_frames.append(exported)
            model_path = MODELS / f"{level.lower()}__{model_name.lower()}.joblib"
            joblib.dump(model, model_path, compress=3)

            if model_name == "LIGHTGBM_ROUTER":
                values = model.feature_importances_.astype(float)
            else:
                classifier = model.named_steps["classifier"]
                imputer = model.named_steps["imputer"]
                names = list(imputer.get_feature_names_out(features))
                values = np.abs(classifier.coef_[0])
                importance_frames.append(pd.DataFrame({
                    "annotation_level": level,
                    "model": model_name,
                    "feature": names,
                    "importance": values,
                    "importance_type": "absolute_standardized_coefficient",
                }))
                values = None
            if values is not None:
                importance_frames.append(pd.DataFrame({
                    "annotation_level": level,
                    "model": model_name,
                    "feature": features,
                    "importance": values,
                    "importance_type": "split_count",
                }))
            training_rows.append({
                "annotation_level": level,
                "model": model_name,
                "fit_candidate_rows": len(fit),
                "fit_queries": fit["query_protein_id"].nunique(),
                "fit_clusters": fit["query_cluster_id_30"].nunique(),
                "fit_positive_rate_unweighted": float(y_fit.mean()),
                "fit_positive_rate_query_weighted": float(np.average(y_fit, weights=weights)),
                "scored_candidate_rows": len(score),
                "seed": SEED,
                "model_path": str(model_path),
            })

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_parquet(RESULTS / "evidencejudge_raw_predictions_classical.parquet", index=False, compression="zstd")
    importance = pd.concat(importance_frames, ignore_index=True)
    importance.to_csv(RESULTS / "evidencejudge_feature_importance_classical.tsv", sep="\t", index=False)
    training = pd.DataFrame(training_rows)
    training.to_csv(RESULTS / "evidencejudge_classical_training_inventory.tsv", sep="\t", index=False)

    checks = [
        ("six_level_model_pairs", len(training) == 6, len(training)),
        ("fit_only_used_for_training", set(matrix.loc[matrix["development_partition"].eq("FIT"), "pair_set"]) == {"validation"}, sorted(matrix.loc[matrix["development_partition"].eq("FIT"), "pair_set"].unique())),
        ("no_fit_rows_exported_as_predictions", "FIT" not in set(predictions["development_partition"]), sorted(predictions["development_partition"].unique())),
        ("probabilities_finite", np.isfinite(predictions["raw_probability"]).all(), len(predictions)),
        ("probabilities_in_unit_interval", predictions["raw_probability"].between(0, 1).all(), [float(predictions["raw_probability"].min()), float(predictions["raw_probability"].max())]),
        ("prediction_grain_unique", not predictions.duplicated(["candidate_row_id", "model"]).any(), len(predictions)),
    ]
    qc = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    qc.to_csv(REPORTS / "phase24_evidencejudge_classical_qc.tsv", sep="\t", index=False)
    failures = qc.loc[~qc["passed"].astype(bool), "check"].tolist()
    summary = {
        "phase": "24D1",
        "stage": "classical_evidencejudge_models",
        "status": "PASS" if not failures else "FAIL",
        "models": ["LOGISTIC_STACK", "LIGHTGBM_ROUTER"],
        "levels": schema["levels"],
        "training_records": len(training),
        "prediction_rows": len(predictions),
        "failures": failures,
    }
    (REPORTS / "phase24_evidencejudge_classical_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"Classical EvidenceJudge QC failed: {failures}")
    (CHECKPOINTS / "CHECKPOINT_24D1_CLASSICAL_PASS").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
