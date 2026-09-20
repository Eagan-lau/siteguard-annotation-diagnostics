#!/usr/bin/env python3
"""Prepare frozen, leakage-audited matrices for the Phase 11 deep model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROW_KEY = ["pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id"]
TARGETS = ["same_ec_l3", "same_ec_l4", "same_exact_rhea"]
FORBIDDEN = {
    "query_ec_l3_ground_truth", "query_ec_l4_ground_truth", "query_rhea_ground_truth",
    "same_ec_l3", "same_ec_l4", "same_exact_rhea", "observed_concordance_depth",
    "is_difficult_case", "candidate_origin", "augmentation_source", "sampling_probability",
    "sample_weight", "mapping_success",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    processed = root / "data/processed"
    work = root / "data/interim/phase11"
    reports = root / "reports"
    work.mkdir(parents=True, exist_ok=True)

    feature_path = processed / "global_features.parquet"
    all_columns = pq.ParquetFile(feature_path).schema_arrow.names
    input_columns = [name for name in all_columns if name not in ROW_KEY + ["query_split"]]
    forbidden = sorted(set(input_columns) & FORBIDDEN | {name for name in input_columns if "ground_truth" in name.lower()})
    if forbidden:
        raise RuntimeError(f"Forbidden deep-model inputs: {forbidden}")
    features = pd.read_parquet(feature_path, columns=ROW_KEY + input_columns)
    label_columns = [
        "query_protein_id", "reference_protein_id", "reference_activity_id", "query_split_expected",
        "query_cluster_id_30", *TARGETS, "sample_weight",
    ]
    training_labels = pd.read_parquet(processed / "training_pairs.parquet", columns=label_columns).assign(pair_set="training")
    population_labels = pd.read_parquet(processed / "population_pairs.parquet", columns=label_columns).assign(pair_set="population")
    train = training_labels.loc[training_labels["query_split_expected"].eq("train")].merge(features, on=ROW_KEY, validate="one_to_one").reset_index(drop=True)
    evaluation = population_labels.loc[population_labels["query_split_expected"].isin(["validation", "test"])].merge(features, on=ROW_KEY, validate="one_to_one").reset_index(drop=True)
    categorical = [name for name in ["reference_ec_l1", "reference_cofactor_class"] if name in input_columns]
    numeric = [name for name in input_columns if name not in categorical]
    train_category = pd.get_dummies(train[categorical].fillna("MISSING").astype(str), prefix=categorical, dtype=np.float32)
    eval_category = pd.get_dummies(evaluation[categorical].fillna("MISSING").astype(str), prefix=categorical, dtype=np.float32)
    eval_category = eval_category.reindex(columns=train_category.columns, fill_value=0.0)
    train_numeric = train[numeric].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    eval_numeric = evaluation[numeric].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    medians = train_numeric.median(axis=0).fillna(0.0)
    train_numeric = train_numeric.fillna(medians)
    eval_numeric = eval_numeric.fillna(medians)
    means = train_numeric.mean(axis=0)
    scales = train_numeric.std(axis=0).replace(0, 1).fillna(1.0)
    train_numeric = (train_numeric - means) / scales
    eval_numeric = (eval_numeric - means) / scales
    model_columns = numeric + train_category.columns.tolist()
    train_x = np.column_stack([train_numeric.to_numpy(np.float32), train_category.to_numpy(np.float32)])
    eval_x = np.column_stack([eval_numeric.to_numpy(np.float32), eval_category.to_numpy(np.float32)])
    train_y = train[TARGETS].to_numpy(np.float32)
    eval_y = evaluation[TARGETS].to_numpy(np.float32)
    train_weight = train["sample_weight"].to_numpy(np.float32).copy()
    train_weight /= train_weight.mean()
    eval_weight = evaluation["sample_weight"].to_numpy(np.float32)
    eval_split = evaluation["query_split_expected"].map({"validation": 0, "test": 1}).to_numpy(np.int8)

    np.save(work / "train_X.npy", train_x)
    np.save(work / "train_y.npy", train_y)
    np.save(work / "train_weight.npy", train_weight)
    np.save(work / "eval_X.npy", eval_x)
    np.save(work / "eval_y.npy", eval_y)
    np.save(work / "eval_weight.npy", eval_weight)
    np.save(work / "eval_split.npy", eval_split)
    evaluation[ROW_KEY + ["query_split_expected", "query_cluster_id_30", *TARGETS, "sample_weight"]].to_parquet(
        work / "evaluation_metadata.parquet", index=False, compression="zstd"
    )
    preprocessing = {
        "model_columns": model_columns, "numeric_columns": numeric, "categorical_columns": categorical,
        "categorical_dummy_columns": train_category.columns.tolist(),
        "numeric_medians": {key: float(value) for key, value in medians.items()},
        "numeric_means": {key: float(value) for key, value in means.items()},
        "numeric_scales": {key: float(value) for key, value in scales.items()},
        "forbidden_fields": forbidden,
    }
    (work / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2) + "\n", encoding="utf-8")
    summary = {
        "phase": 11, "stage": "deep_matrix_preparation", "status": "PASS",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "training_rows": len(train), "evaluation_rows": len(evaluation),
        "validation_rows": int((eval_split == 0).sum()), "test_rows": int((eval_split == 1).sum()),
        "input_features": train_x.shape[1], "targets": TARGETS,
        "forbidden_fields": forbidden, "local_features_in_primary_model": False,
        "phase09_decision": "CHECKPOINT_09_LOCAL_NULL",
    }
    (reports / "phase11_prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
