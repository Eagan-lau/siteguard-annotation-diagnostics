#!/usr/bin/env python3
"""Fit TRAIN-only preprocessing and create role matrices without RETEST outcomes."""
import argparse
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")
KEY = ["query_protein_id", "reference_protein_id", "reference_activity_id"]
TARGETS = ["observed_same_ec_l3", "observed_same_ec_l4", "observed_same_exact_rhea"]


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def emit(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle: json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")


def load_features(path, numeric, categorical):
    return pd.read_parquet(path, columns=KEY + numeric + categorical)


def key_equal(left, right):
    return len(left) == len(right) and all(np.array_equal(left[name].to_numpy(), right[name].to_numpy()) for name in KEY)


def execute(root, feature_root, label_root, output):
    require(output.parent.is_dir() and not output.exists(), "matrix output")
    output.mkdir(exist_ok=False); emit(output / "reservation.json", {"status": "ONE_S4P_MATRIX_PREPARATION_RESERVED", "retest_truth_read": False, "automatic_retry": False})
    state, error = "FAIL_CLOSED", None
    try:
        train_summary = json.loads((feature_root / "role_TRAIN/producer/summary.json").read_text(encoding="utf-8"))
        numeric, categorical = train_summary["numeric_features"], train_summary["categorical_features"]
        require(train_summary["status"] == "PASS_S4N_ROLE_FEATURES" and categorical == ["reference_ec_l1", "reference_cofactor_class"], "TRAIN feature contract")
        train_path = feature_root / "role_TRAIN/producer/features_TRAIN.parquet"
        target_path = feature_root / "role_TRAIN/producer/targets_TRAIN.parquet"
        train = load_features(train_path, numeric, categorical)
        target = pd.read_parquet(target_path)
        require(key_equal(train, target), "TRAIN feature/target keys")
        train_numeric = train[numeric].apply(pd.to_numeric, errors="coerce").astype(np.float32)
        medians = train_numeric.median(axis=0).fillna(0.0)
        train_numeric = train_numeric.fillna(medians)
        means = train_numeric.mean(axis=0)
        scales = train_numeric.std(axis=0).replace(0, 1).fillna(1.0)
        train_numeric = ((train_numeric - means) / scales).astype(np.float32)
        categories = {}
        train_category_parts = []
        for name in categorical:
            values = train[name].fillna("MISSING").astype(str)
            levels = sorted(values.unique().tolist())
            categories[name] = levels
            train_category_parts.append(np.column_stack([(values == level).to_numpy(np.float32) for level in levels]))
        train_x = np.column_stack([train_numeric.to_numpy(np.float32), *train_category_parts]).astype(np.float32)
        require(np.isfinite(train_x).all(), "finite TRAIN matrix")
        np.save(output / "X_TRAIN.npy", train_x)
        y_train = np.column_stack([
            target["observed_same_ec_l3"].astype(bool).to_numpy(np.float32),
            target["observed_same_ec_l4"].astype(bool).to_numpy(np.float32),
            target["observed_same_exact_rhea"].fillna(False).astype(bool).to_numpy(np.float32),
        ])
        mask_train = np.column_stack([
            np.ones(len(target), dtype=np.float32), np.ones(len(target), dtype=np.float32),
            target["exact_rhea_outcome_evaluable"].astype(bool).to_numpy(np.float32),
        ])
        weights = target["sample_weight"].astype(np.float32).to_numpy(copy=True); require(np.isfinite(weights).all() and np.all(weights > 0), "TRAIN weights"); weights /= weights.mean()
        np.save(output / "y_TRAIN.npy", y_train); np.save(output / "mask_TRAIN.npy", mask_train); np.save(output / "weight_TRAIN.npy", weights)
        del train_x, train_numeric, train, target

        role_rows = {"TRAIN": int(len(y_train))}
        for role in ROLES[1:]:
            summary = json.loads((feature_root / f"role_{role}/producer/summary.json").read_text(encoding="utf-8"))
            require(summary["numeric_features"] == numeric and summary["categorical_features"] == categorical and not summary["nontrain_truth_read"], "role feature contract")
            frame = load_features(feature_root / f"role_{role}/producer/features_{role}.parquet", numeric, categorical)
            values = frame[numeric].apply(pd.to_numeric, errors="coerce").astype(np.float32).fillna(medians)
            values = ((values - means) / scales).astype(np.float32)
            category_parts = []
            for name in categorical:
                strings = frame[name].fillna("MISSING").astype(str)
                category_parts.append(np.column_stack([(strings == level).to_numpy(np.float32) for level in categories[name]]))
            matrix = np.column_stack([values.to_numpy(np.float32), *category_parts]).astype(np.float32)
            require(np.isfinite(matrix).all() and matrix.shape[1] > 0, "finite role matrix")
            np.save(output / f"X_{role}.npy", matrix); role_rows[role] = len(frame)
            if role == "DEV":
                label_path = label_root / "role_DEV/labels_DEV.parquet"
                labels = pd.read_parquet(label_path)
                require(key_equal(frame, labels), "DEV feature/label keys")
                y = np.column_stack([
                    labels["observed_same_ec_l3"].astype(bool).to_numpy(np.float32),
                    labels["observed_same_ec_l4"].astype(bool).to_numpy(np.float32),
                    labels["observed_same_exact_rhea"].fillna(False).astype(bool).to_numpy(np.float32),
                ])
                mask = np.column_stack([
                    np.ones(len(labels), dtype=np.float32), np.ones(len(labels), dtype=np.float32),
                    labels["exact_rhea_outcome_evaluable"].astype(bool).to_numpy(np.float32),
                ])
                np.save(output / "y_DEV.npy", y); np.save(output / "mask_DEV.npy", mask)
            del matrix, values, frame
        model_columns = numeric + [f"{name}={level}" for name in categorical for level in categories[name]]
        preprocessing = {
            "status": "FROZEN_S4P_TRAIN_ONLY_PREPROCESSING", "numeric_columns": numeric,
            "categorical_columns": categorical, "categorical_levels": categories,
            "model_columns": model_columns, "numeric_medians": {k: float(v) for k, v in medians.items()},
            "numeric_means": {k: float(v) for k, v in means.items()},
            "numeric_scales": {k: float(v) for k, v in scales.items()},
            "fit_role": "TRAIN", "dev_used_for_preprocessing": False,
            "calibration_roles_used_for_preprocessing": False, "retest_truth_read": False,
        }
        emit(output / "preprocessing.json", preprocessing)
        files = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)} for path in output.glob("*.npy")}
        summary = {"status": "PASS_S4P_MATRIX_PREPARATION", "role_rows": role_rows, "input_features": len(model_columns), "retest_truth_read": False, "files": files, "preprocessing_sha256": sha(output / "preprocessing.json")}
        emit(output / "summary.json", summary); emit(output / "S4P_MATRIX_PREPARATION_PASS.json", {"status": "PASS_S4P_MATRIX_PREPARATION", "summary_sha256": sha(output / "summary.json")}); state = summary["status"]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    emit(output / "terminal.json", {"status": state, "error": error, "retest_truth_read": False, "automatic_retry": False})
    if error: raise RuntimeError(error["message"])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--feature-root", type=Path, required=True); parser.add_argument("--label-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); execute(args.root.resolve(), args.feature_root.resolve(), args.label_root.resolve(), args.output.resolve())


if __name__ == "__main__": main()
