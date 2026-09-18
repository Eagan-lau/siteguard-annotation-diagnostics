#!/usr/bin/env python3
"""Export frozen numeric arrays for the cluster PyTorch module (which lacks PyArrow)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4")).resolve()
RESULTS = ROOT / "results/phase24"
REPORTS = ROOT / "reports"
CHECKPOINTS = ROOT / "checkpoints"
PARTITION_CODE = {"FIT": 0, "CAL": 1, "SELECT": 2, "TEST": 3}


def inner_holdout(cluster_id: object) -> bool:
    digest = hashlib.sha256(f"20260819|DEEPSETS_INNER|{cluster_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10 == 0


def make_arrays(
    frame: pd.DataFrame,
    methods: list[str],
    global_features: list[str],
    stats: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_tool: list[np.ndarray] = []
    availability: list[np.ndarray] = []
    for method in methods:
        support = frame[f"support__{method}"].to_numpy(dtype=np.float32)
        query_score_raw = frame[f"query_score__{method}"].to_numpy(dtype=np.float32)
        query_margin_raw = frame[f"query_margin__{method}"].to_numpy(dtype=np.float32)
        support_score_raw = frame[f"support_score__{method}"].to_numpy(dtype=np.float32)
        support_margin_raw = frame[f"support_margin__{method}"].to_numpy(dtype=np.float32)
        available = np.isfinite(query_score_raw).astype(np.float32)

        def scaled(values: np.ndarray, name: str) -> np.ndarray:
            mean = stats[name]["mean"]
            std = stats[name]["std"]
            return np.nan_to_num((values - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        per_tool.append(np.column_stack([
            support,
            scaled(query_score_raw, f"query_score__{method}"),
            scaled(query_margin_raw, f"query_margin__{method}"),
            scaled(support_score_raw, f"support_score__{method}"),
            scaled(support_margin_raw, f"support_margin__{method}"),
            (~np.isfinite(query_margin_raw)).astype(np.float32),
            (~np.isfinite(support_margin_raw)).astype(np.float32),
        ]).astype(np.float32))
        availability.append(available)
    tool_values = np.stack(per_tool, axis=1)
    masks = np.stack(availability, axis=1).astype(np.float32)
    globals_list = []
    for feature in global_features:
        values = frame[feature].to_numpy(dtype=np.float32)
        globals_list.append(np.nan_to_num(
            (values - stats[feature]["mean"]) / stats[feature]["std"],
            nan=0.0, posinf=0.0, neginf=0.0,
        ))
    return tool_values, masks, np.column_stack(globals_list).astype(np.float32)


def main() -> None:
    if not (CHECKPOINTS / "CHECKPOINT_24D0_MATRIX_PASS").is_file():
        raise RuntimeError("CHECKPOINT_24D0_MATRIX_PASS is required")
    matrix = pd.read_parquet(RESULTS / "evidencejudge_candidate_matrix.parquet")
    schema = json.loads((RESULTS / "evidencejudge_feature_schema.json").read_text(encoding="utf-8"))
    methods = list(schema["methods"])
    method_features = {
        f"{prefix}__{method}"
        for method in methods
        for prefix in ("support", "support_score", "support_margin", "query_score", "query_margin")
    }
    global_features = [feature for feature in schema["feature_columns"] if feature not in method_features]
    inventory: list[dict[str, object]] = []
    for level in schema["levels"]:
        frame = matrix.loc[matrix["annotation_level"].eq(level)].copy()
        frame["inner_holdout"] = frame["query_cluster_id_30"].map(inner_holdout)
        stats_source = frame.loc[frame["development_partition"].eq("FIT") & ~frame["inner_holdout"]]
        numeric_for_stats = []
        for method in methods:
            numeric_for_stats.extend([
                f"query_score__{method}", f"query_margin__{method}",
                f"support_score__{method}", f"support_margin__{method}",
            ])
        numeric_for_stats.extend(global_features)
        stats: dict[str, dict[str, float]] = {}
        for feature in numeric_for_stats:
            values = stats_source[feature].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            mean = float(finite.mean()) if len(finite) else 0.0
            std = float(finite.std()) if len(finite) else 1.0
            stats[feature] = {"mean": mean, "std": max(std, 1e-6)}
        tool_values, availability, global_values = make_arrays(frame, methods, global_features, stats)
        path = RESULTS / f"evidencejudge_deepsets_arrays__{level.lower()}.npz"
        np.savez_compressed(
            path,
            candidate_row_id=frame["candidate_row_id"].to_numpy(dtype=np.int64),
            partition_code=frame["development_partition"].map(PARTITION_CODE).to_numpy(dtype=np.int8),
            inner_holdout=frame["inner_holdout"].to_numpy(dtype=bool),
            tool_values=tool_values,
            availability=availability,
            global_values=global_values,
            labels=frame["correct"].astype(np.float32).to_numpy(),
            weights=frame["query_weight"].astype(np.float32).to_numpy(),
        )
        preprocessing = {
            "annotation_level": level,
            "methods": methods,
            "global_features": global_features,
            "stats": stats,
            "partition_code": PARTITION_CODE,
            "array_path": str(path),
        }
        (RESULTS / f"evidencejudge_deepsets_preprocessing__{level.lower()}.json").write_text(
            json.dumps(preprocessing, indent=2), encoding="utf-8"
        )
        inventory.append({
            "annotation_level": level,
            "candidate_rows": len(frame),
            "fit_train_rows": int(((frame["development_partition"] == "FIT") & ~frame["inner_holdout"]).sum()),
            "fit_inner_holdout_rows": int(((frame["development_partition"] == "FIT") & frame["inner_holdout"]).sum()),
            "tool_shape": list(tool_values.shape),
            "global_shape": list(global_values.shape),
            "array_path": str(path),
        })
    inventory_frame = pd.DataFrame(inventory)
    inventory_frame.to_csv(RESULTS / "evidencejudge_deepsets_array_inventory.tsv", sep="\t", index=False)
    checks = [
        ("all_three_levels_exported", len(inventory_frame) == 3, len(inventory_frame)),
        ("all_array_files_nonempty", all(Path(path).stat().st_size > 0 for path in inventory_frame["array_path"]), inventory_frame["array_path"].tolist()),
        ("fit_and_inner_holdout_nonempty", bool((inventory_frame["fit_train_rows"] > 0).all() and (inventory_frame["fit_inner_holdout_rows"] > 0).all()), inventory_frame[["fit_train_rows", "fit_inner_holdout_rows"]].to_dict(orient="records")),
    ]
    qc = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    qc.to_csv(REPORTS / "phase24_evidencejudge_deepsets_arrays_qc.tsv", sep="\t", index=False)
    failures = qc.loc[~qc["passed"].astype(bool), "check"].tolist()
    summary = {"phase": "24D2A", "stage": "deepsets_numeric_export", "status": "PASS" if not failures else "FAIL", "failures": failures}
    (REPORTS / "phase24_evidencejudge_deepsets_arrays_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"DeepSets array export failed: {failures}")
    (CHECKPOINTS / "CHECKPOINT_24D2A_ARRAYS_PASS").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
