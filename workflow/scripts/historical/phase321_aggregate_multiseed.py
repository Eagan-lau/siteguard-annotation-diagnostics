#!/usr/bin/env python3
"""Aggregate exactly three Phase11 seeds without selecting or refitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


LEVELS = ["EC_L3", "EC_L4", "EXACT_RHEA"]
FROZEN_SHA256 = {
    "data/interim/phase11/eval_y.npy": "b3548c6893e251aa2ba5193d2e9f5d0537adca3de7df2d7aa28f35597f345465",
    "data/interim/phase11/eval_split.npy": "ee2952ddb9826f3880fd9d1aa3f83fa52b4b3aaea694a9f6bda05a34fc2e8180",
    "models/phase11/siteguard_global_multitask.pt": "bc6e0669eb8e69c80dadea6f0f6ba33ce1b47528df12c9af94a354be2d0cb466",
    "data/interim/phase11/deep_eval_predictions.npy": "99fff9ac44fe6f4c6699676873a86c5e228a721c628ef0e444a0b864f4667c02",
    "models/phase11/model_config.json": "55781c0ef35ec71cab3e9994421f301a643c805c979ed2bf479f7ce0d70fa483",
    "models/phase12/calibration_config.json": "bda14fc5c2ac3e86bb29fffb75124822e9d6a83a39eb4e42ba09bf5386ca7c22",
    "models/phase12/isotonic_EC_L3.joblib": "6152ed122f5c1e408197ea8bc655f8a8db2a4d034b7c83147fe50489b8b478",
    "models/phase12/isotonic_EC_L4.joblib": "1287d4a794b450d6d5ccd699fa174a5303c764b56538b7bcda5651f5aa2fccaf",
    "models/phase12/isotonic_EXACT_RHEA.joblib": "a0bbbc88fd006427ba4fdf9d2270513da72fa6e416ed4193972012010c630837",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def ap(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    outcome = y[order].astype(np.float64)
    positives = float(outcome.sum())
    if positives <= 0:
        return float("nan")
    precision = np.cumsum(outcome) / np.arange(1, len(outcome) + 1)
    return float(np.dot(precision, outcome) / positives)


def write_json_x(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_root.resolve()
    output.relative_to(root)
    output.mkdir(parents=True, exist_ok=True)
    targets = [
        output / "phase321_multiseed_summary.json",
        output / "phase321_seed_metrics.tsv",
        output / "phase321_model_manifest.json",
        output / "phase321_formal_report.md",
        output / "CHECKPOINT_321_PHASE11_THREE_SEED_SUCCESSOR_PASS.json",
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite Phase321 aggregate artifacts")

    frozen = {}
    for relative, expected in FROZEN_SHA256.items():
        path = root / relative
        observed = digest(path)
        if observed != expected:
            raise ValueError(f"frozen pre-Phase321 artifact changed: {relative}")
        frozen[relative] = {"bytes": path.stat().st_size, "sha256": observed}

    eval_y = np.load(root / "data/interim/phase11/eval_y.npy")
    eval_split = np.load(root / "data/interim/phase11/eval_split.npy")
    masks = {"validation": eval_split == 0, "test": eval_split == 1}
    run_root = output / "runs"
    sources = {
        20260819: run_root / "seed_20260819" / "deep_eval_predictions_seed_20260819.npy",
        20260820: root / "data/interim/phase11/deep_eval_predictions.npy",
        20260821: run_root / "seed_20260821" / "deep_eval_predictions_seed_20260821.npy",
    }
    configs = {
        20260819: run_root / "seed_20260819" / "model_config.json",
        20260820: root / "models/phase11/model_config.json",
        20260821: run_root / "seed_20260821" / "model_config.json",
    }
    models = {
        20260819: run_root / "seed_20260819" / "siteguard_global_multitask_seed_20260819.pt",
        20260820: root / "models/phase11/siteguard_global_multitask.pt",
        20260821: run_root / "seed_20260821" / "siteguard_global_multitask_seed_20260821.pt",
    }
    rows = []
    model_manifest = []
    metrics_by_split = {name: [] for name in masks}
    for seed in (20260819, 20260820, 20260821):
        config = json.loads(configs[seed].read_text(encoding="utf-8"))
        if int(config.get("seed", -1)) != seed:
            raise ValueError(f"seed/config mismatch: {seed}")
        predictions = np.load(sources[seed], mmap_mode="r")
        if predictions.shape != eval_y.shape or predictions.shape[1] != 3:
            raise ValueError(f"prediction shape mismatch for seed {seed}: {predictions.shape}")
        for split_name, mask in masks.items():
            values = [ap(eval_y[mask, i], predictions[mask, i]) for i in range(3)]
            metrics_by_split[split_name].append(values)
            for level, value in zip(LEVELS, values):
                rows.append((seed, split_name, level, value))
        for role, path in (("model", models[seed]), ("predictions", sources[seed]), ("config", configs[seed])):
            model_manifest.append({
                "seed": seed,
                "role": role,
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            })

    aggregate = {}
    for split_name, matrix in metrics_by_split.items():
        values = np.asarray(matrix, dtype=float)
        aggregate[split_name] = {
            level: {
                "mean": float(values[:, index].mean()),
                "sample_std": float(values[:, index].std(ddof=1)),
                "minimum": float(values[:, index].min()),
                "maximum": float(values[:, index].max()),
            }
            for index, level in enumerate(LEVELS)
        }

    metrics_lines = ["seed\tsplit\tlevel\tauprc"] + [
        f"{seed}\t{split}\t{level}\t{value:.12g}" for seed, split, level, value in rows
    ]
    (output / "phase321_seed_metrics.tsv").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8", newline="\n")
    write_json_x(output / "phase321_model_manifest.json", {
        "format": "siteguard.phase321.three-seed-model-manifest.v1",
        "artifacts": model_manifest,
    })
    summary = {
        "format": "siteguard.phase321.phase11-three-seed-successor-summary.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_PHASE321_PHASE11_THREE_SEED_SUCCESSOR_NO_BEST_SEED_SELECTION_NO_THRESHOLD_REFIT",
        "seeds": [20260819, 20260820, 20260821],
        "primary_seed": 20260819,
        "frozen_legacy_seed": 20260820,
        "new_additive_seed_count": 2,
        "identical_frozen_eval_population": True,
        "test_used_for_model_selection": False,
        "best_seed_only_reporting": False,
        "threshold_refit_performed": False,
        "aggregate_auprc": aggregate,
        "frozen_pre_phase321_artifacts": frozen,
        "models_or_thresholds_overwritten": False,
        "submission_ready": False,
    }
    write_json_x(output / "phase321_multiseed_summary.json", summary)
    report = [
        "# Phase321 Phase11 three-seed successor",
        "",
        f"Status: `{summary['status']}`",
        "",
        "The original seed 20260820 is preserved byte-for-byte. New protected candidates were trained for the task-book primary seed 20260819 and the adjacent independent seed 20260821 on the identical frozen Phase11 arrays. All three seeds are reported; no best-seed-only selection or threshold refit was performed.",
        "",
        "## Aggregate AUPRC",
        "",
        "| Split | Level | Mean | Sample SD | Min | Max |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split_name in ("validation", "test"):
        for level in LEVELS:
            value = aggregate[split_name][level]
            report.append(f"| {split_name} | {level} | {value['mean']:.6f} | {value['sample_std']:.6f} | {value['minimum']:.6f} | {value['maximum']:.6f} |")
    report += [
        "",
        "This successor closes the literal three-seed reporting requirement only. It does not replace the locked production model, refit calibration thresholds, or authorize release/submission.",
    ]
    (output / "phase321_formal_report.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    checkpoint_payload = {
        "format": "siteguard.phase321.three-seed-checkpoint.v1",
        "status": summary["status"],
        "summary_sha256": digest(output / "phase321_multiseed_summary.json"),
        "metrics_sha256": digest(output / "phase321_seed_metrics.tsv"),
        "manifest_sha256": digest(output / "phase321_model_manifest.json"),
        "report_sha256": digest(output / "phase321_formal_report.md"),
        "terminal": False,
    }
    write_json_x(output / "CHECKPOINT_321_PHASE11_THREE_SEED_SUCCESSOR_PASS.json", checkpoint_payload)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
