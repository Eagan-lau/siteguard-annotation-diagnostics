#!/usr/bin/env python3
"""Governed additive recovery for the failed Phase321 aggregate job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


LEVELS = ("EC_L3", "EC_L4", "EXACT_RHEA")
SEEDS = (20260819, 20260820, 20260821)
SOURCE_JOB = "1575884"
FAILED_JOB = "1575886"
FROZEN_SHA256 = {
    "data/interim/phase11/eval_y.npy": "b3548c6893e251aa2ba5193d2e9f5d0537adca3de7df2d7aa28f35597f345465",
    "data/interim/phase11/eval_split.npy": "ee2952ddb9826f3880fd9d1aa3f83fa52b4b3aaea694a9f6bda05a34fc2e8180",
    "models/phase11/siteguard_global_multitask.pt": "bc6e0669eb8e69c80dadea6f0f6ba33ce1b47528df12c9af94a354be2d0cb466",
    "data/interim/phase11/deep_eval_predictions.npy": "99fff9ac44fe6f4c6699676873a86c5e228a721c628ef0e444a0b864f4667c02",
    "models/phase11/model_config.json": "55781c0ef35ec71cab3e9994421f301a643c805c979ed2bf479f7ce0d70fa483",
    "models/phase12/calibration_config.json": "bda14fc5c2ac3e86bb29fffb75124822e9d6a83a39eb4e42ba09bf5386ca7c22",
    "models/phase12/isotonic_EC_L3.joblib": "6152ed122f5c1e408197ea8bc655f8a8a8db2a4d034b7c83147fe50489b8b478",
    "models/phase12/isotonic_EC_L4.joblib": "1287d4a794b450d6d5ccd699fa174a5303c764b56538b7bcda5651f5aa2fccaf",
    "models/phase12/isotonic_EXACT_RHEA.joblib": "a0bbbc88fd006427ba4fdf9d2270513da72fa6e416ed4193972012010c630837",
}
PHASE321_SHA256 = {
    "reports/phase321_phase11_multiseed_20260903/runs/seed_20260819/CHECKPOINT_SINGLE_SEED_PASS.json": "cf6234682e45b42e4e2d65263db19b707cd305145b0e6fd44dee07f5e6fb11e6",
    "reports/phase321_phase11_multiseed_20260903/runs/seed_20260819/run_summary.json": "b04efb8aef76f8f3d0adcf8dbae251e40d97c142ee9b0ea9394f9bb207ad1d7f",
    "reports/phase321_phase11_multiseed_20260903/runs/seed_20260819/model_config.json": "370533050afd72d19a275b647e07e48632e60367b9eb7b37628ce8ddb39e404a",
    "reports/phase321_phase11_multiseed_20260903/runs/seed_20260819/siteguard_global_multitask_seed_20260819.pt": "c0e911ae512f193c87b886b4a13a69b6d1970493dfd728796a436dc8689f85c5",
    "reports/phase321_phase11_multiseed_20260903/runs/seed_20260819/deep_eval_predictions_seed_20260819.npy": "716ac8f4e88ae0848cc504430a7b052a4c6bc8b52d23d782f7f44d7fab7cb294",
    "reports/phase321_phase11_multiseed_20260903/runs/seed_20260821/CHECKPOINT_SINGLE_SEED_PASS.json": "c2d230eeb5c3673b07a01d0c53f7821a8547b1b18a6d3ca49911a4f8b3447e04",
    "reports/phase321_phase11_multiseed_20260903/runs/seed_20260821/run_summary.json": "d4f52e90c0035908ea22e270a94bfac37b0e7456349daf7bc54670647e1e0ae9",
    "reports/phase321_phase11_multiseed_20260903/runs/seed_20260821/model_config.json": "a7746207c5dd274216c737cfec0ef03e17f6a75691c5d33f3b551d9de824d474",
    "reports/phase321_phase11_multiseed_20260903/runs/seed_20260821/siteguard_global_multitask_seed_20260821.pt": "a3f35b4ed0786d584661de335d871e8e0bd17f53814282fa833217acaa7c1ebf",
    "reports/phase321_phase11_multiseed_20260903/runs/seed_20260821/deep_eval_predictions_seed_20260821.npy": "356fa32cc72d13098597474c7259068034093300b6ffab910730bf489395f0d9",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_text_x(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def write_json_x(path: Path, value: object) -> None:
    write_text_x(path, json.dumps(value, indent=2) + "\n")


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    outcome = y[order].astype(np.float64)
    positives = float(outcome.sum())
    if positives <= 0:
        return float("nan")
    precision = np.cumsum(outcome) / np.arange(1, len(outcome) + 1)
    return float(np.dot(precision, outcome) / positives)


def verify_identities(root: Path, expected: dict[str, str]) -> dict[str, dict[str, object]]:
    observed: dict[str, dict[str, object]] = {}
    for relative, expected_sha in expected.items():
        if len(expected_sha) != 64:
            raise ValueError(f"contract SHA-256 is not 64 characters: {relative}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = digest(path)
        if actual != expected_sha:
            raise ValueError(f"identity mismatch: {relative}")
        observed[relative] = {"bytes": path.stat().st_size, "sha256": actual}
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_root.resolve()
    output.relative_to(root)
    if output.exists():
        raise FileExistsError(f"refusing to reuse Phase322 output: {output}")

    frozen = verify_identities(root, FROZEN_SHA256)
    phase321 = verify_identities(root, PHASE321_SHA256)
    output.mkdir(parents=True, exist_ok=False)

    eval_y = np.load(root / "data/interim/phase11/eval_y.npy", mmap_mode="r")
    eval_split = np.load(root / "data/interim/phase11/eval_split.npy", mmap_mode="r")
    masks = {"validation": eval_split == 0, "test": eval_split == 1}
    prediction_paths = {
        20260819: root / "reports/phase321_phase11_multiseed_20260903/runs/seed_20260819/deep_eval_predictions_seed_20260819.npy",
        20260820: root / "data/interim/phase11/deep_eval_predictions.npy",
        20260821: root / "reports/phase321_phase11_multiseed_20260903/runs/seed_20260821/deep_eval_predictions_seed_20260821.npy",
    }
    model_paths = {
        20260819: root / "reports/phase321_phase11_multiseed_20260903/runs/seed_20260819/siteguard_global_multitask_seed_20260819.pt",
        20260820: root / "models/phase11/siteguard_global_multitask.pt",
        20260821: root / "reports/phase321_phase11_multiseed_20260903/runs/seed_20260821/siteguard_global_multitask_seed_20260821.pt",
    }
    config_paths = {
        20260819: root / "reports/phase321_phase11_multiseed_20260903/runs/seed_20260819/model_config.json",
        20260820: root / "models/phase11/model_config.json",
        20260821: root / "reports/phase321_phase11_multiseed_20260903/runs/seed_20260821/model_config.json",
    }

    rows: list[tuple[int, str, str, float]] = []
    metrics_by_split: dict[str, list[list[float]]] = {name: [] for name in masks}
    manifest: list[dict[str, object]] = []
    for seed in SEEDS:
        config = json.loads(config_paths[seed].read_text(encoding="utf-8"))
        if int(config.get("seed", -1)) != seed:
            raise ValueError(f"seed/config mismatch: {seed}")
        predictions = np.load(prediction_paths[seed], mmap_mode="r")
        if predictions.shape != eval_y.shape or predictions.shape[1] != len(LEVELS):
            raise ValueError(f"prediction shape mismatch for seed {seed}: {predictions.shape}")
        for split_name, mask in masks.items():
            values = [average_precision(eval_y[mask, index], predictions[mask, index]) for index in range(3)]
            metrics_by_split[split_name].append(values)
            rows.extend((seed, split_name, level, value) for level, value in zip(LEVELS, values))
        for role, path in (("model", model_paths[seed]), ("predictions", prediction_paths[seed]), ("config", config_paths[seed])):
            manifest.append({
                "seed": seed,
                "role": role,
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            })

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
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

    metrics = ["seed\tsplit\tlevel\tauprc"]
    metrics.extend(f"{seed}\t{split}\t{level}\t{value:.12g}" for seed, split, level, value in rows)
    write_text_x(output / "phase322_seed_metrics.tsv", "\n".join(metrics) + "\n")
    write_json_x(output / "phase322_model_manifest.json", {
        "format": "siteguard.phase322.three-seed-model-manifest.v1",
        "artifacts": manifest,
    })
    write_json_x(output / "phase322_failure_acknowledgement.json", {
        "format": "siteguard.phase322.phase321-failure-acknowledgement.v1",
        "failed_job": FAILED_JOB,
        "failed_job_state": "FAILED",
        "failed_job_exit_code": "1:0",
        "failure": "62-character registered EC_L3 calibrator SHA-256",
        "corrected_sha256": FROZEN_SHA256["models/phase12/isotonic_EC_L3.joblib"],
        "training_rerun": False,
        "phase321_artifacts_overwritten": False,
    })
    summary = {
        "format": "siteguard.phase322.phase321-aggregation-recovery-summary.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_PHASE322_GOVERNED_AGGREGATION_RECOVERY_THREE_SEED_REPORTING_CLOSED_NOT_MODEL_AUTHORITY",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "source_training_job": SOURCE_JOB,
        "failed_aggregate_job": FAILED_JOB,
        "seeds": list(SEEDS),
        "primary_seed": 20260819,
        "frozen_legacy_seed": 20260820,
        "aggregate_auprc": aggregate,
        "frozen_artifacts": frozen,
        "phase321_seed_artifacts": phase321,
        "test_used_for_model_selection": False,
        "best_seed_only_reporting": False,
        "threshold_refit_performed": False,
        "training_rerun": False,
        "models_or_thresholds_overwritten": False,
        "submission_ready": False,
    }
    write_json_x(output / "phase322_multiseed_summary.json", summary)

    report = [
        "# Phase322 governed Phase321 aggregation recovery",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Phase321 Job 1575886 failed closed because one registered SHA-256 string contained 62 rather than 64 hexadecimal characters. The underlying EC-L3 calibrator was identical locally and on Lyra and was not changed. Phase322 corrects only that transcription and aggregates the two successful additive runs with the frozen legacy run.",
        "",
        "| Split | Level | Mean | Sample SD | Min | Max |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split_name in ("validation", "test"):
        for level in LEVELS:
            value = aggregate[split_name][level]
            report.append(f"| {split_name} | {level} | {value['mean']:.6f} | {value['sample_std']:.6f} | {value['minimum']:.6f} | {value['maximum']:.6f} |")
    report.extend([
        "",
        "All three seeds are reported. No best-seed selection, threshold refit, calibration change, production-model replacement, release, or submission action was performed.",
    ])
    write_text_x(output / "phase322_formal_report.md", "\n".join(report) + "\n")

    terminal = {
        "format": "siteguard.phase322.terminal.v1",
        "status": summary["status"],
        "terminal": True,
        "successful": True,
        "independent_audit_required": True,
        "submission_ready": False,
    }
    write_json_x(output / "phase322_terminal.json", terminal)
    outputs = [
        "phase322_seed_metrics.tsv",
        "phase322_model_manifest.json",
        "phase322_failure_acknowledgement.json",
        "phase322_multiseed_summary.json",
        "phase322_formal_report.md",
        "phase322_terminal.json",
    ]
    checkpoint = {
        "format": "siteguard.phase322.checkpoint.v1",
        "status": summary["status"],
        "artifacts": {name: {"bytes": (output / name).stat().st_size, "sha256": digest(output / name)} for name in outputs},
        "terminal_sha256": digest(output / "phase322_terminal.json"),
        "independent_audit_required": True,
        "submission_ready": False,
    }
    write_json_x(output / "CHECKPOINT_322_PHASE321_AGGREGATION_RECOVERY_PASS.json", checkpoint)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
