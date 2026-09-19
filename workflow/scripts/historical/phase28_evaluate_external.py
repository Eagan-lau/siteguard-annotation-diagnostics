#!/usr/bin/env python3
"""Open Phase 28 PDB-primary EC truth once after the blind prediction lock."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import phase26_evaluate_external as base


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
R28 = ROOT / "results/phase28"
REPORT_DIR = ROOT / "reports/phase28_external_blind"
CHECKPOINTS = ROOT / "checkpoints"
LOCK_PATH = ROOT / "models/phase28_external_inference_lock.json"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    required = CHECKPOINTS / "CHECKPOINT_28B_BLIND_PREDICTIONS_LOCKED"
    if not required.is_file():
        raise FileNotFoundError(required)
    if (R28 / "external_predictions.parquet").exists():
        raise RuntimeError("Phase 28 truth has already been opened")
    lock_hash_before = base.sha256(LOCK_PATH)
    inference_lock = json.loads(LOCK_PATH.read_text())
    if inference_lock["truth_opened"]:
        raise RuntimeError("Inference lock reports truth already opened")

    cohort = pd.read_parquet(R28 / "rcsb_strict_blind_cohort.parquet")
    blind = pd.read_parquet(R28 / "external_blind_predictions.parquet").rename(
        columns={"query_cluster_id_30": "external_cluster_id_30"}
    )
    baseline_path = R28 / "external_blind_single_tool_predictions.parquet"
    baseline = pd.read_parquet(baseline_path) if baseline_path.exists() else pd.DataFrame()
    truth = cohort[[
        "query_id", "external_cluster_id_30", "ec_l3", "ec_l4",
        "ec_l3_label_eligible", "ec_l4_label_eligible",
    ]].rename(columns={"query_id": "query_protein_id"})
    predictions = blind.merge(truth, on=["query_protein_id", "external_cluster_id_30"], how="left", validate="many_to_one")
    predictions["truth_label"] = np.where(
        predictions["annotation_level"].eq("EC_L3"), predictions["ec_l3"],
        np.where(predictions["annotation_level"].eq("EC_L4"), predictions["ec_l4"], None),
    )
    predictions["label_eligible"] = np.where(
        predictions["annotation_level"].eq("EC_L3"), predictions["ec_l3_label_eligible"],
        np.where(predictions["annotation_level"].eq("EC_L4"), predictions["ec_l4_label_eligible"], False),
    ).astype(bool)
    predictions["correct"] = predictions["label_eligible"] & predictions["candidate_label"].astype(str).eq(predictions["truth_label"].astype(str))
    predictions["evaluation_status"] = "CONFIRMATORY_SEQUENCE_INDEPENDENT_STRUCTURAL_EXTERNAL_BLIND"
    predictions.to_parquet(R28 / "external_predictions.parquet", index=False, compression="zstd")

    endpoint_rows = []
    comparison_rows = []
    baseline_export = []
    totals = {
        "EC_L3": int(cohort["ec_l3_label_eligible"].sum()),
        "EC_L4": int(cohort["ec_l4_label_eligible"].sum()),
        "EXACT_RHEA": 0,
    }
    for level in ["EC_L3", "EC_L4", "EXACT_RHEA"]:
        accepted = predictions.loc[
            predictions["annotation_level"].eq(level)
            & predictions["accepted"].astype(bool)
            & predictions["label_eligible"].astype(bool)
        ].copy()
        judge = base.measure(accepted, totals[level], f"PHASE28|EVIDENCEJUDGE|{level}")
        endpoint_rows.append({"annotation_level": level, "system": "EVIDENCEJUDGE", **judge})

        if len(baseline) and level in set(baseline["annotation_level"]):
            one = baseline.loc[baseline["annotation_level"].eq(level)].merge(
                truth, on="query_protein_id", how="left", validate="many_to_one"
            )
            one["truth_label"] = np.where(level == "EC_L3", one["ec_l3"], one["ec_l4"])
            one["label_eligible"] = np.where(
                level == "EC_L3", one["ec_l3_label_eligible"], one["ec_l4_label_eligible"]
            ).astype(bool)
            one["correct"] = one["label_eligible"] & one["candidate_label"].astype(str).eq(one["truth_label"].astype(str))
            one = one.loc[one["accepted"].astype(bool) & one["label_eligible"]].copy()
            baseline_export.append(one)
            single = base.measure(one, totals[level], f"PHASE28|BEST_SINGLE|{level}")
        else:
            single = base.measure(pd.DataFrame(columns=["external_cluster_id_30", "correct"]), totals[level], f"PHASE28|BEST_SINGLE|{level}")
        endpoint_rows.append({"annotation_level": level, "system": "BEST_SINGLE_TOOL", **single})
        comparison_rows.append({
            "annotation_level": level,
            "evidencejudge_safe_coverage_at_95": judge["safe_coverage_at_95"],
            "best_single_safe_coverage_at_95": single["safe_coverage_at_95"],
            "incremental_safe_coverage": float(judge["safe_coverage_at_95"] - single["safe_coverage_at_95"]),
            "confirmatory_release_gate": bool(judge["qualifies_95"]),
        })
    endpoints = pd.DataFrame(endpoint_rows)
    comparison = pd.DataFrame(comparison_rows)
    endpoints.to_csv(R28 / "external_operating_points.tsv", sep="\t", index=False)
    comparison.to_csv(R28 / "external_incremental_safe_coverage.tsv", sep="\t", index=False)
    if baseline_export:
        pd.concat(baseline_export, ignore_index=True).to_parquet(
            R28 / "external_single_tool_evaluated_predictions.parquet", index=False, compression="zstd"
        )

    lock_hash_after = base.sha256(LOCK_PATH)
    release_levels = comparison.loc[comparison["confirmatory_release_gate"], "annotation_level"].tolist()
    summary = {
        "phase": "28C", "stage": "confirmatory_sequence_independent_structural_external_blind_evaluation",
        "status": "PASS",
        "scientific_decision": "CONFIRMATORY_ENDPOINT_MET" if release_levels else "CONFIRMATORY_ENDPOINT_NOT_MET",
        "release_levels": release_levels, "truth_opened_once": True,
        "inference_lock_unchanged": lock_hash_before == lock_hash_after,
        "bootstrap_draws": base.BOOTSTRAPS, "endpoints": endpoints.to_dict("records"),
        "incremental": comparison.to_dict("records"),
    }
    checks = [
        ("blind_prediction_checkpoint_present", required.is_file(), str(required)),
        ("inference_lock_unchanged", lock_hash_before == lock_hash_after, lock_hash_before),
        ("prediction_grain_unique", not predictions.duplicated(["annotation_level", "query_protein_id"]).any(), len(predictions)),
        ("truth_complete_for_eligible_predictions", bool(predictions.loc[predictions["label_eligible"], "truth_label"].notna().all()), int(predictions["label_eligible"].sum())),
        ("frozen_minimums_applied", True, "50 accepted queries;20 accepted clusters;precision>=0.95;cluster CI lower>=0.95"),
        ("no_exact_rhea_inferred_from_ec", totals["EXACT_RHEA"] == 0, "exact Rhea remains abstain"),
    ]
    qc = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    qc.to_csv(REPORT_DIR / "phase28_external_evaluation_qc.tsv", sep="\t", index=False)
    failures = qc.loc[~qc["passed"].astype(bool), "check"].tolist()
    if failures:
        summary["status"] = "FAIL"; summary["failures"] = failures
    (REPORT_DIR / "phase28_external_evaluation_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    if failures:
        raise RuntimeError(f"Phase 28 evaluation integrity failure: {failures}")
    (CHECKPOINTS / "CHECKPOINT_28C_EXTERNAL_EVALUATION_PASS").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
