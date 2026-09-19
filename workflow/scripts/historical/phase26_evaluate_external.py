#!/usr/bin/env python3
"""Open the frozen SABIO truth once and evaluate locked external predictions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
R26 = ROOT / "results/phase26"
REPORT_DIR = ROOT / "reports/phase26_external_blind"
CHECKPOINTS = ROOT / "checkpoints"
LOCK_PATH = ROOT / "models/phase26_external_inference_lock.json"
BOOTSTRAPS = 5000
SEED = 20260823


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context_seed(context: str) -> int:
    digest = hashlib.sha256(f"{SEED}|{context}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def cluster_interval(frame: pd.DataFrame, context: str) -> tuple[float, float]:
    if frame.empty:
        return np.nan, np.nan
    grouped = frame.groupby("external_cluster_id_30", observed=True)["correct"].agg(["sum", "count"])
    values = grouped[["sum", "count"]].to_numpy(dtype=float)
    if len(values) == 1:
        value = float(values[0, 0] / values[0, 1])
        return value, value
    rng = np.random.default_rng(context_seed(context))
    draws = np.empty(BOOTSTRAPS, dtype=float)
    for start in range(0, BOOTSTRAPS, 250):
        size = min(250, BOOTSTRAPS - start)
        sampled = values[rng.integers(0, len(values), size=(size, len(values)))]
        draws[start:start + size] = sampled[:, :, 0].sum(axis=1) / sampled[:, :, 1].sum(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def measure(frame: pd.DataFrame, total: int, context: str) -> dict[str, object]:
    queries = len(frame)
    clusters = int(frame["external_cluster_id_30"].nunique()) if queries else 0
    precision = float(frame["correct"].mean()) if queries else np.nan
    ci_low, ci_high = cluster_interval(frame, context)
    coverage = float(queries / total) if total else 0.0
    qualifies = bool(queries >= 50 and clusters >= 20 and precision >= 0.95 and ci_low >= 0.95)
    return {
        "queries_total": total,
        "accepted_queries": queries,
        "accepted_clusters": clusters,
        "coverage": coverage,
        "documented_precision": precision,
        "cluster_ci_low": ci_low,
        "cluster_ci_high": ci_high,
        "minimum_queries_met": queries >= 50,
        "minimum_clusters_met": clusters >= 20,
        "point_precision_met": bool(queries and precision >= 0.95),
        "cluster_ci_lower_met": bool(queries and ci_low >= 0.95),
        "qualifies_95": qualifies,
        "safe_coverage_at_95": coverage if qualifies else 0.0,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    required = CHECKPOINTS / "CHECKPOINT_26A5_EXTERNAL_BLIND_PREDICTIONS_LOCKED"
    if not required.is_file():
        raise FileNotFoundError(required)
    if (R26 / "external_predictions.parquet").exists():
        raise RuntimeError("External evaluation output already exists; holdout can only be opened once")
    lock_hash_before = sha256(LOCK_PATH)
    inference_lock = json.loads(LOCK_PATH.read_text())
    if inference_lock["truth_opened"]:
        raise RuntimeError("Inference lock reports truth already opened")

    cohort = pd.read_parquet(R26 / "sabio_strict_blind_cohort.parquet")
    blind = pd.read_parquet(R26 / "external_blind_predictions.parquet")
    blind = blind.rename(columns={"query_cluster_id_30": "external_cluster_id_30"})
    baseline_path = R26 / "external_blind_single_tool_predictions.parquet"
    baseline = pd.read_parquet(baseline_path) if baseline_path.exists() else pd.DataFrame()
    truth = cohort[[
        "uniprot_accession", "external_cluster_id_30", "ec_l3", "ec_l4",
        "ec_l3_label_eligible", "ec_l4_label_eligible",
    ]].rename(columns={"uniprot_accession": "query_protein_id"})
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
    predictions["evaluation_status"] = "CONFIRMATORY_EXTERNAL_BLIND"
    predictions.to_parquet(R26 / "external_predictions.parquet", index=False, compression="zstd")

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
        judge = measure(accepted, totals[level], f"EVIDENCEJUDGE|{level}")
        endpoint_rows.append({"annotation_level": level, "system": "EVIDENCEJUDGE", **judge})

        if len(baseline) and level in set(baseline["annotation_level"]):
            one = baseline.loc[baseline["annotation_level"].eq(level)].merge(
                truth, left_on="query_protein_id", right_on="query_protein_id", how="left", validate="many_to_one"
            )
            one["truth_label"] = np.where(level == "EC_L3", one["ec_l3"], one["ec_l4"])
            one["label_eligible"] = np.where(level == "EC_L3", one["ec_l3_label_eligible"], one["ec_l4_label_eligible"]).astype(bool)
            one["correct"] = one["label_eligible"] & one["candidate_label"].astype(str).eq(one["truth_label"].astype(str))
            one = one.loc[one["accepted"].astype(bool) & one["label_eligible"]].copy()
            baseline_export.append(one)
            single = measure(one, totals[level], f"BEST_SINGLE|{level}")
        else:
            single = measure(pd.DataFrame(columns=["external_cluster_id_30", "correct"]), totals[level], f"BEST_SINGLE|{level}")
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
    endpoints.to_csv(R26 / "external_operating_points.tsv", sep="\t", index=False)
    comparison.to_csv(R26 / "external_incremental_safe_coverage.tsv", sep="\t", index=False)
    if baseline_export:
        pd.concat(baseline_export, ignore_index=True).to_parquet(
            R26 / "external_single_tool_evaluated_predictions.parquet", index=False, compression="zstd"
        )

    lock_hash_after = sha256(LOCK_PATH)
    release_levels = comparison.loc[comparison["confirmatory_release_gate"], "annotation_level"].tolist()
    summary = {
        "phase": "26B",
        "stage": "confirmatory_external_blind_evaluation",
        "status": "PASS",
        "scientific_decision": "CONFIRMATORY_ENDPOINT_MET" if release_levels else "CONFIRMATORY_ENDPOINT_NOT_MET",
        "release_levels": release_levels,
        "truth_opened_once": True,
        "inference_lock_unchanged": lock_hash_before == lock_hash_after,
        "bootstrap_draws": BOOTSTRAPS,
        "endpoints": endpoints.to_dict("records"),
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
    qc.to_csv(REPORT_DIR / "phase26_external_evaluation_qc.tsv", sep="\t", index=False)
    failures = qc.loc[~qc["passed"].astype(bool), "check"].tolist()
    if failures:
        summary["status"] = "FAIL"
        summary["failures"] = failures
    (REPORT_DIR / "phase26_external_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    if failures:
        raise RuntimeError(f"Phase 26 evaluation integrity failure: {failures}")
    (CHECKPOINTS / "CHECKPOINT_26B_EXTERNAL_EVALUATION_PASS").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
