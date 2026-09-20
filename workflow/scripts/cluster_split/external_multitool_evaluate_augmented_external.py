#!/usr/bin/env python3
"""Open Phase30 truth once, after the augmented blind prediction lock."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
R30 = ROOT / "results/phase30"
REPORTS = ROOT / "reports/phase30_external_blind"
CHECKPOINTS = ROOT / "checkpoints"
LOCK_PATH = ROOT / "models/phase30_external_inference_lock.json"
BOOTSTRAPS = 10000
SEED = 20260830


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(frame: pd.DataFrame, context: str) -> tuple[float, float]:
    groups = frame.groupby("query_cluster_id_30", observed=True)["correct"].agg(["sum", "count"])
    values = groups[["sum", "count"]].to_numpy(float)
    if not len(values):
        return float("nan"), float("nan")
    if np.all(values[:, 0] == values[:, 1]):
        return 1.0, 1.0
    seed = int.from_bytes(hashlib.sha256(f"{SEED}|{context}".encode()).digest()[:8], "big") % (2**32 - 1)
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAPS)
    for start in range(0, BOOTSTRAPS, 250):
        n = min(250, BOOTSTRAPS - start)
        sample = values[rng.integers(0, len(values), size=(n, len(values)))]
        estimates[start : start + n] = sample[:, :, 0].sum(1) / sample[:, :, 1].sum(1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def measure(frame: pd.DataFrame, total: int, context: str) -> dict[str, object]:
    n = len(frame)
    successes = int(frame["correct"].sum()) if n else 0
    clusters = int(frame["query_cluster_id_30"].nunique()) if n else 0
    precision = successes / n if n else float("nan")
    low, high = bootstrap(frame, context) if n else (float("nan"), float("nan"))
    cp_low = float(beta.ppf(0.025, successes, n - successes + 1)) if successes else 0.0
    qualifies = bool(n >= 50 and clusters >= 20 and precision >= 0.95 and low >= 0.95) if n else False
    return {
        "total_eligible_queries": total, "accepted_queries": n, "accepted_clusters": clusters,
        "coverage": n / total if total else 0.0, "precision": precision,
        "cluster_bootstrap_low": low, "cluster_bootstrap_high": high,
        "clopper_pearson_two_sided_low": cp_low, "qualifies_95": qualifies,
        "safe_coverage_at_95": n / total if qualifies and total else 0.0,
    }


def attach_truth(predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    output = predictions.merge(
        truth, left_on=["query_protein_id", "query_cluster_id_30"],
        right_on=["query_id", "external_cluster_id_30"], how="left", validate="many_to_one",
    )
    output["truth_label"] = np.where(
        output["annotation_level"].eq("EC_L3"), output["ec_l3"], output["ec_l4"]
    )
    output["label_eligible"] = np.where(
        output["annotation_level"].eq("EC_L3"), output["ec_l3_label_eligible"], output["ec_l4_label_eligible"]
    ).astype(bool)
    output["correct"] = output["label_eligible"] & output["candidate_label"].astype(str).eq(output["truth_label"].astype(str))
    return output


def main() -> None:
    required = CHECKPOINTS / "CHECKPOINT_30B_BLIND_PREDICTIONS_LOCKED"
    if not required.is_file():
        raise FileNotFoundError(required)
    if (R30 / "external_predictions.parquet").exists():
        raise RuntimeError("Phase30 truth has already been opened")
    lock_hash_before = sha256(LOCK_PATH)
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["truth_opened"]:
        raise RuntimeError("Phase30 inference lock already reports opened truth")

    cohort = pd.read_parquet(R30 / "rcsb_strict_blind_cohort.parquet")
    truth = cohort[[
        "query_id", "external_cluster_id_30", "ec_l3", "ec_l4",
        "ec_l3_label_eligible", "ec_l4_label_eligible",
    ]]
    totals = {
        "EC_L3": int(cohort["ec_l3_label_eligible"].sum()),
        "EC_L4": int(cohort["ec_l4_label_eligible"].sum()),
    }

    blind = pd.read_parquet(R30 / "external_blind_predictions.parquet")
    evaluated = attach_truth(blind, truth)
    evaluated["evaluation_status"] = "PHASE30_LOCKED_RETROSPECTIVE_EXTERNAL_BLIND"
    evaluated.to_parquet(R30 / "external_predictions.parquet", index=False, compression="zstd")

    rules = attach_truth(pd.read_parquet(R30 / "external_blind_rule_predictions.parquet"), truth)
    rules.to_parquet(R30 / "external_rule_predictions.parquet", index=False, compression="zstd")
    singles = attach_truth(pd.read_parquet(R30 / "external_blind_single_tool_predictions.parquet"), truth)
    singles.to_parquet(R30 / "external_single_tool_evaluated_predictions.parquet", index=False, compression="zstd")

    endpoint_rows = []
    accepted = evaluated.loc[evaluated["accepted"].astype(bool) & evaluated["label_eligible"]].copy()
    judge_metric = measure(accepted, totals["EC_L3"], "PHASE30|EVIDENCEJUDGE|EC_L3")
    endpoint_rows.append({"annotation_level": "EC_L3", "system": "AUGMENTED_EVIDENCEJUDGE_LOGISTIC_L2", **judge_metric})
    for level in ("EC_L3", "EC_L4"):
        group = rules.loc[
            rules["annotation_level"].eq(level) & rules["accepted"].astype(bool) & rules["label_eligible"]
        ]
        endpoint_rows.append({
            "annotation_level": level, "system": f"TRANSPARENT_{level}_RULE",
            **measure(group, totals[level], f"PHASE30|RULE|{level}"),
        })
    single_rows = []
    for (level, method), group in singles.groupby(["annotation_level", "method"], observed=True):
        accepted_single = group.loc[group["accepted"].astype(bool) & group["label_eligible"]]
        metric = measure(accepted_single, totals[level], f"PHASE30|SINGLE|{level}|{method}")
        row = {"annotation_level": level, "system": f"SINGLE::{method}", **metric}
        endpoint_rows.append(row)
        single_rows.append({"annotation_level": level, "method": method, **metric})
    endpoints = pd.DataFrame(endpoint_rows)
    endpoints.to_csv(R30 / "external_operating_points.tsv", sep="\t", index=False)
    single_endpoints = pd.DataFrame(single_rows)
    single_endpoints.to_csv(R30 / "external_single_tool_operating_points.tsv", sep="\t", index=False)

    best = single_endpoints.loc[single_endpoints["qualifies_95"].astype(bool)].sort_values(
        ["annotation_level", "safe_coverage_at_95", "precision"],
        ascending=[True, False, False], kind="mergesort",
    ).drop_duplicates("annotation_level")
    comparison = []
    for level in ("EC_L3", "EC_L4"):
        if level == "EC_L3":
            target = endpoints.loc[endpoints["system"].eq("AUGMENTED_EVIDENCEJUDGE_LOGISTIC_L2")].iloc[0]
        else:
            target = endpoints.loc[endpoints["system"].eq("TRANSPARENT_EC_L4_RULE")].iloc[0]
        baseline = best.loc[best["annotation_level"].eq(level)]
        baseline_coverage = float(baseline.iloc[0]["safe_coverage_at_95"]) if len(baseline) else 0.0
        comparison.append({
            "annotation_level": level,
            "evidencejudge_system": target["system"],
            "evidencejudge_safe_coverage_at_95": float(target["safe_coverage_at_95"]),
            "best_single_tool": str(baseline.iloc[0]["method"]) if len(baseline) else "NONE_QUALIFIED",
            "best_single_safe_coverage_at_95": baseline_coverage,
            "incremental_safe_coverage": float(target["safe_coverage_at_95"] - baseline_coverage),
            "confirmatory_release_gate": bool(target["qualifies_95"] and float(target["safe_coverage_at_95"] - baseline_coverage) > 0),
        })
    comparison_frame = pd.DataFrame(comparison)
    comparison_frame.to_csv(R30 / "external_incremental_safe_coverage.tsv", sep="\t", index=False)

    errors = pd.concat([
        evaluated.loc[evaluated["accepted"].astype(bool) & evaluated["label_eligible"] & ~evaluated["correct"]].assign(system="AUGMENTED_EVIDENCEJUDGE_LOGISTIC_L2"),
        rules.loc[rules["accepted"].astype(bool) & rules["label_eligible"] & ~rules["correct"]],
    ], ignore_index=True, sort=False)
    errors.to_csv(R30 / "external_accepted_errors.tsv", sep="\t", index=False)

    lock_hash_after = sha256(LOCK_PATH)
    primary = comparison_frame.loc[comparison_frame["annotation_level"].eq("EC_L3")].iloc[0]
    scientific_decision = "CONFIRMATORY_ENDPOINT_MET" if bool(primary["confirmatory_release_gate"]) else "CONFIRMATORY_ENDPOINT_NOT_MET"
    checks = {
        "blind_checkpoint_present": required.is_file(),
        "inference_lock_unchanged": lock_hash_before == lock_hash_after,
        "one_primary_prediction_per_query": len(evaluated) == len(cohort) and not evaluated["query_protein_id"].duplicated().any(),
        "truth_complete_for_eligible": bool(evaluated.loc[evaluated["label_eligible"], "truth_label"].notna().all()),
        "frozen_minimums_applied": True,
        "exact_rhea_not_claimed": True,
    }
    summary = {
        "phase": "30C", "status": "PASS" if all(checks.values()) else "FAIL",
        "stage": "locked_retrospective_external_blind_evaluation",
        "scientific_decision": scientific_decision,
        "truth_opened_once": True, "bootstrap_draws": BOOTSTRAPS, "bootstrap_seed": SEED,
        "checks": checks, "endpoints": endpoints.to_dict("records"),
        "incremental": comparison_frame.to_dict("records"),
    }
    (REPORTS / "phase30_external_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if not all(checks.values()):
        raise RuntimeError(f"Phase30 evaluation integrity failure: {checks}")
    (CHECKPOINTS / "CHECKPOINT_30C_EXTERNAL_EVALUATION_PASS").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(endpoints.to_string(index=False))
    print(comparison_frame.to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))
    print("CHECKPOINT_30C_EXTERNAL_EVALUATION_PASS")


if __name__ == "__main__":
    main()
