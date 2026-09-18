#!/usr/bin/env python3
"""Open external truth once after the blind prediction lock and evaluate endpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from phase31_train_hit_evidencejudge import canonical_full, parse_enzyme_transfers


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
PHASE_ID = int(os.environ.get("EVIDENCEJUDGE_EXTERNAL_PHASE", "31"))
if PHASE_ID not in {31, 32}:
    raise RuntimeError(f"Unsupported EvidenceJudge external phase: {PHASE_ID}")
RPHASE = ROOT / f"results/phase{PHASE_ID}"
REPORTS = ROOT / f"reports/phase{PHASE_ID}_external_blind"
CHECKPOINTS = ROOT / "checkpoints"
LOCK_PATH = ROOT / f"models/phase{PHASE_ID}_external_inference_lock.json"
ENZYME = ROOT / "data/external/enzyme_release_2026_06_10/enzyme.dat"
BOOTSTRAPS = 10000
SEED = {31: 20260901, 32: 20260915}[PHASE_ID]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_truth(cohort: pd.DataFrame, transfers: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in cohort.itertuples(index=False):
        original_l3 = str(row.ec_l3) if bool(row.ec_l3_label_eligible) else None
        original_l4 = str(row.ec_l4) if bool(row.ec_l4_label_eligible) else None
        canonical_l4, status = canonical_full(original_l4, transfers) if original_l4 else (None, "NO_L4")
        canonical_l3 = canonical_l4.rsplit(".", 1)[0] if canonical_l4 else original_l3
        rows.append({
            "query_id": str(row.query_id), "external_cluster_id_30": str(row.external_cluster_id_30),
            "annotation_level": "EC_L3", "original_truth_label": original_l3,
            "truth_label": canonical_l3, "label_eligible": canonical_l3 is not None,
            "truth_canonicalization_status": status,
        })
        rows.append({
            "query_id": str(row.query_id), "external_cluster_id_30": str(row.external_cluster_id_30),
            "annotation_level": "EC_L4", "original_truth_label": original_l4,
            "truth_label": canonical_l4, "label_eligible": canonical_l4 is not None,
            "truth_canonicalization_status": status,
        })
    return pd.DataFrame(rows)


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
        size = min(250, BOOTSTRAPS - start)
        sample = values[rng.integers(0, len(values), size=(size, len(values)))]
        estimates[start:start + size] = sample[:, :, 0].sum(1) / sample[:, :, 1].sum(1)
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
        "coverage": n / total if total else 0.0, "correct": successes, "precision": precision,
        "cluster_bootstrap_low": low, "cluster_bootstrap_high": high,
        "clopper_pearson_two_sided_low": cp_low, "qualifies_95": qualifies,
        "safe_coverage_at_95": n / total if qualifies and total else 0.0,
    }


def attach_truth(predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    output = predictions.merge(
        truth,
        left_on=["annotation_level", "query_protein_id", "query_cluster_id_30"],
        right_on=["annotation_level", "query_id", "external_cluster_id_30"],
        how="left", validate="many_to_one",
    )
    output["correct"] = (
        output["label_eligible"].astype(bool)
        & output["candidate_label"].astype(str).eq(output["truth_label"].astype(str))
    )
    output["correct_against_raw_pdb_ec"] = (
        output["label_eligible"].astype(bool)
        & output["candidate_label"].astype(str).eq(output["original_truth_label"].astype(str))
    )
    return output


def paired_coverage_bootstrap(
    evaluated: pd.DataFrame,
    singles: pd.DataFrame,
    truth: pd.DataFrame,
    level: str,
    method: str,
) -> dict[str, float]:
    universe = truth.loc[
        truth["annotation_level"].eq(level) & truth["label_eligible"].astype(bool),
        ["query_id", "external_cluster_id_30"],
    ].drop_duplicates("query_id")
    judge = evaluated.loc[
        evaluated["annotation_level"].eq(level), ["query_protein_id", "accepted"]
    ].rename(columns={"query_protein_id": "query_id", "accepted": "judge_accepted"})
    paired = universe.merge(judge, on="query_id", how="left", validate="one_to_one")
    if method == "NONE_QUALIFIED":
        paired["single_accepted"] = False
    else:
        single = singles.loc[
            singles["annotation_level"].eq(level) & singles["method"].eq(method),
            ["query_protein_id", "accepted"],
        ].rename(columns={"query_protein_id": "query_id", "accepted": "single_accepted"})
        paired = paired.merge(single, on="query_id", how="left", validate="one_to_one")
    if paired["judge_accepted"].isna().any():
        raise RuntimeError(f"Incomplete EvidenceJudge coverage inventory for {level}/{method}")
    # A structure/domain tool can be unavailable for a query; in coverage space
    # that query is an abstention, not a missing member of the paired universe.
    paired["single_accepted"] = paired["single_accepted"].fillna(False)
    paired[["judge_accepted", "single_accepted"]] = paired[[
        "judge_accepted", "single_accepted"
    ]].astype(bool)
    groups = paired.groupby("external_cluster_id_30", observed=True).agg(
        judge_accepted=("judge_accepted", "sum"),
        single_accepted=("single_accepted", "sum"),
        queries=("query_id", "count"),
    )
    values = groups[["judge_accepted", "single_accepted", "queries"]].to_numpy(float)
    point = float((values[:, 0].sum() - values[:, 1].sum()) / values[:, 2].sum())
    seed = int.from_bytes(
        hashlib.sha256(f"{SEED}|PAIRED_COVERAGE|{level}|{method}".encode()).digest()[:8], "big"
    ) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAPS)
    for start in range(0, BOOTSTRAPS, 250):
        size = min(250, BOOTSTRAPS - start)
        sample = values[rng.integers(0, len(values), size=(size, len(values)))]
        estimates[start:start + size] = (
            sample[:, :, 0].sum(1) - sample[:, :, 1].sum(1)
        ) / sample[:, :, 2].sum(1)
    return {
        "paired_coverage_difference": point,
        "paired_cluster_bootstrap_low": float(np.quantile(estimates, 0.025)),
        "paired_cluster_bootstrap_high": float(np.quantile(estimates, 0.975)),
    }


def main() -> None:
    checkpoint = CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}B_BLIND_PREDICTIONS_LOCKED"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if (RPHASE / "external_predictions.parquet").exists():
        raise RuntimeError(f"Phase{PHASE_ID} truth has already been opened")
    lock_hash_before = sha256(LOCK_PATH)
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["truth_opened"]:
        raise RuntimeError(f"Phase{PHASE_ID} inference lock already reports opened truth")
    protocol_lock = None
    if PHASE_ID == 32:
        protocol_lock_path = ROOT / "models/phase32_protocol_lock.json"
        protocol_lock = json.loads(protocol_lock_path.read_text(encoding="utf-8"))
        if sha256(protocol_lock_path) != lock["protocol_lock_sha256"]:
            raise RuntimeError("Phase32 protocol lock changed after blind inference")
        if any(
            not (ROOT / path).is_file() or sha256(ROOT / path) != expected
            for path, expected in protocol_lock["script_sha256"].items()
        ):
            raise RuntimeError("Phase32 frozen script changed before truth opening")
    transfers, _ = parse_enzyme_transfers(ENZYME)

    cohort = pd.read_parquet(RPHASE / "rcsb_strict_blind_cohort.parquet")
    if PHASE_ID == 32 and (
        not cohort["tool_training_homology_independent"].astype(bool).all()
        or not cohort["tool_training_homology_preexcluded"].astype(bool).all()
    ):
        raise RuntimeError("Phase32 cohort is not preprediction tool-training-homology clean")
    truth = build_truth(cohort, transfers)
    totals = truth.loc[truth["label_eligible"]].groupby("annotation_level")["query_id"].nunique().to_dict()
    transferred = truth["truth_canonicalization_status"].eq("TRANSFERRED")
    truth.to_csv(RPHASE / "external_canonicalized_truth_audit.tsv", sep="\t", index=False)

    blind = pd.read_parquet(RPHASE / "external_blind_predictions.parquet")
    evaluated = attach_truth(blind, truth)
    evaluated["evaluation_status"] = f"PHASE{PHASE_ID}_LOCKED_RETROSPECTIVE_EXTERNAL_BLIND"
    evaluated.to_parquet(RPHASE / "external_predictions.parquet", index=False, compression="zstd")
    singles = attach_truth(pd.read_parquet(RPHASE / "external_blind_single_tool_predictions.parquet"), truth)
    singles.to_parquet(RPHASE / "external_single_tool_evaluated_predictions.parquet", index=False, compression="zstd")

    endpoint_rows: list[dict[str, object]] = []
    for level in ("EC_L3", "EC_L4"):
        accepted = evaluated.loc[
            evaluated["annotation_level"].eq(level)
            & evaluated["accepted"].astype(bool)
            & evaluated["label_eligible"].astype(bool)
        ]
        endpoint_rows.append({
            "annotation_level": level, "system": "HIT_ANCHORED_EVIDENCEJUDGE_V2",
            **measure(accepted, int(totals.get(level, 0)), f"PHASE{PHASE_ID}|JUDGE|{level}"),
        })
    single_rows: list[dict[str, object]] = []
    for (level, method), group in singles.groupby(["annotation_level", "method"], observed=True):
        accepted = group.loc[group["accepted"].astype(bool) & group["label_eligible"].astype(bool)]
        metric = measure(accepted, int(totals.get(level, 0)), f"PHASE{PHASE_ID}|SINGLE|{level}|{method}")
        row = {"annotation_level": level, "system": f"SINGLE::{method}", **metric}
        endpoint_rows.append(row)
        single_rows.append({"annotation_level": level, "method": method, **metric})
    endpoints = pd.DataFrame(endpoint_rows)
    single_endpoints = pd.DataFrame(single_rows)
    endpoints.to_csv(RPHASE / "external_operating_points.tsv", sep="\t", index=False)
    single_endpoints.to_csv(RPHASE / "external_single_tool_operating_points.tsv", sep="\t", index=False)

    comparison_rows: list[dict[str, object]] = []
    for level in ("EC_L3", "EC_L4"):
        judge = endpoints.loc[
            endpoints["annotation_level"].eq(level)
            & endpoints["system"].eq("HIT_ANCHORED_EVIDENCEJUDGE_V2")
        ].iloc[0]
        eligible_singles = single_endpoints.loc[
            single_endpoints["annotation_level"].eq(level)
            & single_endpoints["qualifies_95"].astype(bool)
        ].sort_values(["safe_coverage_at_95", "precision"], ascending=False, kind="mergesort")
        best_name = str(eligible_singles.iloc[0]["method"]) if len(eligible_singles) else "NONE_QUALIFIED"
        best_coverage = float(eligible_singles.iloc[0]["safe_coverage_at_95"]) if len(eligible_singles) else 0.0
        increment = float(judge["safe_coverage_at_95"] - best_coverage)
        paired = paired_coverage_bootstrap(evaluated, singles, truth, level, best_name)
        release_gate = bool(judge["qualifies_95"] and increment > 0)
        if PHASE_ID == 32:
            release_gate = release_gate and paired["paired_cluster_bootstrap_low"] > 0
        comparison_rows.append({
            "annotation_level": level, "evidencejudge_system": judge["system"],
            "evidencejudge_safe_coverage_at_95": float(judge["safe_coverage_at_95"]),
            "best_single_tool": best_name, "best_single_safe_coverage_at_95": best_coverage,
            "incremental_safe_coverage": increment,
            **paired,
            "confirmatory_release_gate": release_gate,
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(RPHASE / "external_incremental_safe_coverage.tsv", sep="\t", index=False)

    calibration_rows = []
    for level, group in evaluated.loc[evaluated["label_eligible"]].groupby("annotation_level", observed=True):
        y = group["correct"].astype(int).to_numpy()
        score = group["selection_score"].astype(float).to_numpy()
        calibration_rows.append({
            "annotation_level": level, "queries": len(group),
            "brier_score": float(brier_score_loss(y, score)),
            "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
            "average_precision": float(average_precision_score(y, score)),
        })
    pd.DataFrame(calibration_rows).to_csv(RPHASE / "external_discrimination_calibration.tsv", sep="\t", index=False)
    evaluated.loc[
        evaluated["accepted"].astype(bool) & evaluated["label_eligible"].astype(bool) & ~evaluated["correct"]
    ].to_csv(RPHASE / "external_accepted_errors.tsv", sep="\t", index=False)

    raw_sensitivity = []
    for level in ("EC_L3", "EC_L4"):
        group = evaluated.loc[
            evaluated["annotation_level"].eq(level)
            & evaluated["accepted"].astype(bool)
            & evaluated["label_eligible"].astype(bool)
        ].copy()
        group["correct"] = group["correct_against_raw_pdb_ec"]
        raw_sensitivity.append({
            "annotation_level": level, "system": "HIT_ANCHORED_EVIDENCEJUDGE_V2_RAW_PDB_EC_SENSITIVITY",
            **measure(group, int(totals.get(level, 0)), f"PHASE{PHASE_ID}|RAW|{level}"),
        })
    pd.DataFrame(raw_sensitivity).to_csv(RPHASE / "external_raw_pdb_ec_sensitivity.tsv", sep="\t", index=False)

    lock_hash_after = sha256(LOCK_PATH)
    primary = comparison.loc[comparison["annotation_level"].eq("EC_L3")].iloc[0]
    decision = "CONFIRMATORY_ENDPOINT_MET" if bool(primary["confirmatory_release_gate"]) else "CONFIRMATORY_ENDPOINT_NOT_MET"
    checks = {
        "blind_checkpoint_present": checkpoint.is_file(),
        "inference_lock_unchanged": lock_hash_before == lock_hash_after,
        "one_prediction_per_query_and_level": len(evaluated) == 2 * len(cohort)
        and not evaluated.duplicated(["annotation_level", "query_protein_id"]).any(),
        "truth_complete_for_eligible": bool(evaluated.loc[evaluated["label_eligible"], "truth_label"].notna().all()),
        "canonicalization_resource_matches_lock": sha256(ENZYME) == lock["enzyme_sha256"],
        "ec_l3_power_gate": int(totals.get("EC_L3", 0)) >= 120
        and truth.loc[truth["annotation_level"].eq("EC_L3") & truth["label_eligible"], "external_cluster_id_30"].nunique() >= 50,
        "phase32_tool_training_homology_preexcluded": PHASE_ID != 32 or bool(
            cohort["tool_training_homology_independent"].astype(bool).all()
            and cohort["tool_training_homology_preexcluded"].astype(bool).all()
        ),
        "exact_rhea_not_claimed": True,
    }
    summary = {
        "phase": f"{PHASE_ID}C", "status": "PASS" if all(checks.values()) else "FAIL",
        "stage": "locked_retrospective_external_blind_evaluation",
        "scientific_decision": decision, "truth_opened_once": True,
        "bootstrap_draws": BOOTSTRAPS, "bootstrap_seed": SEED,
        "canonicalized_truth_rows": int(transferred.sum()), "checks": checks,
        "endpoints": endpoints.to_dict("records"), "incremental": comparison.to_dict("records"),
        "calibration": calibration_rows,
    }
    (REPORTS / f"phase{PHASE_ID}_external_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if not all(checks.values()):
        raise RuntimeError(f"Phase{PHASE_ID} evaluation integrity failure: {checks}")
    (CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}C_EXTERNAL_EVALUATION_PASS").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(endpoints.to_string(index=False))
    print(comparison.to_string(index=False))
    print(pd.DataFrame(calibration_rows).to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))
    print(f"CHECKPOINT_{PHASE_ID}C_EXTERNAL_EVALUATION_PASS")


if __name__ == "__main__":
    main()
