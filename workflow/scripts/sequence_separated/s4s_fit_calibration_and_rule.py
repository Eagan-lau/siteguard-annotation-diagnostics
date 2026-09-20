#!/usr/bin/env python3
"""Fit CAL_FIT calibrators and freeze a useful-coverage rule on CAL_RULE."""
import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


LEVELS = ("EC_L3", "EC_L4", "EXACT_RHEA")
TARGET = {"EC_L3": "observed_same_ec_l3", "EC_L4": "observed_same_ec_l4", "EXACT_RHEA": "observed_same_exact_rhea"}
REFERENCE = {"EC_L3": "ec_l3", "EC_L4": "ec_l4", "EXACT_RHEA": "canonical_rhea"}
KEY = ["query_protein_id", "reference_protein_id", "reference_activity_id"]
BUDGETS = (10, 25, 50, -1)


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def emit(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle: json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")


def wilson_lower(successes, trials, z=1.959963984540054):
    if trials <= 0: return 0.0
    p = successes / trials; denominator = 1 + z * z / trials
    return (p + z * z / (2 * trials) - z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))) / denominator


def safe_threshold(scores, outcomes, desired, minimum=100):
    if len(scores) < minimum:
        return {"threshold": 1.000001, "accepted": 0, "accepted_correct": 0, "precision": None, "wilson_lower": None, "coverage": 0.0, "correct_coverage": 0.0}
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 1001)))
    selected = None
    for threshold in candidates:
        mask = scores >= threshold; count = int(mask.sum())
        if count < minimum: continue
        success = int(outcomes[mask].sum()); lower = wilson_lower(success, count)
        if lower >= desired:
            selected = {"threshold": float(threshold), "accepted": count, "accepted_correct": success, "precision": success / count, "wilson_lower": lower, "coverage": count / len(scores), "correct_coverage": success / len(scores)}
            break
    return selected or {"threshold": 1.000001, "accepted": 0, "accepted_correct": 0, "precision": None, "wilson_lower": None, "coverage": 0.0, "correct_coverage": 0.0}


def load_role(feature_root, label_root, prediction_root, role):
    metadata = pd.read_parquet(feature_root / f"role_{role}/producer/metadata_{role}.parquet")
    labels = pd.read_parquet(label_root / f"role_{role}/labels_{role}.parquet")
    require(len(metadata) == len(labels) and all(np.array_equal(metadata[name].to_numpy(), labels[name].to_numpy()) for name in KEY), role + " key binding")
    predictions = np.load(prediction_root / f"role_{role}/raw_predictions_{role}.npy")
    require(predictions.shape == (len(metadata), 3), role + " prediction shape")
    frame = metadata.copy()
    for index, level in enumerate(LEVELS): frame["raw_" + level] = predictions[:, index]
    for name in labels.columns:
        if name not in KEY: frame[name] = labels[name].to_numpy()
    return frame


def calibrate(frame, level):
    target = TARGET[level]; mask = frame[target].notna().to_numpy(copy=True)
    if level == "EXACT_RHEA": mask &= frame["exact_rhea_outcome_evaluable"].astype(bool).to_numpy()
    y = frame.loc[mask, target].astype(bool).to_numpy(np.int8); x = frame.loc[mask, "raw_" + level].to_numpy(float)
    require(len(y) >= 100 and len(np.unique(y)) == 2, "calibration target support " + level)
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, y)
    return model, mask


def apply(model, values):
    return model.predict(values.astype(float)).astype(np.float32)


def add_rrf_budget_rank(frame):
    candidate = frame.groupby(KEY[:2], sort=False, as_index=False).agg(rrf60_score=("rrf60_score", "max"), mmseqs_rank=("mmseqs_rank", "min"), foldseek_rank=("foldseek_rank", "min"))
    candidate["rrf60_score"] = candidate["rrf60_score"].fillna(-np.inf)
    candidate = candidate.sort_values(["query_protein_id", "rrf60_score", "reference_protein_id"], ascending=[True, False, True], kind="mergesort")
    candidate["rrf_budget_rank"] = candidate.groupby("query_protein_id", sort=False).cumcount() + 1
    return frame.merge(candidate[KEY[:2] + ["rrf_budget_rank"]], on=KEY[:2], how="left", validate="m:1")


def top_rows(frame, level, budget):
    pool = frame if budget < 0 else frame.loc[frame["rrf_budget_rank"].le(budget)]
    pool = pool.loc[pool[REFERENCE[level]].notna()]
    require(len(pool) > 0, "empty candidate budget")
    indices = pool.groupby("query_protein_id", sort=False)["calibrated_" + level].idxmax()
    return pool.loc[indices].sort_values("query_protein_id").reset_index(drop=True)


def execute(root, feature_root, label_root, prediction_root, output):
    require(output.parent.is_dir() and not output.exists(), "calibration output")
    output.mkdir(exist_ok=False); emit(output / "reservation.json", {"status": "ONE_S4S_CALIBRATION_RULE_RESERVED", "retest_truth_read": False, "automatic_retry": False})
    state, error = "FAIL_CLOSED", None
    try:
        fit = load_role(feature_root, label_root, prediction_root, "CAL_FIT")
        rule = load_role(feature_root, label_root, prediction_root, "CAL_RULE")
        calibrators, fit_metrics = {}, []
        for level in LEVELS:
            model, mask = calibrate(fit, level); fit["calibrated_" + level] = apply(model, fit["raw_" + level].to_numpy()); rule["calibrated_" + level] = apply(model, rule["raw_" + level].to_numpy())
            calibrators[level] = {"x_thresholds": [float(v) for v in model.X_thresholds_], "y_thresholds": [float(v) for v in model.y_thresholds_], "fit_rows": int(mask.sum()), "fit_positives": int(fit.loc[mask, TARGET[level]].astype(bool).sum())}
            fit_metrics.append({"level": level, "rows": int(mask.sum()), "brier_raw": float(np.mean((fit.loc[mask, "raw_" + level].to_numpy() - fit.loc[mask, TARGET[level]].astype(float).to_numpy()) ** 2)), "brier_isotonic": float(np.mean((fit.loc[mask, "calibrated_" + level].to_numpy() - fit.loc[mask, TARGET[level]].astype(float).to_numpy()) ** 2))})
        emit(output / "calibrators.json", {"status": "FROZEN_CAL_FIT_ISOTONIC_CALIBRATORS", "levels": calibrators, "retest_truth_read": False})
        rule = add_rrf_budget_rank(rule)
        ledger = pd.read_parquet(root / "revisions/s4_20260910/truth_free_ledger_attempt_01/producer/protein_role_ledger.parquet", columns=["protein_id", "role"])
        cal_rule_queries = int(ledger.loc[ledger["role"].eq("CAL_RULE"), "protein_id"].nunique())
        require(cal_rule_queries >= rule["query_protein_id"].nunique() > 0, "CAL_RULE query universe")
        records, primary_thresholds = [], {}
        for budget in BUDGETS:
            top_by_level = {level: top_rows(rule, level, budget) for level in LEVELS}
            for desired in (0.80, 0.90, 0.95):
                policy_thresholds = {}
                for level in LEVELS:
                    top = top_by_level[level]
                    evaluable = top[TARGET[level]].notna().to_numpy(copy=True)
                    if level == "EXACT_RHEA": evaluable &= top["exact_rhea_outcome_evaluable"].astype(bool).to_numpy()
                    hierarchy_eligible = evaluable.copy()
                    if level != "EC_L3":
                        hierarchy_eligible &= top["calibrated_EC_L3"].ge(policy_thresholds["EC_L3"]).to_numpy()
                    scores = top.loc[hierarchy_eligible, "calibrated_" + level].to_numpy(float)
                    outcomes = top.loc[hierarchy_eligible, TARGET[level]].astype(bool).to_numpy()
                    result = safe_threshold(scores, outcomes, desired)
                    policy_thresholds[level] = result["threshold"]
                    records.append({
                        "budget": "ALL" if budget < 0 else budget, "level": level,
                        "target_precision_lower_bound": desired, "queries_total": cal_rule_queries,
                        "queries_with_candidate": int(top["query_protein_id"].nunique()),
                        "queries_evaluable": int(evaluable.sum()),
                        "hierarchy_eligible_evaluable": int(hierarchy_eligible.sum()),
                        "ancestor_gate": "NONE" if level == "EC_L3" else "SAME_CANDIDATE_EC_L3",
                        **result,
                        "accepted_coverage_all_queries": float(result["accepted"] / cal_rule_queries),
                        "accepted_correct_coverage_all_queries": float(result["accepted_correct"] / cal_rule_queries),
                    })
                if budget == 50 and desired == 0.80:
                    primary_thresholds.update(policy_thresholds)
        require(set(primary_thresholds) == set(LEVELS), "primary threshold coverage")
        pd.DataFrame(records).to_csv(output / "cal_rule_risk_coverage.tsv", sep="\t", index=False)
        rule_contract = {
            "status": "FROZEN_S4S_USEFUL_COVERAGE_RULE", "candidate_strategy": "RRF_UNION_TOP_50_THEN_MLP_RERANK",
            "scoring_budget_reference_proteins_per_query": 50, "thresholds": primary_thresholds,
            "primary_precision_lower_bound": 0.80, "secondary_precision_targets": [0.90, 0.95],
            "selection_objective": "maximize documented-concordant accepted-query coverage subject to Wilson lower precision bound",
            "hierarchy": "EXACT_RHEA and EC_L4 each require SAME-CANDIDATE EC_L3 acceptance; otherwise EC_L3 or abstain",
            "calibrator_sha256": sha(output / "calibrators.json"), "cal_fit_role_only": True,
            "cal_rule_role_only": True, "retest_truth_read": False,
        }
        emit(output / "release_rule.json", rule_contract)
        summary = {"status": "PASS_S4S_CALIBRATION_AND_RULE_FREEZE", "calibration_metrics": fit_metrics, "primary_thresholds": primary_thresholds, "calibration_rows": len(fit), "rule_rows": len(rule), "retest_truth_read": False, "release_rule_sha256": sha(output / "release_rule.json")}
        emit(output / "summary.json", summary); emit(output / "S4S_CALIBRATION_RULE_PASS.json", {"status": "PASS_S4S_CALIBRATION_AND_RULE_FREEZE", "summary_sha256": sha(output / "summary.json"), "release_rule_sha256": sha(output / "release_rule.json")}); state = summary["status"]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    emit(output / "terminal.json", {"status": state, "error": error, "retest_truth_read": False, "automatic_retry": False})
    if error: raise RuntimeError(error["message"])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--feature-root", type=Path, required=True); parser.add_argument("--label-root", type=Path, required=True); parser.add_argument("--prediction-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); execute(args.root.resolve(), args.feature_root.resolve(), args.label_root.resolve(), args.prediction_root.resolve(), args.output.resolve())


if __name__ == "__main__": main()
