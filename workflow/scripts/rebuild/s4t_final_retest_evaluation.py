#!/usr/bin/env python3
"""Apply the frozen rule once to RETEST and report utility, risk and candidate limits."""
import argparse
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


LEVELS = ("EC_L3", "EC_L4", "EXACT_RHEA")
TARGET = {"EC_L3": "observed_same_ec_l3", "EC_L4": "observed_same_ec_l4", "EXACT_RHEA": "observed_same_exact_rhea"}
REFERENCE = {"EC_L3": "ec_l3", "EC_L4": "ec_l4", "EXACT_RHEA": "canonical_rhea"}
KEY = ["query_protein_id", "reference_protein_id", "reference_activity_id"]


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def emit(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle: json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")


def add_candidate_ranks(frame):
    candidates = frame.groupby(KEY[:2], sort=False, as_index=False).agg(rrf60_score=("rrf60_score", "max"), mmseqs_rank=("mmseqs_rank", "min"), foldseek_rank=("foldseek_rank", "min"))
    candidates["rrf60_score"] = candidates["rrf60_score"].fillna(-np.inf)
    for strategy, column, ascending in (("rrf", "rrf60_score", False), ("sequence", "mmseqs_rank", True), ("structure", "foldseek_rank", True)):
        work = candidates.copy() if strategy == "rrf" else candidates.loc[candidates[column].notna()].copy()
        work = work.sort_values(["query_protein_id", column, "reference_protein_id"], ascending=[True, ascending, True], kind="mergesort"); work[strategy + "_candidate_rank"] = work.groupby("query_protein_id", sort=False).cumcount() + 1
        frame = frame.merge(work[KEY[:2] + [strategy + "_candidate_rank"]], on=KEY[:2], how="left", validate="m:1")
    return frame


def model_top(frame, level, budget):
    pool = frame if budget < 0 else frame.loc[frame["rrf_candidate_rank"].le(budget)]
    pool = pool.loc[pool[REFERENCE[level]].notna()]
    idx = pool.groupby("query_protein_id", sort=False)["calibrated_" + level].idxmax()
    return pool.loc[idx].sort_values("query_protein_id").reset_index(drop=True)


def deterministic_baseline_outcomes(frame, strategy, level):
    rank = strategy + "_candidate_rank"
    pool = frame.loc[frame[rank].eq(1) & frame[REFERENCE[level]].notna()]
    grouped = pool.groupby("query_protein_id", sort=False)[TARGET[level]]
    result = grouped.agg(lambda values: bool(values.fillna(False).astype(bool).any())).rename("correct").to_frame()
    if level == "EXACT_RHEA":
        result["evaluable"] = grouped.agg(lambda values: bool(values.notna().any()))
    else:
        result["evaluable"] = True
    return result.reset_index()


def cluster_bootstrap(final, replicates=2000, seed=20260819):
    accepted = final["final_resolution"].ne("ABSTAIN").to_numpy(np.int64)
    evaluable = (final["final_resolution"].ne("ABSTAIN") & final["outcome_evaluable"].astype(bool)).to_numpy(np.int64)
    correct = (final["documented_concordant"].fillna(False).astype(bool) & evaluable.astype(bool)).to_numpy(np.int64)
    counts = pd.DataFrame({"component_id": final["query_component_id"], "queries": 1, "accepted": accepted, "evaluable_accepted": evaluable, "correct": correct}).groupby("component_id", sort=True, as_index=False).sum()
    require(len(counts) >= 2 and counts["queries"].sum() == len(final), "RETEST component census")
    values = counts[["queries", "accepted", "evaluable_accepted", "correct"]].to_numpy(np.int64)
    rng = np.random.default_rng(seed); rows = []
    for replicate in range(replicates):
        sampled = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        queries, accepted_n, evaluable_n, correct_n = (int(value) for value in sampled)
        rows.append({
            "replicate": replicate, "queries": queries,
            "coverage": accepted_n / queries,
            "useful_correct_coverage": correct_n / queries,
            "precision_among_evaluable_accepted": correct_n / evaluable_n if evaluable_n else np.nan,
        })
    samples = pd.DataFrame(rows)
    intervals = {}
    for metric in ("coverage", "useful_correct_coverage", "precision_among_evaluable_accepted"):
        values = samples[metric].dropna().to_numpy(float)
        intervals[metric] = {"lower_95": float(np.quantile(values, 0.025)), "upper_95": float(np.quantile(values, 0.975)), "finite_replicates": int(len(values))}
    return counts, samples, {"method": "30_PERCENT_SEQUENCE_COMPONENT_CLUSTER_BOOTSTRAP", "replicates": replicates, "seed": seed, "components": int(len(counts)), "intervals": intervals}


def execute(root, feature_root, label_root, prediction_root, calibration_root, output):
    require(output.parent.is_dir() and not output.exists(), "RETEST output")
    checkpoint_path = calibration_root / "S4S_CALIBRATION_RULE_PASS.json"; checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")); rule = json.loads((calibration_root / "release_rule.json").read_text(encoding="utf-8")); calibrators = json.loads((calibration_root / "calibrators.json").read_text(encoding="utf-8"))
    require(checkpoint["status"] == "PASS_S4S_CALIBRATION_AND_RULE_FREEZE" and checkpoint["release_rule_sha256"] == sha(calibration_root / "release_rule.json") and not rule["retest_truth_read"], "pre-RETEST freeze")
    output.mkdir(exist_ok=False); emit(output / "reservation.json", {"status": "ONE_S4T_FINAL_RETEST_RESERVED", "rule_sha256": sha(calibration_root / "release_rule.json"), "automatic_retry": False})
    state, error = "FAIL_CLOSED", None
    try:
        metadata = pd.read_parquet(feature_root / "role_RETEST/producer/metadata_RETEST.parquet")
        labels = pd.read_parquet(label_root / "role_RETEST/labels_RETEST.parquet")
        require(len(metadata) == len(labels) and all(np.array_equal(metadata[name].to_numpy(), labels[name].to_numpy()) for name in KEY), "RETEST key binding")
        ledger_path = root / "revisions/s4_20260910/truth_free_ledger_attempt_01/producer/protein_role_ledger.parquet"
        ledger = pd.read_parquet(ledger_path, columns=["protein_id", "role", "component_id"])
        all_retest_queries = sorted(ledger.loc[ledger["role"].eq("RETEST"), "protein_id"].drop_duplicates().tolist())
        require(all_retest_queries, "empty RETEST query universe")
        all_retest_set = set(all_retest_queries)
        require(set(metadata["query_protein_id"]).issubset(all_retest_set), "candidate query outside RETEST universe")
        raw = np.load(prediction_root / "role_RETEST/raw_predictions_RETEST.npy"); require(raw.shape == (len(metadata), 3), "RETEST predictions")
        frame = metadata.copy()
        for name in labels.columns:
            if name not in KEY: frame[name] = labels[name].to_numpy()
        for index, level in enumerate(LEVELS):
            spec = calibrators["levels"][level]; frame["raw_" + level] = raw[:, index]; frame["calibrated_" + level] = np.interp(raw[:, index], np.asarray(spec["x_thresholds"]), np.asarray(spec["y_thresholds"])).astype(np.float32)
        frame = add_candidate_ranks(frame); queries_with_candidates = frame["query_protein_id"].nunique(); require(queries_with_candidates > 0, "RETEST candidate queries")
        total_queries = len(all_retest_queries)
        rows = []
        primary_top = {}
        for budget in (10, 25, 50, -1):
            for level in LEVELS:
                top = model_top(frame, level, budget); evaluable = top[TARGET[level]].notna()
                if level == "EXACT_RHEA": evaluable &= top["exact_rhea_outcome_evaluable"].astype(bool)
                usable = top.loc[evaluable]; threshold = float(rule["thresholds"][level]); accepted = usable["calibrated_" + level].ge(threshold); correct = usable[TARGET[level]].astype(bool)
                budget_pool = frame if budget < 0 else frame.loc[frame["rrf_candidate_rank"].le(budget)]
                budget_pool = budget_pool.loc[budget_pool[REFERENCE[level]].notna()]
                oracle_group = budget_pool.groupby("query_protein_id")[TARGET[level]]
                oracle = oracle_group.agg(lambda values: bool(values.fillna(False).astype(bool).any()))
                oracle_evaluable = oracle_group.agg(lambda values: bool(values.notna().any())) if level == "EXACT_RHEA" else pd.Series(True, index=oracle.index)
                oracle_usable = oracle.loc[oracle_evaluable]
                accepted_n = int(accepted.sum()); accepted_correct_n = int((accepted & correct).sum())
                rows.append({"analysis": "MODEL_RERANK", "budget": "ALL" if budget < 0 else budget, "level": level, "queries_total": total_queries, "queries_with_candidate": int(top["query_protein_id"].nunique()), "queries_evaluable": len(usable), "accepted": accepted_n, "accepted_correct": accepted_correct_n, "coverage_all_queries": float(accepted_n / total_queries), "coverage_among_evaluable": float(accepted.mean()) if len(usable) else 0.0, "precision_among_accepted": float(correct[accepted].mean()) if accepted.any() else None, "useful_correct_coverage_all_queries": float(accepted_correct_n / total_queries), "top1_accuracy_among_evaluable": float(correct.mean()) if len(usable) else None, "oracle_candidate_coverage_all_queries": float(oracle.sum() / total_queries), "oracle_candidate_coverage_among_evaluable": float(oracle_usable.mean()) if len(oracle_usable) else None, "threshold": threshold})
                if budget == 50: primary_top[level] = top.set_index("query_protein_id")
        for strategy in ("rrf", "sequence", "structure"):
            for level in LEVELS:
                baseline = deterministic_baseline_outcomes(frame, strategy, level)
                usable = baseline.loc[baseline["evaluable"]]; correct = usable["correct"].astype(bool); correct_n = int(correct.sum())
                rows.append({"analysis": strategy.upper() + "_TOP1", "budget": 1, "level": level, "queries_total": total_queries, "queries_with_candidate": len(baseline), "queries_evaluable": len(usable), "accepted": len(usable), "accepted_correct": correct_n, "coverage_all_queries": float(len(usable) / total_queries), "coverage_among_evaluable": 1.0 if len(usable) else 0.0, "precision_among_accepted": float(correct.mean()) if len(usable) else None, "useful_correct_coverage_all_queries": float(correct_n / total_queries), "top1_accuracy_among_evaluable": float(correct.mean()) if len(usable) else None, "oracle_candidate_coverage_all_queries": None, "oracle_candidate_coverage_among_evaluable": None, "threshold": None})
        metrics = pd.DataFrame(rows); metrics.to_csv(output / "retest_candidate_and_utility_metrics.tsv", sep="\t", index=False)
        final_rows = []
        for query in all_retest_queries:
            chosen_level = "ABSTAIN"; chosen = None
            ec3 = primary_top["EC_L3"].loc[query] if query in primary_top["EC_L3"].index else None
            rhea = primary_top["EXACT_RHEA"].loc[query] if query in primary_top["EXACT_RHEA"].index else None
            ec4 = primary_top["EC_L4"].loc[query] if query in primary_top["EC_L4"].index else None
            ec3_ok = ec3 is not None and float(ec3["calibrated_EC_L3"]) >= float(rule["thresholds"]["EC_L3"])
            rhea_ok = rhea is not None and float(rhea["calibrated_EC_L3"]) >= float(rule["thresholds"]["EC_L3"]) and float(rhea["calibrated_EXACT_RHEA"]) >= float(rule["thresholds"]["EXACT_RHEA"])
            ec4_ok = ec4 is not None and float(ec4["calibrated_EC_L3"]) >= float(rule["thresholds"]["EC_L3"]) and float(ec4["calibrated_EC_L4"]) >= float(rule["thresholds"]["EC_L4"])
            if rhea_ok: chosen_level, chosen = "EXACT_RHEA", rhea
            elif ec4_ok: chosen_level, chosen = "EC_L4", ec4
            elif ec3_ok: chosen_level, chosen = "EC_L3", ec3
            if chosen is None:
                final_rows.append({"query_protein_id": query, "final_resolution": "ABSTAIN", "predicted_label": None, "reference_protein_id": None, "reference_activity_id": None, "probability": None, "outcome_evaluable": False, "documented_concordant": None})
            else:
                evaluable = chosen_level != "EXACT_RHEA" or bool(chosen["exact_rhea_outcome_evaluable"])
                concordant = bool(chosen[TARGET[chosen_level]]) if evaluable and pd.notna(chosen[TARGET[chosen_level]]) else None
                final_rows.append({"query_protein_id": query, "final_resolution": chosen_level, "predicted_label": chosen[REFERENCE[chosen_level]], "reference_protein_id": chosen["reference_protein_id"], "reference_activity_id": chosen["reference_activity_id"], "probability": float(chosen["calibrated_" + chosen_level]), "outcome_evaluable": evaluable, "documented_concordant": concordant})
        final = pd.DataFrame(final_rows)
        component_map = ledger.loc[ledger["role"].eq("RETEST"), ["protein_id", "component_id"]].drop_duplicates().set_index("protein_id")["component_id"]
        final.insert(1, "query_component_id", final["query_protein_id"].map(component_map))
        require(final["query_component_id"].notna().all(), "RETEST component coverage")
        final.to_parquet(output / "retest_end_to_end_predictions.parquet", index=False, compression="zstd")
        accepted = final["final_resolution"].ne("ABSTAIN"); evaluable = accepted & final["outcome_evaluable"].astype(bool); correct = final["documented_concordant"].fillna(False).astype(bool)
        component_counts, bootstrap, robustness = cluster_bootstrap(final)
        component_counts.to_csv(output / "retest_component_census.tsv", sep="\t", index=False)
        bootstrap.to_csv(output / "retest_cluster_bootstrap.tsv", sep="\t", index=False)
        emit(output / "cluster_robustness.json", robustness)
        endpoint = {"retest_queries": len(final), "retest_sequence_components": robustness["components"], "queries_with_any_candidate": int(queries_with_candidates), "queries_without_candidate": int(len(final) - queries_with_candidates), "accepted_queries": int(accepted.sum()), "accepted_evaluable_queries": int(evaluable.sum()), "accepted_correct_queries": int((evaluable & correct).sum()), "coverage": float(accepted.mean()), "precision_among_evaluable_accepted": float(correct[evaluable].mean()) if evaluable.any() else None, "unevaluable_accepted_queries": int((accepted & ~evaluable).sum()), "useful_correct_coverage": float((evaluable & correct).sum() / len(final)), "cluster_bootstrap_95_intervals": robustness["intervals"], "resolution_counts": final["final_resolution"].value_counts().to_dict()}
        summary = {"status": "PASS_S4T_FINAL_RETEST_EVALUATION", "rule_sha256": sha(calibration_root / "release_rule.json"), "calibrator_sha256": sha(calibration_root / "calibrators.json"), "retest_labels_joined_after_rule_freeze": True, "endpoint": endpoint, "candidate_metric_rows": len(metrics), "prediction_sha256": sha(output / "retest_end_to_end_predictions.parquet"), "cluster_robustness_sha256": sha(output / "cluster_robustness.json")}
        emit(output / "summary.json", summary); emit(output / "S4T_FINAL_RETEST_PASS.json", {"status": "PASS_S4T_FINAL_RETEST_EVALUATION", "summary_sha256": sha(output / "summary.json"), "rule_sha256": summary["rule_sha256"]}); state = summary["status"]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    emit(output / "terminal.json", {"status": state, "error": error, "automatic_retry": False})
    if error: raise RuntimeError(error["message"])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--feature-root", type=Path, required=True); parser.add_argument("--label-root", type=Path, required=True); parser.add_argument("--prediction-root", type=Path, required=True); parser.add_argument("--calibration-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); execute(args.root.resolve(), args.feature_root.resolve(), args.label_root.resolve(), args.prediction_root.resolve(), args.calibration_root.resolve(), args.output.resolve())


if __name__ == "__main__": main()
