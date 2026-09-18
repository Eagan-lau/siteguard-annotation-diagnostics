#!/usr/bin/env python3
"""Screen domain-robust candidate-level judges using Phase 26/28 as development domains."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
R29 = ROOT / "results/phase29"
REPORTS = ROOT / "reports/phase29_development"
SEED = 20260829
LEVELS = ("EC_L3", "EC_L4")


def partition(cluster: object) -> str:
    value = int.from_bytes(hashlib.sha256(f"{SEED}|{cluster}".encode()).digest()[:8], "big") % 100
    return "FIT" if value < 60 else "CAL" if value < 80 else "SELECT"


def external_candidates(phase: int) -> pd.DataFrame:
    proposals = pd.read_parquet(ROOT / f"results/phase{phase}/external_errorjudge_proposals_blind.parquet")
    keys = ["annotation_level", "query_protein_id", "query_cluster_id_30", "candidate_label"]
    candidates = proposals.sort_values(keys + ["proposal_method"], kind="mergesort").drop_duplicates(keys).copy()
    if phase == 26:
        truth = pd.read_parquet(ROOT / "results/phase26/sabio_strict_blind_cohort.parquet")[[
            "uniprot_accession", "ec_l3", "ec_l4", "ec_l3_label_eligible", "ec_l4_label_eligible",
        ]].rename(columns={"uniprot_accession": "query_protein_id"})
    else:
        truth = pd.read_parquet(ROOT / "results/phase28/rcsb_strict_blind_cohort.parquet")[[
            "query_id", "ec_l3", "ec_l4", "ec_l3_label_eligible", "ec_l4_label_eligible",
        ]].rename(columns={"query_id": "query_protein_id"})
    candidates = candidates.merge(truth, on="query_protein_id", how="left", validate="many_to_one")
    candidates["eligible"] = np.where(
        candidates["annotation_level"].eq("EC_L3"), candidates["ec_l3_label_eligible"],
        np.where(candidates["annotation_level"].eq("EC_L4"), candidates["ec_l4_label_eligible"], False),
    ).astype(bool)
    candidates["truth_label"] = np.where(
        candidates["annotation_level"].eq("EC_L3"), candidates["ec_l3"], candidates["ec_l4"],
    )
    candidates = candidates.loc[candidates["annotation_level"].isin(LEVELS) & candidates["eligible"]].copy()
    candidates["correct"] = candidates["candidate_label"].astype(str).eq(candidates["truth_label"].astype(str))
    candidates["source_dataset"] = f"PHASE{phase}"
    candidates["development_partition"] = candidates["query_cluster_id_30"].map(partition)
    candidates["query_weight"] = 1.0 / candidates.groupby(
        ["annotation_level", "query_protein_id"], observed=True
    )["candidate_label"].transform("nunique")
    return candidates


def internal_candidates() -> pd.DataFrame:
    proposals = pd.read_parquet(ROOT / "results/phase25/errorjudge_development_proposals.parquet")
    keys = ["annotation_level", "query_protein_id", "query_cluster_id_30", "candidate_label"]
    candidates = proposals.sort_values(keys + ["proposal_method"], kind="mergesort").drop_duplicates(keys).copy()
    candidates = candidates.loc[candidates["annotation_level"].isin(LEVELS)].copy()
    candidates["source_dataset"] = "INTERNAL"
    return candidates


def domain_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = frame["query_weight"].astype(float).to_numpy()
    output = np.zeros(len(frame), dtype=float)
    for source, positions in frame.groupby("source_dataset", observed=True).indices.items():
        local = weights[positions]
        output[positions] = local / max(local.sum(), 1e-12)
    output *= len(frame) / max(output.sum(), 1e-12)
    return output


def candidate_winners(frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    output = frame[["source_dataset", "annotation_level", "query_protein_id", "query_cluster_id_30", "candidate_label", "correct"]].copy()
    output["probability"] = probabilities
    output.sort_values(
        ["source_dataset", "annotation_level", "query_protein_id", "probability", "candidate_label"],
        ascending=[True, True, True, False, True], kind="mergesort", inplace=True,
    )
    return output.drop_duplicates(["source_dataset", "annotation_level", "query_protein_id"])


def bootstrap_low(frame: pd.DataFrame, context: str, draws: int = 1000) -> float:
    if frame.empty:
        return float("nan")
    grouped = frame.groupby("query_cluster_id_30", observed=True)["correct"].agg(["sum", "count"])
    values = grouped[["sum", "count"]].to_numpy(float)
    if len(values) == 1:
        return float(values[0, 0] / values[0, 1])
    seed = int.from_bytes(hashlib.sha256(f"{SEED}|{context}".encode()).digest()[:8], "big") % (2**32 - 1)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for start in range(0, draws, 200):
        size = min(200, draws - start)
        sample = values[rng.integers(0, len(values), size=(size, len(values)))]
        estimates[start:start + size] = sample[:, :, 0].sum(1) / sample[:, :, 1].sum(1)
    return float(np.quantile(estimates, 0.025))


def select_threshold(winners: pd.DataFrame, context: str) -> dict[str, object]:
    scores = np.unique(np.quantile(winners["probability"], np.linspace(0, 1, 121)))
    rows = []
    for threshold in scores:
        accepted = winners.loc[winners["probability"].ge(threshold)]
        if len(accepted) < 50 or accepted["query_cluster_id_30"].nunique() < 20:
            continue
        precision = float(accepted["correct"].mean())
        if precision < 0.95:
            continue
        low = bootstrap_low(accepted, f"{context}|{threshold:.12g}")
        external_guard = True
        for source, group in accepted.loc[accepted["source_dataset"].ne("INTERNAL")].groupby("source_dataset", observed=True):
            if len(group) >= 8 and float(group["correct"].mean()) < 0.90:
                external_guard = False
        if low >= 0.95 and external_guard:
            rows.append({
                "threshold": float(threshold), "accepted": len(accepted),
                "clusters": int(accepted["query_cluster_id_30"].nunique()),
                "precision": precision, "ci_low": low,
            })
    if not rows:
        return {"qualified": False, "threshold": float("inf"), "accepted": 0, "precision": float("nan"), "ci_low": float("nan")}
    return {"qualified": True, **max(rows, key=lambda row: (row["accepted"], row["precision"], -row["threshold"]))}


def evaluate(winners: pd.DataFrame, threshold: float, context: str) -> dict[str, object]:
    accepted = winners.loc[winners["probability"].ge(threshold)].copy()
    total = winners["query_protein_id"].nunique()
    return {
        "queries_total": total, "accepted": len(accepted),
        "clusters": int(accepted["query_cluster_id_30"].nunique()),
        "coverage": len(accepted) / total if total else 0.0,
        "precision": float(accepted["correct"].mean()) if len(accepted) else float("nan"),
        "ci_low": bootstrap_low(accepted, context) if len(accepted) else float("nan"),
    }


def main() -> None:
    R29.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    schema = json.loads((ROOT / "results/phase24/evidencejudge_feature_schema.json").read_text())
    features = list(schema["feature_columns"]) + [
        "log1p_length", "pfam_seen_in_fit", "pfam_clan_seen_in_fit", "cath_seen_in_fit",
        "log1p_fit_pfam_frequency", "log1p_fit_pfam_clan_frequency", "log1p_fit_cath_frequency",
    ]
    internal = internal_candidates()
    ext26 = external_candidates(26)
    ext28 = external_candidates(28)
    all_data = pd.concat([internal, ext26, ext28], ignore_index=True, sort=False)
    all_data[features] = all_data[features].apply(pd.to_numeric, errors="coerce")

    configs = [
        ("LGB_L7_M100", 7, 100, 0.8),
        ("LGB_L15_M100", 15, 100, 0.8),
        ("LGB_L15_M300", 15, 300, 0.8),
        ("LGB_L31_M300", 31, 300, 0.7),
    ]
    rows = []
    for level in LEVELS:
        level_data = all_data.loc[all_data["annotation_level"].eq(level)].copy()
        for heldout in ("PHASE26", "PHASE28"):
            other = "PHASE28" if heldout == "PHASE26" else "PHASE26"
            fit = level_data.loc[
                (level_data["source_dataset"].eq("INTERNAL") & level_data["development_partition"].eq("FIT"))
                | (level_data["source_dataset"].eq(other) & level_data["development_partition"].eq("FIT"))
            ]
            cal = level_data.loc[
                (level_data["source_dataset"].eq("INTERNAL") & level_data["development_partition"].eq("CAL"))
                | (level_data["source_dataset"].eq(other) & level_data["development_partition"].eq("CAL"))
            ]
            select = level_data.loc[
                (level_data["source_dataset"].eq("INTERNAL") & level_data["development_partition"].eq("SELECT"))
                | (level_data["source_dataset"].eq(other) & level_data["development_partition"].eq("SELECT"))
            ]
            test = level_data.loc[level_data["source_dataset"].eq(heldout)]
            for name, leaves, min_leaf, fraction in configs:
                model = lgb.LGBMClassifier(
                    objective="binary", n_estimators=350, learning_rate=0.03,
                    num_leaves=leaves, max_depth=-1, min_child_samples=min_leaf,
                    subsample=0.85, colsample_bytree=fraction, reg_lambda=3.0, reg_alpha=0.2,
                    random_state=SEED, n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", "12")), verbosity=-1,
                )
                model.fit(fit[features], fit["correct"].astype(int), sample_weight=domain_balanced_weights(fit))
                raw_cal = model.predict_proba(cal[features])[:, 1]
                calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                calibrator.fit(raw_cal, cal["correct"].astype(int), sample_weight=domain_balanced_weights(cal))
                select_prob = calibrator.predict(model.predict_proba(select[features])[:, 1])
                select_winners = candidate_winners(select, select_prob)
                lock = select_threshold(select_winners, f"{level}|{heldout}|{name}|SELECT")
                test_prob = calibrator.predict(model.predict_proba(test[features])[:, 1])
                test_winners = candidate_winners(test, test_prob)
                metric = evaluate(test_winners, float(lock["threshold"]), f"{level}|{heldout}|{name}|TEST")
                rows.append({
                    "annotation_level": level, "heldout_domain": heldout, "model": name,
                    "lock_qualified": lock["qualified"], "threshold": lock["threshold"],
                    "select_accepted": lock["accepted"], "select_precision": lock["precision"],
                    "select_ci_low": lock["ci_low"], **{f"heldout_{key}": value for key, value in metric.items()},
                })
                print(json.dumps(rows[-1], default=str), flush=True)
    screen = pd.DataFrame(rows)
    screen.to_csv(R29 / "cross_domain_model_screen.tsv", sep="\t", index=False)
    summary = {
        "phase": "29A_DEVELOPMENT_SCREEN", "status": "PASS", "models": [row[0] for row in configs],
        "feature_count": len(features), "internal_candidate_rows": len(internal),
        "phase26_candidate_rows": len(ext26), "phase28_candidate_rows": len(ext28),
        "interpretation": "DEVELOPMENT_ONLY_PHASE26_AND_PHASE28_ALREADY_OPENED_NO_CONFIRMATORY_CLAIM",
    }
    (REPORTS / "phase29_cross_domain_screen_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
