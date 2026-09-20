#!/usr/bin/env python3
"""Develop and freeze HIT-anchored EvidenceJudge v2 after Phase30 is opened.

The model predicts whether HIT-EC's top-1 label is safe to release.  It does
not choose a replacement label.  Development uses only Phases 26, 28 and 30;
any future confirmatory cohort must be acquired after this freeze.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
OUT = ROOT / "results/phase31_development"
REPORTS = ROOT / "reports/phase31_development"
ENZYME = ROOT / "data/external/enzyme_release_2026_06_10/enzyme.dat"
PHASE29 = ROOT / "results/phase29_augmented"
SEED = 20260831
LEVELS = ("EC_L3", "EC_L4")
SOURCES = ("PHASE26", "PHASE28", "PHASE30")
BOOTSTRAPS = 10000
EC_PATTERN = re.compile(r"\b[1-9]\d*\.\d+\.\d+\.\d+\b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_enzyme_transfers(path: Path) -> tuple[dict[str, tuple[str, ...]], set[str]]:
    transfers: dict[str, tuple[str, ...]] = {}
    active: set[str] = set()
    current: str | None = None
    description: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines() + ["//"]:
        if raw.startswith("ID   "):
            current = raw[5:].strip()
            description = []
        elif raw.startswith("DE   ") and current:
            description.append(raw[5:].strip())
        elif raw == "//" and current:
            text = " ".join(description)
            if text.startswith("Transferred entry:"):
                destinations = tuple(dict.fromkeys(EC_PATTERN.findall(text)))
                if destinations:
                    transfers[current] = destinations
            elif not text.startswith("Deleted entry"):
                active.add(current)
            current = None
            description = []
    return transfers, active


def canonical_full(label: object, transfers: dict[str, tuple[str, ...]]) -> tuple[str | None, str]:
    value = str(label)
    visited: set[str] = set()
    while value in transfers:
        if value in visited:
            return None, "TRANSFER_CYCLE"
        visited.add(value)
        destinations = transfers[value]
        if len(destinations) != 1:
            return None, "AMBIGUOUS_TRANSFER"
        value = destinations[0]
    return value, "TRANSFERRED" if visited else "UNCHANGED"


def load_truth(phase: int, transfers: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    if phase == 26:
        frame = pd.read_parquet(ROOT / "results/phase26/sabio_strict_blind_cohort.parquet").rename(
            columns={"uniprot_accession": "query_protein_id"}
        )
    else:
        frame = pd.read_parquet(ROOT / f"results/phase{phase}/rcsb_strict_blind_cohort.parquet").rename(
            columns={"query_id": "query_protein_id"}
        )
    records: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        original_l3 = str(row.ec_l3) if bool(row.ec_l3_label_eligible) else None
        original_l4 = str(row.ec_l4) if bool(row.ec_l4_label_eligible) else None
        canonical_l4, status = canonical_full(original_l4, transfers) if original_l4 else (None, "NO_L4")
        canonical_l3 = canonical_l4.rsplit(".", 1)[0] if canonical_l4 else original_l3
        if original_l3 and canonical_l3:
            records.append({
                "source_dataset": f"PHASE{phase}", "annotation_level": "EC_L3",
                "query_protein_id": str(row.query_protein_id), "truth_label": canonical_l3,
                "original_truth_label": original_l3, "truth_canonicalization_status": status,
            })
        if original_l4 and canonical_l4:
            records.append({
                "source_dataset": f"PHASE{phase}", "annotation_level": "EC_L4",
                "query_protein_id": str(row.query_protein_id), "truth_label": canonical_l4,
                "original_truth_label": original_l4, "truth_canonicalization_status": status,
            })
    return pd.DataFrame(records)


def load_candidates(transfers: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    development = pd.read_parquet(PHASE29 / "augmented_development_candidates.parquet")
    development.drop(columns=["truth_label", "correct"], errors="ignore", inplace=True)
    phase30 = pd.read_parquet(ROOT / "results/phase30/external_augmented_candidate_matrix_blind.parquet")
    phase30["source_dataset"] = "PHASE30"
    data = pd.concat([development, phase30], ignore_index=True, sort=False)
    truths = pd.concat([load_truth(phase, transfers) for phase in (26, 28, 30)], ignore_index=True)
    data = data.merge(
        truths, on=["source_dataset", "annotation_level", "query_protein_id"],
        how="inner", validate="many_to_one",
    )
    data["original_candidate_label"] = data["candidate_label"].astype(str)
    ec4 = data["annotation_level"].eq("EC_L4")
    canonicalized = [canonical_full(value, transfers)[0] for value in data.loc[ec4, "candidate_label"]]
    data.loc[ec4, "candidate_label"] = [value if value is not None else old for value, old in zip(
        canonicalized, data.loc[ec4, "candidate_label"].astype(str), strict=True
    )]
    data["candidate_label"] = data["candidate_label"].astype(str)
    data["correct"] = data["candidate_label"].eq(data["truth_label"])
    data["cluster_key"] = data["source_dataset"].astype(str) + "::" + data["query_cluster_id_30"].astype(str)
    return data


def anchored_rows(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    anchored = data.loc[data["support__HIT_EC"].eq(1)].copy()
    grain = ["source_dataset", "annotation_level", "query_protein_id"]
    if anchored.duplicated(grain).any():
        raise RuntimeError("HIT-EC anchored candidate is not unique per query and level")
    anchored["hit_old_agreement_fraction"] = anchored["hit_old_agreement_count"] / anchored[
        "available_old_tools"
    ].clip(lower=1)
    anchored["hit_old_disagreement_count"] = anchored["available_old_tools"] - anchored[
        "hit_old_agreement_count"
    ]
    anchored["hit_clean_disagree"] = 1 - anchored["hit_clean_agree"]
    anchored["clean_neg_distance"] = -anchored["clean_top1_distance"]
    anchored["clean_log_gmm"] = np.log10(anchored["clean_top1_gmm_confidence"].clip(lower=1e-12))
    anchored["hit_logit_abs"] = anchored["hit_top1_logit"].abs()
    for minimum in range(1, 8):
        anchored[f"hit_old_agree_ge{minimum}"] = anchored["hit_old_agreement_count"].ge(minimum).astype(int)
    for feature in features:
        if feature not in anchored:
            anchored[feature] = np.nan
    return anchored


def domain_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=float)
    for _, positions in frame.groupby("source_dataset", observed=True).indices.items():
        weights[positions] = 1.0 / len(positions)
    return weights * len(frame) / weights.sum()


def fold_for(row: pd.Series, folds: int = 5) -> int:
    value = f"{SEED}|{row['cluster_key']}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % folds


def bootstrap_interval(frame: pd.DataFrame, context: str, draws: int = BOOTSTRAPS) -> tuple[float, float]:
    groups = frame.groupby("cluster_key", observed=True)["correct"].agg(["sum", "count"])
    values = groups[["sum", "count"]].to_numpy(float)
    if not len(values):
        return float("nan"), float("nan")
    if np.all(values[:, 0] == values[:, 1]):
        return 1.0, 1.0
    seed = int.from_bytes(hashlib.sha256(f"{SEED}|{context}".encode()).digest()[:8], "big") % (2**32 - 1)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for start in range(0, draws, 250):
        size = min(250, draws - start)
        sample = values[rng.integers(0, len(values), size=(size, len(values)))]
        estimates[start:start + size] = sample[:, :, 0].sum(1) / sample[:, :, 1].sum(1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def endpoint(frame: pd.DataFrame, total: int, context: str, draws: int = BOOTSTRAPS) -> dict[str, object]:
    accepted = len(frame)
    correct = int(frame["correct"].sum()) if accepted else 0
    clusters = int(frame["cluster_key"].nunique()) if accepted else 0
    precision = correct / accepted if accepted else float("nan")
    low, high = bootstrap_interval(frame, context, draws) if accepted else (float("nan"), float("nan"))
    cp_low = float(beta.ppf(0.025, correct, accepted - correct + 1)) if correct else 0.0
    per_source = {
        source: {
            "accepted": len(group),
            "correct": int(group["correct"].sum()),
            "precision": float(group["correct"].mean()) if len(group) else float("nan"),
        }
        for source, group in frame.groupby("source_dataset", observed=True)
    }
    source_robust = all(
        source in per_source
        and per_source[source]["accepted"] >= 10
        and per_source[source]["precision"] >= 0.95
        for source in SOURCES
    )
    qualifies = bool(
        accepted >= 50 and clusters >= 20 and precision >= 0.95 and low >= 0.95 and source_robust
    ) if accepted else False
    return {
        "total": total, "accepted": accepted, "clusters": clusters,
        "coverage": accepted / total if total else 0.0, "correct": correct, "precision": precision,
        "cluster_bootstrap_low": low, "cluster_bootstrap_high": high,
        "clopper_pearson_two_sided_low": cp_low, "source_robust": source_robust,
        "per_source": per_source, "qualifies_95": qualifies,
    }


def select_threshold(frame: pd.DataFrame, score_column: str, direction: str, context: str) -> dict[str, object]:
    scored = frame.loc[pd.to_numeric(frame[score_column], errors="coerce").notna()].copy()
    scored[score_column] = pd.to_numeric(scored[score_column], errors="coerce")
    values = np.sort(scored[score_column].unique())
    if direction == "HIGH":
        values = values
    elif direction == "LOW":
        values = values[::-1]
    else:
        raise ValueError(direction)
    total = frame["query_protein_id"].nunique()
    finalists: list[tuple[float, pd.DataFrame]] = []
    # Thresholds are traversed from most permissive to most restrictive.
    for threshold in values:
        accepted = scored.loc[
            scored[score_column].ge(threshold) if direction == "HIGH" else scored[score_column].le(threshold)
        ]
        n = len(accepted)
        if n < 50 or accepted["cluster_key"].nunique() < 20 or accepted["correct"].mean() < 0.95:
            continue
        by_source = accepted.groupby("source_dataset", observed=True)["correct"].agg(["size", "mean"])
        if not all(source in by_source.index and by_source.loc[source, "size"] >= 10 and by_source.loc[source, "mean"] >= 0.95 for source in SOURCES):
            continue
        finalists.append((float(threshold), accepted))
    qualified: list[dict[str, object]] = []
    for threshold, accepted in finalists:
        metric = endpoint(accepted, total, f"{context}|{threshold:.12g}", draws=3000)
        if metric["cluster_bootstrap_low"] >= 0.95:
            qualified.append({"threshold": threshold, "direction": direction, **metric})
    if not qualified:
        empty = endpoint(frame.iloc[0:0], total, f"{context}|NONE", draws=3000)
        return {"qualified": False, "threshold": math.inf if direction == "HIGH" else -math.inf,
                "direction": direction, **empty}
    selected = max(qualified, key=lambda row: (row["accepted"], row["precision"], row["cluster_bootstrap_low"]))
    final_selected = scored.loc[
        scored[score_column].ge(selected["threshold"])
        if direction == "HIGH" else scored[score_column].le(selected["threshold"])
    ]
    metric = endpoint(final_selected, total, f"{context}|FINAL|{selected['threshold']:.12g}")
    return {"qualified": bool(metric["qualifies_95"]), "threshold": selected["threshold"],
            "direction": direction, **metric}


def factories(cpu: int) -> dict[str, tuple[object, str]]:
    return {
        "LOGISTIC_L2_C0P03": (
            lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.03, max_iter=5000, class_weight="balanced", random_state=SEED)),
            ]), "model__sample_weight",
        ),
        "LOGISTIC_L2_C0P1": (
            lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", RobustScaler()),
                ("model", LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced", random_state=SEED)),
            ]), "model__sample_weight",
        ),
        "LOGISTIC_L2_C0P3": (
            lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced", random_state=SEED)),
            ]), "model__sample_weight",
        ),
        "HIST_GB_L7": (
            lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("model", HistGradientBoostingClassifier(
                    learning_rate=0.04, max_iter=250, max_leaf_nodes=7,
                    min_samples_leaf=20, l2_regularization=5.0, random_state=SEED,
                )),
            ]), "model__sample_weight",
        ),
        "LGB_L3": (
            lambda: lgb.LGBMClassifier(
                objective="binary", n_estimators=300, learning_rate=0.025, num_leaves=3,
                min_child_samples=25, colsample_bytree=0.8, reg_lambda=7.0, reg_alpha=0.7,
                random_state=SEED, n_jobs=cpu, verbosity=-1,
            ), "sample_weight",
        ),
        "LGB_L7": (
            lambda: lgb.LGBMClassifier(
                objective="binary", n_estimators=350, learning_rate=0.02, num_leaves=7,
                min_child_samples=25, colsample_bytree=0.8, reg_lambda=7.0, reg_alpha=0.7,
                random_state=SEED, n_jobs=cpu, verbosity=-1,
            ), "sample_weight",
        ),
    }


def fit(model: object, weight_name: str, x: pd.DataFrame, y: pd.Series, weights: np.ndarray) -> None:
    model.fit(x, y.astype(int), **{weight_name: weights})


def native_score_specs(features: list[str]) -> dict[str, list[tuple[str, str]]]:
    specs = {
        "HIT_EC": [
            ("hit_top1_logit", "HIGH"), ("hit_top1_softmax", "HIGH"),
            ("hit_softmax_margin12", "HIGH"),
        ],
        "CLEAN": [
            ("clean_top1_distance", "LOW"), ("clean_top1_gmm_confidence", "HIGH"),
            ("clean_distance_margin12", "HIGH"),
        ],
    }
    for method in (
        "SEQUENCE_IDENTITY", "ESM2_SIMILARITY", "FOLDSEEK_IDENTITY", "PFAM_JACCARD",
        "LIGHTGBM_GLOBAL", "DEEP_GLOBAL", "SITEGUARD",
    ):
        variants = []
        for prefix in ("query_score__", "query_margin__"):
            name = prefix + method
            if name in features:
                variants.append((name, "HIGH"))
        specs[method] = variants
    return specs


def build_single_policies(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for level in LEVELS:
        level_data = data.loc[data["annotation_level"].eq(level)]
        for method, variants in native_score_specs(features).items():
            support = f"support__{method}"
            if support not in level_data:
                continue
            predictions = level_data.loc[level_data[support].eq(1)].copy()
            if predictions.duplicated(["source_dataset", "query_protein_id"]).any():
                raise RuntimeError(f"duplicate single-tool row: {level} {method}")
            choices = []
            for score, direction in variants:
                if score not in predictions:
                    continue
                result = select_threshold(predictions, score, direction, f"SINGLE|{level}|{method}|{score}")
                choices.append({"score_variant": score, **result})
            qualified = [choice for choice in choices if choice["qualified"]]
            if qualified:
                choice = max(qualified, key=lambda item: (item["accepted"], item["precision"]))
                rows.append({
                    "annotation_level": level, "method": method,
                    "policy_status": "FROZEN_NATIVE_SCORE_THRESHOLD", **choice,
                })
            else:
                rows.append({
                    "annotation_level": level, "method": method,
                    "policy_status": "ABSTAIN_ALL", "score_variant": "NONE", "direction": "HIGH",
                    "threshold": math.inf, "qualified": False, "total": predictions["query_protein_id"].nunique(),
                    "accepted": 0, "clusters": 0, "coverage": 0.0, "correct": 0,
                    "precision": float("nan"), "cluster_bootstrap_low": float("nan"),
                    "cluster_bootstrap_high": float("nan"), "clopper_pearson_two_sided_low": 0.0,
                    "source_robust": False, "per_source": {}, "qualifies_95": False,
                })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if sha256(ENZYME) != "e3cf02778ecea7c5b3e2813c8120625b5626d779a3657e60b8d27f063082c34a":
        raise RuntimeError("ENZYME release hash mismatch")
    transfers, active = parse_enzyme_transfers(ENZYME)
    freeze29 = json.loads((PHASE29 / "augmented_evidencejudge_freeze.json").read_text())
    base_features = list(freeze29["feature_columns"])
    engineered = [
        "hit_old_agreement_fraction", "hit_old_disagreement_count", "hit_clean_disagree",
        "clean_neg_distance", "clean_log_gmm", "hit_logit_abs",
    ] + [f"hit_old_agree_ge{minimum}" for minimum in range(1, 8)]
    features = base_features + engineered
    all_candidates = load_candidates(transfers)
    anchored = anchored_rows(all_candidates, features)
    anchored[features] = anchored[features].apply(pd.to_numeric, errors="coerce")
    anchored["cv_fold"] = anchored.apply(fold_for, axis=1)
    anchored.to_parquet(OUT / "hit_anchored_development_rows.parquet", index=False, compression="zstd")

    canonical_summary = {
        "enzyme_sha256": sha256(ENZYME), "transfer_entries": len(transfers), "active_entries": len(active),
        "truth_rows_transferred": int(anchored["truth_canonicalization_status"].eq("TRANSFERRED").sum()),
        "candidate_rows_changed": int(anchored["candidate_label"].ne(anchored["original_candidate_label"]).sum()),
        "transferred_truth_examples": anchored.loc[
            anchored["truth_canonicalization_status"].eq("TRANSFERRED"),
            ["source_dataset", "annotation_level", "query_protein_id", "original_truth_label", "truth_label"],
        ].drop_duplicates().to_dict("records"),
    }
    (REPORTS / "ec_canonicalization_summary.json").write_text(
        json.dumps(canonical_summary, indent=2, default=str) + "\n", encoding="utf-8"
    )

    cpu = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    model_rows: list[dict[str, object]] = []
    oof_rows: list[pd.DataFrame] = []
    selected: dict[str, dict[str, object]] = {}
    for level in LEVELS:
        frame = anchored.loc[anchored["annotation_level"].eq(level)].reset_index(drop=True)
        for name, (factory, weight_name) in factories(cpu).items():
            scores = np.full(len(frame), np.nan)
            for fold in range(5):
                train = frame.loc[frame["cv_fold"].ne(fold)]
                test = frame.loc[frame["cv_fold"].eq(fold)]
                model = factory()
                fit(model, weight_name, train[features], train["correct"], domain_weights(train))
                scores[test.index] = model.predict_proba(test[features])[:, 1]
            if np.isnan(scores).any():
                raise RuntimeError(f"missing OOF scores: {level} {name}")
            predicted = frame[[
                "source_dataset", "annotation_level", "query_protein_id", "query_cluster_id_30",
                "cluster_key", "candidate_label", "truth_label", "original_truth_label", "correct",
            ]].copy()
            predicted["model"] = name
            predicted["score"] = scores
            lock = select_threshold(predicted, "score", "HIGH", f"JUDGE|{level}|{name}")
            loco = {}
            for heldout in SOURCES:
                train = frame.loc[frame["source_dataset"].ne(heldout)]
                test = frame.loc[frame["source_dataset"].eq(heldout)]
                model = factory()
                fit(model, weight_name, train[features], train["correct"], domain_weights(train))
                probability = model.predict_proba(test[features])[:, 1]
                loco[heldout] = {
                    "queries": len(test),
                    "auroc": float(roc_auc_score(test["correct"], probability)),
                    "average_precision": float(average_precision_score(test["correct"], probability)),
                }
            model_rows.append({
                "annotation_level": level, "model": name,
                "oof_auroc": float(roc_auc_score(frame["correct"], scores)),
                "oof_average_precision": float(average_precision_score(frame["correct"], scores)),
                "lock_qualified": lock["qualified"], "locked_threshold": lock["threshold"],
                "locked_accepted": lock["accepted"], "locked_clusters": lock["clusters"],
                "locked_coverage": lock["coverage"], "locked_precision": lock["precision"],
                "locked_cluster_bootstrap_low": lock["cluster_bootstrap_low"],
                "locked_cp_low": lock["clopper_pearson_two_sided_low"],
                "locked_source_robust": lock["source_robust"],
                "per_source_json": json.dumps(lock["per_source"], sort_keys=True),
                "leave_one_source_out_json": json.dumps(loco, sort_keys=True),
            })
            predicted["accepted_at_locked_threshold"] = predicted["score"].ge(lock["threshold"])
            oof_rows.append(predicted)
            if lock["qualified"]:
                rank = (int(lock["accepted"]), float(lock["cluster_bootstrap_low"]), float(lock["precision"]))
                if level not in selected or rank > selected[level]["rank"]:
                    selected[level] = {
                        "model": name, "factory": factory, "weight_name": weight_name,
                        "lock": lock, "rank": rank,
                    }

    model_summary = pd.DataFrame(model_rows)
    model_summary.to_csv(OUT / "hit_evidencejudge_model_selection.tsv", sep="\t", index=False)
    pd.concat(oof_rows, ignore_index=True).to_parquet(
        OUT / "hit_evidencejudge_oof_predictions.parquet", index=False, compression="zstd"
    )
    policies = build_single_policies(all_candidates, base_features)
    policies.to_csv(OUT / "single_tool_policy_freeze_v2.tsv", sep="\t", index=False)

    frozen: dict[str, object] = {
        "phase": "31_DEVELOPMENT_AFTER_PHASE30_OPENING",
        "status": "DEVELOPMENT_ONLY_REQUIRES_FRESH_EXTERNAL_CONFIRMATION",
        "task": "selective correctness prediction for HIT-EC top-1; no replacement-label selection",
        "seed": SEED, "folds": 5, "development_sources": list(SOURCES),
        "feature_columns": features, "truth_in_model_inputs": False,
        "enzyme_release": "10-Jun-2026", "enzyme_sha256": sha256(ENZYME),
        "ec_canonicalization": "single-destination ENZYME transferred entries followed to current EC",
        "selection_rule": "maximize OOF accepted queries subject to pooled point and cluster-bootstrap precision >=0.95, n>=50, clusters>=20, and each source n>=10/precision>=0.95",
        "selected": {},
    }
    for level, choice in selected.items():
        frame = anchored.loc[anchored["annotation_level"].eq(level)].reset_index(drop=True)
        model = choice["factory"]()
        fit(model, choice["weight_name"], frame[features], frame["correct"], domain_weights(frame))
        model_path = OUT / f"frozen_{level.lower()}_hit_evidencejudge_v2_{choice['model'].lower()}.joblib"
        joblib.dump(model, model_path)
        frozen["selected"][level] = {
            "model": choice["model"], "threshold": choice["lock"]["threshold"],
            "development_endpoint": choice["lock"], "model_path": str(model_path),
            "model_sha256": sha256(model_path),
        }
    freeze_path = OUT / "hit_evidencejudge_v2_freeze.json"
    freeze_path.write_text(json.dumps(frozen, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    policy_json = {
        "phase": "31_DEVELOPMENT", "status": "FROZEN_FOR_FUTURE_EXTERNAL_ONLY",
        "policies": policies.to_dict("records"),
    }
    (OUT / "single_tool_policy_freeze_v2.json").write_text(
        json.dumps(policy_json, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(canonical_summary, indent=2, default=str))
    print(model_summary.to_string(index=False))
    print(policies[[
        "annotation_level", "method", "policy_status", "score_variant", "direction", "threshold",
        "accepted", "clusters", "precision", "cluster_bootstrap_low",
    ]].to_string(index=False))
    print(json.dumps(frozen, indent=2, sort_keys=True, default=str))
    if "EC_L3" not in selected:
        raise RuntimeError("No EC_L3 EvidenceJudge v2 operating point met the frozen development criteria")
    print("CHECKPOINT_31A_HIT_EVIDENCEJUDGE_V2_FROZEN")


if __name__ == "__main__":
    main()
