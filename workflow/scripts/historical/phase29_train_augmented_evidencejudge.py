#!/usr/bin/env python3
"""Build, cross-validate, and freeze an EvidenceJudge augmented with HIT-EC and CLEAN."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
OUT = ROOT / "results/phase29_augmented"
REPORTS = ROOT / "reports/phase29_augmented"
SEED = 20260829
LEVELS = ("EC_L3", "EC_L4")


def fold_for_cluster(cluster: str, n_folds: int = 5) -> int:
    digest = hashlib.sha256(f"{SEED}|{cluster}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % n_folds


def domain_weights(frame: pd.DataFrame) -> np.ndarray:
    base = frame["query_weight"].to_numpy(float)
    result = np.zeros(len(frame), dtype=float)
    for _, positions in frame.groupby("source_dataset", observed=True).indices.items():
        local = base[positions]
        result[positions] = local / max(local.sum(), 1e-12)
    return result * len(frame) / max(result.sum(), 1e-12)


def load_phase(phase: int) -> pd.DataFrame:
    if phase == 26:
        truth = pd.read_parquet(ROOT / "results/phase26/sabio_strict_blind_cohort.parquet").rename(
            columns={"uniprot_accession": "query_protein_id"}
        )
    else:
        truth = pd.read_parquet(ROOT / "results/phase28/rcsb_strict_blind_cohort.parquet").rename(
            columns={"query_id": "query_protein_id"}
        )
    old = pd.read_parquet(ROOT / f"results/phase{phase}/external_tool_top_predictions_blind.parquet")
    hit = pd.read_csv(ROOT / f"data/interim/phase29_hit_ec/phase{phase}_hit_ec_predictions.tsv", sep="\t").rename(
        columns={
            "query_id": "query_protein_id",
            "sequence_length": "hit_sequence_length",
            "truncated_residue_count": "hit_truncated_residue_count",
        }
    )
    clean = pd.read_csv(ROOT / f"data/interim/phase29_clean/phase{phase}_clean_predictions.tsv", sep="\t").rename(
        columns={
            "query_id": "query_protein_id",
            "sequence_length": "clean_sequence_length",
            "truncated_residue_count": "clean_truncated_residue_count",
        }
    )
    metadata = pd.read_parquet(ROOT / f"results/phase{phase}/external_query_metadata.parquet")
    query = truth.merge(hit, on="query_protein_id", validate="one_to_one").merge(
        clean, on="query_protein_id", validate="one_to_one"
    ).merge(metadata, on="query_protein_id", how="left", validate="one_to_one", suffixes=("", "_metadata"))
    query["clean_ec3_top1"] = query["clean_top1_ec4"].str.rsplit(".", n=1).str[0]
    query["source_dataset"] = f"PHASE{phase}"

    output = []
    for level in LEVELS:
        number = 3 if level == "EC_L3" else 4
        eligible = f"ec_l{number}_label_eligible"
        truth_column = f"ec_l{number}"
        subset = query.loc[query[eligible].astype(bool)].copy()
        old_level = old.loc[old["annotation_level"].eq(level)].copy()
        old_level.sort_values(["query_protein_id", "method", "candidate_label"], kind="mergesort", inplace=True)
        if old_level.duplicated(["query_protein_id", "method"]).any():
            raise RuntimeError(f"phase {phase} {level}: duplicate old tool top predictions")
        method_channel = old_level[["method", "evidence_channel"]].drop_duplicates().set_index("method")[
            "evidence_channel"
        ].to_dict()
        old_by_query = {key: group for key, group in old_level.groupby("query_protein_id", observed=True)}

        for _, row in subset.iterrows():
            query_id = row["query_protein_id"]
            old_rows = old_by_query.get(query_id, pd.DataFrame(columns=old_level.columns))
            old_predictions = {
                str(item.method): {
                    "label": str(item.candidate_label),
                    "score": float(item.raw_score) if pd.notna(item.raw_score) else np.nan,
                    "margin": float(item.top1_margin) if pd.notna(item.top1_margin) else np.nan,
                    "channel": str(item.evidence_channel),
                }
                for item in old_rows.itertuples(index=False)
            }
            hit_label = str(row[f"ec{number}_top1"])
            clean_label = str(row["clean_ec3_top1"] if number == 3 else row["clean_top1_ec4"])
            candidates = sorted({item["label"] for item in old_predictions.values()} | {hit_label, clean_label})
            available_old = len(old_predictions)
            candidate_counts = Counter([item["label"] for item in old_predictions.values()] + [hit_label, clean_label])
            sorted_counts = sorted(candidate_counts.values(), reverse=True)
            second_votes = sorted_counts[1] if len(sorted_counts) > 1 else 0
            total_votes = available_old + 2
            probabilities = np.asarray(list(candidate_counts.values()), dtype=float) / total_votes
            entropy = float(-(probabilities * np.log(probabilities)).sum() / max(math.log(len(probabilities)), 1e-12))
            for candidate in candidates:
                record: dict[str, object] = {
                    "source_dataset": row["source_dataset"],
                    "annotation_level": level,
                    "query_protein_id": query_id,
                    "query_cluster_id_30": row["external_cluster_id_30"],
                    "candidate_label": candidate,
                    "truth_label": str(row[truth_column]),
                    "correct": candidate == str(row[truth_column]),
                    "available_old_tools": available_old,
                    "available_tools_augmented": total_votes,
                    "candidate_set_size": len(candidates),
                    "agreement_entropy_augmented": entropy,
                    "candidate_votes_augmented": candidate_counts[candidate],
                    "candidate_vote_fraction_augmented": candidate_counts[candidate] / total_votes,
                    "second_candidate_votes_augmented": second_votes,
                    "candidate_vote_lead_augmented": candidate_counts[candidate] - second_votes,
                    "candidate_is_plurality": int(candidate_counts[candidate] == sorted_counts[0]),
                    "support__HIT_EC": int(candidate == hit_label),
                    "support__CLEAN": int(candidate == clean_label),
                    "support_score__HIT_EC": float(row[f"ec{number}_top1_softmax"]) if candidate == hit_label else np.nan,
                    "support_score__CLEAN": float(row["clean_top1_gmm_confidence"]) if candidate == clean_label else np.nan,
                    "hit_old_agreement_count": sum(item["label"] == hit_label for item in old_predictions.values()),
                    "clean_old_agreement_count": sum(item["label"] == clean_label for item in old_predictions.values()),
                    "hit_clean_agree": int(hit_label == clean_label),
                    "hit_top1_softmax": float(row[f"ec{number}_top1_softmax"]),
                    "hit_top1_logit": float(row[f"ec{number}_top1_logit"]),
                    "hit_softmax_margin12": float(row[f"ec{number}_softmax_margin12"]),
                    "hit_ec4_sigmoid": float(row["ec4_top1_sigmoid"]),
                    "hit_ec4_sigmoid_margin12": float(row["ec4_sigmoid_margin12"]),
                    "hit_ec3_ec4_consistent": int(row["ec3_ec4_hierarchy_consistent"]),
                    "clean_top1_distance": float(row["clean_top1_distance"]),
                    "clean_top1_gmm_confidence": float(row["clean_top1_gmm_confidence"]),
                    "clean_distance_margin12": float(row["clean_distance_margin12"]),
                    "clean_distance_ratio12": float(row["clean_distance_ratio12"]),
                    "clean_maxsep_n": int(row["clean_maxsep_n"]),
                    "log1p_sequence_length": math.log1p(float(row["hit_sequence_length"])),
                    "log1p_hit_truncation": math.log1p(float(row["hit_truncated_residue_count"])),
                    "nearest_frozen_identity_fraction": float(row.get("nearest_frozen_identity_fraction", np.nan)),
                    "pfam_seen_in_fit": int(pd.notna(row.get("pfam_seen_in_fit")) and bool(row.get("pfam_seen_in_fit"))),
                    "pfam_clan_seen_in_fit": int(pd.notna(row.get("pfam_clan_seen_in_fit")) and bool(row.get("pfam_clan_seen_in_fit"))),
                    "cath_seen_in_fit": int(pd.notna(row.get("cath_seen_in_fit")) and bool(row.get("cath_seen_in_fit"))),
                    "query_structure_available": int(pd.notna(row.get("query_structure_available")) and bool(row.get("query_structure_available"))),
                }
                supporting_channels = set()
                for method, item in old_predictions.items():
                    support = int(candidate == item["label"])
                    record[f"support__{method}"] = support
                    record[f"query_score__{method}"] = item["score"]
                    record[f"query_margin__{method}"] = item["margin"]
                    record[f"support_score__{method}"] = item["score"] if support else np.nan
                    record[f"support_margin__{method}"] = item["margin"] if support else np.nan
                    if support:
                        supporting_channels.add(method_channel[method])
                if candidate == hit_label:
                    supporting_channels.add("hierarchical_transformer")
                if candidate == clean_label:
                    supporting_channels.add("contrastive_embedding")
                record["supporting_channels_augmented"] = len(supporting_channels)
                output.append(record)
    frame = pd.DataFrame(output)
    frame["query_weight"] = 1.0 / frame.groupby(
        ["source_dataset", "annotation_level", "query_protein_id"], observed=True
    )["candidate_label"].transform("nunique")
    frame["cv_fold"] = frame["query_cluster_id_30"].map(fold_for_cluster)
    return frame


def winner_rows(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    result = frame[[
        "source_dataset", "annotation_level", "query_protein_id", "query_cluster_id_30",
        "candidate_label", "truth_label", "correct",
    ]].copy()
    result["score"] = scores
    result.sort_values(
        ["source_dataset", "annotation_level", "query_protein_id", "score", "candidate_label"],
        ascending=[True, True, True, False, True], kind="mergesort", inplace=True,
    )
    return result.drop_duplicates(["source_dataset", "annotation_level", "query_protein_id"])


def bootstrap_interval(frame: pd.DataFrame, context: str, draws: int = 5000) -> tuple[float, float]:
    grouped = frame.groupby("query_cluster_id_30", observed=True)["correct"].agg(["sum", "count"])
    values = grouped[["sum", "count"]].to_numpy(float)
    if not len(values):
        return float("nan"), float("nan")
    if len(values) == 1:
        value = float(values[0, 0] / values[0, 1])
        return value, value
    seed = int.from_bytes(hashlib.sha256(f"{SEED}|{context}".encode()).digest()[:8], "big") % (2**32 - 1)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for start in range(0, draws, 250):
        size = min(250, draws - start)
        sample = values[rng.integers(0, len(values), size=(size, len(values)))]
        estimates[start : start + size] = sample[:, :, 0].sum(1) / sample[:, :, 1].sum(1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def endpoint(frame: pd.DataFrame, context: str, total: int) -> dict[str, object]:
    n = len(frame)
    successes = int(frame["correct"].sum()) if n else 0
    low, high = bootstrap_interval(frame, context) if n else (float("nan"), float("nan"))
    cp_low = float(beta.ppf(0.025, successes, n - successes + 1)) if successes else 0.0
    return {
        "accepted": n,
        "clusters": int(frame["query_cluster_id_30"].nunique()),
        "coverage": n / total if total else 0.0,
        "precision": successes / n if n else float("nan"),
        "cluster_bootstrap_low": low,
        "cluster_bootstrap_high": high,
        "clopper_pearson_two_sided_low": cp_low,
        "primary_safe": bool(n >= 50 and frame["query_cluster_id_30"].nunique() >= 20 and successes / n >= 0.95 and low >= 0.95) if n else False,
    }


def best_threshold(winners: pd.DataFrame, context: str) -> dict[str, object]:
    total = winners["query_protein_id"].nunique()
    candidates = np.unique(np.quantile(winners["score"], np.linspace(0, 1, min(301, len(winners) + 1))))
    rows = []
    for threshold in candidates:
        accepted = winners.loc[winners["score"].ge(threshold)]
        metric = endpoint(accepted, f"{context}|{threshold:.12g}", total)
        if metric["primary_safe"]:
            rows.append({"threshold": float(threshold), **metric})
    if not rows:
        return {"qualified": False, "threshold": float("inf"), **endpoint(winners.iloc[0:0], context, total)}
    return {"qualified": True, **max(rows, key=lambda row: (row["accepted"], row["precision"], -row["threshold"]))}


def model_factories() -> dict[str, object]:
    cpu = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    return {
        "LOGISTIC_L2": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced", random_state=SEED)),
        ]),
        "LGB_L7": lambda: lgb.LGBMClassifier(
            objective="binary", n_estimators=300, learning_rate=0.025, num_leaves=7,
            min_child_samples=25, colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.5,
            random_state=SEED, n_jobs=cpu, verbosity=-1,
        ),
        "LGB_L15": lambda: lgb.LGBMClassifier(
            objective="binary", n_estimators=350, learning_rate=0.025, num_leaves=15,
            min_child_samples=20, colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.5,
            random_state=SEED, n_jobs=cpu, verbosity=-1,
        ),
        "LGB_L31": lambda: lgb.LGBMClassifier(
            objective="binary", n_estimators=350, learning_rate=0.02, num_leaves=31,
            min_child_samples=15, colsample_bytree=0.7, reg_lambda=7.0, reg_alpha=0.7,
            random_state=SEED, n_jobs=cpu, verbosity=-1,
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    data = pd.concat([load_phase(26), load_phase(28)], ignore_index=True, sort=False)
    protected = {
        "source_dataset", "annotation_level", "query_protein_id", "query_cluster_id_30",
        "candidate_label", "truth_label", "correct", "query_weight", "cv_fold",
    }
    features = sorted(column for column in data.columns if column not in protected)
    data[features] = data[features].apply(pd.to_numeric, errors="coerce")
    data.to_parquet(OUT / "augmented_development_candidates.parquet", index=False)

    summary_rows = []
    prediction_rows = []
    selected_models: dict[str, dict[str, object]] = {}
    for level in LEVELS:
        level_data = data.loc[data["annotation_level"].eq(level)].reset_index(drop=True)
        for model_name, factory in model_factories().items():
            oof = np.full(len(level_data), np.nan)
            for fold in range(5):
                train = level_data.loc[level_data["cv_fold"].ne(fold)]
                test = level_data.loc[level_data["cv_fold"].eq(fold)]
                model = factory()
                model.fit(train[features], train["correct"].astype(int), **(
                    {"sample_weight": domain_weights(train)} if model_name.startswith("LGB") else {"model__sample_weight": domain_weights(train)}
                ))
                oof[test.index] = model.predict_proba(test[features])[:, 1]
            if np.isnan(oof).any():
                raise RuntimeError(f"missing OOF predictions for {level} {model_name}")
            winners = winner_rows(level_data, oof)
            lock = best_threshold(winners, f"{level}|{model_name}|OOF")
            all_metric = endpoint(winners, f"{level}|{model_name}|ALL", len(winners))
            summary_rows.append({
                "annotation_level": level,
                "model": model_name,
                "oof_winner_accuracy": all_metric["precision"],
                "lock_qualified": lock["qualified"],
                **{f"locked_{key}": value for key, value in lock.items() if key != "qualified"},
            })
            winners["model"] = model_name
            winners["accepted_at_locked_threshold"] = winners["score"].ge(float(lock["threshold"]))
            prediction_rows.append(winners)

            if lock["qualified"]:
                rank = (int(lock["accepted"]), float(lock["precision"]), float(lock["cluster_bootstrap_low"]))
                previous = selected_models.get(level)
                if previous is None or rank > previous["rank"]:
                    selected_models[level] = {"name": model_name, "factory": factory, "lock": lock, "rank": rank}

        # Transparent consensus rule developed alongside trained models.
        query = level_data.sort_values(
            ["query_protein_id", "support__HIT_EC", "candidate_label"], ascending=[True, False, True], kind="mergesort"
        ).drop_duplicates(["source_dataset", "query_protein_id"])
        if level == "EC_L3":
            rule_accept = query["support__HIT_EC"].eq(1) & (
                query["hit_old_agreement_count"].ge(5)
                | (query["hit_old_agreement_count"].eq(4) & query["clean_distance_margin12"].ge(1.5050787925720217))
            )
            rule_name = "HIT_OLD5_OR_OLD4_CLEAN_MARGIN1P505"
        else:
            rule_accept = query["support__HIT_EC"].eq(1) & (
                query["hit_old_agreement_count"].ge(5)
                | (query["hit_old_agreement_count"].eq(4) & query["clean_distance_margin12"].ge(1.712409257888794))
            )
            rule_name = "HIT_OLD5_OR_OLD4_CLEAN_MARGIN1P712"
        rule_winners = query[[
            "source_dataset", "annotation_level", "query_protein_id", "query_cluster_id_30",
            "candidate_label", "truth_label", "correct",
        ]].copy()
        rule_winners["score"] = rule_accept.astype(float).to_numpy()
        accepted = rule_winners.loc[rule_accept.to_numpy()]
        metric = endpoint(accepted, f"{level}|{rule_name}", len(rule_winners))
        summary_rows.append({
            "annotation_level": level, "model": rule_name,
            "oof_winner_accuracy": float(rule_winners["correct"].mean()),
            "lock_qualified": metric["primary_safe"],
            "locked_threshold": 1.0, **{f"locked_{key}": value for key, value in metric.items()},
        })
        rule_winners["model"] = rule_name
        rule_winners["accepted_at_locked_threshold"] = rule_accept.to_numpy()
        prediction_rows.append(rule_winners)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "augmented_judge_development_endpoints.tsv", sep="\t", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(OUT / "augmented_judge_oof_predictions.parquet", index=False)

    frozen = {
        "phase": "29_AUGMENTED_DEVELOPMENT",
        "status": "DEVELOPMENT_ONLY_NO_CONFIRMATORY_CLAIM",
        "seed": SEED,
        "folds": 5,
        "feature_columns": features,
        "candidate_tools": "existing Phase26/28 tools + HIT-EC v2.0.0 + CLEAN split100",
        "truth_in_model_inputs": False,
        "selected": {},
    }
    for level, choice in selected_models.items():
        final_model = choice["factory"]()
        level_data = data.loc[data["annotation_level"].eq(level)].reset_index(drop=True)
        final_model.fit(level_data[features], level_data["correct"].astype(int), **(
            {"sample_weight": domain_weights(level_data)} if choice["name"].startswith("LGB") else {"model__sample_weight": domain_weights(level_data)}
        ))
        model_path = OUT / f"frozen_{level.lower()}_{choice['name'].lower()}.joblib"
        joblib.dump(final_model, model_path)
        frozen["selected"][level] = {
            "model": choice["name"],
            "threshold": choice["lock"]["threshold"],
            "development_endpoint": choice["lock"],
            "model_path": str(model_path),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        }
    (OUT / "augmented_evidencejudge_freeze.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(frozen, indent=2, sort_keys=True, default=str))
    print("CHECKPOINT_29C_AUGMENTED_EVIDENCEJUDGE_DEVELOPMENT_COMPLETE")


if __name__ == "__main__":
    main()
