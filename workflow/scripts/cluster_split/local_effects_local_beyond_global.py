#!/usr/bin/env python3
"""Test whether mapped catalytic-site context adds value beyond global homology."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest, chi2, spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 20260820
BOOTSTRAPS = 300
ROW_KEY = ["pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id"]
TARGETS = {"EC_L3": "same_ec_l3", "EC_L4": "same_ec_l4", "EXACT_RHEA": "same_exact_rhea"}
BASE = [
    "sequence_identity", "sequence_identity_squared", "sequence_query_coverage",
    "sequence_reference_coverage", "sequence_alignment_fraction", "esm2_t33_cosine",
    "foldseek_identity", "foldseek_identity_squared", "foldseek_alignment_fraction",
    "pfam_jaccard", "primary_pfam_match", "cath_jaccard", "primary_cath_match",
    "length_ratio", "same_taxonomy_group",
]
LOCAL_BLOCKS = {
    "GLOBAL_ONLY": [],
    "GLOBAL_PLUS_L0": ["local0_score"],
    "GLOBAL_PLUS_L1": ["local0_score", "local1_score"],
    "GLOBAL_PLUS_L2": ["local0_score", "local1_score", "local2_score"],
    "GLOBAL_PLUS_L3": ["local0_score", "local1_score", "local2_score", "local3_score"],
}


def prepare_matrix(train: pd.DataFrame, evaluation: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_values: list[np.ndarray] = []
    eval_values: list[np.ndarray] = []
    usable: list[str] = []
    for column in columns:
        tr = pd.to_numeric(train[column], errors="coerce")
        median = float(tr.median()) if tr.notna().any() else 0.0
        tr_fill = tr.fillna(median).to_numpy(float)
        mean = float(tr_fill.mean())
        scale = float(tr_fill.std())
        if scale < 1e-10:
            continue
        usable.append(column)
        train_values.append((tr_fill - mean) / scale)
        ev = pd.to_numeric(evaluation[column], errors="coerce").fillna(median).to_numpy(float)
        eval_values.append((ev - mean) / scale)
    return (
        sm.add_constant(np.column_stack(train_values), has_constant="add"),
        sm.add_constant(np.column_stack(eval_values), has_constant="add"),
        ["const", *usable],
    )


def fit_glm(y: np.ndarray, matrix: np.ndarray, groups: np.ndarray):
    model = sm.GLM(y.astype(float), matrix, family=sm.families.Binomial())
    try:
        return model.fit(maxiter=200, disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    except Exception:  # noqa: BLE001 - retain converged estimates if robust covariance is singular
        return model.fit(maxiter=200, disp=0)


def metric_values(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return (
        float(average_precision_score(y, score, sample_weight=weight)),
        float(roc_auc_score(y, score, sample_weight=weight)),
    )


def cluster_bootstrap_delta(
    frame: pd.DataFrame, y_column: str, base_score: np.ndarray, full_score: np.ndarray, salt: str,
) -> tuple[float, float]:
    groups = frame["query_cluster_id_30"].astype(str).to_numpy()
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(int(hashlib.sha256(f"{SEED}:{salt}".encode()).hexdigest()[:8], 16))
    y = frame[y_column].astype(int).to_numpy()
    weight = frame["sample_weight"].to_numpy(float)
    values: list[float] = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        if len(np.unique(y[selected])) < 2:
            continue
        base = average_precision_score(y[selected], base_score[selected], sample_weight=weight[selected])
        full = average_precision_score(y[selected], full_score[selected], sample_weight=weight[selected])
        values.append(float(full - base))
    return (
        (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        if values else (float("nan"), float("nan"))
    )


def bh_fdr(values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return output
    count = len(valid)
    adjusted = np.minimum.accumulate((valid.to_numpy() * count / np.arange(1, count + 1))[::-1])[::-1]
    output.loc[valid.index] = np.clip(adjusted, 0, 1)
    return output


def build_frame(root: Path) -> pd.DataFrame:
    processed = root / "data/processed"
    label_columns = [
        "query_protein_id", "reference_protein_id", "reference_activity_id", "query_split_expected",
        "query_cluster_id_30", "query_primary_pfam", "same_taxonomy_group",
        "same_ec_l3", "same_ec_l4", "same_exact_rhea", "sample_weight",
    ]
    labels = pd.concat([
        pd.read_parquet(processed / "population_pairs.parquet", columns=label_columns).assign(pair_set="population"),
        pd.read_parquet(processed / "training_pairs.parquet", columns=label_columns).assign(pair_set="training"),
    ], ignore_index=True)
    global_features = pd.read_parquet(
        processed / "global_features.parquet",
        columns=ROW_KEY + [
            "sequence_identity", "sequence_query_coverage", "sequence_reference_coverage",
            "sequence_alignment_fraction", "esm2_t33_cosine", "foldseek_identity",
            "foldseek_alignment_fraction", "pfam_jaccard", "primary_pfam_match",
            "cath_jaccard", "primary_cath_match", "length_ratio",
        ],
    )
    l0 = pd.read_parquet(processed / "local_features_L0.parquet")
    l1 = pd.read_parquet(processed / "local_features_L1.parquet", columns=ROW_KEY + [
        "local1_chemical_class_match", "local1_role_compatibility",
    ])
    l2 = pd.read_parquet(processed / "local_features_L2.parquet", columns=ROW_KEY + [
        "aa_composition_cosine_r8", "property_cosine_r8", "residue_count_ratio_r8",
        "local2_residue_esm2_t12_cosine", "local2_descriptor_site_count",
    ])
    l3 = pd.read_parquet(processed / "local_features_L3.parquet", columns=ROW_KEY + [
        "radial_cosine_r8", "octant_cosine_r8", "sidechain_direction_cosine_r8",
        "query_local_plddt_r8", "reference_local_plddt_r8", "local3_descriptor_site_count",
    ])
    frame = (
        l0.merge(l1, on=ROW_KEY, validate="one_to_one")
        .merge(l2, on=ROW_KEY, validate="one_to_one")
        .merge(l3, on=ROW_KEY, validate="one_to_one")
        .merge(global_features, on=ROW_KEY, validate="one_to_one")
        .merge(labels, on=ROW_KEY, validate="one_to_one")
    )
    if len(frame) != len(l0):
        raise RuntimeError(f"Phase 9 merge mismatch: {len(frame)}/{len(l0)}")
    if not frame["query_split"].eq(frame["query_split_expected"]).all():
        raise RuntimeError("Local-feature split metadata disagrees with registered pair labels")
    frame["sequence_identity_squared"] = frame["sequence_identity"] ** 2
    frame["foldseek_identity_squared"] = frame["foldseek_identity"] ** 2
    frame["local0_score"] = frame["local0_residue_identity"]
    frame["local1_score"] = frame[["local1_chemical_class_match", "local1_role_compatibility"]].mean(axis=1)
    frame["local2_score"] = frame[[
        "aa_composition_cosine_r8", "property_cosine_r8", "residue_count_ratio_r8",
        "local2_residue_esm2_t12_cosine",
    ]].mean(axis=1)
    frame["local3_score"] = frame[[
        "radial_cosine_r8", "octant_cosine_r8", "sidechain_direction_cosine_r8",
    ]].mean(axis=1)
    frame["local_full_score"] = frame[["local0_score", "local1_score", "local2_score", "local3_score"]].mean(axis=1)
    frame["minimum_local_plddt_r8"] = frame[["query_local_plddt_r8", "reference_local_plddt_r8"]].min(axis=1)
    frame["quality_matched"] = (
        frame["mapping_success"].astype(bool)
        & frame["local2_descriptor_site_count"].notna()
        & frame["local3_descriptor_site_count"].notna()
        & frame["minimum_local_plddt_r8"].ge(70.0)
        & frame[["local0_score", "local1_score", "local2_score", "local3_score"]].notna().all(axis=1)
    )
    return frame


def conditional_and_ablation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    population = frame.loc[frame["pair_set"].eq("population") & frame["quality_matched"]].copy()
    train = population.loc[population["query_split"].eq("train")].reset_index(drop=True)
    test = population.loc[population["query_split"].eq("test")].reset_index(drop=True)
    if len(train) < 1000 or len(test) < 500:
        raise RuntimeError(f"Insufficient quality-matched population cohort: train={len(train)}, test={len(test)}")
    records: list[dict] = []
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for level, target in TARGETS.items():
        y_train = train[target].astype(int).to_numpy()
        y_test = test[target].astype(int).to_numpy()
        if min(y_train.sum(), len(y_train) - y_train.sum(), y_test.sum(), len(y_test) - y_test.sum()) < 10:
            raise RuntimeError(f"Insufficient class support for {level}")
        predictions[level] = {}
        null_matrix = np.ones((len(train), 1), dtype=float)
        null_fit = fit_glm(y_train, null_matrix, train["query_cluster_id_30"].to_numpy())
        base_score: np.ndarray | None = None
        base_fit = None
        for model_name, local_columns in LOCAL_BLOCKS.items():
            columns = BASE + local_columns
            x_train, x_test, names = prepare_matrix(train, test, columns)
            fitted = fit_glm(y_train, x_train, train["query_cluster_id_30"].to_numpy())
            test_score = np.asarray(fitted.predict(x_test), dtype=float)
            train_score = np.asarray(fitted.predict(x_train), dtype=float)
            predictions[level][model_name] = test_score
            test_auprc, test_auroc = metric_values(y_test, test_score, test["sample_weight"].to_numpy(float))
            train_auprc, _ = metric_values(y_train, train_score, train["sample_weight"].to_numpy(float))
            if model_name == "GLOBAL_ONLY":
                base_score, base_fit = test_score, fitted
                delta, delta_lower, delta_upper = 0.0, 0.0, 0.0
                lrt, lrt_df, lrt_p = 0.0, 0, float("nan")
            else:
                assert base_score is not None and base_fit is not None
                delta = test_auprc - metric_values(y_test, base_score, test["sample_weight"].to_numpy(float))[0]
                delta_lower, delta_upper = cluster_bootstrap_delta(
                    test, target, base_score, test_score, f"{level}:{model_name}"
                )
                lrt = max(0.0, 2.0 * (fitted.llf - base_fit.llf))
                lrt_df = max(1, len(names) - len(BASE) - 1)
                lrt_p = float(chi2.sf(lrt, lrt_df))
            partial_r2 = float((fitted.llf - base_fit.llf) / abs(null_fit.llf)) if base_fit is not None and model_name != "GLOBAL_ONLY" else 0.0
            coefficient = {name: float(value) for name, value in zip(names, fitted.params, strict=True) if name in local_columns}
            standard_error = {name: float(value) for name, value in zip(names, fitted.bse, strict=True) if name in local_columns}
            pvalues = {name: float(value) for name, value in zip(names, fitted.pvalues, strict=True) if name in local_columns}
            records.append({
                "annotation_level": level, "model": model_name,
                "train_rows": len(train), "test_rows": len(test),
                "train_positives": int(y_train.sum()), "test_positives": int(y_test.sum()),
                "train_auprc": train_auprc, "test_auprc": test_auprc, "test_auroc": test_auroc,
                "delta_auprc_vs_global": delta, "delta_auprc_ci_lower": delta_lower,
                "delta_auprc_ci_upper": delta_upper, "likelihood_ratio": lrt,
                "likelihood_ratio_df": lrt_df, "likelihood_ratio_p": lrt_p,
                "partial_mcfadden_r2_vs_global": partial_r2,
                "local_coefficients_json": json.dumps(coefficient, sort_keys=True),
                "local_odds_ratios_json": json.dumps({key: math.exp(value) for key, value in coefficient.items()}, sort_keys=True),
                "local_standard_errors_json": json.dumps(standard_error, sort_keys=True),
                "local_pvalues_json": json.dumps(pvalues, sort_keys=True),
                "fit_split": "population_train", "evaluation_split": "population_test",
                "mapping_success_used_as_feature": False, "minimum_local_plddt": 70.0,
            })
    conditional = pd.DataFrame(records)
    conditional["likelihood_ratio_fdr_bh"] = bh_fdr(conditional["likelihood_ratio_p"])
    ablation = conditional[[
        "annotation_level", "model", "train_rows", "test_rows", "train_auprc", "test_auprc",
        "test_auroc", "delta_auprc_vs_global", "delta_auprc_ci_lower", "delta_auprc_ci_upper",
    ]].copy()
    return conditional, ablation, predictions


def matched_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    population = frame.loc[frame["pair_set"].eq("population") & frame["quality_matched"]].copy()
    validation = population.loc[population["query_split"].eq("validation")].copy()
    test = population.loc[population["query_split"].eq("test")].copy()
    low_cut = float(validation["local_full_score"].quantile(0.25))
    high_cut = float(validation["local_full_score"].quantile(0.75))
    validation = validation.loc[validation["local_full_score"].le(low_cut) | validation["local_full_score"].ge(high_cut)].copy()
    validation["local_high"] = validation["local_full_score"].ge(high_cut).astype(int)
    test = test.loc[test["local_full_score"].le(low_cut) | test["local_full_score"].ge(high_cut)].copy()
    test["local_high"] = test["local_full_score"].ge(high_cut).astype(int)
    x_val, x_test, _ = prepare_matrix(validation, test, BASE)
    propensity_model = LogisticRegression(max_iter=1000, random_state=SEED)
    propensity_model.fit(x_val[:, 1:], validation["local_high"])
    validation["global_propensity"] = propensity_model.predict_proba(x_val[:, 1:])[:, 1]
    test["global_propensity"] = propensity_model.predict_proba(x_test[:, 1:])[:, 1]
    edges = np.unique(np.quantile(validation["global_propensity"], np.linspace(0, 1, 11)))
    if len(edges) < 4:
        edges = np.linspace(0, 1, 11)
    edges[0], edges[-1] = -np.inf, np.inf
    test["propensity_bin"] = pd.cut(test["global_propensity"], edges, labels=False, include_lowest=True)
    matches: list[dict] = []
    group_columns = ["propensity_bin", "primary_pfam_match", "primary_cath_match", "same_taxonomy_group"]
    match_index = 0
    for _, group in test.groupby(group_columns, dropna=False):
        high = group.loc[group["local_high"].eq(1)].sort_values("global_propensity")
        low = group.loc[group["local_high"].eq(0)].sort_values("global_propensity")
        for (_, high_row), (_, low_row) in zip(high.iterrows(), low.iterrows()):
            if abs(high_row.sequence_identity - low_row.sequence_identity) > 0.15:
                continue
            if abs(high_row.esm2_t33_cosine - low_row.esm2_t33_cosine) > 0.20:
                continue
            if pd.notna(high_row.foldseek_identity) and pd.notna(low_row.foldseek_identity) and abs(high_row.foldseek_identity - low_row.foldseek_identity) > 0.30:
                continue
            match_index += 1
            record = {
                "match_id": match_index,
                "high_pair_set": high_row.pair_set,
                "high_query_protein_id": high_row.query_protein_id,
                "high_reference_protein_id": high_row.reference_protein_id,
                "high_reference_activity_id": high_row.reference_activity_id,
                "low_query_protein_id": low_row.query_protein_id,
                "low_reference_protein_id": low_row.reference_protein_id,
                "low_reference_activity_id": low_row.reference_activity_id,
                "high_local_score": high_row.local_full_score,
                "low_local_score": low_row.local_full_score,
                "high_global_propensity": high_row.global_propensity,
                "low_global_propensity": low_row.global_propensity,
                "sequence_identity_gap": abs(high_row.sequence_identity - low_row.sequence_identity),
                "esm2_similarity_gap": abs(high_row.esm2_t33_cosine - low_row.esm2_t33_cosine),
            }
            for level, target in TARGETS.items():
                record[f"high_{level}"] = int(high_row[target])
                record[f"low_{level}"] = int(low_row[target])
            matches.append(record)
    detail = pd.DataFrame(matches)
    if len(detail) < 50:
        raise RuntimeError(f"Too few held-out matched pairs: {len(detail)}")
    rows: list[dict] = []
    rng = np.random.default_rng(SEED)
    for level in TARGETS:
        high = detail[f"high_{level}"].to_numpy(int)
        low = detail[f"low_{level}"].to_numpy(int)
        differences = high - low
        n10 = int(((high == 1) & (low == 0)).sum())
        n01 = int(((high == 0) & (low == 1)).sum())
        odds = (n10 + 0.5) / (n01 + 0.5)
        pvalue = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
        bootstrap = np.array([
            differences[rng.integers(0, len(differences), size=len(differences))].mean()
            for _ in range(BOOTSTRAPS)
        ])
        rows.append({
            "annotation_level": level, "matched_pairs": len(detail),
            "local_high_concordance": float(high.mean()), "local_low_concordance": float(low.mean()),
            "matched_risk_difference": float(differences.mean()),
            "risk_difference_ci_lower": float(np.percentile(bootstrap, 2.5)),
            "risk_difference_ci_upper": float(np.percentile(bootstrap, 97.5)),
            "discordant_high_positive": n10, "discordant_low_positive": n01,
            "matched_conditional_odds_ratio": odds, "exact_mcnemar_p": pvalue,
            "local_low_cut_validation": low_cut, "local_high_cut_validation": high_cut,
            "matching_split": "population_test", "threshold_selection_split": "population_validation",
            "matching_controls": "global propensity decile + Pfam/CATH match + taxonomy; sequence/ESM/Foldseek calipers",
        })
    output = pd.DataFrame(rows)
    output["mcnemar_fdr_bh"] = bh_fdr(output["exact_mcnemar_p"])
    return output, detail


def family_effects(frame: pd.DataFrame, predictions: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    test = frame.loc[
        frame["pair_set"].eq("population") & frame["quality_matched"] & frame["query_split"].eq("test")
    ].reset_index(drop=True)
    rows: list[dict] = []
    for level, target in TARGETS.items():
        y = test[target].astype(int).to_numpy()
        base = predictions[level]["GLOBAL_ONLY"]
        full = predictions[level]["GLOBAL_PLUS_L3"]
        for family, indices in test.groupby("query_primary_pfam", dropna=True).indices.items():
            index = np.asarray(indices, dtype=int)
            if len(index) < 20:
                continue
            base_values = np.clip(base[index], 1e-7, 1 - 1e-7)
            full_values = np.clip(full[index], 1e-7, 1 - 1e-7)
            base_log_loss = float(-np.mean(y[index] * np.log(base_values) + (1 - y[index]) * np.log(1 - base_values)))
            full_log_loss = float(-np.mean(y[index] * np.log(full_values) + (1 - y[index]) * np.log(1 - full_values)))
            if len(np.unique(y[index])) >= 2:
                base_auprc, _ = metric_values(y[index], base[index])
                full_auprc, _ = metric_values(y[index], full[index])
            else:
                base_auprc, full_auprc = float("nan"), float("nan")
            delta_loss = base_log_loss - full_log_loss
            rows.append({
                "family_type": "Pfam_primary", "family_id": family, "annotation_level": level,
                "test_rows": len(index), "positives": int(y[index].sum()),
                "global_auprc": base_auprc, "global_plus_local_auprc": full_auprc,
                "delta_auprc": full_auprc - base_auprc,
                "global_log_loss": base_log_loss, "global_plus_local_log_loss": full_log_loss,
                "delta_log_loss": delta_loss,
                "effect_direction": "POSITIVE" if delta_loss > 0 else ("NEGATIVE" if delta_loss < 0 else "NULL"),
            })
    return pd.DataFrame(rows)


def af_pdb_robustness(root: Path) -> pd.DataFrame:
    data = pd.read_parquet(root / "data/processed/af_pdb_robustness_input.parquet")
    rows: list[dict] = []
    for radius in [6, 8, 10]:
        for descriptor in ["aa_composition_cosine", "property_cosine", "radial_cosine", "octant_cosine", "sidechain_direction_cosine", "residue_count_ratio"]:
            column = f"af_pdb_{descriptor}_r{radius}"
            values = pd.to_numeric(data[column], errors="coerce").dropna()
            quality = pd.to_numeric(data.loc[values.index, f"af_pdb_af_center_plddt_r{radius}"], errors="coerce")
            correlation = spearmanr(quality, values, nan_policy="omit")
            rows.append({
                "analysis": "AF_PDB_REFERENCE_SITE_DESCRIPTOR_AGREEMENT", "descriptor": descriptor,
                "radius_angstrom": radius, "site_rows": len(values),
                "median_similarity": float(values.median()), "q25": float(values.quantile(.25)),
                "q75": float(values.quantile(.75)), "mean_similarity": float(values.mean()),
                "plddt_spearman_rho": float(correlation.statistic), "plddt_spearman_p": float(correlation.pvalue),
                "conditional_effect_consistency": "PENDING_PAIRED_QUERY_AF_VS_REFERENCE_PDB_SENSITIVITY",
            })
    output = pd.DataFrame(rows)
    output["plddt_spearman_fdr_bh"] = bh_fdr(output["plddt_spearman_p"])
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    results = root / "results/phase09"
    figures = root / "figures/source_data"
    reports = root / "reports"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    frame = build_frame(root)
    conditional, ablation, predictions = conditional_and_ablation(frame)
    matched, match_detail = matched_analysis(frame)
    family = family_effects(frame, predictions)
    robustness = af_pdb_robustness(root)

    family_consistency: dict[str, dict[str, float]] = {}
    for level, group in family.groupby("annotation_level"):
        deltas = group["delta_log_loss"].dropna()
        test = wilcoxon(deltas) if len(deltas) >= 5 and (deltas != 0).any() else None
        family_consistency[level] = {
            "families": len(deltas),
            "positive_fraction": float((deltas > 0).mean()) if len(deltas) else float("nan"),
            "median_delta_log_loss": float(deltas.median()) if len(deltas) else float("nan"),
            "paired_wilcoxon_p": float(test.pvalue) if test is not None else float("nan"),
        }
    for level, values in family_consistency.items():
        mask = conditional["annotation_level"].eq(level)
        for key, value in values.items():
            conditional.loc[mask, f"family_{key}"] = value

    conditional.to_csv(results / "conditional_models.tsv", sep="\t", index=False)
    ablation.to_csv(results / "local_ablation.tsv", sep="\t", index=False)
    matched.to_csv(results / "matched_pair_analysis.tsv", sep="\t", index=False)
    match_detail.to_parquet(results / "matched_pair_instances.parquet", index=False, compression="zstd")
    family.to_csv(results / "family_local_effects.tsv", sep="\t", index=False)
    robustness.to_csv(results / "af_pdb_robustness.tsv", sep="\t", index=False)

    test_map = frame.loc[
        frame["pair_set"].eq("population") & frame["quality_matched"] & frame["query_split"].eq("test"),
        ROW_KEY + ["query_primary_pfam", "sequence_identity", "esm2_t33_cosine", "foldseek_identity", "pfam_jaccard", "cath_jaccard", "local_full_score", *TARGETS.values()],
    ].copy()
    test_map.to_parquet(figures / "Figure3_global_local_map.parquet", index=False, compression="zstd")
    conditional.to_csv(figures / "Figure3_conditional_models.tsv", sep="\t", index=False)
    ablation.to_csv(figures / "Figure3_local_ablation.tsv", sep="\t", index=False)
    matched.to_csv(figures / "Figure3_matched_analysis.tsv", sep="\t", index=False)
    robustness.to_csv(figures / "Figure3_af_pdb_robustness.tsv", sep="\t", index=False)

    cohort_counts = (
        frame.groupby(["pair_set", "query_split"])["quality_matched"]
        .agg(rows="size", quality_matched_rows="sum").reset_index().to_dict("records")
    )
    summary = {
        "phase": 9, "stage": "local_beyond_global", "status": "PASS",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "site_linked_rows": len(frame), "cohort_counts": cohort_counts,
        "conditional_model_rows": len(conditional), "matched_pairs": len(match_detail),
        "family_effect_rows": len(family), "af_pdb_summary_rows": len(robustness),
        "family_consistency": family_consistency,
        "mapping_success_used_as_feature": False,
        "query_ground_truth_used_in_features": False,
        "go_no_go_status": "PENDING_PAIRED_AF_PDB_SENSITIVITY_AND_FINAL_QC",
    }
    (reports / "phase09_analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
