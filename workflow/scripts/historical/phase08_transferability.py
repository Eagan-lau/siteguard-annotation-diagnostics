#!/usr/bin/env python3
"""Estimate the empirical resolution-dependent annotation-transfer landscape."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


TARGETS = {
    "EC_L3": "same_ec_l3",
    "EC_L4": "same_ec_l4",
    "EXACT_RHEA": "same_exact_rhea",
}
ROW_KEY = ["pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id"]
SEED = 20260820
BOOTSTRAPS = 200
FAMILY_BOOTSTRAPS = 300


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{text}".encode()).hexdigest()[:8], 16)


def weighted_rate(frame: pd.DataFrame, target: str) -> float:
    weight = frame["sample_weight"].to_numpy(float)
    outcome = frame[target].to_numpy(float)
    return float(np.dot(weight, outcome) / weight.sum()) if weight.sum() else float("nan")


def block_bootstrap_rates(
    frame: pd.DataFrame, target_columns: list[str], salt: str, replicates: int = BOOTSTRAPS,
) -> dict[str, tuple[float, float]]:
    if frame.empty:
        return {target: (float("nan"), float("nan")) for target in target_columns}
    work = frame[["query_cluster_id_30", "sample_weight", *target_columns]].copy()
    for target in target_columns:
        work[f"__{target}"] = work["sample_weight"] * work[target].astype(float)
    aggregate = work.groupby("query_cluster_id_30", sort=False).agg(
        denominator=("sample_weight", "sum"),
        **{target: (f"__{target}", "sum") for target in target_columns},
    )
    count = len(aggregate)
    if count < 2:
        return {target: (float("nan"), float("nan")) for target in target_columns}
    denominator = aggregate["denominator"].to_numpy(float)
    numerators = aggregate[target_columns].to_numpy(float)
    rng = np.random.default_rng(stable_seed(salt))
    values = np.empty((replicates, len(target_columns)), dtype=np.float64)
    batch = 10
    for start in range(0, replicates, batch):
        size = min(batch, replicates - start)
        indices = rng.integers(0, count, size=(size, count))
        sampled_denominator = denominator[indices].sum(axis=1)
        sampled_numerator = numerators[indices].sum(axis=1)
        values[start:start + size] = sampled_numerator / sampled_denominator[:, None]
    return {
        target: (
            float(np.clip(np.percentile(values[:, index], 2.5), 0.0, 1.0)),
            float(np.clip(np.percentile(values[:, index], 97.5), 0.0, 1.0)),
        )
        for index, target in enumerate(target_columns)
    }


def wilson_lower(successes: float, trials: int, z: float = 1.959963984540054) -> float:
    if trials <= 0:
        return float("nan")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = proportion + z * z / (2.0 * trials)
    margin = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
    return (center - margin) / denominator


def make_curve_rows(
    frame: pd.DataFrame,
    metric: str,
    edges: list[float],
    labels: list[str],
    cohort: str,
) -> list[dict]:
    valid = frame.loc[frame[metric].notna()].copy()
    valid["evidence_bin"] = pd.cut(valid[metric], bins=edges, labels=labels, right=False, include_lowest=True)
    rows: list[dict] = []
    for (split, evidence_bin), group in valid.groupby(["query_split_expected", "evidence_bin"], observed=True):
        intervals = block_bootstrap_rates(group, list(TARGETS.values()), f"curve:{cohort}:{metric}:{split}:{evidence_bin}")
        for level, target in TARGETS.items():
            lower, upper = intervals[target]
            rows.append({
                "cohort": cohort,
                "query_split": split,
                "evidence_metric": metric,
                "evidence_bin": str(evidence_bin),
                "bin_lower": float(group[metric].min()),
                "bin_upper": float(group[metric].max()),
                "annotation_level": level,
                "transfer_probability": weighted_rate(group, target),
                "ci_lower": lower,
                "ci_upper": upper,
                "pair_rows": len(group),
                "query_proteins": group["query_protein_id"].nunique(),
                "query_clusters": group["query_cluster_id_30"].nunique(),
                "inverse_probability_weighted": cohort == "POPULATION_ATLAS",
                "bootstrap_unit": "query_sequence_cluster_30",
            })
    return rows


def threshold_rows(frame: pd.DataFrame, metric: str, thresholds: list[float], cohort: str) -> list[dict]:
    rows: list[dict] = []
    for split in ["validation", "test"]:
        split_frame = frame.loc[frame["query_split_expected"].eq(split) & frame[metric].notna()].copy()
        total_weight = float(split_frame["sample_weight"].sum())
        for threshold in thresholds:
            group = split_frame.loc[split_frame[metric].ge(threshold)]
            if len(group) < 20:
                continue
            intervals = block_bootstrap_rates(group, list(TARGETS.values()), f"threshold:{cohort}:{metric}:{split}:{threshold}")
            for level, target in TARGETS.items():
                lower, upper = intervals[target]
                rows.append({
                    "cohort": cohort,
                    "query_split": split,
                    "evidence_metric": metric,
                    "minimum_threshold": threshold,
                    "annotation_level": level,
                    "precision": weighted_rate(group, target),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "coverage": float(group["sample_weight"].sum() / total_weight) if total_weight else float("nan"),
                    "pair_rows": len(group),
                    "query_clusters": group["query_cluster_id_30"].nunique(),
                    "bootstrap_unit": "query_sequence_cluster_30",
                    "threshold_selected_on_test": False,
                })
    return rows


def test_at_boundary(frame: pd.DataFrame, metric: str, threshold: float, target: str, salt: str) -> dict:
    group = frame.loc[frame[metric].notna() & frame[metric].ge(threshold)]
    if len(group) < 20:
        return {"precision": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "pairs": len(group), "clusters": group["query_cluster_id_30"].nunique()}
    lower, upper = block_bootstrap_rates(group, [target], salt, FAMILY_BOOTSTRAPS)[target]
    return {"precision": weighted_rate(group, target), "ci_lower": lower, "ci_upper": upper, "pairs": len(group), "clusters": group["query_cluster_id_30"].nunique()}


def family_boundaries(frame: pd.DataFrame) -> pd.DataFrame:
    validation = frame.loc[frame["query_split_expected"].eq("validation")].copy()
    test = frame.loc[frame["query_split_expected"].eq("test")].copy()
    metric_thresholds = {
        "sequence_identity": [value / 100 for value in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]],
        "foldseek_identity": [value / 100 for value in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]],
    }
    families = validation.groupby("query_primary_pfam", dropna=True).agg(
        rows=("query_protein_id", "size"), clusters=("query_cluster_id_30", "nunique")
    )
    eligible = families.loc[(families["rows"] >= 100) & (families["clusters"] >= 5)].index.tolist()
    family_values: list[str] = ["ALL_FAMILIES", *eligible]
    rows: list[dict] = []
    for family in family_values:
        validation_family = validation if family == "ALL_FAMILIES" else validation.loc[validation["query_primary_pfam"].eq(family)]
        test_family = test if family == "ALL_FAMILIES" else test.loc[test["query_primary_pfam"].eq(family)]
        for metric, thresholds in metric_thresholds.items():
            valid_family = validation_family.loc[validation_family[metric].notna()]
            for level, target in TARGETS.items():
                for desired in [0.90, 0.95]:
                    selected: dict | None = None
                    tested_candidates = 0
                    for threshold in thresholds:
                        candidate = valid_family.loc[valid_family[metric].ge(threshold)]
                        if len(candidate) < 50 or candidate["query_cluster_id_30"].nunique() < 5:
                            continue
                        successes = int(candidate[target].sum())
                        if wilson_lower(successes, len(candidate)) < desired:
                            continue
                        tested_candidates += 1
                        lower, upper = block_bootstrap_rates(
                            candidate, [target], f"family:{family}:{metric}:{level}:{desired}:{threshold}", FAMILY_BOOTSTRAPS,
                        )[target]
                        if math.isfinite(lower) and lower >= desired:
                            selected = {
                                "threshold": threshold,
                                "precision": weighted_rate(candidate, target),
                                "ci_lower": lower,
                                "ci_upper": upper,
                                "pairs": len(candidate),
                                "clusters": candidate["query_cluster_id_30"].nunique(),
                            }
                            break
                    row = {
                        "family_type": "Pfam_primary",
                        "family_id": family,
                        "evidence_metric": metric,
                        "annotation_level": level,
                        "target_precision": desired,
                        "boundary_status": "VALIDATED_BOUNDARY" if selected else "NO_VALIDATED_BOUNDARY",
                        "minimum_evidence_threshold": selected["threshold"] if selected else float("nan"),
                        "validation_precision": selected["precision"] if selected else float("nan"),
                        "validation_ci_lower": selected["ci_lower"] if selected else float("nan"),
                        "validation_ci_upper": selected["ci_upper"] if selected else float("nan"),
                        "validation_pairs": selected["pairs"] if selected else 0,
                        "validation_clusters": selected["clusters"] if selected else 0,
                        "candidate_thresholds_bootstrapped": tested_candidates,
                        "selection_split": "validation",
                        "test_used_for_selection": False,
                    }
                    if selected:
                        heldout = test_at_boundary(
                            test_family, metric, selected["threshold"], target,
                            f"family_test:{family}:{metric}:{level}:{desired}",
                        )
                    else:
                        heldout = {"precision": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "pairs": 0, "clusters": 0}
                    row.update({f"test_{name}": value for name, value in heldout.items()})
                    rows.append(row)
    return pd.DataFrame(rows)


def deterministic_random_partition(protein_id: str) -> str:
    value = int(hashlib.sha256(f"SiteGuardV4-random-split:{SEED}:{protein_id}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if value < 0.8 else ("validation" if value < 0.9 else "test")


def benchmark_rows(frame: pd.DataFrame, split_family: pd.DataFrame, temporal: pd.DataFrame) -> pd.DataFrame:
    protein_ids = pd.Index(pd.concat([frame["query_protein_id"], frame["reference_protein_id"]]).unique())
    random_map = {protein: deterministic_random_partition(str(protein)) for protein in protein_ids}
    family_map = split_family.set_index("protein_id")["pfam_family_split"].to_dict()
    temporal_map = temporal.set_index("protein_id")["temporal_membership"].to_dict()
    query_random = frame["query_protein_id"].map(random_map)
    reference_random = frame["reference_protein_id"].map(random_map)
    query_family = frame["query_protein_id"].map(family_map)
    reference_family = frame["reference_protein_id"].map(family_map)
    query_temporal = frame["query_protein_id"].map(temporal_map)
    reference_temporal = frame["reference_protein_id"].map(temporal_map)
    cohorts = {
        "random_protein_split": query_random.eq("test") & reference_random.eq("train"),
        "sequence_cluster_split": frame["query_split_expected"].eq("test") & frame["different_sequence_cluster_30"].fillna(False),
        "pfam_family_holdout": query_family.eq("test") & reference_family.eq("train"),
        "temporal_new_in_T1": query_temporal.isin(["NEW_IN_T1", "NEW_AFTER_T1_OR_UNRESOLVED"]) & reference_temporal.eq("T0_T1_SEQUENCE_STABLE"),
    }
    rows: list[dict] = []
    for cohort, mask in cohorts.items():
        group = frame.loc[mask & frame["sequence_identity"].notna()]
        for level, target in TARGETS.items():
            outcome = group[target].astype(int).to_numpy()
            score = group["sequence_identity"].to_numpy(float)
            weight = group["sample_weight"].to_numpy(float)
            estimable = len(group) > 0 and len(np.unique(outcome)) == 2
            rows.append({
                "benchmark_design": cohort,
                "annotation_level": level,
                "pair_rows": len(group),
                "query_proteins": group["query_protein_id"].nunique(),
                "query_clusters": group["query_cluster_id_30"].nunique(),
                "positive_rate": weighted_rate(group, target) if len(group) else float("nan"),
                "sequence_identity_auprc": float(average_precision_score(outcome, score, sample_weight=weight)) if estimable else float("nan"),
                "sequence_identity_auroc": float(roc_auc_score(outcome, score, sample_weight=weight)) if estimable else float("nan"),
                "definition": {
                    "random_protein_split": "deterministic 80/10/10 protein hash; random-test query versus random-train reference",
                    "sequence_cluster_split": "registered test query versus train reference in a different MMseqs2 30% cluster",
                    "pfam_family_holdout": "registered Pfam-family test query versus Pfam-family train reference",
                    "temporal_new_in_T1": "new T1/current query versus T0-T1 stable reference",
                }[cohort],
            })
    return pd.DataFrame(rows)


def stratified_rows(frame: pd.DataFrame) -> pd.DataFrame:
    test = frame.loc[frame["query_split_expected"].eq("test") & frame["sequence_identity"].notna()].copy()
    test["ec_class"] = test["ec_l3"].astype(str).str.split(".").str[0]
    test["domain_architecture"] = np.where(test["query_domain_count"].fillna(0).le(1), "single_or_no_Pfam", "multi_Pfam")
    definitions = {
        "EC_class": "ec_class",
        "taxonomy": "query_taxonomy_group",
        "domain_architecture": "domain_architecture",
        "cofactor_class": "reference_cofactor_class",
        "Pfam_primary": "query_primary_pfam",
        "CATH_primary": "query_primary_cath_superfamily",
    }
    rows: list[dict] = []
    for stratum_type, column in definitions.items():
        counts = test[column].value_counts(dropna=False)
        allowed = set(counts[counts >= 100].index)
        for value, group in test.loc[test[column].isin(allowed)].groupby(column, dropna=False):
            for level, target in TARGETS.items():
                rows.append({
                    "query_split": "test", "stratum_type": stratum_type, "stratum": str(value),
                    "annotation_level": level, "pair_rows": len(group),
                    "query_clusters": group["query_cluster_id_30"].nunique(),
                    "transfer_probability": weighted_rate(group, target),
                })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    processed = root / "data/processed"
    results = root / "results/phase08"
    figure_source = root / "figures/source_data"
    reports = root / "reports"
    results.mkdir(parents=True, exist_ok=True)
    figure_source.mkdir(parents=True, exist_ok=True)

    population_columns = [
        "query_protein_id", "reference_protein_id", "reference_activity_id", "pair_set",
        "query_split_expected", "query_cluster_id_30", "query_primary_pfam",
        "query_primary_cath_superfamily", "query_taxonomy_group", "ec_l3",
        "different_sequence_cluster_30", "same_ec_l3", "same_ec_l4", "same_exact_rhea",
        "sample_weight", "sampling_probability",
    ]
    population = pd.read_parquet(processed / "population_pairs.parquet", columns=population_columns)
    features = pd.read_parquet(
        processed / "global_features.parquet",
        columns=ROW_KEY + ["sequence_identity", "foldseek_identity", "esm2_t33_cosine", "reference_cofactor_class"],
        filters=[("pair_set", "==", "population")],
    )
    # Phase 5 calls the registered set ``population_atlas`` whereas Phase 6
    # normalizes the same rows to ``population``.  Resolve that documented
    # naming difference before the one-to-one content-key merge.
    features = features.loc[features["pair_set"].eq("population")].drop(columns="pair_set")
    content_key = ["query_protein_id", "reference_protein_id", "reference_activity_id"]
    frame = population.merge(features, on=content_key, validate="one_to_one")
    if len(frame) != len(population):
        raise RuntimeError("Population/global-feature merge changed the registered row count")

    split_family = pd.read_parquet(root / "data/splits/split_family.parquet")
    domain_counts = split_family.set_index("protein_id")["pfam_domains_json"].map(
        lambda value: len(json.loads(value)) if isinstance(value, str) and value else 0
    ).to_dict()
    frame["query_domain_count"] = frame["query_protein_id"].map(domain_counts)

    curve_rows: list[dict] = []
    curve_rows += make_curve_rows(
        frame, "sequence_identity", [0, .2, .3, .4, .5, .6, .8, 1.000001],
        ["00-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80-100"], "POPULATION_ATLAS",
    )
    curve_rows += make_curve_rows(
        frame, "foldseek_identity", [0, .1, .2, .3, .5, 1.000001],
        ["00-10", "10-20", "20-30", "30-50", "50-100"], "POPULATION_ATLAS",
    )
    curve_rows += make_curve_rows(
        frame, "esm2_t33_cosine", [-1, .3, .5, .6, .7, .8, .9, .95, 1.000001],
        ["<-0.30", "30-50", "50-60", "60-70", "70-80", "80-90", "90-95", "95-100"], "POPULATION_ATLAS",
    )

    usalign = pd.read_parquet(processed / "selected_usalign_calibration.parquet")
    tm = usalign.merge(
        population[["query_protein_id", "reference_protein_id", "reference_activity_id", "query_split_expected", "query_cluster_id_30", "query_primary_pfam", "sample_weight", *TARGETS.values()]],
        on=["query_protein_id", "reference_protein_id"], how="inner",
    )
    curve_rows += make_curve_rows(
        tm, "tm_mean", [0, .3, .4, .5, .6, .7, .8, .9, .95, 1.000001],
        ["00-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-95", "95-100"],
        "LABEL_BLIND_USALIGN_SUBSET",
    )
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(results / "transferability_curves.tsv", sep="\t", index=False)

    threshold_records: list[dict] = []
    common_thresholds = [value / 100 for value in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]]
    threshold_records += threshold_rows(frame, "sequence_identity", common_thresholds, "POPULATION_ATLAS")
    threshold_records += threshold_rows(frame, "foldseek_identity", common_thresholds, "POPULATION_ATLAS")
    threshold_records += threshold_rows(tm, "tm_mean", [value / 100 for value in [0, 30, 40, 50, 60, 70, 80, 90, 95]], "LABEL_BLIND_USALIGN_SUBSET")
    thresholds = pd.DataFrame(threshold_records)
    thresholds.to_csv(results / "threshold_confidence_intervals.tsv", sep="\t", index=False)

    boundaries = family_boundaries(frame)
    boundaries.to_csv(results / "family_boundaries.tsv", sep="\t", index=False)
    temporal = pd.read_parquet(root / "data/splits/split_temporal.parquet", columns=["protein_id", "temporal_membership"])
    benchmark = benchmark_rows(frame, split_family[["protein_id", "pfam_family_split"]], temporal)
    benchmark.to_csv(results / "benchmark_gap.tsv", sep="\t", index=False)
    stratified = stratified_rows(frame)
    stratified.to_csv(results / "transferability_stratified.tsv", sep="\t", index=False)

    curves.to_csv(figure_source / "Figure2_transferability_curves.tsv", sep="\t", index=False)
    boundaries.to_csv(figure_source / "Figure2_family_boundaries.tsv", sep="\t", index=False)
    benchmark.to_csv(figure_source / "Figure2_benchmark_gap.tsv", sep="\t", index=False)

    summary = {
        "phase": 8,
        "stage": "empirical_transferability",
        "status": "PASS",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "population_rows": len(frame),
        "population_query_proteins": frame["query_protein_id"].nunique(),
        "curve_rows": len(curves),
        "threshold_ci_rows": len(thresholds),
        "eligible_family_metric_target_rows": len(boundaries),
        "validated_family_boundaries": int(boundaries["boundary_status"].eq("VALIDATED_BOUNDARY").sum()),
        "usalign_activity_rows": len(tm),
        "bootstrap_replicates": BOOTSTRAPS,
        "family_bootstrap_replicates": FAMILY_BOOTSTRAPS,
        "bootstrap_unit": "query_sequence_cluster_30",
        "population_inverse_probability_weighted": True,
        "test_used_for_threshold_selection": False,
    }
    (reports / "phase08_transferability_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
