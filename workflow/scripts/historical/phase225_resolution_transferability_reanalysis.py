#!/usr/bin/env python3
"""Audit Phase8 and recompute nested resolution-transferability estimands.

The default source-audit mode is lightweight and uses only frozen Figure2
source data.  Raw-reanalysis mode requires the identity-locked population pair,
global feature, and family split Parquet files.  No Phase99 path is referenced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 20260819
TARGETS = {
    "EC_L3": "depth_ge_1",
    "EC_L4": "depth_ge_2",
    "EXACT_RHEA": "depth_eq_3",
}
RESOLUTION_ORDER = {"EC_L3": 1, "EC_L4": 2, "EXACT_RHEA": 3}
PAIR_KEY = ["query_protein_id", "reference_protein_id", "reference_activity_id"]
EXPECTED_RAW = {
    "data/processed/population_pairs.parquet": {
        "rows": 1_498_893,
        "sha256": "779af333e224981dec2732dda3f25cc39e9f87824aa8c1a30e74ccfa1391404c",
    },
    "data/processed/global_features.parquet": {
        "rows": 2_998_901,
        "sha256": "e304bd237a063b7335bdc580a9eb0f8c76293e2fb666ebac232e6af13ef6be3c",
    },
    "data/splits/split_family.parquet": {
        "rows": 210_788,
        "sha256": "e67b307995649674149046aa38ccf2105d8fc0c83668161490158c2dbb79c340",
    },
}
HISTORICAL_SOURCE = {
    "figures/source_data/Figure2_transferability_curves.tsv":
        "567a442d5c7380c723acad2d7d743d81e1dfe2b3b8dbda999c75ff37b1e2d405",
    "figures/source_data/Figure2_family_boundaries.tsv":
        "892fd21914c746a9e7d3820aac84402a1ddb47d84cfcd0c7210eb381a98ca195",
    "figures/source_data/Figure2_benchmark_gap.tsv":
        "5422994545f2c894a19506e186916ad65a766c76e000e04303ac115a135c031c",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{text}".encode()).hexdigest()[:8], 16)


def require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def prepare_output(output: Path, names: Iterable[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    collisions = [str(output / name) for name in names if (output / name).exists()]
    if collisions:
        raise RuntimeError(f"Phase225 outputs already exist; no overwrite: {collisions}")


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)


def historical_hierarchy_audit(curves: pd.DataFrame) -> pd.DataFrame:
    key = ["cohort", "query_split", "evidence_metric", "evidence_bin"]
    pivot = curves.pivot_table(
        index=key, columns="annotation_level", values="transfer_probability", aggfunc="first"
    ).reset_index()
    require_columns(pivot, TARGETS, "historical curve pivot")
    pivot["ec3_ge_ec4"] = pivot["EC_L3"] + 1e-12 >= pivot["EC_L4"]
    pivot["ec4_ge_exact_rhea"] = pivot["EC_L4"] + 1e-12 >= pivot["EXACT_RHEA"]
    pivot["nested_point_estimates"] = pivot["ec3_ge_ec4"] & pivot["ec4_ge_exact_rhea"]
    pivot["violation_type"] = np.select(
        [~pivot["ec3_ge_ec4"], ~pivot["ec4_ge_exact_rhea"]],
        ["EC3_LT_EC4", "EC4_LT_EXACT_RHEA"],
        default="NONE",
    )
    return pivot


def random_strict_gap(
    benchmark: pd.DataFrame,
    target_semantics: str = "SEPARATE_NON_NESTED_INDICATORS",
) -> pd.DataFrame:
    require_columns(
        benchmark,
        ["benchmark_design", "annotation_level", "positive_rate", "sequence_identity_auprc",
         "sequence_identity_auroc", "pair_rows", "query_proteins", "query_clusters"],
        "historical benchmark",
    )
    indexed = benchmark.set_index(["benchmark_design", "annotation_level"])
    rows: list[dict] = []
    for level in TARGETS:
        random = indexed.loc[("random_protein_split", level)]
        strict = indexed.loc[("sequence_cluster_split", level)]
        random_ap = float(random["sequence_identity_auprc"])
        strict_ap = float(strict["sequence_identity_auprc"])
        rows.append({
            "annotation_level": level,
            "resolution_order": RESOLUTION_ORDER[level],
            "random_auprc": random_ap,
            "strict_auprc": strict_ap,
            "random_minus_strict_auprc": random_ap - strict_ap,
            "strict_to_random_auprc_ratio": strict_ap / random_ap if random_ap else math.nan,
            "relative_auprc_drop": (random_ap - strict_ap) / random_ap if random_ap else math.nan,
            "random_positive_rate": float(random["positive_rate"]),
            "strict_positive_rate": float(strict["positive_rate"]),
            "random_pair_rows": int(random["pair_rows"]),
            "strict_pair_rows": int(strict["pair_rows"]),
            "random_query_clusters": int(random["query_clusters"]),
            "strict_query_clusters": int(strict["query_clusters"]),
            "target_semantics": target_semantics,
        })
    return pd.DataFrame(rows)


def family_feasibility(boundaries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "family_id", "evidence_metric", "annotation_level", "target_precision",
        "boundary_status", "selection_split", "test_used_for_selection", "test_ci_lower",
        "validation_clusters", "test_clusters",
    ]
    require_columns(boundaries, required, "historical family boundaries")
    work = boundaries.copy()
    for column in ["target_precision", "test_ci_lower", "validation_clusters", "test_clusters"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["validation_supported"] = work["boundary_status"].eq("VALIDATED_BOUNDARY")
    work["heldout_confirmed"] = (
        work["validation_supported"]
        & work["test_ci_lower"].ge(work["target_precision"])
        & work["test_clusters"].ge(5)
    )
    family_rows = work.loc[~work["family_id"].eq("ALL_FAMILIES")].copy()
    total_families = int(family_rows["family_id"].nunique())
    rows: list[dict] = []
    for (level, metric, target), group in family_rows.groupby(
        ["annotation_level", "evidence_metric", "target_precision"], sort=True
    ):
        rows.append({
            "annotation_level": level,
            "resolution_order": RESOLUTION_ORDER[str(level)],
            "evidence_metric": metric,
            "target_precision": float(target),
            "candidate_families": total_families,
            "validation_supported_boundaries": int(group["validation_supported"].sum()),
            "validation_supported_families": int(group.loc[group["validation_supported"], "family_id"].nunique()),
            "heldout_confirmed_boundaries": int(group["heldout_confirmed"].sum()),
            "heldout_confirmed_families": int(group.loc[group["heldout_confirmed"], "family_id"].nunique()),
            "test_selected_thresholds": int(group["test_used_for_selection"].astype(str).str.lower().eq("true").sum()),
        })
    details = work.loc[work["validation_supported"]].copy()
    keep = [
        "family_id", "evidence_metric", "annotation_level", "target_precision",
        "minimum_evidence_threshold", "validation_precision", "validation_ci_lower",
        "validation_ci_upper", "validation_pairs", "validation_clusters", "test_precision",
        "test_ci_lower", "test_ci_upper", "test_pairs", "test_clusters", "heldout_confirmed",
    ]
    return pd.DataFrame(rows), details[keep]


def source_audit(root: Path, output: Path) -> dict:
    names = [
        "phase225_historical_curve_inventory.tsv",
        "phase225_hierarchy_consistency_audit.tsv",
        "phase225_random_vs_strict_gap.tsv",
        "phase225_family_boundary_feasibility.tsv",
        "phase225_validation_supported_boundary_details.tsv",
        "phase225_input_identity.tsv",
        "phase225_summary.json",
        "phase225_report.md",
    ]
    prepare_output(output, names)
    identity_rows = []
    for relative, expected in HISTORICAL_SOURCE.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing historical source: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"historical source identity mismatch: {relative}")
        identity_rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": observed, "status": "PASS"})

    curves = pd.read_csv(root / "figures/source_data/Figure2_transferability_curves.tsv", sep="\t")
    boundaries = pd.read_csv(root / "figures/source_data/Figure2_family_boundaries.tsv", sep="\t")
    benchmark = pd.read_csv(root / "figures/source_data/Figure2_benchmark_gap.tsv", sep="\t")
    require_columns(curves, ["cohort", "query_split", "evidence_metric", "evidence_bin",
                             "annotation_level", "transfer_probability", "ci_lower", "ci_upper",
                             "bootstrap_unit", "inverse_probability_weighted"], "historical curves")
    if set(curves["annotation_level"]) != set(TARGETS):
        raise RuntimeError("historical curves do not contain the exact three annotation levels")
    inventory = curves.copy()
    inventory.insert(5, "resolution_order", inventory["annotation_level"].map(RESOLUTION_ORDER))
    inventory["historical_target_semantics"] = "SEPARATE_NON_NESTED_INDICATORS"
    hierarchy = historical_hierarchy_audit(curves)
    gap = random_strict_gap(benchmark)
    feasibility, details = family_feasibility(boundaries)
    write_tsv(inventory, output / names[0])
    write_tsv(hierarchy, output / names[1])
    write_tsv(gap, output / names[2])
    write_tsv(feasibility, output / names[3])
    write_tsv(details, output / names[4])
    write_tsv(pd.DataFrame(identity_rows), output / names[5])

    violations = int((~hierarchy["nested_point_estimates"]).sum())
    validation_supported = int(details.shape[0])
    heldout_confirmed = int(details["heldout_confirmed"].sum())
    distinct_supported = int(details.loc[details["family_id"].ne("ALL_FAMILIES"), "family_id"].nunique())
    summary = {
        "format": "siteguard.phase225.current-transferability-source-audit.v1",
        "phase": 225,
        "date": "2026-09-02",
        "status": "PASS_PHASE225_CURRENT_PHASE8_SOURCE_AUDIT_HIERARCHY_CORRECTION_REQUIRED_RAW_REANALYSIS_NOT_EXECUTED",
        "historical_curve_rows": int(len(curves)),
        "complete_curve_groups": int(len(hierarchy)),
        "hierarchy_violating_groups": violations,
        "ec3_lt_ec4_groups": int((~hierarchy["ec3_ge_ec4"]).sum()),
        "ec4_lt_exact_rhea_groups": int((~hierarchy["ec4_ge_exact_rhea"]).sum()),
        "historical_family_boundary_rows": int(len(boundaries)),
        "validation_supported_boundaries": validation_supported,
        "heldout_confirmed_boundaries": heldout_confirmed,
        "distinct_supported_families": distinct_supported,
        "candidate_families_excluding_pooled": int(boundaries.loc[boundaries["family_id"].ne("ALL_FAMILIES"), "family_id"].nunique()),
        "target_95_validation_supported_boundaries": int(details["target_precision"].eq(0.95).sum()),
        "random_vs_strict": {
            row["annotation_level"]: {
                "random_auprc": row["random_auprc"],
                "strict_auprc": row["strict_auprc"],
                "absolute_gap": row["random_minus_strict_auprc"],
                "relative_drop": row["relative_auprc_drop"],
            }
            for row in gap.to_dict("records")
        },
        "raw_identity_locked_inputs_present_locally": all((root / relative).is_file() for relative in EXPECTED_RAW),
        "raw_reanalysis_executed": False,
        "phase99_paths_read": False,
        "raw_data_modified": False,
        "submission_ready": False,
    }
    (output / names[6]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = f"""# Phase225 current transferability evidence audit

Status: **{summary['status']}**

## Evidence retained

- Historical Phase8 contains {len(curves):,} curve rows from a registered
  {1_498_893:,}-row population/atlas sample and uses 30%-identity query-cluster
  bootstrap intervals.
- The random-minus-strict sequence-identity AUPRC gaps are
  {gap.loc[gap.annotation_level.eq('EC_L3'), 'random_minus_strict_auprc'].iloc[0]:.4f}
  (EC-L3), {gap.loc[gap.annotation_level.eq('EC_L4'), 'random_minus_strict_auprc'].iloc[0]:.4f}
  (EC-L4), and {gap.loc[gap.annotation_level.eq('EXACT_RHEA'), 'random_minus_strict_auprc'].iloc[0]:.4f}
  (Exact Rhea).  This supports the claim that random splitting materially
  overstates sequence-identity performance.
- Phase8 found {validation_supported} validation-supported boundaries across
  {distinct_supported} distinct Pfam families; {heldout_confirmed} boundaries
  retained a held-out cluster-bootstrap lower bound at their target.  No 95%
  validation-supported family boundary was found.

## Correction required

The frozen historical curves use three separate annotation indicators.  Of
{len(hierarchy)} complete evidence-bin groups, {violations} violate the nested
ordering required for `D_obs>=1`, `D_obs>=2`, `D_obs=3`; all observed violations
are EC-L4 below Exact Rhea ({int((~hierarchy['ec4_ge_exact_rhea']).sum())} groups).
The old files remain immutable, but they cannot be relabelled as the final
hierarchical transferability atlas.

## Next hard gate

Run the identity-locked raw Phase225 reanalysis on Lyra, then run the independent
Phase225 auditor.  Until both pass, the article may cite the random-versus-strict
gap and sparse family feasibility, but not a final universal resolution curve or
universal safe identity threshold.

No Phase99 result-bearing path was read or modified.
"""
    (output / names[7]).write_text(report, encoding="utf-8")
    return summary


def weighted_rate(frame: pd.DataFrame, target: str) -> float:
    weights = frame["sample_weight"].to_numpy(float)
    outcomes = frame[target].to_numpy(float)
    denominator = weights.sum()
    return float(np.dot(weights, outcomes) / denominator) if denominator > 0 else math.nan


def block_bootstrap(
    frame: pd.DataFrame, targets: list[str], salt: str, replicates: int,
) -> dict[str, tuple[float, float]]:
    if frame.empty:
        return {target: (math.nan, math.nan) for target in targets}
    work = frame[["query_cluster_id_30", "sample_weight", *targets]].copy()
    for target in targets:
        work[f"num_{target}"] = work["sample_weight"] * work[target].astype(float)
    aggregates = work.groupby("query_cluster_id_30", sort=False).agg(
        denominator=("sample_weight", "sum"),
        **{target: (f"num_{target}", "sum") for target in targets},
    )
    if len(aggregates) < 2:
        return {target: (math.nan, math.nan) for target in targets}
    denominator = aggregates["denominator"].to_numpy(float)
    numerator = aggregates[targets].to_numpy(float)
    rng = np.random.default_rng(stable_seed(salt))
    estimates = np.empty((replicates, len(targets)), dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(aggregates), size=len(aggregates))
        estimates[index] = numerator[sample].sum(axis=0) / denominator[sample].sum()
    return {
        target: (float(np.quantile(estimates[:, i], 0.025)), float(np.quantile(estimates[:, i], 0.975)))
        for i, target in enumerate(targets)
    }


def make_nested_curves(frame: pd.DataFrame, replicates: int) -> pd.DataFrame:
    definitions = {
        "sequence_identity": ([0, .2, .3, .4, .5, .6, .8, 1.000001],
                              ["00-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80-100"]),
        "foldseek_identity": ([0, .1, .2, .3, .5, 1.000001],
                              ["00-10", "10-20", "20-30", "30-50", "50-100"]),
        "esm2_t33_cosine": ([-1, .3, .5, .6, .7, .8, .9, .95, 1.000001],
                            ["<-0.30", "30-50", "50-60", "60-70", "70-80", "80-90", "90-95", "95-100"]),
    }
    rows: list[dict] = []
    for metric, (edges, labels) in definitions.items():
        valid = frame.loc[frame[metric].notna()].copy()
        valid["evidence_bin"] = pd.cut(valid[metric], edges, labels=labels, right=False, include_lowest=True)
        for (split, evidence_bin), group in valid.groupby(["query_split_expected", "evidence_bin"], observed=True):
            intervals = block_bootstrap(group, list(TARGETS.values()), f"curve:{metric}:{split}:{evidence_bin}", replicates)
            for level, target in TARGETS.items():
                lower, upper = intervals[target]
                rows.append({
                    "cohort": "POPULATION_ATLAS",
                    "query_split": split,
                    "evidence_metric": metric,
                    "evidence_bin": str(evidence_bin),
                    "annotation_level": level,
                    "resolution_order": RESOLUTION_ORDER[level],
                    "transfer_probability": weighted_rate(group, target),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "pair_rows": len(group),
                    "query_proteins": group["query_protein_id"].nunique(),
                    "query_clusters": group["query_cluster_id_30"].nunique(),
                    "inverse_probability_weighted": True,
                    "bootstrap_unit": "query_sequence_cluster_30",
                    "target_definition": {"EC_L3": "D_obs>=1", "EC_L4": "D_obs>=2", "EXACT_RHEA": "D_obs==3"}[level],
                })
    return pd.DataFrame(rows)


def deterministic_random_partition(protein_id: str) -> str:
    value = int(hashlib.sha256(f"SiteGuardV4-random-split:{SEED}:{protein_id}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if value < 0.8 else ("validation" if value < 0.9 else "test")


def nested_benchmark(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    protein_ids = pd.Index(pd.concat([frame["query_protein_id"], frame["reference_protein_id"]]).unique())
    random_map = {protein: deterministic_random_partition(str(protein)) for protein in protein_ids}
    query_random = frame["query_protein_id"].map(random_map)
    reference_random = frame["reference_protein_id"].map(random_map)
    cohorts = {
        "random_protein_split": query_random.eq("test") & reference_random.eq("train"),
        "sequence_cluster_split": frame["query_split_expected"].eq("test") & frame["different_sequence_cluster_30"].fillna(False),
    }
    rows: list[dict] = []
    for design, mask in cohorts.items():
        group = frame.loc[mask & frame["sequence_identity"].notna()]
        for level, target in TARGETS.items():
            outcome = group[target].astype(int).to_numpy()
            score = group["sequence_identity"].to_numpy(float)
            weight = group["sample_weight"].to_numpy(float)
            estimable = len(group) > 0 and len(np.unique(outcome)) == 2
            rows.append({
                "benchmark_design": design,
                "annotation_level": level,
                "resolution_order": RESOLUTION_ORDER[level],
                "pair_rows": len(group),
                "query_proteins": group["query_protein_id"].nunique(),
                "query_clusters": group["query_cluster_id_30"].nunique(),
                "positive_rate": weighted_rate(group, target) if len(group) else math.nan,
                "sequence_identity_auprc": float(average_precision_score(outcome, score, sample_weight=weight)) if estimable else math.nan,
                "sequence_identity_auroc": float(roc_auc_score(outcome, score, sample_weight=weight)) if estimable else math.nan,
                "target_definition": {"EC_L3": "D_obs>=1", "EC_L4": "D_obs>=2", "EXACT_RHEA": "D_obs==3"}[level],
            })
    benchmark = pd.DataFrame(rows)
    return benchmark, random_strict_gap(benchmark, "NESTED_D_OBS_ESTIMANDS")


def wilson_lower(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    if trials <= 0:
        return math.nan
    p = successes / trials
    denominator = 1 + z * z / trials
    center = p + z * z / (2 * trials)
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return (center - margin) / denominator


def nested_family_boundaries(frame: pd.DataFrame, replicates: int) -> pd.DataFrame:
    validation = frame.loc[frame["query_split_expected"].eq("validation")]
    test = frame.loc[frame["query_split_expected"].eq("test")]
    counts = validation.groupby("query_primary_pfam", dropna=True).agg(
        rows=("query_protein_id", "size"), clusters=("query_cluster_id_30", "nunique")
    )
    eligible = counts.loc[(counts["rows"] >= 100) & (counts["clusters"] >= 5)].index.tolist()
    thresholds = {
        "sequence_identity": [v / 100 for v in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]],
        "foldseek_identity": [v / 100 for v in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]],
    }
    rows: list[dict] = []
    for family in eligible:
        val_family = validation.loc[validation["query_primary_pfam"].eq(family)]
        test_family = test.loc[test["query_primary_pfam"].eq(family)]
        for metric, candidates in thresholds.items():
            for level, target in TARGETS.items():
                for desired in [0.90, 0.95]:
                    selected = None
                    for threshold in candidates:
                        group = val_family.loc[val_family[metric].notna() & val_family[metric].ge(threshold)]
                        if len(group) < 50 or group["query_cluster_id_30"].nunique() < 5:
                            continue
                        if wilson_lower(int(group[target].sum()), len(group)) < desired:
                            continue
                        low, high = block_bootstrap(group, [target], f"family:{family}:{metric}:{level}:{desired}:{threshold}", replicates)[target]
                        if math.isfinite(low) and low >= desired:
                            selected = (threshold, weighted_rate(group, target), low, high, len(group), group["query_cluster_id_30"].nunique())
                            break
                    row = {
                        "family_id": family, "evidence_metric": metric,
                        "annotation_level": level, "resolution_order": RESOLUTION_ORDER[level],
                        "target_precision": desired,
                        "boundary_status": "VALIDATED_BOUNDARY" if selected else "NO_VALIDATED_BOUNDARY",
                        "minimum_evidence_threshold": selected[0] if selected else math.nan,
                        "validation_precision": selected[1] if selected else math.nan,
                        "validation_ci_lower": selected[2] if selected else math.nan,
                        "validation_ci_upper": selected[3] if selected else math.nan,
                        "validation_pairs": selected[4] if selected else 0,
                        "validation_clusters": selected[5] if selected else 0,
                        "selection_split": "validation", "test_used_for_selection": False,
                    }
                    if selected:
                        group = test_family.loc[test_family[metric].notna() & test_family[metric].ge(selected[0])]
                        low, high = block_bootstrap(group, [target], f"family-test:{family}:{metric}:{level}:{desired}", replicates)[target]
                        row.update({"test_precision": weighted_rate(group, target), "test_ci_lower": low,
                                    "test_ci_upper": high, "test_pairs": len(group),
                                    "test_clusters": group["query_cluster_id_30"].nunique()})
                    else:
                        row.update({"test_precision": math.nan, "test_ci_lower": math.nan,
                                    "test_ci_upper": math.nan, "test_pairs": 0, "test_clusters": 0})
                    rows.append(row)
    return pd.DataFrame(rows)


def raw_reanalysis(root: Path, output: Path, bootstraps: int, family_bootstraps: int) -> dict:
    names = [
        "phase225_nested_transferability_curves.tsv", "phase225_nested_benchmark.tsv",
        "phase225_random_vs_strict_gap.tsv", "phase225_nested_family_boundaries.tsv",
        "phase225_family_boundary_feasibility.tsv", "phase225_input_identity.tsv",
        "phase225_summary.json",
    ]
    prepare_output(output, names)
    identities = []
    for relative, expected in EXPECTED_RAW.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"required raw-reanalysis input absent: {relative}")
        observed = sha256_file(path)
        if observed != expected["sha256"]:
            raise RuntimeError(f"identity mismatch: {relative}")
        identities.append({"path": relative, "bytes": path.stat().st_size, "sha256": observed, "status": "PASS"})

    population_columns = [
        *PAIR_KEY, "pair_set", "query_split_expected", "query_cluster_id_30",
        "query_primary_pfam", "different_sequence_cluster_30", "observed_concordance_depth",
        "sample_weight", "sampling_probability",
    ]
    population = pd.read_parquet(root / "data/processed/population_pairs.parquet", columns=population_columns)
    if len(population) != EXPECTED_RAW["data/processed/population_pairs.parquet"]["rows"]:
        raise RuntimeError("population row-count mismatch")
    if not set(population["pair_set"].astype(str)).issubset({"population_atlas", "population"}):
        raise RuntimeError("non-population pair_set found")
    if population[PAIR_KEY].duplicated().any():
        raise RuntimeError("population content key is not unique")
    if population["sample_weight"].isna().any() or (population["sample_weight"] <= 0).any():
        raise RuntimeError("invalid population sample weights")
    depth = pd.to_numeric(population["observed_concordance_depth"], errors="raise").astype(int)
    if not depth.isin([0, 1, 2, 3]).all():
        raise RuntimeError("invalid observed_concordance_depth")
    population["depth_ge_1"] = depth.ge(1)
    population["depth_ge_2"] = depth.ge(2)
    population["depth_eq_3"] = depth.eq(3)

    family_split = pd.read_parquet(
        root / "data/splits/split_family.parquet",
        columns=["protein_id", "primary_pfam", "pfam_family_split"],
    )
    if len(family_split) != EXPECTED_RAW["data/splits/split_family.parquet"]["rows"]:
        raise RuntimeError("family-split row-count mismatch")
    if family_split["protein_id"].duplicated().any():
        raise RuntimeError("family-split protein key is not unique")
    pfam_map = family_split.set_index("protein_id")["primary_pfam"]
    registered_pfam = population["query_protein_id"].map(pfam_map)
    if registered_pfam.isna().any():
        raise RuntimeError("population query missing from frozen family split")
    pair_pfam = population["query_primary_pfam"].fillna("UNASSIGNED").astype(str)
    split_pfam = registered_pfam.fillna("UNASSIGNED").astype(str)
    if not pair_pfam.eq(split_pfam).all():
        raise RuntimeError("population query Pfam differs from the pre-pair family split")

    features = pd.read_parquet(
        root / "data/processed/global_features.parquet",
        columns=["pair_set", *PAIR_KEY, "sequence_identity", "foldseek_identity", "esm2_t33_cosine"],
        filters=[("pair_set", "==", "population")],
    )
    features = features.loc[features["pair_set"].eq("population")].drop(columns="pair_set")
    if features[PAIR_KEY].duplicated().any():
        raise RuntimeError("population global-feature key is not unique")
    frame = population.merge(features, on=PAIR_KEY, validate="one_to_one")
    if len(frame) != len(population):
        raise RuntimeError("population/global-feature merge changed row count")

    curves = make_nested_curves(frame, bootstraps)
    benchmark, gap = nested_benchmark(frame)
    boundaries = nested_family_boundaries(frame, family_bootstraps)
    feasibility, _ = family_feasibility(boundaries)
    write_tsv(curves, output / names[0])
    write_tsv(benchmark, output / names[1])
    write_tsv(gap, output / names[2])
    write_tsv(boundaries, output / names[3])
    write_tsv(feasibility, output / names[4])
    write_tsv(pd.DataFrame(identities), output / names[5])
    hierarchy = historical_hierarchy_audit(curves)
    violations = int((~hierarchy["nested_point_estimates"]).sum())
    if violations:
        raise RuntimeError(f"nested reanalysis produced {violations} hierarchy violations")
    supported = int(boundaries["boundary_status"].eq("VALIDATED_BOUNDARY").sum())
    confirmed = int((boundaries["boundary_status"].eq("VALIDATED_BOUNDARY") &
                     boundaries["test_ci_lower"].ge(boundaries["target_precision"])).sum())
    summary = {
        "format": "siteguard.phase225.nested-resolution-transferability-reanalysis.v1",
        "phase": 225,
        "status": "PASS_PHASE225_NESTED_RESOLUTION_TRANSFERABILITY_REANALYSIS_PENDING_INDEPENDENT_AUDIT",
        "seed": SEED,
        "population_rows": len(frame),
        "curve_rows": len(curves),
        "complete_curve_groups": len(hierarchy),
        "hierarchy_violating_groups": 0,
        "bootstrap_replicates": bootstraps,
        "family_bootstrap_replicates": family_bootstraps,
        "bootstrap_unit": "query_sequence_cluster_30",
        "validation_supported_boundaries": supported,
        "heldout_confirmed_boundaries": confirmed,
        "test_used_for_threshold_selection": False,
        "phase99_paths_read": False,
        "raw_data_modified": False,
    }
    (output / names[6]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["source-audit", "raw-reanalysis"], required=True)
    parser.add_argument("--bootstraps", type=int, default=200)
    parser.add_argument("--family-bootstraps", type=int, default=300)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    if args.bootstraps < 20 or args.family_bootstraps < 20:
        raise RuntimeError("at least 20 bootstrap replicates are required")
    summary = (
        source_audit(root, output)
        if args.mode == "source-audit"
        else raw_reanalysis(root, output, args.bootstraps, args.family_bootstraps)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
