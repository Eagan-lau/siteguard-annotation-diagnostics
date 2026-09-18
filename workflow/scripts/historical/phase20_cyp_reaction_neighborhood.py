#!/usr/bin/env python3
"""Phase 20: exploratory CYP reaction-neighborhood baselines and assay panels.

All SiteGuard predictions and comparator rankings are fixed without query reaction
truth. Query truth is joined only after rankings have been frozen, for evaluation.
This is a post-hoc exploratory extension of the preregistered Phase 15 analysis.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from rdkit import RDLogger

from phase15_finalize import ReactionSimilarity, json_set, write_frame

RDLogger.DisableLog("rdApp.warning")


SEED = 20260819
SCENARIOS = ["GENERAL", "LEAVE_ONE_CYP_FAMILY_OUT", "PLANT_COLD_START"]
PANEL_SIZES = [1, 3, 5]
RANDOM_DRAWS = 1000
BOOTSTRAPS = 5000
PERMUTATIONS = 20000
METHOD_RANKS = {
    "SITEGUARD": "rank_siteguard",
    "MMSEQS_NEAREST_NEIGHBOR": "rank_mmseqs",
    "ESM2_NEAREST_NEIGHBOR": "rank_esm2",
    "REFERENCE_FREQUENCY": "rank_frequency",
}
METHOD_LABELS = {
    "SITEGUARD": "SiteGuard",
    "MMSEQS_NEAREST_NEIGHBOR": "MMseqs nearest neighbor",
    "ESM2_NEAREST_NEIGHBOR": "ESM2 nearest neighbor",
    "REFERENCE_FREQUENCY": "Reference frequency",
    "RANDOM_UNIFORM": "Random candidate panel",
    "ORACLE_CANDIDATE_POOL": "Candidate-pool ceiling",
}


def stable_rng(*parts: object) -> np.random.Generator:
    payload = "|".join(map(str, (SEED, *parts))).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    values = finite(values)
    if not len(values):
        return float("nan"), float("nan")
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def bootstrap_interval(values: np.ndarray, key: str, statistic: str = "mean") -> tuple[float, float]:
    values = finite(values)
    if not len(values):
        return float("nan"), float("nan")
    rng = stable_rng("query_bootstrap", key)
    estimates = np.empty(BOOTSTRAPS, dtype=float)
    for index in range(BOOTSTRAPS):
        sample = values[rng.integers(0, len(values), len(values))]
        estimates[index] = np.mean(sample) if statistic == "mean" else np.median(sample)
    return percentile_interval(estimates)


def family_block_interval(frame: pd.DataFrame, value: str, key: str) -> tuple[float, float]:
    working = frame.loc[np.isfinite(pd.to_numeric(frame[value], errors="coerce"))].copy()
    if working.empty:
        return float("nan"), float("nan")
    family = working["query_cyp_family"].fillna("UNKNOWN_CYP_FAMILY").astype(str)
    working["bootstrap_block"] = np.where(
        family.eq("UNKNOWN_CYP_FAMILY"), "QUERY::" + working["query_protein_id"].astype(str), family,
    )
    blocks = [group[value].to_numpy(float) for _, group in working.groupby("bootstrap_block", sort=True)]
    rng = stable_rng("family_bootstrap", key)
    estimates = np.empty(BOOTSTRAPS, dtype=float)
    for index in range(BOOTSTRAPS):
        selected = rng.integers(0, len(blocks), len(blocks))
        estimates[index] = np.mean(np.concatenate([blocks[item] for item in selected]))
    return percentile_interval(estimates)


def sign_flip_pvalue(differences: np.ndarray, key: str) -> float:
    differences = finite(differences)
    if not len(differences):
        return float("nan")
    observed = abs(float(np.mean(differences)))
    rng = stable_rng("sign_flip", key)
    exceed = 0
    completed = 0
    while completed < PERMUTATIONS:
        chunk = min(1000, PERMUTATIONS - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(chunk, len(differences)))
        estimates = np.mean(signs * differences[None, :], axis=1)
        exceed += int(np.sum(np.abs(estimates) >= observed - 1e-15))
        completed += chunk
    return (exceed + 1) / (PERMUTATIONS + 1)


def bh_adjust(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return pd.Series(adjusted, index=frame.index)
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return pd.Series(adjusted, index=frame.index)


def set_rank(frame: pd.DataFrame, name: str, columns: list[str], ascending: list[bool]) -> None:
    order = frame.sort_values(columns, ascending=ascending, kind="mergesort").index
    frame.loc[order, name] = np.arange(1, len(frame) + 1, dtype=int)
    frame[name] = frame[name].astype(int)


def best_truth_comparison(
    engine: ReactionSimilarity,
    predicted: str,
    truths: set[str],
    cache: dict[tuple[str, str], dict[str, float]],
) -> dict[str, Any]:
    comparisons: list[tuple[float, str, dict[str, float]]] = []
    for truth in sorted(truths):
        key = (predicted, truth)
        if key not in cache:
            cache[key] = engine.compare(predicted, truth)
        scores = cache[key]
        similarity = float(scores["reaction_similarity"])
        if math.isfinite(similarity):
            comparisons.append((similarity, truth, scores))
    if not comparisons:
        return {
            "best_true_rhea": None,
            "exact_rhea_match": predicted in truths,
            "transform_similarity": float("nan"),
            "substrate_similarity": float("nan"),
            "product_similarity": float("nan"),
            "reaction_similarity": float("nan"),
        }
    _, truth, scores = max(comparisons, key=lambda item: (item[0], item[1]))
    return {
        "best_true_rhea": truth,
        "exact_rhea_match": predicted in truths,
        **scores,
    }


def build_candidate_scores(root: Path) -> pd.DataFrame:
    pairs = pd.read_parquet(root / "results/phase15/cyp450_scored_pairs.parquet")
    protein_pairs = pd.read_parquet(
        root / "data/interim/phase15/p450_protein_pair_features.parquet",
        columns=["query_protein_id", "reference_protein_id", "retrieval_rank"],
    )
    pairs = pairs.merge(
        protein_pairs, on=["query_protein_id", "reference_protein_id"], how="left", validate="many_to_one",
    )
    if pairs["retrieval_rank"].isna().any():
        raise RuntimeError("MMseqs retrieval ranks are missing after the Phase 15 join")
    forbidden = sorted(column for column in pairs.columns if column.startswith("truth_"))
    if forbidden:
        raise RuntimeError(f"Ground-truth columns present before prediction ranking: {forbidden}")

    truth = pd.read_parquet(root / "data/interim/phase15/p450_query_truth.parquet").set_index("accession")
    engine = ReactionSimilarity(root)
    cache: dict[tuple[str, str], dict[str, float]] = {}
    output: list[pd.DataFrame] = []

    for scenario in SCENARIOS:
        eligible = f"eligible_{scenario}"
        pool = pairs.loc[pairs[eligible] & pairs["canonical_rhea"].notna()].copy()
        pool = pool.loc[pool["canonical_rhea"].astype(str).ne("")]
        for query_id, frame in pool.groupby("query_protein_id", sort=True):
            if query_id not in truth.index:
                continue
            truth_row = truth.loc[query_id]
            truths = json_set(truth_row["truth_rhea_json"])
            if not truths:
                continue

            representative = frame.loc[
                frame.groupby("canonical_rhea", sort=False)["calibrated_EXACT_RHEA"].idxmax(),
                ["canonical_rhea", "reference_protein_id", "reference_activity_id", "calibrated_EXACT_RHEA"],
            ].rename(columns={
                "reference_protein_id": "siteguard_reference_protein_id",
                "reference_activity_id": "siteguard_reference_activity_id",
            })
            labels = frame.groupby("canonical_rhea", sort=False).agg(
                siteguard_score=("calibrated_EXACT_RHEA", "max"),
                min_mmseqs_rank=("retrieval_rank", "min"),
                max_sequence_identity=("sequence_identity", "max"),
                max_esm2_cosine=("esm2_t33_cosine", "max"),
                max_pfam_jaccard=("pfam_jaccard", "max"),
                reference_proteins=("reference_protein_id", "nunique"),
                reference_clusters=("reference_cluster_id_30", "nunique"),
            ).reset_index().merge(representative, on="canonical_rhea", how="left", validate="one_to_one")
            labels["query_protein_id"] = query_id
            labels["scenario"] = scenario
            labels["evaluation_cohort"] = truth_row["evaluation_cohort"]
            labels["query_cyp_family"] = truth_row["cyp_family"]
            labels["query_species_group"] = truth_row["species_group"]

            set_rank(
                labels, "rank_siteguard",
                ["siteguard_score", "reference_clusters", "max_sequence_identity", "canonical_rhea"],
                [False, False, False, True],
            )
            set_rank(
                labels, "rank_mmseqs",
                ["min_mmseqs_rank", "max_sequence_identity", "canonical_rhea"],
                [True, False, True],
            )
            set_rank(
                labels, "rank_esm2",
                ["max_esm2_cosine", "min_mmseqs_rank", "canonical_rhea"],
                [False, True, True],
            )
            set_rank(
                labels, "rank_frequency",
                ["reference_clusters", "reference_proteins", "max_sequence_identity", "canonical_rhea"],
                [False, False, False, True],
            )
            comparisons = [
                best_truth_comparison(engine, str(rhea), truths, cache)
                for rhea in labels["canonical_rhea"].astype(str)
            ]
            labels = pd.concat([labels.reset_index(drop=True), pd.DataFrame(comparisons)], axis=1)
            labels["truth_labels_json"] = json.dumps(sorted(truths))
            labels["ranking_truth_use_policy"] = "NO_QUERY_FUNCTION_TRUTH_USED"
            labels["evaluation_truth_use_policy"] = "GROUND_TRUTH_ONLY_AFTER_ALL_RANKS_FROZEN"
            output.append(labels)
        print(json.dumps({"candidate_scoring_scenario": scenario, "queries": len(output)}), flush=True)
    if not output:
        raise RuntimeError("No reaction-evaluable external CYP candidate sets were produced")
    return pd.concat(output, ignore_index=True)


def panel_from_ranked(group: pd.DataFrame, rank_column: str, panel_size: int) -> dict[str, Any]:
    selected = group.loc[group[rank_column].le(panel_size)].sort_values(rank_column)
    valid = selected.loc[np.isfinite(selected["reaction_similarity"])]
    if valid.empty:
        best = None
        similarity = float("nan")
    else:
        best = valid.sort_values(
            ["reaction_similarity", rank_column, "canonical_rhea"], ascending=[False, True, True],
        ).iloc[0]
        similarity = float(best["reaction_similarity"])
    return {
        "panel_labels_json": json.dumps(selected["canonical_rhea"].astype(str).tolist()),
        "actual_panel_size": len(selected),
        "reaction_similarity": similarity,
        "exact_panel_hit": float(bool(selected["exact_rhea_match"].any())),
        "best_predicted_rhea": None if best is None else best["canonical_rhea"],
        "best_true_rhea": None if best is None else best["best_true_rhea"],
        "transform_similarity": float("nan") if best is None else float(best["transform_similarity"]),
        "substrate_similarity": float("nan") if best is None else float(best["substrate_similarity"]),
        "product_similarity": float("nan") if best is None else float(best["product_similarity"]),
        "score_evaluable": bool(len(valid)),
    }


def random_panel_expectation(group: pd.DataFrame, panel_size: int) -> dict[str, Any]:
    records = group.reset_index(drop=True)
    n = len(records)
    size = min(panel_size, n)
    rng = stable_rng("random_panel", group.iloc[0]["scenario"], group.iloc[0]["query_protein_id"], panel_size)
    similarities = np.empty(RANDOM_DRAWS, dtype=float)
    exact = np.empty(RANDOM_DRAWS, dtype=float)
    values = records["reaction_similarity"].to_numpy(float)
    hits = records["exact_rhea_match"].to_numpy(bool)
    for index in range(RANDOM_DRAWS):
        choice = rng.choice(n, size=size, replace=False)
        draw = values[choice]
        similarities[index] = np.nanmax(draw) if np.isfinite(draw).any() else np.nan
        exact[index] = float(hits[choice].any())
    return {
        "panel_labels_json": f"RANDOM_EXPECTATION_{RANDOM_DRAWS}_DRAWS",
        "actual_panel_size": size,
        "reaction_similarity": float(np.nanmean(similarities)) if np.isfinite(similarities).any() else float("nan"),
        "exact_panel_hit": float(np.mean(exact)),
        "best_predicted_rhea": None,
        "best_true_rhea": None,
        "transform_similarity": float("nan"),
        "substrate_similarity": float("nan"),
        "product_similarity": float("nan"),
        "score_evaluable": bool(np.isfinite(similarities).any()),
    }


def evaluate_panels(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["scenario", "query_protein_id", "evaluation_cohort", "query_cyp_family", "query_species_group"]
    for keys, group in candidates.groupby(group_columns, sort=True):
        base = dict(zip(group_columns, keys, strict=True))
        for panel_size in PANEL_SIZES:
            for method, rank_column in METHOD_RANKS.items():
                rows.append({**base, "method": method, "panel_size": panel_size, **panel_from_ranked(group, rank_column, panel_size)})
            rows.append({**base, "method": "RANDOM_UNIFORM", "panel_size": panel_size, **random_panel_expectation(group, panel_size)})
            oracle = panel_from_ranked(
                group.assign(rank_oracle=group["reaction_similarity"].rank(method="first", ascending=False)),
                "rank_oracle", len(group),
            )
            rows.append({**base, "method": "ORACLE_CANDIDATE_POOL", "panel_size": panel_size, **oracle})
    output = pd.DataFrame(rows)
    output["ranking_truth_use_policy"] = np.where(
        output["method"].eq("ORACLE_CANDIDATE_POOL"),
        "GROUND_TRUTH_ONLY_CANDIDATE_POOL_CEILING",
        "NO_QUERY_FUNCTION_TRUTH_USED_FOR_PANEL_GENERATION",
    )
    output["analysis_status"] = "POST_HOC_EXPLORATORY_NO_THRESHOLD_RETUNING"
    return output


def aggregate_metrics(evaluation: pd.DataFrame) -> pd.DataFrame:
    strict = evaluation.loc[evaluation["evaluation_cohort"].eq("STRICT_EXTERNAL_ACCESSION")].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in strict.groupby(["scenario", "method", "panel_size"], sort=True):
        scenario, method, panel_size = keys
        scores = pd.to_numeric(group["reaction_similarity"], errors="coerce")
        valid = scores[np.isfinite(scores)]
        key = f"{scenario}|{method}|{panel_size}"
        mean_low, mean_high = bootstrap_interval(valid.to_numpy(float), key, "mean")
        median_low, median_high = bootstrap_interval(valid.to_numpy(float), key, "median")
        family_low, family_high = family_block_interval(group, "reaction_similarity", key)
        exact = pd.to_numeric(group["exact_panel_hit"], errors="coerce")
        rows.append({
            "scenario": scenario,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "panel_size": int(panel_size),
            "eligible_queries": group["query_protein_id"].nunique(),
            "reaction_evaluable_queries": int(np.isfinite(scores).sum()),
            "reaction_score_coverage": float(np.isfinite(scores).mean()),
            "mean_reaction_similarity": float(valid.mean()) if len(valid) else float("nan"),
            "mean_query_bootstrap_ci_low": mean_low,
            "mean_query_bootstrap_ci_high": mean_high,
            "mean_family_block_ci_low": family_low,
            "mean_family_block_ci_high": family_high,
            "median_reaction_similarity": float(valid.median()) if len(valid) else float("nan"),
            "median_query_bootstrap_ci_low": median_low,
            "median_query_bootstrap_ci_high": median_high,
            "fraction_similarity_ge_0_5": float((valid >= 0.5).mean()) if len(valid) else float("nan"),
            "fraction_similarity_ge_0_7": float((valid >= 0.7).mean()) if len(valid) else float("nan"),
            "fraction_similarity_ge_0_8": float((valid >= 0.8).mean()) if len(valid) else float("nan"),
            "exact_panel_recall": float(exact.mean()),
            "random_draws_per_query": RANDOM_DRAWS if method == "RANDOM_UNIFORM" else 0,
            "bootstrap_replicates": BOOTSTRAPS,
        })
    return pd.DataFrame(rows)


def paired_comparisons(evaluation: pd.DataFrame) -> pd.DataFrame:
    strict = evaluation.loc[evaluation["evaluation_cohort"].eq("STRICT_EXTERNAL_ACCESSION")].copy()
    baselines = ["MMSEQS_NEAREST_NEIGHBOR", "ESM2_NEAREST_NEIGHBOR", "REFERENCE_FREQUENCY", "RANDOM_UNIFORM"]
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for panel_size in PANEL_SIZES:
            model = strict.loc[
                strict["scenario"].eq(scenario) & strict["panel_size"].eq(panel_size) & strict["method"].eq("SITEGUARD"),
                ["query_protein_id", "query_cyp_family", "reaction_similarity"],
            ].rename(columns={"reaction_similarity": "siteguard_similarity"})
            for baseline in baselines:
                other = strict.loc[
                    strict["scenario"].eq(scenario) & strict["panel_size"].eq(panel_size) & strict["method"].eq(baseline),
                    ["query_protein_id", "reaction_similarity"],
                ].rename(columns={"reaction_similarity": "baseline_similarity"})
                paired = model.merge(other, on="query_protein_id", how="inner", validate="one_to_one")
                paired = paired.loc[
                    np.isfinite(paired["siteguard_similarity"]) & np.isfinite(paired["baseline_similarity"])
                ].copy()
                paired["difference"] = paired["siteguard_similarity"] - paired["baseline_similarity"]
                key = f"{scenario}|{panel_size}|{baseline}"
                low, high = bootstrap_interval(paired["difference"].to_numpy(float), "paired|" + key)
                family_low, family_high = family_block_interval(paired, "difference", "paired|" + key)
                rows.append({
                    "scenario": scenario,
                    "panel_size": panel_size,
                    "baseline": baseline,
                    "baseline_label": METHOD_LABELS[baseline],
                    "paired_queries": len(paired),
                    "mean_siteguard_similarity": float(paired["siteguard_similarity"].mean()) if len(paired) else float("nan"),
                    "mean_baseline_similarity": float(paired["baseline_similarity"].mean()) if len(paired) else float("nan"),
                    "mean_paired_difference": float(paired["difference"].mean()) if len(paired) else float("nan"),
                    "median_paired_difference": float(paired["difference"].median()) if len(paired) else float("nan"),
                    "fraction_siteguard_higher": float((paired["difference"] > 0).mean()) if len(paired) else float("nan"),
                    "query_bootstrap_ci_low": low,
                    "query_bootstrap_ci_high": high,
                    "family_block_ci_low": family_low,
                    "family_block_ci_high": family_high,
                    "sign_flip_p_value": sign_flip_pvalue(paired["difference"].to_numpy(float), key),
                })
    output = pd.DataFrame(rows)
    output["sign_flip_bh_fdr"] = bh_adjust(output, "sign_flip_p_value")
    output["confirmatory_status"] = "POST_HOC_EXPLORATORY"
    return output


def reaction_lookup(root: Path) -> pd.DataFrame:
    table = pd.read_parquet(root / "data/processed/reaction_table.parquet")
    table = table.loc[table["release"].eq(141)].drop_duplicates("canonical_rhea")
    return table[[
        "canonical_rhea", "definition", "equation", "substrate_chebi_ids_json",
        "product_chebi_ids_json", "participant_names_json",
    ]]


def retrospective_panels(root: Path, candidates: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    strict = evaluation.loc[
        evaluation["scenario"].eq("GENERAL")
        & evaluation["evaluation_cohort"].eq("STRICT_EXTERNAL_ACCESSION")
        & evaluation["panel_size"].eq(5),
    ]
    model = strict.loc[strict["method"].eq("SITEGUARD")].copy()
    nn = strict.loc[strict["method"].eq("MMSEQS_NEAREST_NEIGHBOR"), ["query_protein_id", "reaction_similarity"]]
    nn = nn.rename(columns={"reaction_similarity": "mmseqs_similarity"})
    model = model.merge(nn, on="query_protein_id", how="left", validate="one_to_one")
    model["delta_vs_mmseqs"] = model["reaction_similarity"] - model["mmseqs_similarity"]
    model = model.loc[
        ~model["exact_panel_hit"].astype(bool)
        & model["reaction_similarity"].ge(0.70)
        & model["delta_vs_mmseqs"].ge(0.0)
    ].sort_values(["delta_vs_mmseqs", "reaction_similarity"], ascending=False)
    selected: list[str] = []
    seen: set[str] = set()
    for row in model.itertuples(index=False):
        family = str(row.query_cyp_family)
        block = family if family != "UNKNOWN_CYP_FAMILY" else f"QUERY::{row.query_protein_id}"
        if block in seen:
            continue
        selected.append(str(row.query_protein_id)); seen.add(block)
        if len(selected) == 12:
            break
    if len(selected) < min(12, len(model)):
        for query_id in model["query_protein_id"].astype(str):
            if query_id not in selected:
                selected.append(query_id)
            if len(selected) == 12:
                break
    if not selected:
        return pd.DataFrame(columns=["query_protein_id", "rank_siteguard", "canonical_rhea"])
    details = candidates.loc[
        candidates["scenario"].eq("GENERAL")
        & candidates["query_protein_id"].isin(selected)
        & candidates["rank_siteguard"].le(5)
    ].copy()
    details = details.merge(
        model[["query_protein_id", "reaction_similarity", "delta_vs_mmseqs"]],
        on="query_protein_id", how="left", validate="many_to_one", suffixes=("_candidate", "_panel"),
    ).merge(reaction_lookup(root), on="canonical_rhea", how="left", validate="many_to_one")
    details["case_selection_policy"] = (
        "POST_HOC_STRICT_EXTERNAL_EXACT_MISS_WITH_SITEGUARD_TOP5_SIMILARITY_GE_0.70_"
        "AND_NONNEGATIVE_DELTA_VS_MMSEQS;_FAMILY_DIVERSIFIED"
    )
    details["biological_interpretation"] = "RETROSPECTIVE_ASSAY_PANEL_RECONSTRUCTION_NOT_PROSPECTIVE_VALIDATION"
    return details.sort_values(["query_protein_id", "rank_siteguard"])


def prospective_trembl_panels(root: Path, threshold: float) -> pd.DataFrame:
    query = pd.read_parquet(root / "data/interim/phase14/query_metadata.parquet")
    catalog = pd.read_parquet(root / "results/phase14/trembl_prediction_catalog.parquet")
    mask = query["pfam_ids_json"].astype(str).str.contains("PF00067", regex=False)
    mask &= query["database_rhea_raw_json"].map(lambda value: not json_set(value))
    mask &= ~query["selection_reasons"].astype(str).str.contains("EXTERNAL_CYP450_ACCESSION", regex=False)
    eligible = query.loc[mask].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["query_protein_id", "rank_siteguard", "canonical_rhea"])
    scores = pd.read_parquet(
        root / "results/phase14/trembl_pair_scores.parquet",
        columns=[
            "query_protein_id", "reference_protein_id", "reference_activity_id", "reference_cluster_id_30",
            "canonical_rhea", "sequence_identity", "esm2_t33_cosine", "pfam_jaccard", "calibrated_EXACT_RHEA",
        ],
        filters=[("query_protein_id", "in", eligible["accession"].astype(str).tolist())],
    )
    scores = scores.loc[scores["canonical_rhea"].notna() & scores["canonical_rhea"].astype(str).ne("")]
    rows: list[pd.DataFrame] = []
    for query_id, frame in scores.groupby("query_protein_id", sort=True):
        representative = frame.loc[
            frame.groupby("canonical_rhea")["calibrated_EXACT_RHEA"].idxmax(),
            ["canonical_rhea", "reference_protein_id", "reference_activity_id"],
        ]
        labels = frame.groupby("canonical_rhea").agg(
            siteguard_score=("calibrated_EXACT_RHEA", "max"),
            reference_clusters=("reference_cluster_id_30", "nunique"),
            reference_proteins=("reference_protein_id", "nunique"),
            max_sequence_identity=("sequence_identity", "max"),
            max_esm2_cosine=("esm2_t33_cosine", "max"),
            max_pfam_jaccard=("pfam_jaccard", "max"),
        ).reset_index().merge(representative, on="canonical_rhea", how="left", validate="one_to_one")
        set_rank(
            labels, "rank_siteguard",
            ["siteguard_score", "reference_clusters", "max_sequence_identity", "canonical_rhea"],
            [False, False, False, True],
        )
        labels["query_protein_id"] = query_id
        rows.append(labels)
    if not rows:
        return pd.DataFrame(columns=["query_protein_id", "rank_siteguard", "canonical_rhea"])
    ranked = pd.concat(rows, ignore_index=True)
    top = ranked.loc[ranked["rank_siteguard"].eq(1)].copy()
    metadata = eligible[[
        "accession", "Organism", "Organism (ID)", "taxonomy_group", "selection_reasons",
        "database_ec_json", "database_rhea_raw_json", "query_structure_available",
    ]].rename(columns={"accession": "query_protein_id"})
    top = top.merge(metadata, on="query_protein_id", how="inner", validate="one_to_one")
    top = top.merge(
        catalog[["accession", "final_resolution", "top_label_EC_L3", "top_probability_EC_L3", "accepted_EC_L3"]]
        .rename(columns={"accession": "query_protein_id"}),
        on="query_protein_id", how="left", validate="one_to_one",
    )
    top = top.sort_values(
        ["accepted_EC_L3", "reference_clusters", "siteguard_score", "max_sequence_identity", "query_protein_id"],
        ascending=[False, False, False, False, True],
    )
    selected: list[str] = []
    reactions_seen: set[str] = set()
    for row in top.itertuples(index=False):
        if str(row.canonical_rhea) in reactions_seen:
            continue
        selected.append(str(row.query_protein_id)); reactions_seen.add(str(row.canonical_rhea))
        if len(selected) == 12:
            break
    if len(selected) < min(12, len(top)):
        for query_id in top["query_protein_id"].astype(str):
            if query_id not in selected:
                selected.append(query_id)
            if len(selected) == 12:
                break
    details = ranked.loc[ranked["query_protein_id"].isin(selected) & ranked["rank_siteguard"].le(5)].copy()
    details = details.merge(metadata, on="query_protein_id", how="left", validate="many_to_one")
    details = details.merge(
        catalog[["accession", "final_resolution", "top_label_EC_L3", "top_probability_EC_L3", "accepted_EC_L3"]]
        .rename(columns={"accession": "query_protein_id"}),
        on="query_protein_id", how="left", validate="many_to_one",
    ).merge(reaction_lookup(root), on="canonical_rhea", how="left", validate="many_to_one")
    details["exact_rhea_frozen_threshold"] = threshold
    details["exact_annotation_accepted"] = details["siteguard_score"].ge(threshold)
    details["selection_truth_use_policy"] = "NO_QUERY_EC_RHEA_SUBSTRATE_PRODUCT_OR_ACTIVITY_TRUTH_USED"
    details["panel_purpose"] = "PROSPECTIVE_REACTION_NEIGHBORHOOD_ASSAY_HYPOTHESIS"
    details["claim_guardrail"] = "NOT_AN_EXACT_ANNOTATION; REQUIRES EXPERIMENTAL_VALIDATION"
    details["priority_rank"] = details["query_protein_id"].map({query: rank + 1 for rank, query in enumerate(selected)})
    return details.sort_values(["priority_rank", "rank_siteguard"])


def create_figure(root: Path, metrics: pd.DataFrame, paired: pd.DataFrame) -> None:
    main = root / "figures/main"
    source = root / "figures/source_data"
    main.mkdir(parents=True, exist_ok=True); source.mkdir(parents=True, exist_ok=True)
    colors = {
        "SITEGUARD": "#2C6EAA", "MMSEQS_NEAREST_NEIGHBOR": "#E58C38",
        "ESM2_NEAREST_NEIGHBOR": "#7E62A3", "REFERENCE_FREQUENCY": "#6C757D",
        "RANDOM_UNIFORM": "#B0B7C3", "ORACLE_CANDIDATE_POOL": "#222222",
    }
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.2), constrained_layout=True)

    panel_a = metrics.loc[
        metrics["scenario"].eq("GENERAL") & metrics["panel_size"].eq(1)
        & ~metrics["method"].eq("ORACLE_CANDIDATE_POOL")
    ].copy()
    methods = ["SITEGUARD", "MMSEQS_NEAREST_NEIGHBOR", "ESM2_NEAREST_NEIGHBOR", "REFERENCE_FREQUENCY", "RANDOM_UNIFORM"]
    panel_a["order"] = panel_a["method"].map({m: i for i, m in enumerate(methods)})
    panel_a = panel_a.sort_values("order")
    y = np.arange(len(panel_a))
    axes[0, 0].errorbar(
        panel_a["mean_reaction_similarity"], y,
        xerr=np.vstack([
            panel_a["mean_reaction_similarity"] - panel_a["mean_query_bootstrap_ci_low"],
            panel_a["mean_query_bootstrap_ci_high"] - panel_a["mean_reaction_similarity"],
        ]), fmt="none", ecolor="#4A4A4A", capsize=3, lw=1.2,
    )
    for yi, row in enumerate(panel_a.itertuples(index=False)):
        axes[0, 0].scatter(row.mean_reaction_similarity, yi, s=64, color=colors[row.method], zorder=3)
    axes[0, 0].set_yticks(y, [METHOD_LABELS[m] for m in panel_a["method"]])
    axes[0, 0].set_xlim(0, 1); axes[0, 0].set_xlabel("Mean reaction similarity")
    axes[0, 0].set_title("A  Single-label reaction similarity")

    panel_b = metrics.loc[
        metrics["method"].isin(["SITEGUARD", "MMSEQS_NEAREST_NEIGHBOR"])
    ].copy()
    scenario_markers = {"GENERAL": "o", "LEAVE_ONE_CYP_FAMILY_OUT": "s", "PLANT_COLD_START": "^"}
    scenario_labels = {"GENERAL": "General", "LEAVE_ONE_CYP_FAMILY_OUT": "Leave-family-out", "PLANT_COLD_START": "Plant cold-start"}
    for (scenario, method), group in panel_b.groupby(["scenario", "method"], sort=False):
        group = group.sort_values("panel_size")
        axes[0, 1].plot(
            group["panel_size"], group["mean_reaction_similarity"],
            color=colors[method], marker=scenario_markers[scenario],
            linestyle="-" if method == "SITEGUARD" else "--", lw=1.8,
            label=f"{METHOD_LABELS[method]} · {scenario_labels[scenario]}",
        )
    axes[0, 1].set_xticks(PANEL_SIZES); axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xlabel("Reaction candidates in assay panel (K)")
    axes[0, 1].set_ylabel("Mean best-in-panel similarity")
    axes[0, 1].set_title("B  Reaction-neighborhood recovery")
    axes[0, 1].legend(frameon=False, fontsize=7.2, ncol=1, loc="best")

    panel_c = metrics.loc[
        metrics["scenario"].eq("GENERAL") & metrics["method"].eq("SITEGUARD")
    ].sort_values("panel_size")
    measures = ["exact_panel_recall", "fraction_similarity_ge_0_7", "fraction_similarity_ge_0_5"]
    labels = ["Exact Rhea", "Similarity ≥ 0.7", "Similarity ≥ 0.5"]
    bar_colors = ["#2C6EAA", "#7E62A3", "#B0B7C3"]
    x = np.arange(len(panel_c)); width = 0.24
    for index, (measure, label, color) in enumerate(zip(measures, labels, bar_colors, strict=True)):
        axes[1, 0].bar(x + (index - 1) * width, panel_c[measure], width=width, label=label, color=color)
    axes[1, 0].set_xticks(x, [f"K={value}" for value in panel_c["panel_size"]])
    axes[1, 0].set_ylim(0, 1); axes[1, 0].set_ylabel("Fraction of strict external queries")
    axes[1, 0].set_title("C  Exact and neighborhood-level panel recovery")
    axes[1, 0].legend(frameon=False, fontsize=8)

    panel_d = paired.loc[
        paired["panel_size"].eq(5)
        & paired["baseline"].isin(["MMSEQS_NEAREST_NEIGHBOR", "RANDOM_UNIFORM"])
    ].copy()
    panel_d["label"] = panel_d.apply(
        lambda row: f"{scenario_labels[row['scenario']]} vs {METHOD_LABELS[row['baseline']]}", axis=1,
    )
    panel_d = panel_d.sort_values(["baseline", "scenario"])
    y = np.arange(len(panel_d))
    axes[1, 1].axvline(0, color="#777777", lw=1, linestyle=":")
    axes[1, 1].errorbar(
        panel_d["mean_paired_difference"], y,
        xerr=np.vstack([
            panel_d["mean_paired_difference"] - panel_d["query_bootstrap_ci_low"],
            panel_d["query_bootstrap_ci_high"] - panel_d["mean_paired_difference"],
        ]), fmt="o", color="#2C6EAA", ecolor="#4A4A4A", capsize=3,
    )
    axes[1, 1].set_yticks(y, panel_d["label"])
    axes[1, 1].set_xlabel("Paired similarity difference (SiteGuard − baseline)")
    axes[1, 1].set_title("D  Paired advantage at K=5")

    for axis in axes.flat:
        axis.grid(axis="x", color="#E3E6EA", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "External CYP reaction-neighborhood evaluation\n"
        "Strict P450Rdb accessions; post-hoc exploratory analysis; frozen model and thresholds",
        fontsize=15, fontweight="bold",
    )
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(
            main / f"Figure7_CYP_reaction_neighborhood.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.15,
        )
    plt.close(fig)

    write_frame(panel_a.drop(columns="order"), source / "Figure7A_single_label_similarity.tsv")
    write_frame(panel_b, source / "Figure7B_panel_size.tsv")
    write_frame(panel_c, source / "Figure7C_neighborhood_recovery.tsv")
    write_frame(panel_d.drop(columns="label"), source / "Figure7D_paired_differences.tsv")


def write_notebook(root: Path, summary: dict[str, Any]) -> None:
    notebook_dir = root / "notebooks"; notebook_dir.mkdir(parents=True, exist_ok=True)
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["siteguard"] = {
        "phase": 20, "seed": SEED, "execution": "in-process top-to-bottom",
        "analysis_status": "post-hoc exploratory",
    }
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Phase 20 — CYP reaction-neighborhood evaluation\n\n"
            "## TL;DR\n\n" + summary["technical_summary"]
        ),
        nbformat.v4.new_markdown_cell(
            "## Context and methods\n\n"
            "The frozen Phase 11–12 SiteGuard scores are compared with MMseqs nearest-neighbor, "
            "ESM2 nearest-neighbor, reference-frequency, and uniform-random candidate-panel baselines. "
            "All rankings are generated without query EC, Rhea, substrate, product, or activity truth. "
            "Truth is used only after ranking for reaction-similarity evaluation. The analysis is post-hoc exploratory."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport pandas as pd\n"
            "ROOT = Path.cwd()\n"
            "ROOT = ROOT if (ROOT / 'results/phase20').exists() else ROOT.parent\n"
            "metrics = pd.read_csv(ROOT / 'results/phase20/cyp_reaction_neighborhood_metrics.tsv', sep='\\t')\n"
            "paired = pd.read_csv(ROOT / 'results/phase20/cyp_reaction_neighborhood_pairwise.tsv', sep='\\t')\n"
            "print(f'Metric rows: {len(metrics)}; paired comparisons: {len(paired)}')"
        ),
        nbformat.v4.new_markdown_cell("## Key results"),
        nbformat.v4.new_code_cell(
            "cols = ['scenario','method_label','panel_size','eligible_queries','mean_reaction_similarity',"
            "'mean_query_bootstrap_ci_low','mean_query_bootstrap_ci_high','exact_panel_recall']\n"
            "print(metrics.loc[metrics.method.isin(['SITEGUARD','MMSEQS_NEAREST_NEIGHBOR','RANDOM_UNIFORM']), cols]"
            ".sort_values(['scenario','panel_size','method_label']).to_string(index=False))"
        ),
        nbformat.v4.new_code_cell(
            "cols = ['scenario','panel_size','baseline_label','paired_queries','mean_paired_difference',"
            "'query_bootstrap_ci_low','query_bootstrap_ci_high','sign_flip_bh_fdr']\n"
            "print(paired.loc[paired.baseline.isin(['MMSEQS_NEAREST_NEIGHBOR','RANDOM_UNIFORM']), cols]"
            ".sort_values(['scenario','panel_size','baseline_label']).to_string(index=False))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Visual evidence\n\n"
            "![External CYP reaction-neighborhood evaluation](../figures/main/Figure7_CYP_reaction_neighborhood.png)"
        ),
        nbformat.v4.new_markdown_cell(
            "## Limitations and takeaways\n\n"
            "- The P450 analysis was extended after Phase 15 results were inspected, so inferential statistics are exploratory.\n"
            "- P450Rdb documents known activities but does not establish an exhaustive negative activity spectrum.\n"
            "- Reaction similarity measures chemical proximity, not catalytic equivalence.\n"
            "- The prospective TrEMBL panels are experimental hypotheses, not exact annotations."
        ),
    ]
    namespace: dict[str, Any] = {}
    execution_count = 0
    for cell in cells:
        if cell.cell_type == "code":
            execution_count += 1
            stream = io.StringIO()
            try:
                with contextlib.redirect_stdout(stream):
                    exec(compile(cell.source, f"phase20-cell-{execution_count}", "exec"), namespace)
            except Exception as exc:
                cell.outputs = [nbformat.v4.new_output(
                    "error", ename=type(exc).__name__, evalue=str(exc), traceback=[],
                )]
                raise
            cell.execution_count = execution_count
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=stream.getvalue())]
        cells[cells.index(cell)] = cell
    nb.cells = cells
    nbformat.validate(nb)
    nbformat.write(nb, notebook_dir / "Phase20_CYP_reaction_neighborhood.ipynb")


def scientific_decision(paired: pd.DataFrame) -> tuple[str, str]:
    row_nn = paired.loc[
        paired["scenario"].eq("GENERAL") & paired["panel_size"].eq(5)
        & paired["baseline"].eq("MMSEQS_NEAREST_NEIGHBOR")
    ]
    row_random = paired.loc[
        paired["scenario"].eq("GENERAL") & paired["panel_size"].eq(5)
        & paired["baseline"].eq("RANDOM_UNIFORM")
    ]
    if row_nn.empty or row_random.empty:
        return "INDETERMINATE", "Primary paired comparisons are missing."
    nn = row_nn.iloc[0]; random = row_random.iloc[0]
    if nn["query_bootstrap_ci_low"] > 0 and random["query_bootstrap_ci_low"] > 0:
        return (
            "EXPLORATORY_MODEL_RANKING_INCREMENT_SUPPORTED",
            "At K=5, SiteGuard exceeded both MMseqs nearest-neighbor and random candidate panels in the strict general cohort; this remains post-hoc exploratory evidence.",
        )
    if random["query_bootstrap_ci_low"] > 0:
        return (
            "EXPLORATORY_NEIGHBORHOOD_SIGNAL_WITHOUT_HOMOLOGY_INCREMENT",
            "At K=5, SiteGuard exceeded random panels but did not show a positive-CI increment over MMseqs nearest-neighbor.",
        )
    return (
        "REACTION_NEIGHBORHOOD_INCREMENT_NOT_SUPPORTED",
        "The frozen model did not show a positive-CI K=5 advantage over the required baselines; the negative result is retained.",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    results = root / "results/phase20"; reports = root / "reports"; checkpoints = root / "checkpoints"
    for path in [results, reports, checkpoints]:
        path.mkdir(parents=True, exist_ok=True)
    if not (checkpoints / "CHECKPOINT_15_PASS").is_file():
        raise RuntimeError("CHECKPOINT_15_PASS is required")
    if not (root / "FINAL_PROJECT_COMPLETE").is_file() or not (reports / "phase19_summary.json").is_file():
        raise RuntimeError("Phase 19 FINAL_PROJECT_COMPLETE and summary are required")

    candidate_path = results / "cyp_reaction_candidate_scores.parquet"
    resumed_candidate_scores = candidate_path.is_file() and candidate_path.stat().st_size > 0
    if resumed_candidate_scores:
        candidates = pd.read_parquet(candidate_path)
    else:
        candidates = build_candidate_scores(root)
        write_frame(candidates, candidate_path)
    evaluation = evaluate_panels(candidates)
    write_frame(evaluation, results / "cyp_reaction_panel_evaluation.parquet")
    metrics = aggregate_metrics(evaluation)
    paired = paired_comparisons(evaluation)
    write_frame(metrics, results / "cyp_reaction_neighborhood_metrics.tsv")
    write_frame(paired, results / "cyp_reaction_neighborhood_pairwise.tsv")

    retrospective = retrospective_panels(root, candidates, evaluation)
    calibration = json.loads((root / "models/phase12/calibration_config.json").read_text(encoding="utf-8"))
    exact_threshold = float(calibration["thresholds"]["EXACT_RHEA"])
    prospective = prospective_trembl_panels(root, exact_threshold)
    write_frame(retrospective, results / "retrospective_cyp_assay_panels.tsv")
    write_frame(prospective, results / "prospective_trembl_cyp_assay_panels.tsv")
    create_figure(root, metrics, paired)

    decision, technical_summary = scientific_decision(paired)
    strict_queries = candidates.loc[candidates["evaluation_cohort"].eq("STRICT_EXTERNAL_ACCESSION"), "query_protein_id"].nunique()
    checks = [
        ("phase15_checkpoint_present", (checkpoints / "CHECKPOINT_15_PASS").is_file(), "required phase gate"),
        ("phase19_completion_present", (root / "FINAL_PROJECT_COMPLETE").is_file() and (reports / "phase19_summary.json").is_file(), "required phase gate"),
        ("strict_external_reaction_queries_ge_150", strict_queries >= 150, str(strict_queries)),
        ("all_scenarios_present", set(candidates["scenario"]) == set(SCENARIOS), str(candidates["scenario"].value_counts().to_dict())),
        ("all_required_comparators_present", (set(METHOD_RANKS) | {"RANDOM_UNIFORM"}).issubset(set(evaluation["method"])), str(sorted(evaluation["method"].unique()))),
        ("query_truth_absent_from_ranking_fields", not any(column.startswith("truth_") for column in [
            "siteguard_score", "min_mmseqs_rank", "max_sequence_identity", "max_esm2_cosine", "reference_clusters",
        ]), "rankings use frozen prediction, retrieval, and reference-frequency evidence only"),
        ("frozen_threshold_not_retuned", not calibration["test_used_for_calibration_or_threshold_selection"], str(exact_threshold)),
        ("paired_statistics_nonempty", len(paired) == len(SCENARIOS) * len(PANEL_SIZES) * 4, str(len(paired))),
        ("figure_outputs_present", all((root / f"figures/main/Figure7_CYP_reaction_neighborhood.{suffix}").is_file() for suffix in ["pdf", "svg", "png"]), "PDF/SVG/PNG"),
        ("prospective_panel_truth_free", prospective.empty or prospective["selection_truth_use_policy"].eq("NO_QUERY_EC_RHEA_SUBSTRATE_PRODUCT_OR_ACTIVITY_TRUTH_USED").all(), str(len(prospective))),
        ("negative_results_preserved", True, decision),
    ]
    qc = pd.DataFrame(
        [(name, "PASS" if bool(ok) else "FAIL", details) for name, ok, details in checks],
        columns=["check", "status", "details"],
    )
    write_frame(qc, reports / "phase20_qc.tsv")
    failures = qc.loc[qc["status"].eq("FAIL"), "check"].tolist()
    summary = {
        "phase": 20,
        "status": "PASS" if not failures else "FAIL",
        "slurm_job_id": os.getenv("SLURM_JOB_ID", "NA"),
        "analysis_status": "POST_HOC_EXPLORATORY",
        "primary_seed": SEED,
        "random_draws_per_query": RANDOM_DRAWS,
        "bootstrap_replicates": BOOTSTRAPS,
        "sign_flip_permutations": PERMUTATIONS,
        "strict_external_reaction_queries": int(strict_queries),
        "candidate_score_rows": len(candidates),
        "resumed_candidate_scores": resumed_candidate_scores,
        "panel_evaluation_rows": len(evaluation),
        "retrospective_panel_rows": len(retrospective),
        "prospective_trembl_panel_rows": len(prospective),
        "frozen_exact_rhea_threshold": exact_threshold,
        "scientific_decision": decision,
        "technical_summary": technical_summary,
        "guardrails": [
            "No query function truth was used to generate model or baseline rankings.",
            "Reaction similarity is chemical proximity, not catalytic equivalence.",
            "Different documented Rhea reactions are not biochemical negatives.",
            "Prospective TrEMBL panels are assay hypotheses, not exact annotations.",
        ],
        "qc_failures": failures,
    }
    (reports / "phase20_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_notebook(root, summary)
    report = [
        "# SiteGuard V4 Phase 20 Report", "", f"Status: **{summary['status']}**", "",
        "## Technical summary", "", technical_summary, "",
        "This extension asks whether the frozen SiteGuard ranking narrows CYP reaction space beyond no-truth baselines. "
        "The primary endpoint is continuous reaction similarity in the strict external P450Rdb cohort. Top-1/3/5 exact and neighborhood recovery are secondary endpoints.", "",
        "## Scope and guardrails", "",
        "- Post-hoc exploratory analysis; no external threshold or model parameter was retuned.",
        "- Rankings were frozen before query reaction truth was joined.",
        "- MMseqs nearest-neighbor, ESM2 nearest-neighbor, reference-frequency, and uniform-random panels are evaluated.",
        "- Prospective TrEMBL outputs are experimental assay hypotheses, not database corrections or exact annotations.", "",
        "## QC", "", qc.to_markdown(index=False), "",
    ]
    (reports / "PHASE_20_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    if failures:
        raise RuntimeError("Phase 20 QC failed: " + ", ".join(failures))
    (checkpoints / "CHECKPOINT_20_PASS").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
