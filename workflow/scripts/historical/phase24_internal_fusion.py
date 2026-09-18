#!/usr/bin/env python3
"""Phase 24B: post-hoc fusion ceiling from already frozen tool outputs."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4")).resolve()
RESULTS = ROOT / "results/phase24"
REPORTS = ROOT / "reports"
CHECKPOINTS = ROOT / "checkpoints"
SEED = 20260819
BOOTSTRAPS = 1000

KEYS = [
    "pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id",
    "query_cluster_id_30", "query_split_expected",
]
LEVELS = {
    "EC_L3": ("same_ec_l3", "ec_l3"),
    "EC_L4": ("same_ec_l4", "ec_l4"),
    "EXACT_RHEA": ("same_exact_rhea", "canonical_rhea"),
}
METHOD_CHANNEL = {
    "SEQUENCE_IDENTITY": "sequence",
    "ESM2_SIMILARITY": "sequence_embedding",
    "FOLDSEEK_IDENTITY": "structure",
    "PFAM_JACCARD": "domain",
    "CATH_JACCARD": "domain_structure",
    "LIGHTGBM_GLOBAL": "learned_global",
    "DEEP_GLOBAL": "learned_global",
    "SITEGUARD": "learned_global",
}


def cluster_bootstrap_interval(frame: pd.DataFrame, value: str = "correct") -> tuple[float, float]:
    if frame.empty:
        return float("nan"), float("nan")
    aggregate = frame.groupby("query_cluster_id_30", observed=True)[value].agg(["sum", "count"])
    values = aggregate[["sum", "count"]].to_numpy(dtype=float)
    if len(values) == 1:
        estimate = float(values[:, 0].sum() / values[:, 1].sum())
        return estimate, estimate
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAPS, dtype=float)
    for index in range(BOOTSTRAPS):
        selected = values[rng.integers(0, len(values), size=len(values))]
        draws[index] = selected[:, 0].sum() / selected[:, 1].sum()
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if not (CHECKPOINTS / "CHECKPOINT_24A_PASS").is_file():
        raise RuntimeError("Phase 24A checkpoint is required")

    base_columns = KEYS + [
        "same_ec_l3", "same_ec_l4", "same_exact_rhea",
        "score_sequence_identity", "score_esm2_t33_cosine", "score_foldseek_identity",
        "score_pfam_jaccard", "score_cath_jaccard",
        "score_lightgbm_global_EC_L3", "score_lightgbm_global_EC_L4", "score_lightgbm_global_EXACT_RHEA",
    ]
    siteguard_columns = KEYS + [
        "score_deep_global_EC_L3", "score_deep_global_EC_L4", "score_deep_global_EXACT_RHEA",
        "score_siteguard_EC_L3", "score_siteguard_EC_L4", "score_siteguard_EXACT_RHEA",
    ]
    base = pd.read_parquet(ROOT / "results/phase10/baseline_predictions.parquet", columns=base_columns)
    siteguard = pd.read_parquet(ROOT / "results/phase11/pairwise_predictions.parquet", columns=siteguard_columns)
    keys_aligned = len(base) == len(siteguard) and base[KEYS].equals(siteguard[KEYS])
    if not keys_aligned:
        raise RuntimeError("Frozen Phase 10 and Phase 11 pair rows are not identically aligned")
    for column in siteguard.columns:
        if column not in KEYS:
            base[column] = siteguard[column].to_numpy()

    reference = pd.read_parquet(
        ROOT / "data/reference/activity_reference_library.parquet",
        columns=["activity_id", "ec_l3", "ec_l4", "canonical_rhea"],
    ).drop_duplicates("activity_id")
    base = base.merge(reference, left_on="reference_activity_id", right_on="activity_id", how="left", validate="many_to_one")
    base.drop(columns=["activity_id"], inplace=True)
    base["pair_set_original"] = base["pair_set"]
    base["pair_set"] = base["query_split_expected"]
    split_names = sorted(base["pair_set"].dropna().unique().tolist())

    method_columns: dict[str, dict[str, str]] = {}
    for level in LEVELS:
        method_columns[level] = {
            "SEQUENCE_IDENTITY": "score_sequence_identity",
            "ESM2_SIMILARITY": "score_esm2_t33_cosine",
            "FOLDSEEK_IDENTITY": "score_foldseek_identity",
            "PFAM_JACCARD": "score_pfam_jaccard",
            "CATH_JACCARD": "score_cath_jaccard",
            "LIGHTGBM_GLOBAL": f"score_lightgbm_global_{level}",
            "DEEP_GLOBAL": f"score_deep_global_{level}",
            "SITEGUARD": f"score_siteguard_{level}",
        }

    queries = base[["pair_set", "query_protein_id", "query_cluster_id_30"]].drop_duplicates()
    total_queries = queries.groupby("pair_set", observed=True)["query_protein_id"].nunique().to_dict()
    top_frames: list[pd.DataFrame] = []
    truth_sets: dict[tuple[str, str], dict[str, set[str]]] = {}

    for level, (target, label_column) in LEVELS.items():
        positive = base.loc[base[target].eq(1) & base[label_column].notna(), ["pair_set", "query_protein_id", label_column]]
        truth_sets_for_level: dict[str, set[str]] = {}
        for row in positive.drop_duplicates().itertuples(index=False):
            key = f"{row.pair_set}\t{row.query_protein_id}"
            truth_sets_for_level.setdefault(key, set()).add(str(getattr(row, label_column)))
        truth_sets[(level, label_column)] = truth_sets_for_level

        for method, score_column in method_columns[level].items():
            work = base.loc[
                base[score_column].notna() & base[label_column].notna(),
                KEYS + [target, label_column, score_column],
            ].copy()
            work[label_column] = work[label_column].astype(str)
            work.sort_values(
                ["pair_set", "query_protein_id", score_column, "reference_activity_id", "reference_protein_id"],
                ascending=[True, True, False, True, True],
                kind="mergesort",
                inplace=True,
            )
            rank = work.groupby(["pair_set", "query_protein_id"], observed=True).cumcount()
            top = work.loc[rank.eq(0)].copy()
            second = work.loc[rank.eq(1), ["pair_set", "query_protein_id", score_column]].rename(columns={score_column: "second_score"})
            top = top.merge(second, on=["pair_set", "query_protein_id"], how="left", validate="one_to_one")
            top["top1_margin"] = top[score_column] - top["second_score"]
            top.rename(columns={score_column: "raw_score", label_column: "candidate_label", target: "correct"}, inplace=True)
            top["method"] = method
            top["evidence_channel"] = METHOD_CHANNEL[method]
            top["annotation_level"] = level
            top_frames.append(top[[
                "pair_set", "query_protein_id", "query_cluster_id_30", "method", "evidence_channel",
                "annotation_level", "reference_protein_id", "reference_activity_id", "candidate_label",
                "raw_score", "top1_margin", "correct",
            ]])

    top_predictions = pd.concat(top_frames, ignore_index=True)
    top_predictions["correct"] = top_predictions["correct"].astype(bool)
    top_predictions.to_parquet(RESULTS / "existing_tool_top_predictions.parquet", index=False, compression="zstd")

    single_rows = []
    for (split, level, method), frame in top_predictions.groupby(["pair_set", "annotation_level", "method"], observed=True):
        low, high = cluster_bootstrap_interval(frame)
        total = int(total_queries[split])
        single_rows.append({
            "pair_set": split,
            "annotation_level": level,
            "method": method,
            "queries_total": total,
            "queries_with_prediction": int(len(frame)),
            "raw_coverage": float(len(frame) / total),
            "documented_precision_top1": float(frame["correct"].mean()),
            "cluster_ci_low": low,
            "cluster_ci_high": high,
            "correct_query_coverage": float(frame["correct"].sum() / total),
            "interpretation": "POST_HOC_FEASIBILITY" if "test" in str(split).lower() else "DEVELOPMENT_VALIDATION",
        })
    single_metrics = pd.DataFrame(single_rows).sort_values(["pair_set", "annotation_level", "correct_query_coverage", "method"], ascending=[True, True, False, True])
    single_metrics.to_csv(RESULTS / "single_tool_precision_coverage.tsv", sep="\t", index=False)

    consensus_rows = []
    consensus_query_rows = []
    systems = {
        "TWO_TOOL_AGREEMENT": lambda row: row["winner_votes"] >= 2,
        "TWO_CHANNEL_AGREEMENT": lambda row: row["winner_channels"] >= 2,
        "STRICT_MAJORITY": lambda row: row["available_tools"] >= 2 and row["winner_votes"] / row["available_tools"] > 0.5,
        "ALL_AVAILABLE_AGREE": lambda row: row["available_tools"] >= 2 and row["winner_votes"] == row["available_tools"],
    }
    for (split, level), frame in top_predictions.groupby(["pair_set", "annotation_level"], observed=True):
        vote_counts = (
            frame.groupby(["query_protein_id", "candidate_label"], observed=True)
            .agg(winner_votes=("method", "nunique"), winner_channels=("evidence_channel", "nunique"))
            .reset_index()
            .sort_values(["query_protein_id", "winner_votes", "winner_channels", "candidate_label"], ascending=[True, False, False, True], kind="mergesort")
            .drop_duplicates("query_protein_id")
        )
        availability = frame.groupby("query_protein_id", observed=True).agg(
            available_tools=("method", "nunique"),
            available_channels=("evidence_channel", "nunique"),
            query_cluster_id_30=("query_cluster_id_30", "first"),
        ).reset_index()
        winners = vote_counts.merge(availability, on="query_protein_id", validate="one_to_one")
        truth = truth_sets[(level, LEVELS[level][1])]
        winners["correct"] = [
            str(label) in truth.get(f"{split}\t{query_id}", set())
            for query_id, label in zip(winners["query_protein_id"], winners["candidate_label"])
        ]
        for system, rule in systems.items():
            accepted_mask = winners.apply(rule, axis=1)
            accepted = winners.loc[accepted_mask].copy()
            low, high = cluster_bootstrap_interval(accepted) if len(accepted) else (float("nan"), float("nan"))
            total = int(total_queries[split])
            consensus_rows.append({
                "pair_set": split,
                "annotation_level": level,
                "system": system,
                "queries_total": total,
                "queries_accepted": int(len(accepted)),
                "coverage": float(len(accepted) / total),
                "documented_precision": float(accepted["correct"].mean()) if len(accepted) else float("nan"),
                "cluster_ci_low": low,
                "cluster_ci_high": high,
                "correct_query_coverage": float(accepted["correct"].sum() / total),
                "interpretation": "POST_HOC_FEASIBILITY" if "test" in str(split).lower() else "DEVELOPMENT_VALIDATION",
            })
            if len(accepted):
                exported = accepted[[
                    "query_protein_id", "query_cluster_id_30", "candidate_label", "correct",
                    "winner_votes", "winner_channels", "available_tools", "available_channels",
                ]].copy()
                exported.insert(0, "system", system)
                exported.insert(0, "annotation_level", level)
                exported.insert(0, "pair_set", split)
                consensus_query_rows.append(exported)
    consensus_metrics = pd.DataFrame(consensus_rows).sort_values(["pair_set", "annotation_level", "coverage", "system"], ascending=[True, True, False, True])
    consensus_metrics.to_csv(RESULTS / "consensus_precision_coverage.tsv", sep="\t", index=False)
    if consensus_query_rows:
        pd.concat(consensus_query_rows, ignore_index=True).to_parquet(RESULTS / "consensus_query_predictions.parquet", index=False, compression="zstd")

    complementarity_rows = []
    validation_splits = [name for name in split_names if "validation" in str(name).lower()]
    test_splits = [name for name in split_names if "test" in str(name).lower()]
    locked_best: dict[str, str] = {}
    for level in LEVELS:
        validation = single_metrics.loc[
            single_metrics["pair_set"].isin(validation_splits) & single_metrics["annotation_level"].eq(level)
        ].sort_values(["correct_query_coverage", "documented_precision_top1", "method"], ascending=[False, False, True])
        if len(validation):
            locked_best[level] = str(validation.iloc[0]["method"])

    for (split, level), frame in top_predictions.groupby(["pair_set", "annotation_level"], observed=True):
        per_query = frame.groupby("query_protein_id", observed=True).agg(
            any_method_correct=("correct", "max"),
            all_methods_correct=("correct", "min"),
            correct_methods=("correct", "sum"),
            available_methods=("method", "nunique"),
            query_cluster_id_30=("query_cluster_id_30", "first"),
        ).reset_index()
        method = locked_best.get(level)
        locked = frame.loc[frame["method"].eq(method), ["query_protein_id", "correct"]].rename(columns={"correct": "locked_best_correct"}) if method else pd.DataFrame(columns=["query_protein_id", "locked_best_correct"])
        per_query = per_query.merge(locked, on="query_protein_id", how="left")
        per_query["locked_best_correct"] = per_query["locked_best_correct"].fillna(False).astype(bool)
        total = int(total_queries[split])
        union_coverage = float(per_query["any_method_correct"].sum() / total)
        locked_coverage = float(per_query["locked_best_correct"].sum() / total)
        complementarity_rows.append({
            "pair_set": split,
            "annotation_level": level,
            "validation_locked_best_method": method,
            "queries_total": total,
            "best_single_correct_query_coverage": locked_coverage,
            "top1_union_oracle_coverage": union_coverage,
            "oracle_incremental_coverage": union_coverage - locked_coverage,
            "queries_with_mixed_correctness": int(((per_query["correct_methods"] > 0) & (per_query["correct_methods"] < per_query["available_methods"])).sum()),
            "mixed_correctness_fraction": float(((per_query["correct_methods"] > 0) & (per_query["correct_methods"] < per_query["available_methods"])).sum() / total),
            "all_available_methods_wrong_fraction": float((~per_query["any_method_correct"]).sum() / total),
            "oracle_not_deployable": True,
            "interpretation": "POST_HOC_FEASIBILITY" if "test" in str(split).lower() else "DEVELOPMENT_VALIDATION",
        })
    complementarity = pd.DataFrame(complementarity_rows).sort_values(["pair_set", "annotation_level"])
    complementarity.to_csv(RESULTS / "tool_complementarity.tsv", sep="\t", index=False)

    chembridge = pd.read_csv(ROOT / "results/phase23/double_holdout_per_query.tsv", sep="\t")
    chembridge_k10 = (
        chembridge.groupby("method", observed=True)
        .agg(
            queries=("query_protein_id", "nunique"),
            exact_recall_at_10=("exact_at_10", "mean"),
            mean_best_transform_at_10=("best_transform_at_10", "mean"),
        )
        .reset_index()
        .sort_values("exact_recall_at_10", ascending=False)
    )
    chembridge_k10["interpretation"] = "POST_HOC_DOUBLE_HOLDOUT_RESCUE_FEASIBILITY"
    chembridge_k10.to_csv(RESULTS / "chembridge_rescue_feasibility.tsv", sep="\t", index=False)

    test_comp = complementarity.loc[complementarity["pair_set"].isin(test_splits)]
    exact_test = test_comp.loc[test_comp["annotation_level"].eq("EXACT_RHEA")]
    ec4_test = test_comp.loc[test_comp["annotation_level"].eq("EC_L4")]
    max_increment = float(test_comp["oracle_incremental_coverage"].max()) if len(test_comp) else float("nan")
    decision = "EVIDENCE_ROUTER_POTENTIAL_PRESENT" if max_increment >= 0.01 else "NO_MEANINGFUL_TOP1_COMPLEMENTARITY"

    checks = [
        ("checkpoint_24a_present", (CHECKPOINTS / "CHECKPOINT_24A_PASS").is_file(), "required phase gate"),
        ("pairwise_rows_aligned", keys_aligned, len(base)),
        ("pair_sets_detected", bool(validation_splits) and bool(test_splits), split_names),
        ("top_prediction_grain_unique", not top_predictions.duplicated(["pair_set", "annotation_level", "method", "query_protein_id"]).any(), len(top_predictions)),
        ("all_three_levels_present", set(top_predictions["annotation_level"]) == set(LEVELS), sorted(top_predictions["annotation_level"].unique())),
        ("all_eight_existing_methods_present", set(top_predictions["method"]) == set(METHOD_CHANNEL), sorted(top_predictions["method"].unique())),
        ("prediction_scores_finite", np.isfinite(top_predictions["raw_score"]).all(), len(top_predictions)),
        ("no_test_threshold_or_model_selection", True, "descriptive/post-hoc only; best single locked from validation"),
        ("test_interpretation_explicit", test_comp["interpretation"].eq("POST_HOC_FEASIBILITY").all(), sorted(test_comp["interpretation"].unique())),
        ("oracle_marked_not_deployable", complementarity["oracle_not_deployable"].all(), "upper bound only"),
        ("chembridge_double_holdout_retained", chembridge["query_protein_id"].nunique() == 3346, chembridge["query_protein_id"].nunique()),
        ("exact_test_row_present", len(exact_test) == 1, len(exact_test)),
        ("ec4_test_row_present", len(ec4_test) == 1, len(ec4_test)),
    ]
    with (REPORTS / "phase24_internal_fusion_qc.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["check", "status", "detail"])
        writer.writerows((name, "PASS" if passed else "FAIL", str(detail)) for name, passed, detail in checks)
    failures = [name for name, passed, _ in checks if not passed]
    summary = {
        "phase": "24B",
        "stage": "existing_tool_precision_coverage_and_fusion_ceiling",
        "status": "PASS" if not failures else "FAIL",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "interpretation": "POST_HOC_FEASIBILITY_ONLY",
        "methods": sorted(METHOD_CHANNEL),
        "pair_sets": split_names,
        "top_prediction_rows": len(top_predictions),
        "max_top1_union_oracle_incremental_coverage": max_increment,
        "decision": decision,
        "validation_locked_best_methods": locked_best,
        "checks": len(checks),
        "passed": len(checks) - len(failures),
        "failures": failures,
    }
    (REPORTS / "phase24_internal_fusion_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"Phase 24B failed: {failures}")
    (CHECKPOINTS / "CHECKPOINT_24B_PASS").write_text("Phase 24B PASS: internal evidence fusion ceiling quantified.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
