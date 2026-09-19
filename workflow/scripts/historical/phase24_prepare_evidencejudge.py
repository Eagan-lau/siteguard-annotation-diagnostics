#!/usr/bin/env python3
"""Build the leakage-controlled candidate matrix for Phase 24 EvidenceJudge."""

from __future__ import annotations

import hashlib
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
METHODS = tuple(METHOD_CHANNEL)
LEVELS = ("EC_L3", "EC_L4", "EXACT_RHEA")
QUERY_KEYS = ["pair_set", "annotation_level", "query_protein_id", "query_cluster_id_30"]


def development_partition(cluster_id: object) -> str:
    digest = hashlib.sha256(f"{SEED}|{cluster_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 60:
        return "FIT"
    if bucket < 80:
        return "CAL"
    return "SELECT"


def normalized_entropy(values: pd.Series) -> float:
    counts = values.value_counts().to_numpy(dtype=float)
    if len(counts) <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(counts)))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    source = RESULTS / "existing_tool_top_predictions.parquet"
    if not (CHECKPOINTS / "CHECKPOINT_24B_PASS").is_file():
        raise RuntimeError("CHECKPOINT_24B_PASS is required")
    if not source.is_file():
        raise FileNotFoundError(source)

    top = pd.read_parquet(source)
    required = set(QUERY_KEYS + ["method", "evidence_channel", "candidate_label", "raw_score", "top1_margin", "correct"])
    missing = sorted(required.difference(top.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    if top.duplicated(QUERY_KEYS[:-1] + ["method"]).any():
        raise RuntimeError("Top-prediction grain is not unique per query/level/method")
    if set(top["method"].unique()) != set(METHODS):
        raise RuntimeError("Frozen method inventory changed")
    if set(top["annotation_level"].unique()) != set(LEVELS):
        raise RuntimeError("Frozen annotation levels changed")

    top = top.copy()
    top["candidate_label"] = top["candidate_label"].astype(str)
    top["correct"] = top["correct"].astype(bool)
    top["development_partition"] = np.where(
        top["pair_set"].eq("validation"),
        top["query_cluster_id_30"].map(development_partition),
        "TEST",
    )

    # Query-level tool state. This includes the score of each tool's own top candidate,
    # never the query truth or a truth-derived chemical feature.
    query_index = top[QUERY_KEYS + ["development_partition"]].drop_duplicates()
    query_wide = query_index.copy()
    for method in METHODS:
        own = top.loc[top["method"].eq(method), QUERY_KEYS + ["raw_score", "top1_margin"]].rename(
            columns={"raw_score": f"query_score__{method}", "top1_margin": f"query_margin__{method}"}
        )
        query_wide = query_wide.merge(own, on=QUERY_KEYS, how="left", validate="one_to_one")

    candidate_keys = QUERY_KEYS + ["candidate_label"]
    candidates = top[candidate_keys].drop_duplicates().sort_values(candidate_keys, kind="mergesort").reset_index(drop=True)
    candidates = candidates.merge(query_wide, on=QUERY_KEYS, how="left", validate="many_to_one")

    # Candidate-specific support from each tool. Raw label IDs are retained only as
    # outputs/grouping keys and are never included in the feature list.
    for method in METHODS:
        own = top.loc[
            top["method"].eq(method),
            candidate_keys + ["raw_score", "top1_margin"],
        ].rename(
            columns={
                "raw_score": f"support_score__{method}",
                "top1_margin": f"support_margin__{method}",
            }
        )
        own[f"support__{method}"] = 1.0
        candidates = candidates.merge(own, on=candidate_keys, how="left", validate="one_to_one")
        candidates[f"support__{method}"] = candidates[f"support__{method}"].fillna(0.0)

    truth = top.groupby(candidate_keys, observed=True)["correct"].agg(["min", "max"]).reset_index()
    if (truth["min"] != truth["max"]).any():
        raise RuntimeError("Candidate truth is inconsistent across supporting tools")
    truth = truth[candidate_keys + ["max"]].rename(columns={"max": "correct"})
    candidates = candidates.merge(truth, on=candidate_keys, how="left", validate="one_to_one")
    candidates["correct"] = candidates["correct"].fillna(False).astype(bool)

    support_columns = [f"support__{method}" for method in METHODS]
    candidates["supporting_tools"] = candidates[support_columns].sum(axis=1).astype(np.int16)
    channel_support_columns: list[str] = []
    for channel in sorted(set(METHOD_CHANNEL.values())):
        methods = [method for method, mapped in METHOD_CHANNEL.items() if mapped == channel]
        column = f"support_channel__{channel}"
        candidates[column] = candidates[[f"support__{method}" for method in methods]].max(axis=1)
        channel_support_columns.append(column)
    candidates["supporting_channels"] = candidates[channel_support_columns].sum(axis=1).astype(np.int16)
    candidates["available_tools"] = candidates[[f"query_score__{method}" for method in METHODS]].notna().sum(axis=1).astype(np.int16)
    candidates["available_channels"] = len(set(METHOD_CHANNEL.values()))
    candidates["tool_vote_fraction"] = candidates["supporting_tools"] / candidates["available_tools"].clip(lower=1)
    candidates["channel_vote_fraction"] = candidates["supporting_channels"] / candidates["available_channels"]

    group_keys = QUERY_KEYS
    query_stats = top.groupby(group_keys, observed=True).agg(
        candidate_set_size=("candidate_label", "nunique"),
        agreement_entropy=("candidate_label", normalized_entropy),
    ).reset_index()
    vote_counts = (
        top.groupby(candidate_keys, observed=True)["method"].nunique().rename("candidate_votes").reset_index()
        .sort_values(group_keys + ["candidate_votes", "candidate_label"], ascending=[True, True, True, True, False, True], kind="mergesort")
    )
    vote_counts["vote_rank"] = vote_counts.groupby(group_keys, observed=True)["candidate_votes"].rank(method="dense", ascending=False)
    second = vote_counts.loc[vote_counts["vote_rank"].eq(2)].groupby(group_keys, observed=True)["candidate_votes"].max().rename("second_candidate_votes").reset_index()
    candidates = candidates.merge(query_stats, on=group_keys, how="left", validate="many_to_one")
    candidates = candidates.merge(second, on=group_keys, how="left", validate="many_to_one")
    candidates["second_candidate_votes"] = candidates["second_candidate_votes"].fillna(0)
    candidates["vote_lead"] = candidates["supporting_tools"] - candidates["second_candidate_votes"]

    candidates.sort_values(candidate_keys, kind="mergesort", inplace=True)
    candidates.reset_index(drop=True, inplace=True)
    candidates.insert(0, "candidate_row_id", np.arange(len(candidates), dtype=np.int64))
    candidates["query_weight"] = 1.0 / candidates["candidate_set_size"].clip(lower=1)

    feature_columns = []
    for method in METHODS:
        feature_columns.extend([
            f"support__{method}",
            f"support_score__{method}",
            f"support_margin__{method}",
            f"query_score__{method}",
            f"query_margin__{method}",
        ])
    feature_columns.extend(channel_support_columns)
    feature_columns.extend([
        "supporting_tools", "supporting_channels", "available_tools", "available_channels",
        "tool_vote_fraction", "channel_vote_fraction", "candidate_set_size",
        "agreement_entropy", "second_candidate_votes", "vote_lead",
    ])
    forbidden_tokens = ("correct", "candidate_label", "query_protein_id", "cluster", "rhea", "ec_l")
    leakage_features = [column for column in feature_columns if any(token in column.lower() for token in forbidden_tokens)]
    if leakage_features:
        raise RuntimeError(f"Forbidden feature names detected: {leakage_features}")

    output = RESULTS / "evidencejudge_candidate_matrix.parquet"
    candidates.to_parquet(output, index=False, compression="zstd")
    schema = {
        "seed": SEED,
        "methods": list(METHODS),
        "method_channels": METHOD_CHANNEL,
        "levels": list(LEVELS),
        "feature_columns": feature_columns,
        "partition_rule": "SHA256(seed|validation_cluster) modulo 100: FIT<60, CAL<80, SELECT otherwise; population test=TEST",
        "forbidden_inputs": ["query truth", "candidate label ID", "query identifier", "cluster identifier", "truth-derived chemistry"],
    }
    (RESULTS / "evidencejudge_feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    partition_rows = (
        candidates.groupby(["pair_set", "development_partition", "annotation_level"], observed=True)
        .agg(
            candidate_rows=("candidate_row_id", "size"),
            queries=("query_protein_id", "nunique"),
            clusters=("query_cluster_id_30", "nunique"),
            positive_candidate_rows=("correct", "sum"),
            mean_candidates_per_query=("candidate_set_size", "mean"),
        )
        .reset_index()
    )
    partition_rows.to_csv(RESULTS / "evidencejudge_development_partitions.tsv", sep="\t", index=False)

    cluster_partition = top.loc[top["pair_set"].eq("validation"), ["query_cluster_id_30", "development_partition"]].drop_duplicates()
    partition_unique = not cluster_partition.duplicated("query_cluster_id_30").any()
    checks = [
        ("input_top_prediction_grain_unique", not top.duplicated(QUERY_KEYS[:-1] + ["method"]).any(), len(top)),
        ("candidate_grain_unique", not candidates.duplicated(candidate_keys).any(), len(candidates)),
        ("candidate_truth_complete", candidates["correct"].notna().all(), int(candidates["correct"].isna().sum())),
        ("cluster_partition_unique", partition_unique, len(cluster_partition)),
        ("development_partitions_present", set(cluster_partition["development_partition"]) == {"FIT", "CAL", "SELECT"}, sorted(cluster_partition["development_partition"].unique())),
        ("test_is_not_development", set(candidates.loc[candidates["pair_set"].eq("test"), "development_partition"]) == {"TEST"}, sorted(candidates.loc[candidates["pair_set"].eq("test"), "development_partition"].unique())),
        ("all_feature_columns_present", set(feature_columns).issubset(candidates.columns), len(feature_columns)),
        ("no_forbidden_feature_names", not leakage_features, leakage_features),
        ("query_weights_sum_to_query_count", bool(np.isclose(candidates.groupby(group_keys, observed=True)["query_weight"].sum().to_numpy(), 1.0).all()), len(candidates)),
        ("no_raw_label_feature", "candidate_label" not in feature_columns, feature_columns),
    ]
    qc = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    qc.to_csv(REPORTS / "phase24_evidencejudge_matrix_qc.tsv", sep="\t", index=False)
    failures = qc.loc[~qc["passed"].astype(bool), "check"].tolist()
    summary = {
        "phase": "24D0",
        "stage": "evidencejudge_candidate_matrix",
        "status": "PASS" if not failures else "FAIL",
        "source": str(source),
        "candidate_rows": len(candidates),
        "queries": int(candidates["query_protein_id"].nunique()),
        "features": len(feature_columns),
        "checks": len(checks),
        "passed": len(checks) - len(failures),
        "failures": failures,
    }
    (REPORTS / "phase24_evidencejudge_matrix_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"EvidenceJudge matrix QC failed: {failures}")
    (CHECKPOINTS / "CHECKPOINT_24D0_MATRIX_PASS").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
