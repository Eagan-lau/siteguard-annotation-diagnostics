"""Truth-free assembly of the frozen 83-feature EvidenceJudge input table.

This module converts version-locked outputs from HIT-EC, CLEAN, and the seven
frozen supporting channels into one HIT-EC-anchored row per query and EC level.
It never reads an outcome label and deliberately rejects truth-like columns.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evidencejudge import load_spec, read_feature_table


LEVELS = ("EC_L3", "EC_L4")
FIXED_SUPPORT_METHODS = (
    "SEQUENCE_IDENTITY",
    "ESM2_SIMILARITY",
    "FOLDSEEK_IDENTITY",
    "PFAM_JACCARD",
    "LIGHTGBM_GLOBAL",
    "DEEP_GLOBAL",
    "SITEGUARD",
)
IDENTIFIER_COLUMNS = (
    "annotation_level",
    "query_protein_id",
    "query_cluster_id_30",
    "candidate_label",
)

COHORT_REQUIRED = ("query_id", "external_cluster_id_30")
METADATA_REQUIRED = ("query_protein_id",)
TOOL_REQUIRED = (
    "query_protein_id",
    "method",
    "evidence_channel",
    "annotation_level",
    "candidate_label",
    "raw_score",
    "top1_margin",
)
HIT_REQUIRED = (
    "query_id",
    "sequence_length",
    "truncated_residue_count",
    "ec3_top1",
    "ec3_top1_current",
    "ec3_top1_softmax",
    "ec3_top1_logit",
    "ec3_softmax_margin12",
    "ec4_top1",
    "ec4_top1_current",
    "ec4_top1_softmax",
    "ec4_top1_logit",
    "ec4_softmax_margin12",
    "ec4_top1_sigmoid",
    "ec4_sigmoid_margin12",
    "ec3_ec4_hierarchy_consistent",
)
CLEAN_REQUIRED = (
    "query_id",
    "sequence_length",
    "truncated_residue_count",
    "clean_top1_ec4",
    "clean_top1_distance",
    "clean_top1_gmm_confidence",
    "clean_distance_margin12",
    "clean_distance_ratio12",
    "clean_maxsep_n",
)

PROHIBITED_EXACT_COLUMNS = {
    "correct",
    "truth",
    "truth_label",
    "ground_truth",
    "outcome",
    "ec_l3",
    "ec_l4",
    "ec_l3_set_json",
    "ec_l4_set_json",
    "ec_l3_label_eligible",
    "ec_l4_label_eligible",
}
PROHIBITED_COLUMN_SUBSTRINGS = (
    "ground_truth",
    "truth_label",
    "is_correct",
    "correctness",
    "experimental_outcome",
    "observed_outcome",
)


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _reject_truth_columns(frame: pd.DataFrame, name: str) -> None:
    bad = []
    for column in frame.columns:
        lowered = str(column).strip().lower()
        if lowered in PROHIBITED_EXACT_COLUMNS or any(token in lowered for token in PROHIBITED_COLUMN_SUBSTRINGS):
            bad.append(str(column))
    if bad:
        raise ValueError(f"{name} contains prohibited truth/outcome columns: {sorted(bad)}")


def _require_unique(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    duplicated = frame.duplicated(columns, keep=False)
    if duplicated.any():
        preview = frame.loc[duplicated, columns].head(5).to_dict("records")
        raise ValueError(f"{name} contains duplicate rows at grain {columns}: {preview}")


def _query_ids(frame: pd.DataFrame, column: str) -> set[str]:
    return set(frame[column].astype(str))


def _number(value: Any) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _flag(value: Any) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return 1
        if normalized in {"false", "0", "no", "n", ""}:
            return 0
        raise ValueError(f"cannot parse boolean evidence flag: {value!r}")
    return int(bool(value))


def _validate_ec_label(value: Any, components: int, context: str) -> str:
    label = str(value).strip()
    parts = label.split(".")
    if len(parts) != components or any(not part for part in parts):
        raise ValueError(
            f"{context} must be a current-EC-normalized EC{components} label with {components} components: {label!r}"
        )
    return label


def _feature_columns(spec_path: str | Path | None) -> tuple[list[str], str]:
    spec, _, spec_sha256 = load_spec(spec_path)
    ec3 = list(spec["models"]["EC_L3"]["feature_columns"])
    ec4 = list(spec["models"]["EC_L4"]["feature_columns"])
    if ec3 != ec4 or len(ec3) != 83:
        raise ValueError("Portable EvidenceJudge EC_L3/EC_L4 schemas must be identical and contain 83 features")
    return ec3, spec_sha256


def validate_tool_versions(versions: dict[str, Any], observed_methods: set[str]) -> dict[str, Any]:
    if not isinstance(versions, dict):
        raise ValueError("tool-versions JSON must be an object")
    ec_release = str(versions.get("ec_release", "")).strip()
    tools = versions.get("tools")
    if not ec_release:
        raise ValueError("tool-versions JSON must declare a non-empty ec_release")
    if not isinstance(tools, dict):
        raise ValueError("tool-versions JSON must contain a tools object")
    required = {"HIT_EC", "CLEAN", *observed_methods}
    missing = sorted(required - set(tools))
    if missing:
        raise ValueError(f"tool-versions JSON is missing observed tools: {missing}")
    empty_versions = sorted(
        name for name in required
        if not isinstance(tools.get(name), dict) or not str(tools[name].get("version", "")).strip()
    )
    if empty_versions:
        raise ValueError(f"tool-versions entries require a non-empty version: {empty_versions}")
    return {"ec_release": ec_release, "tools": {name: tools[name] for name in sorted(required)}}


def assemble_hit_anchored_features(
    cohort: pd.DataFrame,
    metadata: pd.DataFrame,
    tool_predictions: pd.DataFrame,
    hit_ec: pd.DataFrame,
    clean: pd.DataFrame,
    *,
    tool_versions: dict[str, Any],
    spec_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble the exact frozen feature schema without truth or outcomes."""
    frames = {
        "cohort": cohort.copy(),
        "metadata": metadata.copy(),
        "tool_predictions": tool_predictions.copy(),
        "hit_ec": hit_ec.copy(),
        "clean": clean.copy(),
    }
    required = {
        "cohort": COHORT_REQUIRED,
        "metadata": METADATA_REQUIRED,
        "tool_predictions": TOOL_REQUIRED,
        "hit_ec": HIT_REQUIRED,
        "clean": CLEAN_REQUIRED,
    }
    for name, frame in frames.items():
        _require_columns(frame, required[name], name)
        _reject_truth_columns(frame, name)

    cohort = frames["cohort"]
    metadata = frames["metadata"]
    tool_predictions = frames["tool_predictions"]
    hit_ec = frames["hit_ec"]
    clean = frames["clean"]
    for frame, column in ((cohort, "query_id"), (metadata, "query_protein_id"), (tool_predictions, "query_protein_id"), (hit_ec, "query_id"), (clean, "query_id")):
        frame[column] = frame[column].astype(str)

    _require_unique(cohort, ["query_id"], "cohort")
    _require_unique(metadata, ["query_protein_id"], "metadata")
    _require_unique(hit_ec, ["query_id"], "hit_ec")
    _require_unique(clean, ["query_id"], "clean")
    expected_queries = _query_ids(cohort, "query_id")
    query_sets = {
        "metadata": _query_ids(metadata, "query_protein_id"),
        "hit_ec": _query_ids(hit_ec, "query_id"),
        "clean": _query_ids(clean, "query_id"),
    }
    mismatched = {
        name: {"missing": sorted(expected_queries - values), "unexpected": sorted(values - expected_queries)}
        for name, values in query_sets.items() if values != expected_queries
    }
    if mismatched:
        raise ValueError(f"query inventories do not match the cohort: {mismatched}")

    tool_predictions["annotation_level"] = tool_predictions["annotation_level"].astype(str)
    ignored_exact_rows = int(tool_predictions["annotation_level"].eq("EXACT_RHEA").sum())
    tool_predictions = tool_predictions.loc[tool_predictions["annotation_level"].isin(LEVELS)].copy()
    unexpected_queries = sorted(_query_ids(tool_predictions, "query_protein_id") - expected_queries)
    if unexpected_queries:
        raise ValueError(f"tool_predictions contains queries outside the cohort: {unexpected_queries[:8]}")
    tool_predictions["method"] = tool_predictions["method"].astype(str)
    observed_methods = set(tool_predictions["method"])
    unexpected_methods = sorted(observed_methods - set(FIXED_SUPPORT_METHODS))
    if unexpected_methods:
        raise ValueError(
            "tool_predictions contains methods outside the frozen EvidenceJudge schema: "
            f"{unexpected_methods}"
        )
    _require_unique(
        tool_predictions,
        ["query_protein_id", "annotation_level", "method"],
        "tool_predictions",
    )
    channel_counts = tool_predictions.groupby("method", observed=True)["evidence_channel"].nunique(dropna=False)
    unstable_channels = sorted(channel_counts[channel_counts.ne(1)].index.astype(str))
    if unstable_channels:
        raise ValueError(f"a method maps to multiple evidence channels: {unstable_channels}")
    normalized_versions = validate_tool_versions(tool_versions, observed_methods)

    features, spec_sha256 = _feature_columns(spec_path)
    metadata_by_query = metadata.set_index("query_protein_id")
    hit_by_query = hit_ec.set_index("query_id")
    clean_by_query = clean.set_index("query_id")
    old_by_key = {
        (str(query), str(level)): group.sort_values(["method", "candidate_label"], kind="mergesort")
        for (query, level), group in tool_predictions.groupby(
            ["query_protein_id", "annotation_level"], observed=True, sort=False
        )
    }

    records: list[dict[str, Any]] = []
    for cohort_row in cohort.sort_values("query_id", kind="mergesort").itertuples(index=False):
        query_id = str(cohort_row.query_id)
        cluster_id = str(cohort_row.external_cluster_id_30)
        hit = hit_by_query.loc[query_id]
        clean_row = clean_by_query.loc[query_id]
        meta = metadata_by_query.loc[query_id]
        hit_ec4 = _validate_ec_label(hit["ec4_top1"], 4, f"HIT_EC {query_id} ec4_top1")
        current_hit_ec3 = _validate_ec_label(
            hit["ec3_top1_current"], 3, f"HIT_EC {query_id} ec3_top1_current"
        )
        current_hit_ec4 = _validate_ec_label(
            hit["ec4_top1_current"], 4, f"HIT_EC {query_id} ec4_top1_current"
        )
        if current_hit_ec4.rsplit(".", 1)[0] != current_hit_ec3:
            raise ValueError(
                f"HIT_EC {query_id} current EC3 must be derived from current EC4 after canonicalization"
            )
        clean_ec4 = _validate_ec_label(clean_row["clean_top1_ec4"], 4, f"CLEAN {query_id} clean_top1_ec4")
        for level, number in (("EC_L3", 3), ("EC_L4", 4)):
            old_rows = old_by_key.get((query_id, level), tool_predictions.iloc[0:0])
            old_predictions = {
                str(row.method): {
                    "label": _validate_ec_label(row.candidate_label, number, f"{row.method} {query_id} {level}"),
                    "score": _number(row.raw_score),
                    "margin": _number(row.top1_margin),
                    "channel": str(row.evidence_channel),
                }
                for row in old_rows.itertuples(index=False)
            }
            raw_hit_label = _validate_ec_label(hit[f"ec{number}_top1"], number, f"HIT_EC {query_id} ec{number}_top1")
            clean_label = clean_ec4.rsplit(".", 1)[0] if number == 3 else clean_ec4
            votes = [item["label"] for item in old_predictions.values()] + [raw_hit_label, clean_label]
            counts = Counter(votes)
            total_votes = len(votes)
            sorted_counts = sorted(counts.values(), reverse=True)
            second_votes = sorted_counts[1] if len(sorted_counts) > 1 else 0
            proportions = np.asarray(list(counts.values()), dtype=float) / total_votes
            entropy = float(
                -(proportions * np.log(proportions)).sum()
                / max(math.log(len(proportions)), 1e-12)
            )
            record: dict[str, Any] = {
                "annotation_level": level,
                "query_protein_id": query_id,
                "query_cluster_id_30": cluster_id,
                "candidate_label": current_hit_ec3 if number == 3 else current_hit_ec4,
                "available_old_tools": len(old_predictions),
                "available_tools_augmented": total_votes,
                "candidate_set_size": len(counts),
                "agreement_entropy_augmented": entropy,
                "candidate_votes_augmented": counts[raw_hit_label],
                "candidate_vote_fraction_augmented": counts[raw_hit_label] / total_votes,
                "second_candidate_votes_augmented": second_votes,
                "candidate_vote_lead_augmented": counts[raw_hit_label] - second_votes,
                "candidate_is_plurality": int(counts[raw_hit_label] == sorted_counts[0]),
                "support__HIT_EC": 1,
                "support__CLEAN": int(raw_hit_label == clean_label),
                "support_score__HIT_EC": _number(hit[f"ec{number}_top1_softmax"]),
                "support_score__CLEAN": (
                    _number(clean_row["clean_top1_gmm_confidence"])
                    if raw_hit_label == clean_label else np.nan
                ),
                "hit_old_agreement_count": sum(
                    item["label"] == raw_hit_label for item in old_predictions.values()
                ),
                "clean_old_agreement_count": sum(
                    item["label"] == clean_label for item in old_predictions.values()
                ),
                "hit_clean_agree": int(raw_hit_label == clean_label),
                "hit_top1_softmax": _number(hit[f"ec{number}_top1_softmax"]),
                "hit_top1_logit": _number(hit[f"ec{number}_top1_logit"]),
                "hit_softmax_margin12": _number(hit[f"ec{number}_softmax_margin12"]),
                "hit_ec4_sigmoid": _number(hit["ec4_top1_sigmoid"]),
                "hit_ec4_sigmoid_margin12": _number(hit["ec4_sigmoid_margin12"]),
                "hit_ec3_ec4_consistent": _flag(hit["ec3_ec4_hierarchy_consistent"]),
                "clean_top1_distance": _number(clean_row["clean_top1_distance"]),
                "clean_top1_gmm_confidence": _number(clean_row["clean_top1_gmm_confidence"]),
                "clean_distance_margin12": _number(clean_row["clean_distance_margin12"]),
                "clean_distance_ratio12": _number(clean_row["clean_distance_ratio12"]),
                "clean_maxsep_n": _number(clean_row["clean_maxsep_n"]),
                "log1p_sequence_length": math.log1p(_number(hit["sequence_length"])),
                "log1p_hit_truncation": math.log1p(_number(hit["truncated_residue_count"])),
                "nearest_frozen_identity_fraction": _number(
                    getattr(cohort_row, "nearest_frozen_identity_fraction", np.nan)
                ),
                "pfam_seen_in_fit": _flag(meta.get("pfam_seen_in_fit", np.nan)),
                "pfam_clan_seen_in_fit": _flag(meta.get("pfam_clan_seen_in_fit", np.nan)),
                "cath_seen_in_fit": _flag(meta.get("cath_seen_in_fit", np.nan)),
                "query_structure_available": _flag(meta.get("query_structure_available", np.nan)),
            }
            supporting_channels: set[str] = set()
            for method, item in old_predictions.items():
                support = int(item["label"] == raw_hit_label)
                record[f"support__{method}"] = support
                record[f"query_score__{method}"] = item["score"]
                record[f"query_margin__{method}"] = item["margin"]
                record[f"support_score__{method}"] = item["score"] if support else np.nan
                record[f"support_margin__{method}"] = item["margin"] if support else np.nan
                if support:
                    supporting_channels.add(item["channel"])
            supporting_channels.add("hierarchical_transformer")
            if raw_hit_label == clean_label:
                supporting_channels.add("contrastive_embedding")
            record["supporting_channels_augmented"] = len(supporting_channels)
            records.append(record)

    output = pd.DataFrame(records)
    output["hit_old_agreement_fraction"] = output["hit_old_agreement_count"] / output[
        "available_old_tools"
    ].clip(lower=1)
    output["hit_old_disagreement_count"] = output["available_old_tools"] - output[
        "hit_old_agreement_count"
    ]
    output["hit_clean_disagree"] = 1 - output["hit_clean_agree"]
    output["clean_neg_distance"] = -output["clean_top1_distance"]
    output["clean_log_gmm"] = np.log10(output["clean_top1_gmm_confidence"].clip(lower=1e-12))
    output["hit_logit_abs"] = output["hit_top1_logit"].abs()
    for minimum in range(1, 8):
        output[f"hit_old_agree_ge{minimum}"] = output["hit_old_agreement_count"].ge(minimum).astype(int)
    for feature in features:
        if feature not in output:
            output[feature] = np.nan
    output[features] = output[features].apply(pd.to_numeric, errors="coerce")
    output = output[[*IDENTIFIER_COLUMNS, *features]].sort_values(
        ["annotation_level", "query_protein_id"], kind="mergesort"
    ).reset_index(drop=True)
    _require_unique(output, ["query_protein_id", "annotation_level"], "assembled output")
    if len(output) != 2 * len(cohort):
        raise ValueError(f"assembled output has {len(output)} rows; expected {2 * len(cohort)}")
    _reject_truth_columns(output, "assembled output")

    null_counts = output[features].isna().sum()
    qc = {
        "status": "PASS_TRUTH_FREE_83_FEATURE_ASSEMBLY",
        "truth_or_outcome_fields_used": False,
        "queries": len(cohort),
        "rows": len(output),
        "rows_by_level": output.groupby("annotation_level", observed=True).size().astype(int).to_dict(),
        "feature_count": len(features),
        "identifier_count": len(IDENTIFIER_COLUMNS),
        "ignored_exact_rhea_tool_rows": ignored_exact_rows,
        "observed_support_methods": sorted(observed_methods),
        "tool_versions": normalized_versions,
        "model_spec_sha256": spec_sha256,
        "missing_feature_cells": int(null_counts.sum()),
        "features_with_missing_values": int(null_counts.gt(0).sum()),
        "missing_values_preserved_for_frozen_imputation": True,
        "candidate_policy": "HIT_EC_TOP1_CURRENT_EC_NORMALIZED",
        "one_row_per_query_and_level": True,
    }
    return output, qc


def read_adapter_input(path: str | Path) -> pd.DataFrame:
    return read_feature_table(path)


def write_feature_table(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False, compression="zstd")
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False, na_rep="NA")
    else:
        frame.to_csv(path, sep="\t", index=False, na_rep="NA")
