#!/usr/bin/env python3
"""Diagnostic-only analysis after the locked Phase30 evaluation.

Phase30 has already been opened and therefore must never be used for a new
confirmatory claim.  This script only characterizes errors and distribution
shift for designing a separately frozen successor model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
R30 = ROOT / "results/phase30"
REPORT = ROOT / "reports/phase30_external_blind/diagnostic_only"


def truth_for_level(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(frame["annotation_level"].eq("EC_L3"), frame["ec_l3"], frame["ec_l4"]),
        index=frame.index,
    )


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(R30 / "rcsb_strict_blind_cohort.parquet")
    candidate = pd.read_parquet(R30 / "external_augmented_candidate_matrix_blind.parquet")
    judged = pd.read_parquet(R30 / "external_predictions.parquet")
    rules = pd.read_parquet(R30 / "external_rule_predictions.parquet")
    singles = pd.read_parquet(R30 / "external_single_tool_evaluated_predictions.parquet")

    cohort_join = cohort.rename(columns={"query_id": "query_protein_id"})
    identity_columns = [
        column for column in cohort_join.columns
        if column not in {"ec_l3", "ec_l4", "ec_l3_label_eligible", "ec_l4_label_eligible"}
    ]

    # Candidate-level evidence for every accepted prediction and every error.
    key = ["annotation_level", "query_protein_id", "candidate_label"]
    evidence_columns = key + [
        column for column in candidate.columns
        if column.startswith("support__")
        or column in {
            "candidate_votes_augmented", "candidate_vote_fraction_augmented",
            "hit_old_agreement_count", "clean_old_agreement_count", "hit_clean_agree",
            "hit_top1_logit", "hit_top1_softmax", "hit_softmax_margin12",
            "clean_top1_distance", "clean_top1_gmm_confidence",
            "clean_distance_margin12", "nearest_frozen_identity_fraction",
            "supporting_channels_augmented",
        }
    ]
    evidence = candidate[evidence_columns].drop_duplicates(key)

    diagnostic_frames = []
    for system, frame in [
        ("AUGMENTED_EVIDENCEJUDGE_LOGISTIC_L2", judged),
        ("TRANSPARENT_RULE", rules),
    ]:
        accepted = frame.loc[frame["accepted"].astype(bool) & frame["label_eligible"].astype(bool)].copy()
        accepted["diagnostic_system"] = system
        diagnostic_frames.append(accepted)
    accepted_all = pd.concat(diagnostic_frames, ignore_index=True, sort=False)
    accepted_all = accepted_all.merge(evidence, on=key, how="left", validate="many_to_one")
    accepted_all = accepted_all.merge(
        cohort_join[identity_columns], on="query_protein_id", how="left", validate="many_to_one",
        suffixes=("", "_cohort"),
    )
    accepted_all.to_csv(REPORT / "accepted_predictions_with_evidence.tsv", sep="\t", index=False)
    accepted_all.loc[~accepted_all["correct"].astype(bool)].to_csv(
        REPORT / "accepted_errors_with_evidence.tsv", sep="\t", index=False
    )

    # Characterize HIT-EC at every agreement count without selecting a new cutoff.
    hit_candidates = candidate.loc[
        candidate["support__HIT_EC"].eq(1),
        [
            "annotation_level", "query_protein_id", "query_cluster_id_30", "candidate_label",
            "hit_old_agreement_count", "hit_clean_agree", "hit_top1_logit",
            "hit_top1_softmax", "hit_softmax_margin12", "hit_ec4_sigmoid_margin12",
            "clean_distance_margin12", "candidate_votes_augmented",
            "supporting_channels_augmented",
        ],
    ].copy()
    if hit_candidates.duplicated(["annotation_level", "query_protein_id"]).any():
        raise RuntimeError("HIT candidate rows are not unique")
    truth = cohort[[
        "query_id", "external_cluster_id_30", "ec_l3", "ec_l4",
        "ec_l3_label_eligible", "ec_l4_label_eligible",
    ]]
    hit_candidates = hit_candidates.merge(
        truth, left_on=["query_protein_id", "query_cluster_id_30"],
        right_on=["query_id", "external_cluster_id_30"], how="left", validate="many_to_one",
    )
    hit_candidates["truth_label"] = truth_for_level(hit_candidates)
    hit_candidates["label_eligible"] = np.where(
        hit_candidates["annotation_level"].eq("EC_L3"),
        hit_candidates["ec_l3_label_eligible"], hit_candidates["ec_l4_label_eligible"],
    ).astype(bool)
    hit_candidates["correct"] = (
        hit_candidates["label_eligible"]
        & hit_candidates["candidate_label"].astype(str).eq(hit_candidates["truth_label"].astype(str))
    )
    hit_candidates = hit_candidates.merge(
        cohort_join[identity_columns], on="query_protein_id", how="left", validate="many_to_one",
        suffixes=("", "_cohort"),
    )
    hit_candidates.to_csv(REPORT / "hit_ec_all_predictions_with_truth.tsv", sep="\t", index=False)
    hit_candidates.loc[hit_candidates["label_eligible"] & ~hit_candidates["correct"]].to_csv(
        REPORT / "hit_ec_errors.tsv", sep="\t", index=False
    )
    single_accepted_errors = singles.loc[
        singles["accepted"].astype(bool)
        & singles["label_eligible"].astype(bool)
        & ~singles["correct"].astype(bool)
    ].merge(
        cohort_join[identity_columns], on="query_protein_id", how="left", validate="many_to_one",
        suffixes=("", "_cohort"),
    )
    single_accepted_errors.to_csv(REPORT / "single_tool_accepted_errors.tsv", sep="\t", index=False)

    curve_rows: list[dict[str, object]] = []
    for level, group in hit_candidates.loc[hit_candidates["label_eligible"]].groupby("annotation_level"):
        for minimum in range(0, 8):
            selected = group.loc[group["hit_old_agreement_count"].ge(minimum)]
            curve_rows.append({
                "annotation_level": level,
                "diagnostic_rule": f"hit_old_agreement_count>={minimum}",
                "accepted_queries": len(selected),
                "accepted_clusters": selected["query_cluster_id_30"].nunique(),
                "correct": int(selected["correct"].sum()),
                "precision": float(selected["correct"].mean()) if len(selected) else np.nan,
                "coverage": len(selected) / len(group),
            })
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(REPORT / "hit_agreement_diagnostic_curve.tsv", sep="\t", index=False)

    summary = {
        "status": "PASS",
        "interpretation": "DIAGNOSTIC_ONLY_PHASE30_ALREADY_OPENED_NOT_CONFIRMATORY",
        "accepted_error_counts": accepted_all.groupby("diagnostic_system")["correct"]
        .agg(accepted="size", errors=lambda values: int((~values.astype(bool)).sum()))
        .reset_index().to_dict("records"),
        "hit_error_counts": hit_candidates.loc[hit_candidates["label_eligible"]]
        .groupby("annotation_level")["correct"]
        .agg(total="size", errors=lambda values: int((~values.astype(bool)).sum()))
        .reset_index().to_dict("records"),
        "outputs": [
            "accepted_predictions_with_evidence.tsv", "accepted_errors_with_evidence.tsv",
            "hit_ec_all_predictions_with_truth.tsv", "hit_ec_errors.tsv",
            "hit_agreement_diagnostic_curve.tsv", "single_tool_accepted_errors.tsv",
        ],
    }
    (REPORT / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    print(curve.to_string(index=False))
    print("CHECKPOINT_30D_DIAGNOSTIC_ONLY_PASS")


if __name__ == "__main__":
    main()
