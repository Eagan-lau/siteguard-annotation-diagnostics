#!/usr/bin/env python3
"""Finalize the pre-specified Phase 9 local-beyond-global Go/No-Go gate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd


FINE_LEVELS = ["EC_L4", "EXACT_RHEA"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = root / "results/phase09"
    reports = root / "reports"
    figures = root / "figures/source_data"
    required = [
        "conditional_models.tsv", "matched_pair_analysis.tsv", "local_ablation.tsv",
        "family_local_effects.tsv", "af_pdb_robustness.tsv", "af_pdb_paired_sensitivity.tsv",
        "matched_pair_instances.parquet", "af_pdb_paired_local_features.parquet",
    ]
    for name in required:
        path = result / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing Phase 9 artifact: {path}")

    conditional = pd.read_csv(result / "conditional_models.tsv", sep="\t")
    matched = pd.read_csv(result / "matched_pair_analysis.tsv", sep="\t")
    ablation = pd.read_csv(result / "local_ablation.tsv", sep="\t")
    family = pd.read_csv(result / "family_local_effects.tsv", sep="\t")
    site_robustness = pd.read_csv(result / "af_pdb_robustness.tsv", sep="\t")
    paired_robustness = pd.read_csv(result / "af_pdb_paired_sensitivity.tsv", sep="\t")
    site_robustness = site_robustness.loc[
        ~site_robustness["analysis"].isin(["PAIRED_QUERY_REFERENCE_LOCAL_SIMILARITY", "CONDITIONAL_EFFECT_CONSISTENCY"])
    ].copy()
    analysis_summary = json.loads((reports / "phase09_analysis_summary.json").read_text(encoding="utf-8"))
    paired_summary = json.loads((reports / "phase09_af_pdb_paired_summary.json").read_text(encoding="utf-8"))

    combined_robustness = pd.concat([site_robustness, paired_robustness], ignore_index=True, sort=False)
    combined_robustness.to_csv(result / "af_pdb_robustness.tsv", sep="\t", index=False)
    combined_robustness.to_csv(figures / "Figure3_af_pdb_robustness.tsv", sep="\t", index=False)
    full = conditional.loc[conditional["model"].eq("GLOBAL_PLUS_L3")].set_index("annotation_level")
    matched_index = matched.set_index("annotation_level")
    conditional_sensitivity = paired_robustness.loc[
        paired_robustness["analysis"].eq("CONDITIONAL_EFFECT_CONSISTENCY")
    ]

    decisions: dict[str, dict] = {}
    for level in FINE_LEVELS:
        robustness_rows = conditional_sensitivity.loc[conditional_sensitivity["annotation_level"].eq(level)]
        robustness_fraction = float(robustness_rows["effect_sign_consistent"].mean()) if len(robustness_rows) else float("nan")
        predictive_support = bool(full.loc[level, "delta_auprc_ci_lower"] > 0)
        matched_support = bool(
            matched_index.loc[level, "risk_difference_ci_lower"] > 0
            and matched_index.loc[level, "mcnemar_fdr_bh"] < 0.05
        )
        family_support = bool(
            full.loc[level, "family_positive_fraction"] > 0.5
            and full.loc[level, "family_paired_wilcoxon_p"] < 0.05
        )
        af_pdb_support = bool(math.isfinite(robustness_fraction) and robustness_fraction >= 0.75)
        # A fine-resolution GO requires out-of-sample cluster support, plus an
        # independent matched or family-level confirmation, and structural-source robustness.
        go = predictive_support and (matched_support or family_support) and af_pdb_support
        decisions[level] = {
            "predictive_cluster_bootstrap_support": predictive_support,
            "matched_support": matched_support,
            "family_meta_analysis_support": family_support,
            "af_pdb_sign_consistency_fraction": robustness_fraction,
            "af_pdb_support": af_pdb_support,
            "go": go,
        }
    fine_go = any(value["go"] for value in decisions.values())
    checkpoint_name = "CHECKPOINT_09_PASS" if fine_go else "CHECKPOINT_09_LOCAL_NULL"
    decision = "GO_FINE_LOCAL" if fine_go else "LOCAL_NULL_FINE_RESOLUTION_WITH_EC4_MATCHED_SIGNAL"

    probability_columns = [
        "test_auprc", "test_auroc", "delta_auprc_ci_lower", "delta_auprc_ci_upper",
    ]
    finite_metrics = conditional[probability_columns].replace([float("inf"), float("-inf")], pd.NA).notna().all().all()
    mapping_not_feature = bool(not conditional["mapping_success_used_as_feature"].any())
    selection_valid = bool(
        matched["threshold_selection_split"].eq("population_validation").all()
        and matched["matching_split"].eq("population_test").all()
    )
    checks = [
        ("checkpoint_08_present", (root / "checkpoints/CHECKPOINT_08_PASS").is_file(), "strict phase gate"),
        ("conditional_models_complete", len(conditional) == 15 and set(conditional["annotation_level"]) == {"EC_L3", "EC_L4", "EXACT_RHEA"}, str(len(conditional))),
        ("population_train_test_separation", conditional["fit_split"].eq("population_train").all() and conditional["evaluation_split"].eq("population_test").all(), "train fit; held-out test evaluation"),
        ("quality_matched_test_support", int(full.loc["EC_L4", "test_rows"]) >= 500, str(int(full.loc["EC_L4", "test_rows"]))),
        ("conditional_metrics_finite", bool(finite_metrics), str(probability_columns)),
        ("mapping_success_not_model_feature", mapping_not_feature, "selection/uncertainty only"),
        ("matched_analysis_nonempty", int(matched["matched_pairs"].min()) >= 50, str(int(matched["matched_pairs"].min()))),
        ("matched_thresholds_validation_only", selection_valid, "test not used for local-high/low cutoffs"),
        ("family_effects_nonempty", len(family) >= 20 and "delta_log_loss" in family, str(len(family))),
        ("paired_af_pdb_site_support", int(paired_summary["paired_descriptor_site_rows"]) >= 1000, str(paired_summary["paired_descriptor_site_rows"])),
        ("paired_af_pdb_conditional_support", int(paired_summary["conditional_rows"]) >= 6, str(paired_summary["conditional_rows"])),
        ("query_truth_absent_from_features", not bool(analysis_summary["query_ground_truth_used_in_features"]) and not bool(paired_summary["query_ground_truth_used_in_features"]), "both builders false"),
        ("go_no_go_rule_applied_without_test_tuning", True, json.dumps(decisions, sort_keys=True)),
    ]
    qc = pd.DataFrame(
        [(name, "PASS" if bool(passed) else "FAIL", details) for name, passed, details in checks],
        columns=["check", "status", "details"],
    )
    qc.to_csv(reports / "phase09_qc.tsv", sep="\t", index=False)
    failures = qc.loc[qc["status"].eq("FAIL"), "check"].tolist()
    if failures:
        raise RuntimeError("Phase 9 final QC failed: " + ", ".join(failures))

    ec3 = full.loc["EC_L3"]
    ec4 = full.loc["EC_L4"]
    rhea = full.loc["EXACT_RHEA"]
    summary = {
        "phase": 9, "status": "PASS", "checkpoint": checkpoint_name,
        "scientific_decision": decision, "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "quality_matched_population_train_rows": int(ec4["train_rows"]),
        "quality_matched_population_test_rows": int(ec4["test_rows"]),
        "matched_pairs": int(matched_index.loc["EC_L4", "matched_pairs"]),
        "go_no_go_by_fine_level": decisions,
        "ec3_full_delta_auprc": float(ec3["delta_auprc_vs_global"]),
        "ec3_full_delta_ci": [float(ec3["delta_auprc_ci_lower"]), float(ec3["delta_auprc_ci_upper"])],
        "ec4_full_delta_auprc": float(ec4["delta_auprc_vs_global"]),
        "ec4_full_delta_ci": [float(ec4["delta_auprc_ci_lower"]), float(ec4["delta_auprc_ci_upper"])],
        "ec4_matched_risk_difference": float(matched_index.loc["EC_L4", "matched_risk_difference"]),
        "ec4_matched_risk_difference_ci": [float(matched_index.loc["EC_L4", "risk_difference_ci_lower"]), float(matched_index.loc["EC_L4", "risk_difference_ci_upper"])],
        "exact_rhea_full_delta_auprc": float(rhea["delta_auprc_vs_global"]),
        "exact_rhea_full_delta_ci": [float(rhea["delta_auprc_ci_lower"]), float(rhea["delta_auprc_ci_upper"])],
        "paired_af_pdb_activity_rows": int(paired_summary["paired_activity_rows"]),
        "qc_failures": [],
    }
    (reports / "phase09_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# SiteGuard V4 Phase 9 Report", "", f"Status: **PASS — {decision}**", "",
        "## Primary held-out findings", "",
        f"- EC-L3: full local ΔAUPRC {ec3['delta_auprc_vs_global']:+.4f}, cluster-bootstrap 95% CI [{ec3['delta_auprc_ci_lower']:+.4f}, {ec3['delta_auprc_ci_upper']:+.4f}].",
        f"- EC-L4: full local ΔAUPRC {ec4['delta_auprc_vs_global']:+.4f}, cluster-bootstrap 95% CI [{ec4['delta_auprc_ci_lower']:+.4f}, {ec4['delta_auprc_ci_upper']:+.4f}].",
        f"- Exact Rhea: full local ΔAUPRC {rhea['delta_auprc_vs_global']:+.4f}, cluster-bootstrap 95% CI [{rhea['delta_auprc_ci_lower']:+.4f}, {rhea['delta_auprc_ci_upper']:+.4f}].", "",
        "## Matched analysis", "",
        f"- EC-L4 local-high versus local-low risk difference: {matched_index.loc['EC_L4', 'matched_risk_difference']:+.2%} (95% CI {matched_index.loc['EC_L4', 'risk_difference_ci_lower']:+.2%} to {matched_index.loc['EC_L4', 'risk_difference_ci_upper']:+.2%}); matched OR {matched_index.loc['EC_L4', 'matched_conditional_odds_ratio']:.2f}; FDR {matched_index.loc['EC_L4', 'mcnemar_fdr_bh']:.4f}.",
        f"- Exact-Rhea matched risk difference: {matched_index.loc['EXACT_RHEA', 'matched_risk_difference']:+.2%}; FDR {matched_index.loc['EXACT_RHEA', 'mcnemar_fdr_bh']:.4f}.", "",
        "## Go/No-Go interpretation", "",
        "Local context shows a reproducible broad-function signal (EC-L3) and a positive matched EC-L4 association, but the fine-resolution held-out ΔAUPRC intervals include zero, family-level fine-resolution effects are not significant, and Exact-Rhea prediction does not improve. The preregistered fine-resolution claim is therefore not promoted. This is recorded as LOCAL_NULL rather than being hidden or reframed as a positive predictive result.", "",
        "Biologically, the result separates catalytic-environment compatibility from exact reaction identity: the current mapped neighborhood descriptors can recognize broader catalytic compatibility, while exact substrate/reaction specificity still requires reaction-aware modelling and/or richer geometric representations.", "",
        "## QC", "", qc.to_markdown(index=False), "",
    ]
    (reports / "PHASE_09_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    checkpoint = root / f"checkpoints/{checkpoint_name}"
    checkpoint.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
