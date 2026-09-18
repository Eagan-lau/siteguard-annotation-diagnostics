#!/usr/bin/env python3
"""Validate the Phase 8 atlas and create its immutable checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


LEVELS = {"EC_L3", "EC_L4", "EXACT_RHEA"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = root / "results/phase08"
    reports = root / "reports"
    figure_source = root / "figures/source_data"

    paths = {
        "curves": result / "transferability_curves.tsv",
        "boundaries": result / "family_boundaries.tsv",
        "thresholds": result / "threshold_confidence_intervals.tsv",
        "benchmark": result / "benchmark_gap.tsv",
        "stratified": result / "transferability_stratified.tsv",
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing Phase 8 artifact: {name}={path}")
    curves = pd.read_csv(paths["curves"], sep="\t")
    boundaries = pd.read_csv(paths["boundaries"], sep="\t")
    thresholds = pd.read_csv(paths["thresholds"], sep="\t")
    benchmark = pd.read_csv(paths["benchmark"], sep="\t")
    stratified = pd.read_csv(paths["stratified"], sep="\t")
    stage = json.loads((reports / "phase08_transferability_summary.json").read_text(encoding="utf-8"))

    numeric_probability_columns = {
        "curves": (curves, ["transfer_probability", "ci_lower", "ci_upper"]),
        "thresholds": (thresholds, ["precision", "ci_lower", "ci_upper", "coverage"]),
        "benchmark": (benchmark, ["positive_rate", "sequence_identity_auprc", "sequence_identity_auroc"]),
        "stratified": (stratified, ["transfer_probability"]),
    }
    probability_ranges_valid = all(
        frame[columns].apply(lambda series: series.dropna().between(0, 1).all()).all()
        for frame, columns in numeric_probability_columns.values()
    )
    validated = boundaries["boundary_status"].eq("VALIDATED_BOUNDARY")
    boundary_rules_valid = bool((boundaries.loc[validated, "validation_ci_lower"] >= boundaries.loc[validated, "target_precision"]).all())
    heldout_confirmed = boundaries.loc[
        validated & boundaries["test_ci_lower"].ge(boundaries["target_precision"])
    ]
    benchmark_designs = {
        "random_protein_split", "sequence_cluster_split", "pfam_family_holdout", "temporal_new_in_T1"
    }
    sequence_test = curves.loc[
        curves["cohort"].eq("POPULATION_ATLAS")
        & curves["query_split"].eq("test")
        & curves["evidence_metric"].eq("sequence_identity")
    ]
    expected_sequence_bins = {"00-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80-100"}
    source_files = [
        figure_source / "Figure2_transferability_curves.tsv",
        figure_source / "Figure2_family_boundaries.tsv",
        figure_source / "Figure2_benchmark_gap.tsv",
    ]
    curve_ci = curves.dropna(subset=["transfer_probability", "ci_lower", "ci_upper"])
    checks = [
        ("checkpoint_07_present", (root / "checkpoints/CHECKPOINT_07_PASS").is_file(), "strict phase gate"),
        ("registered_population_row_count", int(stage["population_rows"]) == 1_498_893, str(stage["population_rows"])),
        ("population_ipw_registered", bool(stage["population_inverse_probability_weighted"]), "population atlas"),
        ("curve_annotation_levels_complete", set(curves["annotation_level"]) == LEVELS, str(sorted(curves["annotation_level"].unique()))),
        ("sequence_test_bins_complete", set(sequence_test["evidence_bin"]) == expected_sequence_bins and len(sequence_test) == 21, str(sorted(sequence_test["evidence_bin"].unique()))),
        ("bootstrap_intervals_valid", probability_ranges_valid, "all finite observations in [0,1]"),
        ("ci_order_valid", bool((curve_ci["ci_lower"] <= curve_ci["transfer_probability"] + 1e-12).all() and (curve_ci["transfer_probability"] <= curve_ci["ci_upper"] + 1e-12).all()), f"finite_rows={len(curve_ci)}"),
        ("thresholds_not_selected_on_test", not bool(thresholds["threshold_selected_on_test"].any()) and not bool(boundaries["test_used_for_selection"].any()), "validation selection only"),
        ("family_boundary_acceptance_rule", boundary_rules_valid, f"validated={int(validated.sum())}"),
        ("benchmark_designs_complete", set(benchmark["benchmark_design"]) == benchmark_designs and len(benchmark) == 12, str(sorted(benchmark["benchmark_design"].unique()))),
        ("stratified_results_nonempty", len(stratified) > 0, str(len(stratified))),
        ("figure2_source_data_present", all(path.is_file() and path.stat().st_size > 0 for path in source_files), str(source_files)),
        ("cluster_bootstrap_registered", stage.get("bootstrap_unit") == "query_sequence_cluster_30", str(stage.get("bootstrap_unit"))),
    ]
    qc = pd.DataFrame(
        [(name, "PASS" if bool(passed) else "FAIL", details) for name, passed, details in checks],
        columns=["check", "status", "details"],
    )
    qc.to_csv(reports / "phase08_qc.tsv", sep="\t", index=False)
    failures = qc.loc[qc["status"].eq("FAIL"), "check"].tolist()
    status = "PASS" if not failures else "FAIL"

    strict = benchmark.loc[benchmark["benchmark_design"].eq("sequence_cluster_split")].set_index("annotation_level")
    random = benchmark.loc[benchmark["benchmark_design"].eq("random_protein_split")].set_index("annotation_level")
    summary = {
        "phase": 8,
        "status": status,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "population_rows": int(stage["population_rows"]),
        "curve_rows": len(curves),
        "threshold_ci_rows": len(thresholds),
        "family_boundary_rows": len(boundaries),
        "validated_family_boundaries": int(validated.sum()),
        "heldout_test_confirmed_boundaries": int(len(heldout_confirmed)),
        "strict_test_positive_rates": {level: float(strict.loc[level, "positive_rate"]) for level in sorted(LEVELS)},
        "strict_test_sequence_identity_auprc": {level: float(strict.loc[level, "sequence_identity_auprc"]) for level in sorted(LEVELS)},
        "random_sequence_identity_auprc": {level: float(random.loc[level, "sequence_identity_auprc"]) for level in sorted(LEVELS)},
        "universal_sequence_cutoff_claim_supported": False,
        "test_used_for_threshold_selection": False,
        "qc_failures": failures,
    }
    (reports / "phase08_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# SiteGuard V4 Phase 8 Report", "", f"Status: **{status}**", "",
        "## Empirical transferability landscape", "",
        f"- Registered population/atlas rows: {summary['population_rows']:,}",
        f"- Cluster-bootstrap curve estimates: {summary['curve_rows']:,}",
        f"- Threshold/CI rows: {summary['threshold_ci_rows']:,}",
        f"- Family-boundary tests: {summary['family_boundary_rows']:,}",
        f"- Validation-supported family boundaries: {summary['validated_family_boundaries']:,}",
        f"- Boundaries independently confirmed by the held-out test CI: {summary['heldout_test_confirmed_boundaries']:,}", "",
        "## Strict held-out benchmark", "",
        f"- EC-L3 concordance: {strict.loc['EC_L3', 'positive_rate']:.2%}; sequence-identity AUPRC: {strict.loc['EC_L3', 'sequence_identity_auprc']:.4f}",
        f"- EC-L4 concordance: {strict.loc['EC_L4', 'positive_rate']:.2%}; sequence-identity AUPRC: {strict.loc['EC_L4', 'sequence_identity_auprc']:.4f}",
        f"- Exact-Rhea concordance: {strict.loc['EXACT_RHEA', 'positive_rate']:.2%}; sequence-identity AUPRC: {strict.loc['EXACT_RHEA', 'sequence_identity_auprc']:.4f}", "",
        "The registered activity-specific atlas shows a sharp loss of transferable information from EC-L3 to EC-L4/Exact Rhea. Under the strict sequence-cluster split, sequence identity alone is weak and the binned relationship is not universally monotonic. Therefore Phase 8 does not support a universal identity cutoff; only family-specific, validation-supported boundaries are retained, and test replication is reported separately.", "",
        "## Random-versus-strict gap", "",
        *[
            f"- {level}: random AUPRC {random.loc[level, 'sequence_identity_auprc']:.4f} versus strict-cluster AUPRC {strict.loc[level, 'sequence_identity_auprc']:.4f}"
            for level in ["EC_L3", "EC_L4", "EXACT_RHEA"]
        ], "",
        "## QC", "", qc.to_markdown(index=False), "",
    ]
    (reports / "PHASE_08_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    if failures:
        raise RuntimeError("Phase 8 final QC failed: " + ", ".join(failures))
    (root / "checkpoints/CHECKPOINT_08_PASS").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
