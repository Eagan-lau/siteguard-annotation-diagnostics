#!/usr/bin/env python3
"""Finalize SiteGuard V4 Phase 6 after global features and US-align calibration."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


CHUNKS = 10
ROW_KEY = ["pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id"]
FORBIDDEN = {
    "query_ec_l3_ground_truth", "query_ec_l4_ground_truth", "query_rhea_ground_truth",
    "same_ec_l3", "same_ec_l4", "same_exact_rhea", "observed_concordance_depth",
    "canonical_rhea", "ec_l3", "ec_l4", "is_difficult_case", "candidate_origin",
    "augmentation_source", "sampling_probability", "sample_weight",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    reports = root / "reports"
    work = root / "data/interim/phase06"
    usalign_work = work / "usalign"
    output = root / "data/processed/global_features.parquet"
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("global_features.parquet is required before Phase 6 finalization")

    selected = pd.read_parquet(usalign_work / "selected_usalign_pairs.parquet")
    result_frames = [
        pd.read_csv(usalign_work / f"usalign_results_chunk_{chunk}.tsv", sep="\t")
        for chunk in range(CHUNKS)
    ]
    results = pd.concat(result_frames, ignore_index=True).sort_values("usalign_pair_index")
    if len(results) != len(selected) or not results["usalign_pair_index"].is_unique:
        raise RuntimeError(f"US-align result mismatch: {len(results)}/{len(selected)}")
    calibration = selected.merge(
        results, on=["usalign_pair_index", "query_protein_id", "reference_protein_id"],
        validate="one_to_one",
    )
    calibration["tm_mean"] = calibration[["tm_query", "tm_reference"]].mean(axis=1)
    calibration["tm_min"] = calibration[["tm_query", "tm_reference"]].min(axis=1)
    calibration["tm_max"] = calibration[["tm_query", "tm_reference"]].max(axis=1)
    calibration_path = root / "data/processed/selected_usalign_calibration.parquet"
    calibration.to_parquet(calibration_path, index=False, compression="zstd")

    correlations = {
        "foldseek_identity_vs_tm_mean_spearman": float(calibration["foldseek_identity"].rank().corr(calibration["tm_mean"].rank())),
        "foldseek_alignment_fraction_vs_tm_mean_spearman": float(calibration["foldseek_alignment_fraction"].rank().corr(calibration["tm_mean"].rank())),
        "foldseek_bitscore_log1p_vs_tm_mean_spearman": float(calibration["foldseek_bitscore_log1p"].rank().corr(calibration["tm_mean"].rank())),
    }
    feature_schema = pq.ParquetFile(output).schema_arrow
    feature_columns = feature_schema.names
    provenance = json.loads((reports / "global_feature_provenance.json").read_text(encoding="utf-8"))
    forbidden = sorted(
        (set(feature_columns) & FORBIDDEN) |
        {name for name in feature_columns if name.startswith("hard_H") or "ground_truth" in name.lower()}
    )
    unclassified = sorted(set(feature_columns) - set(provenance))
    invalid_provenance = sorted(
        name for name, value in provenance.items()
        if value not in {"IDENTIFIER_OR_SPLIT_METADATA", "QUERY_REFERENCE_DERIVED", "REFERENCE_DERIVED"}
    )
    leakage_path = root / "data/splits/split_leakage_report.tsv"
    split_leakage = pd.read_csv(leakage_path, sep="\t")
    split_leakage_pass = bool((split_leakage["status"] == "PASS").all())
    prepare = json.loads((usalign_work / "usalign_prepare_summary.json").read_text(encoding="utf-8"))
    global_summary = json.loads((reports / "phase06_global_features_summary.json").read_text(encoding="utf-8"))
    esm_summary = json.loads((work / "esm2_t33_v4_summary.json").read_text(encoding="utf-8"))
    direct_summary = json.loads(
        (work / "direct_alignment/direct_alignment_merge_summary.json").read_text(encoding="utf-8")
    )
    correlations_finite = all(math.isfinite(value) for value in correlations.values())
    checks = [
        ("checkpoint_05_present", (root / "checkpoints/CHECKPOINT_05_PASS").is_file(), "strict phase gate"),
        ("global_features_exist", output.is_file() and output.stat().st_size > 0, str(output.stat().st_size)),
        ("global_feature_rows_match", int(global_summary["rows"]) == pq.ParquetFile(output).metadata.num_rows, str(global_summary["rows"])),
        ("ground_truth_fields_absent", not forbidden, str(forbidden)),
        ("all_feature_columns_classified", not unclassified, str(unclassified)),
        ("provenance_classes_valid", not invalid_provenance, str(invalid_provenance)),
        ("split_leakage_report_pass", split_leakage_pass, str(leakage_path)),
        ("esm2_merge_pass", esm_summary.get("status") == "PASS", str(esm_summary.get("status"))),
        ("direct_alignment_merge_pass", direct_summary.get("status") == "PASS", str(direct_summary.get("status"))),
        ("usalign_selection_label_blind", prepare.get("ground_truth_fields_used") == [], str(prepare.get("selection_rule"))),
        ("usalign_results_complete", len(calibration) == int(prepare["selected_pairs"]), f"{len(calibration)}/{prepare['selected_pairs']}"),
        ("usalign_tm_scores_valid", calibration[["tm_query", "tm_reference"]].apply(lambda values: values.between(0, 1)).all().all(), "range=[0,1]"),
        ("usalign_correlations_finite", correlations_finite, str(correlations)),
    ]
    qc = pd.DataFrame(
        [(name, "PASS" if passed else "FAIL", details) for name, passed, details in checks],
        columns=["check", "status", "details"],
    )
    qc.to_csv(reports / "phase06_qc.tsv", sep="\t", index=False)
    failures = qc[qc["status"] == "FAIL"]
    status = "PASS" if failures.empty else "FAIL"
    summary = {
        "phase": 6, "status": status, "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "global_feature_rows": int(global_summary["rows"]),
        "unique_protein_pairs": int(global_summary["unique_protein_pairs"]),
        "model_input_features": int(global_summary["model_input_features"]),
        "esm2_proteins": int(esm_summary["proteins"]),
        "direct_mmseqs_pairs": int(direct_summary["direct_mmseqs_rows"]),
        "direct_foldseek_pairs": int(direct_summary["direct_foldseek_rows"]),
        "usalign_calibration_pairs": len(calibration), "usalign_correlations": correlations,
        "ground_truth_fields_in_matrix": forbidden, "qc_failures": failures["check"].tolist(),
    }
    (reports / "phase06_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# SiteGuard V4 Phase 6 Report", "", f"Status: **{status}**", "",
        "## Global model inputs", "",
        f"- Activity-transfer rows: {summary['global_feature_rows']:,}",
        f"- Unique query/reference protein pairs: {summary['unique_protein_pairs']:,}",
        f"- Model-input features: {summary['model_input_features']:,}",
        f"- ESM2-t33 protein embeddings: {summary['esm2_proteins']:,}",
        f"- Direct MMseqs fills: {summary['direct_mmseqs_pairs']:,}",
        f"- Direct Foldseek fills: {summary['direct_foldseek_pairs']:,}", "",
        "Retrieval ranks, candidate origin, sampling weights, query EC/Rhea, observed concordance, and hard-case outcomes are absent from the model matrix. Direct known-pair alignments remove sampling-route missingness.", "",
        "## Label-blind US-align calibration", "",
        f"- Calibration pairs: {summary['usalign_calibration_pairs']:,}",
        *[f"- {name}: {value:.4f}" for name, value in correlations.items()], "",
        "US-align values are a structural calibration artifact and are not used as a sparsely observed model input.", "",
        "## QC", "", qc.to_markdown(index=False), "",
    ]
    (reports / "PHASE_06_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    if not failures.empty:
        raise RuntimeError("Phase 6 final QC failed: " + ", ".join(failures["check"]))
    checkpoint = root / "checkpoints/CHECKPOINT_06_PASS"
    checkpoint.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
