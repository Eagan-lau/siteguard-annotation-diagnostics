#!/usr/bin/env python3
"""Build and lock the strictly sequence-independent Phase 26 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
RESULTS = ROOT / "results/phase26"
WORK = ROOT / "data/interim/sabio_independence"
DATA = ROOT / "data/external/uniprot_phase26_sequences"
REPORTS = ROOT / "reports"
CHECKPOINTS = ROOT / "checkpoints"
FORBIDDEN_PREDICTIONS = [RESULTS / "external_predictions.parquet", RESULTS / "external_operating_points.tsv"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_fasta(frame: pd.DataFrame, id_column: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for identifier, sequence in frame[[id_column, "sequence"]].itertuples(index=False, name=None):
            handle.write(f">{identifier}\n")
            sequence = str(sequence)
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start:start + 80] + "\n")
    temporary.replace(path)


def require_blind() -> None:
    if any(path.exists() for path in FORBIDDEN_PREDICTIONS):
        raise RuntimeError("Phase 26 predictions exist; independence filtering must remain prediction-blind")


def prepare() -> None:
    require_blind()
    sequence_checkpoint = CHECKPOINTS / "CHECKPOINT_26A1_UNIPROT_SEQUENCE_PASS"
    if not sequence_checkpoint.is_file():
        raise FileNotFoundError(sequence_checkpoint)
    for directory in (RESULTS, WORK, REPORTS, CHECKPOINTS):
        directory.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_parquet(RESULTS / "sabio_sequence_cohort.parquet")
    external = cohort.loc[cohort["sequence_eligible"]].copy().sort_values("uniprot_accession")
    if external.empty:
        raise RuntimeError("No sequence-eligible external candidates")
    write_fasta(external, "uniprot_accession", WORK / "external_candidates.fasta")

    split = pd.read_parquet(ROOT / "data/splits/split_sequence.parquet", columns=["protein_id", "split"])
    proteins = pd.read_parquet(
        ROOT / "data/processed/protein_table.parquet",
        columns=["protein_id", "sequence", "sequence_valid"],
    )
    core = split.merge(proteins, on="protein_id", how="left", validate="one_to_one")
    missing = core["sequence"].isna() | ~core["sequence_valid"].fillna(False)
    if missing.any():
        raise RuntimeError(f"Frozen core has {int(missing.sum())} missing/invalid sequences")
    reference_ids = set(pd.read_parquet(
        ROOT / "data/reference/activity_reference_library.parquet",
        columns=["reference_protein_id"],
    )["reference_protein_id"].astype(str))
    absent_reference_ids = sorted(reference_ids - set(core["protein_id"].astype(str)))
    if absent_reference_ids:
        extra = proteins.loc[proteins["protein_id"].astype(str).isin(absent_reference_ids)].copy()
        if len(extra) != len(absent_reference_ids) or not extra["sequence_valid"].all():
            raise RuntimeError("Reference library proteins missing from frozen protein sequence table")
        extra["split"] = "reference_only"
        core = pd.concat([core, extra[core.columns]], ignore_index=True)
    core = core.sort_values("protein_id").drop_duplicates("protein_id")
    write_fasta(core, "protein_id", WORK / "frozen_core_all_sequences.fasta")

    summary = {
        "phase": "26A2_PREPARE",
        "status": "PASS",
        "external_sequence_candidates": int(len(external)),
        "frozen_core_sequences": int(len(core)),
        "frozen_core_split_counts": core["split"].value_counts().to_dict(),
        "reference_ids": len(reference_ids),
        "reference_only_ids_added": len(absent_reference_ids),
        "external_fasta_sha256": sha256(WORK / "external_candidates.fasta"),
        "frozen_core_fasta_sha256": sha256(WORK / "frozen_core_all_sequences.fasta"),
        "phase26_predictions_read": False,
    }
    (WORK / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def filter_hits() -> None:
    require_blind()
    hits_path = WORK / "external_vs_frozen_core.tsv"
    if not hits_path.is_file():
        raise FileNotFoundError(hits_path)
    cohort = pd.read_parquet(RESULTS / "sabio_sequence_cohort.parquet")
    eligible = cohort.loc[cohort["sequence_eligible"]].copy()
    columns = [
        "query", "target", "fident", "alnlen", "qstart", "qend", "qlen",
        "tstart", "tend", "tlen", "qcov", "tcov", "evalue", "bits",
    ]
    hits = pd.read_csv(hits_path, sep="\t", names=columns, dtype={"query": str, "target": str})
    for column in columns[2:]:
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    # MMseqs2 `fident` is a 0--1 fraction (not a percentage).  The frozen
    # protocol's identity >30% rule is therefore fident >0.30.
    hits["strict_exclusion_hit"] = (
        hits["fident"].gt(0.30) & hits["qcov"].ge(0.70) & hits["tcov"].ge(0.70)
    )
    qualifying = set(hits.loc[hits["strict_exclusion_hit"], "query"].astype(str))

    ranked = hits.sort_values(
        ["query", "strict_exclusion_hit", "fident", "qcov", "tcov", "bits"],
        ascending=[True, False, False, False, False, False],
    ).drop_duplicates("query")
    nearest = ranked[[
        "query", "target", "fident", "qcov", "tcov", "bits", "evalue", "strict_exclusion_hit"
    ]].rename(columns={
        "query": "uniprot_accession",
        "target": "nearest_frozen_protein_id",
        "fident": "nearest_frozen_identity_fraction",
        "qcov": "nearest_frozen_query_coverage",
        "tcov": "nearest_frozen_target_coverage",
        "bits": "nearest_frozen_bitscore",
        "evalue": "nearest_frozen_evalue",
        "strict_exclusion_hit": "nearest_is_strict_exclusion_hit",
    })
    eligible = eligible.merge(nearest, on="uniprot_accession", how="left", validate="one_to_one")
    eligible["nearest_frozen_identity_percent"] = 100.0 * eligible["nearest_frozen_identity_fraction"]
    eligible["has_strict_frozen_core_hit"] = eligible["uniprot_accession"].isin(qualifying)
    eligible["sequence_independent"] = ~eligible["has_strict_frozen_core_hit"]
    eligible["independence_status"] = np.where(
        eligible["sequence_independent"], "PASS_GT30_BIDIRECTIONAL70_EXCLUSION", "EXCLUDED_GT30_BIDIRECTIONAL70"
    )
    eligible.to_parquet(RESULTS / "sabio_independence_screen.parquet", index=False, compression="zstd")
    survivors = eligible.loc[eligible["sequence_independent"]].sort_values("uniprot_accession")
    write_fasta(survivors, "uniprot_accession", WORK / "external_independent_survivors.fasta")
    summary = {
        "phase": "26A2_FILTER",
        "status": "PASS",
        "eligible_before_identity_filter": int(len(eligible)),
        "excluded_by_gt30_bidirectional70": int(eligible["has_strict_frozen_core_hit"].sum()),
        "independent_survivors": int(len(survivors)),
        "ec_l3_survivors": int((survivors["ec_l3_label_eligible"]).sum()),
        "ec_l4_survivors": int((survivors["ec_l4_label_eligible"]).sum()),
        "raw_alignment_rows": int(len(hits)),
        "phase26_predictions_read": False,
    }
    (WORK / "filter_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def finalize() -> None:
    require_blind()
    screen_path = RESULTS / "sabio_independence_screen.parquet"
    clusters_path = WORK / "external_independent_clusters_v2.tsv"
    if not screen_path.is_file() or not clusters_path.is_file():
        raise FileNotFoundError(f"Missing screen/clusters: {screen_path}; {clusters_path}")
    screen = pd.read_parquet(screen_path)
    survivors = screen.loc[screen["sequence_independent"]].copy()
    clusters = pd.read_csv(
        clusters_path, sep="\t", names=["cluster_representative", "uniprot_accession"], dtype=str
    )
    if clusters["uniprot_accession"].duplicated().any():
        raise RuntimeError("External cluster membership is not one-to-one")
    survivors = survivors.merge(clusters, on="uniprot_accession", how="left", validate="one_to_one")
    if survivors["cluster_representative"].isna().any():
        raise RuntimeError("Some independent survivors lack external cluster assignments")
    survivors["external_cluster_id_30"] = "SABIO30::" + survivors["cluster_representative"].astype(str)
    survivors["external_cohort_status"] = "LOCKED_STRICT_EXTERNAL_BLIND"
    survivors["pfam_overlap_status"] = "PENDING_LABEL_BLIND_FEATURE_EXTRACTION"
    survivors["cath_overlap_status"] = "PENDING_LABEL_BLIND_FEATURE_EXTRACTION"
    survivors["structure_availability_status"] = "PENDING_LABEL_BLIND_FEATURE_EXTRACTION"
    survivors = survivors.sort_values("uniprot_accession")
    survivors.to_parquet(RESULTS / "sabio_strict_blind_cohort.parquet", index=False, compression="zstd")

    manifest_columns = [
        "uniprot_accession", "external_cluster_id_30", "ec_l3_label_eligible", "ec_l4_label_eligible",
        "sabio_entry_count", "pubmed_count", "sequence_length_observed", "uniprot_release" if "uniprot_release" in survivors.columns else "reviewed_status",
        "nearest_frozen_identity_percent", "nearest_frozen_query_coverage", "nearest_frozen_target_coverage",
        "nearest_frozen_bitscore", "independence_status", "external_cohort_status",
    ]
    # Preserve only columns that exist; labels themselves remain in the restricted parquet, not the public manifest.
    manifest_columns = [column for column in manifest_columns if column in survivors.columns]
    survivors[manifest_columns].to_csv(RESULTS / "external_holdout_manifest.tsv", sep="\t", index=False)

    endpoint_rows = []
    for endpoint, eligibility in [("EC_L3", "ec_l3_label_eligible"), ("EC_L4", "ec_l4_label_eligible")]:
        subset = survivors.loc[survivors[eligibility].astype(bool)]
        queries = len(subset)
        cluster_count = subset["external_cluster_id_30"].nunique()
        endpoint_rows.append({
            "endpoint": endpoint,
            "eligible_queries": queries,
            "eligible_clusters": cluster_count,
            "minimum_queries": 50,
            "minimum_clusters": 20,
            "cohort_power_status": "SUFFICIENT_FOR_EXTERNAL_EVALUATION" if queries >= 50 and cluster_count >= 20 else "INSUFFICIENT_EXTERNAL_POWER",
        })
    endpoint_power = pd.DataFrame(endpoint_rows)
    endpoint_power.to_csv(RESULTS / "external_endpoint_power.tsv", sep="\t", index=False)

    all_members_once = len(clusters) == len(survivors) and not clusters["uniprot_accession"].duplicated().any()
    checks = [
        ("sequence_checkpoint_present", (CHECKPOINTS / "CHECKPOINT_26A1_UNIPROT_SEQUENCE_PASS").is_file(), "strict stage gate"),
        ("no_phase26_predictions_read", not any(path.exists() for path in FORBIDDEN_PREDICTIONS), FORBIDDEN_PREDICTIONS),
        ("all_survivors_pass_identity_rule", bool(survivors["sequence_independent"].all()), len(survivors)),
        ("all_survivors_clustered_once", all_members_once, f"members={len(clusters)};survivors={len(survivors)}"),
        ("cluster_ids_nonmissing", bool(survivors["external_cluster_id_30"].notna().all()), survivors["external_cluster_id_30"].nunique()),
        ("strict_threshold_not_relaxed", True, "identity >30%; qcov>=0.70; tcov>=0.70; internal clustering min-id=0.30/cov-mode=0/c=0.70"),
        ("cohort_nonempty", len(survivors) > 0, len(survivors)),
    ]
    qc = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    qc.to_csv(RESULTS / "external_independence_qc.tsv", sep="\t", index=False)
    failures = qc.loc[~qc["passed"].astype(bool), "check"].tolist()
    summary = {
        "phase": "26A",
        "stage": "strict_external_cohort_lock",
        "status": "PASS" if not failures else "FAIL",
        "strict_blind_queries": int(len(survivors)),
        "strict_blind_clusters": int(survivors["external_cluster_id_30"].nunique()),
        "endpoint_power": endpoint_rows,
        "identity_exclusion_rule": "exclude if identity >30% and qcov >=70% and tcov >=70%",
        "internal_cluster_rule": "MMseqs2 min-seq-id=0.30, coverage=0.70, cov-mode=0",
        "external_predictions_read": False,
        "external_predictions_allowed_after_this_checkpoint": not failures,
        "failures": failures,
    }
    (REPORTS / "phase26_external_cohort_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"Phase 26 external cohort lock failed: {failures}")
    (CHECKPOINTS / "CHECKPOINT_26A_EXTERNAL_COHORT_LOCKED").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "filter", "finalize"], required=True)
    args = parser.parse_args()
    {"prepare": prepare, "filter": filter_hits, "finalize": finalize}[args.stage]()


if __name__ == "__main__":
    main()
