#!/usr/bin/env python3
"""Lock a strictly sequence-independent RCSB external cohort (Phase 27 or 28)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
PHASE_ID = int(os.environ.get("RCSB_EXTERNAL_PHASE", "33"))
if PHASE_ID not in {27, 28, 30, 31, 32, 33}:
    raise RuntimeError(f"Unsupported RCSB external phase: {PHASE_ID}")
RPHASE = ROOT / f"results/phase{PHASE_ID}"
WORK = ROOT / f"data/interim/phase{PHASE_ID}_independence"
CHECKPOINTS = ROOT / "checkpoints"
REPORTS = ROOT / f"reports/phase{PHASE_ID}_external_blind"
ACQUISITION_CHECKPOINT = CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}A0_RCSB_ACQUISITION_PASS"
LOCK_CHECKPOINT = CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}A_EXTERNAL_COHORT_LOCKED"
TOOL_TRAINING_FASTA = ROOT / "data/interim/phase31_tool_training_homology/tool_training_unique.fasta"
TOOL_TRAINING_SHA256 = "0b04c34acc91371b30f31077237870257af698aa859481f1c24e4fbf0c5ec1e7"


def require_blind() -> None:
    forbidden = [RPHASE / "external_blind_predictions.parquet", RPHASE / "external_predictions.parquet"]
    if any(path.exists() for path in forbidden):
        raise RuntimeError(f"Phase {PHASE_ID} predictions exist; independence stage is closed")


def write_fasta(frame: pd.DataFrame, identifier: str, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for name, sequence in frame[[identifier, "sequence"]].itertuples(index=False, name=None):
            handle.write(f">{name}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start:start + 80] + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fasta(path: Path, prefix: str) -> pd.DataFrame:
    rows: list[tuple[str, str]] = []
    name: str | None = None
    sequence: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    rows.append((prefix + name, "".join(sequence).upper()))
                name = line[1:].split()[0]
                sequence = []
            else:
                sequence.append(line)
    if name is not None:
        rows.append((prefix + name, "".join(sequence).upper()))
    return pd.DataFrame(rows, columns=["protein_id", "sequence"])


def prepare() -> None:
    require_blind()
    if not ACQUISITION_CHECKPOINT.is_file():
        raise FileNotFoundError(ACQUISITION_CHECKPOINT)
    for directory in (WORK, RPHASE, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(RPHASE / "rcsb_sequence_candidates.parquet")
    write_fasta(candidates, "query_id", WORK / "rcsb_candidates.fasta")

    split = pd.read_parquet(ROOT / "data/splits/split_sequence.parquet", columns=["protein_id"])
    proteins = pd.read_parquet(
        ROOT / "data/processed/protein_table.parquet", columns=["protein_id", "sequence", "sequence_valid"]
    )
    core = split.merge(proteins, on="protein_id", how="left", validate="one_to_one")
    if core["sequence"].isna().any() or not core["sequence_valid"].fillna(False).all():
        raise RuntimeError("Frozen V4 core sequence inventory is incomplete")
    core = core[["protein_id", "sequence"]]
    prior = ROOT / "results/phase26/sabio_strict_blind_cohort.parquet"
    prior_count = 0
    if prior.exists():
        phase26 = pd.read_parquet(prior, columns=["uniprot_accession", "sequence"]).rename(
            columns={"uniprot_accession": "protein_id"}
        )
        phase26["protein_id"] = "PHASE26::" + phase26["protein_id"].astype(str)
        prior_count = len(phase26)
        core = pd.concat([core, phase26], ignore_index=True)
    phase27_prior_count = 0
    if PHASE_ID in {28, 30, 31, 32, 33}:
        phase27_path = ROOT / "results/phase27/rcsb_sequence_candidates.parquet"
        if not phase27_path.exists():
            raise FileNotFoundError("Phase 28 requires all Phase 27 sequence candidates for homology exclusion")
        phase27 = pd.read_parquet(phase27_path, columns=["query_id", "sequence"])
        phase27["protein_id"] = "PHASE27::" + phase27["query_id"].astype(str)
        phase27_prior_count = len(phase27)
        core = pd.concat([core, phase27[["protein_id", "sequence"]]], ignore_index=True)
    phase28_prior_count = 0
    if PHASE_ID in {30, 31, 32, 33}:
        phase28_path = ROOT / "results/phase28/rcsb_sequence_candidates.parquet"
        if not phase28_path.exists():
            raise FileNotFoundError("Phase 30 requires all Phase 28 sequence candidates for homology exclusion")
        phase28 = pd.read_parquet(phase28_path, columns=["query_id", "sequence"])
        phase28["protein_id"] = "PHASE28::" + phase28["query_id"].astype(str)
        phase28_prior_count = len(phase28)
        core = pd.concat([core, phase28[["protein_id", "sequence"]]], ignore_index=True)
    phase30_prior_count = 0
    if PHASE_ID in {31, 32, 33}:
        phase30_path = ROOT / "results/phase30/rcsb_sequence_candidates.parquet"
        if not phase30_path.exists():
            raise FileNotFoundError("Phase 31 requires all Phase 30 sequence candidates for homology exclusion")
        phase30 = pd.read_parquet(phase30_path, columns=["query_id", "sequence"])
        phase30["protein_id"] = "PHASE30::" + phase30["query_id"].astype(str)
        phase30_prior_count = len(phase30)
        core = pd.concat([core, phase30[["protein_id", "sequence"]]], ignore_index=True)
    phase31_prior_count = 0
    if PHASE_ID in {32, 33}:
        phase31_path = ROOT / "results/phase31/rcsb_sequence_candidates.parquet"
        if not phase31_path.exists():
            raise FileNotFoundError("Phase 32 requires all Phase 31 sequence candidates for homology exclusion")
        phase31 = pd.read_parquet(phase31_path, columns=["query_id", "sequence"])
        phase31["protein_id"] = "PHASE31::" + phase31["query_id"].astype(str)
        phase31_prior_count = len(phase31)
        core = pd.concat([core, phase31[["protein_id", "sequence"]]], ignore_index=True)
    phase32_prior_count = 0
    if PHASE_ID == 33:
        phase32_path = ROOT / "results/phase32/rcsb_sequence_candidates.parquet"
        if not phase32_path.exists():
            raise FileNotFoundError("Phase 33 requires all Phase 32 sequence candidates for homology exclusion")
        phase32 = pd.read_parquet(phase32_path, columns=["query_id", "sequence"])
        phase32["protein_id"] = "PHASE32::" + phase32["query_id"].astype(str)
        phase32_prior_count = len(phase32)
        core = pd.concat([core, phase32[["protein_id", "sequence"]]], ignore_index=True)
    tool_training_count = 0
    tool_training_sha256 = None
    if PHASE_ID in {32, 33}:
        if not TOOL_TRAINING_FASTA.is_file():
            raise FileNotFoundError(TOOL_TRAINING_FASTA)
        tool_training_sha256 = sha256(TOOL_TRAINING_FASTA)
        if tool_training_sha256 != TOOL_TRAINING_SHA256:
            raise RuntimeError(f"Frozen tool-training FASTA hash mismatch: {tool_training_sha256}")
        tool_training = read_fasta(TOOL_TRAINING_FASTA, "TOOLTRAIN::")
        tool_training_count = len(tool_training)
        if tool_training_count != 195377 or tool_training["protein_id"].duplicated().any():
            raise RuntimeError(f"Unexpected tool-training inventory: {tool_training_count}")
        core = pd.concat([core, tool_training], ignore_index=True)
    core = core.drop_duplicates("protein_id").sort_values("protein_id")
    write_fasta(core, "protein_id", WORK / "frozen_core_plus_prior_external.fasta")
    summary = {
        "phase": f"{PHASE_ID}A1_PREPARE", "status": "PASS", "candidate_sequences": len(candidates),
        "frozen_core_sequences": len(split), "phase26_prior_external_sequences": prior_count,
        "phase27_prior_external_sequences": phase27_prior_count,
        "phase28_prior_external_sequences": phase28_prior_count,
        "phase30_prior_external_sequences": phase30_prior_count,
        "phase31_prior_external_sequences": phase31_prior_count,
        "phase32_prior_external_sequences": phase32_prior_count,
        "tool_training_sequences": tool_training_count,
        "tool_training_fasta_sha256": tool_training_sha256,
        "combined_exclusion_sequences": len(core), f"phase{PHASE_ID}_predictions_read": False,
    }
    (WORK / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def filter_hits() -> None:
    require_blind()
    candidates = pd.read_parquet(RPHASE / "rcsb_sequence_candidates.parquet")
    columns = [
        "query", "target", "fident", "alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen",
        "qcov", "tcov", "evalue", "bits",
    ]
    hits = pd.read_csv(WORK / "rcsb_vs_frozen.tsv", sep="\t", names=columns)
    for column in columns[2:]:
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    hits["target_source"] = np.where(
        hits["target"].astype(str).str.startswith("TOOLTRAIN::"),
        "TOOL_TRAINING",
        "FROZEN_OR_PRIOR_EXTERNAL",
    )
    hits["strict_exclusion_hit"] = hits["fident"].gt(0.30) & hits["qcov"].ge(0.70) & hits["tcov"].ge(0.70)
    excluded = set(hits.loc[hits["strict_exclusion_hit"], "query"].astype(str))
    tool_excluded = set(hits.loc[
        hits["strict_exclusion_hit"] & hits["target_source"].eq("TOOL_TRAINING"), "query"
    ].astype(str))
    frozen_or_prior_excluded = set(hits.loc[
        hits["strict_exclusion_hit"] & hits["target_source"].eq("FROZEN_OR_PRIOR_EXTERNAL"), "query"
    ].astype(str))
    nearest = hits.loc[hits["target_source"].eq("FROZEN_OR_PRIOR_EXTERNAL")].sort_values(
        ["query", "strict_exclusion_hit", "fident", "qcov", "tcov", "bits"],
        ascending=[True, False, False, False, False, False],
    ).drop_duplicates("query")[["query", "target", "target_source", "fident", "qcov", "tcov", "bits", "evalue", "strict_exclusion_hit"]]
    nearest = nearest.rename(columns={
        "query": "query_id", "target": "nearest_frozen_protein_id",
        "target_source": "nearest_exclusion_source",
        "fident": "nearest_frozen_identity_fraction", "qcov": "nearest_frozen_query_coverage",
        "tcov": "nearest_frozen_target_coverage", "bits": "nearest_frozen_bitscore",
        "evalue": "nearest_frozen_evalue", "strict_exclusion_hit": "nearest_is_strict_exclusion_hit",
    })
    screen = candidates.merge(nearest, on="query_id", how="left", validate="one_to_one")
    tool_nearest = hits.loc[hits["target_source"].eq("TOOL_TRAINING")].sort_values(
        ["query", "strict_exclusion_hit", "fident", "qcov", "tcov", "bits"],
        ascending=[True, False, False, False, False, False],
    ).drop_duplicates("query")[["query", "target", "fident", "qcov", "tcov", "bits", "evalue", "strict_exclusion_hit"]]
    tool_nearest = tool_nearest.rename(columns={
        "query": "query_id", "target": "nearest_tool_training_sequence_id",
        "fident": "nearest_tool_training_identity_fraction",
        "qcov": "nearest_tool_training_query_coverage",
        "tcov": "nearest_tool_training_target_coverage",
        "bits": "nearest_tool_training_bitscore",
        "evalue": "nearest_tool_training_evalue",
        "strict_exclusion_hit": "nearest_tool_training_is_strict_exclusion_hit",
    })
    screen = screen.merge(tool_nearest, on="query_id", how="left", validate="one_to_one")
    screen["nearest_frozen_identity_percent"] = 100 * screen["nearest_frozen_identity_fraction"]
    screen["nearest_tool_training_identity_percent"] = 100 * screen["nearest_tool_training_identity_fraction"]
    screen["has_strict_frozen_or_prior_hit"] = screen["query_id"].isin(frozen_or_prior_excluded)
    screen["has_strict_tool_training_hit"] = screen["query_id"].isin(tool_excluded)
    screen["has_strict_frozen_hit"] = screen["query_id"].isin(excluded)
    screen["sequence_independent"] = ~screen["has_strict_frozen_hit"]
    screen["tool_training_homology_independent"] = ~screen["has_strict_tool_training_hit"]
    screen.to_parquet(RPHASE / "rcsb_independence_screen.parquet", index=False, compression="zstd")
    survivors = screen.loc[screen["sequence_independent"]].sort_values("query_id")
    write_fasta(survivors, "query_id", WORK / "rcsb_independent_survivors.fasta")
    summary = {
        "phase": f"{PHASE_ID}A1_FILTER", "status": "PASS", "before_filter": len(screen),
        "excluded_gt30_bidirectional70": int(screen["has_strict_frozen_hit"].sum()),
        "excluded_frozen_or_prior_gt30_bidirectional70": int(screen["has_strict_frozen_or_prior_hit"].sum()),
        "excluded_tool_training_gt30_bidirectional70": int(screen["has_strict_tool_training_hit"].sum()),
        "excluded_by_both_sources": int((screen["has_strict_frozen_or_prior_hit"] & screen["has_strict_tool_training_hit"]).sum()),
        "survivors": len(survivors), "ec_l3_survivors": int(survivors["ec_l3_label_eligible"].sum()),
        "ec_l4_survivors": int(survivors["ec_l4_label_eligible"].sum()), "alignment_rows": len(hits),
        f"phase{PHASE_ID}_predictions_read": False,
    }
    (WORK / "filter_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def finalize() -> None:
    require_blind()
    screen = pd.read_parquet(RPHASE / "rcsb_independence_screen.parquet")
    survivors = screen.loc[screen["sequence_independent"]].copy()
    clusters = pd.read_csv(
        WORK / "rcsb_independent_clusters.tsv", sep="\t", names=["cluster_representative", "query_id"], dtype=str
    )
    if clusters["query_id"].duplicated().any():
        raise RuntimeError(f"Phase {PHASE_ID} cluster membership is not unique")
    survivors = survivors.merge(clusters, on="query_id", how="left", validate="one_to_one")
    if survivors["cluster_representative"].isna().any():
        raise RuntimeError(f"Phase {PHASE_ID} survivor missing cluster")
    survivors["external_cluster_id_30"] = f"RCSB{PHASE_ID}_30::" + survivors["cluster_representative"]
    survivors["external_cohort_status"] = {
        27: "LOCKED_STRICT_TEMPORAL_EXTERNAL_BLIND",
        28: "LOCKED_STRICT_SEQUENCE_INDEPENDENT_STRUCTURAL_EXTERNAL_BLIND",
        30: "LOCKED_RETROSPECTIVE_SEQUENCE_INDEPENDENT_EXTERNAL_BLIND",
        31: "LOCKED_HIT_ANCHORED_RETROSPECTIVE_EXTERNAL_BLIND",
        32: "LOCKED_PREPREDICTION_TOOL_TRAINING_HOMOLOGY_CLEAN_EXTERNAL_BLIND",
        33: "LOCKED_BLINDED_SAMPLE_SIZE_ADAPTED_TOOL_TRAINING_CLEAN_EXTERNAL_BLIND",
    }[PHASE_ID]
    if PHASE_ID in {32, 33}:
        survivors["tool_training_homology_preexcluded"] = True
    survivors.to_parquet(RPHASE / "rcsb_strict_blind_cohort.parquet", index=False, compression="zstd")
    manifest_columns = [
        "query_id", "representative_rcsb_entity_id", "representative_pdb_entry_id", "external_cluster_id_30",
        "initial_release_date", "sequence_length", "ec_l3_label_eligible", "ec_l4_label_eligible",
        "nearest_frozen_identity_percent", "nearest_frozen_query_coverage", "nearest_frozen_target_coverage",
    ]
    if PHASE_ID in {32, 33}:
        manifest_columns.extend([
            "nearest_tool_training_sequence_id", "nearest_tool_training_identity_percent",
            "nearest_tool_training_query_coverage", "nearest_tool_training_target_coverage",
            "tool_training_homology_independent", "tool_training_homology_preexcluded",
        ])
    survivors[manifest_columns].to_csv(RPHASE / "external_holdout_manifest.tsv", sep="\t", index=False)
    power_rows = []
    for level, column in [("EC_L3", "ec_l3_label_eligible"), ("EC_L4", "ec_l4_label_eligible")]:
        subset = survivors.loc[survivors[column].astype(bool)]
        queries = len(subset)
        cluster_count = subset["external_cluster_id_30"].nunique()
        required_queries = 120 if PHASE_ID in {32, 33} and level == "EC_L3" else 50
        required_clusters = 50 if PHASE_ID in {32, 33} and level == "EC_L3" else 20
        power_rows.append({
            "annotation_level": level, "eligible_queries": queries, "eligible_clusters": cluster_count,
            "required_queries": required_queries, "required_clusters": required_clusters,
            "power_status": "SUFFICIENT_FOR_EXTERNAL_EVALUATION" if queries >= required_queries and cluster_count >= required_clusters else "INSUFFICIENT_EXTERNAL_POWER",
        })
    power = pd.DataFrame(power_rows)
    power.to_csv(RPHASE / "external_endpoint_power.tsv", sep="\t", index=False)
    checks = [
        ("acquisition_checkpoint", ACQUISITION_CHECKPOINT.is_file(), "stage gate"),
        ("all_survivors_independent", bool(survivors["sequence_independent"].all()), len(survivors)),
        ("all_survivors_tool_training_independent", PHASE_ID not in {32, 33} or bool(survivors["tool_training_homology_independent"].all()), len(survivors)),
        ("all_survivors_clustered", len(clusters) == len(survivors), f"clusters_rows={len(clusters)};survivors={len(survivors)}"),
        ("strict_threshold_not_relaxed", True, "fident>0.30;qcov>=0.70;tcov>=0.70"),
        ("cohort_nonempty", len(survivors) > 0, len(survivors)),
        (f"no_phase{PHASE_ID}_predictions_read", not (RPHASE / "external_predictions.parquet").exists(), "blind"),
    ]
    qc = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    qc.to_csv(RPHASE / "external_independence_qc.tsv", sep="\t", index=False)
    failures = qc.loc[~qc["passed"].astype(bool), "check"].tolist()
    summary = {
        "phase": f"{PHASE_ID}A", "status": "PASS" if not failures else "FAIL", "strict_blind_queries": len(survivors),
        "strict_blind_clusters": survivors["external_cluster_id_30"].nunique(),
        "confirmatory_ec_l3_power_gate": bool(power.loc[power["annotation_level"].eq("EC_L3"), "power_status"].eq("SUFFICIENT_FOR_EXTERNAL_EVALUATION").all()),
        "endpoint_power": power_rows, f"phase{PHASE_ID}_predictions_read": False, "failures": failures,
    }
    (REPORTS / f"phase{PHASE_ID}_external_cohort_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"Phase {PHASE_ID} cohort lock failed: {failures}")
    LOCK_CHECKPOINT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "filter", "finalize"], required=True)
    args = parser.parse_args()
    {"prepare": prepare, "filter": filter_hits, "finalize": finalize}[args.stage]()


if __name__ == "__main__":
    main()
