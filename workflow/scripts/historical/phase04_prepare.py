#!/usr/bin/env python3
"""Prepare leakage-safe Phase 4 sequence/structure retrieval inputs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in frame[["protein_id", "sequence"]].itertuples(index=False):
            handle.write(f">{row.protein_id}\n")
            for start in range(0, len(row.sequence), 80):
                handle.write(row.sequence[start:start + 80] + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    source = args.source_root.resolve()
    work = project / "data" / "interim" / "phase04"
    reference_dir = project / "data" / "reference"
    work.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    split = pd.read_parquet(project / "data" / "splits" / "split_sequence.parquet")
    proteins = pd.read_parquet(
        project / "data" / "processed" / "protein_table.parquet",
        columns=["protein_id", "sequence", "sequence_valid", "fragment_status"],
    )
    protein_sequences = proteins[
        proteins["sequence_valid"] & (proteins["fragment_status"] == "complete")
    ][["protein_id", "sequence"]]
    benchmark = split.merge(protein_sequences, on="protein_id", how="inner", validate="one_to_one")
    references = benchmark[benchmark["split"] == "train"].sort_values("protein_id").copy()
    queries = benchmark[benchmark["split"].isin(["validation", "test"])].sort_values("protein_id").copy()
    if set(references["protein_id"]) & set(queries["protein_id"]):
        raise RuntimeError("Reference/query protein overlap detected")

    write_fasta(references, work / "reference_sequences.fasta")
    write_fasta(queries, work / "query_sequences.fasta")

    query_truth = queries[[
        "protein_id", "split", "cluster_id_30", "documented_activity_count",
        "ec_l1_json", "ec_l3_json", "ec_l4_json", "canonical_rhea_json",
    ]].copy()
    query_truth["provenance"] = "GROUND_TRUTH_ONLY_EVALUATION_DO_NOT_JOIN_TO_MODEL_FEATURES"
    write_parquet(query_truth, work / "retrieval_query_truth.parquet")

    activities = pd.read_parquet(project / "data" / "processed" / "activity_table_canonical.parquet")
    allowed = activities[
        activities["protein_id"].isin(set(references["protein_id"]))
        & activities["evidence_tier"].isin(["GOLD", "SILVER"])
        & activities["canonical_ec"].notna()
    ].copy()
    allowed = allowed.rename(columns={"protein_id": "reference_protein_id"})
    allowed["reference_partition"] = "train"
    allowed["provenance"] = "REFERENCE_ACTIVITY_ALLOWED_AT_INFERENCE"
    write_parquet(allowed, reference_dir / "activity_reference_library.parquet")

    af_index = pd.read_parquet(
        source / "data" / "raw" / "alphafold" / "bulk_swissprot_v6" / "afdb_bulk_file_index.parquet"
    )
    af_index = af_index.sort_values(["uniprot_accession", "fragment", "filename"]).drop_duplicates("uniprot_accession")
    membership = split[["protein_id", "split", "cluster_id_30"]].merge(
        af_index[["uniprot_accession", "filename", "model_version", "compressed_size"]],
        left_on="protein_id", right_on="uniprot_accession", how="left", validate="one_to_one",
    ).drop(columns=["uniprot_accession"])
    membership["has_structure"] = membership["filename"].notna()
    membership["structure_source"] = membership["has_structure"].map(
        {True: "AlphaFoldDB_SwissProt_v6", False: "UNAVAILABLE_IN_FROZEN_BULK_INDEX"}
    )
    write_parquet(membership, work / "afdb_membership.parquet")

    raw_inventory = source / "data" / "manifests" / "resource_inventory.tsv"
    summary = {
        "stage": "phase04_prepare",
        "created_at": now(),
        "references": len(references),
        "queries": len(queries),
        "query_split_counts": queries["split"].value_counts().to_dict(),
        "reference_activity_rows": len(allowed),
        "reference_activity_proteins": allowed["reference_protein_id"].nunique(),
        "structure_coverage": membership.groupby("split")["has_structure"].agg(["sum", "count", "mean"]).reset_index().to_dict("records"),
        "reference_fasta_sha256": sha256_file(work / "reference_sequences.fasta"),
        "query_fasta_sha256": sha256_file(work / "query_sequences.fasta"),
        "raw_inventory_sha256": sha256_file(raw_inventory),
    }
    (work / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
