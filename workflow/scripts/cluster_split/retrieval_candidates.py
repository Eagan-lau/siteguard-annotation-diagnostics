#!/usr/bin/env python3
"""Convert raw sequence/structure hits to leakage-safe, cluster-deduplicated candidates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


AF_NAME = re.compile(r"AF-([A-Z0-9]+)-F\d+-model_v\d+", re.IGNORECASE)


def normalize_structure_name(value: str) -> str:
    match = AF_NAME.search(str(value))
    return match.group(1).upper() if match else str(value).split()[0]


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--modality", choices=["mmseqs", "foldseek"], required=True)
    parser.add_argument("--raw-hits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    root = args.project_root.resolve()
    split = pd.read_parquet(
        root / "data" / "splits" / "split_sequence.parquet",
        columns=["protein_id", "split", "cluster_id_30"],
    )
    partition = split.set_index("protein_id")["split"].to_dict()
    cluster = split.set_index("protein_id")["cluster_id_30"].to_dict()

    columns = [
        "query", "target", "fident", "alnlen", "qstart", "qend", "qlen",
        "tstart", "tend", "tlen", "qcov", "tcov", "evalue", "bits",
    ]
    if args.modality == "foldseek":
        columns += ["lddt", "qtmscore", "ttmscore", "alntmscore", "rmsd", "prob"]
    hits = pd.read_csv(args.raw_hits, sep="\t", names=columns, header=None, low_memory=False)
    if args.modality == "foldseek":
        hits["query"] = hits["query"].map(normalize_structure_name)
        hits["target"] = hits["target"].map(normalize_structure_name)
    hits = hits.rename(columns={"query": "query_protein_id", "target": "reference_protein_id"})
    hits["_input_order"] = range(len(hits))
    hits["query_partition"] = hits["query_protein_id"].map(partition)
    hits["reference_partition"] = hits["reference_protein_id"].map(partition)
    hits["reference_cluster_id_30"] = hits["reference_protein_id"].map(cluster)
    unknown_queries = int(hits["query_partition"].isna().sum())
    unknown_references = int(hits["reference_partition"].isna().sum())
    if unknown_queries or unknown_references:
        raise RuntimeError(f"Unknown IDs after normalization: queries={unknown_queries}, references={unknown_references}")
    invalid_query = int((~hits["query_partition"].isin(["validation", "test"])).sum())
    invalid_reference = int((hits["reference_partition"] != "train").sum())
    self_hits = int((hits["query_protein_id"] == hits["reference_protein_id"]).sum())
    if invalid_query or invalid_reference or self_hits:
        raise RuntimeError(
            f"Leakage guard failed: invalid_query={invalid_query}, invalid_reference={invalid_reference}, self_hits={self_hits}"
        )

    hits = hits.sort_values(["query_protein_id", "_input_order"], kind="stable")
    hits["raw_rank"] = hits.groupby("query_protein_id", sort=False).cumcount() + 1
    before = len(hits)
    hits = hits.drop_duplicates(["query_protein_id", "reference_cluster_id_30"], keep="first")
    hits["modality_rank"] = hits.groupby("query_protein_id", sort=False).cumcount() + 1
    hits = hits[hits["modality_rank"] <= args.top_k].copy()
    hits["modality"] = args.modality
    hits["candidate_provenance"] = "QUERY_DERIVED_RETRIEVAL_PLUS_TRAIN_REFERENCE_METADATA"
    hits = hits.drop(columns=["_input_order"])

    forbidden = {"canonical_ec", "canonical_rhea", "ec_l3_json", "ec_l4_json", "canonical_rhea_json"}
    if forbidden & set(hits.columns):
        raise RuntimeError("Query ground-truth field leaked into candidate schema")
    write_parquet(hits, args.output)
    query_counts = hits.groupby("query_protein_id")["modality_rank"].max()
    summary = {
        "modality": args.modality,
        "raw_rows": before,
        "deduplicated_top_k_rows": len(hits),
        "queries_with_candidates": int(query_counts.size),
        "queries_with_100_candidates": int((query_counts >= args.top_k).sum()),
        "median_candidates": float(query_counts.median()) if len(query_counts) else 0.0,
        "maximum_rank": int(query_counts.max()) if len(query_counts) else 0,
        "reference_partitions": sorted(hits["reference_partition"].unique().tolist()),
        "query_partitions": sorted(hits["query_partition"].unique().tolist()),
        "reference_clusters_removed": before - len(hits),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
