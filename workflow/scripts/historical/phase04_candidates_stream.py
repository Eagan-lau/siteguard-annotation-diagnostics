#!/usr/bin/env python3
"""Stream raw retrieval hits into cluster-deduplicated Parquet candidates."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


AF_NAME = re.compile(r"AF-([A-Z0-9]+)-F\d+-model_v\d+", re.IGNORECASE)
BASE_COLUMNS = (
    "fident", "alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen",
    "qcov", "tcov", "evalue", "bits",
)
FOLD_COLUMNS = ("lddt", "qtmscore", "ttmscore", "alntmscore", "rmsd", "prob")
INTEGER_COLUMNS = {"alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen"}


def normalize_structure_name(value: str) -> str:
    match = AF_NAME.search(value)
    return match.group(1).upper() if match else value.split()[0]


def schema(modality: str, foldseek_basic: bool) -> pa.Schema:
    fields = [
        pa.field("query_protein_id", pa.string()),
        pa.field("reference_protein_id", pa.string()),
    ]
    extended_columns = FOLD_COLUMNS if modality == "foldseek" and not foldseek_basic else ()
    for column in BASE_COLUMNS + extended_columns:
        fields.append(pa.field(column, pa.int64() if column in INTEGER_COLUMNS else pa.float64()))
    fields += [
        pa.field("query_partition", pa.string()),
        pa.field("reference_partition", pa.string()),
        pa.field("reference_cluster_id_30", pa.string()),
        pa.field("raw_rank", pa.int32()),
        pa.field("modality_rank", pa.int16()),
        pa.field("modality", pa.string()),
        pa.field("candidate_provenance", pa.string()),
    ]
    return pa.schema(fields)


def empty_batch(output_schema: pa.Schema) -> dict[str, list]:
    return {field.name: [] for field in output_schema}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--modality", choices=["mmseqs", "foldseek"], required=True)
    parser.add_argument("--raw-hits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-rows", type=int, default=100000)
    parser.add_argument("--foldseek-basic", action="store_true", help="Foldseek Phase 4 hits without CIGAR-derived TM/LDDT/RMSD fields")
    parser.add_argument(
        "--allowed-query-partitions",
        nargs="+",
        default=["validation", "test"],
        choices=["train", "validation", "test"],
        help="Partitions permitted on the query side; Phase 4 defaults to held-out queries.",
    )
    parser.add_argument(
        "--skip-self-hits",
        action="store_true",
        help="Drop self hits for train-to-train retrieval instead of treating them as an invariant failure.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    split = pd.read_parquet(
        root / "data" / "splits" / "split_sequence.parquet",
        columns=["protein_id", "split", "cluster_id_30"],
    )
    partition = split.set_index("protein_id")["split"].to_dict()
    cluster = split.set_index("protein_id")["cluster_id_30"].to_dict()
    output_schema = schema(args.modality, args.foldseek_basic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, output_schema, compression="zstd")
    batch = empty_batch(output_schema)
    raw_rows = 0
    written_rows = 0
    current_query: str | None = None
    completed_queries: set[str] = set()
    seen_clusters: set[str] = set()
    raw_rank = 0
    modality_rank = 0
    candidate_counts: list[int] = []
    numeric_columns = BASE_COLUMNS + (FOLD_COLUMNS if args.modality == "foldseek" and not args.foldseek_basic else ())

    def flush() -> None:
        nonlocal batch
        if batch["query_protein_id"]:
            writer.write_table(pa.Table.from_pydict(batch, schema=output_schema))
            batch = empty_batch(output_schema)

    try:
        with args.raw_hits.open("r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 2 + len(numeric_columns):
                    raise RuntimeError(f"Unexpected {args.modality} raw field count: {len(fields)}")
                raw_rows += 1
                query, reference = fields[0], fields[1]
                if args.modality == "foldseek":
                    query, reference = normalize_structure_name(query), normalize_structure_name(reference)
                if query != current_query:
                    if current_query is not None:
                        completed_queries.add(current_query)
                        candidate_counts.append(modality_rank)
                    if query in completed_queries:
                        raise RuntimeError(f"Raw search output is not query-grouped: {query}")
                    current_query = query
                    seen_clusters = set()
                    raw_rank = 0
                    modality_rank = 0
                raw_rank += 1
                query_partition = partition.get(query)
                reference_partition = partition.get(reference)
                reference_cluster = cluster.get(reference)
                if query_partition not in set(args.allowed_query_partitions):
                    raise RuntimeError(f"Invalid or unknown query partition for {query}: {query_partition}")
                if reference_partition != "train" or reference_cluster is None:
                    raise RuntimeError(f"Invalid or unknown reference for {reference}: {reference_partition}")
                if query == reference:
                    if args.skip_self_hits:
                        continue
                    raise RuntimeError(f"Self hit in strict retrieval: {query}")
                if reference_cluster in seen_clusters or modality_rank >= args.top_k:
                    continue
                seen_clusters.add(reference_cluster)
                modality_rank += 1
                values = fields[2:]
                batch["query_protein_id"].append(query)
                batch["reference_protein_id"].append(reference)
                for column, value in zip(numeric_columns, values):
                    batch[column].append(int(float(value)) if column in INTEGER_COLUMNS else float(value))
                batch["query_partition"].append(query_partition)
                batch["reference_partition"].append(reference_partition)
                batch["reference_cluster_id_30"].append(reference_cluster)
                batch["raw_rank"].append(raw_rank)
                batch["modality_rank"].append(modality_rank)
                batch["modality"].append(args.modality)
                batch["candidate_provenance"].append("QUERY_DERIVED_RETRIEVAL_PLUS_TRAIN_REFERENCE_METADATA")
                written_rows += 1
                if len(batch["query_protein_id"]) >= args.batch_rows:
                    flush()
        if current_query is not None:
            candidate_counts.append(modality_rank)
        flush()
    finally:
        writer.close()
    temporary.replace(args.output)
    summary = {
        "modality": args.modality,
        "raw_rows": raw_rows,
        "deduplicated_top_k_rows": written_rows,
        "queries_with_candidates": len(candidate_counts),
        "queries_with_100_candidates": sum(value >= args.top_k for value in candidate_counts),
        "median_candidates": float(statistics.median(candidate_counts)) if candidate_counts else 0.0,
        "maximum_rank": max(candidate_counts, default=0),
        "reference_partitions": ["train"],
        "query_partitions": sorted(args.allowed_query_partitions),
        "reference_clusters_removed_or_beyond_top_k": raw_rows - written_rows,
        "streaming_parser": True,
        "foldseek_basic_fields": bool(args.modality == "foldseek" and args.foldseek_basic),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
