#!/usr/bin/env python3
"""Select a deterministic label-blind structural calibration panel for US-align."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


SEED = 20260819
CHUNKS = 10
PER_STRATUM = 400


def full_structure_lookup(db: Path) -> pl.DataFrame:
    lookup = pl.read_csv(
        Path(f"{db}.lookup"), separator="\t", has_header=False,
        new_columns=["db_key", "structure_name", "source_file"],
        schema_overrides={"db_key": pl.UInt64, "structure_name": pl.String, "source_file": pl.UInt32},
    ).select("db_key", "structure_name")
    valid = pl.read_csv(
        Path(f"{db}.index"), separator="\t", has_header=False,
        new_columns=["db_key", "offset", "record_size"],
        schema_overrides={"db_key": pl.UInt64, "offset": pl.UInt64, "record_size": pl.UInt64},
    ).select("db_key")
    frame = (
        valid.join(lookup, on="db_key", how="left", validate="1:1")
        .with_columns(
            pl.col("structure_name").str.extract(r"^AF-(.+)-F[0-9]+-model_v[0-9]+$", 1).alias("protein_id")
        )
        .select("db_key", "structure_name", "protein_id")
    )
    if frame["protein_id"].null_count() or frame["protein_id"].n_unique() != frame.height:
        raise RuntimeError("Full AlphaFoldDB Foldseek lookup is not protein-unique")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data/interim/phase06/usalign"
    work.mkdir(parents=True, exist_ok=True)
    pair_path = root / "data/interim/phase06/global_pair_features.parquet"
    if not pair_path.is_file():
        raise RuntimeError("global_pair_features.parquet is required before US-align selection")
    pairs = (
        pl.read_parquet(pair_path)
        .filter(pl.col("both_structures_available") & pl.col("foldseek_identity").is_not_null())
        .with_columns(
            pl.when(pl.col("foldseek_identity") < 0.10).then(pl.lit("00-10"))
            .when(pl.col("foldseek_identity") < 0.20).then(pl.lit("10-20"))
            .when(pl.col("foldseek_identity") < 0.30).then(pl.lit("20-30"))
            .when(pl.col("foldseek_identity") < 0.50).then(pl.lit("30-50"))
            .otherwise(pl.lit("50-100")).alias("foldseek_identity_bin"),
            pl.concat_str(["query_protein_id", "reference_protein_id"], separator="|")
            .hash(seed=SEED).alias("selection_hash"),
        )
        .sort(["query_split", "foldseek_identity_bin", "selection_hash"])
        .group_by(["query_split", "foldseek_identity_bin"], maintain_order=True)
        .head(PER_STRATUM)
    )
    structures = full_structure_lookup(root / "databases/foldseek/full_afdb")
    selected = (
        pairs.join(
            structures.rename({
                "protein_id": "query_protein_id", "db_key": "query_structure_key",
                "structure_name": "query_structure_name",
            }), on="query_protein_id", how="left", validate="m:1",
        )
        .join(
            structures.rename({
                "protein_id": "reference_protein_id", "db_key": "reference_structure_key",
                "structure_name": "reference_structure_name",
            }), on="reference_protein_id", how="left", validate="m:1",
        )
    )
    key_columns = ["query_structure_key", "reference_structure_key"]
    if any(selected[name].null_count() for name in key_columns):
        raise RuntimeError("Selected US-align pairs cannot be mapped to full AlphaFoldDB keys")
    selected = selected.with_row_index("usalign_pair_index").with_columns(
        pl.concat_str([
            pl.lit(str(work / "pdb")),
            pl.concat_str([pl.col("query_structure_name"), pl.lit(".pdb")]),
        ], separator="/")
        .alias("query_pdb"),
        pl.concat_str([
            pl.lit(str(work / "pdb")),
            pl.concat_str([pl.col("reference_structure_name"), pl.lit(".pdb")]),
        ], separator="/")
        .alias("reference_pdb"),
    )
    selected.write_parquet(work / "selected_usalign_pairs.parquet", compression="zstd")
    keys = pl.concat([
        selected.select(pl.col("query_structure_key").alias("db_key")),
        selected.select(pl.col("reference_structure_key").alias("db_key")),
    ]).unique().sort("db_key")
    keys.write_csv(work / "selected_structure.keys", include_header=False)
    chunk_counts: list[int] = []
    for chunk in range(CHUNKS):
        subset = selected.filter(pl.col("usalign_pair_index") % CHUNKS == chunk).select(
            "usalign_pair_index", "query_protein_id", "reference_protein_id", "query_pdb", "reference_pdb"
        ).sort("usalign_pair_index")
        subset.write_csv(work / f"usalign_pairs_chunk_{chunk}.tsv", separator="\t")
        chunk_counts.append(subset.height)
    stratum_counts = (
        selected.group_by(["query_split", "foldseek_identity_bin"]).len()
        .sort(["query_split", "foldseek_identity_bin"]).to_dicts()
    )
    summary = {
        "status": "PASS", "selection_rule": "label-blind deterministic hash within query-split and Foldseek-identity bins",
        "selected_pairs": selected.height, "selected_structures": keys.height,
        "chunks": CHUNKS, "chunk_counts": chunk_counts, "stratum_counts": stratum_counts,
        "ground_truth_fields_used": [], "model_feature_use": "CALIBRATION_ONLY_NOT_SPARSE_MODEL_INPUT",
    }
    (work / "usalign_prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
