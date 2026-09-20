#!/usr/bin/env python3
"""Prepare activity-linked reference sites and exact known-pair mapping alignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


SEED = 20260819
SHARDS = 4
PAIR_KEY = ["query_protein_id", "reference_protein_id"]


def sequence_lookup(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(
        path, separator="\t", has_header=False,
        new_columns=["db_key", "protein_id", "source_file"],
        schema_overrides={"db_key": pl.UInt64, "protein_id": pl.String, "source_file": pl.UInt32},
    ).select("db_key", "protein_id")
    if frame["protein_id"].n_unique() != frame.height:
        raise RuntimeError(f"Non-unique sequence lookup: {path}")
    return frame


def write_shards(frame: pl.DataFrame, stem: str, work: Path) -> list[int]:
    keyed = frame.select("query_db_key", "target_db_key").unique().with_columns(
        (pl.concat_str(["query_db_key", "target_db_key"], separator="|").hash(seed=SEED) % SHARDS)
        .alias("shard"),
        pl.lit(2000, dtype=pl.Int32).alias("prefilter_score"),
        pl.lit(0, dtype=pl.Int32).alias("diagonal"),
    )
    counts: list[int] = []
    for shard in range(SHARDS):
        subset = keyed.filter(pl.col("shard") == shard).select(
            "query_db_key", "target_db_key", "prefilter_score", "diagonal"
        ).sort(["query_db_key", "target_db_key"])
        subset.write_csv(work / f"site_{stem}_chunk_{shard}.tsv", separator="\t", include_header=False)
        counts.append(subset.height)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if not (root / "checkpoints/CHECKPOINT_06_PASS").is_file():
        raise RuntimeError("CHECKPOINT_06_PASS is required before Phase 7")
    work = root / "data/interim/phase07"
    work.mkdir(parents=True, exist_ok=True)

    sites = pl.read_parquet(root / "data/processed/catalytic_site_table.parquet").filter(
        pl.col("included_in_site_resolved_cohort") & pl.col("activity_id").is_not_null() &
        (pl.col("residue_number") > 0)
    ).with_columns(
        pl.when(pl.col("site_confidence").str.starts_with("TIER_1")).then(1)
        .when(pl.col("site_confidence").str.starts_with("TIER_2")).then(2)
        .when(pl.col("site_confidence").str.starts_with("TIER_3")).then(3)
        .otherwise(9).alias("site_priority")
    )
    # Same activity/residue can be represented by multiple resources. Preserve the strongest
    # evidence record while retaining distinct catalytic residues.
    sites = sites.sort([
        "activity_id", "protein_id", "residue_number", "site_priority", "catalytic_site_id",
    ]).unique(
        subset=["activity_id", "protein_id", "residue_number"], keep="first", maintain_order=True,
    )
    site_activities = sites.select(pl.col("activity_id").alias("reference_activity_id")).unique()
    pair_columns = [
        "query_protein_id", "reference_protein_id", "reference_activity_id",
        "query_split_expected", "query_has_structure", "reference_has_structure",
    ]
    pair_frames = []
    for pair_set, filename in [
        ("population", "population_pairs.parquet"), ("training", "training_pairs.parquet"),
    ]:
        pair_frames.append(
            pl.read_parquet(root / "data/processed" / filename, columns=pair_columns)
            .join(site_activities, on="reference_activity_id", how="inner")
            .with_columns(pl.lit(pair_set).alias("pair_set"))
        )
    pair_rows = pl.concat(pair_frames, how="vertical_relaxed")
    if pair_rows.select(pl.struct([
        "pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id",
    ]).n_unique()).item() != pair_rows.height:
        raise RuntimeError("Activity-linked Phase 7 pair rows are not unique")
    protein_pairs = pair_rows.group_by(PAIR_KEY).agg(
        pl.col("query_split_expected").drop_nulls().first(),
        pl.col("query_has_structure").first(), pl.col("reference_has_structure").first(),
    )
    inconsistent = pair_rows.group_by(PAIR_KEY).agg(
        pl.col("query_split_expected").n_unique().alias("n")
    ).filter(pl.col("n") != 1).height
    if inconsistent:
        raise RuntimeError(f"Site-linked protein pairs have inconsistent query splits: {inconsistent}")

    mmseqs = root / "databases/mmseqs"
    reference = sequence_lookup(mmseqs / "reference_seq.lookup")
    evaluation = sequence_lookup(mmseqs / "query_seq.lookup")
    train = (
        protein_pairs.filter(pl.col("query_split_expected") == "train")
        .join(reference.rename({"protein_id": "query_protein_id", "db_key": "query_db_key"}),
              on="query_protein_id", how="left", validate="m:1")
        .join(reference.rename({"protein_id": "reference_protein_id", "db_key": "target_db_key"}),
              on="reference_protein_id", how="left", validate="m:1")
    )
    eval_pairs = (
        protein_pairs.filter(pl.col("query_split_expected") != "train")
        .join(evaluation.rename({"protein_id": "query_protein_id", "db_key": "query_db_key"}),
              on="query_protein_id", how="left", validate="m:1")
        .join(reference.rename({"protein_id": "reference_protein_id", "db_key": "target_db_key"}),
              on="reference_protein_id", how="left", validate="m:1")
    )
    for name, frame in {"train": train, "eval": eval_pairs}.items():
        missing = frame.filter(pl.col("query_db_key").is_null() | pl.col("target_db_key").is_null()).height
        if missing:
            raise RuntimeError(f"Phase 7 {name} pairs absent from MMseqs databases: {missing}")

    sites.write_parquet(work / "reference_sites_selected.parquet", compression="zstd")
    pair_rows.write_parquet(work / "site_pair_rows.parquet", compression="zstd")
    protein_pairs.write_parquet(work / "site_protein_pairs.parquet", compression="zstd")
    shard_counts = {
        "train": write_shards(train, "train", work),
        "eval": write_shards(eval_pairs, "eval", work),
    }
    summary = {
        "status": "PASS", "selected_reference_sites": sites.height,
        "selected_reference_activities": sites["activity_id"].n_unique(),
        "activity_pair_rows": pair_rows.height, "unique_protein_pairs": protein_pairs.height,
        "both_structures_pair_rows": pair_rows.filter(
            pl.col("query_has_structure") & pl.col("reference_has_structure")
        ).height,
        "shards": SHARDS, "shard_counts": shard_counts,
        "site_selection": "included_in_site_resolved_cohort; exact reference_activity_id; strongest evidence per activity/protein/residue",
        "query_ground_truth_used": False,
    }
    (work / "site_alignment_prepare_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
