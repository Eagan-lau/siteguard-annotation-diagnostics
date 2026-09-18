#!/usr/bin/env python3
"""Prepare known protein pairs for leakage-safe direct MMseqs/Foldseek alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


SEED = 20260819
SHARDS = 4
PAIR_KEY = ["query_protein_id", "reference_protein_id"]
MMSEQS_FEATURES = ["mmseqs_rank", "mmseqs_fident", "mmseqs_qcov", "mmseqs_tcov", "mmseqs_bits"]
FOLDSEEK_FEATURES = ["foldseek_rank", "foldseek_fident", "foldseek_qcov", "foldseek_tcov", "foldseek_bits"]


def sequence_lookup(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(
        path, separator="\t", has_header=False,
        new_columns=["db_key", "protein_id", "source_file"],
        schema_overrides={"db_key": pl.UInt64, "protein_id": pl.String, "source_file": pl.UInt32},
    ).select("db_key", "protein_id")
    if frame["protein_id"].n_unique() != frame.height:
        raise RuntimeError(f"Sequence lookup is not protein-unique: {path}")
    return frame


def structure_lookup(db: Path) -> pl.DataFrame:
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
            pl.col("structure_name")
            .str.extract(r"^AF-(.+)-F[0-9]+-model_v[0-9]+$", 1)
            .alias("protein_id")
        )
        .select("db_key", "protein_id")
    )
    if frame["protein_id"].null_count() or frame["protein_id"].n_unique() != frame.height:
        raise RuntimeError(f"Structure lookup cannot be mapped uniquely to proteins: {db}")
    return frame


def write_shards(frame: pl.DataFrame, stem: str, work: Path) -> list[int]:
    keyed = (
        frame.select("query_db_key", "target_db_key")
        .unique()
        .with_columns(
            (pl.concat_str([pl.col("query_db_key"), pl.col("target_db_key")], separator="|")
             .hash(seed=SEED) % SHARDS).alias("shard"),
            pl.lit(2000, dtype=pl.Int32).alias("prefilter_score"),
            pl.lit(0, dtype=pl.Int32).alias("diagonal"),
        )
    )
    counts: list[int] = []
    for shard in range(SHARDS):
        output = work / f"direct_{stem}_chunk_{shard}.tsv"
        subset = (
            keyed.filter(pl.col("shard") == shard)
            .select("query_db_key", "target_db_key", "prefilter_score", "diagonal")
            .sort(["query_db_key", "target_db_key"])
        )
        subset.write_csv(output, separator="\t", include_header=False)
        counts.append(subset.height)
    return counts


def attach_keys(
    pairs: pl.DataFrame, query_map: pl.DataFrame, target_map: pl.DataFrame,
) -> pl.DataFrame:
    return (
        pairs.join(
            query_map.rename({"db_key": "query_db_key", "protein_id": "query_protein_id"}),
            on="query_protein_id",
            how="left", validate="m:1",
        )
        .join(
            target_map.rename({"db_key": "target_db_key", "protein_id": "reference_protein_id"}),
            on="reference_protein_id",
            how="left", validate="m:1",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if not (root / "checkpoints/CHECKPOINT_05_PASS").is_file():
        raise RuntimeError("CHECKPOINT_05_PASS is required before direct-alignment preparation")
    work = root / "data/interim/phase06/direct_alignment"
    work.mkdir(parents=True, exist_ok=True)

    columns = PAIR_KEY + ["query_split_expected"] + MMSEQS_FEATURES + FOLDSEEK_FEATURES
    pair_frames = [
        pl.read_parquet(root / "data/processed/population_pairs.parquet", columns=columns),
        pl.read_parquet(root / "data/processed/training_pairs.parquet", columns=columns),
    ]
    pairs = (
        pl.concat(pair_frames, how="vertical_relaxed")
        .group_by(PAIR_KEY)
        .agg(
            pl.col("query_split_expected").drop_nulls().first(),
            *[pl.col(name).drop_nulls().first() for name in MMSEQS_FEATURES + FOLDSEEK_FEATURES],
        )
    )
    inconsistent_splits = (
        pl.concat(pair_frames, how="vertical_relaxed")
        .group_by(PAIR_KEY).agg(pl.col("query_split_expected").n_unique().alias("n"))
        .filter(pl.col("n") != 1).height
    )
    if inconsistent_splits:
        raise RuntimeError(f"Protein pairs have inconsistent query splits: {inconsistent_splits}")
    seq_needed = pairs.filter(pl.any_horizontal([pl.col(name).is_null() for name in MMSEQS_FEATURES]))
    struct_needed = pairs.filter(pl.any_horizontal([pl.col(name).is_null() for name in FOLDSEEK_FEATURES]))

    mmseqs = root / "databases/mmseqs"
    seq_ref = sequence_lookup(mmseqs / "reference_seq.lookup")
    seq_eval = sequence_lookup(mmseqs / "query_seq.lookup")
    seq_train_pairs = attach_keys(
        seq_needed.filter(pl.col("query_split_expected") == "train"), seq_ref, seq_ref,
    )
    seq_eval_pairs = attach_keys(
        seq_needed.filter(pl.col("query_split_expected") != "train"), seq_eval, seq_ref,
    )
    for name, frame in {"sequence_train": seq_train_pairs, "sequence_eval": seq_eval_pairs}.items():
        missing_keys = frame.filter(pl.col("query_db_key").is_null() | pl.col("target_db_key").is_null()).height
        if missing_keys:
            raise RuntimeError(f"{name} has {missing_keys} pairs absent from sequence databases")

    foldseek = root / "databases/foldseek"
    struct_ref = structure_lookup(foldseek / "reference_structure")
    struct_eval = structure_lookup(foldseek / "query_structure")
    struct_train_pairs = attach_keys(
        struct_needed.filter(pl.col("query_split_expected") == "train"), struct_ref, struct_ref,
    )
    struct_eval_pairs = attach_keys(
        struct_needed.filter(pl.col("query_split_expected") != "train"), struct_eval, struct_ref,
    )
    struct_train_available = struct_train_pairs.filter(
        pl.col("query_db_key").is_not_null() & pl.col("target_db_key").is_not_null()
    )
    struct_eval_available = struct_eval_pairs.filter(
        pl.col("query_db_key").is_not_null() & pl.col("target_db_key").is_not_null()
    )

    shard_counts = {
        "sequence_train": write_shards(seq_train_pairs, "sequence_train", work),
        "sequence_eval": write_shards(seq_eval_pairs, "sequence_eval", work),
        "structure_train": write_shards(struct_train_available, "structure_train", work),
        "structure_eval": write_shards(struct_eval_available, "structure_eval", work),
    }
    manifest = pairs.select(
        PAIR_KEY + ["query_split_expected"] + MMSEQS_FEATURES + FOLDSEEK_FEATURES
    ).with_columns(
        pl.any_horizontal([pl.col(name).is_null() for name in MMSEQS_FEATURES]).alias("direct_mmseqs_required"),
        pl.any_horizontal([pl.col(name).is_null() for name in FOLDSEEK_FEATURES]).alias("direct_foldseek_required"),
    )
    manifest.write_parquet(work / "direct_alignment_manifest.parquet", compression="zstd")
    summary = {
        "status": "PASS", "unique_protein_pairs": pairs.height,
        "mmseqs_pairs_missing_retrieval_features": seq_needed.height,
        "foldseek_pairs_missing_retrieval_features": struct_needed.height,
        "direct_mmseqs_scheduled": seq_train_pairs.height + seq_eval_pairs.height,
        "direct_foldseek_scheduled": struct_train_available.height + struct_eval_available.height,
        "foldseek_pairs_without_both_structures": (
            struct_train_pairs.height + struct_eval_pairs.height -
            struct_train_available.height - struct_eval_available.height
        ),
        "shards": SHARDS, "shard_counts": shard_counts,
        "feature_policy": "Direct known-pair alignments fill retrieval-origin missingness without query functional labels",
    }
    (work / "direct_alignment_prepare_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
