#!/usr/bin/env python3
"""Merge direct known-pair MMseqs/Foldseek shards into model-safe feature tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


SHARDS = 4
RAW_COLUMNS = [
    "query", "target", "fident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue", "bits",
]


def read_shards(work: Path, classes: list[str], modality: str) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for name in classes:
        for shard in range(SHARDS):
            path = work / f"direct_{name}_chunk_{shard}_hits.tsv"
            if not path.is_file():
                raise RuntimeError(f"Missing direct-alignment output: {path}")
            if path.stat().st_size == 0:
                continue
            frame = pl.read_csv(
                path, separator="\t", has_header=False, new_columns=RAW_COLUMNS,
                schema_overrides={
                    "query": pl.String, "target": pl.String, "fident": pl.Float32,
                    "alnlen": pl.UInt32, "qlen": pl.UInt32, "tlen": pl.UInt32,
                    "qcov": pl.Float32, "tcov": pl.Float32, "evalue": pl.Float64,
                    "bits": pl.Float32,
                },
            )
            frames.append(frame)
    if not frames:
        return pl.DataFrame(schema={
            "query_protein_id": pl.String, "reference_protein_id": pl.String,
            f"direct_{modality}_fident": pl.Float32, f"direct_{modality}_alnlen": pl.UInt32,
            f"direct_{modality}_qlen": pl.UInt32, f"direct_{modality}_tlen": pl.UInt32,
            f"direct_{modality}_qcov": pl.Float32, f"direct_{modality}_tcov": pl.Float32,
            f"direct_{modality}_evalue": pl.Float64, f"direct_{modality}_bits": pl.Float32,
        })
    combined = pl.concat(frames, how="vertical")
    if modality == "foldseek":
        combined = combined.with_columns(
            pl.col("query").str.extract(r"^AF-(.+)-F[0-9]+-model_v[0-9]+$", 1),
            pl.col("target").str.extract(r"^AF-(.+)-F[0-9]+-model_v[0-9]+$", 1),
        )
        if combined["query"].null_count() or combined["target"].null_count():
            raise RuntimeError("Foldseek identifiers could not be normalized to UniProt accessions")
    combined = combined.rename({
        "query": "query_protein_id", "target": "reference_protein_id",
        **{name: f"direct_{modality}_{name}" for name in RAW_COLUMNS[2:]},
    })
    if combined.select(pl.struct(["query_protein_id", "reference_protein_id"]).n_unique()).item() != combined.height:
        raise RuntimeError(f"Duplicate direct {modality} protein pairs")
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data/interim/phase06/direct_alignment"
    prepare = json.loads((work / "direct_alignment_prepare_summary.json").read_text(encoding="utf-8"))
    sequence = read_shards(work, ["sequence_train", "sequence_eval"], "mmseqs")
    structure = read_shards(work, ["structure_train", "structure_eval"], "foldseek")
    expected_sequence = int(prepare["direct_mmseqs_scheduled"])
    expected_structure = int(prepare["direct_foldseek_scheduled"])
    if sequence.height != expected_sequence or structure.height != expected_structure:
        raise RuntimeError(
            "Direct-alignment merge count mismatch: "
            f"sequence={sequence.height}/{expected_sequence}, structure={structure.height}/{expected_structure}"
        )
    sequence.write_parquet(work / "direct_mmseqs_alignments.parquet", compression="zstd")
    structure.write_parquet(work / "direct_foldseek_alignments.parquet", compression="zstd")
    summary = {
        "status": "PASS", "direct_mmseqs_rows": sequence.height,
        "direct_foldseek_rows": structure.height,
        "mmseqs_unique_pairs": sequence.height, "foldseek_unique_pairs": structure.height,
        "label_fields_used": [],
        "provenance": "QUERY_REFERENCE_DERIVED_DIRECT_KNOWN_PAIR_ALIGNMENT",
    }
    (work / "direct_alignment_merge_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
