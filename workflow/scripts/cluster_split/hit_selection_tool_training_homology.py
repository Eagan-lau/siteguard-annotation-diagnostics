#!/usr/bin/env python3
"""Prepare and score the pre-specified strict tool-training homology sensitivity set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
WORK = ROOT / "data/interim/hit_selection_tool_training_homology"
R31 = ROOT / "results/phase31"
CHECKPOINTS = ROOT / "checkpoints"


def fasta_records(path: Path):
    name: str | None = None
    sequence: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(sequence).upper()
                name = line[1:].split()[0]
                sequence = []
            else:
                sequence.append(line)
    if name is not None:
        yield name, "".join(sequence).upper()


def prepare() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts = {"CLEAN": 0, "HIT_EC": 0}
    output = WORK / "tool_training_unique.fasta"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in pd.read_csv(
            ROOT / "external_tools/CLEAN/app/data/split100.csv", sep="\t", usecols=["Sequence"], chunksize=50000
        ):
            for sequence in chunk["Sequence"].astype(str).str.upper():
                digest = hashlib.sha256(sequence.encode()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest); counts["CLEAN"] += 1
                handle.write(f">CLEAN::{digest[:20]}\n{sequence}\n")
        for _, sequence in fasta_records(ROOT / "external_tools/HIT-EC/data/new-28245.fasta"):
            digest = hashlib.sha256(sequence.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest); counts["HIT_EC"] += 1
            handle.write(f">HIT_EC::{digest[:20]}\n{sequence}\n")
    summary = {
        "phase": "31S1_PREPARE", "status": "PASS", "unique_training_sequences": len(seen),
        "unique_first_seen_by_source": counts, "truth_columns_read": [],
    }
    (WORK / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def evaluate() -> None:
    cohort = pd.read_parquet(R31 / "rcsb_strict_blind_cohort.parquet", columns=["query_id"])
    columns = [
        "query", "target", "fident", "alnlen", "qstart", "qend", "qlen",
        "tstart", "tend", "tlen", "qcov", "tcov", "evalue", "bits",
    ]
    hits = pd.read_csv(WORK / "phase31_vs_tool_training.tsv", sep="\t", names=columns)
    for column in columns[2:]:
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    hits["strict_tool_training_homology"] = (
        hits["fident"].gt(0.30) & hits["qcov"].ge(0.70) & hits["tcov"].ge(0.70)
    )
    nearest = hits.sort_values(
        ["query", "strict_tool_training_homology", "fident", "qcov", "tcov", "bits"],
        ascending=[True, False, False, False, False, False], kind="mergesort",
    ).drop_duplicates("query")
    nearest = nearest.rename(columns={
        "query": "query_id", "target": "nearest_tool_training_sequence",
        "fident": "nearest_tool_training_identity", "qcov": "nearest_tool_training_query_coverage",
        "tcov": "nearest_tool_training_target_coverage", "bits": "nearest_tool_training_bitscore",
        "evalue": "nearest_tool_training_evalue",
    })
    screen = cohort.merge(nearest[[
        "query_id", "nearest_tool_training_sequence", "nearest_tool_training_identity",
        "nearest_tool_training_query_coverage", "nearest_tool_training_target_coverage",
        "nearest_tool_training_bitscore", "nearest_tool_training_evalue", "strict_tool_training_homology",
    ]], on="query_id", how="left", validate="one_to_one")
    screen["strict_tool_training_homology"] = screen["strict_tool_training_homology"].fillna(False).astype(bool)
    screen["tool_training_homology_independent"] = ~screen["strict_tool_training_homology"]
    screen.to_parquet(R31 / "tool_training_homology_sensitivity_screen.parquet", index=False, compression="zstd")
    summary = {
        "phase": "31S1_FILTER", "status": "PASS", "queries": len(screen),
        "excluded_gt30_bidirectional70": int(screen["strict_tool_training_homology"].sum()),
        "strict_sensitivity_queries": int(screen["tool_training_homology_independent"].sum()),
        "alignment_rows": len(hits), "truth_columns_read": [],
    }
    (WORK / "filter_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (CHECKPOINTS / "CHECKPOINT_31S1_TOOL_TRAINING_HOMOLOGY_PASS").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "evaluate"], required=True)
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
