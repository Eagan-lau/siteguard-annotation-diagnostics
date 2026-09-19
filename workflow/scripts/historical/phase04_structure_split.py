#!/usr/bin/env python3
"""Append independent Foldseek structure-cluster holdout membership to Phase 3 structure splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


AF_NAME = re.compile(r"AF-([A-Z0-9]+)-F\d+-model_v\d+", re.IGNORECASE)


def accession(value: str) -> str:
    match = AF_NAME.search(str(value))
    return match.group(1).upper() if match else str(value).split()[0]


def stable_split(group: str, seed: int) -> str:
    if not group or group == "UNASSIGNED":
        return "UNASSIGNED"
    digest = hashlib.sha256(f"{seed}|foldseek_structure_cluster|{group}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if value < 0.70 else ("validation" if value < 0.85 else "test")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data" / "interim" / "phase04"
    membership = pd.read_parquet(work / "afdb_membership.parquet")
    expected = set(membership.loc[membership["has_structure"], "protein_id"])
    mapping: dict[str, str] = {}
    with args.clusters.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            representative, member = accession(fields[0]), accession(fields[1])
            if member in expected:
                mapping[member] = representative
    for protein_id in expected - set(mapping):
        mapping[protein_id] = protein_id

    structure_path = root / "data" / "splits" / "split_structure.parquet"
    structure = pd.read_parquet(structure_path)
    structure["foldseek_structure_cluster"] = structure["protein_id"].map(mapping).fillna("UNASSIGNED")
    structure["foldseek_structure_cluster_split"] = structure["foldseek_structure_cluster"].map(
        lambda value: stable_split(value, args.seed)
    )
    assigned = structure[structure["foldseek_structure_cluster"] != "UNASSIGNED"]
    overlap = int((assigned.groupby("foldseek_structure_cluster")["foldseek_structure_cluster_split"].nunique() > 1).sum())
    fractions = assigned["foldseek_structure_cluster_split"].value_counts(normalize=True).to_dict()
    ratio_ok = all(abs(fractions.get(name, 0.0) - target) <= 0.05 for name, target in (("train", 0.70), ("validation", 0.15), ("test", 0.15)))
    if overlap or not ratio_ok or assigned["protein_id"].nunique() != len(expected):
        raise RuntimeError(f"Structure split QC failed: overlap={overlap}, ratio_ok={ratio_ok}, assigned={len(assigned)}, expected={len(expected)}")
    temporary = structure_path.with_suffix(".parquet.tmp")
    structure.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(structure_path)
    summary = {
        "status": "PASS",
        "structure_proteins": len(expected),
        "structure_clusters": assigned["foldseek_structure_cluster"].nunique(),
        "unassigned_no_structure": int((structure["foldseek_structure_cluster"] == "UNASSIGNED").sum()),
        "cluster_overlap": overlap,
        "split_counts": assigned["foldseek_structure_cluster_split"].value_counts().to_dict(),
        "split_fractions": fractions,
        "tm_score_threshold": 0.5,
        "bidirectional_coverage_threshold": 0.5,
    }
    (root / "reports" / "phase04_structure_split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
