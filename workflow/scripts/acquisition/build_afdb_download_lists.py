#!/usr/bin/env python3
"""Build AlphaFold API supplement and selected-PAE lists from frozen catalogs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def read_ids(path: Path) -> set[str]:
    accessions: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        for token in re.split(r"[,;\s]+", line.strip().upper()):
            if not token:
                continue
            if not re.fullmatch(r"[A-Z0-9]+(?:-\d+)?", token):
                raise ValueError(f"invalid UniProt accession token {token!r} in {path}")
            accessions.add(token)
    return accessions


def write_ids(path: Path, values: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in sorted(values)), encoding="utf-8")


def write_sources(path: Path, rows: dict[str, set[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["accession", "selection_reasons"], delimiter="\t")
        writer.writeheader()
        for accession in sorted(rows):
            writer.writerow({"accession": accession, "selection_reasons": ";".join(sorted(rows[accession]))})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--long-protein-threshold", type=int, default=1000)
    args = parser.parse_args()
    lists = args.root / "data/derived_download_lists"
    catalog = pd.read_parquet(args.root / "data/staging/uniprot/current_swissprot_download_catalog.parquet")
    index = pd.read_parquet(
        args.root / "data/raw/alphafold/bulk_swissprot_v6/afdb_bulk_file_index.parquet",
        columns=["uniprot_accession"],
    )
    bulk = set(index["uniprot_accession"].astype(str).str.upper())
    current = read_ids(lists / "current_complete_ec_or_rhea_accessions.txt")
    mcsa = read_ids(lists / "mcsa_reference_uniprot_ids.txt")
    p450 = read_ids(lists / "p450_uniprot_accessions.txt")
    long_current = set(
        catalog.loc[
            catalog["accession"].astype(str).str.upper().isin(current)
            & (catalog["sequence_length"].fillna(0).astype(int) > args.long_protein_threshold),
            "accession",
        ]
        .astype(str)
        .str.upper()
    )

    pae_sources: dict[str, set[str]] = defaultdict(set)
    for accession in mcsa:
        pae_sources[accession].add("MCSA_REFERENCE")
    for accession in p450:
        pae_sources[accession].add("EXTERNAL_CYP450")
    for accession in long_current:
        pae_sources[accession].add(f"CURRENT_SWISSPROT_ENZYME_LENGTH_GT_{args.long_protein_threshold}")
    pae = set(pae_sources)

    supplement_sources: dict[str, set[str]] = defaultdict(set)
    for accession in current - bulk:
        supplement_sources[accession].add("CURRENT_ENZYME_MISSING_FROM_AFDB_V6_BULK")
    for accession in (mcsa | p450) - bulk:
        supplement_sources[accession].add("SELECTED_VALIDATION_ACCESSION_MISSING_FROM_AFDB_V6_BULK")
    supplement = set(supplement_sources)
    write_ids(lists / "afdb_api_supplement_accessions.txt", supplement)
    write_sources(lists / "afdb_api_supplement_accessions.tsv", supplement_sources)
    write_ids(lists / "afdb_pae_accessions.txt", pae)
    write_sources(lists / "afdb_pae_accessions.tsv", pae_sources)
    write_ids(lists / "afdb_long_enzyme_accessions.txt", long_current)
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "afdb_bulk_accessions": len(bulk),
        "current_swissprot_enzyme_accessions": len(current),
        "current_enzymes_in_bulk": len(current & bulk),
        "current_enzymes_missing_from_bulk": len(current - bulk),
        "mcsa_accessions": len(mcsa),
        "p450_accessions": len(p450),
        "long_current_enzymes": len(long_current),
        "api_supplement_accessions": len(supplement),
        "selected_pae_accessions": len(pae),
        "status": "PASS" if current and bulk and pae else "FAIL",
    }
    output = args.root / "data/manifests/afdb_download_lists_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
