#!/usr/bin/env python3
"""Build source-linked P450 accession, PDB, and PubChem download lists."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


UNIPROT = re.compile(
    r"(?<![A-Z0-9])(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[0-9]+)?(?![A-Z0-9])",
    re.IGNORECASE,
)
PDB = re.compile(r"(?<![A-Z0-9])[0-9][A-Z0-9]{3}(?![A-Z0-9])", re.IGNORECASE)


def add_uniprot(value: object, source: str, rows: dict[str, set[str]]) -> None:
    if pd.isna(value):
        return
    for match in UNIPROT.findall(str(value).upper()):
        rows[match].add(source)


def add_cid(value: object, source: str, rows: dict[int, set[str]]) -> None:
    if pd.isna(value):
        return
    for token in re.findall(r"(?<![0-9])[0-9]+(?![0-9])", str(value)):
        cid = int(token)
        if cid > 0:
            rows[cid].add(source)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin1", low_memory=False)


def sifts_p450_pdbs(path: Path) -> set[str]:
    values: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader((line for line in handle if not line.startswith("#")), delimiter="\t")
        header = next(reader)
        normalized = [name.strip().upper() for name in header]
        pdb_index = normalized.index("PDB") if "PDB" in normalized else 0
        pfam_index = next((i for i, name in enumerate(normalized) if "PFAM" in name or name in {"SP_PRIMARY", "PFAM_ID"}), None)
        for row in reader:
            if not row:
                continue
            joined = "\t".join(row)
            if "PF00067" not in joined:
                continue
            if pfam_index is not None and pfam_index < len(row) and "PF00067" not in row[pfam_index] and "PF00067" not in joined:
                continue
            pdb_id = row[pdb_index].strip().lower()
            if re.fullmatch(r"[0-9][a-z0-9]{3}", pdb_id):
                values.add(pdb_id)
    return values


def write_list(path: Path, values: list[str | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def write_sources(path: Path, key: str, rows: dict[object, set[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key, "sources", "source_count"], delimiter="\t")
        writer.writeheader()
        for value in sorted(rows, key=lambda item: str(item)):
            sources = sorted(rows[value])
            writer.writerow({key: value, "sources": ";".join(sources), "source_count": len(sources)})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    raw = args.root / "data/raw/cyp450"
    lists = args.root / "data/derived_download_lists"
    manifests = args.root / "data/manifests"
    accession_sources: dict[str, set[str]] = defaultdict(set)
    cid_sources: dict[int, set[str]] = defaultdict(set)
    pdb_sources: dict[str, set[str]] = defaultdict(set)

    p450s = read_csv(raw / "p450rdb_v2/2. P450s.CSV")
    reactions = read_csv(raw / "p450rdb_v2/1. Reactions.CSV")
    compounds = read_csv(raw / "p450rdb_v2/3. Compounds.CSV")
    for value in p450s.get("Uniprot ID", pd.Series(dtype=str)):
        add_uniprot(value, "P450Rdb:P450s", accession_sources)
    for value in reactions.get("Uniprot ID", pd.Series(dtype=str)):
        add_uniprot(value, "P450Rdb:Reactions", accession_sources)
    for frame_name, frame in (("P450Rdb:Reactions", reactions), ("P450Rdb:Compounds", compounds)):
        for column in frame.columns:
            if "cid" in column.lower():
                for value in frame[column]:
                    add_cid(value, f"{frame_name}:{column}", cid_sources)

    plant_path = args.root / "data/staging/cyp450/plantp450_entries.parquet"
    if plant_path.is_file():
        plant = pd.read_parquet(plant_path)
        for column in plant.columns:
            lowered = column.lower()
            if "accession" in lowered:
                for value in plant[column]:
                    add_uniprot(value, f"PlantP450:{column}", accession_sources)
            if "cid" in lowered:
                for value in plant[column]:
                    add_cid(value, f"PlantP450:{column}", cid_sources)

    pfam_sifts = args.root / "data/raw/sifts/flatfiles/pdb_chain_pfam.tsv.gz"
    for pdb_id in sifts_p450_pdbs(pfam_sifts):
        pdb_sources[pdb_id].add("SIFTS:PF00067")
    structure_html = raw / "plantp450_erda/structure.html"
    if structure_html.is_file():
        text = structure_html.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"(?:rcsb|pdb)[^\n<>]{0,160}", text, flags=re.IGNORECASE):
            for pdb_id in PDB.findall(match.group(0)):
                pdb_sources[pdb_id.lower()].add("PlantP450:structure_page")

    accessions = sorted(accession_sources)
    pdb_ids = sorted(pdb_sources)
    cids = sorted(cid_sources)
    write_list(lists / "p450_uniprot_accessions.txt", accessions)
    write_sources(lists / "p450_uniprot_accessions.tsv", "accession", accession_sources)
    write_list(lists / "p450_pdb_ids.txt", pdb_ids)
    write_sources(lists / "p450_pdb_ids.tsv", "pdb_id", pdb_sources)
    write_list(lists / "external_pubchem_cids.txt", cids)
    write_sources(lists / "external_pubchem_cids.tsv", "cid", cid_sources)
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "snapshot_date": "2026-08-18",
        "p450_uniprot_accessions": len(accessions),
        "p450_pdb_ids": len(pdb_ids),
        "external_pubchem_cids": len(cids),
        "source_versions": {"P450Rdb": "v2 snapshot 2026-08-18", "PlantP450": "snapshot 2026-08-18", "SIFTS": "snapshot 2026-08-18"},
        "status": "PASS" if accessions and cids and pdb_ids else "FAIL",
    }
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "p450_derived_lists_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
