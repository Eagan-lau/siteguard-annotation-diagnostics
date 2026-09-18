#!/usr/bin/env python3
"""Stream Swiss-Prot XML to build download-stage enzyme/PDB/site lists."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree


# UniProt changed the XML namespace from http to https.  The frozen 2026_02
# release uses this value; matching the legacy namespace causes iterparse to
# retain the entire document because no entry end-event is ever selected.
NS_URI = "https://uniprot.org/uniprot"
NS = {"u": NS_URI}
COMPLETE_EC = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$")
CATALYTIC_FEATURE_TYPES = {"active site", "binding site", "metal ion-binding site"}
SCHEMA = pa.schema(
    [
        ("accession", pa.string()),
        ("secondary_accessions", pa.list_(pa.string())),
        ("entry_name", pa.string()),
        ("entry_version", pa.int32()),
        ("sequence_version", pa.int32()),
        ("sequence_length", pa.int32()),
        ("fragment", pa.bool_()),
        ("complete_ec", pa.list_(pa.string())),
        ("rhea", pa.list_(pa.string())),
        ("pdb_ids", pa.list_(pa.string())),
        ("catalytic_feature_count", pa.int16()),
    ]
)


def integer(value: str | None) -> int:
    return int(value) if value and value.isdigit() else 0


def parse_entry(entry: etree._Element) -> dict[str, Any]:
    accessions = [value.text.strip() for value in entry.findall("u:accession", NS) if value.text]
    if not accessions:
        raise ValueError("UniProt entry has no accession")
    name = entry.findtext("u:name", default="", namespaces=NS)
    sequence_node = entry.find("u:sequence", NS)
    sequence_length = 0
    if sequence_node is not None:
        sequence_length = integer(sequence_node.get("length"))
        if not sequence_length and sequence_node.text:
            sequence_length = sum(not char.isspace() for char in sequence_node.text)
    fragment = False if sequence_node is None else bool(sequence_node.get("fragment"))
    ec_values = sorted(
        {
            node.text.strip()
            for node in entry.findall(".//u:ecNumber", NS)
            if node.text and COMPLETE_EC.match(node.text.strip())
        }
    )
    rhea_values: set[str] = set()
    pdb_ids: set[str] = set()
    for reference in entry.findall(".//u:dbReference", NS):
        ref_type = reference.get("type", "")
        ref_id = reference.get("id", "")
        if ref_type == "Rhea" and ref_id:
            rhea_values.add(ref_id)
        elif ref_type == "PDB" and ref_id:
            pdb_ids.add(ref_id.lower())
    catalytic_count = sum(
        feature.get("type", "").lower() in CATALYTIC_FEATURE_TYPES
        for feature in entry.findall("u:feature", NS)
    )
    return {
        "accession": accessions[0],
        "secondary_accessions": accessions[1:],
        "entry_name": name,
        "entry_version": integer(entry.get("version")),
        "sequence_version": 0 if sequence_node is None else integer(sequence_node.get("version")),
        "sequence_length": sequence_length,
        "fragment": fragment,
        "complete_ec": ec_values,
        "rhea": sorted(rhea_values),
        "pdb_ids": sorted(pdb_ids),
        "catalytic_feature_count": catalytic_count,
    }


def write_values(path: Path, values: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in sorted(values)), encoding="utf-8")


def assert_xml_namespace(path: Path) -> None:
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as handle:
        prefix = handle.read(4096)
    match = re.search(r'<uniprot\b[^>]*\bxmlns="([^"]+)"', prefix)
    if not match:
        raise ValueError("could not determine the UniProt XML namespace")
    if match.group(1) != NS_URI:
        raise ValueError(
            f"unexpected UniProt XML namespace {match.group(1)!r}; expected {NS_URI!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--lists", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    assert_xml_namespace(args.xml)

    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.catalog.with_suffix(args.catalog.suffix + ".tmp")
    writer = pq.ParquetWriter(
        temporary,
        SCHEMA,
        compression="zstd",
        use_dictionary=False,
    )
    entries = 0
    enzyme_accessions: set[str] = set()
    site_accessions: set[str] = set()
    pdb_ids: set[str] = set()
    batch: list[dict[str, Any]] = []
    try:
        with gzip.open(args.xml, "rb") as handle:
            context = etree.iterparse(handle, events=("end",), tag=f"{{{NS_URI}}}entry", huge_tree=True)
            for _, entry in context:
                row = parse_entry(entry)
                entries += 1
                if row["complete_ec"] or row["rhea"]:
                    enzyme_accessions.add(row["accession"])
                    pdb_ids.update(row["pdb_ids"])
                if row["catalytic_feature_count"]:
                    site_accessions.add(row["accession"])
                batch.append(row)
                if len(batch) >= args.batch_size:
                    writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                    batch.clear()
                entry.clear()
                parent = entry.getparent()
                while parent is not None and entry.getprevious() is not None:
                    del parent[0]
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
    finally:
        writer.close()
    temporary.replace(args.catalog)

    write_values(args.lists / "current_swissprot_complete_ec_or_rhea.txt", enzyme_accessions)
    write_values(args.lists / "current_complete_ec_or_rhea_accessions.txt", enzyme_accessions)
    write_values(args.lists / "current_swissprot_catalytic_site_accessions.txt", site_accessions)
    write_values(args.lists / "swissprot_enzyme_pdb_ids.txt", pdb_ids)
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_xml": str(args.xml.resolve()),
        "entries": entries,
        "complete_ec_or_rhea_accessions": len(enzyme_accessions),
        "catalytic_site_accessions": len(site_accessions),
        "unique_pdb_ids": len(pdb_ids),
        "selection": "reviewed Swiss-Prot entries in release 2026_02 with complete numeric EC or explicit Rhea; fragments retained with flag",
        "status": "PASS" if entries > 500_000 and enzyme_accessions else "FAIL",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
