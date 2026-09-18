#!/usr/bin/env python3
"""Build deterministic M-CSA UniProt/PDB/EC download lists."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable


def write_values(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = sorted({value.strip() for value in values if value and value.strip()})
    path.write_text("".join(f"{value}\n" for value in normalized), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", type=Path, required=True)
    parser.add_argument("--residues", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    entries: list[dict[str, Any]] = json.loads(args.entries.read_text(encoding="utf-8"))
    residues: list[dict[str, Any]] = json.loads(args.residues.read_text(encoding="utf-8"))
    uniprot_ids: set[str] = set()
    pdb_ids: set[str] = set()
    ec_ids: set[str] = set()
    for entry in entries:
        reference = entry.get("reference_uniprot_id")
        if reference:
            uniprot_ids.add(str(reference))
        ec_ids.update(str(value) for value in entry.get("all_ecs", []) if value)
        for sequence in entry.get("protein", {}).get("sequences", []):
            if sequence.get("uniprot_id") and entry.get("is_reference_uniprot_id"):
                uniprot_ids.add(str(sequence["uniprot_id"]))
    reference_residue_count = 0
    for residue in residues:
        for sequence in residue.get("residue_sequences", []):
            if sequence.get("is_reference") and sequence.get("uniprot_id"):
                uniprot_ids.add(str(sequence["uniprot_id"]))
                reference_residue_count += 1
        for chain in residue.get("residue_chains", []):
            if chain.get("is_reference") and chain.get("pdb_id"):
                pdb_ids.add(str(chain["pdb_id"]).lower())
    write_values(args.output_dir / "mcsa_reference_uniprot_ids.txt", uniprot_ids)
    write_values(args.output_dir / "mcsa_reference_pdb_ids.txt", pdb_ids)
    write_values(args.output_dir / "mcsa_ec_ids.txt", ec_ids)
    rows = [
        {"metric": "entries", "value": len(entries)},
        {"metric": "residues", "value": len(residues)},
        {"metric": "reference_uniprot_ids", "value": len(uniprot_ids)},
        {"metric": "reference_pdb_ids", "value": len(pdb_ids)},
        {"metric": "ec_ids", "value": len(ec_ids)},
        {"metric": "reference_residue_sequence_rows", "value": reference_residue_count},
    ]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "entries_path": str(args.entries),
        "residues_path": str(args.residues),
        "selection": "M-CSA reference entries and residue mappings only; homologous propagated residues excluded",
        "counts": {row["metric"]: row["value"] for row in rows},
    }
    (args.output_dir / "mcsa_lists_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance["counts"]))
    return 0 if entries and residues and uniprot_ids and pdb_ids else 1


if __name__ == "__main__":
    raise SystemExit(main())

