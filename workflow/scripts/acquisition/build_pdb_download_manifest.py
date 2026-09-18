#!/usr/bin/env python3
"""Build a dynamic archive/assembly/SIFTS download manifest from PDB IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="snapshot_2026-08-18")
    parser.add_argument("--include-assembly1", action="store_true")
    args = parser.parse_args()
    ids = sorted(
        {
            raw.strip().lower()
            for raw in args.ids.read_text(encoding="utf-8").splitlines()
            if raw.strip() and not raw.startswith("#")
        }
    )
    invalid = [value for value in ids if len(value) != 4 or not value.isalnum()]
    if invalid:
        raise SystemExit(f"invalid PDB IDs: {invalid[:20]}")
    resources: list[dict[str, object]] = []
    for pdb_id in ids:
        resources.append(
            {
                "id": f"pdb_archive::{pdb_id}",
                "database": "pdb",
                "version": args.version,
                "url": f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz",
                "relative_path": f"pdb/archive_mmcif/{pdb_id}.cif.gz",
                "required": True,
            }
        )
        if args.include_assembly1:
            resources.append(
                {
                    "id": f"pdb_assembly1::{pdb_id}",
                    "database": "pdb",
                    "version": args.version,
                    "url": f"https://files.rcsb.org/download/{pdb_id.upper()}-assembly1.cif.gz",
                    "relative_path": f"pdb/assembly1_mmcif/{pdb_id}-assembly1.cif.gz",
                    "required": False,
                }
            )
        middle = pdb_id[1:3]
        resources.append(
            {
                "id": f"sifts_xml::{pdb_id}",
                "database": "sifts",
                "version": args.version,
                "url": f"https://ftp.ebi.ac.uk/pub/databases/msd/sifts/split_xml/{middle}/{pdb_id}.xml.gz",
                "relative_path": f"sifts/selected_xml/{middle}/{pdb_id}.xml.gz",
                "required": True,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"groups": [], "singletons": resources}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pdb_ids": len(ids), "resources": len(resources)}))
    return 0 if ids else 1


if __name__ == "__main__":
    raise SystemExit(main())

