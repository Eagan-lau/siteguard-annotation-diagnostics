#!/usr/bin/env python3
"""Safely unpack the small/medium frozen archives required after download."""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path


def safe_members(archive: tarfile.TarFile, destination: Path) -> list[tarfile.TarInfo]:
    base = destination.resolve()
    members: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != base and base not in target.parents:
            raise ValueError(f"archive path escapes destination: {member.name}")
        if member.issym() or member.islnk():
            if Path(member.linkname).is_absolute():
                raise ValueError(f"absolute archive link is not allowed: {member.name}")
            link_base = Path(member.name).parent if member.issym() else Path()
            link_target = (destination / link_base / member.linkname).resolve()
            if link_target != base and base not in link_target.parents:
                raise ValueError(f"archive link escapes destination: {member.name}")
        members.append(member)
    return members


def unpack(source: Path, destination: Path) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, "r:*") as archive:
        members = safe_members(archive, destination)
        rows = [
            {
                "member": member.name,
                "size": member.size,
                "type": (
                    "file"
                    if member.isfile()
                    else "directory"
                    if member.isdir()
                    else "symlink"
                    if member.issym()
                    else "hardlink"
                    if member.islnk()
                    else "other"
                ),
            }
            for member in members
        ]
        archive.extractall(destination, members=members)
    inventory = destination.parent / f"{destination.name}_file_inventory.tsv"
    with inventory.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["member", "size", "type"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "source": str(source),
        "destination": str(destination),
        "members": len(rows),
        "file_members": sum(row["type"] == "file" for row in rows),
        "total_member_bytes": sum(int(row["size"]) for row in rows),
        "inventory": str(inventory),
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    raw = args.root / "data" / "raw"
    tasks = [
        (raw / "uniprot/historical_2023_01/knowledgebase/uniprot_sprot-only2023_01.tar.gz", raw / "uniprot/historical_2023_01/extracted"),
        (raw / "uniprot/historical_2026_01/knowledgebase/uniprot_sprot-only2026_01.tar.gz", raw / "uniprot/historical_2026_01/extracted"),
        (raw / "rhea/release_126/126.tar.bz2", raw / "rhea/release_126/extracted"),
        (raw / "rhea/release_140/140.tar.bz2", raw / "rhea/release_140/extracted"),
        (raw / "rhea/release_141/141.tar.bz2", raw / "rhea/release_141/extracted"),
        (raw / "cath/v4_4_0/s40/cath-dataset-nonredundant-S40-v4_4_0.pdb.tgz", raw / "cath/v4_4_0/s40/extracted"),
    ]
    missing = [str(source) for source, _ in tasks if not source.is_file()]
    if missing:
        raise SystemExit(f"required archives missing: {missing}")
    results = [unpack(source, destination) for source, destination in tasks]
    payload = {"archives": results, "status": "PASS"}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archives": len(results), "members": sum(int(row["members"]) for row in results), "status": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
