#!/usr/bin/env python3
"""Union selected PDB identifiers with explicit source provenance."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path


def read_ids(path: Path) -> set[str]:
    return {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.root / "data/derived_download_lists"
    inputs = {
        "MCSA_REFERENCE": output_dir / "mcsa_reference_pdb_ids.txt",
        "CURRENT_SWISSPROT_ENZYME": output_dir / "swissprot_enzyme_pdb_ids.txt",
        "CYP450": output_dir / "p450_pdb_ids.txt",
    }
    sources: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = {}
    for source, path in inputs.items():
        values = read_ids(path)
        counts[source] = len(values)
        for pdb_id in values:
            sources[pdb_id].add(source)
    invalid = sorted(value for value in sources if len(value) != 4 or not value.isalnum())
    if invalid:
        raise SystemExit(f"invalid PDB IDs: {invalid[:20]}")
    values = sorted(sources)
    (output_dir / "all_selected_pdb_ids.txt").write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    with (output_dir / "all_selected_pdb_ids.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdb_id", "selection_reasons"], delimiter="\t")
        writer.writeheader()
        for value in values:
            writer.writerow({"pdb_id": value, "selection_reasons": ";".join(sorted(sources[value]))})
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_counts": counts,
        "all_selected_pdb_ids": len(values),
        "status": "PASS" if values else "FAIL",
    }
    path = args.root / "data/manifests/all_selected_pdb_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0 if values else 1


if __name__ == "__main__":
    raise SystemExit(main())
