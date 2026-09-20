#!/usr/bin/env python3
"""Map frozen AlphaFold DB entries to Phase 4 split-specific Foldseek DB keys."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


AF_NAME = re.compile(r"AF-([A-Z0-9]+)-F\d+-model_v\d+", re.IGNORECASE)


def accession(value: str) -> str | None:
    match = AF_NAME.search(str(value))
    return match.group(1).upper() if match else None


def write_keys(values: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(f"{int(value)}\n" for value in values), encoding="ascii")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--lookup", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data" / "interim" / "phase04"
    membership = pd.read_parquet(work / "afdb_membership.parquet")
    lookup = pd.read_csv(args.lookup, sep="\t", names=["db_key", "structure_name", "file_number"], dtype={"db_key": "int64", "structure_name": str, "file_number": "int64"})
    lookup["protein_id"] = lookup["structure_name"].map(accession)
    lookup = lookup[lookup["protein_id"].notna()].copy()
    allowed = membership[membership["has_structure"]][["protein_id", "split", "cluster_id_30"]]
    mapped = lookup.merge(allowed, on="protein_id", how="inner", validate="many_to_one")
    if mapped["protein_id"].nunique() != len(allowed):
        missing = sorted(set(allowed["protein_id"]) - set(mapped["protein_id"]))
        raise RuntimeError(f"Foldseek lookup missing indexed proteins: {len(missing)}; first={missing[:10]}")
    mapped = mapped.sort_values(["protein_id", "db_key"]).reset_index(drop=True)
    mapped.to_parquet(work / "foldseek_name_map.parquet", index=False, compression="zstd")
    write_keys(mapped.loc[mapped["split"] == "train", "db_key"], work / "foldseek_reference.keys")
    write_keys(mapped.loc[mapped["split"].isin(["validation", "test"]), "db_key"], work / "foldseek_query.keys")
    write_keys(mapped["db_key"], work / "foldseek_benchmark.keys")
    summary = {
        "full_lookup_entries": len(lookup),
        "benchmark_structure_entries": len(mapped),
        "benchmark_structure_proteins": mapped["protein_id"].nunique(),
        "reference_structure_entries": int((mapped["split"] == "train").sum()),
        "query_structure_entries": int(mapped["split"].isin(["validation", "test"]).sum()),
        "split_structure_proteins": mapped.groupby("split")["protein_id"].nunique().to_dict(),
    }
    (work / "foldseek_keys_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
