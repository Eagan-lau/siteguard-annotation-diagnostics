#!/usr/bin/env python3
"""Select the one RCSB entity chain per locked Phase 28 query from a Foldseek DB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookup", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    args = parser.parse_args()
    lookup = pd.read_csv(args.lookup, sep="\t", names=["db_key", "structure_name", "file_number"], dtype={"db_key": int, "structure_name": str})
    manifest = pd.read_parquet(args.manifest)
    cohort = pd.read_parquet(args.cohort, columns=["query_id", "representative_asym_ids_json"])
    inventory = set(lookup["structure_name"].astype(str))
    rows = []
    missing = []
    joined = manifest.merge(cohort, left_on="query_protein_id", right_on="query_id", validate="one_to_one")
    for row in joined.itertuples(index=False):
        candidates = [f"{row.structure_token}_{asym}" for asym in json.loads(row.representative_asym_ids_json)]
        available = [value for value in candidates if value in inventory]
        if available:
            rows.append({"query_protein_id": row.query_protein_id, "structure_name": available[0]})
        else:
            missing.append(str(row.query_protein_id))
    selected_names = pd.DataFrame(rows)
    lookup_unique = lookup.sort_values(["structure_name", "db_key"], kind="mergesort").drop_duplicates("structure_name")
    selected = lookup_unique.merge(selected_names, on="structure_name", how="inner", validate="one_to_one")
    if selected.empty or selected["query_protein_id"].duplicated().any():
        raise RuntimeError("No unique Foldseek-readable RCSB entity chains could be selected")
    args.keys.write_text("".join(f"{value}\n" for value in selected["db_key"].astype(int)), encoding="ascii")
    selected.to_parquet(args.keys.with_suffix(".mapping.parquet"), index=False, compression="zstd")
    args.keys.with_suffix(".summary.json").write_text(json.dumps({
        "locked_queries": len(manifest), "foldseek_readable_entity_chains": len(selected),
        "foldseek_missing_queries": len(missing), "missing_query_ids": missing,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
