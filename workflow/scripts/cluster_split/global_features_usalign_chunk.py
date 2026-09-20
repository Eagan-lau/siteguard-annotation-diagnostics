#!/usr/bin/env python3
"""Run a chunk of deterministic US-align calibration pairs."""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "usalign_pair_index", "query_protein_id", "reference_protein_id", "tm_query",
    "tm_reference", "rmsd", "identity_query", "identity_reference", "identity_aligned",
    "query_length", "reference_length", "aligned_length",
]


def align_one(item: tuple, executable: str) -> dict[str, object]:
    index, query_id, reference_id, query_pdb, reference_pdb = item
    result = subprocess.run(
        [executable, str(query_pdb), str(reference_pdb), "-outfmt", "2"],
        check=True, capture_output=True, text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line and not line.startswith("#")]
    if len(lines) != 1:
        raise RuntimeError(f"Unexpected US-align output for pair {index}: {result.stdout[-1000:]}")
    fields = lines[0].split()
    if len(fields) != 11:
        raise RuntimeError(f"Unexpected US-align field count for pair {index}: {fields}")
    values = fields[2:]
    return {
        "usalign_pair_index": int(index), "query_protein_id": str(query_id),
        "reference_protein_id": str(reference_id), "tm_query": float(values[0]),
        "tm_reference": float(values[1]), "rmsd": float(values[2]),
        "identity_query": float(values[3]), "identity_reference": float(values[4]),
        "identity_aligned": float(values[5]), "query_length": int(values[6]),
        "reference_length": int(values[7]), "aligned_length": int(values[8]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--usalign", default="USalign")
    args = parser.parse_args()
    work = args.project_root.resolve() / "data/interim/phase06/usalign"
    source = pd.read_csv(work / f"usalign_pairs_chunk_{args.chunk}.tsv", sep="\t")
    items = list(source.itertuples(index=False, name=None))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda item: align_one(item, args.usalign), items))
    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS).sort_values("usalign_pair_index")
    path = work / f"usalign_results_chunk_{args.chunk}.tsv"
    temporary = path.with_suffix(".tsv.tmp")
    output.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)
    if len(output) != len(source):
        raise RuntimeError(f"US-align chunk row mismatch: {len(output)}/{len(source)}")
    print({"chunk": args.chunk, "pairs": len(output)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
