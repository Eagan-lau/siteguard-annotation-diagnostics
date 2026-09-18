#!/usr/bin/env python3
"""Prepare deterministic Foldseek train-query chunks after the Phase 4 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=6)
    args = parser.parse_args()
    root = args.project_root.resolve()
    checkpoint = root / "checkpoints" / "CHECKPOINT_04_PASS"
    if not checkpoint.is_file():
        raise RuntimeError("CHECKPOINT_04_PASS is required before Phase 5 pair retrieval")
    source = root / "data" / "interim" / "phase04" / "foldseek_reference.keys"
    keys = [line.strip() for line in source.read_text(encoding="ascii").splitlines() if line.strip()]
    if len(keys) != len(set(keys)) or not keys:
        raise RuntimeError("Foldseek train/reference keys are empty or non-unique")
    work = root / "data" / "interim" / "phase05"
    work.mkdir(parents=True, exist_ok=True)
    sizes: list[int] = []
    observed: set[str] = set()
    for index in range(args.chunks):
        start = len(keys) * index // args.chunks
        end = len(keys) * (index + 1) // args.chunks
        subset = keys[start:end]
        if observed.intersection(subset):
            raise RuntimeError("Foldseek train chunks overlap")
        observed.update(subset)
        (work / f"foldseek_train_query_chunk_{index}.keys").write_text(
            "".join(f"{key}\n" for key in subset), encoding="ascii"
        )
        sizes.append(len(subset))
    summary = {
        "status": "PASS" if observed == set(keys) else "FAIL",
        "source": str(source),
        "total_train_structure_keys": len(keys),
        "chunks": args.chunks,
        "chunk_sizes": sizes,
        "disjoint_and_complete": observed == set(keys),
        "checkpoint_04_verified": True,
    }
    (work / "train_retrieval_chunks_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "PASS":
        raise RuntimeError("Train retrieval chunk QC failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
