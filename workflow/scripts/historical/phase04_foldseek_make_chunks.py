#!/usr/bin/env python3
"""Split Foldseek query database keys into deterministic contiguous chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=3)
    args = parser.parse_args()
    keys = [line.strip() for line in args.keys.read_text(encoding="ascii").splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sizes = []
    for index in range(args.chunks):
        start = len(keys) * index // args.chunks
        end = len(keys) * (index + 1) // args.chunks
        subset = keys[start:end]
        (args.output_dir / f"foldseek_query_chunk_{index}.keys").write_text(
            "".join(f"{key}\n" for key in subset), encoding="ascii"
        )
        sizes.append(len(subset))
    result = {"total_keys": len(keys), "chunks": args.chunks, "chunk_sizes": sizes, "disjoint": sum(sizes) == len(keys)}
    (args.output_dir / "foldseek_query_chunks_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
