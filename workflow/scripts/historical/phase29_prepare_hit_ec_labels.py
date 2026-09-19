#!/usr/bin/env python3
"""Export the official HIT-EC EC label order without touching evaluation truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path


EXPECTED_LEVEL_SIZES = [7, 72, 268, 4255]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hit-ec-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.hit_ec_root / "utils" / "label_encoder.pkl"
    with source.open("rb") as handle:
        encoder = pickle.load(handle)
    level4 = [str(item) for item in encoder.classes_]
    levels = [sorted({".".join(ec.split(".")[:n]) for ec in level4}) for n in range(1, 5)]
    observed = [len(labels) for labels in levels]
    if observed != EXPECTED_LEVEL_SIZES:
        raise RuntimeError(f"unexpected HIT-EC label dimensions: {observed}")

    args.output.mkdir(parents=True, exist_ok=True)
    for level, labels in enumerate(levels, start=1):
        (args.output / f"level{level}_labels.txt").write_text(
            "\n".join(labels) + "\n", encoding="utf-8"
        )
    manifest = {
        "source": str(source),
        "source_sha256": sha256(source),
        "level_sizes": observed,
        "derivation": "lexicographically sorted unique dot-prefixes of official level-4 LabelEncoder classes",
        "truth_inputs_used": False,
    }
    (args.output / "label_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
