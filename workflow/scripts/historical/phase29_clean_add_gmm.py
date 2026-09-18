#!/usr/bin/env python3
"""Append official CLEAN GMM ensemble confidence to already locked raw distances."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


def confidence(distance: float, models: list[object]) -> float:
    values = []
    for model in models:
        means = model.means_
        true_component = 0 if means[0][0] < means[1][0] else 1
        values.append(float(model.predict_proba([[distance]])[0][true_component]))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gmm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.gmm.open("rb") as handle:
        models = pickle.load(handle)
    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        for rank in range(1, 11):
            key = f"clean_top{rank}_distance"
            row[f"clean_top{rank}_gmm_confidence"] = confidence(float(row[key]), models)
        row["clean_gmm_margin12"] = (
            float(row["clean_top1_gmm_confidence"]) - float(row["clean_top2_gmm_confidence"])
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CLEAN_GMM_ENRICHED {len(rows)} {args.output}")


if __name__ == "__main__":
    main()
