#!/usr/bin/env python3
"""Prepare unique residue-centred sequence windows for Local-2 ESM2 context."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


FLANK = 64


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data/interim/phase07"
    mapping = pd.read_parquet(root / "data/processed/site_mapping.parquet")
    mapping = mapping.loc[mapping["mapping_status"].eq("PASS")].copy()
    proteins = pd.read_parquet(
        root / "data/interim/phase03/benchmark_proteins.parquet", columns=["protein_id", "sequence"]
    ).set_index("protein_id")["sequence"].to_dict()
    contexts: dict[tuple[str, int], dict[str, object]] = {}
    for row in mapping.itertuples(index=False):
        for protein_id, position, aligned_aa, role in [
            (str(row.query_protein_id), int(row.mapped_query_position), str(row.query_aligned_residue), "query"),
            (str(row.reference_protein_id), int(row.reference_site_position), str(row.reference_aligned_residue), "reference"),
        ]:
            key = (protein_id, position)
            if key in contexts:
                continue
            sequence = proteins.get(protein_id, "")
            if not sequence or position < 1 or position > len(sequence):
                raise RuntimeError(f"Residue context outside frozen sequence: {protein_id}:{position}")
            start = max(1, position - FLANK)
            end = min(len(sequence), position + FLANK)
            window = sequence[start - 1:end]
            center_offset = position - start
            observed = sequence[position - 1]
            context_id = hashlib.sha256(f"{protein_id}|{position}".encode()).hexdigest()[:24]
            contexts[key] = {
                "context_id": context_id, "protein_id": protein_id, "residue_position": position,
                "window_start": start, "window_end": end, "center_offset": center_offset,
                "sequence": window, "sequence_residue": observed,
                "alignment_residue": aligned_aa, "first_role_observed": role,
                "sequence_alignment_residue_match": observed == aligned_aa,
            }
    frame = pd.DataFrame(contexts.values()).sort_values(["protein_id", "residue_position"]).reset_index(drop=True)
    frame["embedding_row"] = range(len(frame))
    if not frame["context_id"].is_unique:
        raise RuntimeError("Residue context identifiers are not unique")
    frame.to_parquet(work / "residue_esm2_contexts.parquet", index=False, compression="zstd")
    frame.to_csv(
        work / "residue_esm2_contexts.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    summary = {
        "status": "PASS", "successful_site_mapping_rows": len(mapping),
        "unique_residue_contexts": len(frame), "proteins": frame["protein_id"].nunique(),
        "flank_residues": FLANK, "maximum_window_length": int(frame["sequence"].str.len().max()),
        "sequence_alignment_residue_match_rate": float(frame["sequence_alignment_residue_match"].mean()),
        "query_ground_truth_used": False,
    }
    (work / "residue_esm2_prepare_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
