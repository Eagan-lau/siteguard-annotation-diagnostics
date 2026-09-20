#!/usr/bin/env python3
"""Merge site-level structural descriptors into Local-2 and Local-3 pair tables."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


ROW_KEY = ["pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id"]
SITE_KEY = ROW_KEY + ["reference_site_id"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=8)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data/interim/phase07"
    processed = root / "data/processed"
    reports = root / "reports"
    frames = [pd.read_parquet(work / f"local_structure_site_chunk_{chunk}.parquet") for chunk in range(args.chunks)]
    structure = pd.concat(frames, ignore_index=True)
    if structure.duplicated(SITE_KEY).any():
        raise RuntimeError("Structural site descriptors are not site-row unique")
    mapping = pd.read_parquet(processed / "site_mapping.parquet")
    enriched = mapping.merge(structure, on=SITE_KEY, how="left", validate="one_to_one")
    enriched.to_parquet(processed / "site_mapping.parquet", index=False, compression="zstd")
    pair_rows = pd.read_parquet(work / "site_pair_rows.parquet")[ROW_KEY + ["query_split_expected"]]
    descriptor_rows = enriched.loc[enriched["local_descriptor_status"].eq("PASS")].copy()

    l2_columns = [name for name in structure.columns if name.startswith(("aa_composition_cosine_", "property_cosine_", "residue_count_ratio_"))]
    l3_columns = [name for name in structure.columns if name.startswith(("radial_cosine_", "octant_cosine_", "sidechain_direction_cosine_", "query_local_plddt_", "reference_local_plddt_", "query_center_plddt_", "reference_center_plddt_"))]
    if descriptor_rows.empty:
        raise RuntimeError("No successful structural local descriptors")
    grouped = descriptor_rows.groupby(ROW_KEY, sort=False)
    l2 = grouped[l2_columns].mean().reset_index()
    l2["local2_descriptor_site_count"] = grouped.size().to_numpy()
    l2["feature_provenance"] = "QUERY_REFERENCE_DERIVED_AFDB_LOCAL_COMPOSITION;NO_QUERY_TRUTH"
    l3 = grouped[l3_columns].mean().reset_index()
    l3["local3_descriptor_site_count"] = grouped.size().to_numpy()
    l3["feature_provenance"] = "QUERY_REFERENCE_DERIVED_AFDB_COARSE_GEOMETRY;NO_QUERY_TRUTH"
    base = pair_rows.rename(columns={"query_split_expected": "query_split"})
    l2 = base.merge(l2, on=ROW_KEY, how="left", validate="one_to_one")
    l3 = base.merge(l3, on=ROW_KEY, how="left", validate="one_to_one")
    l2.to_parquet(processed / "local_features_L2.parquet", index=False, compression="zstd")
    l3.to_parquet(processed / "local_features_L3.parquet", index=False, compression="zstd")

    qc = (
        enriched.groupby(["query_split", "reference_site_source", "mapping_status", "local_descriptor_status"], dropna=False)
        .size().rename("site_mappings").reset_index()
    )
    totals = qc.groupby(["query_split", "reference_site_source"])["site_mappings"].transform("sum")
    qc["fraction"] = qc["site_mappings"] / totals
    qc.to_csv(reports / "site_mapping_qc.tsv", sep="\t", index=False)
    summary = {
        "phase": 7, "stage": "local_structure_L2_L3", "status": "PASS",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "structural_site_rows": len(structure),
        "descriptor_pass_site_rows": int((enriched["local_descriptor_status"] == "PASS").sum()),
        "L2_pair_rows": len(l2), "L3_pair_rows": len(l3),
        "pairs_with_L2": int(l2["local2_descriptor_site_count"].notna().sum()),
        "pairs_with_L3": int(l3["local3_descriptor_site_count"].notna().sum()),
        "query_ground_truth_used": False,
        "checkpoint_07_status": "PENDING_RESIDUE_ESM2_AND_AF_PDB_ROBUSTNESS",
    }
    (reports / "phase07_local_structure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
