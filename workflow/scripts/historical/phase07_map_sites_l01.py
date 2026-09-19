#!/usr/bin/env python3
"""Project exact reference-activity sites and compute leakage-safe Local-0/Local-1."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROW_KEY = ["pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id"]
AA_CLASS = {
    **{aa: "HYDROPHOBIC" for aa in "AILMFWVY"},
    **{aa: "POLAR" for aa in "STNQCGP"},
    **{aa: "POSITIVE" for aa in "KRH"},
    **{aa: "NEGATIVE" for aa in "DE"},
}
ROLE_RESIDUES = {
    "ACID_BASE": set("DEKRHSTYC"),
    "NUCLEOPHILE": set("STCYKDE"),
    "METAL_LIGAND": set("DECH"),
    "ELECTROSTATIC_STABILIZER": set("KRHDE"),
    "SUBSTRATE_POSITIONING": set("STNQKRHFWY"),
}


def role_class(site: dict[str, Any]) -> str:
    text = " ".join(str(site.get(name, "")) for name in [
        "catalytic_role", "roles_json", "functional_location", "metal_cofactor_role",
        "uniprot_feature_type",
    ]).lower()
    if any(token in text for token in ["metal", "ligand", "coordination", "cofactor"]):
        return "METAL_LIGAND"
    if any(token in text for token in ["nucleophil", "covalent", "electron donor"]):
        return "NUCLEOPHILE"
    if any(token in text for token in ["acid", "base", "proton", "deproton"]):
        return "ACID_BASE"
    if any(token in text for token in ["electrostatic", "charge stabil", "transition state"]):
        return "ELECTROSTATIC_STABILIZER"
    if any(token in text for token in ["position", "substrate", "binding", "orient"]):
        return "SUBSTRATE_POSITIONING"
    return "UNRESOLVED"


def map_targets(
    query_start: int, reference_start: int, query_alignment: str,
    reference_alignment: str, targets: set[int],
) -> dict[int, tuple[int, str, str, str]]:
    if not query_alignment or len(query_alignment) != len(reference_alignment):
        return {position: (0, "", "", "NO_VALID_BACKTRACE") for position in targets}
    query_position = int(query_start) - 1
    reference_position = int(reference_start) - 1
    result: dict[int, tuple[int, str, str, str]] = {}
    for query_aa, reference_aa in zip(query_alignment, reference_alignment, strict=True):
        if query_aa != "-":
            query_position += 1
        if reference_aa != "-":
            reference_position += 1
        if reference_aa != "-" and reference_position in targets:
            result[reference_position] = (
                (0, "", reference_aa, "REFERENCE_SITE_ALIGNS_TO_QUERY_GAP")
                if query_aa == "-" else
                (query_position, query_aa, reference_aa, "PASS")
            )
    for position in targets - set(result):
        result[position] = (0, "", "", "REFERENCE_SITE_OUTSIDE_ALIGNMENT")
    return result


def safe_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if not (root / "checkpoints/CHECKPOINT_06_PASS").is_file():
        raise RuntimeError("CHECKPOINT_06_PASS is required before Phase 7 mapping")
    work = root / "data/interim/phase07"
    processed = root / "data/processed"
    reports = root / "reports"
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    pair_rows = pd.read_parquet(work / "site_pair_rows.parquet")
    sites = pd.read_parquet(work / "reference_sites_selected.parquet")
    alignments = pd.read_parquet(work / "site_pair_alignments.parquet")
    if pair_rows.duplicated(ROW_KEY).any():
        raise RuntimeError("Phase 7 activity pair rows are not row-key unique")
    sites_by_activity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sites.to_dict("records"):
        sites_by_activity[str(row["activity_id"])].append(row)
    pair_rows_by_protein: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows.to_dict("records"):
        pair_rows_by_protein[(str(row["query_protein_id"]), str(row["reference_protein_id"]))].append(row)

    mapping_rows: list[dict[str, Any]] = []
    l0_rows: list[dict[str, Any]] = []
    l1_rows: list[dict[str, Any]] = []
    observed_keys: set[tuple[str, str]] = set()
    for index, alignment in enumerate(alignments.itertuples(index=False), start=1):
        protein_key = (str(alignment.query_protein_id), str(alignment.reference_protein_id))
        observed_keys.add(protein_key)
        activity_rows = pair_rows_by_protein.get(protein_key, [])
        target_positions = {
            int(site["residue_number"])
            for pair in activity_rows
            for site in sites_by_activity.get(str(pair["reference_activity_id"]), [])
            if int(site["residue_number"]) > 0
        }
        mapped_positions = map_targets(
            int(alignment.query_start), int(alignment.reference_start),
            str(alignment.query_alignment), str(alignment.reference_alignment), target_positions,
        )
        for pair in activity_rows:
            activity_sites = sites_by_activity.get(str(pair["reference_activity_id"]), [])
            pair_mappings: list[dict[str, Any]] = []
            for site in activity_sites:
                position = int(site["residue_number"])
                query_position, query_aa, reference_aa, status = mapped_positions[position]
                expected_reference_aa = str(site.get("residue_type", "") or "").strip().upper()
                role = role_class(site)
                role_known = role in ROLE_RESIDUES
                role_compatible = bool(status == "PASS" and role_known and query_aa in ROLE_RESIDUES[role])
                chemical_class_match = bool(
                    status == "PASS" and AA_CLASS.get(query_aa) is not None and
                    AA_CLASS.get(query_aa) == AA_CLASS.get(reference_aa)
                )
                mapping = {
                    **{name: pair[name] for name in ROW_KEY},
                    "query_split": pair["query_split_expected"],
                    "reference_site_id": site["catalytic_site_id"],
                    "reference_site_source": site["site_source"],
                    "reference_site_confidence": site["site_confidence"],
                    "reference_site_position": position,
                    "reference_site_residue_expected": expected_reference_aa,
                    "reference_aligned_residue": reference_aa,
                    "mapped_query_position": query_position if query_position else np.nan,
                    "query_aligned_residue": query_aa,
                    "mapping_status": status,
                    "residue_identity_match": bool(status == "PASS" and query_aa == reference_aa),
                    "reference_residue_table_alignment_match": bool(
                        not expected_reference_aa or expected_reference_aa == reference_aa
                    ),
                    "chemical_class_match": chemical_class_match,
                    "role_class": role,
                    "role_resolved": role_known,
                    "role_compatible": role_compatible,
                    "mapping_method": "MMSEQS2_DIRECT_KNOWN_PAIR_GAPPED_BACKTRACE",
                    "feature_provenance": "REFERENCE_SITE_PLUS_QUERY_REFERENCE_ALIGNMENT;NO_QUERY_TRUTH",
                }
                mapping_rows.append(mapping)
                pair_mappings.append(mapping)
            successful = [row for row in pair_mappings if row["mapping_status"] == "PASS"]
            role_rows = [row for row in successful if row["role_resolved"]]
            base = {**{name: pair[name] for name in ROW_KEY}, "query_split": pair["query_split_expected"]}
            l0_rows.append({
                **base, "reference_site_count": len(pair_mappings), "mapped_site_count": len(successful),
                "mapping_success": bool(successful),
                "mapping_success_fraction": len(successful) / len(pair_mappings) if pair_mappings else 0.0,
                "local0_residue_identity": safe_mean([
                    float(row["residue_identity_match"]) for row in successful
                ]),
                "feature_provenance": "QUERY_REFERENCE_DERIVED;MAPPING_SUCCESS_OPERATIONAL_ONLY",
            })
            l1_rows.append({
                **base,
                "local1_chemical_class_match": safe_mean([
                    float(row["chemical_class_match"]) for row in successful
                ]),
                "local1_role_compatibility": safe_mean([
                    float(row["role_compatible"]) for row in role_rows
                ]),
                "local1_role_coverage": len(role_rows) / len(successful) if successful else 0.0,
                "resolved_role_site_count": len(role_rows),
                "feature_provenance": "REFERENCE_ROLE_PLUS_QUERY_MAPPED_RESIDUE;NO_QUERY_TRUTH",
            })
        if index % 10_000 == 0:
            print(json.dumps({"protein_pair_alignments": index, "site_mapping_rows": len(mapping_rows)}), flush=True)

    if observed_keys != set(pair_rows_by_protein):
        raise RuntimeError(f"Alignment/pair-key coverage mismatch: {len(observed_keys)}/{len(pair_rows_by_protein)}")
    mapping = pd.DataFrame(mapping_rows)
    l0 = pd.DataFrame(l0_rows)
    l1 = pd.DataFrame(l1_rows)
    if len(l0) != len(pair_rows) or len(l1) != len(pair_rows):
        raise RuntimeError(f"Local feature row mismatch: L0={len(l0)}, L1={len(l1)}, pairs={len(pair_rows)}")
    mapping.to_parquet(processed / "site_mapping.parquet", index=False, compression="zstd")
    l0.to_parquet(processed / "local_features_L0.parquet", index=False, compression="zstd")
    l1.to_parquet(processed / "local_features_L1.parquet", index=False, compression="zstd")

    qc = (
        mapping.groupby(["query_split", "reference_site_source", "mapping_status"], dropna=False)
        .size().rename("site_mappings").reset_index()
    )
    totals = qc.groupby(["query_split", "reference_site_source"])["site_mappings"].transform("sum")
    qc["fraction"] = qc["site_mappings"] / totals
    qc.to_csv(reports / "site_mapping_qc.tsv", sep="\t", index=False)
    status_counts = mapping["mapping_status"].value_counts().to_dict()
    summary = {
        "phase": 7, "stage": "site_mapping_L0_L1", "status": "PASS",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "activity_pair_rows": len(pair_rows), "site_mapping_rows": len(mapping),
        "mapping_status_counts": status_counts,
        "mapping_success_rate": float((mapping["mapping_status"] == "PASS").mean()),
        "pairs_with_mapping": int(l0["mapping_success"].sum()),
        "role_resolved_site_mappings": int(mapping["role_resolved"].sum()),
        "query_ground_truth_used": False,
        "checkpoint_07_status": "PENDING_L2_L3_AND_AF_PDB_ROBUSTNESS",
    }
    (reports / "phase07_l01_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
