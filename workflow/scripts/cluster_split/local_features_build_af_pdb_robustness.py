#!/usr/bin/env python3
"""Build an AlphaFold-versus-PDB local-descriptor robustness table.

Only reference-side, curated M-CSA residue mappings are used.  The resulting
table is an analysis input/uncertainty control and never exposes query labels
to a model feature matrix.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pandas as pd

from local_features_local_structure_chunk import RADII, StructureStore, cosine, descriptor


def parse_pdb_chain(path: Path, chain_name: str) -> tuple[dict[int, dict[str, Any]], str]:
    try:
        content = path.read_bytes()
        if content.startswith(b"\x1f\x8b"):
            content = gzip.decompress(content)
        document = gemmi.cif.read_string(content.decode("utf-8"))
        structure = gemmi.make_structure_from_block(document.sole_block())
        if not structure:
            return {}, "EMPTY_STRUCTURE"
        chains = [chain for chain in structure[0] if chain.name == chain_name]
        if not chains:
            chains = [chain for chain in structure[0] if chain.name.upper() == chain_name.upper()]
        if not chains:
            return {}, "CHAIN_UNAVAILABLE"
        residues: dict[int, dict[str, Any]] = {}
        for residue in chains[0]:
            atoms = {
                atom.name.strip(): np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32)
                for atom in residue
            }
            if "CA" not in atoms:
                continue
            residues[int(residue.seqid.num)] = {
                "aa": gemmi.find_tabulated_residue(residue.name).one_letter_code,
                "atoms": atoms,
                "plddt": float("nan"),
            }
        return residues, "PASS" if residues else "NO_CA_RESIDUES"
    except Exception as exc:  # noqa: BLE001 - failure is retained explicitly
        return {}, f"{type(exc).__name__}:{exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    source = args.source_root.resolve()

    sites = pd.read_parquet(root / "data/interim/phase07/reference_sites_selected.parquet")
    sites = sites.loc[
        sites["site_source"].eq("M-CSA")
        & sites["pdb_id"].notna()
        & sites["pdb_chain"].notna()
        & sites["pdb_residue_number"].notna()
        & sites["residue_number"].notna()
    ].drop_duplicates("catalytic_site_id").copy()
    sites["pdb_id"] = sites["pdb_id"].astype(str).str.lower()
    sites["pdb_chain"] = sites["pdb_chain"].astype(str)
    sites["pdb_residue_number"] = sites["pdb_residue_number"].astype(int)
    sites["residue_number"] = sites["residue_number"].astype(int)

    af_store = StructureStore(source, set(sites["protein_id"].astype(str)), cache_size=1024)
    pdb_root = source / "data/raw/pdb/archive_mmcif"
    rows: list[dict[str, Any]] = []
    for group_index, ((pdb_id, chain_name), frame) in enumerate(
        sites.groupby(["pdb_id", "pdb_chain"], sort=True), start=1
    ):
        pdb_path = pdb_root / f"{pdb_id}.cif.gz"
        pdb_residues, pdb_status = (
            parse_pdb_chain(pdb_path, chain_name) if pdb_path.is_file() else ({}, "PDB_FILE_UNAVAILABLE")
        )
        for site in frame.itertuples(index=False):
            af_residues, af_status = af_store.get(str(site.protein_id))
            row: dict[str, Any] = {
                "reference_site_id": site.catalytic_site_id,
                "reference_protein_id": site.protein_id,
                "reference_activity_id": site.activity_id,
                "site_source": site.site_source,
                "site_confidence": site.site_confidence,
                "pdb_id": pdb_id,
                "pdb_chain": chain_name,
                "pdb_residue_number": int(site.pdb_residue_number),
                "uniprot_residue_number": int(site.residue_number),
                "expected_residue": site.residue_type,
                "af_structure_status": af_status,
                "pdb_structure_status": pdb_status,
                "af_site_available": int(site.residue_number) in af_residues,
                "pdb_site_available": int(site.pdb_residue_number) in pdb_residues,
                "feature_provenance": "FROZEN_AFDB_V6_AND_PDB_2026-08-18;REFERENCE_SIDE_ONLY;NO_QUERY_TRUTH",
            }
            all_valid = True
            for radius in RADII:
                af_value = descriptor(af_residues, int(site.residue_number), radius)
                pdb_value = descriptor(pdb_residues, int(site.pdb_residue_number), radius)
                prefix = f"r{int(radius)}"
                if af_value is None or pdb_value is None:
                    all_valid = False
                    for name in [
                        "aa_composition_cosine", "property_cosine", "radial_cosine",
                        "octant_cosine", "sidechain_direction_cosine", "residue_count_ratio",
                        "af_residue_count", "pdb_residue_count", "af_center_plddt",
                    ]:
                        row[f"af_pdb_{name}_{prefix}"] = float("nan")
                    continue
                row[f"af_pdb_aa_composition_cosine_{prefix}"] = cosine(af_value["aa"], pdb_value["aa"])
                row[f"af_pdb_property_cosine_{prefix}"] = cosine(af_value["properties"], pdb_value["properties"])
                row[f"af_pdb_radial_cosine_{prefix}"] = cosine(af_value["radial"], pdb_value["radial"])
                row[f"af_pdb_octant_cosine_{prefix}"] = cosine(af_value["octants"], pdb_value["octants"])
                row[f"af_pdb_sidechain_direction_cosine_{prefix}"] = cosine(af_value["sidechain"], pdb_value["sidechain"])
                row[f"af_pdb_residue_count_ratio_{prefix}"] = min(af_value["residue_count"], pdb_value["residue_count"]) / max(af_value["residue_count"], pdb_value["residue_count"], 1)
                row[f"af_pdb_af_residue_count_{prefix}"] = af_value["residue_count"]
                row[f"af_pdb_pdb_residue_count_{prefix}"] = pdb_value["residue_count"]
                row[f"af_pdb_af_center_plddt_{prefix}"] = af_value["center_plddt"]
            row["descriptor_status"] = "PASS" if all_valid else "SITE_OR_DESCRIPTOR_UNAVAILABLE"
            rows.append(row)
        if group_index % 100 == 0:
            print(json.dumps({"pdb_chains": group_index, "site_rows": len(rows)}), flush=True)

    output = pd.DataFrame(rows)
    out_path = root / "data/processed/af_pdb_robustness_input.parquet"
    output.to_parquet(out_path, index=False, compression="zstd")
    summary = {
        "status": "PASS" if len(output) else "FAIL",
        "site_rows": int(len(output)),
        "unique_sites": int(output["reference_site_id"].nunique()),
        "proteins": int(output["reference_protein_id"].nunique()),
        "pdb_entries": int(output["pdb_id"].nunique()),
        "descriptor_pass": int(output["descriptor_status"].eq("PASS").sum()),
        "af_site_available": int(output["af_site_available"].sum()),
        "pdb_site_available": int(output["pdb_site_available"].sum()),
        "query_ground_truth_used": False,
    }
    (root / "reports/phase07_af_pdb_robustness_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
