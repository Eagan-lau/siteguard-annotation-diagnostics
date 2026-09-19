#!/usr/bin/env python3
"""Run SiteGuard V4 Phase 7 acceptance checks and write its checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROW_KEY = ["pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id"]
SITE_KEY = ROW_KEY + ["reference_site_id"]
FEATURE_FILES = [f"local_features_L{level}.parquet" for level in range(4)]
FORBIDDEN_EXACT = {
    "query_ec_l3_ground_truth", "query_ec_l4_ground_truth", "query_rhea_ground_truth",
    "same_ec_l3", "same_ec_l4", "same_exact_rhea", "observed_concordance_depth",
    "canonical_rhea", "ec_l3", "ec_l4", "is_difficult_case", "candidate_origin",
    "augmentation_source", "sampling_probability", "sample_weight",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    processed = root / "data/processed"
    reports = root / "reports"

    paths = {
        "site_mapping": processed / "site_mapping.parquet",
        "af_pdb": processed / "af_pdb_robustness_input.parquet",
        **{f"L{level}": processed / f"local_features_L{level}.parquet" for level in range(4)},
    }
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required Phase 7 artifact is absent: {name}={path}")

    mapping = pd.read_parquet(paths["site_mapping"])
    local = {f"L{level}": pd.read_parquet(paths[f"L{level}"]) for level in range(4)}
    robustness = pd.read_parquet(paths["af_pdb"])
    l01_summary = read_json(reports / "phase07_l01_summary.json")
    structure_summary = read_json(reports / "phase07_local_structure_summary.json")
    esm_summary = read_json(reports / "phase07_residue_esm2_summary.json")
    robustness_summary = read_json(reports / "phase07_af_pdb_robustness_summary.json")

    expected_rows = int(l01_summary["activity_pair_rows"])
    forbidden: dict[str, list[str]] = {}
    for level, frame in local.items():
        names = set(frame.columns)
        hits = sorted(
            (names & FORBIDDEN_EXACT)
            | {name for name in names if "ground_truth" in name.lower() or name.startswith("hard_H")}
        )
        if hits:
            forbidden[level] = hits

    split_leakage = pd.read_csv(root / "data/splits/split_leakage_report.tsv", sep="\t")
    mapping_counts = mapping["mapping_status"].value_counts(dropna=False).to_dict()
    expected_mapping_statuses = {
        "PASS", "REFERENCE_SITE_OUTSIDE_ALIGNMENT", "REFERENCE_SITE_ALIGNS_TO_QUERY_GAP"
    }
    pair_key_sets_equal = all(
        set(map(tuple, frame[ROW_KEY].itertuples(index=False, name=None)))
        == set(map(tuple, local["L0"][ROW_KEY].itertuples(index=False, name=None)))
        for frame in local.values()
    )
    expected_structural_columns = {
        "aa_composition_cosine_r6", "property_cosine_r8", "residue_count_ratio_r10",
        "radial_cosine_r6", "octant_cosine_r8", "sidechain_direction_cosine_r10",
    }
    robustness_measurements = [
        name for name in robustness.columns if name.startswith("af_pdb_") and name.endswith(("_r6", "_r8", "_r10"))
    ]
    checks = [
        ("checkpoint_06_present", (root / "checkpoints/CHECKPOINT_06_PASS").is_file(), "strict phase gate"),
        ("site_mapping_nonempty", len(mapping) > 0, str(len(mapping))),
        ("site_mapping_key_unique", not mapping.duplicated(SITE_KEY).any(), str(SITE_KEY)),
        ("site_mapping_row_count_stable", len(mapping) == int(l01_summary["site_mapping_rows"]), f"{len(mapping)}/{l01_summary['site_mapping_rows']}"),
        ("mapping_failures_explicit", set(mapping_counts).issubset(expected_mapping_statuses), str(mapping_counts)),
        ("successful_mapping_exists", int(mapping_counts.get("PASS", 0)) > 0, str(mapping_counts.get("PASS", 0))),
        ("mapped_residue_esm2_complete", mapping.loc[mapping["mapping_status"].eq("PASS"), "local2_residue_esm2_t12_cosine"].notna().all(), str(esm_summary.get("site_rows_with_residue_esm2"))),
        ("all_local_levels_expected_rows", all(len(frame) == expected_rows for frame in local.values()), str({k: len(v) for k, v in local.items()})),
        ("all_local_row_keys_unique", all(not frame.duplicated(ROW_KEY).any() for frame in local.values()), str(ROW_KEY)),
        ("all_local_row_key_sets_equal", pair_key_sets_equal, "L0-L3"),
        ("structural_columns_present", expected_structural_columns.issubset(set(local["L2"].columns) | set(local["L3"].columns)), str(sorted(expected_structural_columns))),
        ("residue_esm2_present_in_L2", "local2_residue_esm2_t12_cosine" in local["L2"], "frozen ESM2-t12 central residue"),
        ("forbidden_query_truth_absent", not forbidden, str(forbidden)),
        ("split_leakage_report_pass", bool((split_leakage["status"] == "PASS").all()), str(root / "data/splits/split_leakage_report.tsv")),
        ("local_structure_stage_pass", structure_summary.get("status") == "PASS", str(structure_summary.get("status"))),
        ("residue_esm2_stage_pass", esm_summary.get("status") == "PASS", str(esm_summary.get("status"))),
        ("af_pdb_stage_pass", robustness_summary.get("status") == "PASS", str(robustness_summary.get("status"))),
        ("af_pdb_site_key_unique", robustness["reference_site_id"].is_unique, str(len(robustness))),
        ("af_pdb_descriptor_rows_exist", int(robustness["descriptor_status"].eq("PASS").sum()) > 0, str(robustness["descriptor_status"].value_counts().to_dict())),
        ("af_pdb_radii_complete", len(robustness_measurements) >= 18, str(sorted(robustness_measurements))),
        ("query_truth_not_used_by_builders", not any(bool(value.get("query_ground_truth_used", False)) for value in [l01_summary, structure_summary, esm_summary, robustness_summary]), "all false"),
    ]
    qc = pd.DataFrame(
        [(name, "PASS" if bool(passed) else "FAIL", details) for name, passed, details in checks],
        columns=["check", "status", "details"],
    )
    qc.to_csv(reports / "phase07_qc.tsv", sep="\t", index=False)
    failures = qc.loc[qc["status"].eq("FAIL"), "check"].tolist()
    status = "PASS" if not failures else "FAIL"

    mapping_pass = int(mapping["mapping_status"].eq("PASS").sum())
    structure_pass = int(mapping["local_descriptor_status"].eq("PASS").sum())
    af_pdb_pass = int(robustness["descriptor_status"].eq("PASS").sum())
    summary = {
        "phase": 7,
        "status": status,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "activity_pair_rows": expected_rows,
        "site_mapping_rows": int(len(mapping)),
        "successful_site_mappings": mapping_pass,
        "mapping_success_rate": mapping_pass / len(mapping),
        "pairs_with_mapped_sites": int(local["L0"]["mapped_site_count"].gt(0).sum()),
        "pairs_with_structure_descriptors": int(local["L2"]["local2_descriptor_site_count"].notna().sum()),
        "site_rows_with_structure_descriptors": structure_pass,
        "site_rows_with_residue_esm2": int(mapping["local2_residue_esm2_t12_cosine"].notna().sum()),
        "af_pdb_site_rows": int(len(robustness)),
        "af_pdb_descriptor_pass": af_pdb_pass,
        "af_pdb_descriptor_pass_rate": af_pdb_pass / len(robustness),
        "local_radii_angstrom": [6, 8, 10],
        "mapping_success_is_uncertainty_only": True,
        "query_ground_truth_used": False,
        "qc_failures": failures,
    }
    (reports / "phase07_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# SiteGuard V4 Phase 7 Report", "", f"Status: **{status}**", "",
        "## Site mapping and local representations", "",
        f"- Activity-transfer rows: {expected_rows:,}",
        f"- Reference-site mapping rows: {len(mapping):,}",
        f"- Successfully aligned site rows: {mapping_pass:,} ({mapping_pass / len(mapping):.2%})",
        f"- Pair rows with at least one mapped site: {summary['pairs_with_mapped_sites']:,}",
        f"- Site rows with AlphaFold local descriptors: {structure_pass:,}",
        f"- Site rows with residue-centred ESM2-t12 similarity: {summary['site_rows_with_residue_esm2']:,}", "",
        "Local-0 encodes residue identity; Local-1 chemical/role compatibility; Local-2 local composition plus frozen residue-centred ESM2; Local-3 coarse 3D geometry at 6, 8, and 10 Å.", "",
        "## AlphaFold–PDB robustness input", "",
        f"- Curated M-CSA site rows: {len(robustness):,}",
        f"- Descriptor-complete AF–PDB comparisons: {af_pdb_pass:,} ({af_pdb_pass / len(robustness):.2%})",
        f"- Proteins: {robustness['reference_protein_id'].nunique():,}; PDB entries: {robustness['pdb_id'].nunique():,}", "",
        "This table is a reference-side robustness/control artifact. Mapping success and mapping confidence remain uncertainty controls, not evidence that local context adds biological value.", "",
        "## QC", "", qc.to_markdown(index=False), "",
    ]
    (reports / "PHASE_07_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    if failures:
        raise RuntimeError("Phase 7 final QC failed: " + ", ".join(failures))
    (root / "checkpoints/CHECKPOINT_07_PASS").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
