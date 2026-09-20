#!/usr/bin/env python3
"""Summarize already-computed Phase 9 paired AF/PDB local features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr


TARGETS = {"EC_L3": "same_ec_l3", "EC_L4": "same_ec_l4", "EXACT_RHEA": "same_exact_rhea"}


def standard_matrix(frame: pd.DataFrame, local_column: str) -> tuple[np.ndarray, list[str]]:
    columns = [
        "sequence_identity", "esm2_t33_cosine", "foldseek_identity", "foldseek_alignment_fraction",
        "pfam_jaccard", "cath_jaccard", "length_ratio", local_column,
    ]
    vectors: list[np.ndarray] = []
    usable: list[str] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        values = values.fillna(median).to_numpy(float)
        scale = float(values.std())
        if scale < 1e-10:
            continue
        vectors.append((values - values.mean()) / scale)
        usable.append(column)
    return sm.add_constant(np.column_stack(vectors), has_constant="add"), ["const", *usable]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    results = root / "results/phase09"
    site_frame = pd.read_parquet(results / "af_pdb_paired_site_features.parquet")
    pair_frame = pd.read_parquet(results / "af_pdb_paired_local_features.parquet")
    valid = site_frame.loc[site_frame["paired_descriptor_status"].eq("PASS")]
    rows: list[dict] = []
    for radius in [6, 8, 10]:
        for layer in ["local2", "local3"]:
            af_column, pdb_column = f"af_{layer}_r{radius}", f"pdb_{layer}_r{radius}"
            values = pair_frame[[af_column, pdb_column]].dropna()
            correlation = spearmanr(values[af_column], values[pdb_column])
            rows.append({
                "analysis": "PAIRED_QUERY_REFERENCE_LOCAL_SIMILARITY", "descriptor": layer,
                "radius_angstrom": radius, "pair_rows": len(values),
                "af_pdb_spearman_rho": float(correlation.statistic), "af_pdb_spearman_p": float(correlation.pvalue),
                "median_absolute_difference": float((values[af_column] - values[pdb_column]).abs().median()),
                "mean_af_local_similarity": float(values[af_column].mean()),
                "mean_pdb_local_similarity": float(values[pdb_column].mean()),
            })
            population = pair_frame.loc[pair_frame["pair_set"].eq("population")].dropna(subset=[af_column, pdb_column])
            for level, target in TARGETS.items():
                if len(population) < 100 or population[target].sum() < 10:
                    continue
                coefficients = {}
                for structure_name, local_column in [("AF", af_column), ("PDB", pdb_column)]:
                    matrix, names = standard_matrix(population, local_column)
                    fit = sm.GLM(population[target].astype(int), matrix, family=sm.families.Binomial()).fit(
                        maxiter=200, disp=0, cov_type="cluster",
                        cov_kwds={"groups": population["query_cluster_id_30"]},
                    )
                    index = names.index(local_column)
                    coefficients[structure_name] = (
                        float(fit.params.iloc[index]), float(fit.bse.iloc[index]), float(fit.pvalues.iloc[index])
                    )
                rows.append({
                    "analysis": "CONDITIONAL_EFFECT_CONSISTENCY", "descriptor": layer,
                    "radius_angstrom": radius, "annotation_level": level, "pair_rows": len(population),
                    "af_local_beta": coefficients["AF"][0], "af_local_se": coefficients["AF"][1], "af_local_p": coefficients["AF"][2],
                    "pdb_local_beta": coefficients["PDB"][0], "pdb_local_se": coefficients["PDB"][1], "pdb_local_p": coefficients["PDB"][2],
                    "effect_sign_consistent": np.sign(coefficients["AF"][0]) == np.sign(coefficients["PDB"][0]),
                    "mapping_success_used_as_feature": False,
                })
    output = pd.DataFrame(rows)
    output.to_csv(results / "af_pdb_paired_sensitivity.tsv", sep="\t", index=False)
    summary = {
        "phase": 9, "stage": "paired_AF_PDB_sensitivity", "status": "PASS",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "candidate_site_rows": len(site_frame), "paired_descriptor_site_rows": len(valid),
        "paired_activity_rows": len(pair_frame),
        "population_activity_rows": int(pair_frame["pair_set"].eq("population").sum()),
        "sensitivity_rows": len(output),
        "conditional_rows": int(output["analysis"].eq("CONDITIONAL_EFFECT_CONSISTENCY").sum()),
        "reused_completed_paired_features": True,
        "query_ground_truth_used_in_features": False,
    }
    (root / "reports/phase09_af_pdb_paired_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
