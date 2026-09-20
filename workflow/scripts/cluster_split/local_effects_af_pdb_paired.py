#!/usr/bin/env python3
"""Paired AlphaFold/PDB sensitivity analysis for mapped M-CSA reference sites."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from local_features_local_structure_chunk import RADII, StructureStore, cosine, descriptor


ROW_KEY = ["pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id"]
TARGETS = {"EC_L3": "same_ec_l3", "EC_L4": "same_ec_l4", "EXACT_RHEA": "same_exact_rhea"}


class PDBStore:
    def __init__(self, source_root: Path, cache_size: int = 128) -> None:
        self.root = source_root / "data/raw/pdb/archive_mmcif"
        self.cache_size = cache_size
        self.cache: OrderedDict[tuple[str, str], tuple[dict[int, dict[str, Any]], str]] = OrderedDict()

    def get(self, pdb_id: str, chain_name: str) -> tuple[dict[int, dict[str, Any]], str]:
        key = (pdb_id.lower(), chain_name)
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        path = self.root / f"{key[0]}.cif.gz"
        try:
            content = gzip.decompress(path.read_bytes())
            document = gemmi.cif.read_string(content.decode("utf-8"))
            structure = gemmi.make_structure_from_block(document.sole_block())
            chains = [chain for chain in structure[0] if chain.name == chain_name]
            if not chains:
                chains = [chain for chain in structure[0] if chain.name.upper() == chain_name.upper()]
            if not chains:
                value = ({}, "CHAIN_UNAVAILABLE")
            else:
                residues: dict[int, dict[str, Any]] = {}
                for residue in chains[0]:
                    atoms = {
                        atom.name.strip(): np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32)
                        for atom in residue
                    }
                    if "CA" in atoms:
                        residues[int(residue.seqid.num)] = {
                            "aa": gemmi.find_tabulated_residue(residue.name).one_letter_code,
                            "atoms": atoms, "plddt": float("nan"),
                        }
                value = (residues, "PASS" if residues else "NO_CA_RESIDUES")
        except Exception as exc:  # noqa: BLE001
            value = ({}, f"{type(exc).__name__}:{exc}")
        self.cache[key] = value
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value


def local_scores(query: dict, reference: dict) -> tuple[float, float]:
    count_ratio = min(query["residue_count"], reference["residue_count"]) / max(query["residue_count"], reference["residue_count"], 1)
    local2 = float(np.nanmean([
        cosine(query["aa"], reference["aa"]), cosine(query["properties"], reference["properties"]), count_ratio,
    ]))
    local3 = float(np.nanmean([
        cosine(query["radial"], reference["radial"]), cosine(query["octants"], reference["octants"]),
        cosine(query["sidechain"], reference["sidechain"]),
    ]))
    return local2, local3


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
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    source = args.source_root.resolve()
    processed = root / "data/processed"
    results = root / "results/phase09"

    mapping = pd.read_parquet(processed / "site_mapping.parquet")
    robustness = pd.read_parquet(processed / "af_pdb_robustness_input.parquet")
    site_columns = [
        "reference_site_id", "pdb_id", "pdb_chain", "pdb_residue_number", "uniprot_residue_number"
    ]
    mapping = mapping.loc[
        mapping["mapping_status"].eq("PASS") & mapping["local_descriptor_status"].eq("PASS")
    ].merge(robustness[site_columns], on="reference_site_id", how="inner", validate="many_to_one")
    needed = set(mapping["query_protein_id"].astype(str)) | set(mapping["reference_protein_id"].astype(str))
    af_store = StructureStore(source, needed, cache_size=512)
    pdb_store = PDBStore(source)
    rows: list[dict] = []
    for pair_index, ((query_id, reference_id), group) in enumerate(
        mapping.groupby(["query_protein_id", "reference_protein_id"], sort=True), start=1
    ):
        query_structure, query_status = af_store.get(str(query_id))
        reference_structure, reference_status = af_store.get(str(reference_id))
        descriptor_cache: dict[tuple, dict | None] = {}
        for site in group.itertuples(index=False):
            pdb_structure, pdb_status = pdb_store.get(str(site.pdb_id), str(site.pdb_chain))
            row = {name: getattr(site, name) for name in ROW_KEY + ["reference_site_id"]}
            row.update({
                "pdb_id": site.pdb_id, "pdb_chain": site.pdb_chain,
                "pdb_residue_number": int(site.pdb_residue_number),
                "query_residue_number": int(site.mapped_query_position),
                "reference_residue_number": int(site.reference_site_position),
                "query_af_status": query_status, "reference_af_status": reference_status,
                "reference_pdb_status": pdb_status,
            })
            complete = True
            for radius in RADII:
                query_key = ("query", int(site.mapped_query_position), radius)
                af_key = ("reference_af", int(site.reference_site_position), radius)
                pdb_key = (str(site.pdb_id), str(site.pdb_chain), int(site.pdb_residue_number), radius)
                if query_key not in descriptor_cache:
                    descriptor_cache[query_key] = descriptor(query_structure, query_key[1], radius)
                if af_key not in descriptor_cache:
                    descriptor_cache[af_key] = descriptor(reference_structure, af_key[1], radius)
                if pdb_key not in descriptor_cache:
                    descriptor_cache[pdb_key] = descriptor(pdb_structure, int(site.pdb_residue_number), radius)
                qd, ad, pd_value = descriptor_cache[query_key], descriptor_cache[af_key], descriptor_cache[pdb_key]
                if qd is None or ad is None or pd_value is None:
                    complete = False
                    for name in ["af_local2", "pdb_local2", "af_local3", "pdb_local3"]:
                        row[f"{name}_r{int(radius)}"] = float("nan")
                    continue
                af_l2, af_l3 = local_scores(qd, ad)
                pdb_l2, pdb_l3 = local_scores(qd, pd_value)
                row[f"af_local2_r{int(radius)}"] = af_l2
                row[f"pdb_local2_r{int(radius)}"] = pdb_l2
                row[f"af_local3_r{int(radius)}"] = af_l3
                row[f"pdb_local3_r{int(radius)}"] = pdb_l3
            row["paired_descriptor_status"] = "PASS" if complete else "SITE_OR_DESCRIPTOR_UNAVAILABLE"
            rows.append(row)
        if pair_index % 1000 == 0:
            print(json.dumps({"protein_pairs": pair_index, "site_rows": len(rows)}), flush=True)
    site_frame = pd.DataFrame(rows)
    site_frame.to_parquet(results / "af_pdb_paired_site_features.parquet", index=False, compression="zstd")
    valid = site_frame.loc[site_frame["paired_descriptor_status"].eq("PASS")]
    score_columns = [name for name in valid.columns if name.startswith(("af_local", "pdb_local"))]
    pair_frame = valid.groupby(ROW_KEY, sort=False)[score_columns].mean().reset_index()

    label_columns = [
        "query_protein_id", "reference_protein_id", "reference_activity_id", "query_cluster_id_30",
        "same_ec_l3", "same_ec_l4", "same_exact_rhea",
    ]
    labels = pd.concat([
        pd.read_parquet(processed / "population_pairs.parquet", columns=label_columns).assign(pair_set="population"),
        pd.read_parquet(processed / "training_pairs.parquet", columns=label_columns).assign(pair_set="training"),
    ], ignore_index=True)
    features = pd.read_parquet(processed / "global_features.parquet", columns=ROW_KEY + [
        "sequence_identity", "esm2_t33_cosine", "foldseek_identity", "foldseek_alignment_fraction",
        "pfam_jaccard", "cath_jaccard", "length_ratio",
    ])
    pair_frame = pair_frame.merge(labels, on=ROW_KEY, validate="one_to_one").merge(features, on=ROW_KEY, validate="one_to_one")
    pair_frame.to_parquet(results / "af_pdb_paired_local_features.parquet", index=False, compression="zstd")

    summary_rows: list[dict] = []
    for radius in [6, 8, 10]:
        for layer in ["local2", "local3"]:
            af_column, pdb_column = f"af_{layer}_r{radius}", f"pdb_{layer}_r{radius}"
            values = pair_frame[[af_column, pdb_column]].dropna()
            correlation = spearmanr(values[af_column], values[pdb_column])
            summary_rows.append({
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
                summary_rows.append({
                    "analysis": "CONDITIONAL_EFFECT_CONSISTENCY", "descriptor": layer,
                    "radius_angstrom": radius, "annotation_level": level, "pair_rows": len(population),
                    "af_local_beta": coefficients["AF"][0], "af_local_se": coefficients["AF"][1], "af_local_p": coefficients["AF"][2],
                    "pdb_local_beta": coefficients["PDB"][0], "pdb_local_se": coefficients["PDB"][1], "pdb_local_p": coefficients["PDB"][2],
                    "effect_sign_consistent": np.sign(coefficients["AF"][0]) == np.sign(coefficients["PDB"][0]),
                    "mapping_success_used_as_feature": False,
                })
    output = pd.DataFrame(summary_rows)
    output.to_csv(results / "af_pdb_paired_sensitivity.tsv", sep="\t", index=False)
    summary = {
        "phase": 9, "stage": "paired_AF_PDB_sensitivity", "status": "PASS",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "candidate_site_rows": len(mapping), "paired_descriptor_site_rows": len(valid),
        "paired_activity_rows": len(pair_frame),
        "population_activity_rows": int(pair_frame["pair_set"].eq("population").sum()),
        "sensitivity_rows": len(output),
        "conditional_rows": int(output["analysis"].eq("CONDITIONAL_EFFECT_CONSISTENCY").sum()),
        "query_ground_truth_used_in_features": False,
    }
    (root / "reports/phase09_af_pdb_paired_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
