#!/usr/bin/env python3
"""Build frozen-model features for the P450 multi-scenario stress test."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from phase14_build_features import composition, cosine_rows, json_ids


LEVELS = ["EC_L3", "EC_L4", "EXACT_RHEA"]


def cyp_family(value: object) -> str:
    match = re.search(r"CYP(\d+)", str(value).upper())
    return f"CYP{match.group(1)}" if match else "UNKNOWN_CYP_FAMILY"


def lineage_group(value: object) -> str:
    try:
        lineage = {str(item).lower() for item in json.loads(str(value))}
    except (TypeError, ValueError, json.JSONDecodeError):
        lineage = set()
    if "viridiplantae" in lineage or "streptophyta" in lineage:
        return "Plant"
    if "fungi" in lineage:
        return "Fungi"
    if "metazoa" in lineage:
        return "Animal"
    if "bacteria" in lineage:
        return "Bacteria"
    if "archaea" in lineage:
        return "Archaea"
    return "Other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root, source = args.project_root.resolve(), args.source_root.resolve()
    work = root / "data/interim/phase15"
    reports = root / "reports"
    if not (root / "checkpoints/CHECKPOINT_14_PASS").is_file():
        raise RuntimeError("CHECKPOINT_14_PASS is required")

    hit_columns = [
        "query_protein_id", "reference_protein_id", "fident", "alnlen", "qstart", "qend", "qlen",
        "tstart", "tend", "tlen", "qcov", "tcov", "evalue", "bits",
    ]
    hits = pd.read_csv(work / "p450_mmseqs_hits.tsv", sep="\t", names=hit_columns)
    for column in ["fident", "qcov", "tcov", "bits", "evalue"]:
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    for column in ["fident", "qcov", "tcov"]:
        if hits[column].max() > 1:
            hits[column] /= 100.0
    hits = hits.sort_values(
        ["query_protein_id", "bits", "reference_protein_id"], ascending=[True, False, True],
    ).drop_duplicates(["query_protein_id", "reference_protein_id"])
    hits["retrieval_rank_raw"] = hits.groupby("query_protein_id").cumcount() + 1
    hits = hits.loc[hits["retrieval_rank_raw"].le(500)].reset_index(drop=True)
    if hits.empty:
        raise RuntimeError("MMseqs produced no P450 candidates")

    query = pd.read_parquet(work / "p450_query_metadata.parquet")
    query["pfam_set"] = query["pfam_ids_json"].map(json_ids)
    query["composition"] = query["sequence"].map(composition)
    query_lookup = query.set_index("accession")
    reference_ids = set(hits["reference_protein_id"].astype(str))
    proteins = pd.read_parquet(
        root / "data/processed/protein_table.parquet",
        columns=[
            "protein_id", "sequence", "length", "pfam_domains_json", "entry_name", "protein_name",
            "gene_names_json", "lineage_json",
        ],
    )
    proteins = proteins.loc[proteins["protein_id"].isin(reference_ids)].copy()
    family = pd.read_parquet(
        root / "data/splits/split_family.parquet",
        columns=["protein_id", "primary_pfam", "primary_pfam_clan"],
    )
    sequence_split = pd.read_parquet(
        root / "data/splits/split_sequence.parquet", columns=["protein_id", "cluster_id_30", "split"],
    )
    membership = pd.read_parquet(
        root / "data/interim/phase04/afdb_membership.parquet", columns=["protein_id", "has_structure"],
    )
    reference = proteins.merge(family, on="protein_id", how="left", validate="one_to_one")
    reference = reference.merge(sequence_split, on="protein_id", how="left", validate="one_to_one")
    reference = reference.merge(membership, on="protein_id", how="left", validate="one_to_one")
    if not reference["split"].eq("train").all():
        raise RuntimeError("P450 external reference library contains a non-training protein")
    combined = (
        reference["entry_name"].astype(str) + " " + reference["protein_name"].astype(str) + " "
        + reference["gene_names_json"].astype(str)
    )
    reference["reference_is_cyp"] = (
        reference["protein_name"].astype(str).str.contains("cytochrome P450", case=False, regex=False)
        | combined.str.contains(r"CYP[0-9]", case=False, regex=True)
    )
    reference["reference_cyp_family"] = combined.map(cyp_family)
    reference["reference_species_group"] = reference["lineage_json"].map(lineage_group)
    reference["pfam_set"] = reference["pfam_domains_json"].map(json_ids)
    reference["composition"] = reference["sequence"].map(composition)
    reference_lookup = reference.set_index("protein_id")

    hits["query_cyp_family"] = hits["query_protein_id"].map(query_lookup["cyp_family"])
    hits["query_species_group"] = hits["query_protein_id"].map(query_lookup["species_group"])
    hits["reference_cyp_family"] = hits["reference_protein_id"].map(reference_lookup["reference_cyp_family"])
    hits["reference_is_cyp"] = hits["reference_protein_id"].map(reference_lookup["reference_is_cyp"]).fillna(False)
    hits["reference_species_group"] = hits["reference_protein_id"].map(reference_lookup["reference_species_group"]).fillna("Other")
    hits["same_cyp_family"] = (
        hits["query_cyp_family"].ne("UNKNOWN_CYP_FAMILY")
        & hits["query_cyp_family"].eq(hits["reference_cyp_family"])
    )
    hits["eligible_GENERAL"] = hits["retrieval_rank_raw"].le(50)
    leave_pool = hits.loc[
        hits["query_cyp_family"].ne("UNKNOWN_CYP_FAMILY")
        & ~hits["same_cyp_family"]
        & ~(hits["reference_is_cyp"] & hits["reference_cyp_family"].eq("UNKNOWN_CYP_FAMILY"))
    ].copy()
    leave_pool["scenario_rank"] = leave_pool.groupby("query_protein_id").cumcount() + 1
    leave_keys = set(map(tuple, leave_pool.loc[leave_pool["scenario_rank"].le(50), ["query_protein_id", "reference_protein_id"]].to_numpy()))
    plant_pool = hits.loc[
        hits["query_species_group"].eq("Plant")
        & ~hits["reference_species_group"].isin(["Plant", "Other"])
    ].copy()
    plant_pool["scenario_rank"] = plant_pool.groupby("query_protein_id").cumcount() + 1
    plant_keys = set(map(tuple, plant_pool.loc[plant_pool["scenario_rank"].le(50), ["query_protein_id", "reference_protein_id"]].to_numpy()))
    keys = list(zip(hits["query_protein_id"], hits["reference_protein_id"], strict=True))
    hits["eligible_LEAVE_ONE_CYP_FAMILY_OUT"] = [key in leave_keys for key in keys]
    hits["eligible_PLANT_COLD_START"] = [key in plant_keys for key in keys]
    hits = hits.loc[
        hits[["eligible_GENERAL", "eligible_LEAVE_ONE_CYP_FAMILY_OUT", "eligible_PLANT_COLD_START"]].any(axis=1)
    ].reset_index(drop=True)

    old_embedding = np.load(source / "05_features/esm2_t33_embeddings.npy", mmap_mode="r")
    v4_embedding = np.load(root / "data/interim/phase06/esm2_t33_v4_embeddings.npy", mmap_mode="r")
    v4_index = pd.read_csv(root / "data/interim/phase06/esm2_t33_v4_index.tsv", sep="\t", dtype={"protein_id": str})
    reference_embedding = v4_index.set_index("protein_id")["row_index"].to_dict()
    query_embedding = query_lookup["esm2_source_row"].astype(int).to_dict()
    missing = set(hits["reference_protein_id"]) - set(reference_embedding)
    if missing:
        raise RuntimeError(f"Reference ESM2 rows missing: {len(missing)}")

    row_count = len(hits)
    esm_cosine = np.empty(row_count, dtype=np.float32)
    composition_cosine = np.empty(row_count, dtype=np.float32)
    pfam_jaccard = np.empty(row_count, dtype=np.float32)
    query_ids = hits["query_protein_id"].astype(str).to_numpy()
    ref_ids = hits["reference_protein_id"].astype(str).to_numpy()
    chunk_size = 50_000
    for start in range(0, row_count, chunk_size):
        end = min(start + chunk_size, row_count)
        qids, rids = query_ids[start:end], ref_ids[start:end]
        qrows = np.fromiter((query_embedding[value] for value in qids), dtype=np.int64, count=len(qids))
        rrows = np.fromiter((reference_embedding[value] for value in rids), dtype=np.int64, count=len(rids))
        esm_cosine[start:end] = cosine_rows(
            np.asarray(old_embedding[qrows], dtype=np.float32), np.asarray(v4_embedding[rrows], dtype=np.float32),
        )
        qc = np.stack([query_lookup.at[value, "composition"] for value in qids])
        rc = np.stack([reference_lookup.at[value, "composition"] for value in rids])
        composition_cosine[start:end] = cosine_rows(qc, rc)
        for offset, (query_id, reference_id) in enumerate(zip(qids, rids, strict=True)):
            first, second = query_lookup.at[query_id, "pfam_set"], reference_lookup.at[reference_id, "pfam_set"]
            union = first | second
            pfam_jaccard[start + offset] = len(first & second) / len(union) if union else 0.0
        print(json.dumps({"p450_protein_pair_features": end, "total": row_count}), flush=True)

    query_lengths = query_lookup.loc[query_ids, "sequence_length"].to_numpy(float)
    reference_lengths = reference_lookup.loc[ref_ids, "length"].to_numpy(float)
    protein_pairs = pd.DataFrame({
        "query_protein_id": query_ids, "reference_protein_id": ref_ids,
        "retrieval_rank": hits["retrieval_rank_raw"].to_numpy(np.int16),
        "reference_cluster_id_30": [reference_lookup.at[value, "cluster_id_30"] for value in ref_ids],
        "sequence_identity": hits["fident"].to_numpy(np.float32),
        "sequence_query_coverage": hits["qcov"].to_numpy(np.float32),
        "sequence_reference_coverage": hits["tcov"].to_numpy(np.float32),
        "sequence_bitscore_log1p": np.log1p(hits["bits"].clip(lower=0).to_numpy(np.float32)),
        "sequence_alignment_fraction": np.minimum(hits["qcov"], hits["tcov"]).to_numpy(np.float32),
        "length_ratio": np.minimum(query_lengths, reference_lengths) / np.maximum(query_lengths, reference_lengths),
        "aa_composition_cosine": composition_cosine, "esm2_t33_cosine": esm_cosine,
        "foldseek_identity": np.nan, "foldseek_query_coverage": np.nan,
        "foldseek_reference_coverage": np.nan, "foldseek_bitscore_log1p": np.nan,
        "foldseek_alignment_fraction": np.nan, "both_structures_available": False,
        "query_structure_available": [bool(query_lookup.at[value, "query_structure_available"]) for value in query_ids],
        "reference_structure_available": [bool(reference_lookup.at[value, "has_structure"]) for value in ref_ids],
        "pfam_jaccard": pfam_jaccard,
        "primary_pfam_match": [
            bool(query_lookup.at[q, "primary_pfam"]) and query_lookup.at[q, "primary_pfam"] == reference_lookup.at[r, "primary_pfam"]
            for q, r in zip(query_ids, ref_ids, strict=True)
        ],
        "primary_pfam_clan_match": [
            bool(query_lookup.at[q, "primary_pfam_clan"])
            and query_lookup.at[q, "primary_pfam_clan"] == reference_lookup.at[r, "primary_pfam_clan"]
            for q, r in zip(query_ids, ref_ids, strict=True)
        ],
        "cath_jaccard": 0.0, "primary_cath_match": False,
        "query_cyp_family": hits["query_cyp_family"].to_numpy(),
        "reference_cyp_family": hits["reference_cyp_family"].to_numpy(),
        "query_species_group": hits["query_species_group"].to_numpy(),
        "reference_species_group": hits["reference_species_group"].to_numpy(),
        "eligible_GENERAL": hits["eligible_GENERAL"].to_numpy(bool),
        "eligible_LEAVE_ONE_CYP_FAMILY_OUT": hits["eligible_LEAVE_ONE_CYP_FAMILY_OUT"].to_numpy(bool),
        "eligible_PLANT_COLD_START": hits["eligible_PLANT_COLD_START"].to_numpy(bool),
    })
    protein_pairs.to_parquet(work / "p450_protein_pair_features.parquet", index=False, compression="zstd")

    activities = pl.read_parquet(root / "data/reference/activity_reference_library.parquet").select(
        pl.col("activity_id").alias("reference_activity_id"), "reference_protein_id", "ec_l3", "ec_l4",
        "canonical_rhea", "evidence_tier",
    )
    reactions = pl.read_parquet(root / "data/processed/reaction_features.parquet").select(
        "canonical_rhea", pl.col("smiles_parse_valid").alias("reference_smiles_parse_valid"),
        pl.col("substrate_component_count").alias("reference_substrate_component_count"),
        pl.col("product_component_count").alias("reference_product_component_count"),
        pl.col("participant_count").alias("reference_participant_count"),
        pl.col("unique_participant_count").alias("reference_unique_participant_count"),
        pl.col("currency_participant_count").alias("reference_currency_participant_count"),
        pl.col("currency_fraction").alias("reference_currency_fraction"),
        pl.col("substrate_mw_sum").alias("reference_substrate_mw_sum"),
        pl.col("product_mw_sum").alias("reference_product_mw_sum"),
        pl.col("substrate_logp_sum").alias("reference_substrate_logp_sum"),
        pl.col("product_logp_sum").alias("reference_product_logp_sum"),
        pl.col("substrate_tpsa_sum").alias("reference_substrate_tpsa_sum"),
        pl.col("product_tpsa_sum").alias("reference_product_tpsa_sum"),
        pl.col("substrate_heavy_atoms").alias("reference_substrate_heavy_atoms"),
        pl.col("product_heavy_atoms").alias("reference_product_heavy_atoms"),
        pl.col("heavy_atom_change").alias("reference_heavy_atom_change"),
        pl.col("reaction_complexity").alias("reference_reaction_complexity"),
        pl.col("cofactor_class").alias("reference_cofactor_class"),
        pl.lit(True).alias("reference_reaction_available"),
    )
    rows = pl.from_pandas(protein_pairs).join(activities, on="reference_protein_id", how="inner", validate="m:m")
    rows = rows.with_columns(
        pl.col("ec_l3").str.extract(r"^([0-9]+)", 1).cast(pl.UInt8, strict=False).alias("reference_ec_l1")
    ).join(reactions, on="canonical_rhea", how="left", validate="m:1")
    rows = rows.with_columns(pl.col("reference_reaction_available").fill_null(False)).with_row_index("pair_index")
    feature_path = work / "p450_pair_features.parquet"
    rows.write_parquet(feature_path, compression="zstd")
    pair_rows, scored_queries = rows.height, rows["query_protein_id"].n_unique()
    del rows, protein_pairs, hits

    preprocessing = json.loads((root / "data/interim/phase11/preprocessing.json").read_text(encoding="utf-8"))
    numeric = preprocessing["numeric_columns"]
    dummy_columns = preprocessing["categorical_dummy_columns"]
    model_columns = preprocessing["model_columns"]
    medians, means, scales = preprocessing["numeric_medians"], preprocessing["numeric_means"], preprocessing["numeric_scales"]
    matrix = np.lib.format.open_memmap(
        work / "p450_X.npy", mode="w+", dtype=np.float32, shape=(pair_rows, len(model_columns)),
    )
    tree_prediction = np.lib.format.open_memmap(
        work / "p450_tree_predictions.npy", mode="w+", dtype=np.float32, shape=(pair_rows, len(LEVELS)),
    )
    boosters = {
        level: lgb.Booster(model_file=str(root / "models/phase10" / f"lightgbm_global_{level}.txt"))
        for level in LEVELS
    }
    offset, batch_size = 0, 100_000
    for batch in pl.read_parquet(feature_path).iter_slices(n_rows=batch_size):
        frame = batch.to_pandas()
        numeric_frame = frame[numeric].apply(pd.to_numeric, errors="coerce").astype(np.float32)
        for column in numeric:
            numeric_frame[column] = numeric_frame[column].fillna(float(medians[column]))
            numeric_frame[column] = (numeric_frame[column] - float(means[column])) / float(scales[column])
        dummy_values: dict[str, np.ndarray] = {}
        for column in dummy_columns:
            if column.startswith("reference_ec_l1_"):
                value = column.removeprefix("reference_ec_l1_")
                ec_values = pd.to_numeric(frame["reference_ec_l1"], errors="coerce").map(
                    lambda item: str(int(item)) if pd.notna(item) else "MISSING"
                )
                dummy_values[column] = ec_values.eq(value).to_numpy(np.float32)
            elif column.startswith("reference_cofactor_class_"):
                value = column.removeprefix("reference_cofactor_class_")
                dummy_values[column] = frame["reference_cofactor_class"].fillna("MISSING").astype(str).eq(value).to_numpy(np.float32)
            else:
                raise RuntimeError(f"Unknown dummy column: {column}")
        values = np.column_stack([
            numeric_frame.to_numpy(np.float32),
            np.column_stack([dummy_values[column] for column in dummy_columns]),
        ])
        end = offset + len(values)
        matrix[offset:end] = values
        for level_index, level in enumerate(LEVELS):
            tree_prediction[offset:end, level_index] = boosters[level].predict(values).astype(np.float32)
        offset = end
        print(json.dumps({"p450_model_matrix_rows": offset, "total": pair_rows}), flush=True)
    matrix.flush(); tree_prediction.flush()
    if offset != pair_rows:
        raise RuntimeError(f"P450 matrix row mismatch: {offset}/{pair_rows}")
    summary = {
        "phase": 15, "stage": "p450_external_features_and_tree", "status": "PASS",
        "slurm_job_id": os.getenv("SLURM_JOB_ID", "NA"), "mmseqs_union_protein_pairs": row_count,
        "activity_conditioned_pairs": pair_rows, "retrieved_queries": int(scored_queries),
        "panel_queries": len(query), "model_features": len(model_columns),
        "scenario_eligible_protein_pairs": {
            scenario: int(pd.read_parquet(work / "p450_protein_pair_features.parquet", columns=[f"eligible_{scenario}"])[f"eligible_{scenario}"].sum())
            for scenario in ["GENERAL", "LEAVE_ONE_CYP_FAMILY_OUT", "PLANT_COLD_START"]
        },
        "direct_structure_features_in_external_primary_model": False,
        "p450_truth_used_as_model_input": False, "external_thresholds_tuned": False,
    }
    (reports / "phase15_feature_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
