#!/usr/bin/env python3
"""Build the leakage-safe SiteGuard V4 global pair-feature matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAIR_KEY = ["query_protein_id", "reference_protein_id"]
ROW_KEY = ["pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id"]
MMSEQS_SOURCE = ["mmseqs_fident", "mmseqs_qcov", "mmseqs_tcov", "mmseqs_bits"]
FOLDSEEK_SOURCE = ["foldseek_fident", "foldseek_qcov", "foldseek_tcov", "foldseek_bits"]
GROUND_TRUTH_FORBIDDEN = {
    "query_ec_l3_ground_truth", "query_ec_l4_ground_truth", "query_rhea_ground_truth",
    "same_ec_l3", "same_ec_l4", "same_exact_rhea", "observed_concordance_depth",
    "is_difficult_case", "canonical_rhea", "ec_l3", "ec_l4",
}


def json_ids(value: object) -> frozenset[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return frozenset()
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return frozenset()
    if not isinstance(parsed, list):
        return frozenset()
    values: list[str] = []
    for item in parsed:
        if isinstance(item, dict) and item.get("id"):
            values.append(str(item["id"]))
        elif item:
            values.append(str(item))
    return frozenset(values)


def composition(sequence: str) -> np.ndarray:
    counts = np.fromiter((sequence.count(amino_acid) for amino_acid in AMINO_ACIDS), dtype=np.float32)
    total = float(counts.sum())
    return counts / total if total else counts


def jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def build_protein_metadata(root: Path) -> pd.DataFrame:
    benchmark = pd.read_parquet(
        root / "data/interim/phase03/benchmark_proteins.parquet",
        columns=["protein_id", "sequence", "pfam_domains_json"],
    )
    family = pd.read_parquet(
        root / "data/splits/split_family.parquet",
        columns=["protein_id", "primary_pfam", "primary_pfam_clan"],
    )
    structure = pd.read_parquet(
        root / "data/splits/split_structure.parquet",
        columns=["protein_id", "cath_superfamilies_json", "primary_cath_superfamily"],
    )
    membership = pd.read_parquet(
        root / "data/interim/phase04/afdb_membership.parquet",
        columns=["protein_id", "has_structure"],
    )
    frame = benchmark.merge(family, on="protein_id", validate="one_to_one")
    frame = frame.merge(structure, on="protein_id", validate="one_to_one")
    frame = frame.merge(membership, on="protein_id", validate="one_to_one")
    if not frame["protein_id"].is_unique:
        raise RuntimeError("Protein metadata is not protein-unique")
    frame["protein_length"] = frame["sequence"].str.len().astype(np.uint32)
    frame["pfam_set"] = frame["pfam_domains_json"].map(json_ids)
    frame["cath_set"] = frame["cath_superfamilies_json"].map(json_ids)
    return frame.reset_index(drop=True)


def aggregate_protein_pairs(root: Path) -> pl.DataFrame:
    columns = PAIR_KEY + ["query_split_expected"] + MMSEQS_SOURCE + FOLDSEEK_SOURCE
    frames = [
        pl.read_parquet(root / "data/processed/population_pairs.parquet", columns=columns),
        pl.read_parquet(root / "data/processed/training_pairs.parquet", columns=columns),
    ]
    return (
        pl.concat(frames, how="vertical_relaxed")
        .group_by(PAIR_KEY)
        .agg(
            pl.col("query_split_expected").drop_nulls().first(),
            *[pl.col(name).drop_nulls().first() for name in MMSEQS_SOURCE + FOLDSEEK_SOURCE],
        )
    )


def attach_direct_alignments(root: Path, pairs: pl.DataFrame) -> pl.DataFrame:
    work = root / "data/interim/phase06/direct_alignment"
    direct_sequence = pl.read_parquet(work / "direct_mmseqs_alignments.parquet")
    direct_structure = pl.read_parquet(work / "direct_foldseek_alignments.parquet")
    pairs = pairs.join(direct_sequence, on=PAIR_KEY, how="left", validate="1:1")
    pairs = pairs.join(direct_structure, on=PAIR_KEY, how="left", validate="1:1")
    pairs = pairs.with_columns(
        pl.coalesce(["mmseqs_fident", "direct_mmseqs_fident"]).cast(pl.Float32).alias("sequence_identity"),
        pl.coalesce(["mmseqs_qcov", "direct_mmseqs_qcov"]).cast(pl.Float32).alias("sequence_query_coverage"),
        pl.coalesce(["mmseqs_tcov", "direct_mmseqs_tcov"]).cast(pl.Float32).alias("sequence_reference_coverage"),
        pl.coalesce(["mmseqs_bits", "direct_mmseqs_bits"]).cast(pl.Float32).alias("sequence_bitscore"),
        pl.coalesce(["foldseek_fident", "direct_foldseek_fident"]).cast(pl.Float32).alias("foldseek_identity"),
        pl.coalesce(["foldseek_qcov", "direct_foldseek_qcov"]).cast(pl.Float32).alias("foldseek_query_coverage"),
        pl.coalesce(["foldseek_tcov", "direct_foldseek_tcov"]).cast(pl.Float32).alias("foldseek_reference_coverage"),
        pl.coalesce(["foldseek_bits", "direct_foldseek_bits"]).cast(pl.Float32).alias("foldseek_bitscore"),
    )
    return pairs.select(
        PAIR_KEY + ["query_split_expected", "sequence_identity", "sequence_query_coverage",
                    "sequence_reference_coverage", "sequence_bitscore", "foldseek_identity",
                    "foldseek_query_coverage", "foldseek_reference_coverage", "foldseek_bitscore"]
    )


def compute_dense_pair_features(
    pairs: pl.DataFrame, proteins: pd.DataFrame, embedding: np.ndarray, embedding_index: pd.DataFrame,
) -> pl.DataFrame:
    protein_rows = proteins[[
        "protein_id", "protein_length", "primary_pfam", "primary_pfam_clan",
        "primary_cath_superfamily", "has_structure",
    ]].copy()
    protein_rows["protein_row"] = np.arange(len(protein_rows), dtype=np.int64)
    embedding_map = embedding_index.set_index("protein_id")["row_index"]
    protein_rows["embedding_row"] = protein_rows["protein_id"].map(embedding_map)
    if protein_rows["embedding_row"].isna().any():
        raise RuntimeError("ESM2 index does not cover every V4 benchmark protein")
    pmeta = pl.from_pandas(protein_rows)
    pair_frame = (
        pairs.join(
            pmeta.rename({
                "protein_id": "query_protein_id", "protein_row": "query_protein_row",
                "embedding_row": "query_embedding_row", "protein_length": "query_length",
                "primary_pfam": "query_primary_pfam", "primary_pfam_clan": "query_primary_pfam_clan",
                "primary_cath_superfamily": "query_primary_cath", "has_structure": "query_has_structure",
            }), on="query_protein_id", how="left", validate="m:1",
        )
        .join(
            pmeta.rename({
                "protein_id": "reference_protein_id", "protein_row": "reference_protein_row",
                "embedding_row": "reference_embedding_row", "protein_length": "reference_length",
                "primary_pfam": "reference_primary_pfam", "primary_pfam_clan": "reference_primary_pfam_clan",
                "primary_cath_superfamily": "reference_primary_cath", "has_structure": "reference_has_structure",
            }), on="reference_protein_id", how="left", validate="m:1",
        )
    )
    index_columns = [
        "query_protein_row", "reference_protein_row", "query_embedding_row", "reference_embedding_row",
    ]
    if any(pair_frame[name].null_count() for name in index_columns):
        raise RuntimeError("Protein pair cannot be mapped to metadata/ESM2 rows")

    compositions = np.stack([composition(value) for value in proteins["sequence"]], axis=0)
    pfam_sets = proteins["pfam_set"].tolist()
    cath_sets = proteins["cath_set"].tolist()
    query_protein_rows = pair_frame["query_protein_row"].to_numpy().astype(np.int64, copy=False)
    reference_protein_rows = pair_frame["reference_protein_row"].to_numpy().astype(np.int64, copy=False)
    query_embedding_rows = pair_frame["query_embedding_row"].to_numpy().astype(np.int64, copy=False)
    reference_embedding_rows = pair_frame["reference_embedding_row"].to_numpy().astype(np.int64, copy=False)
    row_count = pair_frame.height
    esm_cosine = np.empty(row_count, dtype=np.float32)
    composition_cosine = np.empty(row_count, dtype=np.float32)
    pfam_jaccard = np.empty(row_count, dtype=np.float32)
    cath_jaccard = np.empty(row_count, dtype=np.float32)
    chunk_size = 10_000
    for start in range(0, row_count, chunk_size):
        end = min(start + chunk_size, row_count)
        qe = np.asarray(embedding[query_embedding_rows[start:end]], dtype=np.float32)
        re = np.asarray(embedding[reference_embedding_rows[start:end]], dtype=np.float32)
        denominator = np.linalg.norm(qe, axis=1) * np.linalg.norm(re, axis=1)
        esm_cosine[start:end] = np.divide(
            np.einsum("ij,ij->i", qe, re), denominator,
            out=np.zeros(end - start, dtype=np.float32), where=denominator > 0,
        )
        qc = compositions[query_protein_rows[start:end]]
        rc = compositions[reference_protein_rows[start:end]]
        denominator = np.linalg.norm(qc, axis=1) * np.linalg.norm(rc, axis=1)
        composition_cosine[start:end] = np.divide(
            np.einsum("ij,ij->i", qc, rc), denominator,
            out=np.zeros(end - start, dtype=np.float32), where=denominator > 0,
        )
        for offset, (query_row, reference_row) in enumerate(zip(
            query_protein_rows[start:end], reference_protein_rows[start:end], strict=True,
        )):
            pfam_jaccard[start + offset] = jaccard(pfam_sets[query_row], pfam_sets[reference_row])
            cath_jaccard[start + offset] = jaccard(cath_sets[query_row], cath_sets[reference_row])
        if end % 250_000 < chunk_size or end == row_count:
            print(json.dumps({"dense_pair_features": end, "total": row_count}), flush=True)

    pair_frame = pair_frame.with_columns(
        pl.Series("esm2_t33_cosine", esm_cosine),
        pl.Series("aa_composition_cosine", composition_cosine),
        pl.Series("pfam_jaccard", pfam_jaccard),
        pl.Series("cath_jaccard", cath_jaccard),
        (pl.min_horizontal("query_length", "reference_length") /
         pl.max_horizontal("query_length", "reference_length")).cast(pl.Float32).alias("length_ratio"),
        (pl.col("query_primary_pfam") == pl.col("reference_primary_pfam"))
        .fill_null(False).alias("primary_pfam_match"),
        (pl.col("query_primary_pfam_clan") == pl.col("reference_primary_pfam_clan"))
        .fill_null(False).alias("primary_pfam_clan_match"),
        (pl.col("query_primary_cath") == pl.col("reference_primary_cath"))
        .fill_null(False).alias("primary_cath_match"),
        (pl.col("query_has_structure") & pl.col("reference_has_structure")).alias("both_structures_available"),
        pl.col("sequence_bitscore").clip(lower_bound=0).log1p().cast(pl.Float32).alias("sequence_bitscore_log1p"),
        pl.min_horizontal("sequence_query_coverage", "sequence_reference_coverage")
        .cast(pl.Float32).alias("sequence_alignment_fraction"),
        pl.col("foldseek_bitscore").clip(lower_bound=0).log1p().cast(pl.Float32).alias("foldseek_bitscore_log1p"),
        pl.min_horizontal("foldseek_query_coverage", "foldseek_reference_coverage")
        .cast(pl.Float32).alias("foldseek_alignment_fraction"),
    )
    return pair_frame.select(
        PAIR_KEY + [
            pl.col("query_split_expected").alias("query_split"),
            "sequence_identity", "sequence_query_coverage", "sequence_reference_coverage",
            "sequence_bitscore_log1p", "sequence_alignment_fraction", "length_ratio",
            "aa_composition_cosine", "esm2_t33_cosine", "foldseek_identity",
            "foldseek_query_coverage", "foldseek_reference_coverage", "foldseek_bitscore_log1p",
            "foldseek_alignment_fraction", "both_structures_available", "pfam_jaccard",
            "primary_pfam_match", "primary_pfam_clan_match", "cath_jaccard", "primary_cath_match",
        ]
    )


def reaction_features(root: Path) -> pl.DataFrame:
    source = pl.read_parquet(root / "data/processed/reaction_features.parquet")
    numeric = [
        "smiles_parse_valid", "substrate_component_count", "product_component_count",
        "participant_count", "unique_participant_count", "currency_participant_count",
        "currency_fraction", "substrate_mw_sum", "product_mw_sum", "substrate_logp_sum",
        "product_logp_sum", "substrate_tpsa_sum", "product_tpsa_sum",
        "substrate_heavy_atoms", "product_heavy_atoms", "heavy_atom_change",
        "reaction_complexity", "cofactor_class",
    ]
    return source.select(
        "canonical_rhea", *[pl.col(name).alias(f"reference_{name}") for name in numeric],
        pl.lit(True).alias("reference_reaction_available"),
    )


def build_activity_rows(root: Path, pair_features: pl.DataFrame) -> pl.DataFrame:
    selected = PAIR_KEY + ["reference_activity_id", "ec_l3", "canonical_rhea"]
    frames = []
    for pair_set, filename in [
        ("population", "population_pairs.parquet"), ("training", "training_pairs.parquet"),
    ]:
        frames.append(
            pl.read_parquet(root / "data/processed" / filename, columns=selected)
            .with_columns(pl.lit(pair_set).alias("pair_set"))
        )
    rows = pl.concat(frames, how="vertical_relaxed")
    reactions = reaction_features(root)
    rows = rows.join(pair_features, on=PAIR_KEY, how="left", validate="m:1")
    rows = rows.with_columns(
        pl.col("ec_l3").str.extract(r"^([0-9]+)", 1).cast(pl.UInt8, strict=False).alias("reference_ec_l1")
    ).join(reactions, on="canonical_rhea", how="left", validate="m:1")
    rows = rows.with_columns(
        pl.col("reference_reaction_available").fill_null(False),
    ).drop(["ec_l3", "canonical_rhea"])
    return rows.select(ROW_KEY + [name for name in rows.columns if name not in ROW_KEY])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if not (root / "checkpoints/CHECKPOINT_05_PASS").is_file():
        raise RuntimeError("CHECKPOINT_05_PASS is required before global feature construction")
    work = root / "data/interim/phase06"
    reports = root / "reports"
    work.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    embedding = np.load(work / "esm2_t33_v4_embeddings.npy", mmap_mode="r")
    embedding_index = pd.read_csv(work / "esm2_t33_v4_index.tsv", sep="\t", dtype={"protein_id": str})
    if len(embedding_index) != embedding.shape[0] or embedding.shape[1] != 1280:
        raise RuntimeError("V4 ESM2 index/matrix mismatch")

    proteins = build_protein_metadata(root)
    pairs = attach_direct_alignments(root, aggregate_protein_pairs(root))
    pair_features = compute_dense_pair_features(pairs, proteins, embedding, embedding_index)
    pair_features.write_parquet(work / "global_pair_features.parquet", compression="zstd")
    features = build_activity_rows(root, pair_features)

    forbidden = sorted(GROUND_TRUTH_FORBIDDEN & set(features.columns))
    forbidden.extend(sorted(name for name in features.columns if name.startswith("hard_H")))
    forbidden.extend(sorted(name for name in features.columns if "ground_truth" in name.lower()))
    if forbidden:
        failure = {"status": "FAIL", "forbidden_columns": sorted(set(forbidden))}
        (reports / "LEAKAGE_FAILURE_REPORT.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"GROUND_TRUTH_ONLY fields entered feature matrix: {sorted(set(forbidden))}")

    expected_rows = sum(
        pl.scan_parquet(root / "data/processed" / filename).select(pl.len()).collect().item()
        for filename in ["population_pairs.parquet", "training_pairs.parquet"]
    )
    unique_rows = features.select(pl.struct(ROW_KEY).n_unique()).item()
    sequence_columns = [
        "sequence_identity", "sequence_query_coverage", "sequence_reference_coverage",
        "sequence_bitscore_log1p", "sequence_alignment_fraction",
    ]
    sequence_missing = sum(int(features[name].null_count()) for name in sequence_columns)
    foldseek_columns = [
        "foldseek_identity", "foldseek_query_coverage", "foldseek_reference_coverage",
        "foldseek_bitscore_log1p", "foldseek_alignment_fraction",
    ]
    structure_rows = features.filter(pl.col("both_structures_available"))
    foldseek_missing_with_structures = sum(
        int(structure_rows[name].null_count()) for name in foldseek_columns
    )
    qc_checks = [
        ("checkpoint_05_present", True, "strict phase gate"),
        ("row_count_matches_pair_tables", features.height == expected_rows, f"{features.height}/{expected_rows}"),
        ("row_key_unique", unique_rows == features.height, f"{unique_rows}/{features.height}"),
        ("ground_truth_fields_absent", not forbidden, str(forbidden)),
        ("sequence_features_complete", sequence_missing == 0, str(sequence_missing)),
        ("foldseek_complete_when_both_structures", int(foldseek_missing_with_structures) == 0, str(foldseek_missing_with_structures)),
        ("esm2_complete", features["esm2_t33_cosine"].null_count() == 0, str(features["esm2_t33_cosine"].null_count())),
    ]
    qc = pd.DataFrame(
        [(name, "PASS" if passed else "FAIL", details) for name, passed, details in qc_checks],
        columns=["check", "status", "details"],
    )
    qc.to_csv(reports / "phase06_global_features_qc.tsv", sep="\t", index=False)
    failures = qc[qc["status"] == "FAIL"]
    if not failures.empty:
        raise RuntimeError("Phase 6 global feature QC failed: " + ", ".join(failures["check"]))

    output = root / "data/processed/global_features.parquet"
    features.write_parquet(output, compression="zstd")
    metadata_columns = set(ROW_KEY + ["query_split"])
    provenance = {
        name: (
            "IDENTIFIER_OR_SPLIT_METADATA" if name in metadata_columns else
            "REFERENCE_DERIVED" if name.startswith("reference_") else
            "QUERY_REFERENCE_DERIVED"
        )
        for name in features.columns
    }
    (reports / "global_feature_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    missingness = {name: float(features[name].null_count() / features.height) for name in features.columns}
    summary = {
        "phase": 6, "stage": "global_features", "status": "PASS",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "rows": features.height, "unique_protein_pairs": pair_features.height,
        "model_input_features": len(features.columns) - len(metadata_columns),
        "ground_truth_fields_in_matrix": [], "missingness": missingness,
        "checkpoint_06_status": "PENDING_SELECTED_USALIGN_CALIBRATION_AND_FINAL_QC",
    }
    (reports / "phase06_global_features_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ["status", "rows", "unique_protein_pairs", "model_input_features"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
