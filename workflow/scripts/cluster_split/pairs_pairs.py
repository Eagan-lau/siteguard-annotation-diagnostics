#!/usr/bin/env python3
"""Build population and balanced activity-transfer pair sets after strict splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


SEED = 20260819
PAIR_KEY = ["query_protein_id", "reference_protein_id", "reference_activity_id"]
RETRIEVAL_COLUMNS = [
    "query_protein_id", "reference_protein_id", "fident", "qcov", "tcov", "bits",
    "modality_rank", "modality", "query_partition", "reference_cluster_id_30",
]


def json_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    try:
        parsed = json.loads(str(value))
        return [str(item) for item in parsed if item] if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def stable_u64(text: str, seed: int = SEED) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{text}".encode()).digest()[:8], "big")


def cycle_indices(size: int, key: str) -> Iterable[int]:
    if size <= 0:
        return
    start = stable_u64(key) % size
    for offset in range(size):
        yield (start + offset) % size


def numeric_smd(treated: pd.Series, control: pd.Series) -> tuple[float, float, float, int, int]:
    left = pd.to_numeric(treated, errors="coerce").dropna()
    right = pd.to_numeric(control, errors="coerce").dropna()
    if left.empty or right.empty:
        return float("nan"), float("nan"), float("nan"), len(left), len(right)
    pooled = math.sqrt((float(left.var(ddof=1)) + float(right.var(ddof=1))) / 2.0)
    smd = 0.0 if pooled == 0 else (float(left.mean()) - float(right.mean())) / pooled
    return smd, float(left.mean()), float(right.mean()), len(left), len(right)


def categorical_max_smd(treated: pd.Series, control: pd.Series) -> tuple[float, str, int, int]:
    left = treated.fillna("MISSING").astype(str)
    right = control.fillna("MISSING").astype(str)
    levels = sorted(set(left).union(right))
    best, best_level = 0.0, ""
    for level in levels:
        p1, p0 = float((left == level).mean()), float((right == level).mean())
        pooled = math.sqrt(max((p1 * (1 - p1) + p0 * (1 - p0)) / 2.0, 0.0))
        value = 0.0 if pooled == 0 else (p1 - p0) / pooled
        if abs(value) > abs(best):
            best, best_level = value, level
    return best, best_level, len(left), len(right)


def candidate_scan(path: Path, modality: str) -> pl.LazyFrame:
    prefix = "mmseqs" if modality == "mmseqs" else "foldseek"
    return (
        pl.scan_parquet(path)
        .filter(pl.col("modality_rank") <= 50)
        .select(RETRIEVAL_COLUMNS)
        .with_columns(
            pl.when(pl.col("modality") == modality).then(pl.col("modality_rank")).otherwise(None).alias(f"{prefix}_rank"),
            pl.when(pl.col("modality") == modality).then(pl.col("fident")).otherwise(None).alias(f"{prefix}_fident"),
            pl.when(pl.col("modality") == modality).then(pl.col("qcov")).otherwise(None).alias(f"{prefix}_qcov"),
            pl.when(pl.col("modality") == modality).then(pl.col("tcov")).otherwise(None).alias(f"{prefix}_tcov"),
            pl.when(pl.col("modality") == modality).then(pl.col("bits")).otherwise(None).alias(f"{prefix}_bits"),
        )
        .drop(["fident", "qcov", "tcov", "bits", "modality_rank", "modality"])
    )


def build_retrieval_universe(root: Path, work: Path) -> pl.DataFrame:
    inputs = [
        (root / "data/processed/mmseqs_candidates.parquet", "mmseqs"),
        (root / "data/processed/foldseek_candidates.parquet", "foldseek"),
        (work / "mmseqs_train_candidates.parquet", "mmseqs"),
        (work / "foldseek_train_candidates.parquet", "foldseek"),
    ]
    for path, _ in inputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing retrieval candidates: {path}")
    universe = (
        pl.concat([candidate_scan(path, modality) for path, modality in inputs], how="diagonal_relaxed")
        .group_by(["query_protein_id", "reference_protein_id"])
        .agg(
            pl.col("query_partition").drop_nulls().first(),
            pl.col("reference_cluster_id_30").drop_nulls().first(),
            pl.col("mmseqs_rank").min(),
            pl.col("mmseqs_fident").max(),
            pl.col("mmseqs_qcov").max(),
            pl.col("mmseqs_tcov").max(),
            pl.col("mmseqs_bits").max(),
            pl.col("foldseek_rank").min(),
            pl.col("foldseek_fident").max(),
            pl.col("foldseek_qcov").max(),
            pl.col("foldseek_tcov").max(),
            pl.col("foldseek_bits").max(),
        )
        .with_columns(
            pl.when(pl.col("mmseqs_rank").is_not_null() & pl.col("foldseek_rank").is_not_null())
            .then(pl.lit("mmseqs+foldseek"))
            .when(pl.col("mmseqs_rank").is_not_null()).then(pl.lit("mmseqs_only"))
            .otherwise(pl.lit("foldseek_only")).alias("candidate_source_class"),
            pl.min_horizontal([pl.col("mmseqs_rank"), pl.col("foldseek_rank")]).alias("best_retrieval_rank"),
        )
        .with_columns(
            pl.when(pl.col("best_retrieval_rank") <= 10).then(pl.lit("01-10"))
            .when(pl.col("best_retrieval_rank") <= 20).then(pl.lit("11-20"))
            .otherwise(pl.lit("21-50")).alias("retrieval_rank_bin")
        )
        .collect(engine="streaming")
    )
    if universe.select(pl.struct(["query_protein_id", "reference_protein_id"]).n_unique()).item() != universe.height:
        raise RuntimeError("Retrieval protein-pair universe is not unique")
    if universe.filter(pl.col("query_protein_id") == pl.col("reference_protein_id")).height:
        raise RuntimeError("Retrieval universe contains self pairs")
    output = work / "retrieval_pair_universe.parquet"
    universe.write_parquet(output, compression="zstd")
    return universe


def metadata_frames(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark = pd.read_parquet(root / "data/interim/phase03/benchmark_proteins.parquet")
    split = pd.read_parquet(root / "data/splits/split_sequence.parquet", columns=["protein_id", "split", "cluster_id_30"])
    family = pd.read_parquet(
        root / "data/splits/split_family.parquet",
        columns=["protein_id", "primary_pfam", "primary_pfam_clan"],
    )
    structure = pd.read_parquet(
        root / "data/splits/split_structure.parquet",
        columns=["protein_id", "primary_cath_superfamily", "foldseek_structure_cluster"],
    )
    taxonomy = pd.read_parquet(
        root / "data/splits/split_taxonomy.parquet",
        columns=["protein_id", "taxonomy_group"],
    )
    af = pd.read_parquet(root / "data/interim/phase04/afdb_membership.parquet", columns=["protein_id", "has_structure"])
    meta = benchmark.merge(split, on="protein_id", validate="one_to_one")
    meta = meta.merge(family, on="protein_id", validate="one_to_one")
    meta = meta.merge(structure, on="protein_id", validate="one_to_one")
    meta = meta.merge(taxonomy, on="protein_id", validate="one_to_one")
    meta = meta.merge(af, on="protein_id", how="left", validate="one_to_one")
    meta["has_structure"] = meta["has_structure"].fillna(False).astype(bool)
    meta["protein_length"] = meta["sequence"].str.len().astype(int)
    meta["length_bin"] = pd.cut(
        meta["protein_length"], bins=[0, 150, 250, 400, 600, 1000, np.inf],
        labels=["0001-0150", "0151-0250", "0251-0400", "0401-0600", "0601-1000", "1001+"],
    ).astype(str)
    for source, target in [
        ("ec_l3_json", "ec_l3_set"), ("ec_l4_json", "ec_l4_set"),
        ("canonical_rhea_json", "rhea_set"),
    ]:
        meta[target] = meta[source].map(json_list)
    keep = [
        "protein_id", "split", "cluster_id_30", "protein_length", "length_bin", "fragment_status",
        "taxonomy_group", "primary_pfam", "primary_pfam_clan", "primary_cath_superfamily",
        "foldseek_structure_cluster", "has_structure", "ec_l3_set", "ec_l4_set", "rhea_set",
    ]
    meta = meta[keep]
    activities = pd.read_parquet(
        root / "data/reference/activity_reference_library.parquet",
        columns=["activity_id", "reference_protein_id", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier"],
    ).rename(columns={"activity_id": "reference_activity_id"})
    if not activities["reference_activity_id"].is_unique:
        raise RuntimeError("Reference activity identifiers are not unique")
    return meta, activities


def write_augmentations(root: Path, work: Path, meta: pd.DataFrame, activities: pd.DataFrame, config: dict) -> Path:
    counts = config["augmentation_per_query"]
    activity_rows = activities.sort_values(["reference_protein_id", "reference_activity_id"]).reset_index(drop=True)
    by_rhea: dict[str, list[int]] = defaultdict(list)
    by_ec4: dict[str, list[int]] = defaultdict(list)
    by_ec3: dict[str, list[int]] = defaultdict(list)
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, row in activity_rows.iterrows():
        if pd.notna(row.canonical_rhea):
            by_rhea[str(row.canonical_rhea)].append(index)
        by_ec4[str(row.ec_l4)].append(index)
        by_ec3[str(row.ec_l3)].append(index)
        by_reference[str(row.reference_protein_id)].append(index)

    train_meta = meta.loc[meta["split"] == "train"].copy()
    family_refs: dict[str, list[str]] = defaultdict(list)
    matched_refs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in train_meta.sort_values("protein_id").itertuples(index=False):
        family_refs[str(row.primary_pfam)].append(row.protein_id)
        matched_refs[(str(row.length_bin), str(row.taxonomy_group))].append(row.protein_id)
    all_refs = sorted(by_reference)
    truth = meta.set_index("protein_id")[["ec_l3_set", "ec_l4_set", "rhea_set"]].to_dict("index")
    output = work / "augmented_activity_candidates.parquet"
    schema = pa.schema([
        pa.field("query_protein_id", pa.string()), pa.field("reference_protein_id", pa.string()),
        pa.field("reference_activity_id", pa.string()), pa.field("augmentation_source", pa.string()),
        pa.field("candidate_sampling_probability", pa.float64()),
    ])
    temporary = output.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    batch: list[dict] = []
    rows_written = 0

    def flush() -> None:
        nonlocal batch, rows_written
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            rows_written += len(batch)
            batch = []

    def activity_pick(
        query_id: str, pool: list[int], k: int, source: str,
        predicate: Callable[[pd.Series], bool] | None = None,
    ) -> list[tuple[int, float]]:
        chosen: list[tuple[int, float]] = []
        if k <= 0 or not pool:
            return chosen
        for position in cycle_indices(len(pool), f"{query_id}|{source}"):
            index = pool[position]
            row = activity_rows.iloc[index]
            if row.reference_protein_id == query_id or (predicate is not None and not predicate(row)):
                continue
            chosen.append((index, min(1.0, k / len(pool))))
            if len(chosen) >= k:
                break
        return chosen

    def protein_pick(query_id: str, pool: list[str], k: int, source: str, divergent: bool = True) -> list[tuple[int, float]]:
        chosen: list[tuple[int, float]] = []
        if k <= 0 or not pool:
            return chosen
        q = truth[query_id]
        for position in cycle_indices(len(pool), f"{query_id}|{source}"):
            reference = pool[position]
            if reference == query_id:
                continue
            indices = by_reference.get(reference, [])
            for aindex in cycle_indices(len(indices), f"{query_id}|{reference}|{source}"):
                index = indices[aindex]
                activity = activity_rows.iloc[index]
                is_divergent = (
                    str(activity.ec_l4) not in set(q["ec_l4_set"])
                    and (pd.isna(activity.canonical_rhea) or str(activity.canonical_rhea) not in set(q["rhea_set"]))
                )
                if divergent and not is_divergent:
                    continue
                chosen.append((index, min(1.0, k / len(pool))))
                break
            if len(chosen) >= k:
                break
        return chosen

    try:
        for query in meta.sort_values("protein_id").itertuples(index=False):
            query_id = query.protein_id
            selected: dict[tuple[str, str], tuple[set[str], float]] = {}

            def add(index: int, probability: float, source: str) -> None:
                activity = activity_rows.iloc[index]
                key = (str(activity.reference_protein_id), str(activity.reference_activity_id))
                if key not in selected:
                    selected[key] = ({source}, probability)
                else:
                    sources, previous = selected[key]
                    sources.add(source)
                    selected[key] = (sources, max(previous, probability))

            for label in query.rhea_set:
                for index, probability in activity_pick(query_id, by_rhea.get(label, []), counts["same_exact_rhea_per_label"], f"same_exact_rhea:{label}"):
                    add(index, probability, "same_exact_rhea")
            for label in query.ec_l4_set:
                for index, probability in activity_pick(query_id, by_ec4.get(label, []), counts["same_ec_l4_per_label"], f"same_ec_l4:{label}"):
                    add(index, probability, "same_ec_l4")
            for label in query.ec_l3_set:
                for index, probability in activity_pick(query_id, by_ec3.get(label, []), counts["same_ec_l3_per_label"], f"same_ec_l3:{label}"):
                    add(index, probability, "same_ec_l3")
            for index, probability in protein_pick(query_id, family_refs.get(str(query.primary_pfam), []), counts["same_family_divergent"], "same_family_divergent"):
                add(index, probability, "same_family_divergent")
            match_pool = matched_refs.get((str(query.length_bin), str(query.taxonomy_group)), [])
            for index, probability in protein_pick(query_id, match_pool, counts["matched_controls"], "matched_control"):
                add(index, probability, "matched_control")
            for index, probability in protein_pick(query_id, all_refs, counts["stratified_random_controls"], "stratified_random_control"):
                add(index, probability, "stratified_random_control")
            for (reference_id, activity_id), (sources, probability) in selected.items():
                batch.append({
                    "query_protein_id": query_id,
                    "reference_protein_id": reference_id,
                    "reference_activity_id": activity_id,
                    "augmentation_source": "+".join(sorted(sources)),
                    "candidate_sampling_probability": float(probability),
                })
                if len(batch) >= 100_000:
                    flush()
        flush()
    finally:
        writer.close()
    temporary.replace(output)
    (work / "augmentation_summary.json").write_text(
        json.dumps({"rows": rows_written, "queries": len(meta), "seed": SEED}, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def polars_metadata(meta: pd.DataFrame, activities: pd.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    converted = meta.copy()
    for column in ["ec_l3_set", "ec_l4_set", "rhea_set"]:
        converted[column] = converted[column].map(lambda values: list(values))
    pmeta = pl.from_pandas(converted)
    pactivities = pl.from_pandas(activities).with_columns(
        pl.when(pl.col("canonical_rhea").is_not_null())
        .then(pl.col("canonical_rhea"))
        .otherwise(pl.concat_str([pl.lit("EC4:"), pl.col("ec_l4")]))
        .alias("activity_group")
    )
    activity_frequency = pactivities.group_by("activity_group").len().rename({"len": "activity_frequency"})
    pactivities = pactivities.join(activity_frequency, on="activity_group", how="left")
    return pmeta, pactivities


def enrich_instances(instances: pl.DataFrame, pmeta: pl.DataFrame, pactivities: pl.DataFrame) -> pl.DataFrame:
    query_meta = pmeta.select(
        pl.col("protein_id").alias("query_protein_id"), pl.col("split").alias("query_split_expected"),
        pl.col("cluster_id_30").alias("query_cluster_id_30"), pl.col("protein_length").alias("query_length"),
        pl.col("length_bin").alias("query_length_bin"), pl.col("fragment_status").alias("query_fragment_status"),
        pl.col("taxonomy_group").alias("query_taxonomy_group"), pl.col("primary_pfam").alias("query_primary_pfam"),
        pl.col("primary_pfam_clan").alias("query_primary_pfam_clan"),
        pl.col("primary_cath_superfamily").alias("query_primary_cath_superfamily"),
        pl.col("foldseek_structure_cluster").alias("query_structure_cluster"),
        pl.col("has_structure").alias("query_has_structure"),
        pl.col("ec_l3_set").alias("query_ec_l3_ground_truth"),
        pl.col("ec_l4_set").alias("query_ec_l4_ground_truth"),
        pl.col("rhea_set").alias("query_rhea_ground_truth"),
    )
    reference_meta = pmeta.select(
        pl.col("protein_id").alias("reference_protein_id"), pl.col("split").alias("reference_partition_verified"),
        pl.col("cluster_id_30").alias("reference_cluster_id_30_verified"),
        pl.col("protein_length").alias("reference_length"), pl.col("length_bin").alias("reference_length_bin"),
        pl.col("fragment_status").alias("reference_fragment_status"),
        pl.col("taxonomy_group").alias("reference_taxonomy_group"),
        pl.col("primary_pfam").alias("reference_primary_pfam"),
        pl.col("primary_pfam_clan").alias("reference_primary_pfam_clan"),
        pl.col("primary_cath_superfamily").alias("reference_primary_cath_superfamily"),
        pl.col("foldseek_structure_cluster").alias("reference_structure_cluster"),
        pl.col("has_structure").alias("reference_has_structure"),
    )
    result = (
        instances.join(pactivities, on=["reference_activity_id", "reference_protein_id"], how="left", validate="m:1")
        .join(query_meta, on="query_protein_id", how="left", validate="m:1")
        .join(reference_meta, on="reference_protein_id", how="left", validate="m:1")
        .with_columns(
            pl.col("query_ec_l3_ground_truth").list.contains(pl.col("ec_l3")).fill_null(False).alias("same_ec_l3"),
            pl.col("query_ec_l4_ground_truth").list.contains(pl.col("ec_l4")).fill_null(False).alias("same_ec_l4"),
            pl.when(pl.col("canonical_rhea").is_not_null())
            .then(pl.col("query_rhea_ground_truth").list.contains(pl.col("canonical_rhea")))
            .otherwise(False).fill_null(False).alias("same_exact_rhea"),
        )
        .with_columns(
            pl.when(pl.col("same_exact_rhea")).then(3)
            .when(pl.col("same_ec_l4")).then(2)
            .when(pl.col("same_ec_l3")).then(1).otherwise(0).cast(pl.Int8).alias("observed_concordance_depth"),
            (pl.col("query_primary_pfam") == pl.col("reference_primary_pfam")).fill_null(False).alias("same_pfam_family"),
            (pl.col("query_primary_cath_superfamily") == pl.col("reference_primary_cath_superfamily"))
            .fill_null(False).alias("same_cath_superfamily"),
            (pl.col("query_taxonomy_group") == pl.col("reference_taxonomy_group")).fill_null(False).alias("same_taxonomy_group"),
            (pl.col("query_cluster_id_30") != pl.col("reference_cluster_id_30_verified")).fill_null(False).alias("different_sequence_cluster_30"),
            (pl.col("query_structure_cluster") != pl.col("reference_structure_cluster"))
            .fill_null(False).alias("different_structure_cluster"),
        )
        .with_columns(
            (pl.col("same_ec_l3") & ~pl.col("same_ec_l4")).alias("hard_H1_same_EC3_different_EC4"),
            (pl.col("same_ec_l4") & pl.col("canonical_rhea").is_not_null() &
             (pl.col("query_rhea_ground_truth").list.len() > 0) & ~pl.col("same_exact_rhea"))
            .alias("hard_H2_same_EC4_different_Rhea"),
            ((pl.col("mmseqs_fident") >= 0.40) & (pl.col("mmseqs_qcov") >= 0.70) &
             (pl.col("mmseqs_tcov") >= 0.70) & ~pl.col("same_exact_rhea"))
            .fill_null(False).alias("hard_H3_high_sequence_annotation_divergence"),
            ((pl.col("foldseek_rank") <= 10) & (pl.col("foldseek_qcov") >= 0.70) &
             (pl.col("foldseek_tcov") >= 0.70) & ~pl.col("same_exact_rhea"))
            .fill_null(False).alias("hard_H4_foldseek_high_annotation_divergence_provisional"),
            ((pl.col("same_pfam_family") | pl.col("same_cath_superfamily")) & ~pl.col("same_exact_rhea"))
            .alias("hard_H5_same_family_annotation_divergence"),
            (pl.col("same_exact_rhea") & pl.col("different_sequence_cluster_30") &
             (pl.col("mmseqs_fident").is_null() | (pl.col("mmseqs_fident") < 0.30)))
            .fill_null(False).alias("hard_H6_low_sequence_same_Rhea"),
            (pl.col("same_exact_rhea") & pl.col("query_has_structure") & pl.col("reference_has_structure") &
             pl.col("different_structure_cluster"))
            .fill_null(False).alias("hard_H7_different_structure_cluster_same_Rhea_provisional"),
            pl.lit(False).alias("hard_H8_global_high_local_low_fine_divergence"),
            pl.lit("PENDING_PHASE_07_LOCAL_FEATURES").alias("hard_H8_status"),
            pl.lit("GROUND_TRUTH_ONLY_OUTCOME_AND_SAMPLING").alias("query_label_provenance"),
            pl.lit("REFERENCE_DERIVED").alias("reference_activity_provenance"),
            pl.lit("QUERY_REFERENCE_DERIVED").alias("retrieval_feature_provenance"),
        )
    )
    hard_columns = [name for name in result.columns if name.startswith("hard_H") and name != "hard_H8_status"]
    return result.with_columns(pl.any_horizontal([pl.col(name) for name in hard_columns]).alias("is_difficult_case"))


def deterministic_bernoulli(frame: pl.DataFrame, probability: float, salt: str) -> pl.DataFrame:
    threshold = int(min(max(probability, 0.0), 1.0) * (2**64 - 1))
    return frame.filter(
        pl.concat_str([pl.col(column) for column in PAIR_KEY] + [pl.lit(salt)], separator="|")
        .hash(seed=SEED).cast(pl.UInt64) <= threshold
    )


def build_matching_balance(training: pl.DataFrame, reports: Path) -> tuple[pd.DataFrame, int]:
    columns = [
        *PAIR_KEY, "is_difficult_case", "same_exact_rhea", "mmseqs_fident", "mmseqs_qcov", "mmseqs_tcov",
        "query_length", "reference_length", "query_length_bin", "query_taxonomy_group", "same_taxonomy_group",
        "query_primary_pfam", "same_pfam_family", "ec_l3", "query_has_structure", "reference_has_structure",
        "query_fragment_status",
    ]
    data = training.select(columns).to_pandas()
    data["sequence_identity_bin"] = pd.cut(
        data["mmseqs_fident"], [-np.inf, 0.2, 0.3, 0.4, 0.6, 0.8, np.inf], include_lowest=True
    ).astype(str)
    data["alignment_coverage"] = data[["mmseqs_qcov", "mmseqs_tcov"]].min(axis=1)
    data["coverage_bin"] = pd.cut(data["alignment_coverage"], [-np.inf, 0.3, 0.5, 0.7, 0.9, np.inf]).astype(str)
    data["both_structure"] = data["query_has_structure"] & data["reference_has_structure"]
    data["match_stratum"] = data[
        ["sequence_identity_bin", "coverage_bin", "query_length_bin", "query_taxonomy_group", "ec_l3",
         "same_pfam_family", "both_structure", "query_fragment_status"]
    ].fillna("MISSING").astype(str).agg("|".join, axis=1)
    treated = data[data["is_difficult_case"]].copy()
    controls = data[data["same_exact_rhea"] & ~data["is_difficult_case"]].copy()
    treated["_hash"] = treated.apply(lambda row: stable_u64("T|" + "|".join(str(row[k]) for k in PAIR_KEY)), axis=1)
    controls["_hash"] = controls.apply(lambda row: stable_u64("C|" + "|".join(str(row[k]) for k in PAIR_KEY)), axis=1)
    matched_t, matched_c = [], []
    control_groups = {key: group.sort_values("_hash") for key, group in controls.groupby("match_stratum", sort=False)}
    total_cap = 50_000
    matched_total = 0
    for key, group in treated.groupby("match_stratum", sort=False):
        control = control_groups.get(key)
        if control is None:
            continue
        n = min(len(group), len(control), total_cap - matched_total)
        if n <= 0:
            break
        matched_t.append(group.sort_values("_hash").head(n))
        matched_c.append(control.head(n))
        matched_total += n
    mt = pd.concat(matched_t, ignore_index=True) if matched_t else treated.head(0)
    mc = pd.concat(matched_c, ignore_index=True) if matched_c else controls.head(0)
    rows: list[dict] = []
    for variable in ["mmseqs_fident", "alignment_coverage", "query_length", "reference_length"]:
        smd, tmean, cmean, nt, nc = numeric_smd(mt[variable], mc[variable])
        rows.append({"variable": variable, "type": "numeric", "smd": smd, "treated_summary": tmean,
                     "control_summary": cmean, "treated_n": nt, "control_n": nc, "worst_level": ""})
    for variable in [
        "sequence_identity_bin", "coverage_bin", "query_length_bin", "query_taxonomy_group", "ec_l3",
        "same_pfam_family", "both_structure", "query_fragment_status",
    ]:
        smd, level, nt, nc = categorical_max_smd(mt[variable], mc[variable])
        rows.append({"variable": variable, "type": "categorical_max_binary", "smd": smd,
                     "treated_summary": "", "control_summary": "", "treated_n": nt, "control_n": nc,
                     "worst_level": level})
    result = pd.DataFrame(rows)
    result.to_csv(reports / "matching_balance.tsv", sep="\t", index=False)
    return result, len(mt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data/interim/phase05"
    processed = root / "data/processed"
    reports = root / "reports"
    checkpoints = root / "checkpoints"
    for directory in [work, processed, reports, checkpoints]:
        directory.mkdir(parents=True, exist_ok=True)
    if not (checkpoints / "CHECKPOINT_04_PASS").is_file():
        raise RuntimeError("CHECKPOINT_04_PASS is required before Phase 5")
    config = yaml.safe_load((root / "configs/pair_sampling_v4.yaml").read_text(encoding="utf-8"))
    seed = int(config["seed"])
    if seed != SEED:
        raise RuntimeError(f"Seed mismatch: config={seed}, code={SEED}")

    retrieval = build_retrieval_universe(root, work)
    meta, activities = metadata_frames(root)
    pmeta, pactivities = polars_metadata(meta, activities)
    augmentation_path = write_augmentations(root, work, meta, activities, config)

    retrieval_instances = retrieval.join(
        pactivities.select(["reference_activity_id", "reference_protein_id"]),
        on="reference_protein_id", how="inner", validate="m:m",
    ).with_columns(
        pl.lit("retrieval_top50_union").alias("candidate_origin"),
        pl.lit(1.0).alias("candidate_sampling_probability"),
        pl.lit("").alias("augmentation_source"),
    )
    population_universe_n = retrieval_instances.height
    population_target = int(config["population_atlas"]["target_pairs"])
    population_probability = min(1.0, population_target / population_universe_n)
    population = deterministic_bernoulli(retrieval_instances, population_probability, "population")
    population = enrich_instances(population, pmeta, pactivities).with_columns(
        pl.lit("population_atlas").alias("pair_set"),
        pl.lit(population_probability).alias("sampling_probability"),
        pl.lit(1.0 / population_probability).alias("sample_weight"),
        pl.lit("SRS_WITHOUT_LABEL_BALANCING_FROM_RETRIEVAL_TOP50_ACTIVITY_UNIVERSE").alias("sampling_design"),
    )
    population.write_parquet(processed / "population_pairs.parquet", compression="zstd")

    augmentation = pl.read_parquet(augmentation_path)
    augmentation = augmentation.join(
        pactivities.select(["reference_activity_id", "reference_protein_id"]),
        on=["reference_activity_id", "reference_protein_id"], how="inner", validate="m:1",
    ).with_columns(
        pl.lit("supervised_augmentation").alias("candidate_origin"),
        pl.lit(None, dtype=pl.String).alias("query_partition"),
        pl.lit(None, dtype=pl.String).alias("reference_cluster_id_30"),
        pl.lit(None, dtype=pl.Int16).alias("mmseqs_rank"), pl.lit(None, dtype=pl.Float64).alias("mmseqs_fident"),
        pl.lit(None, dtype=pl.Float64).alias("mmseqs_qcov"), pl.lit(None, dtype=pl.Float64).alias("mmseqs_tcov"),
        pl.lit(None, dtype=pl.Float64).alias("mmseqs_bits"), pl.lit(None, dtype=pl.Int16).alias("foldseek_rank"),
        pl.lit(None, dtype=pl.Float64).alias("foldseek_fident"), pl.lit(None, dtype=pl.Float64).alias("foldseek_qcov"),
        pl.lit(None, dtype=pl.Float64).alias("foldseek_tcov"), pl.lit(None, dtype=pl.Float64).alias("foldseek_bits"),
        pl.lit("augmentation").alias("candidate_source_class"), pl.lit(None, dtype=pl.Int16).alias("best_retrieval_rank"),
        pl.lit("augmentation").alias("retrieval_rank_bin"),
    )
    base_training = population.drop([column for column in population.columns if column not in retrieval_instances.columns], strict=False)
    common = sorted(set(base_training.columns).union(augmentation.columns))
    for column in common:
        if column not in base_training.columns:
            base_training = base_training.with_columns(pl.lit(None).alias(column))
        if column not in augmentation.columns:
            augmentation = augmentation.with_columns(pl.lit(None).alias(column))
    training_pool = (
        pl.concat([base_training.select(common), augmentation.select(common)], how="diagonal_relaxed")
        .sort([*PAIR_KEY, "candidate_origin"])
        .unique(subset=PAIR_KEY, keep="first")
    )
    training_pool = enrich_instances(training_pool, pmeta, pactivities)
    family_frequency = (
        pmeta.filter(pl.col("split") == "train").group_by("primary_pfam").len()
        .rename({"primary_pfam": "reference_primary_pfam", "len": "family_frequency"})
    )
    training_pool = training_pool.join(family_frequency, on="reference_primary_pfam", how="left").with_columns(
        pl.col("family_frequency").fill_null(1),
    )
    # Independent Bernoulli PPS: rare activities/families and difficult cases receive more inclusion mass.
    training_pool = training_pool.with_columns(
        ((1.0 / pl.col("activity_frequency").cast(pl.Float64).sqrt())
         * (1.0 / pl.col("family_frequency").cast(pl.Float64).sqrt())
         * pl.when(pl.col("is_difficult_case")).then(3.0).otherwise(1.0)
         * pl.when(pl.col("observed_concordance_depth") > 0).then(1.5).otherwise(1.0))
        .alias("balance_score")
    )
    target = int(config["balanced_training"]["target_pairs"])
    weights = training_pool["balance_score"].to_numpy()
    low, high = 0.0, max(1.0, target / max(weights.sum(), 1e-12) * 10.0)
    while np.minimum(1.0, high * weights).sum() < target:
        high *= 2.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if np.minimum(1.0, middle * weights).sum() < target:
            low = middle
        else:
            high = middle
    probabilities = np.minimum(1.0, high * weights)
    training_pool = training_pool.with_columns(pl.Series("sampling_probability", probabilities))
    hashes = training_pool.select(
        pl.concat_str([pl.col(column) for column in PAIR_KEY] + [pl.lit("training")], separator="|")
        .hash(seed=SEED).cast(pl.UInt64).alias("_hash")
    )["_hash"].to_numpy()
    uniforms = hashes.astype(np.float64) / float(2**64 - 1)
    training = training_pool.with_columns(pl.Series("_hash", hashes)).filter(pl.Series(uniforms <= probabilities)).with_columns(
        (1.0 / pl.col("sampling_probability")).alias("sample_weight"),
        pl.lit("balanced_training").alias("pair_set"),
        pl.lit("INDEPENDENT_PPS_ACTIVITY_FAMILY_HARD_CASE_BALANCED").alias("sampling_design"),
    )
    # Deterministic cap guard if Bernoulli variation exceeds the registered combined limit.
    max_training = int(config["size_guard"]["maximum_pairs_combined"]) - population.height
    if training.height > max_training:
        training = training.sort("_hash").head(max_training)
    training = training.drop("_hash", strict=False)
    training.write_parquet(processed / "training_pairs.parquet", compression="zstd")
    difficult = training.filter(pl.col("is_difficult_case"))
    difficult.write_parquet(processed / "difficult_cases.parquet", compression="zstd")

    balance, matched_n = build_matching_balance(training, reports)
    manifest_rows = [
        ("retrieval_protein_pair_universe", retrieval.height, 1.0, "Top50 MMseqs union Top50 Foldseek; all splits to train refs"),
        ("retrieval_activity_instance_universe", population_universe_n, population_probability, "Protein pairs expanded by train reference activities"),
        ("population_pairs", population.height, population_probability, "Label-blind SRS hash Bernoulli"),
        ("supervised_augmentation_candidates", augmentation.height, float(augmentation["candidate_sampling_probability"].mean()), "Registered positive/family/control samplers"),
        ("training_candidate_pool", training_pool.height, float(training_pool["sampling_probability"].mean()), "Population base plus supervised augmentation"),
        ("training_pairs", training.height, float(training["sampling_probability"].mean()), "Activity/family/hard-case PPS"),
        ("difficult_cases", difficult.height, 1.0, "H1-H7 observed/provisional; H8 deferred to Phase 7"),
    ]
    sampling_manifest = pd.DataFrame(manifest_rows, columns=["stage", "rows", "mean_sampling_probability", "definition"])
    sampling_manifest.to_csv(reports / "sampling_manifest.tsv", sep="\t", index=False)

    minimum = int(config["size_guard"]["minimum_pairs_each"])
    maximum = int(config["size_guard"]["maximum_pairs_combined"])
    output_paths = [
        processed / "population_pairs.parquet", processed / "training_pairs.parquet",
        processed / "difficult_cases.parquet", reports / "matching_balance.tsv", reports / "sampling_manifest.tsv",
    ]
    checks = [
        ("checkpoint_04_present", (checkpoints / "CHECKPOINT_04_PASS").is_file(), "strict phase gate"),
        ("population_minimum_size", population.height >= minimum, str(population.height)),
        ("training_minimum_size", training.height >= minimum, str(training.height)),
        ("combined_size_guard", population.height + training.height <= maximum, str(population.height + training.height)),
        ("population_unique_instances", population.select(pl.struct(PAIR_KEY).n_unique()).item() == population.height, str(population.height)),
        ("training_unique_instances", training.select(pl.struct(PAIR_KEY).n_unique()).item() == training.height, str(training.height)),
        ("no_self_pairs", population.filter(pl.col("query_protein_id") == pl.col("reference_protein_id")).height == 0 and training.filter(pl.col("query_protein_id") == pl.col("reference_protein_id")).height == 0, "population+training"),
        ("all_references_train", population.filter(pl.col("reference_partition_verified") != "train").height == 0 and training.filter(pl.col("reference_partition_verified") != "train").height == 0, "reference_partition_verified"),
        ("sampling_probabilities_valid", population.filter((pl.col("sampling_probability") <= 0) | (pl.col("sampling_probability") > 1)).height == 0 and training.filter((pl.col("sampling_probability") <= 0) | (pl.col("sampling_probability") > 1)).height == 0, "0 < p <= 1"),
        ("population_label_blind_sampling", population["sampling_design"].n_unique() == 1 and population["sampling_design"][0].startswith("SRS_WITHOUT_LABEL_BALANCING"), "registered population design"),
        ("difficult_cases_nonempty", difficult.height > 0, str(difficult.height)),
        ("matching_pairs_nonempty", matched_n > 0, str(matched_n)),
        ("matching_balance_written", len(balance) >= 8, str(len(balance))),
        ("h8_explicitly_deferred", difficult["hard_H8_status"].n_unique() == 1 and difficult["hard_H8_status"][0] == "PENDING_PHASE_07_LOCAL_FEATURES", "no premature local claim"),
        ("ground_truth_provenance_explicit", population["query_label_provenance"].n_unique() == 1 and training["query_label_provenance"].n_unique() == 1, "GROUND_TRUTH_ONLY outcome/sampling"),
        ("all_outputs_exist", all(path.is_file() and path.stat().st_size > 0 for path in output_paths), "five required outputs"),
    ]
    qc = pd.DataFrame([(name, "PASS" if passed else "FAIL", details) for name, passed, details in checks], columns=["check", "status", "details"])
    qc.to_csv(reports / "phase05_qc.tsv", sep="\t", index=False)
    failures = qc[qc["status"] == "FAIL"]
    summary = {
        "phase": 5, "status": "PASS" if failures.empty else "FAIL", "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "seed": SEED, "retrieval_protein_pairs": retrieval.height,
        "retrieval_activity_instance_universe": population_universe_n,
        "population_pairs": population.height, "training_pairs": training.height,
        "difficult_cases": difficult.height, "matched_pairs_per_arm": matched_n,
        "population_exact_rhea_rate": float(population["same_exact_rhea"].mean()),
        "training_exact_rhea_rate": float(training["same_exact_rhea"].mean()),
        "hard_case_counts": {column: int(training[column].sum()) for column in training.columns if column.startswith("hard_H") and column != "hard_H8_status"},
        "qc_failures": failures["check"].tolist(),
    }
    (reports / "phase05_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# SiteGuard V4 Phase 5 Report", "", f"Status: **{summary['status']}**", "",
        "## Pair universes", "",
        f"- Retrieval protein-pair universe: {retrieval.height:,}",
        f"- Retrieval activity-transfer universe: {population_universe_n:,}",
        f"- Population/atlas pairs: {population.height:,} (label-blind SRS; IPW retained)",
        f"- Balanced training pairs: {training.height:,} (activity/family/hard-case PPS)",
        f"- Difficult cases: {difficult.height:,}", "",
        "The population set is not class-balanced and is the only Phase 5 set suitable for estimating annotation-concordance prevalence. The balanced set is for model fitting only.", "",
        "H4 and H7 are provisional structure-retrieval/cluster categories pending exact global structure features in Phase 6. H8 is intentionally deferred until mapped local features exist in Phase 7.", "",
        "Query EC/Rhea fields are retained only as GROUND_TRUTH_ONLY outcomes and sampling strata. They are prohibited from later feature matrices.", "",
        "## QC", "", qc.to_markdown(index=False), "",
    ]
    (reports / "PHASE_05_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    if not failures.empty:
        raise RuntimeError("Phase 5 QC failed: " + ", ".join(failures["check"]))
    checkpoint = checkpoints / "CHECKPOINT_05_PASS"
    checkpoint.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
