#!/usr/bin/env python3
"""Build role-isolated SiteGuard feature tables without reading non-TRAIN outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path

import numpy as np


ROOT = Path("/globalsc/ulg/plgen/yugenliu/SiteGuard/V4")
ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")
KEY = ["query_protein_id", "reference_protein_id", "reference_activity_id"]
REACTION_NUMERIC = [
    "smiles_parse_valid", "substrate_component_count", "product_component_count",
    "participant_count", "unique_participant_count", "currency_participant_count",
    "currency_fraction", "substrate_mw_sum", "product_mw_sum", "substrate_logp_sum",
    "product_logp_sum", "substrate_tpsa_sum", "product_tpsa_sum",
    "substrate_heavy_atoms", "product_heavy_atoms", "heavy_atom_change",
    "reaction_complexity",
]
NUMERIC = [
    "sequence_identity", "sequence_query_coverage", "sequence_reference_coverage",
    "sequence_bitscore_log1p", "sequence_alignment_fraction", "length_ratio",
    "aa_composition_cosine", "esm2_t33_cosine", "foldseek_identity",
    "foldseek_query_coverage", "foldseek_reference_coverage",
    "foldseek_bitscore_log1p", "foldseek_alignment_fraction",
    "both_structures_available", "pfam_jaccard", "primary_pfam_match",
    "primary_pfam_clan_match", "cath_jaccard", "primary_cath_match",
] + ["reference_" + name for name in REACTION_NUMERIC] + ["reference_reaction_available"]
CATEGORICAL = ["reference_ec_l1", "reference_cofactor_class"]
META = KEY + [
    "query_role", "query_node", "reference_node", "query_component_id",
    "reference_component_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3",
    "ec_l4", "canonical_rhea", "evidence_tier", "mmseqs_rank", "foldseek_rank",
    "source_class", "rrf60_score",
]
TARGET = KEY + [
    "observed_same_ec_l3", "observed_same_ec_l4", "exact_rhea_outcome_evaluable",
    "observed_same_exact_rhea", "sample_weight",
]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def json_set(value) -> frozenset[str]:
    if value is None:
        return frozenset()
    try:
        parsed = json.loads(str(value))
    except (ValueError, TypeError):
        return frozenset()
    if not isinstance(parsed, list):
        return frozenset()
    values = []
    for item in parsed:
        if isinstance(item, dict):
            item = item.get("id")
        if item:
            values.append(str(item))
    return frozenset(values)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def composition(sequence: str) -> np.ndarray:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    result = np.fromiter((sequence.count(aa) for aa in alphabet), dtype=np.float32)
    total = float(result.sum())
    return result / total if total else result


def protein_assets(root: Path):
    import pandas as pd
    import pyarrow.parquet as pq

    benchmark = pq.read_table(
        root / "data/interim/phase03/benchmark_proteins.parquet",
        columns=["protein_id", "sequence", "pfam_domains_json"],
    ).to_pandas()
    family = pq.read_table(
        root / "data/splits/split_family.parquet",
        columns=["protein_id", "primary_pfam", "primary_pfam_clan"],
    ).to_pandas()
    structure = pq.read_table(
        root / "data/splits/split_structure.parquet",
        columns=["protein_id", "cath_superfamilies_json", "primary_cath_superfamily"],
    ).to_pandas()
    membership = pq.read_table(
        root / "data/interim/phase04/afdb_membership.parquet",
        columns=["protein_id", "has_structure"],
    ).to_pandas()
    roles = pq.read_table(
        root / "revisions/s4_20260910/truth_free_ledger_attempt_01/producer/protein_role_ledger.parquet",
        columns=["protein_id", "node_id", "component_id", "role"],
    ).to_pandas()
    frame = benchmark.merge(family, on="protein_id", validate="one_to_one")
    frame = frame.merge(structure, on="protein_id", validate="one_to_one")
    frame = frame.merge(membership, on="protein_id", validate="one_to_one")
    frame = frame.merge(roles, on="protein_id", validate="one_to_one").sort_values("protein_id").reset_index(drop=True)
    require(len(frame) == 210788 and frame["protein_id"].is_unique, "protein universe")
    index = pd.read_csv(root / "data/interim/phase06/esm2_t33_v4_index.tsv", sep="\t", dtype={"protein_id": str})
    require({"protein_id", "row_index"} <= set(index.columns) and index["protein_id"].is_unique, "ESM index")
    embedding_row = index.set_index("protein_id")["row_index"]
    frame["embedding_row"] = frame["protein_id"].map(embedding_row)
    require(not frame["embedding_row"].isna().any(), "ESM coverage")
    embedding = np.load(root / "data/interim/phase06/esm2_t33_v4_embeddings.npy", mmap_mode="r")
    require(embedding.shape == (210788, 1280), "ESM matrix identity")
    sequences = frame["sequence"].astype(str).tolist()
    assets = {
        "ids": frame["protein_id"].tolist(),
        "row": {value: index for index, value in enumerate(frame["protein_id"])},
        "node_role": frame[["protein_id", "node_id", "component_id", "role"]],
        "length": np.asarray([len(value) for value in sequences], dtype=np.float32),
        "composition": np.stack([composition(value) for value in sequences]),
        "pfam": [json_set(value) for value in frame["pfam_domains_json"]],
        "primary_pfam": frame["primary_pfam"].where(frame["primary_pfam"].notna(), None).tolist(),
        "primary_clan": frame["primary_pfam_clan"].where(frame["primary_pfam_clan"].notna(), None).tolist(),
        "cath": [json_set(value) for value in frame["cath_superfamilies_json"]],
        "primary_cath": frame["primary_cath_superfamily"].where(frame["primary_cath_superfamily"].notna(), None).tolist(),
        "has_structure": frame["has_structure"].astype(bool).to_numpy(),
        "embedding_row": frame["embedding_row"].astype(np.int64).to_numpy(),
        "embedding": embedding,
    }
    return assets


def reaction_assets(root: Path) -> dict:
    import pyarrow.parquet as pq

    columns = ["canonical_rhea", *REACTION_NUMERIC, "cofactor_class"]
    rows = pq.read_table(root / "data/processed/reaction_features.parquet", columns=columns).to_pylist()
    result = {}
    for row in rows:
        key = row.pop("canonical_rhea")
        require(isinstance(key, str) and key and key not in result, "reaction feature key")
        result[key] = row
    return result


def build_base(root: Path, role: str, work: Path) -> Path:
    import polars as pl

    base = work / "role_pair_base.parquet"
    if role == "TRAIN":
        source = root / "revisions/s4_20260910/balanced_train_sampler_attempt_01/producer/balanced_train_pairs.parquet"
        plan = root / "revisions/s4_20260910/direct_pair_measurement_plan_attempt_01/producer/direct_pair_measurement_manifest.parquet"
        mm = root / "revisions/s4_20260910/uniform_direct_alignment_attempt_01/reducer/direct_mmseqs_node_alignments.parquet"
        fs = root / "revisions/s4_20260910/uniform_direct_alignment_attempt_01/reducer/direct_foldseek_node_alignments.parquet"
        for path in (source, plan, mm, fs):
            require(path.is_file(), "TRAIN feature upstream " + str(path))
        plan_lf = pl.scan_parquet(plan).select(KEY[:2] + ["query_node", "reference_node", "structure_availability"])
        mm_lf = pl.scan_parquet(mm).select(
            "query_node", "reference_node",
            *[pl.col(name).alias("mmseqs_" + name) for name in ("fident", "qcov", "tcov", "bits")],
        )
        fs_lf = pl.scan_parquet(fs).select(
            "query_node", "reference_node",
            *[pl.col(name).alias("foldseek_" + name) for name in ("fident", "qcov", "tcov", "bits")],
        )
        lf = (
            pl.scan_parquet(source)
            .join(plan_lf, on=KEY[:2], how="left", validate="m:1", suffix="_plan")
            .join(mm_lf, on=["query_node", "reference_node"], how="left", validate="m:1", suffix="_direct")
            .join(fs_lf, on=["query_node", "reference_node"], how="left", validate="m:1", suffix="_direct")
            .with_columns(
                *[pl.col("mmseqs_" + name + "_direct").alias("mmseqs_" + name) for name in ("fident", "qcov", "tcov", "bits")],
                *[pl.col("foldseek_" + name + "_direct").alias("foldseek_" + name) for name in ("fident", "qcov", "tcov", "bits")],
            )
        )
    else:
        source = root / f"revisions/s4_20260910/activity_expansion_attempt_01/expansion/expansion_{role}/activity_candidates.parquet"
        ledger = root / "revisions/s4_20260910/truth_free_ledger_attempt_01/producer/protein_role_ledger.parquet"
        require(source.is_file() and ledger.is_file(), "evaluation feature upstream")
        query_map = (
            pl.scan_parquet(ledger)
            .filter(pl.col("role") == role)
            .select(
                "node_id", pl.col("protein_id").alias("query_protein_id"),
                pl.col("component_id").alias("query_component_id"),
            )
        )
        lf = (
            pl.scan_parquet(source)
            .join(query_map, left_on="query_node", right_on="node_id", how="inner", validate="m:m")
        )
    frame = lf.sort(KEY).collect(engine="streaming")
    require(frame.height > 0, "empty role pair base")
    frame.write_parquet(base, compression="zstd")
    return base


def arrow_schemas():
    import pyarrow as pa

    features = pa.schema([(name, pa.string()) for name in KEY] + [(name, pa.float32()) for name in NUMERIC] + [(name, pa.string()) for name in CATEGORICAL])
    metadata_types = {
        **{name: pa.string() for name in KEY + ["query_role", "canonical_ec", "ec_l1", "ec_l2", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier", "source_class"]},
        **{name: pa.int32() for name in ["query_node", "reference_node", "query_component_id", "reference_component_id"]},
        "mmseqs_rank": pa.int32(), "foldseek_rank": pa.int32(), "rrf60_score": pa.float64(),
    }
    metadata = pa.schema([(name, metadata_types[name]) for name in META])
    targets = pa.schema([(name, pa.string()) for name in KEY] + [
        ("observed_same_ec_l3", pa.bool_()), ("observed_same_ec_l4", pa.bool_()),
        ("exact_rhea_outcome_evaluable", pa.bool_()), ("observed_same_exact_rhea", pa.bool_()),
        ("sample_weight", pa.float32()),
    ])
    return features, metadata, targets


class Sink:
    def __init__(self, path: Path, schema):
        import pyarrow.parquet as pq
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.schema = schema
        self.count = 0

    def write(self, rows):
        import pyarrow as pa
        if rows:
            self.writer.write_table(pa.Table.from_pylist(rows, schema=self.schema))
            self.count += len(rows)

    def close(self):
        self.writer.close()


def number(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def dense_rows(rows, assets, reactions):
    qidx = np.asarray([assets["row"].get(row["query_protein_id"], -1) for row in rows], dtype=np.int64)
    ridx = np.asarray([assets["row"].get(row["reference_protein_id"], -1) for row in rows], dtype=np.int64)
    require(np.all(qidx >= 0) and np.all(ridx >= 0), "pair protein metadata coverage")
    qe = np.asarray(assets["embedding"][assets["embedding_row"][qidx]], dtype=np.float32)
    re = np.asarray(assets["embedding"][assets["embedding_row"][ridx]], dtype=np.float32)
    denominator = np.linalg.norm(qe, axis=1) * np.linalg.norm(re, axis=1)
    esm = np.divide(np.einsum("ij,ij->i", qe, re), denominator, out=np.zeros(len(rows), dtype=np.float32), where=denominator > 0)
    qc, rc = assets["composition"][qidx], assets["composition"][ridx]
    denominator = np.linalg.norm(qc, axis=1) * np.linalg.norm(rc, axis=1)
    comp = np.divide(np.einsum("ij,ij->i", qc, rc), denominator, out=np.zeros(len(rows), dtype=np.float32), where=denominator > 0)
    output = []
    for pos, row in enumerate(rows):
        q, r = qidx[pos], ridx[pos]
        sequence_bits = number(row.get("mmseqs_bits"))
        sequence_qcov = number(row.get("mmseqs_qcov"))
        sequence_tcov = number(row.get("mmseqs_tcov"))
        fold_bits = number(row.get("foldseek_bits"))
        fold_qcov = number(row.get("foldseek_qcov"))
        fold_tcov = number(row.get("foldseek_tcov"))
        rhea = row.get("canonical_rhea")
        reaction = reactions.get(rhea, {}) if rhea else {}
        feature = {
            **{name: row[name] for name in KEY},
            "sequence_identity": number(row.get("mmseqs_fident")),
            "sequence_query_coverage": sequence_qcov,
            "sequence_reference_coverage": sequence_tcov,
            "sequence_bitscore_log1p": math.log1p(max(sequence_bits, 0.0)) if sequence_bits is not None else None,
            "sequence_alignment_fraction": min(sequence_qcov, sequence_tcov) if sequence_qcov is not None and sequence_tcov is not None else None,
            "length_ratio": float(min(assets["length"][q], assets["length"][r]) / max(assets["length"][q], assets["length"][r])),
            "aa_composition_cosine": float(comp[pos]), "esm2_t33_cosine": float(esm[pos]),
            "foldseek_identity": number(row.get("foldseek_fident")),
            "foldseek_query_coverage": fold_qcov, "foldseek_reference_coverage": fold_tcov,
            "foldseek_bitscore_log1p": math.log1p(max(fold_bits, 0.0)) if fold_bits is not None else None,
            "foldseek_alignment_fraction": min(fold_qcov, fold_tcov) if fold_qcov is not None and fold_tcov is not None else None,
            "both_structures_available": float(fold_bits is not None),
            "pfam_jaccard": jaccard(assets["pfam"][q], assets["pfam"][r]),
            "primary_pfam_match": float(assets["primary_pfam"][q] is not None and assets["primary_pfam"][q] == assets["primary_pfam"][r]),
            "primary_pfam_clan_match": float(assets["primary_clan"][q] is not None and assets["primary_clan"][q] == assets["primary_clan"][r]),
            "cath_jaccard": jaccard(assets["cath"][q], assets["cath"][r]),
            "primary_cath_match": float(assets["primary_cath"][q] is not None and assets["primary_cath"][q] == assets["primary_cath"][r]),
            "reference_ec_l1": row.get("ec_l1"),
            "reference_cofactor_class": reaction.get("cofactor_class"),
            "reference_reaction_available": float(bool(reaction)),
        }
        for name in REACTION_NUMERIC:
            feature["reference_" + name] = number(reaction.get(name))
        output.append(feature)
    return output


def execute(root: Path, role: str, output: Path, batch_rows: int):
    import pyarrow.parquet as pq

    require(role in ROLES and output.parent.is_dir() and not output.exists(), "role/output")
    output.mkdir(exist_ok=False)
    emit(output / "reservation.json", {"status": "ONE_S4N_ROLE_FEATURE_BUILD_RESERVED", "role": role, "nontrain_truth_read": False, "automatic_retry": False})
    state, error = "FAIL_CLOSED", None
    feature_sink = metadata_sink = target_sink = None
    try:
        base = build_base(root, role, output)
        assets, reactions = protein_assets(root), reaction_assets(root)
        f_schema, m_schema, t_schema = arrow_schemas()
        feature_sink = Sink(output / f"features_{role}.parquet", f_schema)
        metadata_sink = Sink(output / f"metadata_{role}.parquet", m_schema)
        if role == "TRAIN":
            target_sink = Sink(output / "targets_TRAIN.parquet", t_schema)
        parquet = pq.ParquetFile(base)
        seen, previous = set(), None
        for batch in parquet.iter_batches(batch_size=batch_rows):
            rows = batch.to_pylist()
            features = dense_rows(rows, assets, reactions)
            metadata, targets = [], []
            for row in rows:
                key = tuple(row[name] for name in KEY)
                require(previous is None or key > previous, "strict role pair ordering")
                previous = key
                metadata.append({name: row.get(name) for name in META})
                if role == "TRAIN":
                    require(row.get("query_role") == "TRAIN" and row.get("sampling_applied") is True, "TRAIN scope")
                    targets.append({name: row.get(name) for name in TARGET})
                else:
                    require(row.get("query_role") == role and row.get("query_truth_read") is False, "nontrain truth-free scope")
            feature_sink.write(features); metadata_sink.write(metadata)
            if target_sink is not None:
                target_sink.write(targets)
        feature_sink.close(); metadata_sink.close()
        if target_sink is not None:
            target_sink.close()
        require(feature_sink.count == metadata_sink.count > 0 and (target_sink is None or target_sink.count == feature_sink.count), "role row reconciliation")
        files = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)} for path in output.glob("*.parquet")}
        summary = {
            "status": "PASS_S4N_ROLE_FEATURES", "role": role, "rows": feature_sink.count,
            "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
            "nontrain_truth_read": False, "old_split_columns_read": False,
            "query_ground_truth_in_features": False, "reference_features_only": True,
            "files": files,
        }
        emit(output / "summary.json", summary)
        state = summary["status"]
    except BaseException as exc:
        for sink in (feature_sink, metadata_sink, target_sink):
            if sink is not None:
                try: sink.close()
                except BaseException: pass
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    emit(output / "terminal.json", {"status": state, "error": error, "role": role, "automatic_retry": False})
    if error:
        raise RuntimeError(error["message"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=50000)
    args = parser.parse_args()
    execute(args.root.resolve(), args.role, args.output.resolve(), args.batch_rows)


if __name__ == "__main__":
    main()
