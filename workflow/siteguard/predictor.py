"""Sequence-to-candidate SiteGuard inference using frozen project assets."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .assets import validate_asset_root
from .calibration import load_frozen_calibrator
from .decision import LEVELS, decide_highest_supported_resolution
from .fasta import read_fasta, sequence_sha256
from .model import blend_scores, network_class


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
LABEL_COLUMNS = {"EC_L3": "ec_l3", "EC_L4": "ec_l4", "EXACT_RHEA": "canonical_rhea"}


def composition(sequence: str) -> np.ndarray:
    values = np.fromiter((sequence.count(amino_acid) for amino_acid in AMINO_ACIDS), dtype=np.float32)
    return values / values.sum() if values.sum() else values


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator else 0.0


def json_ids(value: object) -> frozenset[str]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return frozenset()
    output = []
    for item in parsed if isinstance(parsed, list) else []:
        if isinstance(item, dict) and item.get("id"):
            output.append(str(item["id"]))
        elif item:
            output.append(str(item))
    return frozenset(output)


def _query_pfam(path: str | Path | None) -> dict[str, frozenset[str]]:
    if path is None:
        return {}
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if not {"protein_id", "pfam_ids"}.issubset(frame.columns):
        raise ValueError("Query Pfam TSV must contain protein_id and pfam_ids columns")
    return {
        row.protein_id: frozenset(value for value in str(row.pfam_ids).split(";") if value)
        for row in frame.itertuples(index=False)
    }


def _run_mmseqs(fasta: Path, root: Path, work: Path, executable: str, threads: int) -> pd.DataFrame:
    output = work / "mmseqs_hits.tsv"
    temporary = work / "mmseqs_tmp"
    command = [
        executable, "easy-search", str(fasta), str(root / "databases/mmseqs/reference_seq"), str(output), str(temporary),
        "--format-output", "query,target,fident,qcov,tcov,bits,evalue,alnlen", "--max-seqs", "50",
        "-s", "7.5", "-e", "1000", "--threads", str(threads),
    ]
    subprocess.run(command, check=True)
    names = ["query_protein_id", "reference_protein_id", "fident", "qcov", "tcov", "bits", "evalue", "alnlen"]
    hits = pd.read_csv(output, sep="\t", names=names)
    hits = hits.loc[hits["query_protein_id"].ne(hits["reference_protein_id"])].copy()
    for column in ["fident", "qcov", "tcov", "bits", "evalue"]:
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    for column in ["fident", "qcov", "tcov"]:
        if hits[column].max() > 1:
            hits[column] /= 100.0
    hits = hits.sort_values(
        ["query_protein_id", "bits", "reference_protein_id"], ascending=[True, False, True],
    ).drop_duplicates(["query_protein_id", "reference_protein_id"])
    hits["retrieval_rank"] = hits.groupby("query_protein_id").cumcount() + 1
    return hits.loc[hits["retrieval_rank"].le(50)].reset_index(drop=True)


def _build_pairs(
    root: Path, records: dict[str, str], query_embeddings: np.ndarray,
    embedding_rows: dict[str, int], hits: pd.DataFrame, pfam: dict[str, frozenset[str]],
) -> pd.DataFrame:
    reference_ids = set(hits["reference_protein_id"].astype(str))
    protein = pd.read_parquet(
        root / "data/processed/protein_table.parquet",
        columns=["protein_id", "sequence", "length", "pfam_domains_json"],
    )
    protein = protein.loc[protein["protein_id"].isin(reference_ids)]
    family = pd.read_parquet(
        root / "data/splits/split_family.parquet",
        columns=["protein_id", "primary_pfam", "primary_pfam_clan"],
    )
    split = pd.read_parquet(
        root / "data/splits/split_sequence.parquet", columns=["protein_id", "cluster_id_30", "split"],
    )
    structures = pd.read_parquet(
        root / "data/interim/phase04/afdb_membership.parquet", columns=["protein_id", "has_structure"],
    )
    reference = protein.merge(family, on="protein_id", how="left", validate="one_to_one")
    reference = reference.merge(split, on="protein_id", how="left", validate="one_to_one")
    reference = reference.merge(structures, on="protein_id", how="left", validate="one_to_one")
    if not reference["split"].eq("train").all():
        raise RuntimeError("Reference database includes non-training proteins")
    reference["pfam_set"] = reference["pfam_domains_json"].map(json_ids)
    reference["composition"] = reference["sequence"].map(composition)
    reference_lookup = reference.set_index("protein_id")
    reference_matrix = np.load(root / "data/interim/phase06/esm2_t33_v4_embeddings.npy", mmap_mode="r")
    reference_index = pd.read_csv(root / "data/interim/phase06/esm2_t33_v4_index.tsv", sep="\t")
    reference_rows = reference_index.set_index("protein_id")["row_index"].to_dict()
    rows = []
    for hit in hits.itertuples(index=False):
        query_id, reference_id = str(hit.query_protein_id), str(hit.reference_protein_id)
        if reference_id not in reference_lookup.index or reference_id not in reference_rows:
            continue
        query_sequence, reference_row = records[query_id], reference_lookup.loc[reference_id]
        query_pfam, reference_pfam = pfam.get(query_id, frozenset()), reference_row["pfam_set"]
        union = query_pfam | reference_pfam
        rows.append({
            "query_protein_id": query_id, "reference_protein_id": reference_id,
            "reference_cluster_id_30": reference_row["cluster_id_30"], "retrieval_rank": int(hit.retrieval_rank),
            "sequence_identity": float(hit.fident), "sequence_query_coverage": float(hit.qcov),
            "sequence_reference_coverage": float(hit.tcov),
            "sequence_bitscore_log1p": math.log1p(max(0.0, float(hit.bits))),
            "sequence_alignment_fraction": min(float(hit.qcov), float(hit.tcov)),
            "length_ratio": min(len(query_sequence), int(reference_row["length"])) / max(len(query_sequence), int(reference_row["length"])),
            "aa_composition_cosine": cosine(composition(query_sequence), reference_row["composition"]),
            "esm2_t33_cosine": cosine(
                np.asarray(query_embeddings[embedding_rows[query_id]], dtype=np.float32),
                np.asarray(reference_matrix[int(reference_rows[reference_id])], dtype=np.float32),
            ),
            "foldseek_identity": np.nan, "foldseek_query_coverage": np.nan,
            "foldseek_reference_coverage": np.nan, "foldseek_bitscore_log1p": np.nan,
            "foldseek_alignment_fraction": np.nan, "both_structures_available": False,
            "query_structure_available": False, "reference_structure_available": bool(reference_row["has_structure"]),
            "pfam_jaccard": len(query_pfam & reference_pfam) / len(union) if union else 0.0,
            "primary_pfam_match": bool(query_pfam) and sorted(query_pfam)[0] == reference_row["primary_pfam"],
            "primary_pfam_clan_match": False, "cath_jaccard": 0.0, "primary_cath_match": False,
        })
    protein_pairs = pd.DataFrame(rows)
    activities = pd.read_parquet(root / "data/reference/activity_reference_library.parquet")
    activities = activities.loc[activities["reference_protein_id"].isin(reference_ids), [
        "activity_id", "reference_protein_id", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier",
    ]].rename(columns={"activity_id": "reference_activity_id"})
    reactions = pd.read_parquet(root / "data/processed/reaction_features.parquet")
    reaction_columns = {
        "smiles_parse_valid": "reference_smiles_parse_valid",
        "substrate_component_count": "reference_substrate_component_count",
        "product_component_count": "reference_product_component_count",
        "participant_count": "reference_participant_count",
        "unique_participant_count": "reference_unique_participant_count",
        "currency_participant_count": "reference_currency_participant_count",
        "currency_fraction": "reference_currency_fraction",
        "substrate_mw_sum": "reference_substrate_mw_sum", "product_mw_sum": "reference_product_mw_sum",
        "substrate_logp_sum": "reference_substrate_logp_sum", "product_logp_sum": "reference_product_logp_sum",
        "substrate_tpsa_sum": "reference_substrate_tpsa_sum", "product_tpsa_sum": "reference_product_tpsa_sum",
        "substrate_heavy_atoms": "reference_substrate_heavy_atoms",
        "product_heavy_atoms": "reference_product_heavy_atoms", "heavy_atom_change": "reference_heavy_atom_change",
        "reaction_complexity": "reference_reaction_complexity", "cofactor_class": "reference_cofactor_class",
    }
    reactions = reactions[["canonical_rhea", *reaction_columns]].rename(columns=reaction_columns)
    pairs = protein_pairs.merge(activities, on="reference_protein_id", how="inner", validate="many_to_many")
    pairs["reference_ec_l1"] = pd.to_numeric(pairs["ec_l3"].astype(str).str.extract(r"^([0-9]+)")[0], errors="coerce")
    pairs = pairs.merge(reactions, on="canonical_rhea", how="left", validate="many_to_one")
    pairs["reference_reaction_available"] = pairs["reference_smiles_parse_valid"].notna()
    return pairs.reset_index(drop=True)


def _model_matrix(root: Path, pairs: pd.DataFrame) -> np.ndarray:
    preprocessing = json.loads((root / "data/interim/phase11/preprocessing.json").read_text(encoding="utf-8"))
    numeric = preprocessing["numeric_columns"]
    frame = pairs[numeric].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    for column in numeric:
        frame[column] = frame[column].fillna(float(preprocessing["numeric_medians"][column]))
        frame[column] = (
            frame[column] - float(preprocessing["numeric_means"][column])
        ) / float(preprocessing["numeric_scales"][column])
    dummy = []
    for column in preprocessing["categorical_dummy_columns"]:
        if column.startswith("reference_ec_l1_"):
            value = column.removeprefix("reference_ec_l1_")
            values = pd.to_numeric(pairs["reference_ec_l1"], errors="coerce").map(
                lambda item: str(int(item)) if pd.notna(item) else "MISSING"
            )
        elif column.startswith("reference_cofactor_class_"):
            value = column.removeprefix("reference_cofactor_class_")
            values = pairs["reference_cofactor_class"].fillna("MISSING").astype(str)
        else:
            raise RuntimeError(f"Unknown preprocessing dummy: {column}")
        dummy.append(values.eq(value).to_numpy(np.float32))
    return np.column_stack([frame.to_numpy(np.float32), np.column_stack(dummy)])


def predict_fasta(
    fasta: str | Path, asset_root: str | Path, embeddings_path: str | Path,
    embedding_index_path: str | Path, output: str | Path, workdir: str | Path,
    query_pfam: str | Path | None = None, mmseqs: str = "mmseqs", threads: int = 8,
    device: str = "auto",
) -> pd.DataFrame:
    try:
        import lightgbm as lgb
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError("Install siteguard-enzyme[inference] to run prediction") from exc
    root, work = Path(asset_root).resolve(), Path(workdir).resolve()
    status = validate_asset_root(root)
    if status["status"] != "PASS":
        raise RuntimeError(f"Incomplete asset root: {status['missing']}")
    work.mkdir(parents=True, exist_ok=True)
    records = read_fasta(fasta)
    embedding_matrix = np.load(embeddings_path, mmap_mode="r")
    embedding_index = pd.read_csv(embedding_index_path, sep="\t")
    required = {"protein_id", "row_index", "sequence_sha256"}
    if not required.issubset(embedding_index.columns):
        raise ValueError(f"Embedding index lacks {sorted(required - set(embedding_index.columns))}")
    embedding_rows = embedding_index.set_index("protein_id")["row_index"].astype(int).to_dict()
    hashes = embedding_index.set_index("protein_id")["sequence_sha256"].astype(str).to_dict()
    for identifier, sequence in records.items():
        if identifier not in embedding_rows or hashes.get(identifier) != sequence_sha256(sequence):
            raise ValueError(f"Embedding missing or sequence hash mismatch for {identifier}")
    hits = _run_mmseqs(Path(fasta).resolve(), root, work, mmseqs, threads)
    pairs = _build_pairs(root, records, embedding_matrix, embedding_rows, hits, _query_pfam(query_pfam))
    if pairs.empty:
        raise RuntimeError("No reference activities were retrieved; SiteGuard cannot score this input")
    values = _model_matrix(root, pairs)
    tree = np.column_stack([
        lgb.Booster(model_file=str(root / "models/phase10" / f"lightgbm_global_{level}.txt")).predict(values)
        for level in LEVELS
    ]).astype(np.float32)
    checkpoint = torch.load(root / "models/phase11/siteguard_global_multitask.pt", map_location="cpu")
    target_device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    Model = network_class()
    model = Model(values.shape[1], int(checkpoint["width"]), float(checkpoint["dropout"])).to(target_device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    with torch.inference_mode():
        deep = torch.sigmoid(model(torch.from_numpy(values).to(target_device))).cpu().numpy().astype(np.float32)
    model_config = json.loads((root / "models/phase11/siteguard_model_config.json").read_text(encoding="utf-8"))
    blended = blend_scores(deep, tree, model_config["deep_blend_weights"], LEVELS)
    calibration = json.loads((root / "models/phase12/calibration_config.json").read_text(encoding="utf-8"))
    thresholds = {level: float(calibration["thresholds"][level]) for level in LEVELS}
    for index, level in enumerate(LEVELS):
        calibrator = load_frozen_calibrator(root, level)
        pairs[f"probability_{level}"] = calibrator.predict(blended[:, index]).astype(np.float32)
    output_rows = []
    for query_id in records:
        labels, probabilities, references, activities = {}, {}, {}, {}
        query_pairs = pairs.loc[pairs["query_protein_id"].eq(query_id)]
        for level in LEVELS:
            label_column, score_column = LABEL_COLUMNS[level], f"probability_{level}"
            candidates = query_pairs.loc[query_pairs[label_column].notna() & query_pairs[label_column].astype(str).ne("")]
            if candidates.empty:
                labels[level], probabilities[level], references[level], activities[level] = "", 0.0, "", ""
                continue
            label_scores = candidates.groupby(label_column, sort=False)[score_column].max()
            best_label = sorted(label_scores.index.astype(str), key=lambda value: (-float(label_scores.loc[value]), value))[0]
            best = candidates.loc[candidates[label_column].astype(str).eq(best_label)].sort_values(
                [score_column, "reference_protein_id"], ascending=[False, True],
            ).iloc[0]
            labels[level], probabilities[level] = best_label, float(best[score_column])
            references[level], activities[level] = str(best["reference_protein_id"]), str(best["reference_activity_id"])
        decision = decide_highest_supported_resolution(probabilities, labels, thresholds)
        output_rows.append({
            "query_protein_id": query_id, **decision,
            **{f"top_label_{level}": labels[level] for level in LEVELS},
            **{f"top_probability_{level}": probabilities[level] for level in LEVELS},
            **{f"top_reference_{level}": references[level] for level in LEVELS},
            **{f"top_reference_activity_{level}": activities[level] for level in LEVELS},
            "model_scope": "reaction-resolved enzyme catalytic-function annotation transfer",
            "query_pfam_supplied": query_id in _query_pfam(query_pfam),
            "direct_structure_features_used": False,
        })
    result = pd.DataFrame(output_rows)
    result.to_csv(output, sep="\t", index=False)
    pairs.to_parquet(work / "siteguard_pair_scores.parquet", index=False, compression="zstd")
    return result
