"""Inference asset discovery and validation."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_ASSETS = {
    "mmseqs_reference": "databases/mmseqs/reference_seq.dbtype",
    "reference_library": "data/reference/activity_reference_library.parquet",
    "protein_table": "data/processed/protein_table.parquet",
    "family_split": "data/splits/split_family.parquet",
    "sequence_split": "data/splits/split_sequence.parquet",
    "structure_membership": "data/interim/phase04/afdb_membership.parquet",
    "reference_embeddings": "data/interim/phase06/esm2_t33_v4_embeddings.npy",
    "reference_embedding_index": "data/interim/phase06/esm2_t33_v4_index.tsv",
    "reaction_features": "data/processed/reaction_features.parquet",
    "preprocessing": "data/interim/phase11/preprocessing.json",
    "tree_ec3": "models/phase10/lightgbm_global_EC_L3.txt",
    "tree_ec4": "models/phase10/lightgbm_global_EC_L4.txt",
    "tree_rhea": "models/phase10/lightgbm_global_EXACT_RHEA.txt",
    "deep_model": "models/phase11/siteguard_global_multitask.pt",
    "model_config": "models/phase11/siteguard_model_config.json",
    "calibration_config": "models/phase12/calibration_config.json",
    "calibrator_ec3": "models/phase12/isotonic_EC_L3.joblib",
    "calibrator_ec4": "models/phase12/isotonic_EC_L4.joblib",
    "calibrator_rhea": "models/phase12/isotonic_EXACT_RHEA.joblib",
}

CHEMBRIDGE_ASSETS = {
    "locked_model": "models/phase23/LOCKED_MODEL.json",
    "reaction_ids": "data/interim/phase23/reaction_ids.npy",
    "reaction_catalog": "data/interim/phase23/reaction_catalog.tsv",
}


def validate_asset_root(root: str | Path) -> dict[str, object]:
    base = Path(root).resolve()
    missing = [relative for relative in REQUIRED_ASSETS.values() if not (base / relative).is_file()]
    calibration = base / REQUIRED_ASSETS["calibration_config"]
    thresholds = {}
    if calibration.is_file():
        thresholds = json.loads(calibration.read_text(encoding="utf-8")).get("thresholds", {})
    return {
        "asset_root": str(base),
        "status": "PASS" if not missing else "FAIL",
        "missing": missing,
        "thresholds": thresholds,
        "scope": "reaction-resolved enzyme catalytic-function annotation transfer",
    }


def validate_chembridge_assets(root: str | Path) -> dict[str, object]:
    base = Path(root).resolve()
    missing = [relative for relative in CHEMBRIDGE_ASSETS.values() if not (base / relative).is_file()]
    locked = {}
    lock_path = base / CHEMBRIDGE_ASSETS["locked_model"]
    if lock_path.is_file():
        locked = json.loads(lock_path.read_text(encoding="utf-8"))
        view = str(locked.get("view", "")).lower()
        if view:
            relative = f"data/interim/phase23/reaction_latent_{view}.npy"
            if not (base / relative).is_file():
                missing.append(relative)
        for relative in locked.get("checkpoints", []):
            if not (base / str(relative)).is_file():
                missing.append(str(relative))
    return {
        "asset_root": str(base), "status": "PASS" if not missing else "FAIL",
        "missing": sorted(set(missing)), "locked_model": locked.get("selected_key"),
        "scope": "full-catalog known-reaction candidate retrieval",
    }
