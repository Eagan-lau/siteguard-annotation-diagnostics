"""ChemBridge full-catalog known-reaction retrieval for frozen ESM2 embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .assets import validate_chembridge_assets
from .fasta import read_fasta, sequence_sha256


def chembridge_network_class(architecture: str):
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError("Install siteguard-enzyme[inference] to load ChemBridge") from exc

    class LinearTower(nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            self.network = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim))

        def forward(self, values):
            return nn.functional.normalize(self.network(values), dim=1)

    class MLPTower(nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            self.input_norm = nn.LayerNorm(input_dim)
            self.input = nn.Linear(input_dim, 768)
            self.block = nn.Sequential(
                nn.LayerNorm(768), nn.Linear(768, 768), nn.GELU(), nn.Dropout(0.10),
                nn.Linear(768, 768), nn.Dropout(0.10),
            )
            self.neck = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.05))
            self.output = nn.Linear(512, output_dim)

        def forward(self, values):
            hidden = nn.functional.gelu(self.input(self.input_norm(values)))
            hidden = hidden + self.block(hidden)
            return nn.functional.normalize(self.output(self.neck(hidden)), dim=1)

    return LinearTower if architecture == "LINEAR" else MLPTower


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def _embedding_rows(
    matrix_path: str | Path, index_path: str | Path, fasta: str | Path | None,
) -> tuple[list[str], np.ndarray]:
    matrix = np.load(matrix_path, mmap_mode="r")
    index = pd.read_csv(index_path, sep="\t")
    required = {"protein_id", "row_index"}
    if not required.issubset(index.columns):
        raise ValueError(f"Embedding index lacks {sorted(required - set(index.columns))}")
    if index["protein_id"].duplicated().any() or index["row_index"].duplicated().any():
        raise ValueError("Embedding index must be one-to-one")
    if fasta is None:
        ordered = index.sort_values("row_index")
    else:
        records = read_fasta(fasta)
        lookup = index.set_index("protein_id")
        missing = sorted(set(records) - set(lookup.index.astype(str)))
        if missing:
            raise ValueError(f"Embedding index is missing FASTA identifiers: {missing[:5]}")
        if "sequence_sha256" not in index.columns:
            raise ValueError("Embedding index needs sequence_sha256 when --input-fasta is supplied")
        for identifier, sequence in records.items():
            if str(lookup.loc[identifier, "sequence_sha256"]) != sequence_sha256(sequence):
                raise ValueError(f"Embedding sequence hash mismatch for {identifier}")
        ordered = lookup.loc[list(records)].reset_index()
    rows = ordered["row_index"].to_numpy(np.int64)
    if rows.min(initial=0) < 0 or rows.max(initial=-1) >= len(matrix):
        raise ValueError("Embedding row index is outside the matrix")
    return ordered["protein_id"].astype(str).tolist(), np.asarray(matrix[rows], dtype=np.float32)


def propose_reactions(
    asset_root: str | Path, query_embeddings: str | Path, embedding_index: str | Path,
    output: str | Path, top_k: int = 10, input_fasta: str | Path | None = None,
    device: str = "auto",
) -> pd.DataFrame:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError("Install siteguard-enzyme[inference] to run ChemBridge") from exc
    root = Path(asset_root).resolve()
    status = validate_chembridge_assets(root)
    if status["status"] != "PASS":
        raise RuntimeError(f"Incomplete ChemBridge asset root: {status['missing']}")
    if top_k < 1 or top_k > 100:
        raise ValueError("top_k must be between 1 and 100")
    locked = json.loads((root / "models/phase23/LOCKED_MODEL.json").read_text(encoding="utf-8"))
    if locked.get("status") != "LOCKED_FROM_VALIDATION_ONLY" or locked.get("test_loaded") is not False:
        raise RuntimeError("ChemBridge lock certificate is invalid")
    identifiers, features = _embedding_rows(query_embeddings, embedding_index, input_fasta)
    target_device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    architecture, view = str(locked["architecture"]), str(locked["view"]).lower()
    Model = chembridge_network_class(architecture)
    seed_predictions = []
    values = torch.from_numpy(features).to(target_device)
    for relative in locked["checkpoints"]:
        checkpoint = torch.load(root / str(relative), map_location=target_device)
        if checkpoint.get("architecture") != architecture or str(checkpoint.get("view", "")).lower() != view:
            raise RuntimeError(f"Checkpoint does not match locked architecture/view: {relative}")
        model = Model(int(checkpoint["input_dim"]), int(checkpoint["output_dim"])).to(target_device)
        model.load_state_dict(checkpoint["state_dict"]); model.eval()
        with torch.inference_mode():
            seed_predictions.append(model(values).float().cpu().numpy())
        del model
    predictions = _unit(np.mean(np.stack(seed_predictions), axis=0))
    candidates = np.asarray(np.load(root / f"data/interim/phase23/reaction_latent_{view}.npy"), dtype=np.float32)
    reaction_ids = np.load(root / "data/interim/phase23/reaction_ids.npy").astype(str)
    if len(candidates) != len(reaction_ids):
        raise RuntimeError("Reaction latent matrix and ID index are misaligned")
    mean_scores = predictions @ candidates.T
    top = np.argpartition(-mean_scores, kth=top_k - 1, axis=1)[:, :top_k]
    top = np.stack([row[np.argsort(-mean_scores[index, row], kind="mergesort")] for index, row in enumerate(top)])
    release = pd.read_csv(root / "data/interim/phase23/reaction_catalog.tsv", sep="\t").drop_duplicates("canonical_rhea")
    metadata = release.set_index("canonical_rhea").to_dict("index")
    rows = []
    seed_stack = np.stack(seed_predictions)
    for query_index, (identifier, panel) in enumerate(zip(identifiers, top)):
        panel_scores = seed_stack[:, query_index, :] @ candidates[panel].T
        sorted_scores = mean_scores[query_index, panel]
        margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else float("nan")
        for rank, candidate_index in enumerate(panel, start=1):
            rhea = str(reaction_ids[candidate_index]); details = metadata.get(rhea, {})
            rows.append({
                "query_protein_id": identifier, "rank": rank, "candidate_rhea": rhea,
                "catalog_cosine_score": float(mean_scores[query_index, candidate_index]),
                "seed_score_sd": float(np.std(panel_scores[:, rank - 1], ddof=1)) if len(seed_predictions) > 1 else 0.0,
                "top1_margin": margin if rank == 1 else float("nan"),
                "reaction_definition": details.get("definition", ""), "reaction_equation": details.get("equation", ""),
                "reaction_smiles_lr": details.get("reaction_smiles_lr", ""),
                "model": locked["selected_key"], "reaction_view": locked["view"],
                "candidate_catalog_size": len(reaction_ids),
                "claim_level": "KNOWN_REACTION_CANDIDATE_PANEL_NOT_EXPERIMENTALLY_VALIDATED",
                "score_policy": "RANKING_SCORE_NOT_CALIBRATED_PROBABILITY",
                "query_function_truth_used": False,
            })
    result = pd.DataFrame(rows)
    result.to_csv(output, sep="\t", index=False)
    return result
