"""Local, restart-free ESM2 mean pooling for inference-sized FASTA files."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from .fasta import read_fasta, sequence_sha256


def embed_fasta(
    fasta: str | Path,
    model_path: str | Path,
    output: str | Path,
    index_path: str | Path,
    token_budget: int = 6000,
    window_length: int = 1000,
    overlap: int = 200,
) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError("Install siteguard-enzyme[embedding] to compute ESM2 embeddings") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("ESM2-t33 embedding requires an allocated CUDA GPU")
    records = read_fasta(fasta)
    identifiers, sequences = list(records), list(records.values())
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).cuda().eval()
    fragments: list[tuple[int, str]] = []
    fragment_counts: list[int] = []
    for protein_index, sequence in enumerate(sequences):
        if len(sequence) <= window_length:
            values = [sequence]
        else:
            step = window_length - overlap
            starts = list(range(0, len(sequence) - window_length + 1, step))
            if starts[-1] != len(sequence) - window_length:
                starts.append(len(sequence) - window_length)
            values = [sequence[start:start + window_length] for start in starts]
        fragment_counts.append(len(values))
        fragments.extend((protein_index, value) for value in values)
    fragments.sort(key=lambda item: len(item[1]), reverse=True)
    sums = np.zeros((len(sequences), int(model.config.hidden_size)), dtype=np.float64)
    weights = np.zeros(len(sequences), dtype=np.float64)
    start = 0
    while start < len(fragments):
        end, used = start, 0
        while end < len(fragments):
            length = len(fragments[end][1]) + 2
            if end > start and used + length > token_budget:
                break
            used += length
            end += 1
        batch = fragments[start:end]
        encoded = tokenizer([value for _, value in batch], padding=True, return_tensors="pt")
        encoded = {key: value.cuda() for key, value in encoded.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            hidden = model(**encoded).last_hidden_state
        for batch_index, (protein_index, fragment) in enumerate(batch):
            length = len(fragment)
            vector = hidden[batch_index, 1:length + 1].float().mean(dim=0).cpu().numpy()
            sums[protein_index] += vector * length
            weights[protein_index] += length
        start = end
    matrix = (sums / weights[:, None]).astype(np.float16)
    np.save(output, matrix)
    with Path(index_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["protein_id", "row_index", "sequence_length", "sequence_sha256", "window_count"],
            delimiter="\t",
        )
        writer.writeheader()
        for row_index, (identifier, sequence) in enumerate(records.items()):
            writer.writerow({
                "protein_id": identifier, "row_index": row_index, "sequence_length": len(sequence),
                "sequence_sha256": sequence_sha256(sequence), "window_count": fragment_counts[row_index],
            })
