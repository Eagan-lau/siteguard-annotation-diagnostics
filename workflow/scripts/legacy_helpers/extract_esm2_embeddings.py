#!/usr/bin/env python3
"""Extract restartable mean-pooled ESM2 embeddings using a frozen local snapshot."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402


def read_sequences(path: Path) -> tuple[list[str], list[str]]:
    proteins: list[str] = []
    sequences: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            protein, sequence = line.rstrip("\n").split("\t", 1)
            proteins.append(protein)
            sequences.append(sequence)
    return proteins, sequences


def windows(sequence: str, length: int, overlap: int) -> list[str]:
    if len(sequence) <= length:
        return [sequence]
    step = length - overlap
    starts = list(range(0, max(1, len(sequence) - length + 1), step))
    final = len(sequence) - length
    if starts[-1] != final:
        starts.append(final)
    return [sequence[start : start + length] for start in starts]


def embed_chunk(
    model: torch.nn.Module,
    tokenizer: Any,
    sequences: list[str],
    device: torch.device,
    token_budget: int,
    window_length: int,
    overlap: int,
) -> tuple[np.ndarray, list[int]]:
    fragments: list[tuple[int, str]] = []
    counts: list[int] = []
    for protein_index, sequence in enumerate(sequences):
        values = windows(sequence, window_length, overlap)
        counts.append(len(values))
        fragments.extend((protein_index, value) for value in values)
    fragments.sort(key=lambda item: len(item[1]), reverse=True)
    sums = np.zeros((len(sequences), int(model.config.hidden_size)), dtype=np.float64)
    weights = np.zeros(len(sequences), dtype=np.float64)
    start = 0
    while start < len(fragments):
        end = start
        used = 0
        while end < len(fragments):
            length = len(fragments[end][1]) + 2
            if end > start and used + length > token_budget:
                break
            used += length
            end += 1
        batch = fragments[start:end]
        encoded = tokenizer([value for _, value in batch], padding=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            hidden = model(**encoded).last_hidden_state
        for batch_index, (protein_index, fragment) in enumerate(batch):
            length = len(fragment)
            vector = hidden[batch_index, 1 : length + 1].float().mean(dim=0).cpu().numpy()
            sums[protein_index] += vector * length
            weights[protein_index] += length
        start = end
    return (sums / weights[:, None]).astype(np.float16), counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--protein-chunk", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=6000)
    parser.add_argument("--window-length", type=int, default=1000)
    parser.add_argument("--window-overlap", type=int, default=200)
    args = parser.parse_args()
    proteins, sequences = read_sequences(args.sequences)
    order_hash = hashlib.sha256("\n".join(proteins).encode()).hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("ESM2-650M extraction requires an allocated CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).to(device).eval()
    hidden_size = int(model.config.hidden_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.state.is_file():
        state = json.loads(args.state.read_text(encoding="utf-8"))
        if state["protein_order_sha256"] != order_hash or int(state["proteins"]) != len(proteins):
            raise RuntimeError("ESM resume state does not match frozen sequence panel")
        next_index = int(state["next_index"])
        matrix = np.lib.format.open_memmap(args.output, mode="r+")
    else:
        next_index = 0
        matrix = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.float16, shape=(len(proteins), hidden_size))
    window_counts = np.array(
        [len(windows(sequence, args.window_length, args.window_overlap)) for sequence in sequences],
        dtype=np.int32,
    )
    for start in range(next_index, len(proteins), args.protein_chunk):
        end = min(len(proteins), start + args.protein_chunk)
        vectors, counts = embed_chunk(model, tokenizer, sequences[start:end], device, args.token_budget, args.window_length, args.window_overlap)
        matrix[start:end] = vectors
        matrix.flush()
        if counts != window_counts[start:end].tolist():
            raise RuntimeError("window-count bookkeeping mismatch")
        next_index = end
        temporary = args.state.with_suffix(args.state.suffix + ".tmp")
        temporary.write_text(json.dumps({
            "protein_order_sha256": order_hash, "proteins": len(proteins), "next_index": next_index,
            "hidden_size": hidden_size, "dtype": "float16",
        }, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, args.state)
        if next_index % 640 == 0 or next_index == len(proteins):
            print(json.dumps({"embedded_proteins": next_index, "total": len(proteins)}), flush=True)
    with args.index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protein_id", "row_index", "sequence_length", "window_count"], delimiter="\t")
        writer.writeheader()
        for index, (protein, sequence) in enumerate(zip(proteins, sequences, strict=True)):
            writer.writerow({"protein_id": protein, "row_index": index, "sequence_length": len(sequence), "window_count": int(window_counts[index])})
    summary = {
        "proteins": len(proteins), "hidden_size": hidden_size, "dtype": "float16",
        "model_path": str(args.model), "protein_order_sha256": order_hash,
        "window_length": args.window_length, "window_overlap": args.window_overlap,
        "device": str(device), "cuda_device": torch.cuda.get_device_name(0), "status": "PASS",
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
