#!/usr/bin/env python3
"""Deterministic CLEAN split100 inference that never reads query truth labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ESM_LAYER = 33
TRUNCATION_LENGTH = 1022


class LayerNormNet(nn.Module):
    """Exact projection architecture in CLEAN.app.src.CLEAN.model.LayerNormNet."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1280, 512, dtype=torch.float32, device=device)
        self.ln1 = nn.LayerNorm(512, dtype=torch.float32, device=device)
        self.fc2 = nn.Linear(512, 512, dtype=torch.float32, device=device)
        self.ln2 = nn.LayerNorm(512, dtype=torch.float32, device=device)
        self.fc3 = nn.Linear(512, 128, dtype=torch.float32, device=device)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.dropout(self.ln1(self.fc1(inputs))))
        hidden = torch.relu(self.dropout(self.ln2(self.fc2(hidden))))
        return self.fc3(hidden)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    sequence: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(sequence).upper()))
                name = line[1:].split()[0]
                sequence = []
            else:
                if name is None:
                    raise ValueError(f"sequence before header in {path}")
                sequence.append(line)
    if name is not None:
        records.append((name, "".join(sequence).upper()))
    if not records:
        raise ValueError(f"no records in {path}")
    if len({name for name, _ in records}) != len(records):
        raise ValueError(f"duplicate FASTA identifiers in {path}")
    return records


def ordered_ec_counts(split100: Path) -> tuple[list[str], list[int]]:
    ec_members: OrderedDict[str, set[str]] = OrderedDict()
    with split100.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if len(header) < 2:
            raise ValueError("unexpected CLEAN split100 header")
        for row in reader:
            if len(row) < 2:
                continue
            sequence_id = row[0]
            for ec in row[1].split(";"):
                ec = ec.strip()
                if ec:
                    ec_members.setdefault(ec, set()).add(sequence_id)
    return list(ec_members), [len(ids) for ids in ec_members.values()]


def cluster_centres(embeddings: torch.Tensor, counts: list[int]) -> torch.Tensor:
    if int(sum(counts)) != int(embeddings.shape[0]):
        raise RuntimeError(
            f"CLEAN training embedding/count mismatch: {embeddings.shape[0]} vs {sum(counts)}"
        )
    centres = []
    start = 0
    for count in counts:
        centres.append(embeddings[start : start + count].float().mean(dim=0))
        start += count
    return torch.stack(centres)


def make_batches(records: list[tuple[str, str]], token_budget: int) -> list[list[int]]:
    sized = sorted((min(len(sequence), TRUNCATION_LENGTH) + 2, index) for index, (_, sequence) in enumerate(records))
    batches: list[list[int]] = []
    current: list[int] = []
    max_len = 0
    for length, index in sized:
        candidate_max = max(max_len, length)
        if current and candidate_max * (len(current) + 1) > token_budget:
            batches.append(current)
            current = []
            max_len = 0
        current.append(index)
        max_len = max(max_len, length)
    if current:
        batches.append(current)
    return batches


def maximum_separation(distances: np.ndarray) -> int:
    """Official CLEAN default: first above-mean separation gradient, capped at one call if >=5."""
    gamma = np.append(distances[1:], np.repeat(distances[-1], 10))
    separation = np.abs(distances - np.mean(gamma))
    gradients = np.abs(separation[:-1] - separation[1:])
    large = np.where(gradients > np.mean(gradients))[0]
    index = int(large[0]) if len(large) else 0
    return 0 if index >= 5 else index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--esm-weights", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    records = read_fasta(args.fasta)
    split100 = args.clean_root / "app" / "data" / "split100.csv"
    ec_labels, ec_counts = ordered_ec_counts(split100)

    from esm import pretrained

    esm_model, alphabet = pretrained.load_model_and_alphabet_local(args.esm_weights)
    esm_model.to(device).eval()
    batch_converter = alphabet.get_batch_converter(TRUNCATION_LENGTH)

    projection = LayerNormNet(device)
    projection_path = args.clean_root / "app" / "data" / "pretrained" / "split100.pth"
    projection.load_state_dict(torch.load(projection_path, map_location=device), strict=True)
    projection.eval()

    train_embedding_path = args.clean_root / "app" / "data" / "pretrained" / "100.pt"
    train_embeddings = torch.load(train_embedding_path, map_location="cpu")
    centres = cluster_centres(train_embeddings, ec_counts).to(device)
    if centres.shape != (len(ec_labels), 128):
        raise RuntimeError(f"unexpected CLEAN centres shape: {tuple(centres.shape)}")

    rows_by_index: dict[int, dict[str, object]] = {}
    batches = make_batches(records, args.token_budget)
    with torch.inference_mode():
        for batch_number, indices in enumerate(batches, start=1):
            batch_records = [records[index] for index in indices]
            labels, strings, tokens = batch_converter(batch_records)
            output = esm_model(tokens.to(device), repr_layers=[ESM_LAYER], return_contacts=False)
            representations = output["representations"][ESM_LAYER]
            means = []
            for representation, sequence in zip(representations, strings):
                length = min(TRUNCATION_LENGTH, len(sequence))
                means.append(representation[1 : length + 1].mean(dim=0))
            query_embeddings = projection(torch.stack(means))
            distances = torch.cdist(query_embeddings.float(), centres.float(), p=2)
            values, nearest = torch.topk(distances, k=10, largest=False, dim=1)
            for local_index, record_index in enumerate(indices):
                query_id, sequence = records[record_index]
                distance_values = values[local_index].detach().cpu().numpy()
                ec_indices = nearest[local_index].detach().cpu().numpy()
                maxsep_index = maximum_separation(distance_values)
                row: dict[str, object] = {
                    "query_id": query_id,
                    "sequence_length": len(sequence),
                    "truncated_residue_count": max(0, len(sequence) - TRUNCATION_LENGTH),
                    "clean_maxsep_n": maxsep_index + 1,
                }
                for rank, (ec_index, distance) in enumerate(zip(ec_indices, distance_values), start=1):
                    row[f"clean_top{rank}_ec4"] = ec_labels[int(ec_index)]
                    row[f"clean_top{rank}_distance"] = float(distance)
                row["clean_distance_margin12"] = float(distance_values[1] - distance_values[0])
                row["clean_distance_ratio12"] = float(distance_values[0] / max(distance_values[1], 1e-12))
                row["clean_maxsep_ec4"] = ";".join(ec_labels[int(index)] for index in ec_indices[: maxsep_index + 1])
                row["clean_maxsep_distances"] = ";".join(f"{value:.8g}" for value in distance_values[: maxsep_index + 1])
                rows_by_index[record_index] = row
            print(f"CLEAN_PROGRESS {batch_number}/{len(batches)} ({len(rows_by_index)}/{len(records)})", flush=True)

    rows = [rows_by_index[index] for index in range(len(records))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "tool": "CLEAN",
        "tool_git_commit": "f2bf2a4f497fa2cc87dac2a1bb314fee587c0a15",
        "train_split": "split100",
        "input_fasta": str(args.fasta),
        "input_fasta_sha256": sha256(args.fasta),
        "esm_weights_sha256": sha256(args.esm_weights),
        "projection_sha256": sha256(projection_path),
        "train_embeddings_sha256": sha256(train_embedding_path),
        "n_queries": len(records),
        "n_ec_centres": len(ec_labels),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "token_budget": args.token_budget,
        "truncation_length": TRUNCATION_LENGTH,
        "truth_inputs_used": False,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("CHECKPOINT_29B_CLEAN_PREDICTIONS_LOCKED")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
