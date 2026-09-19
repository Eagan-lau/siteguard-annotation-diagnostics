#!/usr/bin/env python3
"""Run official HIT-EC v2.0.0 weights on FASTA without reading any truth labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch


LEVEL_SIZES = [7, 72, 268, 4255]
OFFSETS = np.cumsum([0] + LEVEL_SIZES[:-1]).tolist()
TOKEN_IDS = {
    "l": 1, "a": 2, "g": 3, "v": 4, "e": 5, "i": 6, "s": 7,
    "d": 8, "k": 9, "r": 10, "t": 11, "p": 12, "n": 13,
    "f": 14, "q": 15, "y": 16, "h": 17, "m": 18, "c": 19,
    "w": 20, "x": 21,
}
MAX_LEN = 1024
BOS_ID = 22


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    chunks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks).upper()))
                name = line[1:].split()[0]
                chunks = []
            else:
                if name is None:
                    raise ValueError(f"sequence before FASTA header in {path}")
                chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks).upper()))
    if not records:
        raise ValueError(f"no FASTA records in {path}")
    ids = [name for name, _ in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate FASTA identifiers in {path}")
    return records


def encode(sequence: str) -> tuple[torch.Tensor, int, int]:
    # Equivalent to the repository's char-level Keras Tokenizer: lowercase,
    # silently omit characters outside the 21-symbol learned vocabulary.
    tokens = [TOKEN_IDS[aa] for aa in sequence.lower() if aa in TOKEN_IDS]
    unknown_count = len(sequence) - len(tokens)
    truncated_count = max(0, len(tokens) - (MAX_LEN - 1))
    tokens = [BOS_ID] + tokens[: MAX_LEN - 1]
    tokens.extend([0] * (MAX_LEN - len(tokens)))
    return torch.tensor([tokens], dtype=torch.int32), unknown_count, truncated_count


def load_labels(label_dir: Path) -> list[list[str]]:
    levels = []
    for level, expected in enumerate(LEVEL_SIZES, start=1):
        labels = (label_dir / f"level{level}_labels.txt").read_text(encoding="utf-8").splitlines()
        if len(labels) != expected:
            raise RuntimeError(f"level {level}: expected {expected} labels, found {len(labels)}")
        levels.append(labels)
    return levels


def install_runtime_shims() -> None:
    """Provide two optional training-only dependencies needed at import time."""
    module = types.ModuleType("pytorch_lightning")

    class LightningModule(torch.nn.Module):
        pass

    module.LightningModule = LightningModule
    sys.modules["pytorch_lightning"] = module
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable=None, *args, **kwargs: iterable
    sys.modules.setdefault("tqdm", tqdm_module)


def top_fields(logits: torch.Tensor, labels: list[str], prefix: str) -> dict[str, object]:
    probabilities = torch.softmax(logits.float(), dim=0)
    values, indices = torch.topk(probabilities, k=5)
    raw_values = logits.float()[indices]
    result: dict[str, object] = {}
    for rank, (idx, probability, raw) in enumerate(zip(indices.tolist(), values.tolist(), raw_values.tolist()), start=1):
        result[f"{prefix}_top{rank}"] = labels[idx]
        result[f"{prefix}_top{rank}_softmax"] = probability
        result[f"{prefix}_top{rank}_logit"] = raw
    result[f"{prefix}_softmax_margin12"] = float(values[0] - values[1])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hit-ec-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    records = read_fasta(args.fasta)
    labels = load_labels(args.labels)

    install_runtime_shims()
    sys.path.insert(0, str(args.hit_ec_root))
    Model = importlib.import_module("model.model").Model
    config = {"ah": 2, "dr": 0.1, "beta": 0.59, "output_dims": LEVEL_SIZES}
    model = Model(config)
    checkpoint_path = args.hit_ec_root / "utils" / "model.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint incompatibility: {incompatible}")
    device = torch.device(args.device)
    model.to(device).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for index, (query_id, sequence) in enumerate(records, start=1):
            encoded, unknown_count, truncated_count = encode(sequence)
            logits = model(encoded.to(device), mode="infer")[0].detach().cpu()
            row: dict[str, object] = {
                "query_id": query_id,
                "sequence_length": len(sequence),
                "unknown_residue_count": unknown_count,
                "truncated_residue_count": truncated_count,
            }
            for level, (offset, size) in enumerate(zip(OFFSETS, LEVEL_SIZES), start=1):
                level_logits = logits[offset : offset + size]
                row.update(top_fields(level_logits, labels[level - 1], f"ec{level}"))
                if level == 4:
                    sigmoid = torch.sigmoid(level_logits.float())
                    top_values, top_indices = torch.topk(sigmoid, k=5)
                    for rank, (idx, value) in enumerate(zip(top_indices.tolist(), top_values.tolist()), start=1):
                        row[f"ec4_top{rank}_sigmoid"] = value
                    row["ec4_sigmoid_margin12"] = float(top_values[0] - top_values[1])
                    row["ec4_n_sigmoid_gt_0_4"] = int((sigmoid > 0.4).sum().item())
            row["ec4_top1_parent_ec3"] = str(row["ec4_top1"]).rsplit(".", 1)[0]
            row["ec3_ec4_hierarchy_consistent"] = int(row["ec3_top1"] == row["ec4_top1_parent_ec3"])
            rows.append(row)
            if index == 1 or index % 10 == 0 or index == len(records):
                print(f"HIT_EC_PROGRESS {index}/{len(records)}", flush=True)

    fieldnames = list(rows[0].keys())
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "tool": "HIT-EC",
        "tool_git_commit": "c4779b0",
        "release": "v2.0.0",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "input_fasta": str(args.fasta),
        "input_fasta_sha256": sha256(args.fasta),
        "n_queries": len(records),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "device": str(device),
        "truth_inputs_used": False,
        "sequence_handling": "official char vocabulary; BOS=22; pad/truncate to 1024",
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CHECKPOINT_29A_HIT_EC_PREDICTIONS_LOCKED")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
