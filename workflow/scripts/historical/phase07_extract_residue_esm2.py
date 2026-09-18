#!/usr/bin/env python3
"""Extract frozen ESM2-t12 central-residue embeddings for prepared local windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    contexts = (
        pd.read_csv(args.contexts, sep="\t", compression="gzip")
        if args.contexts.name.endswith(".tsv.gz") else pd.read_parquet(args.contexts)
    ).sort_values("embedding_row").reset_index(drop=True)
    if contexts["embedding_row"].tolist() != list(range(len(contexts))):
        raise RuntimeError("Residue ESM2 embedding rows are not contiguous")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Residue ESM2 extraction requires CUDA")
    model.to(device).eval()
    hidden_size = int(model.config.hidden_size)
    start = 0
    if args.state.is_file() and args.output.is_file():
        state = json.loads(args.state.read_text(encoding="utf-8"))
        if int(state.get("rows", -1)) == len(contexts) and int(state.get("hidden_size", -1)) == hidden_size:
            start = int(state.get("next_index", 0))
            matrix = np.lib.format.open_memmap(args.output, mode="r+", dtype=np.float16, shape=(len(contexts), hidden_size))
        else:
            start = 0
    if start == 0:
        matrix = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.float16, shape=(len(contexts), hidden_size))
    for batch_start in range(start, len(contexts), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(contexts))
        batch = contexts.iloc[batch_start:batch_end]
        encoded = tokenizer(batch["sequence"].tolist(), padding=True, return_tensors="pt")
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            hidden = model(**encoded).last_hidden_state
        centers = torch.tensor(batch["center_offset"].to_numpy() + 1, device=device, dtype=torch.long)
        selected = hidden[torch.arange(len(batch), device=device), centers].float().cpu().numpy()
        matrix[batch_start:batch_end] = selected.astype(np.float16)
        matrix.flush()
        args.state.write_text(json.dumps({
            "rows": len(contexts), "hidden_size": hidden_size, "next_index": batch_end,
        }, indent=2) + "\n", encoding="utf-8")
        if batch_end % 10_000 < args.batch_size or batch_end == len(contexts):
            print(json.dumps({"embedded_residue_contexts": batch_end, "total": len(contexts)}), flush=True)
    norms = np.linalg.norm(np.asarray(matrix, dtype=np.float32), axis=1)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise RuntimeError("Residue ESM2 matrix contains invalid rows")
    summary = {
        "status": "PASS", "residue_contexts": len(contexts), "hidden_size": hidden_size,
        "dtype": "float16", "model_path": str(args.model), "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0), "pooling": "CENTRAL_RESIDUE_LAST_HIDDEN_STATE",
        "minimum_l2_norm": float(norms.min()), "maximum_l2_norm": float(norms.max()),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
