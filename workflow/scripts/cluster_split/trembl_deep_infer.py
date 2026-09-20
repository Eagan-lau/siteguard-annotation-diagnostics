#!/usr/bin/env python3
"""Apply the frozen Phase 11 deep model to Phase 14 external features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width * 2, width), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values + self.block(values))


class SiteGuardGlobalNet(nn.Module):
    def __init__(self, input_features: int, width: int = 256, dropout: float = 0.15) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_features), nn.Linear(input_features, width), nn.GELU())
        self.trunk = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(3)])
        self.neck = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(128))
        self.head = nn.Linear(128, 3)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.neck(self.trunk(self.input(values))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=65_536)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data/interim/phase14"
    values = np.load(work / "external_X.npy", mmap_mode="r")
    checkpoint = torch.load(root / "models/phase11/siteguard_global_multitask.pt", map_location="cpu")
    if values.shape[1] != int(checkpoint["input_features"]):
        raise RuntimeError(f"Feature mismatch: {values.shape}/{checkpoint['input_features']}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    model = SiteGuardGlobalNet(values.shape[1], int(checkpoint["width"]), float(checkpoint["dropout"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    output = np.lib.format.open_memmap(
        work / "external_deep_predictions.npy", mode="w+", dtype=np.float32, shape=(len(values), 3),
    )
    with torch.inference_mode():
        for start in range(0, len(values), args.batch_size):
            end = min(start + args.batch_size, len(values))
            batch = torch.from_numpy(np.asarray(values[start:end], dtype=np.float32)).to(device)
            output[start:end] = torch.sigmoid(model(batch)).cpu().numpy().astype(np.float32)
            if end % 500_000 < args.batch_size or end == len(values):
                print(json.dumps({"deep_inference_rows": end, "total": len(values)}), flush=True)
    output.flush()
    summary = {
        "phase": 14, "stage": "deep_external_inference", "status": "PASS",
        "slurm_job_id": os.getenv("SLURM_JOB_ID", "NA"), "rows": len(values),
        "features": values.shape[1], "device": torch.cuda.get_device_name(0),
        "weights": "frozen Phase 11 validation-selected checkpoint", "trembl_used_for_training": False,
    }
    (root / "reports/phase14_deep_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
