#!/usr/bin/env python3
"""Apply the frozen Phase 11 deep model to Phase 15 P450 features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from phase14_deep_infer import SiteGuardGlobalNet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=65_536)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data/interim/phase15"
    values = np.load(work / "p450_X.npy", mmap_mode="r")
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
        work / "p450_deep_predictions.npy", mode="w+", dtype=np.float32, shape=(len(values), 3),
    )
    with torch.inference_mode():
        for start in range(0, len(values), args.batch_size):
            end = min(start + args.batch_size, len(values))
            batch = torch.from_numpy(np.asarray(values[start:end], dtype=np.float32)).to(device)
            output[start:end] = torch.sigmoid(model(batch)).cpu().numpy().astype(np.float32)
            print(json.dumps({"p450_deep_inference_rows": end, "total": len(values)}), flush=True)
    output.flush()
    summary = {
        "phase": 15, "stage": "p450_deep_external_inference", "status": "PASS",
        "slurm_job_id": os.getenv("SLURM_JOB_ID", "NA"), "rows": len(values),
        "features": values.shape[1], "device": torch.cuda.get_device_name(0),
        "weights": "frozen Phase 11 validation-selected checkpoint", "p450_used_for_training": False,
    }
    (root / "reports/phase15_deep_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
