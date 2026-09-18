#!/usr/bin/env python3
"""Apply the frozen Phase 11 neural pair scorer to Phase 28 blind pairs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from siteguard.model import network_class


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
PHASE_ID = int(os.environ.get("RCSB_EXTERNAL_PHASE", "28"))
if PHASE_ID not in {28, 30, 31, 32}:
    raise RuntimeError(f"Unsupported structural external phase: {PHASE_ID}")
WORK = ROOT / f"data/interim/phase{PHASE_ID}_inference"
CHECKPOINTS = ROOT / "checkpoints"
REPORTS = ROOT / f"reports/phase{PHASE_ID}_external_blind"


def main() -> None:
    required = CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}A5A_EXTERNAL_PAIR_EVIDENCE_PASS"
    if not required.is_file():
        raise FileNotFoundError(required)
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated CUDA GPU is unavailable")
    values = np.load(WORK / "external_model_matrix.npy", mmap_mode="r")
    checkpoint = torch.load(ROOT / "models/phase11/siteguard_global_multitask.pt", map_location="cpu")
    Model = network_class()
    model = Model(values.shape[1], int(checkpoint["width"]), float(checkpoint["dropout"])).cuda()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    output = np.empty((len(values), 3), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(values), 8192):
            batch = torch.from_numpy(np.asarray(values[start:start + 8192], dtype=np.float32)).cuda()
            output[start:start + len(batch)] = torch.sigmoid(model(batch)).cpu().numpy()
    if not np.isfinite(output).all():
        raise RuntimeError("Non-finite frozen deep scores")
    np.save(WORK / "external_deep_global_scores.npy", output)
    summary = {"phase": f"{PHASE_ID}A5B", "status": "PASS", "shape": list(output.shape), "model_frozen_phase": 11}
    (REPORTS / f"phase{PHASE_ID}_deep_score_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}A5B_EXTERNAL_DEEP_SCORE_PASS").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
