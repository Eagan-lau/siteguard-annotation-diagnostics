#!/usr/bin/env python3
"""Run the fixed three-seed ensemble on one frozen role matrix."""
import argparse
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROLES = ("DEV", "CAL_FIT", "CAL_RULE", "RETEST")
SEEDS = (20260819, 20260820, 20260821)


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def emit(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle: json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")


class ResidualBlock(nn.Module):
    def __init__(self, width=256, dropout=0.15):
        super().__init__(); self.block = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 2, width), nn.Dropout(dropout)); self.norm = nn.LayerNorm(width)
    def forward(self, values): return self.norm(values + self.block(values))


class SiteGuardGlobalNet(nn.Module):
    def __init__(self, input_features, width=256, dropout=0.15):
        super().__init__(); self.input = nn.Sequential(nn.LayerNorm(input_features), nn.Linear(input_features, width), nn.GELU()); self.trunk = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(3)]); self.neck = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(128)); self.head = nn.Linear(128, 3)
    def forward(self, values): return self.head(self.neck(self.trunk(self.input(values))))


@torch.no_grad()
def predict(model, values, device, batch_size=65536):
    output = []; model.eval()
    for start in range(0, len(values), batch_size):
        batch = torch.from_numpy(np.asarray(values[start:start + batch_size])).to(device)
        output.append(torch.sigmoid(model(batch)).cpu().numpy().astype(np.float32))
    return np.concatenate(output)


def execute(matrices, models, role_index, output):
    require(0 <= role_index < len(ROLES) and output.parent.is_dir() and not output.exists(), "role/output")
    role = ROLES[role_index]; output.mkdir(exist_ok=False); emit(output / "reservation.json", {"status": "ONE_S4R_ENSEMBLE_INFERENCE_RESERVED", "role": role, "retest_truth_read": False, "automatic_retry": False})
    state, error = "FAIL_CLOSED", None
    try:
        require(torch.cuda.is_available(), "CUDA required")
        values = np.load(matrices / f"X_{role}.npy", mmap_mode="r"); require(values.ndim == 2 and len(values) > 0, "role matrix")
        device = torch.device("cuda"); total = np.zeros((len(values), 3), dtype=np.float64); model_hashes = {}
        for seed in SEEDS:
            path = models / f"seed_{seed}/siteguard_global_multitask.pt"
            checkpoint = torch.load(path, map_location=device); require(checkpoint["seed"] == seed and checkpoint["input_features"] == values.shape[1], "model identity")
            model = SiteGuardGlobalNet(checkpoint["input_features"], checkpoint["width"], checkpoint["dropout"]).to(device); model.load_state_dict(checkpoint["state_dict"])
            total += predict(model, values, device).astype(np.float64); model_hashes[str(seed)] = sha(path); del model; torch.cuda.empty_cache()
        ensemble = (total / len(SEEDS)).astype(np.float32); require(np.isfinite(ensemble).all() and np.all((ensemble >= 0) & (ensemble <= 1)), "ensemble probabilities")
        destination = output / f"raw_predictions_{role}.npy"; np.save(destination, ensemble)
        summary = {"status": "PASS_S4R_ENSEMBLE_INFERENCE", "role": role, "rows": len(values), "seeds": list(SEEDS), "ensemble": "arithmetic_mean_probability_precalibration", "retest_truth_read": False, "model_sha256": model_hashes, "prediction_sha256": sha(destination)}
        emit(output / "summary.json", summary); emit(output / "S4R_ENSEMBLE_INFERENCE_PASS.json", {"status": "PASS_S4R_ENSEMBLE_INFERENCE", "role": role, "summary_sha256": sha(output / "summary.json")}); state = summary["status"]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    emit(output / "terminal.json", {"status": state, "role": role, "error": error, "automatic_retry": False})
    if error: raise RuntimeError(error["message"])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--matrices", type=Path, required=True); parser.add_argument("--models", type=Path, required=True); parser.add_argument("--role-index", type=int, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); execute(args.matrices.resolve(), args.models.resolve(), args.role_index, args.output.resolve())


if __name__ == "__main__": main()
