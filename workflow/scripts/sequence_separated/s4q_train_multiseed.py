#!/usr/bin/env python3
"""Train the fixed residual MLP on the sequence-isolated S4 matrices."""
import argparse
import hashlib
import json
import os
import random
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEEDS = (20260819, 20260820, 20260821)
TARGETS = ("EC_L3", "EC_L4", "EXACT_RHEA")


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


def average_precision(y, score, mask):
    y, score = y[mask].astype(np.float64), score[mask]
    positives = float(y.sum())
    if positives <= 0: return float("nan")
    ordered = y[np.argsort(-score, kind="mergesort")]
    return float(np.dot(np.cumsum(ordered) / np.arange(1, len(ordered) + 1), ordered) / positives)


@torch.no_grad()
def predict(model, values, device, batch_size=65536):
    model.eval(); output = []
    for start in range(0, len(values), batch_size):
        batch = torch.from_numpy(np.asarray(values[start:start + batch_size])).to(device)
        output.append(torch.sigmoid(model(batch)).cpu().numpy().astype(np.float32))
    return np.concatenate(output)


def execute(matrices, seed_index, output, epochs, batch_size, patience):
    require(0 <= seed_index < len(SEEDS) and output.parent.is_dir() and not output.exists(), "seed/output")
    seed = SEEDS[seed_index]; output.mkdir(exist_ok=False); emit(output / "reservation.json", {"status": "ONE_S4Q_MODEL_SEED_RESERVED", "seed": seed, "automatic_retry": False})
    state, error = "FAIL_CLOSED", None
    try:
        require(torch.cuda.is_available(), "CUDA required")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        train_x = np.load(matrices / "X_TRAIN.npy", mmap_mode="r"); train_y_np = np.load(matrices / "y_TRAIN.npy"); train_mask_np = np.load(matrices / "mask_TRAIN.npy"); train_weight_np = np.load(matrices / "weight_TRAIN.npy")
        dev_x = np.load(matrices / "X_DEV.npy", mmap_mode="r"); dev_y = np.load(matrices / "y_DEV.npy"); dev_mask = np.load(matrices / "mask_DEV.npy").astype(bool)
        require(train_x.ndim == 2 and train_y_np.shape == train_mask_np.shape == (len(train_x), 3) and len(train_weight_np) == len(train_x), "TRAIN matrices")
        require(dev_x.shape[1] == train_x.shape[1] and dev_y.shape == dev_mask.shape == (len(dev_x), 3), "DEV matrices")
        train_y = torch.from_numpy(train_y_np); train_mask = torch.from_numpy(train_mask_np); train_weight = torch.from_numpy(train_weight_np)
        weighted_positive = (train_y * train_mask * train_weight[:, None]).sum(dim=0)
        weighted_negative = ((1 - train_y) * train_mask * train_weight[:, None]).sum(dim=0)
        positive_weight = torch.sqrt(weighted_negative / weighted_positive.clamp_min(1)).clamp(1.0, 10.0)
        dataset = TensorDataset(torch.from_numpy(np.asarray(train_x)), train_y, train_mask, train_weight)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=4, pin_memory=True, persistent_workers=True)
        device = torch.device("cuda"); model = SiteGuardGlobalNet(train_x.shape[1]).to(device); positive_weight = positive_weight.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5); scaler = torch.cuda.amp.GradScaler()
        best_score, best_epoch, stale, history = -float("inf"), 0, 0, []
        checkpoint = output / "siteguard_global_multitask.pt"
        for epoch in range(1, epochs + 1):
            model.train(); total_loss = total_rows = 0
            for features, target, mask, weight in loader:
                features, target, mask, weight = features.to(device, non_blocking=True), target.to(device, non_blocking=True), mask.to(device, non_blocking=True), weight.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(features); element = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight, reduction="none")
                    row_loss = (element * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                    probabilities = torch.sigmoid(logits); hierarchy = torch.relu(probabilities[:, 1] - probabilities[:, 0]).pow(2)
                    loss = ((row_loss + 0.1 * hierarchy) * weight).mean()
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(optimizer); scaler.update()
                total_loss += float(loss.detach()) * len(features); total_rows += len(features)
            scheduler.step(); dev_prediction = predict(model, dev_x, device)
            aps = [average_precision(dev_y[:, i], dev_prediction[:, i], dev_mask[:, i]) for i in range(3)]
            selection = 0.2 * aps[0] + 0.4 * aps[1] + 0.4 * aps[2]
            record = {"epoch": epoch, "train_loss": total_loss / total_rows, "dev_auprc": aps, "selection_score": selection, "learning_rate": optimizer.param_groups[0]["lr"]}; history.append(record); print(json.dumps(record), flush=True)
            if selection > best_score + 1e-5:
                best_score, best_epoch, stale = selection, epoch, 0
                torch.save({"state_dict": model.state_dict(), "input_features": train_x.shape[1], "width": 256, "dropout": 0.15, "targets": TARGETS, "seed": seed, "best_epoch": epoch, "dev_selection_score": selection}, checkpoint)
            else:
                stale += 1
                if stale >= patience: break
        saved = torch.load(checkpoint, map_location=device); model.load_state_dict(saved["state_dict"]); final_dev = predict(model, dev_x, device); np.save(output / "predictions_DEV.npy", final_dev)
        emit(output / "training_history.json", {"epochs": history})
        summary = {"status": "PASS_S4Q_MODEL_SEED", "seed": seed, "best_epoch": best_epoch, "epochs_run": len(history), "dev_selection_score": best_score, "dev_auprc": [average_precision(dev_y[:, i], final_dev[:, i], dev_mask[:, i]) for i in range(3)], "training_rows": len(train_x), "dev_rows": len(dev_x), "input_features": train_x.shape[1], "architecture": "fixed_3_block_residual_MLP_256_neck128", "exact_rhea_masked_when_not_evaluable": True, "retest_truth_read": False, "device": torch.cuda.get_device_name(0), "torch_version": torch.__version__, "model_sha256": sha(checkpoint)}
        emit(output / "summary.json", summary); emit(output / "S4Q_MODEL_SEED_PASS.json", {"status": "PASS_S4Q_MODEL_SEED", "seed": seed, "summary_sha256": sha(output / "summary.json")}); state = summary["status"]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    emit(output / "terminal.json", {"status": state, "seed": seed, "error": error, "automatic_retry": False})
    if error: raise RuntimeError(error["message"])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--matrices", type=Path, required=True); parser.add_argument("--seed-index", type=int, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=8192); parser.add_argument("--patience", type=int, default=6); args = parser.parse_args(); execute(args.matrices.resolve(), args.seed_index, args.output.resolve(), args.epochs, args.batch_size, args.patience)


if __name__ == "__main__": main()
