#!/usr/bin/env python3
"""Train the SiteGuard residual MLP using T0 functional knowledge only."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 20260821
LEVELS = ["EC_L3", "EC_L4", "EXACT_RHEA"]


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


class SiteGuardTemporalNet(nn.Module):
    def __init__(self, input_features: int, width: int = 256, dropout: float = 0.15) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_features), nn.Linear(input_features, width), nn.GELU())
        self.trunk = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(3)])
        self.neck = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(128))
        self.head = nn.Linear(128, len(LEVELS))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.neck(self.trunk(self.input(values))))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    outcome = y[order].astype(np.float64)
    positives = float(outcome.sum())
    if positives <= 0:
        return float("nan")
    precision = np.cumsum(outcome) / np.arange(1, len(outcome) + 1)
    return float(np.dot(precision, outcome) / positives)


@torch.no_grad()
def predict(model: nn.Module, values: torch.Tensor, device: torch.device, batch_size: int = 65_536) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = values[start:start + batch_size].to(device, non_blocking=True)
        outputs.append(torch.sigmoid(model(batch)).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs) if outputs else np.empty((0, len(LEVELS)), dtype=np.float32)


def masked_ap(y: np.ndarray, score: np.ndarray, mask: np.ndarray) -> list[float]:
    return [average_precision(y[mask[:, index], index], score[mask[:, index], index]) for index in range(len(LEVELS))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data/interim/phase13"
    models = root / "models/phase13"
    reports = root / "reports"
    models.mkdir(parents=True, exist_ok=True)
    if not (root / "checkpoints/CHECKPOINT_12_PASS").is_file():
        raise RuntimeError("CHECKPOINT_12_PASS is required")
    if not (reports / "phase13_prepare_summary.json").is_file():
        raise RuntimeError("Phase 13 preparation summary is missing")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 13 temporal deep training requires CUDA")
    device = torch.device("cuda")
    train_x = torch.from_numpy(np.load(work / "train_X.npy"))
    train_y = torch.from_numpy(np.load(work / "train_y.npy"))
    train_mask = torch.from_numpy(np.load(work / "train_mask.npy"))
    validation_x = torch.from_numpy(np.load(work / "validation_X.npy"))
    validation_y = np.load(work / "validation_y.npy")
    validation_mask = np.load(work / "validation_mask.npy").astype(bool)
    temporal_x = torch.from_numpy(np.load(work / "temporal_X.npy"))

    weighted_positive = (train_y * train_mask.float()).sum(dim=0)
    weighted_negative = ((1 - train_y) * train_mask.float()).sum(dim=0)
    positive_weight = torch.sqrt(weighted_negative / weighted_positive.clamp_min(1)).clamp(1.0, 10.0).to(device)
    dataset = TensorDataset(train_x, train_y, train_mask)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    model = SiteGuardTemporalNet(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    history: list[dict] = []
    best_score, best_epoch, stale = -float("inf"), 0, 0
    checkpoint_path = models / "siteguard_t0_multitask.pt"
    validation_tensor = validation_x.pin_memory()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_rows = 0.0, 0
        for features, targets, masks in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(features)
                element = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=positive_weight, reduction="none",
                )
                mask_float = masks.float()
                supervised = (element * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
                probability = torch.sigmoid(logits)
                hierarchy = (
                    torch.relu(probability[:, 1] - probability[:, 0]).pow(2)
                    + torch.relu(probability[:, 2] - probability[:, 0]).pow(2)
                )
                loss = (supervised + 0.1 * hierarchy).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(features)
            total_rows += len(features)
        scheduler.step()
        validation_prediction = predict(model, validation_tensor, device)
        validation_ap = masked_ap(validation_y, validation_prediction, validation_mask)
        selection_score = 0.2 * validation_ap[0] + 0.4 * validation_ap[1] + 0.4 * validation_ap[2]
        record = {
            "epoch": epoch, "train_loss": total_loss / total_rows,
            "t0_validation_auprc": validation_ap, "selection_score": selection_score,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if selection_score > best_score + 1e-5:
            best_score, best_epoch, stale = selection_score, epoch, 0
            torch.save({
                "state_dict": model.state_dict(), "input_features": train_x.shape[1],
                "width": 256, "dropout": 0.15, "targets": LEVELS, "seed": SEED,
                "best_epoch": best_epoch, "t0_validation_selection_score": best_score,
                "functional_training_snapshot": "T0 UniProt 2023_01 + Rhea 126",
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    validation_prediction = predict(model, validation_tensor, device)
    temporal_prediction = predict(model, temporal_x.pin_memory(), device)
    np.save(work / "deep_validation_predictions.npy", validation_prediction)
    np.save(work / "deep_temporal_predictions.npy", temporal_prediction)
    (models / "temporal_training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    config = {
        "architecture": "function-time-frozen reaction-aware multi-task residual MLP",
        "input_features": int(train_x.shape[1]), "width": 256, "residual_blocks": 3,
        "neck_width": 128, "dropout": 0.15, "targets": LEVELS,
        "optimizer": "AdamW", "batch_size": args.batch_size, "best_epoch": best_epoch,
        "positive_weight": positive_weight.detach().cpu().tolist(), "hierarchy_penalty_weight": 0.1,
        "seed": SEED, "functional_training_snapshot": "UniProt 2023_01 + Rhea 126",
        "t1_ground_truth_used_for_training_or_model_selection": False,
    }
    (models / "temporal_model_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    summary = {
        "phase": 13, "stage": "train_t0_deep", "status": "PASS",
        "slurm_job_id": os.getenv("SLURM_JOB_ID", "NA"), "device": torch.cuda.get_device_name(0),
        "training_rows": len(train_x), "validation_rows": len(validation_x),
        "temporal_prediction_rows": len(temporal_x), "best_epoch": best_epoch,
        "epochs_run": len(history), "best_t0_validation_selection_score": best_score,
        "t0_validation_auprc": masked_ap(validation_y, validation_prediction, validation_mask),
        "t1_used_for_model_selection": False,
    }
    (reports / "phase13_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
