#!/usr/bin/env python3
"""Train the reaction-aware multi-task SiteGuard global neural network."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 20260820


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
        self.trunk = nn.Sequential(ResidualBlock(width, dropout), ResidualBlock(width, dropout), ResidualBlock(width, dropout))
        self.neck = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(128))
        self.head = nn.Linear(128, 3)

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
    return np.concatenate(outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data/interim/phase11"
    model_dir = root / "models/phase11"
    reports = root / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 11 requires a CUDA GPU")
    device = torch.device("cuda")
    train_x = torch.from_numpy(np.load(work / "train_X.npy"))
    train_y = torch.from_numpy(np.load(work / "train_y.npy"))
    train_weight = torch.from_numpy(np.load(work / "train_weight.npy"))
    eval_x = torch.from_numpy(np.load(work / "eval_X.npy"))
    eval_y = np.load(work / "eval_y.npy")
    eval_split = np.load(work / "eval_split.npy")
    validation_mask = eval_split == 0
    test_mask = eval_split == 1

    weighted_positive = (train_y * train_weight[:, None]).sum(dim=0)
    weighted_negative = ((1 - train_y) * train_weight[:, None]).sum(dim=0)
    positive_weight = torch.sqrt(weighted_negative / weighted_positive).clamp(1.0, 10.0).to(device)
    dataset = TensorDataset(train_x, train_y, train_weight)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    model = SiteGuardGlobalNet(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    history: list[dict] = []
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    checkpoint_path = model_dir / "siteguard_global_multitask.pt"
    eval_tensor = eval_x.pin_memory()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_rows = 0.0, 0
        for features, target, sample_weight in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(features)
                element_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, target, pos_weight=positive_weight, reduction="none"
                ).mean(dim=1)
                probabilities = torch.sigmoid(logits)
                hierarchy_penalty = (
                    torch.relu(probabilities[:, 1] - probabilities[:, 0]).pow(2)
                    + torch.relu(probabilities[:, 2] - probabilities[:, 0]).pow(2)
                )
                loss = ((element_loss + 0.1 * hierarchy_penalty) * sample_weight).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(features)
            total_rows += len(features)
        scheduler.step()
        evaluation = predict(model, eval_tensor, device)
        validation_ap = [average_precision(eval_y[validation_mask, index], evaluation[validation_mask, index]) for index in range(3)]
        selection_score = 0.2 * validation_ap[0] + 0.4 * validation_ap[1] + 0.4 * validation_ap[2]
        row = {
            "epoch": epoch, "train_loss": total_loss / total_rows,
            "validation_auprc": validation_ap,
            "selection_score": selection_score, "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if selection_score > best_score + 1e-5:
            best_score, best_epoch, stale = selection_score, epoch, 0
            torch.save({
                "state_dict": model.state_dict(), "input_features": train_x.shape[1],
                "width": 256, "dropout": 0.15, "targets": ["EC_L3", "EC_L4", "EXACT_RHEA"],
                "seed": SEED, "best_epoch": best_epoch, "validation_selection_score": best_score,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    final_predictions = predict(model, eval_tensor, device)
    np.save(work / "deep_eval_predictions.npy", final_predictions)
    (model_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    config = {
        "architecture": "reaction-aware numerical multi-task residual MLP",
        "input_features": int(train_x.shape[1]), "width": 256, "residual_blocks": 3,
        "neck_width": 128, "dropout": 0.15, "targets": ["EC_L3", "EC_L4", "EXACT_RHEA"],
        "optimizer": "AdamW", "initial_learning_rate": 1e-3, "weight_decay": 1e-4,
        "batch_size": args.batch_size, "maximum_epochs": args.epochs,
        "best_epoch": best_epoch, "validation_selection_score": best_score,
        "positive_weight": positive_weight.detach().cpu().tolist(),
        "hierarchy_penalty_weight": 0.1, "seed": SEED,
        "local_features": "excluded from primary model after CHECKPOINT_09_LOCAL_NULL",
        "test_used_for_model_selection": False,
    }
    (model_dir / "model_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    summary = {
        "phase": 11, "stage": "deep_training", "status": "PASS",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "device": torch.cuda.get_device_name(0), "pytorch_version": torch.__version__,
        "training_rows": len(train_x), "evaluation_rows": len(eval_x),
        "best_epoch": best_epoch, "epochs_run": len(history),
        "best_validation_selection_score": best_score,
        "validation_auprc": [average_precision(eval_y[validation_mask, i], final_predictions[validation_mask, i]) for i in range(3)],
        "test_auprc": [average_precision(eval_y[test_mask, i], final_predictions[test_mask, i]) for i in range(3)],
        "test_used_for_model_selection": False,
    }
    (reports / "phase11_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
