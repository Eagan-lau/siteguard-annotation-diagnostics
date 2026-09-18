#!/usr/bin/env python3
"""Train one additive Phase11 successor seed without touching frozen outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from phase11_train_deep import SiteGuardGlobalNet, average_precision, predict


INPUT_SHA256 = {
    "train_X.npy": "e222e675dad8a0b66c241247bbb7054033f2697b6251827238395077f9d92f37",
    "train_y.npy": "dfc0ec3226ef1ba0faeb39dd374f0fa961078c5b2a91376ef671ebbcd627d24f",
    "train_weight.npy": "36f41ff82d3e3070dc6cb9fcfb3a2229dd84ee6b5bd0f00bd23a11dbff420151",
    "eval_X.npy": "3e94d9221f1922439ac3df4692f94ac8c692d57cd1d0e5c55787d3dbb2a5ed21",
    "eval_y.npy": "b3548c6893e251aa2ba5193d2e9f5d0537adca3de7df2d7aa28f35597f345465",
    "eval_split.npy": "ee2952ddb9826f3880fd9d1aa3f83fa52b4b3aaea694a9f6bda05a34fc2e8180",
}
ALLOWED_NEW_SEEDS = {20260819, 20260821}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json_exclusive(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=sorted(ALLOWED_NEW_SEEDS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()

    root = args.project_root.resolve()
    work = root / "data/interim/phase11"
    output = args.output_dir.resolve()
    output.relative_to(root)
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    write_json_exclusive(output / "RUNNING.json", {
        "format": "siteguard.phase321.multiseed-running.v1",
        "seed": args.seed,
        "started_utc": started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "status": "RUNNING_NOT_AUTHORITY",
    })

    observed_inputs: dict[str, dict[str, object]] = {}
    for name, expected in INPUT_SHA256.items():
        path = work / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = digest(path)
        if observed != expected:
            raise ValueError(f"frozen Phase11 input identity mismatch: {name}")
        observed_inputs[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("Phase321 requires a CUDA GPU")
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
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y, train_weight),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    model = SiteGuardGlobalNet(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    history: list[dict[str, object]] = []
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    checkpoint_path = output / f"siteguard_global_multitask_seed_{seed}.pt"
    eval_tensor = eval_x.pin_memory()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
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
        validation_ap = [
            average_precision(eval_y[validation_mask, index], evaluation[validation_mask, index])
            for index in range(3)
        ]
        selection_score = 0.2 * validation_ap[0] + 0.4 * validation_ap[1] + 0.4 * validation_ap[2]
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_rows,
            "validation_auprc": validation_ap,
            "selection_score": selection_score,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if selection_score > best_score + 1e-5:
            best_score, best_epoch, stale = selection_score, epoch, 0
            torch.save({
                "state_dict": model.state_dict(),
                "input_features": train_x.shape[1],
                "width": 256,
                "dropout": 0.15,
                "targets": ["EC_L3", "EC_L4", "EXACT_RHEA"],
                "seed": seed,
                "best_epoch": best_epoch,
                "validation_selection_score": best_score,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    final_predictions = predict(model, eval_tensor, device)
    predictions_path = output / f"deep_eval_predictions_seed_{seed}.npy"
    np.save(predictions_path, final_predictions)
    write_json_exclusive(output / "training_history.json", history)
    config = {
        "format": "siteguard.phase321.multiseed-model-config.v1",
        "architecture": "reaction-aware numerical multi-task residual MLP",
        "input_features": int(train_x.shape[1]),
        "width": 256,
        "residual_blocks": 3,
        "neck_width": 128,
        "dropout": 0.15,
        "targets": ["EC_L3", "EC_L4", "EXACT_RHEA"],
        "optimizer": "AdamW",
        "initial_learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": args.batch_size,
        "maximum_epochs": args.epochs,
        "best_epoch": best_epoch,
        "validation_selection_score": best_score,
        "positive_weight": positive_weight.detach().cpu().tolist(),
        "hierarchy_penalty_weight": 0.1,
        "seed": seed,
        "local_features": "excluded after CHECKPOINT_09_LOCAL_NULL",
        "test_used_for_model_selection": False,
        "threshold_refit_performed": False,
    }
    write_json_exclusive(output / "model_config.json", config)
    summary = {
        "format": "siteguard.phase321.multiseed-run-summary.v1",
        "phase": 321,
        "status": "PASS_PHASE321_SINGLE_SEED_ADDITIVE_CANDIDATE_NOT_MODEL_SELECTION_AUTHORITY",
        "seed": seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"),
        "device": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "training_rows": len(train_x),
        "evaluation_rows": len(eval_x),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_selection_score": best_score,
        "validation_auprc": [average_precision(eval_y[validation_mask, i], final_predictions[validation_mask, i]) for i in range(3)],
        "test_auprc": [average_precision(eval_y[test_mask, i], final_predictions[test_mask, i]) for i in range(3)],
        "test_used_for_model_selection": False,
        "threshold_refit_performed": False,
        "frozen_inputs": observed_inputs,
        "artifacts": {
            "model": {"path": str(checkpoint_path), "bytes": checkpoint_path.stat().st_size, "sha256": digest(checkpoint_path)},
            "predictions": {"path": str(predictions_path), "bytes": predictions_path.stat().st_size, "sha256": digest(predictions_path)},
        },
    }
    write_json_exclusive(output / "run_summary.json", summary)
    write_json_exclusive(output / "CHECKPOINT_SINGLE_SEED_PASS.json", {
        "format": "siteguard.phase321.single-seed-checkpoint.v1",
        "status": summary["status"],
        "seed": seed,
        "summary_sha256": digest(output / "run_summary.json"),
        "terminal": False,
    })
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
