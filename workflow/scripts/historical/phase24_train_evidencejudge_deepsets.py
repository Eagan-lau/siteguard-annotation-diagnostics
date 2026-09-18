#!/usr/bin/env python3
"""Train a masked DeepSets EvidenceJudge over per-tool prediction evidence."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4")).resolve()
RESULTS = ROOT / "results/phase24"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models/phase24/evidencejudge"
CHECKPOINTS = ROOT / "checkpoints"
SEEDS = (20260819, 20260820, 20260821)
EPOCHS = 20
PATIENCE = 4
BATCH_SIZE = 4096


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ToolSetEncoder(nn.Module):
    def __init__(self, tools: int, tool_numeric_dim: int, global_dim: int) -> None:
        super().__init__()
        self.tool_embedding = nn.Embedding(tools, 8)
        self.phi = nn.Sequential(
            nn.Linear(tool_numeric_dim + 8, 48),
            nn.GELU(),
            nn.LayerNorm(48),
            nn.Dropout(0.10),
            nn.Linear(48, 48),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(96 + global_dim, 96),
            nn.GELU(),
            nn.LayerNorm(96),
            nn.Dropout(0.15),
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, tool_values: torch.Tensor, availability: torch.Tensor, global_values: torch.Tensor) -> torch.Tensor:
        batch, tools, _ = tool_values.shape
        indices = torch.arange(tools, device=tool_values.device).unsqueeze(0).expand(batch, -1)
        encoded = self.phi(torch.cat([tool_values, self.tool_embedding(indices)], dim=2))
        mask = availability.unsqueeze(2)
        count = mask.sum(dim=1).clamp(min=1.0)
        pooled_mean = (encoded * mask).sum(dim=1) / count
        pooled_max = encoded.masked_fill(mask.eq(0), -1e9).max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        return self.head(torch.cat([pooled_mean, pooled_max, global_values], dim=1)).squeeze(1)


def weighted_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    numerator = 0.0
    denominator = 0.0
    with torch.no_grad():
        for tool_values, availability, global_values, labels, weights in loader:
            tool_values = tool_values.to(device)
            availability = availability.to(device)
            global_values = global_values.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            losses = nn.functional.binary_cross_entropy_with_logits(
                model(tool_values, availability, global_values), labels, reduction="none"
            )
            numerator += float((losses * weights).sum().item())
            denominator += float(weights.sum().item())
    return numerator / max(denominator, 1e-12)


def predict(model: nn.Module, arrays: tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device) -> np.ndarray:
    tool_values, availability, global_values = arrays
    dataset = TensorDataset(
        torch.from_numpy(tool_values), torch.from_numpy(availability), torch.from_numpy(global_values)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    result: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for tool_batch, availability_batch, global_batch in loader:
            logits = model(tool_batch.to(device), availability_batch.to(device), global_batch.to(device))
            result.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def main() -> None:
    if not (CHECKPOINTS / "CHECKPOINT_24D2A_ARRAYS_PASS").is_file():
        raise RuntimeError("CHECKPOINT_24D2A_ARRAYS_PASS is required")
    MODELS.mkdir(parents=True, exist_ok=True)
    schema = json.loads((RESULTS / "evidencejudge_feature_schema.json").read_text(encoding="utf-8"))
    methods = list(schema["methods"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction_rows: list[tuple[int, float, float]] = []
    inventory: list[dict[str, object]] = []

    for level in schema["levels"]:
        arrays = np.load(RESULTS / f"evidencejudge_deepsets_arrays__{level.lower()}.npz")
        partition_code = arrays["partition_code"]
        inner = arrays["inner_holdout"]
        train_mask = (partition_code == 0) & ~inner
        early_mask = (partition_code == 0) & inner
        score_mask = partition_code != 0
        train_arrays = (arrays["tool_values"][train_mask], arrays["availability"][train_mask], arrays["global_values"][train_mask])
        early_arrays = (arrays["tool_values"][early_mask], arrays["availability"][early_mask], arrays["global_values"][early_mask])
        score_arrays = (arrays["tool_values"][score_mask], arrays["availability"][score_mask], arrays["global_values"][score_mask])
        train_dataset = TensorDataset(
            torch.from_numpy(train_arrays[0]), torch.from_numpy(train_arrays[1]), torch.from_numpy(train_arrays[2]),
            torch.from_numpy(arrays["labels"][train_mask]), torch.from_numpy(arrays["weights"][train_mask]),
        )
        early_dataset = TensorDataset(
            torch.from_numpy(early_arrays[0]), torch.from_numpy(early_arrays[1]), torch.from_numpy(early_arrays[2]),
            torch.from_numpy(arrays["labels"][early_mask]), torch.from_numpy(arrays["weights"][early_mask]),
        )
        early_loader = DataLoader(early_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        seed_predictions: list[np.ndarray] = []

        for seed in SEEDS:
            seed_everything(seed)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
                num_workers=0, pin_memory=torch.cuda.is_available(),
            )
            model = ToolSetEncoder(len(methods), train_arrays[0].shape[2], train_arrays[2].shape[1]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            best_loss = float("inf")
            best_state: dict[str, torch.Tensor] | None = None
            best_epoch = 0
            stale = 0
            for epoch in range(1, EPOCHS + 1):
                model.train()
                for tool_values, availability, global_values, labels, weights in train_loader:
                    tool_values = tool_values.to(device, non_blocking=True)
                    availability = availability.to(device, non_blocking=True)
                    global_values = global_values.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    weights = weights.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    losses = nn.functional.binary_cross_entropy_with_logits(
                        model(tool_values, availability, global_values), labels, reduction="none"
                    )
                    loss = (losses * weights).sum() / weights.sum().clamp(min=1e-12)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                current = weighted_loss(model, early_loader, device)
                if current < best_loss - 1e-5:
                    best_loss = current
                    best_epoch = epoch
                    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= PATIENCE:
                        break
            if best_state is None:
                raise RuntimeError(f"DeepSets failed to train for {level}, seed {seed}")
            model.load_state_dict(best_state)
            model_path = MODELS / f"{level.lower()}__deepsets_seed{seed}.pt"
            torch.save({
                "state_dict": best_state,
                "methods": methods,
                "tool_numeric_dim": train_arrays[0].shape[2],
                "global_dim": train_arrays[2].shape[1],
                "seed": seed,
                "best_epoch": best_epoch,
                "early_loss": best_loss,
            }, model_path)
            seed_predictions.append(predict(model, score_arrays, device))
            inventory.append({
                "annotation_level": level,
                "model": "DEEPSETS_ROUTER",
                "seed": seed,
                "fit_candidate_rows": int(train_mask.sum()),
                "inner_holdout_candidate_rows": int(early_mask.sum()),
                "best_epoch": best_epoch,
                "inner_weighted_bce": best_loss,
                "device": str(device),
                "model_path": str(model_path),
            })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        stacked = np.stack(seed_predictions, axis=1)
        prediction_rows.extend(zip(
            arrays["candidate_row_id"][score_mask].astype(int).tolist(),
            stacked.mean(axis=1).astype(float).tolist(),
            stacked.std(axis=1).astype(float).tolist(),
        ))

    predictions = pd.DataFrame(prediction_rows, columns=["candidate_row_id", "raw_probability", "seed_probability_sd"])
    predictions["model"] = "DEEPSETS_ROUTER"
    predictions.to_csv(RESULTS / "evidencejudge_raw_predictions_deepsets.tsv.gz", sep="\t", index=False, compression="gzip")
    inventory_frame = pd.DataFrame(inventory)
    inventory_frame.to_csv(RESULTS / "evidencejudge_deepsets_training_inventory.tsv", sep="\t", index=False)
    checks = [
        ("nine_seed_level_runs", len(inventory_frame) == 9, len(inventory_frame)),
        ("gpu_was_used", str(device).startswith("cuda"), str(device)),
        ("probabilities_finite", np.isfinite(predictions["raw_probability"]).all(), len(predictions)),
        ("probabilities_in_unit_interval", predictions["raw_probability"].between(0, 1).all(), [float(predictions["raw_probability"].min()), float(predictions["raw_probability"].max())]),
        ("prediction_grain_unique", not predictions.duplicated(["candidate_row_id"]).any(), len(predictions)),
        ("seed_sd_available", predictions["seed_probability_sd"].notna().all(), len(predictions)),
    ]
    qc = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    qc.to_csv(REPORTS / "phase24_evidencejudge_deepsets_qc.tsv", sep="\t", index=False)
    failures = qc.loc[~qc["passed"].astype(bool), "check"].tolist()
    summary = {
        "phase": "24D2",
        "stage": "deepsets_evidencejudge",
        "status": "PASS" if not failures else "FAIL",
        "device": str(device),
        "seeds": list(SEEDS),
        "training_records": len(inventory_frame),
        "prediction_rows": len(predictions),
        "failures": failures,
    }
    (REPORTS / "phase24_evidencejudge_deepsets_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"DeepSets EvidenceJudge QC failed: {failures}")
    (CHECKPOINTS / "CHECKPOINT_24D2_DEEPSETS_PASS").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
