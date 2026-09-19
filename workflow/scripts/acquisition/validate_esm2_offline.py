#!/usr/bin/env python3
"""Load and execute every frozen ESM2 snapshot with networking disabled."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    for item in config["models"]:
        model_dir = args.root / "data" / "raw" / item["relative_dir"]
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
        encoded = tokenizer("MKTAYIAKQRQISFVKSHFSRQ", return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded).last_hidden_state
        rows.append(
            {
                "name": item["name"],
                "repository": item["repository"],
                "revision": item["revision"],
                "local_dir": str(model_dir),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "output_shape": list(output.shape),
                "finite": bool(torch.isfinite(output).all().item()),
                "status": "PASS" if torch.isfinite(output).all().item() else "FAIL",
            }
        )
        del output, encoded, model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "offline_environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "models": rows,
        "status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "device": payload["device"], "models": len(rows)}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

