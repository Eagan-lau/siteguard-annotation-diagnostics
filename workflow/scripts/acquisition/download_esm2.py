#!/usr/bin/env python3
"""Freeze pinned ESM2 snapshots into real project-owned files."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--endpoint")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for model in config["models"]:
        destination = args.root / "data" / "raw" / model["relative_dir"]
        destination.mkdir(parents=True, exist_ok=True)
        resolved = snapshot_download(
            repo_id=model["repository"],
            revision=model["revision"],
            local_dir=destination,
            max_workers=8,
            endpoint=args.endpoint,
        )
        required_names = {"config.json", "tokenizer_config.json"}
        present_names = {path.name for path in destination.rglob("*") if path.is_file()}
        has_weights = bool(present_names & {"model.safetensors", "pytorch_model.bin"})
        missing = sorted(required_names - present_names)
        if missing or not has_weights:
            raise RuntimeError(f"incomplete snapshot for {model['name']}: missing={missing}, weights={has_weights}")
        for path in sorted(destination.rglob("*")):
            if not path.is_file() or ".cache" in path.parts:
                continue
            rows.append(
                {
                    "model": model["name"],
                    "repository": model["repository"],
                    "revision": model["revision"],
                    "resolved_local_dir": str(resolved),
                    "relative_path": str(path.relative_to(destination)),
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "huggingface_endpoint": args.endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "files": rows,
        "status": "PASS",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"models": len(config["models"]), "files": len(rows), "status": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

