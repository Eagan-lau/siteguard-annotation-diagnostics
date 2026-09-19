"""Command-line adapter from frozen mainstream-tool outputs to EvidenceJudge."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evidence_features import (
    assemble_hit_anchored_features,
    read_adapter_input,
    write_feature_table,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen truth-free 83-feature HIT-EC candidate table from "
            "version-locked HIT-EC, CLEAN, and supporting-tool outputs."
        )
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--tool-predictions", type=Path, required=True)
    parser.add_argument("--hit-ec", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--tool-versions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-spec", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    inputs = {
        "cohort": args.cohort.resolve(),
        "metadata": args.metadata.resolve(),
        "tool_predictions": args.tool_predictions.resolve(),
        "hit_ec": args.hit_ec.resolve(),
        "clean": args.clean.resolve(),
        "tool_versions": args.tool_versions.resolve(),
    }
    missing = [name for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing adapter inputs: {missing}")
    output = args.output.resolve()
    manifest = args.manifest.resolve()
    collisions = [path for path in (output, manifest) if path.exists()]
    if collisions and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing outputs: {collisions}")
    versions: dict[str, Any] = json.loads(inputs["tool_versions"].read_text(encoding="utf-8"))

    assembled, qc = assemble_hit_anchored_features(
        read_adapter_input(inputs["cohort"]),
        read_adapter_input(inputs["metadata"]),
        read_adapter_input(inputs["tool_predictions"]),
        read_adapter_input(inputs["hit_ec"]),
        read_adapter_input(inputs["clean"]),
        tool_versions=versions,
        spec_path=args.model_spec,
    )
    write_feature_table(assembled, output)
    result = {
        "format": "siteguard.evidencejudge.feature-assembly.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        **qc,
        "input_sha256": {str(path): sha256(path) for path in inputs.values()},
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256(output),
        "manifest": str(manifest),
        "deployment_boundary": (
            "FEATURE_ASSEMBLY_ONLY_NO_PRECISION_ESTIMATE_NO_COHORT_AUTHORIZATION_"
            "RUN_EVIDENCE_JUDGE_WITH_UNKNOWN_UNTIL_EXTERNAL_VALIDATION"
        ),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "queries": result["queries"],
        "rows": result["rows"],
        "features": result["feature_count"],
        "truth_or_outcome_fields_used": result["truth_or_outcome_fields_used"],
        "output": result["output"],
        "manifest": result["manifest"],
        "next_safe_step": "siteguard evidence-judge --cohort-status unknown",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
