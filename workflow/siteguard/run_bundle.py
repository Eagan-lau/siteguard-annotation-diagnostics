"""Atomic writer and reader for the eight-file SiteGuard V4 run contract."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from . import __version__


BUNDLE_FILES = (
    "predictions.tsv",
    "predictions.parquet",
    "references.tsv",
    "activities.tsv",
    "evidence.json",
    "report.html",
    "run_manifest.json",
    "warnings.log",
)
PROJECT_PRIMARY_SEED = 20260819
ALLOWED_WORKFLOWS = frozenset({"build-db", "annotate", "transfer", "audit", "report"})
PASS_STATUS = "PASS_COMPLETE_EIGHT_FILE_BUNDLE"
NON_SELF_BUNDLE_FILES = tuple(name for name in BUNDLE_FILES if name != "run_manifest.json")
DECLARED_DEPENDENCIES = (
    "joblib",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "lightgbm",
    "torch",
    "transformers",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _validate_frames(predictions: pd.DataFrame, references: pd.DataFrame, activities: pd.DataFrame) -> None:
    for name, frame in (
        ("predictions", predictions), ("references", references), ("activities", activities)
    ):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        if len(set(frame.columns)) != len(frame.columns):
            raise ValueError(f"{name} contains duplicate columns")
    if "query_protein_id" not in predictions.columns:
        raise ValueError("predictions must contain query_protein_id")
    if predictions["query_protein_id"].astype(str).str.strip().eq("").any():
        raise ValueError("predictions contains an empty query_protein_id")


def _prediction_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load TSV/Parquet and derive the canonical TSV view of Parquet values."""
    tsv = pd.read_csv(path / "predictions.tsv", sep="\t", dtype=str, keep_default_na=False)
    parquet = pd.read_parquet(path / "predictions.parquet")
    canonical_text = io.StringIO()
    parquet.to_csv(canonical_text, sep="\t", index=False, na_rep="NA")
    canonical_text.seek(0)
    canonical_tsv = pd.read_csv(canonical_text, sep="\t", dtype=str, keep_default_na=False)
    return tsv, parquet, canonical_tsv


def _validate_prediction_parity(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tsv, parquet, canonical_tsv = _prediction_tables(path)
    if list(tsv.columns) != list(parquet.columns) or len(tsv) != len(parquet):
        raise ValueError("Bundle predictions TSV/Parquet schema or row-count mismatch")
    if "query_protein_id" not in tsv.columns:
        raise ValueError("Bundle predictions are missing query_protein_id")
    if tsv["query_protein_id"].astype(str).tolist() != parquet["query_protein_id"].astype(str).tolist():
        raise ValueError("Bundle predictions TSV/Parquet query order mismatch")
    if not tsv.equals(canonical_tsv):
        raise ValueError("Bundle predictions TSV/Parquet cell content mismatch")
    return tsv, parquet


def _software_identities() -> dict[str, dict[str, Any]]:
    package_root = Path(__file__).resolve().parent
    files = sorted(package_root.glob("*.py"))
    bundled_spec = package_root / "data/evidencejudge_v2.json"
    if bundled_spec.is_file():
        files.append(bundled_spec)
    identities = {
        path.relative_to(package_root.parent).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    }
    pyproject = package_root.parent / "pyproject.toml"
    if pyproject.is_file():
        identities["pyproject.toml"] = {
            "bytes": pyproject.stat().st_size,
            "sha256": sha256_file(pyproject),
        }
    return dict(sorted(identities.items()))


def _source_tree_sha256(software_files: Mapping[str, Mapping[str, Any]]) -> str:
    payload = json.dumps(software_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _code_state(software_files: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Describe real code state without inventing a VCS identity."""
    package_root = Path(__file__).resolve().parent
    repository = next((parent for parent in (package_root.parent, *package_root.parents) if (parent / ".git").exists()), None)
    source_digest = _source_tree_sha256(software_files)
    if repository is None:
        return {
            "status": "NOT_A_GIT_REPOSITORY",
            "commit": "NOT_A_GIT_REPOSITORY",
            "source_tree_sha256": source_digest,
            "source_tree_file_count": len(software_files),
            "source_tree_scope": "PACKAGED_SITEGUARD_SOURCE_PLUS_PYPROJECT",
        }
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {
            "status": "GIT_METADATA_UNAVAILABLE",
            "commit": "GIT_METADATA_UNAVAILABLE",
            "source_tree_sha256": source_digest,
            "source_tree_file_count": len(software_files),
            "source_tree_scope": "PACKAGED_SITEGUARD_SOURCE_PLUS_PYPROJECT",
            "detail": type(error).__name__,
        }
    return {
        "status": "GIT_WORKTREE_DIRTY" if dirty else "GIT_WORKTREE_CLEAN",
        "commit": commit,
        "dirty": dirty,
        "source_tree_sha256": source_digest,
        "source_tree_file_count": len(software_files),
        "source_tree_scope": "PACKAGED_SITEGUARD_SOURCE_PLUS_PYPROJECT",
    }


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in DECLARED_DEPENDENCIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "NOT_INSTALLED"
    return versions


def write_bundle_atomic(
    output_dir: str | Path,
    *,
    workflow: str,
    predictions: pd.DataFrame,
    references: pd.DataFrame,
    activities: pd.DataFrame,
    evidence: Mapping[str, Any],
    report_html: str,
    inputs: Mapping[str, str | Path],
    assets: Mapping[str, str | Path] | None = None,
    parameters: Mapping[str, Any] | None = None,
    workflow_seed: int | None = None,
    production_model_seed: int | None = None,
    warnings: Sequence[str] = (),
) -> Path:
    """Write exactly eight files to a new directory, then install it atomically.

    The target is exclusive-create. Any validation or write failure removes only the
    sibling temporary directory and never exposes a partial official bundle.
    """
    target = Path(output_dir).resolve()
    if workflow not in ALLOWED_WORKFLOWS:
        raise ValueError(f"Unsupported SiteGuard workflow: {workflow}")
    if target.exists():
        raise FileExistsError(f"Output directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    _validate_frames(predictions, references, activities)
    input_identities = {name: file_identity(path) for name, path in sorted(inputs.items())}
    asset_identities = {name: file_identity(path) for name, path in sorted((assets or {}).items())}
    lock_path = target.parent / f".{target.name}.siteguard-create.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise FileExistsError(f"Output directory is reserved by another writer: {target}") from error
    stage: Path | None = None
    try:
        if target.exists():
            raise FileExistsError(f"Output directory already exists: {target}")
        stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
        predictions.to_csv(stage / "predictions.tsv", sep="\t", index=False, na_rep="NA")
        predictions.to_parquet(stage / "predictions.parquet", index=False)
        references.to_csv(stage / "references.tsv", sep="\t", index=False, na_rep="NA")
        activities.to_csv(stage / "activities.tsv", sep="\t", index=False, na_rep="NA")
        (stage / "evidence.json").write_text(
            json.dumps(_jsonable(dict(evidence)), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (stage / "report.html").write_text(report_html, encoding="utf-8")
        (stage / "warnings.log").write_text(
            "".join(f"{str(item).rstrip()}\n" for item in warnings), encoding="utf-8"
        )

        written = list(NON_SELF_BUNDLE_FILES)
        output_identities = {
            name: {
                "bytes": (stage / name).stat().st_size,
                "sha256": sha256_file(stage / name),
            }
            for name in written
        }
        software_files = _software_identities()
        code_state = _code_state(software_files)
        manifest = {
            "format": "siteguard.v4.eight-file-run-manifest.v1",
            "workflow": workflow,
            "status": PASS_STATUS,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "siteguard_version": __version__,
            "software_files": software_files,
            "code_commit": code_state["commit"],
            "code_state": code_state,
            "dependency_versions": _dependency_versions(),
            "dependency_inventory_scope": "DECLARED_RUNTIME_AND_OPTIONAL_DEPENDENCIES",
            "truth_free_inference": True,
            "project_primary_seed": PROJECT_PRIMARY_SEED,
            "workflow_seed": int(workflow_seed) if workflow_seed is not None else None,
            "production_model_seed": (
                int(production_model_seed) if production_model_seed is not None else None
            ),
            "parameters": _jsonable(parameters or {}),
            "hardware": {
                "machine": platform.machine(),
                "processor": platform.processor(),
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            "inputs": input_identities,
            "assets": asset_identities,
            "outputs_excluding_self": output_identities,
            "contract_files": list(BUNDLE_FILES),
            "warnings": list(warnings),
        }
        (stage / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        actual = sorted(path.name for path in stage.iterdir())
        if actual != sorted(BUNDLE_FILES):
            raise RuntimeError(f"Eight-file contract violation: {actual}")
        try:
            _validate_prediction_parity(stage)
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        if target.exists():
            raise FileExistsError(f"Output directory appeared during assembly: {target}")
        os.rename(stage, target)
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
    return target


def read_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"SiteGuard bundle is not a directory: {root}")
    entries = list(root.iterdir())
    actual = sorted(path.name for path in entries)
    if actual != sorted(BUNDLE_FILES):
        raise ValueError(f"Not an exact SiteGuard eight-file bundle: {root}")
    invalid_entry = next((path for path in entries if not path.is_file() or path.is_symlink()), None)
    if invalid_entry is not None:
        raise ValueError(f"Bundle entry must be a regular file: {invalid_entry.name}")
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "siteguard.v4.eight-file-run-manifest.v1":
        raise ValueError("Unsupported run manifest format")
    if manifest.get("status") != PASS_STATUS:
        raise ValueError("Bundle manifest status is not the complete PASS status")
    if manifest.get("workflow") not in ALLOWED_WORKFLOWS:
        raise ValueError("Bundle manifest workflow is not allowed")
    if manifest.get("contract_files") != list(BUNDLE_FILES):
        raise ValueError("Bundle manifest contract_files is not the exact ordered contract")
    outputs = manifest.get("outputs_excluding_self")
    if not isinstance(outputs, dict) or set(outputs) != set(NON_SELF_BUNDLE_FILES) or len(outputs) != len(NON_SELF_BUNDLE_FILES):
        raise ValueError("Bundle manifest must bind exactly the seven non-self outputs")
    for name in NON_SELF_BUNDLE_FILES:
        identity = outputs[name]
        if not isinstance(identity, dict) or set(identity) != {"bytes", "sha256"}:
            raise ValueError(f"Invalid output identity record: {name}")
        if not isinstance(identity["bytes"], int) or identity["bytes"] < 0:
            raise ValueError(f"Invalid output byte count: {name}")
        if not isinstance(identity["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is None:
            raise ValueError(f"Invalid output SHA-256: {name}")
        path = root / name
        if path.stat().st_size != int(identity["bytes"]) or sha256_file(path) != identity["sha256"]:
            raise ValueError(f"Bundle file identity mismatch: {name}")
    _, predictions_parquet = _validate_prediction_parity(root)
    return {
        "root": root,
        "predictions": predictions_parquet,
        "references": pd.read_csv(root / "references.tsv", sep="\t", dtype=str, keep_default_na=False),
        "activities": pd.read_csv(root / "activities.tsv", sep="\t", dtype=str, keep_default_na=False),
        "evidence": json.loads((root / "evidence.json").read_text(encoding="utf-8")),
        "manifest": manifest,
        "warnings": (root / "warnings.log").read_text(encoding="utf-8").splitlines(),
    }
