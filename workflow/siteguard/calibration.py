"""Identity-bound loading of the three frozen SiteGuard calibrators.

The Phase12 joblib streams were created by NumPy 2 and contain the private
module reference ``numpy._core.multiarray``.  NumPy 1.25 exposes the same
implementation at ``numpy.core.multiarray``.  This module supplies the two
minimal, temporary aliases required for those exact frozen local assets.

It is intentionally not a general compatibility unpickler.  Joblib/pickle
must only be used for trusted files, so callers select a registered logical
level rather than supplying an arbitrary pickle path.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import BinaryIO, Iterator

import joblib
import numpy as np


PHASE319_MANIFEST_PATH = "models/phase319_runtime_asset_manifest.json"
PHASE319_MANIFEST_BYTES = 14_218
PHASE319_MANIFEST_SHA256 = "54aabe0a8aaf7b7989ce8e3f9d13e40be8bf535b8ec5d75c2ee7d319760175a8"
PHASE319_MANIFEST_FORMAT = "siteguard.phase319.runtime-asset-manifest.v1"
PHASE319_MANIFEST_STATUS = "FROZEN_PHASE319_IDENTITY_COMPLETE_RUNTIME_ASSET_MANIFEST"


@dataclass(frozen=True)
class FrozenCalibratorAsset:
    path: str
    bytes: int
    sha256: str


FROZEN_CALIBRATORS = {
    "EC_L3": FrozenCalibratorAsset(
        "models/phase12/isotonic_EC_L3.joblib",
        2_583,
        "6152ed122f5c1e408197ea8bc655f8a8a8db2a4d034b7c83147fe50489b8b478",
    ),
    "EC_L4": FrozenCalibratorAsset(
        "models/phase12/isotonic_EC_L4.joblib",
        2_094,
        "1287d4a794b450d6d5ccd699fa174a5303c764b56538b7bcda5651f5aa2fccaf",
    ),
    "EXACT_RHEA": FrozenCalibratorAsset(
        "models/phase12/isotonic_EXACT_RHEA.joblib",
        1_888,
        "a0bbbc88fd006427ba4fdf9d2270513da72fa6e416ed4193972012010c630837",
    ),
}

# Ordered: the parent package must be present before its observed child module.
NUMPY_CORE_COMPAT_ALIASES = (
    ("numpy._core", "numpy.core"),
    ("numpy._core.multiarray", "numpy.core.multiarray"),
)

_ALIAS_LOCK = threading.RLock()
_MISSING = object()


def _sha256_handle(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _read_identity_bound_manifest(root: Path) -> dict[str, object]:
    path = root / PHASE319_MANIFEST_PATH
    if path.is_symlink():
        raise RuntimeError(f"Frozen runtime manifest must not be a symlink: {path}")
    with path.open("rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"Frozen runtime manifest is not a regular file: {path}")
        if metadata.st_size != PHASE319_MANIFEST_BYTES:
            raise RuntimeError(f"Frozen runtime manifest byte mismatch: {path}")
        if _sha256_handle(handle) != PHASE319_MANIFEST_SHA256:
            raise RuntimeError(f"Frozen runtime manifest SHA-256 mismatch: {path}")
        payload = json.load(handle)
    if payload.get("format") != PHASE319_MANIFEST_FORMAT:
        raise RuntimeError("Frozen runtime manifest format mismatch")
    if payload.get("status") != PHASE319_MANIFEST_STATUS:
        raise RuntimeError("Frozen runtime manifest status mismatch")
    return payload


def _verify_manifest_asset(payload: dict[str, object], asset: FrozenCalibratorAsset) -> None:
    files = payload.get("files")
    if not isinstance(files, list):
        raise RuntimeError("Frozen runtime manifest files are malformed")
    matches = [entry for entry in files if isinstance(entry, dict) and entry.get("path") == asset.path]
    if len(matches) != 1:
        raise RuntimeError(f"Frozen calibrator must have exactly one manifest entry: {asset.path}")
    entry = matches[0]
    if entry.get("bytes") != asset.bytes or entry.get("sha256") != asset.sha256:
        raise RuntimeError(f"Frozen calibrator manifest identity mismatch: {asset.path}")
    if entry.get("validation") != {"kind": "opaque"}:
        raise RuntimeError(f"Frozen calibrator manifest validation mismatch: {asset.path}")


@contextmanager
def _temporary_numpy_core_aliases() -> Iterator[None]:
    """Install only missing Phase329 aliases and restore exact prior state."""

    with _ALIAS_LOCK:
        # Snapshot every target before the first import.  This guarantees exact
        # restoration even if importing the first source raises before the
        # installation loop reaches a later, pre-existing target.
        previous = {
            target: sys.modules.get(target, _MISSING)
            for target, _source in NUMPY_CORE_COMPAT_ALIASES
        }
        try:
            for target, source in NUMPY_CORE_COMPAT_ALIASES:
                if previous[target] is _MISSING:
                    module = importlib.import_module(source)
                    if not isinstance(module, ModuleType):  # pragma: no cover - import invariant
                        raise RuntimeError(f"Compatibility alias source is not a module: {source}")
                    sys.modules[target] = module
            yield
        finally:
            for target, _source in reversed(NUMPY_CORE_COMPAT_ALIASES):
                prior = previous[target]
                if prior is _MISSING:
                    sys.modules.pop(target, None)
                else:
                    sys.modules[target] = prior  # type: ignore[assignment]


def _validate_calibrator(calibrator: object, level: str) -> None:
    cls = type(calibrator)
    if (cls.__module__, cls.__name__) != ("sklearn.isotonic", "IsotonicRegression"):
        raise RuntimeError(f"Unexpected frozen calibrator class for {level}: {cls.__module__}.{cls.__name__}")
    expected_scalars = {
        "increasing": True,
        "increasing_": True,
        "out_of_bounds": "clip",
        "y_min": 0.0,
        "y_max": 1.0,
    }
    for name, expected in expected_scalars.items():
        if getattr(calibrator, name, _MISSING) != expected:
            raise RuntimeError(f"Unexpected frozen calibrator attribute for {level}: {name}")
    x_values = np.asarray(getattr(calibrator, "X_thresholds_", None))
    y_values = np.asarray(getattr(calibrator, "y_thresholds_", None))
    if x_values.dtype != np.dtype("float64") or y_values.dtype != np.dtype("float64"):
        raise RuntimeError(f"Frozen calibrator threshold dtype mismatch for {level}")
    if x_values.ndim != 1 or y_values.ndim != 1 or len(x_values) < 2 or len(x_values) != len(y_values):
        raise RuntimeError(f"Frozen calibrator threshold shape mismatch for {level}")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise RuntimeError(f"Frozen calibrator contains non-finite thresholds for {level}")
    if not np.all(np.diff(x_values) > 0) or not np.all(np.diff(y_values) >= 0):
        raise RuntimeError(f"Frozen calibrator thresholds are not monotone for {level}")
    if not callable(getattr(calibrator, "predict", None)) or not callable(getattr(calibrator, "f_", None)):
        raise RuntimeError(f"Frozen calibrator is not fitted for {level}")


def load_frozen_calibrator(asset_root: str | Path, level: str) -> object:
    """Load one exact manifest-bound SiteGuard calibrator.

    ``level`` is a closed selector, not a path.  The selected file is opened and
    hashed before and after deserialization from the same file descriptor.
    """

    if level not in FROZEN_CALIBRATORS:
        raise ValueError(f"Unknown frozen calibrator level: {level!r}")
    root = Path(asset_root).resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"Asset root is not a directory: {root}")
    asset = FROZEN_CALIBRATORS[level]
    manifest = _read_identity_bound_manifest(root)
    _verify_manifest_asset(manifest, asset)
    path = root / asset.path
    if path.is_symlink():
        raise RuntimeError(f"Frozen calibrator must not be a symlink: {path}")
    with path.open("rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"Frozen calibrator is not a regular file: {path}")
        if metadata.st_size != asset.bytes:
            raise RuntimeError(f"Frozen calibrator byte mismatch: {path}")
        if _sha256_handle(handle) != asset.sha256:
            raise RuntimeError(f"Frozen calibrator SHA-256 mismatch: {path}")
        with _temporary_numpy_core_aliases():
            calibrator = joblib.load(handle)
        if _sha256_handle(handle) != asset.sha256 or os.fstat(handle.fileno()).st_size != asset.bytes:
            raise RuntimeError(f"Frozen calibrator changed during load: {path}")
    _validate_calibrator(calibrator, level)
    return calibrator


__all__ = ["FROZEN_CALIBRATORS", "NUMPY_CORE_COMPAT_ALIASES", "load_frozen_calibrator"]
