"""Portable inference for the frozen EvidenceJudge V2 reliability layer."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LEVELS = ("EC_L3", "EC_L4")
COHORT_STATUSES = ("validated-similar", "unknown", "shifted")


def _default_spec_path() -> Path:
    return Path(resources.files("siteguard").joinpath("data/evidencejudge_v2.json"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_spec(path: str | Path | None = None) -> tuple[dict[str, object], Path, str]:
    spec_path = Path(path).resolve() if path is not None else _default_spec_path()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("format") != "siteguard.evidencejudge.portable.v1":
        raise ValueError(f"Unsupported EvidenceJudge spec format: {spec.get('format')!r}")
    if sorted(spec.get("models", {})) != sorted(LEVELS):
        raise ValueError("EvidenceJudge spec must contain EC_L3 and EC_L4 models")
    return spec, spec_path, _sha256(spec_path)


def _numeric_matrix(frame: pd.DataFrame, columns: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    columns = list(columns)
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        preview = ", ".join(missing_columns[:8])
        raise ValueError(f"Missing {len(missing_columns)} required EvidenceJudge features: {preview}")
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return numeric, np.isnan(numeric)


def score_level(frame: pd.DataFrame, model: dict[str, object]) -> np.ndarray:
    """Reproduce a frozen sklearn imputer/scaler/logistic pipeline from JSON."""
    values, missing = _numeric_matrix(frame, model["feature_columns"])
    statistics = np.asarray(model["imputer_statistics"], dtype=float)
    values = np.where(missing, statistics[None, :], values)
    indicator_indices = np.asarray(model["missing_indicator_indices"], dtype=int)
    if len(indicator_indices):
        values = np.concatenate([values, missing[:, indicator_indices].astype(float)], axis=1)
    center = np.asarray(model["scaler_center"], dtype=float)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    if values.shape[1] != len(center) or len(center) != len(scale) or len(scale) != len(coefficients):
        raise ValueError("Portable EvidenceJudge model dimensions are inconsistent")
    transformed = (values - center[None, :]) / scale[None, :]
    logits = transformed @ coefficients + float(model["intercept"])
    # Stable sigmoid without scipy or scikit-learn at inference time.
    positive = logits >= 0
    probabilities = np.empty_like(logits, dtype=float)
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)
    return probabilities


def judge_frame(
    frame: pd.DataFrame,
    *,
    cohort_status: str = "unknown",
    spec_path: str | Path | None = None,
) -> pd.DataFrame:
    """Score HIT-EC candidates and apply row- plus cohort-level deferral logic."""
    if cohort_status not in COHORT_STATUSES:
        raise ValueError(f"cohort_status must be one of {COHORT_STATUSES}")
    required_identifiers = ["query_protein_id", "annotation_level", "candidate_label"]
    missing_identifiers = [column for column in required_identifiers if column not in frame.columns]
    if missing_identifiers:
        raise ValueError(f"Input is missing required identifier columns: {missing_identifiers}")
    levels = frame["annotation_level"].astype(str)
    unexpected = sorted(set(levels) - set(LEVELS))
    if unexpected:
        raise ValueError(f"Unsupported annotation levels: {unexpected}")
    if frame.duplicated(["query_protein_id", "annotation_level"]).any():
        raise ValueError("Input contains duplicate query/annotation-level rows")

    spec, resolved_spec, spec_sha256 = load_spec(spec_path)
    output = frame.copy()
    output["evidencejudge_score"] = np.nan
    output["evidencejudge_threshold"] = np.nan
    output["frozen_threshold_accept"] = False
    output["evidencejudge_model"] = ""
    for level in LEVELS:
        mask = levels.eq(level)
        if not mask.any():
            continue
        model = spec["models"][level]
        scores = score_level(output.loc[mask], model)
        threshold = float(model["threshold"])
        output.loc[mask, "evidencejudge_score"] = scores
        output.loc[mask, "evidencejudge_threshold"] = threshold
        output.loc[mask, "frozen_threshold_accept"] = scores >= threshold
        output.loc[mask, "evidencejudge_model"] = str(model["model_name"])

    output["cohort_applicability_status"] = cohort_status
    output["final_decision"] = "DEFER"
    if cohort_status == "validated-similar":
        output.loc[output["frozen_threshold_accept"].astype(bool), "final_decision"] = "ACCEPT"
    output["decision_reason"] = np.select(
        [
            output["frozen_threshold_accept"].astype(bool) & output["final_decision"].eq("ACCEPT"),
            ~output["frozen_threshold_accept"].astype(bool),
            output["cohort_applicability_status"].eq("shifted"),
        ],
        [
            "ROW_THRESHOLD_MET_AND_COHORT_VALIDATED_SIMILAR",
            "ROW_THRESHOLD_NOT_MET",
            "COHORT_SHIFTED_REQUIRE_REVALIDATION",
        ],
        default="COHORT_APPLICABILITY_UNKNOWN_REQUIRE_AUDIT",
    )
    output["model_spec_sha256"] = spec_sha256
    output["model_spec"] = (
        str(resolved_spec) if spec_path is not None
        else "bundled:siteguard/data/evidencejudge_v2.json"
    )
    output["deployment_boundary"] = str(spec["deployment_boundary"])
    return output


def read_feature_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def write_judgements(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif suffix == ".csv":
        frame.to_csv(path, index=False, na_rep="NA")
    else:
        frame.to_csv(path, sep="\t", index=False, na_rep="NA")
