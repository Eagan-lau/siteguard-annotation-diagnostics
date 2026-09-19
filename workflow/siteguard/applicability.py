"""Truth-free cohort-shift audit for EvidenceJudge safety vetoes.

The audit intentionally cannot grant ``validated-similar`` status.  It may
only detect a sufficiently strong covariate shift or leave applicability
unknown pending labelled external validation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evidencejudge import LEVELS, load_spec, read_feature_table


AUDIT_FORMAT = "siteguard.evidencejudge.applicability-audit.v1"
DEFAULT_SEED = 20261111
DEFAULT_FOLDS = 5
DEFAULT_PERMUTATIONS = 1_000
DEFAULT_BOOTSTRAPS = 10_000
MIN_ROWS = 100
MIN_CLUSTERS = 50
AUC_THRESHOLD = 0.75
LOWER_BOUND_THRESHOLD = 0.65
PERMUTATION_P_THRESHOLD = 0.01
FORBIDDEN_INPUT_TOKENS = ("truth", "correct", "outcome", "label_eligible")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rng(seed: int, context: str) -> np.random.Generator:
    payload = hashlib.sha256(f"{seed}|{context}".encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(payload, "big") % (2**32 - 1))


def _sklearn_components():
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise ImportError(
            "The cohort audit requires scikit-learn; install siteguard-enzyme[audit]."
        ) from exc
    return (
        SimpleImputer,
        LogisticRegression,
        roc_auc_score,
        StratifiedGroupKFold,
        Pipeline,
        StandardScaler,
    )


def _make_model(seed: int):
    SimpleImputer, LogisticRegression, _, _, Pipeline, StandardScaler = _sklearn_components()
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=0.1,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5_000,
            random_state=seed,
        )),
    ])


def _oriented_auc(y: np.ndarray, probability: np.ndarray) -> float:
    _, _, roc_auc_score, _, _, _ = _sklearn_components()
    auc = float(roc_auc_score(y, probability))
    return max(auc, 1.0 - auc)


def _oof_predictions(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    folds: int,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, _, StratifiedGroupKFold, _, _ = _sklearn_components()
    cv_seed = int(_rng(seed, f"CV|{context}").integers(1, 2**31 - 1))
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=cv_seed)
    probability = np.full(len(y), np.nan, dtype=float)
    fold_id = np.full(len(y), -1, dtype=int)
    for fold, (train, test) in enumerate(cv.split(x, y, groups)):
        model = _make_model(seed)
        model.fit(x.iloc[train], y[train])
        probability[test] = model.predict_proba(x.iloc[test])[:, 1]
        fold_id[test] = fold
    if not np.isfinite(probability).all() or (fold_id < 0).any():
        raise RuntimeError("Incomplete out-of-fold domain predictions")
    return probability, fold_id


def _cluster_bootstrap_interval(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    bootstraps: int,
    context: str,
) -> tuple[float, float]:
    frame = pd.DataFrame({"y": y, "p": probability, "group": groups})
    grouped = {
        int(domain): [part[["y", "p"]].to_numpy() for _, part in subset.groupby("group", sort=True)]
        for domain, subset in frame.groupby("y", sort=True)
    }
    rng = _rng(seed, f"BOOTSTRAP|{context}")
    estimates = np.empty(bootstraps, dtype=float)
    for index in range(bootstraps):
        sampled = []
        for domain in (0, 1):
            clusters = grouped[domain]
            choices = rng.integers(0, len(clusters), size=len(clusters))
            sampled.extend(clusters[value] for value in choices)
        values = np.concatenate(sampled, axis=0)
        estimates[index] = _oriented_auc(values[:, 0].astype(int), values[:, 1])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _cluster_permutation_pvalue(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    observed_auc: float,
    *,
    seed: int,
    folds: int,
    permutations: int,
    context: str,
) -> tuple[float, np.ndarray]:
    group_frame = pd.DataFrame({"group": groups, "domain": y}).drop_duplicates()
    if int(group_frame.groupby("group")["domain"].nunique().max()) != 1:
        raise ValueError(
            "Reference and target sequence-cluster identifiers overlap; this v1 exact "
            "cluster-label permutation audit requires disjoint cohort cluster inventories."
        )
    group_frame = group_frame.sort_values("group").reset_index(drop=True)
    original = group_frame["domain"].to_numpy(dtype=int)
    rng = _rng(seed, f"PERMUTATION|{context}")
    null_auc = np.empty(permutations, dtype=float)
    for index in range(permutations):
        permuted = rng.permutation(original)
        mapping = dict(zip(group_frame["group"], permuted))
        permuted_y = np.fromiter((mapping[group] for group in groups), dtype=int, count=len(groups))
        probability, _ = _oof_predictions(
            x, permuted_y, groups, seed=seed, folds=folds, context=f"{context}|PERM|{index}"
        )
        null_auc[index] = _oriented_auc(permuted_y, probability)
    pvalue = (1.0 + float(np.sum(null_auc >= observed_auc))) / (permutations + 1.0)
    return pvalue, null_auc


def _validate_inputs(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    *,
    folds: int,
) -> None:
    required = {"query_protein_id", "annotation_level", "query_cluster_id_30", "candidate_label"}
    for name, frame in (("reference", reference), ("target", target)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} table is missing identifiers: {missing}")
        missing_features = [feature for feature in features if feature not in frame.columns]
        if missing_features:
            raise ValueError(f"{name} table is missing {len(missing_features)} audit features")
        if frame.duplicated(["query_protein_id", "annotation_level"]).any():
            raise ValueError(f"{name} table contains duplicate query/annotation-level rows")
        forbidden = sorted(
            column for column in frame.columns
            if column not in required and any(token in column.lower() for token in FORBIDDEN_INPUT_TOKENS)
        )
        if forbidden:
            raise ValueError(f"{name} table contains prohibited truth/outcome fields: {forbidden}")
    if folds < 2:
        raise ValueError("folds must be at least 2")


def audit_cohort_frames(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    *,
    spec_path: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    folds: int = DEFAULT_FOLDS,
    permutations: int = DEFAULT_PERMUTATIONS,
    bootstraps: int = DEFAULT_BOOTSTRAPS,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Audit covariate shift and return JSON-ready summary and diagnostic tables."""
    if permutations < 99:
        raise ValueError("At least 99 permutations are required for the fixed p<=0.01 rule")
    if bootstraps < 100:
        raise ValueError("At least 100 cluster bootstraps are required")
    spec, resolved_spec, spec_sha256 = load_spec(spec_path)
    features = list(spec["models"]["EC_L3"]["feature_columns"])
    if features != list(spec["models"]["EC_L4"]["feature_columns"]):
        raise ValueError("EC_L3 and EC_L4 audit feature schemas differ")
    _validate_inputs(reference, target, features, folds=folds)

    level_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    permutation_frames: list[pd.DataFrame] = []
    for level in LEVELS:
        left = reference.loc[reference["annotation_level"].astype(str).eq(level)].copy()
        right = target.loc[target["annotation_level"].astype(str).eq(level)].copy()
        if left.empty or right.empty:
            raise ValueError(f"Both cohorts must contain {level} rows")
        left["audit_domain"] = 0
        right["audit_domain"] = 1
        combined = pd.concat([left, right], ignore_index=True)
        x = combined[features].apply(pd.to_numeric, errors="coerce")
        y = combined["audit_domain"].to_numpy(dtype=int)
        groups = combined["query_cluster_id_30"].astype(str).to_numpy()
        probability, fold_id = _oof_predictions(
            x, y, groups, seed=seed, folds=folds, context=level
        )
        auc = _oriented_auc(y, probability)
        ci_low, ci_high = _cluster_bootstrap_interval(
            y, probability, groups, seed=seed, bootstraps=bootstraps, context=level
        )
        permutation_p, null_auc = _cluster_permutation_pvalue(
            x, y, groups, auc, seed=seed, folds=folds,
            permutations=permutations, context=level,
        )
        reference_rows = int((y == 0).sum())
        target_rows = int((y == 1).sum())
        reference_clusters = int(pd.Series(groups[y == 0]).nunique())
        target_clusters = int(pd.Series(groups[y == 1]).nunique())
        sufficient = (
            reference_rows >= MIN_ROWS and target_rows >= MIN_ROWS
            and reference_clusters >= MIN_CLUSTERS and target_clusters >= MIN_CLUSTERS
        )
        shifted = bool(
            sufficient and auc >= AUC_THRESHOLD and ci_low >= LOWER_BOUND_THRESHOLD
            and permutation_p <= PERMUTATION_P_THRESHOLD
        )
        level_rows.append({
            "annotation_level": level,
            "reference_rows": reference_rows,
            "target_rows": target_rows,
            "reference_clusters": reference_clusters,
            "target_clusters": target_clusters,
            "domain_auc": auc,
            "cluster_bootstrap_low": ci_low,
            "cluster_bootstrap_high": ci_high,
            "cluster_permutation_p": permutation_p,
            "auc_threshold": AUC_THRESHOLD,
            "lower_bound_threshold": LOWER_BOUND_THRESHOLD,
            "p_threshold": PERMUTATION_P_THRESHOLD,
            "level_shift_status": "SHIFT_DETECTED" if shifted else "INCONCLUSIVE",
        })
        predictions = combined[[
            "query_protein_id", "annotation_level", "query_cluster_id_30"
        ]].copy()
        predictions["domain"] = np.where(y == 0, "REFERENCE", "TARGET")
        predictions["domain_probability"] = probability
        predictions["fold"] = fold_id
        prediction_frames.append(predictions)
        permutation_frames.append(pd.DataFrame({
            "annotation_level": level,
            "permutation": np.arange(permutations),
            "null_auc": null_auc,
        }))

    level_table = pd.DataFrame(level_rows)
    status = "shifted" if level_table["level_shift_status"].eq("SHIFT_DETECTED").any() else "unknown"
    result: dict[str, object] = {
        "format": AUDIT_FORMAT,
        "analysis_status": "POST_CONFIRMATORY_EXPLORATORY_TRUTH_FREE_SAFETY_VETO",
        "model_spec": str(resolved_spec),
        "model_spec_sha256": spec_sha256,
        "feature_count": len(features),
        "truth_or_outcome_fields_used": False,
        "automatic_cohort_status": status,
        "automatic_status_scope": "SAFETY_VETO_ONLY_NEVER_GRANTS_VALIDATED_SIMILAR",
        "parameters": {
            "seed": seed,
            "folds": folds,
            "permutations": permutations,
            "bootstraps": bootstraps,
            "min_rows": MIN_ROWS,
            "min_clusters": MIN_CLUSTERS,
            "auc_threshold": AUC_THRESHOLD,
            "lower_bound_threshold": LOWER_BOUND_THRESHOLD,
            "permutation_p_threshold": PERMUTATION_P_THRESHOLD,
        },
        "levels": level_table.to_dict(orient="records"),
        "limitations": [
            "Detects covariate shift but does not estimate accuracy, precision, or calibration.",
            "An inconclusive result remains unknown and cannot grant validated-similar status.",
            "Requires labelled external validation before an accept decision is enabled.",
        ],
    }
    return result, pd.concat(prediction_frames, ignore_index=True), pd.concat(permutation_frames, ignore_index=True)


def audit_cohort_files(
    reference_path: str | Path,
    target_path: str | Path,
    *,
    output_path: str | Path,
    spec_path: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    folds: int = DEFAULT_FOLDS,
    permutations: int = DEFAULT_PERMUTATIONS,
    bootstraps: int = DEFAULT_BOOTSTRAPS,
) -> dict[str, object]:
    """Run the audit from files and write an immutable JSON decision record."""
    reference_path = Path(reference_path).resolve()
    target_path = Path(target_path).resolve()
    output_path = Path(output_path).resolve()
    result, predictions, permutation_null = audit_cohort_frames(
        read_feature_table(reference_path),
        read_feature_table(target_path),
        spec_path=spec_path,
        seed=seed,
        folds=folds,
        permutations=permutations,
        bootstraps=bootstraps,
    )
    result.update({
        "reference": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "target": str(target_path),
        "target_sha256": _sha256(target_path),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    stem = output_path.with_suffix("")
    predictions.to_parquet(stem.with_name(stem.name + "_domain_oof.parquet"), index=False)
    permutation_null.to_parquet(stem.with_name(stem.name + "_permutation_null.parquet"), index=False)
    return result


def apply_audit_veto(
    requested_status: str,
    audit_path: str | Path,
    *,
    target_path: str | Path,
    model_spec_sha256: str,
) -> tuple[str, dict[str, object], str]:
    """Verify an audit record and apply its one-way safety veto."""
    audit_path = Path(audit_path).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("format") != AUDIT_FORMAT:
        raise ValueError(f"Unsupported applicability-audit format: {audit.get('format')!r}")
    if audit.get("automatic_status_scope") != "SAFETY_VETO_ONLY_NEVER_GRANTS_VALIDATED_SIMILAR":
        raise ValueError("Applicability audit is missing the required safety-veto scope")
    if audit.get("target_sha256") != _sha256(Path(target_path).resolve()):
        raise ValueError("Applicability audit target hash does not match the EvidenceJudge input")
    if audit.get("model_spec_sha256") != model_spec_sha256:
        raise ValueError("Applicability audit model-spec hash does not match the scoring model")
    automatic = str(audit.get("automatic_cohort_status"))
    if automatic not in {"unknown", "shifted"}:
        raise ValueError("Applicability audit may only return unknown or shifted")
    effective = "shifted" if automatic == "shifted" else requested_status
    return effective, audit, _sha256(audit_path)
