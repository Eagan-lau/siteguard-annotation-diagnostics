"""Truth-free annotation orchestration with EvidenceJudge fail-closed decisions."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evidencejudge import LEVELS as JUDGE_LEVELS, judge_frame, load_spec, read_feature_table
from .fasta import read_fasta, sequence_sha256
from .identifiers import canonicalize_ec, canonicalize_rhea, ec_l3_from_l4
from .reporting import render_predictions_html
from .run_bundle import read_bundle, write_bundle_atomic


BASE_LEVELS = ("EC_L3", "EC_L4", "EXACT_RHEA")
CANDIDATE_COLUMNS = [
    "query_protein_id",
    "reference_id",
    "activity_id",
    *[item for level in BASE_LEVELS for item in (f"top_label_{level}", f"top_probability_{level}")],
    *[item for level in BASE_LEVELS for item in (f"top_reference_{level}", f"top_reference_activity_{level}")],
    "source_mode",
]
FORBIDDEN_EVIDENCE_COLUMNS = {"ground_truth", "true_label", "is_correct", "outcome", "target_correct"}


def _canonicalize_candidate_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["top_label_EC_L3"] = frame["top_label_EC_L3"].map(
        lambda value: canonicalize_ec(value, level=3)
    )
    frame["top_label_EC_L4"] = frame["top_label_EC_L4"].map(
        lambda value: canonicalize_ec(value, level=4)
    )
    frame["top_label_EXACT_RHEA"] = frame["top_label_EXACT_RHEA"].map(canonicalize_rhea)
    inconsistent = frame.apply(
        lambda row: bool(row["top_label_EC_L4"])
        and ec_l3_from_l4(row["top_label_EC_L4"]) != row["top_label_EC_L3"],
        axis=1,
    )
    if inconsistent.any():
        raise ValueError("Candidate EC-L4 labels are inconsistent with their EC-L3 ancestors")
    return frame


def _precomputed_candidates(path: str | Path, query_ids: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = [
        "query_protein_id", "reference_id", "activity_id",
        *[item for level in BASE_LEVELS for item in (f"top_label_{level}", f"top_probability_{level}")],
        "source_mode",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Precomputed candidate table missing columns: {missing}")
    for level in BASE_LEVELS:
        frame[f"top_reference_{level}"] = frame["reference_id"]
        frame[f"top_reference_activity_{level}"] = frame["activity_id"]
    frame = frame[CANDIDATE_COLUMNS].copy()
    if frame["query_protein_id"].duplicated().any():
        raise ValueError("Precomputed candidate table must contain one row per query")
    if set(frame["query_protein_id"]) != query_ids:
        raise ValueError("Precomputed candidate query identifiers do not exactly match the FASTA")
    for level in BASE_LEVELS:
        frame[f"top_probability_{level}"] = pd.to_numeric(frame[f"top_probability_{level}"], errors="raise")
        if (~frame[f"top_probability_{level}"].between(0.0, 1.0)).any():
            raise ValueError(f"Probability outside [0,1] for {level}")
    return _canonicalize_candidate_labels(frame)


def _candidate_rows_from_prediction_frame(result: pd.DataFrame) -> pd.DataFrame:
    """Preserve the independently selected provenance pair for every level."""
    rows: list[dict[str, Any]] = []
    for row in result.to_dict("records"):
        per_level = {
            level: (
                str(row.get(f"top_reference_{level}", "")),
                str(row.get(f"top_reference_activity_{level}", "")),
            )
            for level in BASE_LEVELS
        }
        rows.append({
            "query_protein_id": str(row["query_protein_id"]),
            "reference_id": per_level["EC_L3"][0],
            "activity_id": per_level["EC_L3"][1],
            **{f"top_label_{level}": str(row.get(f"top_label_{level}", "")) for level in BASE_LEVELS},
            **{f"top_probability_{level}": float(row.get(f"top_probability_{level}", 0.0)) for level in BASE_LEVELS},
            **{f"top_reference_{level}": per_level[level][0] for level in BASE_LEVELS},
            **{f"top_reference_activity_{level}": per_level[level][1] for level in BASE_LEVELS},
            "source_mode": "FROZEN_SITEGUARD_PREDICTOR",
        })
    return _canonicalize_candidate_labels(pd.DataFrame(rows, columns=CANDIDATE_COLUMNS))


def _bind_precomputed_candidates(
    candidates: pd.DataFrame,
    references: pd.DataFrame,
    activities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uniquely bind every candidate to one declared reference/activity row."""
    required_activity = {"reference_id", "activity_id", "ec_l3", "ec_l4", "rhea_id"}
    missing = sorted(required_activity - set(activities.columns))
    if missing:
        raise ValueError(f"Reference activity bundle missing binding columns: {missing}")
    if references["reference_id"].astype(str).duplicated().any():
        raise ValueError("Reference bundle contains duplicate reference_id values")
    if activities[["reference_id", "activity_id"]].astype(str).duplicated().any():
        raise ValueError("Reference bundle contains duplicate reference/activity pairs")

    activity_labels = activities[["reference_id", "activity_id", "ec_l3", "ec_l4", "rhea_id"]].copy()
    activity_labels["reference_id"] = activity_labels["reference_id"].astype(str)
    activity_labels["activity_id"] = activity_labels["activity_id"].astype(str)
    activity_labels["activity_ec_l3"] = activity_labels.pop("ec_l3").map(
        lambda value: canonicalize_ec(value, level=3)
    )
    activity_labels["activity_ec_l4"] = activity_labels.pop("ec_l4").map(
        lambda value: canonicalize_ec(value, level=4)
    )
    activity_labels["activity_rhea"] = activity_labels.pop("rhea_id").map(canonicalize_rhea)
    activity_inconsistent = activity_labels.apply(
        lambda row: bool(row["activity_ec_l4"])
        and ec_l3_from_l4(row["activity_ec_l4"]) != row["activity_ec_l3"],
        axis=1,
    )
    if activity_inconsistent.any():
        raise ValueError("Reference activity bundle contains an inconsistent EC hierarchy")

    bound = candidates.merge(
        activity_labels,
        on=["reference_id", "activity_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not bound["_merge"].eq("both").all():
        raise ValueError("Every precomputed candidate must bind to exactly one reference/activity pair")
    comparisons = {
        "EC-L3": ("top_label_EC_L3", "activity_ec_l3"),
        "EC-L4": ("top_label_EC_L4", "activity_ec_l4"),
        "Exact Rhea": ("top_label_EXACT_RHEA", "activity_rhea"),
    }
    for label, (candidate_column, activity_column) in comparisons.items():
        if not bound[candidate_column].eq(bound[activity_column]).all():
            raise ValueError(f"Precomputed candidate {label} label does not match its bound activity")

    output_references = candidates[["query_protein_id", "reference_id"]].merge(
        references, on="reference_id", how="left", validate="many_to_one"
    )
    if output_references.isna().any(axis=None):
        raise ValueError("Precomputed candidate reference join produced a null row")
    output_references["retrieval_channel"] = "PRECOMPUTED_FIXTURE_OR_EXTERNAL"
    output_references["retrieval_rank"] = 1

    output_activities = candidates[["query_protein_id", "reference_id", "activity_id"]].merge(
        activities, on=["reference_id", "activity_id"], how="left", validate="many_to_one"
    )
    if output_activities.isna().any(axis=None):
        raise ValueError("Precomputed candidate activity join produced a null row")
    output_activities["ec_l3"] = output_activities["ec_l3"].map(
        lambda value: canonicalize_ec(value, level=3)
    )
    output_activities["ec_l4"] = output_activities["ec_l4"].map(
        lambda value: canonicalize_ec(value, level=4)
    )
    output_activities["rhea_id"] = output_activities["rhea_id"].map(canonicalize_rhea)
    return output_references, output_activities


def _predict_candidates(
    input_fasta: Path,
    asset_root: Path,
    query_embeddings: Path,
    embedding_index: Path,
    *,
    query_pfam: Path | None,
    mmseqs: str,
    threads: int,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from .predictor import predict_fasta

    with tempfile.TemporaryDirectory(prefix="siteguard-annotate-predict-") as tmp:
        work = Path(tmp)
        output = work / "predictions.tsv"
        result = predict_fasta(
            input_fasta, asset_root, query_embeddings, embedding_index, output, work,
            query_pfam, mmseqs, threads, device,
        )
        pairs = pd.read_parquet(work / "siteguard_pair_scores.parquet")
        reference_columns = [column for column in (
            "query_protein_id", "reference_protein_id", "retrieval_rank", "sequence_identity",
            "sequence_query_coverage", "sequence_reference_coverage", "sequence_bitscore_log1p",
        ) if column in pairs]
        references = pairs[reference_columns].drop_duplicates().rename(
            columns={"reference_protein_id": "reference_id"}
        )
        activity_columns = [column for column in (
            "query_protein_id", "reference_protein_id", "reference_activity_id", "ec_l3", "ec_l4",
            "canonical_rhea", "evidence_tier",
        ) if column in pairs]
        activities = pairs[activity_columns].drop_duplicates().rename(columns={
            "reference_protein_id": "reference_id",
            "reference_activity_id": "activity_id",
            "canonical_rhea": "rhea_id",
        })
    return _candidate_rows_from_prediction_frame(result), references, activities


def _strict_base_decision(row: pd.Series, thresholds: dict[str, float]) -> tuple[str, str, float, str]:
    p3, p4, pr = (float(row[f"top_probability_{level}"]) for level in BASE_LEVELS)
    l3, l4, lr = (str(row[f"top_label_{level}"]).strip() for level in BASE_LEVELS)
    ec3 = bool(l3) and p3 >= thresholds["EC_L3"]
    ec4 = ec3 and bool(l4) and p4 >= thresholds["EC_L4"]
    rhea = ec4 and bool(lr) and pr >= thresholds["EXACT_RHEA"]
    if rhea:
        return "EXACT_RHEA", lr, pr, "STRICT_EC3_EC4_RHEA_HIERARCHY_MET"
    if ec4:
        return "EC_L4", l4, p4, "STRICT_EC3_EC4_HIERARCHY_MET"
    if ec3:
        return "EC_L3", l3, p3, "EC3_THRESHOLD_MET"
    return "ABSTAIN", "", 0.0, "EC3_THRESHOLD_OR_LABEL_NOT_MET"


def _judge_rows(
    candidates: pd.DataFrame,
    evidence_features: str | Path | None,
    *,
    cohort_status: str,
    model_spec: str | Path | None,
) -> pd.DataFrame:
    spec, _, _ = load_spec(model_spec)
    if evidence_features is None:
        if cohort_status == "validated-similar":
            raise ValueError("validated-similar annotation requires explicit EvidenceJudge feature rows")
        feature_columns = list(dict.fromkeys(
            column for level in JUDGE_LEVELS for column in spec["models"][level]["feature_columns"]
        ))
        rows = []
        for candidate in candidates.to_dict("records"):
            for level in JUDGE_LEVELS:
                rows.append({
                    "query_protein_id": candidate["query_protein_id"],
                    "annotation_level": level,
                    "candidate_label": candidate[f"top_label_{level}"],
                    **{column: np.nan for column in feature_columns},
                })
        features = pd.DataFrame(rows)
    else:
        features = read_feature_table(evidence_features)
        forbidden = sorted(FORBIDDEN_EVIDENCE_COLUMNS.intersection(features.columns))
        if forbidden:
            raise ValueError(f"Outcome/truth fields are forbidden in inference evidence: {forbidden}")
        expected = {(str(row.query_protein_id), level) for row in candidates.itertuples() for level in JUDGE_LEVELS}
        observed = set(zip(features["query_protein_id"].astype(str), features["annotation_level"].astype(str)))
        if observed != expected:
            raise ValueError("Evidence feature rows must exactly cover every query at EC_L3 and EC_L4")
        expected_labels = {
            (str(row.query_protein_id), level): str(getattr(row, f"top_label_{level}"))
            for row in candidates.itertuples() for level in JUDGE_LEVELS
        }
        for row in features[["query_protein_id", "annotation_level", "candidate_label"]].itertuples(index=False):
            if str(row.candidate_label) != expected_labels[(str(row.query_protein_id), str(row.annotation_level))]:
                raise ValueError("EvidenceJudge candidate_label does not match the precomputed candidate")
    return judge_frame(features, cohort_status=cohort_status, spec_path=model_spec)


def annotate_bundle(
    input_fasta: str | Path,
    output_dir: str | Path,
    *,
    calibration_config: str | Path,
    reference_bundle: str | Path | None = None,
    precomputed_candidates: str | Path | None = None,
    asset_root: str | Path | None = None,
    query_embeddings: str | Path | None = None,
    embedding_index: str | Path | None = None,
    evidence_features: str | Path | None = None,
    model_spec: str | Path | None = None,
    cohort_status: str = "unknown",
    query_pfam: str | Path | None = None,
    mmseqs: str = "mmseqs",
    threads: int = 8,
    device: str = "auto",
) -> Path:
    fasta_path = Path(input_fasta).resolve(strict=True)
    records = read_fasta(fasta_path)
    query_ids = set(records)
    calibration_path = Path(calibration_config).resolve(strict=True)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    thresholds = {level: float(calibration["thresholds"][level]) for level in BASE_LEVELS}

    fixture_mode = precomputed_candidates is not None
    predictor_mode = any(item is not None for item in (asset_root, query_embeddings, embedding_index))
    if fixture_mode == predictor_mode:
        raise ValueError("Choose exactly one candidate source: precomputed candidates or asset-root/embedding predictor")
    inputs: dict[str, str | Path] = {"input_fasta": fasta_path, "calibration_config": calibration_path}
    assets: dict[str, str | Path] = {}
    if fixture_mode:
        if reference_bundle is None:
            raise ValueError("Precomputed candidate mode requires --reference-bundle")
        database = read_bundle(reference_bundle)
        if database["manifest"].get("workflow") != "build-db":
            raise ValueError("Precomputed candidate mode requires a build-db reference bundle")
        candidates = _precomputed_candidates(precomputed_candidates, query_ids)
        references = database["references"].copy()
        activities = database["activities"].copy()
        inputs["precomputed_candidates"] = Path(precomputed_candidates)
        assets["reference_bundle_manifest"] = Path(reference_bundle) / "run_manifest.json"
        references, activities = _bind_precomputed_candidates(candidates, references, activities)
    else:
        if asset_root is None or query_embeddings is None or embedding_index is None:
            raise ValueError("Predictor mode requires --asset-root, --query-embeddings and --embedding-index")
        candidates, references, activities = _predict_candidates(
            fasta_path, Path(asset_root), Path(query_embeddings), Path(embedding_index),
            query_pfam=Path(query_pfam) if query_pfam else None,
            mmseqs=mmseqs, threads=threads, device=device,
        )
        assets.update({
            "asset_root_calibration": Path(asset_root) / "models/phase12/calibration_config.json",
            "query_embeddings": Path(query_embeddings),
            "embedding_index": Path(embedding_index),
        })

    judgements = _judge_rows(
        candidates, evidence_features, cohort_status=cohort_status, model_spec=model_spec
    )
    if evidence_features is not None:
        inputs["evidence_features"] = Path(evidence_features)
    if model_spec is not None:
        assets["evidencejudge_model_spec"] = Path(model_spec)

    judge_lookup = {
        (str(row.query_protein_id), str(row.annotation_level)): row
        for row in judgements.itertuples(index=False)
    }
    prediction_rows: list[dict[str, Any]] = []
    evidence_queries: list[dict[str, Any]] = []
    warnings = [
        "Structure, local-residue and chemistry evidence were not fabricated; unavailable channels are recorded as false.",
        "EvidenceJudge has no Exact Rhea acceptance endpoint; Exact Rhea candidates can only be downgraded or deferred.",
    ]
    for candidate in candidates.itertuples(index=False):
        series = pd.Series(candidate._asdict())
        base_level, base_label, base_probability, base_reason = _strict_base_decision(series, thresholds)
        base_reference = str(getattr(candidate, f"top_reference_{base_level}", "")) if base_level in BASE_LEVELS else ""
        base_activity = str(getattr(candidate, f"top_reference_activity_{base_level}", "")) if base_level in BASE_LEVELS else ""
        final_level, final_label, final_probability = "ABSTAIN", "", 0.0
        final_reference, final_activity = "", ""
        final_decision, reason = "DEFER", base_reason
        judge_level = "EC_L4" if base_level in {"EC_L4", "EXACT_RHEA"} else "EC_L3"
        judge = judge_lookup[(str(candidate.query_protein_id), judge_level)]
        if base_level != "ABSTAIN" and str(judge.final_decision) == "ACCEPT":
            if base_level == "EXACT_RHEA":
                final_level = "EC_L4"
                final_label = str(candidate.top_label_EC_L4)
                final_probability = float(candidate.top_probability_EC_L4)
                final_reference = str(candidate.top_reference_EC_L4)
                final_activity = str(candidate.top_reference_activity_EC_L4)
                reason = "EXACT_RHEA_EVIDENCEJUDGE_UNAVAILABLE_DOWNGRADED_TO_ACCEPTED_EC_L4"
            else:
                final_level, final_label, final_probability = base_level, base_label, base_probability
                final_reference = str(getattr(candidate, f"top_reference_{base_level}"))
                final_activity = str(getattr(candidate, f"top_reference_activity_{base_level}"))
                reason = "BASE_HIERARCHY_AND_EVIDENCEJUDGE_ACCEPT_WITH_USER_ASSERTED_VALIDATED_SCOPE"
            final_decision = "ACCEPT"
        elif base_level != "ABSTAIN":
            reason = str(judge.decision_reason)
        prediction_rows.append({
            "query_protein_id": str(candidate.query_protein_id),
            "sequence_sha256": sequence_sha256(records[str(candidate.query_protein_id)]),
            "candidate_reference_id": str(candidate.reference_id),
            "candidate_activity_id": str(candidate.activity_id),
            **{f"top_reference_{level}": str(getattr(candidate, f"top_reference_{level}")) for level in BASE_LEVELS},
            **{f"top_reference_activity_{level}": str(getattr(candidate, f"top_reference_activity_{level}")) for level in BASE_LEVELS},
            "base_highest_supported_level": base_level,
            "base_highest_supported_label": base_label,
            "base_highest_probability": base_probability,
            "base_reference_id": base_reference,
            "base_activity_id": base_activity,
            "candidate_ec_l3": str(candidate.top_label_EC_L3),
            "probability_ec_l3": float(candidate.top_probability_EC_L3),
            "candidate_ec_l4": str(candidate.top_label_EC_L4),
            "probability_ec_l4": float(candidate.top_probability_EC_L4),
            "candidate_rhea": str(candidate.top_label_EXACT_RHEA),
            "probability_rhea": float(candidate.top_probability_EXACT_RHEA),
            "evidencejudge_level": judge_level,
            "evidencejudge_score": float(judge.evidencejudge_score),
            "evidencejudge_threshold": float(judge.evidencejudge_threshold),
            "cohort_status": cohort_status,
            "final_decision": final_decision,
            "recommended_level": final_level,
            "recommended_label": final_label,
            "recommended_probability": final_probability,
            "recommended_reference_id": final_reference,
            "recommended_activity_id": final_activity,
            "abstention": final_decision != "ACCEPT",
            "decision_reason": reason,
        })
        evidence_queries.append({
            "query_protein_id": str(candidate.query_protein_id),
            "sequence_sha256": sequence_sha256(records[str(candidate.query_protein_id)]),
            "candidate_source_mode": str(candidate.source_mode),
            "per_level_candidate_provenance": {
                level: {
                    "reference_id": str(getattr(candidate, f"top_reference_{level}")),
                    "activity_id": str(getattr(candidate, f"top_reference_activity_{level}")),
                }
                for level in BASE_LEVELS
            },
            "sequence_evidence_available": True,
            "structure_evidence_available": False,
            "local_evidence_available": False,
            "chemistry_evidence_available": False,
            "evidencejudge_rows": {
                level: {
                    "score": float(judge_lookup[(str(candidate.query_protein_id), level)].evidencejudge_score),
                    "threshold": float(judge_lookup[(str(candidate.query_protein_id), level)].evidencejudge_threshold),
                    "decision": str(judge_lookup[(str(candidate.query_protein_id), level)].final_decision),
                    "reason": str(judge_lookup[(str(candidate.query_protein_id), level)].decision_reason),
                }
                for level in JUDGE_LEVELS
            },
        })
    predictions = pd.DataFrame(prediction_rows)
    evidence = {
        "format": "siteguard.annotation-evidence.v1",
        "workflow": "annotate",
        "truth_free": True,
        "query_outcomes_used": False,
        "cohort_status": cohort_status,
        "queries": evidence_queries,
        "boundary": "EvidenceJudge judges supplied EC3/EC4 candidates; it does not select replacement labels or accept Exact Rhea.",
    }
    return write_bundle_atomic(
        output_dir,
        workflow="annotate",
        predictions=predictions,
        references=references,
        activities=activities,
        evidence=evidence,
        report_html=render_predictions_html(predictions, title="SiteGuard annotation report"),
        inputs=inputs,
        assets=assets,
        parameters={
            "cohort_status": cohort_status,
            "candidate_mode": "precomputed" if fixture_mode else "frozen_predictor",
            "thresholds": thresholds,
            "threads": int(threads),
            "device": device,
        },
        production_model_seed=20260820 if not fixture_mode else None,
        warnings=warnings,
    )
