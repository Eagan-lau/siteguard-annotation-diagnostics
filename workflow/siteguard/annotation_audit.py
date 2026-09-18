"""Compare existing annotations with an already materialized SiteGuard evidence bundle."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .identifiers import canonicalize_ec, canonicalize_rhea, ec_l3_from_l4
from .reporting import render_predictions_html
from .run_bundle import read_bundle, write_bundle_atomic


EXISTING_COLUMNS = ["query_protein_id", "ec_l3", "ec_l4", "rhea_id", "annotation_source"]
LEVEL_RANK = {"EC_L3": 1, "EC_L4": 2, "EXACT_RHEA": 3}


def _canonical_existing(row: pd.Series) -> dict[str, str]:
    original = {
        "EC_L3": str(row["ec_l3"]),
        "EC_L4": str(row["ec_l4"]),
        "EXACT_RHEA": str(row["rhea_id"]),
    }
    canonical = {
        "EC_L3": canonicalize_ec(original["EC_L3"], level=3),
        "EC_L4": canonicalize_ec(original["EC_L4"], level=4),
        "EXACT_RHEA": canonicalize_rhea(original["EXACT_RHEA"]),
    }
    if canonical["EC_L3"] and canonical["EC_L4"] and ec_l3_from_l4(canonical["EC_L4"]) != canonical["EC_L3"]:
        raise ValueError("Existing annotation contains an inconsistent EC hierarchy")
    for level in ("EXACT_RHEA", "EC_L4", "EC_L3"):
        if canonical[level]:
            return {
                "level": level,
                "label": canonical[level],
                "original_label": original[level],
                "ec_l3": canonical["EC_L3"],
                "ec_l4": canonical["EC_L4"],
                "rhea_id": canonical["EXACT_RHEA"],
            }
    return {
        "level": "", "label": "", "original_label": "",
        "ec_l3": "", "ec_l4": "", "rhea_id": "",
    }


def _model_label(row: pd.Series, level: str) -> str:
    columns = {
        "EC_L3": "candidate_ec_l3",
        "EC_L4": "candidate_ec_l4",
        "EXACT_RHEA": "candidate_rhea",
    }
    value = row.get(columns[level], "")
    if level == "EC_L3":
        return canonicalize_ec(value, level=3)
    if level == "EC_L4":
        return canonicalize_ec(value, level=4)
    return canonicalize_rhea(value)


def _accepted_activity_mapping(
    model_row: pd.Series,
    activities: pd.DataFrame,
    query_id: str,
    accepted_rhea: str,
) -> tuple[dict[str, str] | None, str]:
    """Resolve the EC ancestors of accepted Rhea evidence from activity rows only."""
    required = {"ec_l3", "ec_l4", "rhea_id"}
    if not required.issubset(activities.columns):
        return None, "ACTIVITY_MAPPING_COLUMNS_ABSENT"
    candidates = activities.copy()
    if "query_protein_id" in candidates.columns:
        candidates = candidates.loc[candidates["query_protein_id"].astype(str).eq(query_id)]
    reference_id = str(
        model_row.get("recommended_reference_id", model_row.get("candidate_reference_id", ""))
    ).strip()
    activity_id = str(
        model_row.get("recommended_activity_id", model_row.get("candidate_activity_id", ""))
    ).strip()
    if reference_id and activity_id and {"reference_id", "activity_id"}.issubset(candidates.columns):
        candidates = candidates.loc[
            candidates["reference_id"].astype(str).eq(reference_id)
            & candidates["activity_id"].astype(str).eq(activity_id)
        ]
        basis = "BOUND_REFERENCE_ACTIVITY_MAPPING"
    else:
        canonical_rhea = candidates["rhea_id"].map(canonicalize_rhea)
        candidates = candidates.loc[canonical_rhea.eq(accepted_rhea)]
        basis = "UNIQUE_QUERY_ACTIVITY_RHEA_MAPPING"
    mappings = {
        (
            canonicalize_ec(row.ec_l3, level=3),
            canonicalize_ec(row.ec_l4, level=4),
            canonicalize_rhea(row.rhea_id),
        )
        for row in candidates[["ec_l3", "ec_l4", "rhea_id"]].itertuples(index=False)
    }
    mappings = {item for item in mappings if item[2] == accepted_rhea}
    if len(mappings) != 1:
        return None, f"{basis}_NOT_UNIQUE"
    ec_l3, ec_l4, rhea = next(iter(mappings))
    if not ec_l3 or not ec_l4 or ec_l3_from_l4(ec_l4) != ec_l3:
        raise ValueError("Accepted Rhea activity mapping has an invalid EC hierarchy")
    return {"EC_L3": ec_l3, "EC_L4": ec_l4, "EXACT_RHEA": rhea}, basis


def _audit_status(
    existing_level: str,
    existing_label: str,
    model_row: pd.Series,
    activities: pd.DataFrame,
    query_id: str,
) -> tuple[str, str, str, str]:
    """Return status, model label at comparison level, reason and basis."""
    accepted = str(model_row.get("final_decision", "")) == "ACCEPT"
    accepted_level = str(model_row.get("recommended_level", ""))
    if accepted_level not in LEVEL_RANK:
        candidate = _model_label(model_row, existing_level)
        if candidate:
            return "UNSUPPORTED", candidate, "MODEL_EVALUATED_LEVEL_BUT_DID_NOT_ACCEPT_SUPPORT", "DEFERRED_CANDIDATE"
        return "UNKNOWN", "", "MODEL_HAS_NO_CANDIDATE_AT_EXISTING_RESOLUTION", "NO_COMPARABLE_EVIDENCE"
    if not accepted:
        candidate = _model_label(model_row, existing_level)
        if candidate:
            return "UNSUPPORTED", candidate, "MODEL_EVALUATED_LEVEL_BUT_DID_NOT_ACCEPT_SUPPORT", "DEFERRED_CANDIDATE"
        return "UNKNOWN", "", "MODEL_HAS_NO_CANDIDATE_AT_EXISTING_RESOLUTION", "NO_COMPARABLE_EVIDENCE"

    accepted_labels = {level: _model_label(model_row, level) for level in LEVEL_RANK}
    if accepted_level == "EC_L4":
        accepted_ec4 = canonicalize_ec(model_row.get("recommended_label", accepted_labels["EC_L4"]), level=4)
        if accepted_ec4 and accepted_labels["EC_L4"] and accepted_ec4 != accepted_labels["EC_L4"]:
            raise ValueError("Accepted EC-L4 recommendation differs from its candidate label")
        accepted_labels["EC_L4"] = accepted_ec4 or accepted_labels["EC_L4"]
        ancestor = ec_l3_from_l4(accepted_labels["EC_L4"])
        if accepted_labels["EC_L3"] and accepted_labels["EC_L3"] != ancestor:
            raise ValueError("Accepted EC-L4 evidence has an inconsistent EC-L3 candidate")
        accepted_labels["EC_L3"] = ancestor
    elif accepted_level == "EC_L3":
        accepted_ec3 = canonicalize_ec(model_row.get("recommended_label", accepted_labels["EC_L3"]), level=3)
        if accepted_ec3 and accepted_labels["EC_L3"] and accepted_ec3 != accepted_labels["EC_L3"]:
            raise ValueError("Accepted EC-L3 recommendation differs from its candidate label")
        accepted_labels["EC_L3"] = accepted_ec3 or accepted_labels["EC_L3"]
    else:
        accepted_rhea = canonicalize_rhea(
            model_row.get("recommended_label", accepted_labels["EXACT_RHEA"])
        )
        if accepted_rhea and accepted_labels["EXACT_RHEA"] and accepted_rhea != accepted_labels["EXACT_RHEA"]:
            raise ValueError("Accepted Rhea recommendation differs from its candidate label")
        accepted_labels["EXACT_RHEA"] = accepted_rhea or accepted_labels["EXACT_RHEA"]

    if LEVEL_RANK[accepted_level] < LEVEL_RANK[existing_level]:
        candidate = accepted_labels[existing_level]
        if candidate:
            return "UNSUPPORTED", candidate, "ACCEPTED_EVIDENCE_IS_SHALLOWER_THAN_EXISTING_ANNOTATION", "SHALLOWER_ACCEPTED_LEVEL"
        return "UNKNOWN", "", "MODEL_HAS_NO_CANDIDATE_AT_EXISTING_RESOLUTION", "NO_COMPARABLE_EVIDENCE"

    basis = "SAME_RESOLUTION_ACCEPTED_EVIDENCE"
    if accepted_level == "EXACT_RHEA" and existing_level in {"EC_L3", "EC_L4"}:
        mapping, basis = _accepted_activity_mapping(
            model_row, activities, query_id, accepted_labels["EXACT_RHEA"]
        )
        if mapping is None:
            return "UNKNOWN", "", "EXACT_RHEA_ANCESTOR_REQUIRES_UNIQUE_ACTIVITY_MAPPING", basis
        accepted_labels.update(mapping)
    elif accepted_level == "EC_L4" and existing_level == "EC_L3":
        basis = "ACCEPTED_EC_L4_CHILD_TO_EC_L3_ANCESTOR"

    comparable = accepted_labels[existing_level]
    if not comparable:
        return "UNKNOWN", "", "MODEL_HAS_NO_CANONICAL_LABEL_AT_COMPARABLE_RESOLUTION", basis
    if comparable == existing_label:
        return "SUPPORTED", comparable, "ACCEPTED_MODEL_EVIDENCE_SUPPORTS_EXISTING_ANNOTATION", basis
    return "CONFLICT", comparable, "ACCEPTED_MODEL_EVIDENCE_IS_HIERARCHICALLY_INCOMPATIBLE", basis


def audit_annotation_bundle(
    existing_annotations: str | Path,
    model_bundle: str | Path,
    output_dir: str | Path,
) -> Path:
    existing_path = Path(existing_annotations).resolve(strict=True)
    existing = pd.read_csv(existing_path, sep="\t", dtype=str, keep_default_na=False)
    missing = [column for column in EXISTING_COLUMNS if column not in existing]
    if missing:
        raise ValueError(f"Existing annotation table missing columns: {missing}")
    existing = existing[EXISTING_COLUMNS].copy()
    if existing["query_protein_id"].duplicated().any():
        raise ValueError("Existing annotation table contains duplicate queries")
    model = read_bundle(model_bundle)
    model_predictions = model["predictions"].copy()
    if model_predictions["query_protein_id"].astype(str).duplicated().any():
        raise ValueError("Model bundle contains duplicate query predictions")
    model_lookup = {
        str(row["query_protein_id"]): row for _, row in model_predictions.iterrows()
    }
    rows = []
    evidence_rows = []
    for existing_row in existing.itertuples(index=False):
        query_id = str(existing_row.query_protein_id)
        canonical = _canonical_existing(pd.Series(existing_row._asdict()))
        level = canonical["level"]
        existing_label = canonical["label"]
        model_row = model_lookup.get(query_id)
        if not level or model_row is None:
            status = "UNKNOWN"
            model_label = ""
            reason = "NO_EXISTING_LABEL_OR_NO_MODEL_EVIDENCE"
            comparison_basis = "NO_COMPARABLE_EVIDENCE"
        else:
            status, model_label, reason, comparison_basis = _audit_status(
                level, existing_label, model_row, model["activities"], query_id
            )
        row = {
            "query_protein_id": query_id,
            "existing_level": level,
            "existing_label": existing_label,
            "existing_label_original": canonical["original_label"],
            "existing_ec_l3_original": str(existing_row.ec_l3),
            "existing_ec_l4_original": str(existing_row.ec_l4),
            "existing_rhea_id_original": str(existing_row.rhea_id),
            "existing_ec_l3_canonical": canonical["ec_l3"],
            "existing_ec_l4_canonical": canonical["ec_l4"],
            "existing_rhea_id_canonical": canonical["rhea_id"],
            "annotation_source": str(existing_row.annotation_source),
            "model_label": model_label,
            "audit_status": status,
            "decision_reason": reason,
            "comparison_basis": comparison_basis,
        }
        rows.append(row)
        evidence_rows.append(dict(row))
    predictions = pd.DataFrame(rows)
    evidence = {
        "format": "siteguard.annotation-audit-evidence.v1",
        "workflow": "audit",
        "truth_free": True,
        "query_outcomes_used": False,
        "status_definitions": {
            "SUPPORTED": "accepted evidence matches the existing label at the same or a deeper mapped resolution",
            "CONFLICT": "accepted evidence is incompatible at a comparable or deeper mapped resolution",
            "UNSUPPORTED": "the model evaluated the level but deferred, or accepted only a shallower level",
            "UNKNOWN": "the existing label or comparable model evidence is absent",
        },
        "canonicalization": "Common EC:/whitespace and RHEA:/numeric presentations are normalized; originals are retained.",
        "rhea_fallback_rule": "Exact Rhea may support EC ancestors only through a unique activity-table mapping.",
        "rows": evidence_rows,
    }
    return write_bundle_atomic(
        output_dir,
        workflow="audit",
        predictions=predictions,
        references=model["references"],
        activities=model["activities"],
        evidence=evidence,
        report_html=render_predictions_html(predictions, title="SiteGuard annotation audit"),
        inputs={
            "existing_annotations": existing_path,
            "model_predictions": Path(model_bundle) / "predictions.parquet",
            "model_manifest": Path(model_bundle) / "run_manifest.json",
        },
        parameters={"comparison_scope": "existing EC/Rhea versus accepted/deferred model evidence"},
        warnings=("UNKNOWN and UNSUPPORTED are not negative biological labels.",),
    )
