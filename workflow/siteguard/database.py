"""Build a small, redistributable reference/activity bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .fasta import read_fasta, sequence_sha256
from .identifiers import canonicalize_ec, canonicalize_rhea, ec_l3_from_l4
from .reporting import render_predictions_html
from .run_bundle import write_bundle_atomic


REFERENCE_COLUMNS = ["reference_id", "sequence_sha256", "sequence_length", "sequence", "source"]
ACTIVITY_COLUMNS = ["activity_id", "reference_id", "ec_l3", "ec_l4", "rhea_id", "evidence_tier", "source"]
EMPTY_PREDICTION_COLUMNS = ["query_protein_id", "final_decision", "decision_reason"]


def _load_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path).resolve(strict=True)
    # The committed .yaml is deliberately JSON-compatible, avoiding an undeclared YAML dependency.
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format") != "siteguard.reference-db-config.v1":
        raise ValueError("Unsupported reference database configuration")
    return config


def build_reference_bundle(
    reference_fasta: str | Path,
    activities_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> Path:
    records = read_fasta(reference_fasta)
    config = _load_config(config_path)
    references = pd.DataFrame([
        {
            "reference_id": identifier,
            "sequence_sha256": sequence_sha256(sequence),
            "sequence_length": len(sequence),
            "sequence": sequence,
            "source": str(config.get("reference_source", "user-supplied")),
        }
        for identifier, sequence in records.items()
    ], columns=REFERENCE_COLUMNS)
    activities = pd.read_csv(activities_path, sep="\t", dtype=str, keep_default_na=False)
    missing = [column for column in ACTIVITY_COLUMNS if column not in activities]
    if missing:
        raise ValueError(f"Activity table missing columns: {missing}")
    activities = activities[ACTIVITY_COLUMNS].copy()
    if activities.empty:
        raise ValueError("Activity table is empty")
    if activities["activity_id"].duplicated().any():
        raise ValueError("Activity table contains duplicate activity_id values")
    if activities.duplicated(["reference_id", "activity_id"]).any():
        raise ValueError("Activity table contains duplicate reference/activity pairs")
    activities["ec_l3"] = activities["ec_l3"].map(lambda value: canonicalize_ec(value, level=3))
    activities["ec_l4"] = activities["ec_l4"].map(lambda value: canonicalize_ec(value, level=4))
    activities["rhea_id"] = activities["rhea_id"].map(canonicalize_rhea)
    inconsistent = activities.apply(
        lambda row: bool(row["ec_l4"]) and ec_l3_from_l4(row["ec_l4"]) != row["ec_l3"],
        axis=1,
    )
    if inconsistent.any():
        raise ValueError("Activity table contains an EC-L4 label inconsistent with its EC-L3 ancestor")
    unknown = sorted(set(activities["reference_id"]) - set(references["reference_id"]))
    if unknown:
        raise ValueError(f"Activities refer to unknown references: {unknown[:5]}")
    predictions = pd.DataFrame(columns=EMPTY_PREDICTION_COLUMNS)
    evidence = {
        "format": "siteguard.reference-bundle-evidence.v1",
        "workflow": "build-db",
        "truth_free": True,
        "reference_count": len(references),
        "activity_count": len(activities),
        "config": config,
        "scope": "reference identities and declared activities; no query correctness evidence",
    }
    return write_bundle_atomic(
        output_dir,
        workflow="build-db",
        predictions=predictions,
        references=references,
        activities=activities,
        evidence=evidence,
        report_html=render_predictions_html(predictions, title="SiteGuard reference database build"),
        inputs={
            "reference_fasta": reference_fasta,
            "activities": activities_path,
            "config": config_path,
        },
        parameters={"database_format": config["format"]},
        warnings=("This miniature bundle is a reference/activity fixture, not a trained-model asset pack.",),
    )
