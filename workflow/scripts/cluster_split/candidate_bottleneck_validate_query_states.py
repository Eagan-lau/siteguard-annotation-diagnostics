#!/usr/bin/env python3
"""Independent fail-closed audit for the frozen Phase433 descriptive repair.

This module deliberately does not import, execute, or dynamically load the
Phase433 producer.  It verifies opaque identities before loading the Phase12
row-level table, reconstructs every persisted estimand, and creates the final
checkpoint only after all checks pass.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FORMAL_STATUS = "FROZEN_IDENTITY_LOCK_READY_FOR_SINGLE_PHASE433_RUN"
CONTRACT_FORMAT = "siteguard.phase433.frozen-phase12-rowlevel-bottleneck-lock.v1"
FORMAL_RELATIVE = "models/phase433_frozen_phase12_rowlevel_bottleneck_lock.json"
RECEIPT_RELATIVE = "reports/phase433_remote_identity_preflight_20260906.json"
EXPECTED_OUTPUT_RELATIVE = "reports/phase433_frozen_phase12_rowlevel_bottleneck_closure_20260906"
LEVELS = ("EC_L3", "EC_L4", "EXACT_RHEA")
SIX_STATES = (
    "CANDIDATE_ABSENT_REJECTED",
    "CANDIDATE_ABSENT_ACCEPTED_WRONG",
    "CANDIDATE_PRESENT_TOP1_WRONG_REJECTED",
    "CANDIDATE_PRESENT_TOP1_WRONG_ACCEPTED",
    "TOP1_CORRECT_REJECTED",
    "TOP1_CORRECT_ACCEPTED",
)
COMPATIBLE_STATES = (
    "CANDIDATE_ABSENT_REJECTED",
    "CANDIDATE_PRESENT_TOP1_WRONG_REJECTED",
    "TOP1_CORRECT_REJECTED",
    "TOP1_WRONG_ACCEPTED_UNSUPPORTED_TRANSFER",
    "TOP1_CORRECT_ACCEPTED",
)
BOOTSTRAP_METRICS = SIX_STATES + (
    "CANDIDATE_AVAILABLE",
    "TOP1_CORRECT_GIVEN_CANDIDATE_AVAILABLE",
    "COVERAGE",
    "ACCEPTED_PRECISION",
    "TOP1_CORRECT_REJECTED_FRACTION",
    "TOP1_WRONG_ACCEPTED_FRACTION",
)
PRODUCER_FILES = (
    "phase433_input_identity.tsv",
    "phase433_query_stage_ledger.parquet",
    "phase433_exact_six_state_partition.tsv",
    "phase433_phase227_compatible_partition.tsv",
    "phase433_cluster_bootstrap.tsv",
    "phase433_exploratory_strata.tsv",
    "phase433_producer_check_matrix.tsv",
    "phase433_producer_summary.json",
    "phase433_producer_report.md",
    "PHASE433_PRODUCER_PASS_NOT_CONFIRMATORY_NOT_SUBMISSION_READY.json",
)
AUDIT_FILES = (
    "phase433_independent_check_matrix.tsv",
    "phase433_independent_identity_manifest.tsv",
    "phase433_independent_summary.json",
    "phase433_independent_report.md",
    "CHECKPOINT_433_PASS_FROZEN_PHASE12_ROWLEVEL_BOTTLENECK_CLOSURE_DESCRIPTIVE_NOT_CONFIRMATORY_NOT_PROJECT_COMPLETE.json",
)
LOCKED_INPUT_PATHS = (
    "data/splits/split_sequence.parquet", "data/splits/split_family.parquet",
    "data/processed/protein_table.parquet", "results/phase12/end_to_end_metrics.tsv",
    "results/phase12/error_decomposition.tsv", "reports/phase12_summary.json",
    "reports/phase12_qc.tsv",
    "reports/phase227_candidate_error_decomposition_v2_20260902/phase227_summary.json",
    "reports/phase227_candidate_error_decomposition_v2_20260902/phase227_phase12_exclusive_partition.tsv",
)
IMPLEMENTATION_PATHS = (
    "PROJECT/Phase433_frozen_phase12_rowlevel_bottleneck_closure_protocol.md",
    "scripts/candidate_bottleneck_query_states.py",
    "scripts/candidate_bottleneck_validate_query_states.py",
    "scripts/phase433_submit_frozen_phase12_rowlevel_bottleneck_closure.sh",
)
EXPECTED_STRATA = {
    "pfam_mapping": ["PFAM_WITH_CLAN", "PFAM_NO_CLAN", "NO_PFAM"],
    "taxonomy_domain": ["BACTERIA", "ARCHAEA", "EUKARYOTA", "VIRUSES", "OTHER_OR_UNKNOWN"],
    "structure_availability": ["PDB_AVAILABLE", "AF_ONLY", "NO_STRUCTURE"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def add_check(rows: list[dict[str, Any]], check_id: str, passed: bool, observed: Any) -> None:
    rows.append({
        "check_id": check_id,
        "status": "PASS" if passed else "FAIL",
        "observed": json.dumps(observed, ensure_ascii=False, sort_keys=True)
        if isinstance(observed, (dict, list, tuple)) else str(observed),
    })


def require_checks(rows: list[dict[str, Any]], context: str) -> None:
    failures = [row for row in rows if row["status"] != "PASS"]
    if failures:
        detail = "\n".join(f"{row['check_id']}: {row['observed']}" for row in failures)
        raise RuntimeError(f"{context} failed closed:\n{detail}")


def safe_file(root: Path, relative: str, forbidden: Iterable[str]) -> Path:
    normalized = relative.replace("\\", "/")
    if Path(relative).is_absolute() or ".." in Path(normalized).parts:
        raise RuntimeError(f"unsafe relative path: {relative}")
    if any(token.lower() in normalized.lower() for token in forbidden):
        raise RuntimeError(f"forbidden path token: {relative}")
    candidate = root / Path(normalized)
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError(f"missing, non-regular, or symlink input: {relative}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"input escapes V4 root: {relative}") from exc
    return resolved


def safe_analysis_dir(root: Path, relative: str) -> Path:
    normalized = relative.replace("\\", "/")
    if Path(relative).is_absolute() or ".." in Path(normalized).parts or "phase99" in normalized.lower():
        raise RuntimeError(f"unsafe analysis directory: {relative}")
    path = root / Path(normalized)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("analysis directory is missing, non-directory, or a symlink")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("analysis directory escapes V4 root") from exc
    return resolved


def parquet_metadata(path: Path) -> tuple[int, int, int, list[str], list[str], list[str], list[str]]:
    parquet = pq.ParquetFile(path)
    return (
        int(parquet.metadata.num_rows),
        int(parquet.metadata.num_columns),
        int(parquet.metadata.num_row_groups),
        list(parquet.schema_arrow.names),
        list(parquet.schema.names),
        [str(field.type) for field in parquet.schema_arrow],
        [str(parquet.schema.column(index).physical_type) for index in range(parquet.metadata.num_columns)],
    )


def truth_bool(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any():
        raise RuntimeError(f"null boolean values in {name}")
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise RuntimeError(f"invalid boolean values in {name}")
    return normalized.isin({"true", "1"})


def scalar_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false", "1", "0"}:
        raise RuntimeError(f"invalid boolean scalar in {name}: {value!r}")
    return normalized in {"true", "1"}


def family_stratum(primary_pfam: pd.Series, primary_clan: pd.Series) -> pd.Series:
    pfam = primary_pfam.fillna("").astype(str).str.strip()
    clan = primary_clan.fillna("").astype(str).str.strip()
    has_pfam = pfam.ne("") & ~pfam.str.upper().isin({"UNASSIGNED", "NAN", "NONE", "NA"})
    has_clan = clan.ne("") & ~clan.str.upper().isin({"NO_CLAN", "UNASSIGNED", "NAN", "NONE", "NA"})
    return pd.Series(np.select(
        [~has_pfam, has_pfam & ~has_clan],
        ["NO_PFAM", "PFAM_NO_CLAN"],
        default="PFAM_WITH_CLAN",
    ), index=primary_pfam.index)


def taxonomy_stratum(lineage: pd.Series) -> pd.Series:
    text = lineage.fillna("").astype(str).str.upper()
    return pd.Series(np.select(
        [text.str.contains("BACTERIA", regex=False), text.str.contains("ARCHAEA", regex=False),
         text.str.contains("EUKARYOTA", regex=False), text.str.contains("VIRUSES", regex=False)],
        ["BACTERIA", "ARCHAEA", "EUKARYOTA", "VIRUSES"],
        default="OTHER_OR_UNKNOWN",
    ), index=lineage.index)


def structure_stratum(pdb_available: pd.Series, af_available: pd.Series) -> pd.Series:
    pdb_flag = truth_bool(pdb_available, "pdb_availability")
    af_flag = truth_bool(af_available, "alphafold_availability")
    return pd.Series(np.select(
        [pdb_flag, ~pdb_flag & af_flag],
        ["PDB_AVAILABLE", "AF_ONLY"],
        default="NO_STRUCTURE",
    ), index=pdb_available.index)


def classify(available: pd.Series, correct: pd.Series, accepted: pd.Series) -> pd.Series:
    if (correct & ~available).any():
        raise RuntimeError("top_correct=true with oracle_candidate_available=false")
    state = pd.Series(np.select(
        [
            ~available & ~accepted,
            ~available & accepted,
            available & ~correct & ~accepted,
            available & ~correct & accepted,
            available & correct & ~accepted,
            available & correct & accepted,
        ],
        SIX_STATES,
        default="INVALID",
    ), index=available.index)
    if state.eq("INVALID").any():
        raise RuntimeError("unclassified Phase433 state")
    return state


def metric_values(counts: np.ndarray) -> dict[str, float]:
    values_array = np.asarray(counts, dtype=float)
    total = float(values_array.sum())
    values = {
        state: values_array[index] / total if total else math.nan
        for index, state in enumerate(SIX_STATES)
    }
    available = float(values_array[2:].sum())
    accepted = float(values_array[[1, 3, 5]].sum())
    values.update({
        "CANDIDATE_AVAILABLE": available / total if total else math.nan,
        "TOP1_CORRECT_GIVEN_CANDIDATE_AVAILABLE": float(values_array[[4, 5]].sum()) / available if available else math.nan,
        "COVERAGE": accepted / total if total else math.nan,
        "ACCEPTED_PRECISION": float(values_array[5]) / accepted if accepted else math.nan,
        "TOP1_CORRECT_REJECTED_FRACTION": float(values_array[4]) / total if total else math.nan,
        "TOP1_WRONG_ACCEPTED_FRACTION": float(values_array[[1, 3]].sum()) / total if total else math.nan,
    })
    return values


def context_seed(level: str, primary_seed: int) -> int:
    payload = f"{primary_seed}|PHASE433|{level}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def independent_bootstrap(frame: pd.DataFrame, level: str, replicates: int, primary_seed: int) -> list[dict[str, Any]]:
    cluster_labels = sorted(frame["cluster_id_30"].astype(str).unique())
    cluster_lookup = {label: position for position, label in enumerate(cluster_labels)}
    state_lookup = {state: position for position, state in enumerate(SIX_STATES)}
    count_matrix = np.zeros((len(cluster_labels), len(SIX_STATES)), dtype=np.int64)
    row_clusters = frame["cluster_id_30"].astype(str).map(cluster_lookup).to_numpy(dtype=np.int64)
    row_states = frame["stage"].map(state_lookup).to_numpy(dtype=np.int64)
    np.add.at(count_matrix, (row_clusters, row_states), 1)
    point = metric_values(count_matrix.sum(axis=0))
    seed = context_seed(level, primary_seed)
    generator = np.random.default_rng(seed)
    draws = np.empty((replicates, len(BOOTSTRAP_METRICS)), dtype=np.float64)
    number_clusters = len(cluster_labels)
    for index in range(replicates):
        selected = generator.integers(0, number_clusters, size=number_clusters)
        multiplicity = np.bincount(selected, minlength=number_clusters)
        values = metric_values(multiplicity @ count_matrix)
        draws[index] = [values[metric] for metric in BOOTSTRAP_METRICS]
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return [{
        "annotation_level": level,
        "metric": metric,
        "estimate": point[metric],
        "ci_percentile_low": float(low[position]),
        "ci_percentile_high": float(high[position]),
        "bootstrap_unit": "cluster_id_30",
        "clusters": number_clusters,
        "queries": len(frame),
        "replicates": replicates,
        "context_seed": seed,
        "interpretation": "DESCRIPTIVE_CLUSTER_BOOTSTRAP_NOT_RELEASE_CERTIFICATION",
    } for position, metric in enumerate(BOOTSTRAP_METRICS)]


def construct_ledger(paths: dict[str, Path], row_key: str, checks: list[dict[str, Any]]) -> pd.DataFrame:
    phase12 = pd.read_parquet(paths[row_key])
    add_check(checks, "AUDIT_ROWLEVEL_QUERY_UNIQUE", len(phase12) == 27639 and phase12["query_protein_id"].notna().all() and phase12["query_protein_id"].is_unique, len(phase12))
    sequence = pd.read_parquet(paths["data/splits/split_sequence.parquet"], columns=["protein_id", "cluster_id_30", "split"])
    family = pd.read_parquet(paths["data/splits/split_family.parquet"], columns=["protein_id", "primary_pfam", "primary_pfam_clan"])
    protein = pd.read_parquet(paths["data/processed/protein_table.parquet"], columns=[
        "protein_id", "lineage_json", "alphafold_availability", "pdb_availability",
    ])
    add_check(checks, "AUDIT_SEQUENCE_KEY_UNIQUE", sequence["protein_id"].is_unique, len(sequence))
    add_check(checks, "AUDIT_FAMILY_KEY_UNIQUE", family["protein_id"].is_unique, len(family))
    add_check(checks, "AUDIT_PROTEIN_KEY_UNIQUE", protein["protein_id"].is_unique, len(protein))
    joined = phase12.merge(sequence, left_on="query_protein_id", right_on="protein_id", how="left", validate="one_to_one").drop(columns="protein_id")
    joined = joined.merge(family, left_on="query_protein_id", right_on="protein_id", how="left", validate="one_to_one").drop(columns="protein_id")
    joined = joined.merge(protein, left_on="query_protein_id", right_on="protein_id", how="left", validate="one_to_one").drop(columns="protein_id")
    required = ["cluster_id_30", "split", "lineage_json", "alphafold_availability", "pdb_availability"]
    add_check(checks, "AUDIT_JOIN_COMPLETE", not joined[required].isna().any().any(), joined[required].isna().sum().to_dict())
    add_check(checks, "AUDIT_FROZEN_TEST_ONLY", joined["split"].eq("test").all(), joined["split"].value_counts(dropna=False).to_dict())
    joined["pfam_mapping_stratum"] = family_stratum(joined["primary_pfam"], joined["primary_pfam_clan"])
    joined["taxonomy_domain_stratum"] = taxonomy_stratum(joined["lineage_json"])
    joined["structure_availability_stratum"] = structure_stratum(joined["pdb_availability"], joined["alphafold_availability"])
    parts: list[pd.DataFrame] = []
    for level_order, level in enumerate(LEVELS, start=1):
        available = truth_bool(joined[f"oracle_candidate_available_{level}"], f"oracle_candidate_available_{level}")
        correct = truth_bool(joined[f"top_correct_{level}"], f"top_correct_{level}")
        accepted = truth_bool(joined[f"accepted_{level}"], f"accepted_{level}")
        part = joined[[
            "query_protein_id", "cluster_id_30", "pfam_mapping_stratum",
            "taxonomy_domain_stratum", "structure_availability_stratum",
        ]].copy()
        part.insert(1, "annotation_level", level)
        part.insert(2, "level_order", level_order)
        part["oracle_candidate_available"] = available.to_numpy()
        part["top_correct"] = correct.to_numpy()
        part["accepted"] = accepted.to_numpy()
        part["stage"] = classify(available, correct, accepted).to_numpy()
        parts.append(part)
    ledger = pd.concat(parts, ignore_index=True).sort_values(
        ["level_order", "query_protein_id"], kind="stable"
    ).reset_index(drop=True)
    add_check(checks, "AUDIT_LEDGER_ROWS", len(ledger) == 82917, len(ledger))
    add_check(checks, "AUDIT_LEDGER_UNIQUE", not ledger.duplicated(["annotation_level", "query_protein_id"]).any(), len(ledger))
    return ledger


def construct_partitions(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact_rows: list[dict[str, Any]] = []
    compatible_rows: list[dict[str, Any]] = []
    compatible_mapping = {
        "CANDIDATE_ABSENT_REJECTED": ("CANDIDATE_ABSENT_REJECTED",),
        "CANDIDATE_PRESENT_TOP1_WRONG_REJECTED": ("CANDIDATE_PRESENT_TOP1_WRONG_REJECTED",),
        "TOP1_CORRECT_REJECTED": ("TOP1_CORRECT_REJECTED",),
        "TOP1_WRONG_ACCEPTED_UNSUPPORTED_TRANSFER": (
            "CANDIDATE_ABSENT_ACCEPTED_WRONG", "CANDIDATE_PRESENT_TOP1_WRONG_ACCEPTED",
        ),
        "TOP1_CORRECT_ACCEPTED": ("TOP1_CORRECT_ACCEPTED",),
    }
    for level_order, level in enumerate(LEVELS, start=1):
        block = ledger.loc[ledger["annotation_level"].eq(level)]
        counts = block["stage"].value_counts().reindex(SIX_STATES, fill_value=0)
        denominator = len(block)
        for state_order, state in enumerate(SIX_STATES, start=1):
            count = int(counts[state])
            exact_rows.append({
                "annotation_level": level, "level_order": level_order,
                "state": state, "state_order": state_order,
                "queries": count, "denominator": denominator, "fraction": count / denominator,
                "exclusive": True, "point_identified": True,
                "interpretation": "DOCUMENTED_ANNOTATION_CONCORDANCE_ONLY",
            })
        for component_order, component in enumerate(COMPATIBLE_STATES, start=1):
            count = int(sum(counts[state] for state in compatible_mapping[component]))
            compatible_rows.append({
                "annotation_level": level, "level_order": level_order,
                "component": component, "component_order": component_order,
                "queries": count, "denominator": denominator, "fraction": count / denominator,
                "exclusive_partition": True, "point_identified": True,
                "phase227_prior_status": "WAS_BOUNDED" if component in COMPATIBLE_STATES[:2] else "WAS_POINT_IDENTIFIED",
            })
    return pd.DataFrame(exact_rows), pd.DataFrame(compatible_rows)


def construct_strata(ledger: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    dimensions = {
        "PFAM_MAPPING": ("pfam_mapping_stratum", contract["exploratory_strata"]["pfam_mapping"]),
        "TAXONOMY_DOMAIN": ("taxonomy_domain_stratum", contract["exploratory_strata"]["taxonomy_domain"]),
        "STRUCTURE_AVAILABILITY": ("structure_availability_stratum", contract["exploratory_strata"]["structure_availability"]),
    }
    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        level_frame = ledger.loc[ledger["annotation_level"].eq(level)]
        for dimension, (column, categories) in dimensions.items():
            for category in categories:
                block = level_frame.loc[level_frame[column].eq(category)]
                counts = block["stage"].value_counts().reindex(SIX_STATES, fill_value=0).to_numpy(dtype=int)
                values = metric_values(counts)
                for metric in BOOTSTRAP_METRICS:
                    rows.append({
                        "annotation_level": level, "stratum_dimension": dimension,
                        "stratum": category, "metric": metric, "queries": len(block),
                        "clusters": int(block["cluster_id_30"].nunique()),
                        "estimate": values[metric],
                        "status": "NONCONFIRMATORY_EXPLORATORY_POINT_ESTIMATE_ONLY",
                    })
    return pd.DataFrame(rows)


def frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> tuple[bool, str]:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True), right.reset_index(drop=True),
            check_dtype=False, check_like=False, rtol=1e-12, atol=1e-12,
        )
        return True, f"rows={len(left)} columns={len(left.columns)}"
    except AssertionError as exc:
        return False, str(exc)[:2000]


def independent_source_check(auditor_path: Path) -> dict[str, Any]:
    tree = ast.parse(auditor_path.read_text(encoding="utf-8"), filename=str(auditor_path))
    imports: list[str] = []
    dangerous_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "compile", "__import__"}:
                dangerous_calls.append(node.func.id)
    forbidden_imports = [name for name in imports if name.startswith(("subprocess", "socket", "requests")) or "candidate_bottleneck_query_states" in name]
    return {"imports": sorted(imports), "forbidden_imports": forbidden_imports, "dangerous_calls": dangerous_calls}


def reconcile_prior(paths: dict[str, Path], exact: pd.DataFrame, compatible: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    prior = pd.read_csv(paths["reports/phase227_candidate_error_decomposition_v2_20260902/phase227_phase12_exclusive_partition.tsv"], sep="\t").set_index(["annotation_level", "component"])
    current = compatible.set_index(["annotation_level", "component"])
    exact_indexed = exact.set_index(["annotation_level", "state"])
    for level in LEVELS:
        add_check(checks, f"AUDIT_EXACT_PARTITION_CLOSE::{level}", int(current.loc[level, "queries"].sum()) == 27639, int(current.loc[level, "queries"].sum()))
        for component in COMPATIBLE_STATES:
            value = int(current.loc[(level, component), "queries"])
            old = prior.loc[(level, component)]
            low, high = int(old["numerator_lower"]), int(old["numerator_upper"])
            add_check(checks, f"AUDIT_PHASE227_BOUND::{level}::{component}", low <= value <= high, {"value": value, "lower": low, "upper": high})
            if scalar_bool(old["point_identified"], f"phase227 point_identified {level} {component}"):
                add_check(checks, f"AUDIT_PHASE227_POINT::{level}::{component}", value == low == high, {"value": value, "prior": low})
    phase12_summary = json.loads(paths["reports/phase12_summary.json"].read_text(encoding="utf-8"))
    phase12_qc = pd.read_csv(paths["reports/phase12_qc.tsv"], sep="\t")
    add_check(checks, "AUDIT_PHASE12_STATUS", phase12_summary.get("status") == "PASS" and str(phase12_summary.get("slurm_job_id")) == "1496373", phase12_summary)
    add_check(checks, "AUDIT_PHASE12_QC", phase12_qc["status"].eq("PASS").all(), phase12_qc["status"].value_counts().to_dict())
    phase12_metrics = pd.read_csv(paths["results/phase12/end_to_end_metrics.tsv"], sep="\t")
    per_level = phase12_metrics.loc[phase12_metrics["analysis"].eq("per_level_top_candidate")].set_index("annotation_level")
    add_check(checks, "AUDIT_PHASE12_METRIC_ROWS_EXACT", len(per_level) == len(LEVELS) and per_level.index.is_unique and set(per_level.index) == set(LEVELS), list(per_level.index))
    error = pd.read_csv(paths["results/phase12/error_decomposition.tsv"], sep="\t")
    error_indexed = error.set_index(["annotation_level", "error_component"])
    expected_error_components = {
        "NO_POSITIVE_CANDIDATE_RETRIEVAL_LIMIT",
        "POSITIVE_PRESENT_BUT_TOP1_WRONG_RANKING",
        "TOP1_CORRECT_BUT_ABSTAINED",
        "ACCEPTED_WRONG_OVERANNOTATION",
        "ACCEPTED_CORRECT",
    }
    add_check(checks, "AUDIT_PHASE12_ERROR_ROWS_EXACT", len(error) == len(LEVELS) * len(expected_error_components) and error_indexed.index.is_unique and set(error["annotation_level"]) == set(LEVELS) and set(error["error_component"]) == expected_error_components and error["evaluation_split"].eq("population_test").all(), {"rows": len(error), "levels": sorted(error["annotation_level"].unique()), "components": sorted(error["error_component"].unique())})
    for level in LEVELS:
        accepted = int(current.loc[(level, "TOP1_WRONG_ACCEPTED_UNSUPPORTED_TRANSFER"), "queries"] + current.loc[(level, "TOP1_CORRECT_ACCEPTED"), "queries"])
        correct = int(current.loc[(level, "TOP1_CORRECT_ACCEPTED"), "queries"])
        saved = per_level.loc[level]
        add_check(checks, f"AUDIT_PHASE12_QUERY_DENOMINATOR::{level}", int(saved["queries"]) == 27639, int(saved["queries"]))
        add_check(checks, f"AUDIT_PHASE12_ACCEPTED::{level}", accepted == int(saved["accepted_queries"]), {"observed": accepted, "saved": int(saved["accepted_queries"])})
        precision = correct / accepted if accepted else math.nan
        add_check(checks, f"AUDIT_PHASE12_PRECISION::{level}", abs(precision - float(saved["precision_among_accepted"])) < 1e-12, {"observed": precision, "saved": float(saved["precision_among_accepted"])})
        state = {name: int(exact_indexed.loc[(level, name), "queries"]) for name in SIX_STATES}
        reconstructed = {
            "NO_POSITIVE_CANDIDATE_RETRIEVAL_LIMIT": state["CANDIDATE_ABSENT_REJECTED"] + state["CANDIDATE_ABSENT_ACCEPTED_WRONG"],
            "POSITIVE_PRESENT_BUT_TOP1_WRONG_RANKING": state["CANDIDATE_PRESENT_TOP1_WRONG_REJECTED"] + state["CANDIDATE_PRESENT_TOP1_WRONG_ACCEPTED"],
            "TOP1_CORRECT_BUT_ABSTAINED": state["TOP1_CORRECT_REJECTED"],
            "ACCEPTED_WRONG_OVERANNOTATION": state["CANDIDATE_ABSENT_ACCEPTED_WRONG"] + state["CANDIDATE_PRESENT_TOP1_WRONG_ACCEPTED"],
            "ACCEPTED_CORRECT": state["TOP1_CORRECT_ACCEPTED"],
        }
        for component, observed in reconstructed.items():
            saved_error = error_indexed.loc[(level, component)]
            add_check(checks, f"AUDIT_PHASE12_ERROR_COMPONENT::{level}::{component}", observed == int(saved_error["queries"]) and abs(observed / 27639 - float(saved_error["fraction_of_queries"])) < 1e-12, {"observed": observed, "saved": int(saved_error["queries"])})
        candidate_available = sum(state[name] for name in SIX_STATES[2:])
        add_check(checks, f"AUDIT_PHASE12_ORACLE_COVERAGE::{level}", abs(candidate_available / 27639 - float(saved["oracle_candidate_coverage"])) < 1e-12, {"observed": candidate_available / 27639, "saved": float(saved["oracle_candidate_coverage"])})


def write_tsv_exclusive(frame: pd.DataFrame, path: Path) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)


def write_json_exclusive(payload: dict[str, Any], path: Path) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    contract_path = args.contract if args.contract.is_absolute() else root / args.contract
    contract_path = contract_path.resolve(strict=True)
    expected_contract = (root / FORMAL_RELATIVE).resolve(strict=True)
    if contract_path != expected_contract:
        raise RuntimeError(f"formal contract must be exactly {FORMAL_RELATIVE}")
    if contract_path.is_symlink() or not contract_path.is_file():
        raise RuntimeError("formal contract is missing, non-regular, or a symlink")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    add_check(checks, "AUDIT_CONTRACT_FORMAT", contract.get("format") == CONTRACT_FORMAT, contract.get("format"))
    add_check(checks, "AUDIT_CONTRACT_STATUS", contract.get("status") == FORMAL_STATUS, contract.get("status"))
    add_check(checks, "AUDIT_LEVELS", tuple(contract.get("levels", [])) == LEVELS, contract.get("levels"))
    add_check(checks, "AUDIT_SIX_STATES", tuple(contract.get("six_state_order", [])) == SIX_STATES, contract.get("six_state_order"))
    add_check(checks, "AUDIT_COMPATIBLE_STATES", tuple(contract.get("phase227_compatible_order", [])) == COMPATIBLE_STATES, contract.get("phase227_compatible_order"))
    runtime = contract.get("runtime", {})
    add_check(checks, "AUDIT_PYTHON_VERSION", runtime.get("expected_version") == ".".join(map(str, sys.version_info[:3])) == "3.11.15", ".".join(map(str, sys.version_info[:3])))
    add_check(checks, "AUDIT_PYTHON_ISOLATED_NO_BYTECODE", runtime.get("invocation_flags") == ["-I", "-B"] and sys.flags.isolated == 1 and sys.dont_write_bytecode, {"isolated": sys.flags.isolated, "dont_write_bytecode": sys.dont_write_bytecode})
    expected_python = Path(str(runtime.get("python_path", ""))).resolve(strict=False)
    add_check(checks, "AUDIT_PYTHON_PATH", Path(sys.executable).resolve(strict=True) == expected_python, str(Path(sys.executable).resolve(strict=True)))
    add_check(checks, "AUDIT_BOOTSTRAP", contract.get("cluster_bootstrap", {}).get("replicates") == 10000 and contract.get("cluster_bootstrap", {}).get("unit") == "cluster_id_30" and contract.get("cluster_bootstrap", {}).get("percentiles") == [0.025, 0.975], contract.get("cluster_bootstrap"))
    governance = contract.get("governance", {})
    add_check(checks, "AUDIT_GOVERNANCE", bool(governance) and all(value is False for value in governance.values()), governance)
    forbidden = tuple(contract.get("forbidden_path_tokens", ()))
    add_check(checks, "AUDIT_PHASE99_FORBIDDEN", any(str(token).lower() == "phase99" for token in forbidden), forbidden)
    add_check(checks, "AUDIT_LOGICAL_ROOT_EQUIVALENT", Path(str(contract.get("logical_v4_root", ""))).resolve(strict=True) == root, contract.get("logical_v4_root"))
    add_check(checks, "AUDIT_OUTPUT_DIRECTORY_EXACT", contract.get("expected_output_directory") == EXPECTED_OUTPUT_RELATIVE, contract.get("expected_output_directory"))
    add_check(checks, "AUDIT_LOCKED_INPUT_PATHS_EXACT", tuple(item.get("path") for item in contract.get("locked_inputs", [])) == LOCKED_INPUT_PATHS, [item.get("path") for item in contract.get("locked_inputs", [])])
    add_check(checks, "AUDIT_IMPLEMENTATION_PATHS_EXACT", tuple(item.get("path") for item in contract.get("implementation_identities", [])) == IMPLEMENTATION_PATHS, [item.get("path") for item in contract.get("implementation_identities", [])])
    add_check(checks, "AUDIT_STRATA_CATEGORIES_EXACT", contract.get("exploratory_strata", {}).get("pfam_mapping") == EXPECTED_STRATA["pfam_mapping"] and contract.get("exploratory_strata", {}).get("taxonomy_domain") == EXPECTED_STRATA["taxonomy_domain"] and contract.get("exploratory_strata", {}).get("structure_availability") == EXPECTED_STRATA["structure_availability"], contract.get("exploratory_strata"))
    require_checks(checks, "independent contract admission")

    finalization = contract.get("remote_identity_finalization", {})
    receipt_relative = str(finalization.get("receipt_path", ""))
    add_check(checks, "AUDIT_REMOTE_RECEIPT_PATH_EXACT", receipt_relative == RECEIPT_RELATIVE, receipt_relative)
    require_checks(checks, "independent remote receipt path admission")
    receipt_path = safe_file(root, receipt_relative, forbidden)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    contract_hash = sha256_file(contract_path)
    add_check(checks, "AUDIT_REMOTE_RECEIPT_REQUIRED", finalization.get("receipt_required") is True, finalization)
    add_check(checks, "AUDIT_REMOTE_RECEIPT_STATUS", receipt.get("status") == "PASS_PHASE433_REMOTE_IDENTITY_LOCK_FINALIZED_NO_TRUTH_ROWS_OPENED", receipt.get("status"))
    add_check(checks, "AUDIT_REMOTE_RECEIPT_CONTRACT", receipt.get("formal_lock_sha256") == contract_hash, receipt.get("formal_lock_sha256"))
    add_check(checks, "AUDIT_REMOTE_RECEIPT_NO_TRUTH_ROWS", receipt.get("phase12_dataframe_rows_loaded") == 0, receipt.get("phase12_dataframe_rows_loaded"))
    add_check(checks, "AUDIT_REMOTE_RECEIPT_NO_OUTPUT", receipt.get("analysis_output_directory_created") is False, receipt.get("analysis_output_directory_created"))
    require_checks(checks, "independent remote identity receipt admission")

    row_spec = contract.get("phase12_rowlevel_input", {})
    expected_row_hash = row_spec.get("sha256")
    if not is_sha256(expected_row_hash):
        raise RuntimeError("Phase12 row-level SHA-256 is absent or malformed; audit refuses truth opening")
    row_key = str(row_spec.get("path", ""))
    row_path = safe_file(root, row_key, forbidden)
    observed_row_bytes = row_path.stat().st_size
    observed_row_hash = sha256_file(row_path)
    add_check(checks, "AUDIT_PHASE12_OPAQUE_BYTES", observed_row_bytes == int(row_spec.get("observed_remote_bytes", -1)), observed_row_bytes)
    add_check(checks, "AUDIT_PHASE12_OPAQUE_SHA256", observed_row_hash == expected_row_hash, observed_row_hash)
    require_checks(checks, "independent Phase12 opaque-byte identity")

    row_count, column_count, row_groups, column_names, physical_names, arrow_types, physical_types = parquet_metadata(row_path)
    expected_columns = list(row_spec.get("expected_columns", []))
    expected_row_groups = row_spec.get("observed_row_groups")
    add_check(checks, "AUDIT_PHASE12_ROWS", row_count == int(row_spec.get("expected_rows", -1)), row_count)
    add_check(checks, "AUDIT_PHASE12_COLUMNS", column_count == len(expected_columns) == 24, column_count)
    add_check(checks, "AUDIT_PHASE12_SCHEMA", column_names == expected_columns, column_names)
    add_check(checks, "AUDIT_PHASE12_PHYSICAL_SCHEMA", physical_names == expected_columns, physical_names)
    add_check(checks, "AUDIT_PHASE12_SCHEMA_UNAMBIGUOUS", len(column_names) == len(set(column_names)), column_names)
    add_check(checks, "AUDIT_PHASE12_ROW_GROUP_LOCK_PRESENT", isinstance(expected_row_groups, int) and expected_row_groups > 0, expected_row_groups)
    if isinstance(expected_row_groups, int):
        add_check(checks, "AUDIT_PHASE12_ROW_GROUPS", row_groups == expected_row_groups, row_groups)
    add_check(checks, "AUDIT_PHASE12_ARROW_TYPES", arrow_types == row_spec.get("observed_arrow_types"), arrow_types)
    add_check(checks, "AUDIT_PHASE12_PHYSICAL_TYPES", physical_types == row_spec.get("observed_physical_types"), physical_types)
    require_checks(checks, "independent Phase12 metadata identity")
    receipt_row = receipt.get("phase12_rowlevel", {})
    add_check(checks, "AUDIT_REMOTE_RECEIPT_PHASE12", receipt_row.get("path") == row_key and receipt_row.get("bytes") == observed_row_bytes and receipt_row.get("sha256") == observed_row_hash and receipt_row.get("rows") == row_count and receipt_row.get("columns") == column_count and receipt_row.get("row_groups") == row_groups and receipt_row.get("arrow_types") == arrow_types and receipt_row.get("physical_types") == physical_types, receipt_row)
    require_checks(checks, "independent remote receipt Phase12 binding")
    paths: dict[str, Path] = {row_key: row_path, receipt_relative: receipt_path}
    identity_rows: list[dict[str, Any]] = [{
        "role": "PHASE12_ROWLEVEL_TRUTH_BEARING_INPUT", "path": row_key,
        "bytes": observed_row_bytes, "sha256": observed_row_hash,
        "status": "PASS_HASHED_BEFORE_DATAFRAME_LOAD",
    }, {
        "role": "REMOTE_IDENTITY_PREFLIGHT_RECEIPT", "path": receipt_relative,
        "bytes": receipt_path.stat().st_size, "sha256": sha256_file(receipt_path),
        "status": "PASS_BINDS_FINAL_LOCK_BEFORE_DATAFRAME_LOAD",
    }]
    for item in contract.get("locked_inputs", []):
        relative = str(item["path"])
        path = safe_file(root, relative, forbidden)
        observed_bytes, observed_hash = path.stat().st_size, sha256_file(path)
        add_check(checks, f"AUDIT_INPUT_BYTES::{relative}", observed_bytes == int(item["bytes"]), observed_bytes)
        add_check(checks, f"AUDIT_INPUT_SHA256::{relative}", observed_hash == item["sha256"], observed_hash)
        if path.suffix.lower() == ".parquet" and "rows" in item:
            rows, _, _, names, physical_names, _, _ = parquet_metadata(path)
            add_check(checks, f"AUDIT_INPUT_ROWS::{relative}", rows == int(item["rows"]), rows)
            required_columns = list(item.get("required_columns", []))
            add_check(checks, f"AUDIT_INPUT_REQUIRED_SCHEMA::{relative}", bool(required_columns) and len(names) == len(set(names)) and all(name in names for name in required_columns), {"required": required_columns, "observed": names})
            add_check(checks, f"AUDIT_INPUT_REQUIRED_PHYSICAL_SCHEMA::{relative}", bool(required_columns) and all(name in physical_names for name in required_columns), {"required": required_columns, "observed": physical_names})
        paths[relative] = path
        identity_rows.append({"role": item["role"], "path": relative, "bytes": observed_bytes, "sha256": observed_hash, "status": "PASS_LOCKED_INPUT"})
    auditor_path: Path | None = None
    for item in contract.get("implementation_identities", []):
        relative = str(item["path"])
        if not isinstance(item.get("bytes"), int) or not is_sha256(item.get("sha256")):
            raise RuntimeError(f"implementation identity is not finalized: {relative}")
        path = safe_file(root, relative, forbidden)
        observed_bytes, observed_hash = path.stat().st_size, sha256_file(path)
        add_check(checks, f"AUDIT_IMPLEMENTATION_BYTES::{relative}", observed_bytes == item["bytes"], observed_bytes)
        add_check(checks, f"AUDIT_IMPLEMENTATION_SHA256::{relative}", observed_hash == item["sha256"], observed_hash)
        identity_rows.append({"role": "IMPLEMENTATION", "path": relative, "bytes": observed_bytes, "sha256": observed_hash, "status": "PASS_IMPLEMENTATION_IDENTITY"})
        if relative.endswith("candidate_bottleneck_validate_query_states.py"):
            auditor_path = path
    if auditor_path is None:
        raise RuntimeError("auditor implementation is absent from the finalized contract")
    observed_locked = []
    for item in contract.get("locked_inputs", []):
        path = paths[str(item["path"])]
        metadata = parquet_metadata(path) if path.suffix.lower() == ".parquet" else None
        observed_locked.append({
            "role": item["role"], "path": str(item["path"]), "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "metadata_rows": metadata[0] if metadata else None,
            "metadata_columns": metadata[1] if metadata else None,
            "metadata_row_groups": metadata[2] if metadata else None,
        })
    observed_impl = []
    for item in contract.get("implementation_identities", []):
        path = safe_file(root, str(item["path"]), forbidden)
        observed_impl.append({"path": str(item["path"]), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    add_check(checks, "AUDIT_REMOTE_RECEIPT_LOCKED_INPUTS", receipt.get("locked_inputs") == observed_locked, {"receipt": len(receipt.get("locked_inputs", [])), "observed": len(observed_locked)})
    add_check(checks, "AUDIT_REMOTE_RECEIPT_IMPLEMENTATIONS", receipt.get("formal_implementation_identities") == observed_impl, {"receipt": len(receipt.get("formal_implementation_identities", [])), "observed": len(observed_impl)})
    source_check = independent_source_check(auditor_path)
    add_check(checks, "AUDIT_INDEPENDENT_IMPORTS", not source_check["forbidden_imports"] and not source_check["dangerous_calls"], source_check)
    require_checks(checks, "independent identity admission")

    configured_dir = safe_analysis_dir(root, str(contract.get("expected_output_directory", "")))
    supplied_dir = args.analysis_dir if args.analysis_dir.is_absolute() else root / args.analysis_dir
    supplied_dir = supplied_dir.resolve(strict=True)
    if supplied_dir != configured_dir:
        raise RuntimeError("analysis directory does not exactly match finalized contract")
    observed_names = sorted(path.name for path in configured_dir.iterdir())
    add_check(checks, "AUDIT_PRODUCER_FILESET_EXACT", observed_names == sorted(PRODUCER_FILES), observed_names)
    for name in PRODUCER_FILES:
        candidate = configured_dir / name
        add_check(checks, f"AUDIT_PRODUCER_REGULAR::{name}", candidate.is_file() and not candidate.is_symlink(), name)
    add_check(checks, "AUDIT_TARGETS_ABSENT", all(not (configured_dir / name).exists() for name in AUDIT_FILES), list(AUDIT_FILES))
    require_checks(checks, "independent output admission")

    producer_summary = json.loads((configured_dir / PRODUCER_FILES[7]).read_text(encoding="utf-8"))
    producer_gate = json.loads((configured_dir / PRODUCER_FILES[9]).read_text(encoding="utf-8"))
    expected_producer_status = "PASS_PHASE433_PRODUCER_FROZEN_PHASE12_ROWLEVEL_BOTTLENECK_CLOSURE_DESCRIPTIVE_NOT_CONFIRMATORY_NOT_SUBMISSION_READY"
    add_check(checks, "AUDIT_PRODUCER_STATUS", producer_summary.get("status") == expected_producer_status, producer_summary.get("status"))
    add_check(checks, "AUDIT_PRODUCER_GATE", producer_gate.get("status") == "PASS_PRODUCER_NOT_INDEPENDENT_AUTHORITY", producer_gate.get("status"))
    add_check(checks, "AUDIT_CONTRACT_HASH_RECORDED", producer_summary.get("contract_sha256") == producer_gate.get("contract_sha256") == contract_hash, contract_hash)
    safe_flags = ("confirmatory_endpoint_created", "model_changed", "threshold_changed", "raw_data_modified", "manuscript_modified", "phase99_read_or_used", "project_complete", "submission_ready", "submission_authorized")
    add_check(checks, "AUDIT_PRODUCER_GOVERNANCE", all(producer_summary.get(flag) is False for flag in safe_flags), {flag: producer_summary.get(flag) for flag in safe_flags})
    recorded_summary = producer_gate.get("summary", {})
    recorded_report = producer_gate.get("report", {})
    add_check(checks, "AUDIT_PRODUCER_GATE_SUMMARY_IDENTITY", recorded_summary.get("bytes") == (configured_dir / PRODUCER_FILES[7]).stat().st_size and recorded_summary.get("sha256") == sha256_file(configured_dir / PRODUCER_FILES[7]), recorded_summary)
    add_check(checks, "AUDIT_PRODUCER_GATE_REPORT_IDENTITY", recorded_report.get("bytes") == (configured_dir / PRODUCER_FILES[8]).stat().st_size and recorded_report.get("sha256") == sha256_file(configured_dir / PRODUCER_FILES[8]), recorded_report)
    producer_checks = pd.read_csv(configured_dir / PRODUCER_FILES[6], sep="\t")
    add_check(checks, "AUDIT_PRODUCER_CHECKS_PASS", len(producer_checks) > 0 and producer_checks["status"].eq("PASS").all(), producer_checks["status"].value_counts().to_dict())
    producer_identities = pd.read_csv(configured_dir / PRODUCER_FILES[0], sep="\t")
    expected_identity = {(row["path"], str(row["sha256"]), int(row["bytes"])) for row in identity_rows}
    observed_identity = {(str(row.path), str(row.sha256), int(row.bytes)) for row in producer_identities.itertuples(index=False)}
    add_check(checks, "AUDIT_PRODUCER_INPUT_IDENTITIES", observed_identity == expected_identity, {"expected": len(expected_identity), "observed": len(observed_identity)})

    ledger = construct_ledger(paths, row_key, checks)
    exact, compatible = construct_partitions(ledger)
    reconcile_prior(paths, exact, compatible, checks)
    bootstrap_rows: list[dict[str, Any]] = []
    for level in LEVELS:
        bootstrap_rows.extend(independent_bootstrap(
            ledger.loc[ledger["annotation_level"].eq(level)], level,
            int(contract["cluster_bootstrap"]["replicates"]), int(contract["seed"]),
        ))
    bootstrap = pd.DataFrame(bootstrap_rows)
    strata = construct_strata(ledger, contract)

    saved_ledger = pd.read_parquet(configured_dir / PRODUCER_FILES[1])
    saved_exact = pd.read_csv(configured_dir / PRODUCER_FILES[2], sep="\t")
    saved_compatible = pd.read_csv(configured_dir / PRODUCER_FILES[3], sep="\t")
    saved_bootstrap = pd.read_csv(configured_dir / PRODUCER_FILES[4], sep="\t")
    saved_strata = pd.read_csv(configured_dir / PRODUCER_FILES[5], sep="\t")
    for check_id, expected, observed in (
        ("AUDIT_LEDGER_REPRODUCED", ledger, saved_ledger),
        ("AUDIT_EXACT_PARTITION_REPRODUCED", exact, saved_exact),
        ("AUDIT_COMPATIBLE_PARTITION_REPRODUCED", compatible, saved_compatible),
        ("AUDIT_CLUSTER_BOOTSTRAP_REPRODUCED", bootstrap, saved_bootstrap),
        ("AUDIT_EXPLORATORY_STRATA_REPRODUCED", strata, saved_strata),
    ):
        passed, detail = frames_equal(expected, observed)
        add_check(checks, check_id, passed, detail)
    add_check(checks, "AUDIT_SIX_STATE_EXACT_CLOSURE", all(int(exact.loc[exact["annotation_level"].eq(level), "queries"].sum()) == 27639 for level in LEVELS), exact.groupby("annotation_level")["queries"].sum().to_dict())
    add_check(checks, "AUDIT_STRATA_NONCONFIRMATORY", strata["status"].eq("NONCONFIRMATORY_EXPLORATORY_POINT_ESTIMATE_ONLY").all(), strata["status"].value_counts().to_dict())
    strata_denominators = strata.loc[strata["metric"].eq("COVERAGE")].groupby(["annotation_level", "stratum_dimension"], sort=False)["queries"].sum()
    add_check(checks, "AUDIT_STRATA_PARTITIONS_CLOSE", len(strata_denominators) == len(LEVELS) * 3 and strata_denominators.eq(27639).all(), strata_denominators.to_dict())
    add_check(checks, "AUDIT_NO_PHASE99_INPUT", all("phase99" not in str(path).lower() for path in paths.values()), [str(path) for path in paths.values()])
    require_checks(checks, "Phase433 independent recomputation")

    for name in PRODUCER_FILES:
        path = configured_dir / name
        identity_rows.append({
            "role": "PRODUCER_OUTPUT", "path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "status": "PASS_INDEPENDENTLY_HASHED",
        })
    check_path, identity_path, summary_path, report_path, checkpoint_path = [configured_dir / name for name in AUDIT_FILES]
    write_tsv_exclusive(pd.DataFrame(checks), check_path)
    write_tsv_exclusive(pd.DataFrame(identity_rows), identity_path)
    summary = {
        "format": "siteguard.phase433.independent-frozen-phase12-rowlevel-bottleneck-audit.v1",
        "phase": 433,
        "status": "PASS_PHASE433_INDEPENDENT_AUDIT_DESCRIPTIVE_NOT_CONFIRMATORY_NOT_SUBMISSION_READY",
        "contract_sha256": contract_hash,
        "phase12_rowlevel_sha256": observed_row_hash,
        "queries": 27639,
        "levels": list(LEVELS),
        "ledger_rows": len(ledger),
        "checks_total": len(checks),
        "checks_failed": 0,
        "independent_recomputation": True,
        "producer_imported_or_executed": False,
        "confirmatory_endpoint_created": False,
        "model_changed": False,
        "threshold_changed": False,
        "raw_data_modified": False,
        "manuscript_modified": False,
        "phase99_read_or_used": False,
        "project_complete": False,
        "submission_ready": False,
        "submission_authorized": False,
        "audit_tables": {
            check_path.name: {"bytes": check_path.stat().st_size, "sha256": sha256_file(check_path)},
            identity_path.name: {"bytes": identity_path.stat().st_size, "sha256": sha256_file(identity_path)},
        },
    }
    write_json_exclusive(summary, summary_path)
    report = "\n".join([
        "# Phase433 independent audit",
        "",
        f"Status: **{summary['status']}**",
        "",
        "The standalone auditor reproduced the frozen Phase12 six-state exact partition, the Phase227-compatible view, all 10,000 cluster-bootstrap replicates per annotation level, and every pre-fixed exploratory stratum.",
        "",
        "This closes a descriptive bookkeeping ambiguity only. It creates no confirmatory endpoint, biochemical truth claim, model or threshold change, manuscript update, release decision, or submission authority.",
        "",
        "No Phase99 artifact was read or used.",
        "",
    ])
    with report_path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(report)
    checkpoint = {
        "phase": 433,
        "status": "PASS_FROZEN_PHASE12_ROWLEVEL_BOTTLENECK_CLOSURE_DESCRIPTIVE_NOT_CONFIRMATORY_NOT_PROJECT_COMPLETE",
        "contract_sha256": contract_hash,
        "phase12_rowlevel_sha256": observed_row_hash,
        "independent_summary": {"bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)},
        "independent_report": {"bytes": report_path.stat().st_size, "sha256": sha256_file(report_path)},
        "independent_check_matrix": {"bytes": check_path.stat().st_size, "sha256": sha256_file(check_path)},
        "independent_identity_manifest": {"bytes": identity_path.stat().st_size, "sha256": sha256_file(identity_path)},
        "checkpoint_created_last": True,
        "confirmatory_endpoint_created": False,
        "model_changed": False,
        "threshold_changed": False,
        "raw_data_modified": False,
        "manuscript_modified": False,
        "phase99_read_or_used": False,
        "project_complete": False,
        "submission_ready": False,
        "submission_authorized": False,
    }
    write_json_exclusive(checkpoint, checkpoint_path)
    print(json.dumps({
        "status": checkpoint["status"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
