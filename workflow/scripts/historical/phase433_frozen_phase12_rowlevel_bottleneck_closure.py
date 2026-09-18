#!/usr/bin/env python3
"""Build the identity-locked Phase433 row-level Phase12 bottleneck repair.

The program has a write-free ``--preflight-only`` mode.  Both modes require a
finalized contract with a real SHA-256 for the Phase12 Parquet.  The Parquet is
hashed and its metadata is checked before any truth-bearing row is loaded.
"""

from __future__ import annotations

import argparse
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
    "scripts/phase433_frozen_phase12_rowlevel_bottleneck_closure.py",
    "scripts/phase433_independent_frozen_phase12_rowlevel_bottleneck_audit.py",
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


def stable_seed(level: str, base_seed: int) -> int:
    material = f"{base_seed}|PHASE433|{level}".encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


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
        detail = "\n".join(f"{x['check_id']}: {x['observed']}" for x in failures)
        raise RuntimeError(f"{context} failed closed:\n{detail}")


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def safe_file(root: Path, relative: str, forbidden: Iterable[str]) -> Path:
    normalized = relative.replace("\\", "/")
    if Path(relative).is_absolute() or ".." in Path(normalized).parts:
        raise RuntimeError(f"unsafe relative path: {relative}")
    lowered = normalized.lower()
    if any(token.lower() in lowered for token in forbidden):
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


def safe_output(root: Path, relative: str) -> Path:
    normalized = relative.replace("\\", "/")
    if Path(relative).is_absolute() or ".." in Path(normalized).parts:
        raise RuntimeError(f"unsafe output path: {relative}")
    if "phase99" in normalized.lower():
        raise RuntimeError("Phase99 output path is forbidden")
    output = (root / Path(normalized)).resolve(strict=False)
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("output escapes V4 root") from exc
    return output


def parquet_metadata(path: Path) -> tuple[int, int, int, list[str], list[str], list[str], list[str]]:
    metadata = pq.ParquetFile(path)
    return (
        int(metadata.metadata.num_rows),
        int(metadata.metadata.num_columns),
        int(metadata.metadata.num_row_groups),
        list(metadata.schema_arrow.names),
        list(metadata.schema.names),
        [str(field.type) for field in metadata.schema_arrow],
        [str(metadata.schema.column(index).physical_type) for index in range(metadata.metadata.num_columns)],
    )


def preflight(root: Path, contract_path: Path, output_override: Path | None) -> tuple[dict[str, Any], dict[str, Path], list[dict[str, Any]], Path]:
    checks: list[dict[str, Any]] = []
    expected_contract = (root / FORMAL_RELATIVE).resolve(strict=True)
    if contract_path != expected_contract:
        raise RuntimeError(f"formal contract must be exactly {FORMAL_RELATIVE}")
    if contract_path.is_symlink() or not contract_path.is_file():
        raise RuntimeError("formal contract is missing, non-regular, or a symlink")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    add_check(checks, "CONTRACT_FORMAT", contract.get("format") == CONTRACT_FORMAT, contract.get("format"))
    add_check(checks, "CONTRACT_FORMAL_STATUS", contract.get("status") == FORMAL_STATUS, contract.get("status"))
    add_check(checks, "PRIMARY_SEED", contract.get("seed") == 20260819, contract.get("seed"))
    add_check(checks, "LEVELS_EXACT", tuple(contract.get("levels", [])) == LEVELS, contract.get("levels"))
    add_check(checks, "SIX_STATE_ORDER", tuple(contract.get("six_state_order", [])) == SIX_STATES, contract.get("six_state_order"))
    add_check(checks, "COMPATIBLE_STATE_ORDER", tuple(contract.get("phase227_compatible_order", [])) == COMPATIBLE_STATES, contract.get("phase227_compatible_order"))
    runtime = contract.get("runtime", {})
    add_check(checks, "PYTHON_VERSION", runtime.get("expected_version") == ".".join(map(str, sys.version_info[:3])) == "3.11.15", ".".join(map(str, sys.version_info[:3])))
    add_check(checks, "PYTHON_ISOLATED_NO_BYTECODE", runtime.get("invocation_flags") == ["-I", "-B"] and sys.flags.isolated == 1 and sys.dont_write_bytecode, {"isolated": sys.flags.isolated, "dont_write_bytecode": sys.dont_write_bytecode})
    expected_python = Path(str(runtime.get("python_path", ""))).resolve(strict=False)
    add_check(checks, "PYTHON_PATH", Path(sys.executable).resolve(strict=True) == expected_python, str(Path(sys.executable).resolve(strict=True)))
    governance = contract.get("governance", {})
    add_check(checks, "GOVERNANCE_ALL_FALSE", bool(governance) and all(value is False for value in governance.values()), governance)
    bootstrap = contract.get("cluster_bootstrap", {})
    add_check(checks, "BOOTSTRAP_REPLICATES", bootstrap.get("replicates") == 10000, bootstrap.get("replicates"))
    add_check(checks, "BOOTSTRAP_UNIT", bootstrap.get("unit") == "cluster_id_30", bootstrap.get("unit"))
    add_check(checks, "BOOTSTRAP_PERCENTILES", bootstrap.get("percentiles") == [0.025, 0.975], bootstrap.get("percentiles"))
    forbidden = tuple(contract.get("forbidden_path_tokens", ()))
    add_check(checks, "FORBIDDEN_PHASE99", any(str(x).lower() == "phase99" for x in forbidden), forbidden)
    add_check(checks, "LOGICAL_ROOT_EQUIVALENT", Path(str(contract.get("logical_v4_root", ""))).resolve(strict=True) == root, contract.get("logical_v4_root"))
    add_check(checks, "OUTPUT_DIRECTORY_EXACT", contract.get("expected_output_directory") == EXPECTED_OUTPUT_RELATIVE, contract.get("expected_output_directory"))
    add_check(checks, "LOCKED_INPUT_PATHS_EXACT", tuple(item.get("path") for item in contract.get("locked_inputs", [])) == LOCKED_INPUT_PATHS, [item.get("path") for item in contract.get("locked_inputs", [])])
    add_check(checks, "IMPLEMENTATION_PATHS_EXACT", tuple(item.get("path") for item in contract.get("implementation_identities", [])) == IMPLEMENTATION_PATHS, [item.get("path") for item in contract.get("implementation_identities", [])])
    add_check(checks, "STRATA_CATEGORIES_EXACT", contract.get("exploratory_strata", {}).get("pfam_mapping") == EXPECTED_STRATA["pfam_mapping"] and contract.get("exploratory_strata", {}).get("taxonomy_domain") == EXPECTED_STRATA["taxonomy_domain"] and contract.get("exploratory_strata", {}).get("structure_availability") == EXPECTED_STRATA["structure_availability"], contract.get("exploratory_strata"))
    require_checks(checks, "contract admission")

    finalization = contract.get("remote_identity_finalization", {})
    receipt_relative = str(finalization.get("receipt_path", ""))
    add_check(checks, "REMOTE_IDENTITY_RECEIPT_PATH_EXACT", receipt_relative == RECEIPT_RELATIVE, receipt_relative)
    require_checks(checks, "remote identity receipt path admission")
    receipt_path = safe_file(root, receipt_relative, forbidden)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    contract_hash = sha256_file(contract_path)
    add_check(checks, "REMOTE_IDENTITY_RECEIPT_REQUIRED", finalization.get("receipt_required") is True, finalization)
    add_check(checks, "REMOTE_IDENTITY_RECEIPT_STATUS", receipt.get("status") == "PASS_PHASE433_REMOTE_IDENTITY_LOCK_FINALIZED_NO_TRUTH_ROWS_OPENED", receipt.get("status"))
    add_check(checks, "REMOTE_IDENTITY_RECEIPT_BINDS_CONTRACT", receipt.get("formal_lock_sha256") == contract_hash, receipt.get("formal_lock_sha256"))
    add_check(checks, "REMOTE_IDENTITY_RECEIPT_NO_TRUTH_ROWS", receipt.get("phase12_dataframe_rows_loaded") == 0, receipt.get("phase12_dataframe_rows_loaded"))
    add_check(checks, "REMOTE_IDENTITY_RECEIPT_NO_ANALYSIS_OUTPUT", receipt.get("analysis_output_directory_created") is False, receipt.get("analysis_output_directory_created"))
    require_checks(checks, "remote identity receipt admission")

    row_spec = contract.get("phase12_rowlevel_input", {})
    row_hash = row_spec.get("sha256")
    if not is_sha256(row_hash):
        raise RuntimeError("Phase12 row-level SHA-256 is absent or malformed; remote read-only hash lock is required before truth opening")
    row_path = safe_file(root, str(row_spec.get("path", "")), forbidden)
    row_bytes = row_path.stat().st_size
    row_observed_hash = sha256_file(row_path)
    add_check(checks, "PHASE12_ROWLEVEL_BYTES", row_bytes == int(row_spec.get("observed_remote_bytes", -1)), row_bytes)
    add_check(checks, "PHASE12_ROWLEVEL_SHA256", row_observed_hash == row_hash, row_observed_hash)
    require_checks(checks, "Phase12 opaque-byte identity")

    rows, columns, row_groups, names, physical_names, arrow_types, physical_types = parquet_metadata(row_path)
    expected_names = list(row_spec.get("expected_columns", []))
    expected_row_groups = row_spec.get("observed_row_groups")
    add_check(checks, "PHASE12_ROWLEVEL_ROWS", rows == int(row_spec.get("expected_rows", -1)), rows)
    add_check(checks, "PHASE12_ROWLEVEL_COLUMNS", columns == len(expected_names) == 24, columns)
    add_check(checks, "PHASE12_ROWLEVEL_SCHEMA_EXACT", names == expected_names, names)
    add_check(checks, "PHASE12_ROWLEVEL_PHYSICAL_SCHEMA_EXACT", physical_names == expected_names, physical_names)
    add_check(checks, "PHASE12_ROWLEVEL_SCHEMA_UNAMBIGUOUS", len(names) == len(set(names)), names)
    add_check(checks, "PHASE12_ROWLEVEL_ROW_GROUP_LOCK_PRESENT", isinstance(expected_row_groups, int) and expected_row_groups > 0, expected_row_groups)
    if isinstance(expected_row_groups, int):
        add_check(checks, "PHASE12_ROWLEVEL_ROW_GROUPS", row_groups == expected_row_groups, row_groups)
    add_check(checks, "PHASE12_ROWLEVEL_ARROW_TYPES", arrow_types == row_spec.get("observed_arrow_types"), arrow_types)
    add_check(checks, "PHASE12_ROWLEVEL_PHYSICAL_TYPES", physical_types == row_spec.get("observed_physical_types"), physical_types)
    require_checks(checks, "Phase12 metadata identity")

    receipt_row = receipt.get("phase12_rowlevel", {})
    add_check(checks, "REMOTE_IDENTITY_RECEIPT_BINDS_PHASE12", receipt_row.get("path") == str(row_spec.get("path")) and receipt_row.get("bytes") == row_bytes and receipt_row.get("sha256") == row_observed_hash and receipt_row.get("rows") == rows and receipt_row.get("columns") == columns and receipt_row.get("row_groups") == row_groups and receipt_row.get("arrow_types") == arrow_types and receipt_row.get("physical_types") == physical_types, receipt_row)
    require_checks(checks, "remote identity receipt Phase12 binding")
    paths: dict[str, Path] = {str(row_spec["path"]): row_path, receipt_relative: receipt_path}
    identity_rows = [{
        "role": "PHASE12_ROWLEVEL_TRUTH_BEARING_INPUT",
        "path": str(row_spec["path"]),
        "bytes": row_bytes,
        "rows": rows,
        "columns": columns,
        "row_groups": row_groups,
        "sha256": row_observed_hash,
        "status": "PASS_HASHED_BEFORE_DATAFRAME_LOAD",
    }, {
        "role": "REMOTE_IDENTITY_PREFLIGHT_RECEIPT",
        "path": receipt_relative,
        "bytes": receipt_path.stat().st_size,
        "rows": "NA",
        "columns": "NA",
        "row_groups": "NA",
        "sha256": sha256_file(receipt_path),
        "status": "PASS_BINDS_FINAL_LOCK_BEFORE_DATAFRAME_LOAD",
    }]
    for item in contract.get("locked_inputs", []):
        relative = str(item["path"])
        path = safe_file(root, relative, forbidden)
        observed_bytes = path.stat().st_size
        observed_hash = sha256_file(path)
        add_check(checks, f"INPUT_BYTES::{relative}", observed_bytes == int(item["bytes"]), observed_bytes)
        add_check(checks, f"INPUT_SHA256::{relative}", observed_hash == item["sha256"], observed_hash)
        paths[relative] = path
        meta_rows: int | str = "NA"
        meta_cols: int | str = "NA"
        meta_groups: int | str = "NA"
        if path.suffix.lower() == ".parquet":
            meta_rows, meta_cols, meta_groups, meta_names, physical_names, _, _ = parquet_metadata(path)
            if "rows" in item:
                add_check(checks, f"INPUT_ROWS::{relative}", meta_rows == int(item["rows"]), meta_rows)
            required_columns = list(item.get("required_columns", []))
            add_check(checks, f"INPUT_REQUIRED_SCHEMA::{relative}", bool(required_columns) and len(meta_names) == len(set(meta_names)) and all(name in meta_names for name in required_columns), {"required": required_columns, "observed": meta_names})
            add_check(checks, f"INPUT_REQUIRED_PHYSICAL_SCHEMA::{relative}", bool(required_columns) and all(name in physical_names for name in required_columns), {"required": required_columns, "observed": physical_names})
        identity_rows.append({
            "role": item["role"], "path": relative, "bytes": observed_bytes,
            "rows": meta_rows, "columns": meta_cols, "row_groups": meta_groups,
            "sha256": observed_hash, "status": "PASS_LOCKED_INPUT",
        })

    for item in contract.get("implementation_identities", []):
        relative = str(item["path"])
        if not is_sha256(item.get("sha256")) or not isinstance(item.get("bytes"), int):
            raise RuntimeError(f"implementation identity is not finalized: {relative}")
        path = safe_file(root, relative, forbidden)
        observed_bytes = path.stat().st_size
        observed_hash = sha256_file(path)
        add_check(checks, f"IMPLEMENTATION_BYTES::{relative}", observed_bytes == item["bytes"], observed_bytes)
        add_check(checks, f"IMPLEMENTATION_SHA256::{relative}", observed_hash == item["sha256"], observed_hash)
        identity_rows.append({
            "role": "IMPLEMENTATION", "path": relative, "bytes": observed_bytes,
            "rows": "NA", "columns": "NA", "row_groups": "NA",
            "sha256": observed_hash, "status": "PASS_IMPLEMENTATION_IDENTITY",
        })
    receipt_locked = receipt.get("locked_inputs", [])
    receipt_impl = receipt.get("formal_implementation_identities", [])
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
    add_check(checks, "REMOTE_IDENTITY_RECEIPT_BINDS_LOCKED_INPUTS", receipt_locked == observed_locked, {"receipt": len(receipt_locked) if isinstance(receipt_locked, list) else "INVALID", "observed": len(observed_locked)})
    add_check(checks, "REMOTE_IDENTITY_RECEIPT_BINDS_IMPLEMENTATIONS", receipt_impl == observed_impl, {"receipt": len(receipt_impl), "observed": len(observed_impl)})
    require_checks(checks, "identity preflight")

    configured_output = safe_output(root, str(contract.get("expected_output_directory", "")))
    if output_override is not None:
        override = output_override if output_override.is_absolute() else root / output_override
        override = override.resolve(strict=False)
        if override != configured_output:
            raise RuntimeError("output override does not exactly match the contract")
    if configured_output.exists() or configured_output.is_symlink():
        raise RuntimeError(f"Phase433 output directory already exists: {configured_output}")
    return contract, paths, checks + identity_rows, configured_output


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
    """Parse a persisted scalar boolean without Python string truthiness."""
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


def classify_states(available: pd.Series, correct: pd.Series, accepted: pd.Series) -> pd.Series:
    if (correct & ~available).any():
        raise RuntimeError("top_correct=true with oracle_candidate_available=false")
    states = np.select(
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
    )
    result = pd.Series(states, index=available.index)
    if result.eq("INVALID").any():
        raise RuntimeError("unclassified Phase433 state")
    return result


def metric_values(state_counts: np.ndarray) -> dict[str, float]:
    counts = np.asarray(state_counts, dtype=float)
    total = float(counts.sum())
    values = {state: counts[index] / total if total else math.nan for index, state in enumerate(SIX_STATES)}
    available = float(counts[2:].sum())
    accepted = float(counts[[1, 3, 5]].sum())
    values.update({
        "CANDIDATE_AVAILABLE": available / total if total else math.nan,
        "TOP1_CORRECT_GIVEN_CANDIDATE_AVAILABLE": float(counts[[4, 5]].sum()) / available if available else math.nan,
        "COVERAGE": accepted / total if total else math.nan,
        "ACCEPTED_PRECISION": float(counts[5]) / accepted if accepted else math.nan,
        "TOP1_CORRECT_REJECTED_FRACTION": float(counts[4]) / total if total else math.nan,
        "TOP1_WRONG_ACCEPTED_FRACTION": float(counts[[1, 3]].sum()) / total if total else math.nan,
    })
    return values


def bootstrap_level(level_frame: pd.DataFrame, level: str, replicates: int, base_seed: int) -> list[dict[str, Any]]:
    clusters = level_frame["cluster_id_30"].astype(str)
    cluster_names = sorted(clusters.unique())
    cluster_index = {name: index for index, name in enumerate(cluster_names)}
    state_index = {name: index for index, name in enumerate(SIX_STATES)}
    matrix = np.zeros((len(cluster_names), len(SIX_STATES)), dtype=np.int64)
    cluster_codes = clusters.map(cluster_index).to_numpy(dtype=np.int64)
    state_codes = level_frame["stage"].map(state_index).to_numpy(dtype=np.int64)
    np.add.at(matrix, (cluster_codes, state_codes), 1)
    point = metric_values(matrix.sum(axis=0))
    samples = np.empty((replicates, len(BOOTSTRAP_METRICS)), dtype=np.float64)
    seed = stable_seed(level, base_seed)
    rng = np.random.default_rng(seed)
    number_clusters = len(cluster_names)
    for draw in range(replicates):
        picked = rng.integers(0, number_clusters, size=number_clusters)
        weights = np.bincount(picked, minlength=number_clusters)
        counts = weights @ matrix
        values = metric_values(counts)
        samples[draw, :] = [values[name] for name in BOOTSTRAP_METRICS]
    low, high = np.quantile(samples, [0.025, 0.975], axis=0)
    return [{
        "annotation_level": level,
        "metric": metric,
        "estimate": point[metric],
        "ci_percentile_low": float(low[index]),
        "ci_percentile_high": float(high[index]),
        "bootstrap_unit": "cluster_id_30",
        "clusters": number_clusters,
        "queries": len(level_frame),
        "replicates": replicates,
        "context_seed": seed,
        "interpretation": "DESCRIPTIVE_CLUSTER_BOOTSTRAP_NOT_RELEASE_CERTIFICATION",
    } for index, metric in enumerate(BOOTSTRAP_METRICS)]


def build_strata(ledger: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
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
                state_counts = block["stage"].value_counts().reindex(SIX_STATES, fill_value=0).to_numpy(dtype=int)
                values = metric_values(state_counts)
                clusters = int(block["cluster_id_30"].nunique())
                for metric in BOOTSTRAP_METRICS:
                    rows.append({
                        "annotation_level": level,
                        "stratum_dimension": dimension,
                        "stratum": category,
                        "metric": metric,
                        "queries": len(block),
                        "clusters": clusters,
                        "estimate": values[metric],
                        "status": "NONCONFIRMATORY_EXPLORATORY_POINT_ESTIMATE_ONLY",
                    })
    return pd.DataFrame(rows)


def build_ledger(paths: dict[str, Path], row_path_key: str, checks: list[dict[str, Any]]) -> pd.DataFrame:
    end = pd.read_parquet(paths[row_path_key])
    add_check(checks, "ROWLEVEL_QUERY_UNIQUE", len(end) == 27639 and end["query_protein_id"].notna().all() and end["query_protein_id"].is_unique, len(end))
    sequence = pd.read_parquet(paths["data/splits/split_sequence.parquet"], columns=["protein_id", "cluster_id_30", "split"])
    family = pd.read_parquet(paths["data/splits/split_family.parquet"], columns=["protein_id", "primary_pfam", "primary_pfam_clan"])
    protein = pd.read_parquet(paths["data/processed/protein_table.parquet"], columns=[
        "protein_id", "lineage_json", "alphafold_availability", "pdb_availability",
    ])
    add_check(checks, "SEQUENCE_JOIN_KEY_UNIQUE", sequence["protein_id"].is_unique, len(sequence))
    add_check(checks, "FAMILY_JOIN_KEY_UNIQUE", family["protein_id"].is_unique, len(family))
    add_check(checks, "PROTEIN_JOIN_KEY_UNIQUE", protein["protein_id"].is_unique, len(protein))
    joined = end.merge(sequence, left_on="query_protein_id", right_on="protein_id", how="left", validate="one_to_one")
    joined = joined.drop(columns=["protein_id"])
    joined = joined.merge(family, left_on="query_protein_id", right_on="protein_id", how="left", validate="one_to_one").drop(columns=["protein_id"])
    joined = joined.merge(protein, left_on="query_protein_id", right_on="protein_id", how="left", validate="one_to_one").drop(columns=["protein_id"])
    required_join = ["cluster_id_30", "split", "lineage_json", "alphafold_availability", "pdb_availability"]
    add_check(checks, "JOIN_COMPLETE", not joined[required_join].isna().any().any(), joined[required_join].isna().sum().to_dict())
    add_check(checks, "ALL_QUERIES_FROZEN_TEST", joined["split"].eq("test").all(), joined["split"].value_counts(dropna=False).to_dict())
    joined["pfam_mapping_stratum"] = family_stratum(joined["primary_pfam"], joined["primary_pfam_clan"])
    joined["taxonomy_domain_stratum"] = taxonomy_stratum(joined["lineage_json"])
    joined["structure_availability_stratum"] = structure_stratum(joined["pdb_availability"], joined["alphafold_availability"])
    ledger_parts: list[pd.DataFrame] = []
    for ordinal, level in enumerate(LEVELS, start=1):
        available = truth_bool(joined[f"oracle_candidate_available_{level}"], f"oracle_candidate_available_{level}")
        correct = truth_bool(joined[f"top_correct_{level}"], f"top_correct_{level}")
        accepted = truth_bool(joined[f"accepted_{level}"], f"accepted_{level}")
        part = joined[[
            "query_protein_id", "cluster_id_30", "pfam_mapping_stratum",
            "taxonomy_domain_stratum", "structure_availability_stratum",
        ]].copy()
        part.insert(1, "annotation_level", level)
        part.insert(2, "level_order", ordinal)
        part["oracle_candidate_available"] = available.to_numpy()
        part["top_correct"] = correct.to_numpy()
        part["accepted"] = accepted.to_numpy()
        part["stage"] = classify_states(available, correct, accepted).to_numpy()
        ledger_parts.append(part)
    ledger = pd.concat(ledger_parts, ignore_index=True)
    ledger = ledger.sort_values(["level_order", "query_protein_id"], kind="stable").reset_index(drop=True)
    add_check(checks, "LEDGER_ROWS", len(ledger) == 27639 * 3, len(ledger))
    add_check(checks, "LEDGER_UNIQUE", not ledger.duplicated(["annotation_level", "query_protein_id"]).any(), len(ledger))
    return ledger


def partitions(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact_rows: list[dict[str, Any]] = []
    compatible_rows: list[dict[str, Any]] = []
    mapping = {
        "CANDIDATE_ABSENT_REJECTED": ("CANDIDATE_ABSENT_REJECTED",),
        "CANDIDATE_PRESENT_TOP1_WRONG_REJECTED": ("CANDIDATE_PRESENT_TOP1_WRONG_REJECTED",),
        "TOP1_CORRECT_REJECTED": ("TOP1_CORRECT_REJECTED",),
        "TOP1_WRONG_ACCEPTED_UNSUPPORTED_TRANSFER": (
            "CANDIDATE_ABSENT_ACCEPTED_WRONG", "CANDIDATE_PRESENT_TOP1_WRONG_ACCEPTED",
        ),
        "TOP1_CORRECT_ACCEPTED": ("TOP1_CORRECT_ACCEPTED",),
    }
    for level_order, level in enumerate(LEVELS, start=1):
        frame = ledger.loc[ledger["annotation_level"].eq(level)]
        counts = frame["stage"].value_counts().reindex(SIX_STATES, fill_value=0)
        n = len(frame)
        for state_order, state in enumerate(SIX_STATES, start=1):
            count = int(counts[state])
            exact_rows.append({
                "annotation_level": level, "level_order": level_order,
                "state": state, "state_order": state_order,
                "queries": count, "denominator": n, "fraction": count / n,
                "exclusive": True, "point_identified": True,
                "interpretation": "DOCUMENTED_ANNOTATION_CONCORDANCE_ONLY",
            })
        for state_order, state in enumerate(COMPATIBLE_STATES, start=1):
            count = int(sum(counts[x] for x in mapping[state]))
            compatible_rows.append({
                "annotation_level": level, "level_order": level_order,
                "component": state, "component_order": state_order,
                "queries": count, "denominator": n, "fraction": count / n,
                "exclusive_partition": True, "point_identified": True,
                "phase227_prior_status": "WAS_BOUNDED" if state in COMPATIBLE_STATES[:2] else "WAS_POINT_IDENTIFIED",
            })
    return pd.DataFrame(exact_rows), pd.DataFrame(compatible_rows)


def reconcile_aggregates(paths: dict[str, Path], exact: pd.DataFrame, compatible: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    summary = json.loads(paths["reports/phase12_summary.json"].read_text(encoding="utf-8"))
    qc = pd.read_csv(paths["reports/phase12_qc.tsv"], sep="\t")
    add_check(checks, "PHASE12_STATUS", summary.get("status") == "PASS" and str(summary.get("slurm_job_id")) == "1496373", summary)
    add_check(checks, "PHASE12_QC", qc["status"].eq("PASS").all(), qc["status"].value_counts().to_dict())
    prior = pd.read_csv(paths["reports/phase227_candidate_error_decomposition_v2_20260902/phase227_phase12_exclusive_partition.tsv"], sep="\t")
    current = compatible.set_index(["annotation_level", "component"])
    prior = prior.set_index(["annotation_level", "component"])
    exact_indexed = exact.set_index(["annotation_level", "state"])
    for level in LEVELS:
        block = current.loc[level]
        add_check(checks, f"EXACT_PARTITION_CLOSE::{level}", int(block["queries"].sum()) == 27639, int(block["queries"].sum()))
        for component in COMPATIBLE_STATES:
            value = int(current.loc[(level, component), "queries"])
            old = prior.loc[(level, component)]
            low = int(old["numerator_lower"])
            high = int(old["numerator_upper"])
            add_check(checks, f"PHASE227_BOUND_CONTAINS::{level}::{component}", low <= value <= high, {"value": value, "lower": low, "upper": high})
            if scalar_bool(old["point_identified"], f"phase227 point_identified {level} {component}"):
                add_check(checks, f"PHASE227_POINT_MATCH::{level}::{component}", value == low == high, {"value": value, "prior": low})
    metrics = pd.read_csv(paths["results/phase12/end_to_end_metrics.tsv"], sep="\t")
    per_level = metrics.loc[metrics["analysis"].eq("per_level_top_candidate")].set_index("annotation_level")
    add_check(checks, "PHASE12_METRIC_ROWS_EXACT", len(per_level) == len(LEVELS) and per_level.index.is_unique and set(per_level.index) == set(LEVELS), list(per_level.index))
    error = pd.read_csv(paths["results/phase12/error_decomposition.tsv"], sep="\t")
    error_indexed = error.set_index(["annotation_level", "error_component"])
    expected_error_components = {
        "NO_POSITIVE_CANDIDATE_RETRIEVAL_LIMIT",
        "POSITIVE_PRESENT_BUT_TOP1_WRONG_RANKING",
        "TOP1_CORRECT_BUT_ABSTAINED",
        "ACCEPTED_WRONG_OVERANNOTATION",
        "ACCEPTED_CORRECT",
    }
    add_check(checks, "PHASE12_ERROR_ROWS_EXACT", len(error) == len(LEVELS) * len(expected_error_components) and error_indexed.index.is_unique and set(error["annotation_level"]) == set(LEVELS) and set(error["error_component"]) == expected_error_components and error["evaluation_split"].eq("population_test").all(), {"rows": len(error), "levels": sorted(error["annotation_level"].unique()), "components": sorted(error["error_component"].unique())})
    for level in LEVELS:
        accepted = int(current.loc[(level, "TOP1_WRONG_ACCEPTED_UNSUPPORTED_TRANSFER"), "queries"] + current.loc[(level, "TOP1_CORRECT_ACCEPTED"), "queries"])
        correct = int(current.loc[(level, "TOP1_CORRECT_ACCEPTED"), "queries"])
        row = per_level.loc[level]
        add_check(checks, f"PHASE12_QUERY_DENOMINATOR::{level}", int(row["queries"]) == 27639, int(row["queries"]))
        add_check(checks, f"PHASE12_ACCEPTED_MATCH::{level}", accepted == int(row["accepted_queries"]), {"observed": accepted, "saved": int(row["accepted_queries"])})
        precision = correct / accepted if accepted else math.nan
        add_check(checks, f"PHASE12_PRECISION_MATCH::{level}", abs(precision - float(row["precision_among_accepted"])) < 1e-12, {"observed": precision, "saved": float(row["precision_among_accepted"])})
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
            add_check(checks, f"PHASE12_ERROR_COMPONENT_MATCH::{level}::{component}", observed == int(saved_error["queries"]) and abs(observed / 27639 - float(saved_error["fraction_of_queries"])) < 1e-12, {"observed": observed, "saved": int(saved_error["queries"])})
        candidate_available = sum(state[name] for name in SIX_STATES[2:])
        add_check(checks, f"PHASE12_ORACLE_COVERAGE_MATCH::{level}", abs(candidate_available / 27639 - float(row["oracle_candidate_coverage"])) < 1e-12, {"observed": candidate_available / 27639, "saved": float(row["oracle_candidate_coverage"])})


def report_text(summary: dict[str, Any], compatible: pd.DataFrame) -> str:
    table = compatible.pivot(index="component", columns="annotation_level", values="fraction")
    lines = [
        "# Phase433 frozen Phase12 row-level bottleneck closure",
        "",
        f"Status: **{summary['status']}**",
        "",
        "This is a deterministic descriptive repair of the frozen Phase12 row-level output. It changes no model, calibrator, threshold, raw data, manuscript, release or submission state.",
        "",
        "## Exact Phase227-compatible partition",
        "",
        "| Outcome | EC-L3 | EC-L4 | Exact Rhea |",
        "|---|---:|---:|---:|",
    ]
    for state in COMPATIBLE_STATES:
        lines.append(
            f"| {state.replace('_', ' ').title()} | {table.loc[state, 'EC_L3']:.2%} | "
            f"{table.loc[state, 'EC_L4']:.2%} | {table.loc[state, 'EXACT_RHEA']:.2%} |"
        )
    lines.extend([
        "",
        "Candidate absence means only that the documented positive label is absent from the frozen candidate set. It is not biochemical absence. Cluster-bootstrap intervals are descriptive sensitivity intervals and do not create a new 95% release certificate.",
        "",
        "Exploratory Pfam-mapping, taxonomy-domain and structure-availability strata were fixed before execution and cannot select a model, threshold or claim.",
        "",
        "No Phase99 artifact was read or used.",
    ])
    return "\n".join(lines) + "\n"


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    contract_path = args.contract if args.contract.is_absolute() else root / args.contract
    contract_path = contract_path.resolve(strict=True)
    contract, paths, mixed_checks, output = preflight(root, contract_path, args.output_dir)
    preflight_checks = [row for row in mixed_checks if "check_id" in row]
    identity_rows = [row for row in mixed_checks if "role" in row]
    if args.preflight_only:
        print(json.dumps({
            "status": "PASS_PHASE433_READ_ONLY_PREFLIGHT_HASH_LOCK",
            "checks": len(preflight_checks),
            "phase12_sha256": contract["phase12_rowlevel_input"]["sha256"],
            "phase12_rows_loaded": 0,
            "output_created": False,
        }, indent=2))
        return 0

    checks = list(preflight_checks)
    row_key = str(contract["phase12_rowlevel_input"]["path"])
    ledger = build_ledger(paths, row_key, checks)
    exact, compatible = partitions(ledger)
    for level in LEVELS:
        add_check(checks, f"SIX_STATE_CLOSE::{level}", int(exact.loc[exact["annotation_level"].eq(level), "queries"].sum()) == 27639, int(exact.loc[exact["annotation_level"].eq(level), "queries"].sum()))
    reconcile_aggregates(paths, exact, compatible, checks)
    bootstrap_rows: list[dict[str, Any]] = []
    for level in LEVELS:
        bootstrap_rows.extend(bootstrap_level(
            ledger.loc[ledger["annotation_level"].eq(level)], level,
            int(contract["cluster_bootstrap"]["replicates"]), int(contract["seed"]),
        ))
    bootstrap = pd.DataFrame(bootstrap_rows)
    strata = build_strata(ledger, contract)
    add_check(checks, "BOOTSTRAP_ROWS", len(bootstrap) == len(LEVELS) * len(BOOTSTRAP_METRICS), len(bootstrap))
    expected_strata = sum(len(contract["exploratory_strata"][name]) for name in ("pfam_mapping", "taxonomy_domain", "structure_availability"))
    add_check(checks, "STRATA_ROWS", len(strata) == len(LEVELS) * expected_strata * len(BOOTSTRAP_METRICS), len(strata))
    strata_denominators = strata.loc[strata["metric"].eq("COVERAGE")].groupby(["annotation_level", "stratum_dimension"], sort=False)["queries"].sum()
    add_check(checks, "STRATA_PARTITIONS_CLOSE", len(strata_denominators) == len(LEVELS) * 3 and strata_denominators.eq(27639).all(), strata_denominators.to_dict())
    add_check(checks, "NO_PHASE99_PATH", all("phase99" not in str(path).lower() for path in paths.values()), [str(path) for path in paths.values()])
    require_checks(checks, "Phase433 producer analysis")

    output.mkdir(parents=False, exist_ok=False)
    destinations = {name: output / name for name in PRODUCER_FILES}
    write_tsv(pd.DataFrame(identity_rows), destinations[PRODUCER_FILES[0]])
    ledger.to_parquet(destinations[PRODUCER_FILES[1]], index=False, compression="zstd")
    write_tsv(exact, destinations[PRODUCER_FILES[2]])
    write_tsv(compatible, destinations[PRODUCER_FILES[3]])
    write_tsv(bootstrap, destinations[PRODUCER_FILES[4]])
    write_tsv(strata, destinations[PRODUCER_FILES[5]])
    write_tsv(pd.DataFrame(checks), destinations[PRODUCER_FILES[6]])
    summary = {
        "format": "siteguard.phase433.frozen-phase12-rowlevel-bottleneck-summary.v1",
        "phase": 433,
        "status": "PASS_PHASE433_PRODUCER_FROZEN_PHASE12_ROWLEVEL_BOTTLENECK_CLOSURE_DESCRIPTIVE_NOT_CONFIRMATORY_NOT_SUBMISSION_READY",
        "contract_sha256": sha256_file(contract_path),
        "phase12_rowlevel_sha256": contract["phase12_rowlevel_input"]["sha256"],
        "queries": 27639,
        "levels": list(LEVELS),
        "ledger_rows": len(ledger),
        "cluster_bootstrap_replicates": int(contract["cluster_bootstrap"]["replicates"]),
        "exact_partition": {
            level: {
                row.state: {"queries": int(row.queries), "fraction": float(row.fraction)}
                for row in exact.loc[exact["annotation_level"].eq(level)].itertuples(index=False)
            } for level in LEVELS
        },
        "phase227_compatible_partition": {
            level: {
                row.component: {"queries": int(row.queries), "fraction": float(row.fraction)}
                for row in compatible.loc[compatible["annotation_level"].eq(level)].itertuples(index=False)
            } for level in LEVELS
        },
        "checks_total": len(checks),
        "checks_failed": 0,
        "scientific_interpretation": "DETERMINISTIC_DESCRIPTIVE_REPAIR_ONLY",
        "confirmatory_endpoint_created": False,
        "model_changed": False,
        "threshold_changed": False,
        "raw_data_modified": False,
        "manuscript_modified": False,
        "phase99_read_or_used": False,
        "project_complete": False,
        "submission_ready": False,
        "submission_authorized": False,
        "outputs": {},
    }
    for name in PRODUCER_FILES[:7]:
        path = destinations[name]
        summary["outputs"][name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    destinations[PRODUCER_FILES[7]].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = report_text(summary, compatible)
    destinations[PRODUCER_FILES[8]].write_text(report, encoding="utf-8")
    producer_gate = {
        "phase": 433,
        "status": "PASS_PRODUCER_NOT_INDEPENDENT_AUTHORITY",
        "contract_sha256": sha256_file(contract_path),
        "summary": {"bytes": destinations[PRODUCER_FILES[7]].stat().st_size, "sha256": sha256_file(destinations[PRODUCER_FILES[7]])},
        "report": {"bytes": destinations[PRODUCER_FILES[8]].stat().st_size, "sha256": sha256_file(destinations[PRODUCER_FILES[8]])},
        "independent_audit_required": True,
        "confirmatory_endpoint_created": False,
        "model_changed": False,
        "threshold_changed": False,
        "phase99_read_or_used": False,
        "project_complete": False,
        "submission_ready": False,
        "submission_authorized": False,
    }
    destinations[PRODUCER_FILES[9]].write_text(json.dumps(producer_gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"], "output_dir": str(output),
        "producer_gate_sha256": sha256_file(destinations[PRODUCER_FILES[9]]),
        "independent_audit_required": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
