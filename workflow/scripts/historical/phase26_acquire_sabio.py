#!/usr/bin/env python3
"""Acquire and pre-filter the frozen SABIO-RK external cohort.

Raw SABIO pages are license-restricted and are not copied into public software
artifacts.  This stage does not read any Phase 26 model prediction.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
PROTOCOL = ROOT / "PROJECT/Phase26_SABIO_external_blind_protocol.md"
DATA = ROOT / "data/external/sabio_rk"
RAW = DATA / "raw_restricted"
RESULTS = ROOT / "results/phase26"
REPORTS = ROOT / "reports"
CHECKPOINTS = ROOT / "checkpoints"
OPENAPI_URL = "https://sabiork.h-its.org/openapi/export-api.json"
EXPORT_URL = "https://sabiork.h-its.org/export-api/sabio/kinlaw-entry/csv"
QUERY = 'EnzymeType:"wildtype" AND UniProtKB_AC:* AND ECNumber:* AND PubMedID:*'
PAGE_SIZE = 1000
USER_AGENT = "SiteGuard-academic-external-validation/4.0 (non-commercial research; contact via project manifest)"
FULL_EC = re.compile(r"^[1-9]\d*\.(?:\d+|-)\.(?:\d+|-)\.(?:\d+|-)$")
ACCESSION = re.compile(r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_with_retry(session: requests.Session, url: str, **kwargs: object) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = session.get(url, timeout=(20, 180), **kwargs)
            if response.status_code == 429:
                time.sleep(65)
                continue
            response.raise_for_status()
            return response
        except (requests.RequestException, OSError) as error:
            last_error = error
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Request failed after retries: {url}") from last_error


def split_accessions(value: object) -> list[str]:
    text = "" if value is None else str(value).upper()
    return sorted(set(ACCESSION.findall(text)))


def main() -> None:
    for path in (DATA, RAW, RESULTS, REPORTS, CHECKPOINTS):
        path.mkdir(parents=True, exist_ok=True)
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    forbidden = [RESULTS / "external_predictions.parquet", RESULTS / "external_operating_points.tsv"]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("Phase 26 prediction outputs already exist; acquisition must remain prediction-blind")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/csv,application/json"})
    openapi = request_with_retry(session, OPENAPI_URL)
    openapi_path = RAW / "export_api_openapi.json.gz"
    with gzip.open(openapi_path, "wb") as handle:
        handle.write(openapi.content)
    openapi_json = openapi.json()

    first = request_with_retry(
        session,
        EXPORT_URL,
        params={"q": QUERY, "page": 1, "pageSize": PAGE_SIZE},
    )
    total_count = int(first.headers["X-Total-Count"])
    total_pages = int(first.headers["X-Total-Pages"])
    page_size_returned = int(first.headers["X-Page-Size"])
    if page_size_returned != PAGE_SIZE:
        raise RuntimeError(f"SABIO page-size mismatch: requested={PAGE_SIZE};returned={page_size_returned}")

    frames = []
    page_manifest = []
    for page in range(1, total_pages + 1):
        target = RAW / f"kinlaw_wildtype_uniprot_ec_pubmed__page_{page:04d}.csv.gz"
        if target.is_file():
            with gzip.open(target, "rt", encoding="utf-8") as handle:
                text = handle.read()
            source = "RESUMED_EXISTING_PAGE"
            response_headers = {}
        else:
            response = first if page == 1 else request_with_retry(
                session,
                EXPORT_URL,
                params={"q": QUERY, "page": page, "pageSize": PAGE_SIZE},
            )
            text = response.text
            with gzip.open(target, "wt", encoding="utf-8", newline="") as handle:
                handle.write(text)
            source = "DOWNLOADED"
            response_headers = {
                "date": response.headers.get("Date"),
                "content_disposition": response.headers.get("Content-Disposition"),
                "x_page": response.headers.get("X-Page"),
                "x_total_count": response.headers.get("X-Total-Count"),
                "x_total_pages": response.headers.get("X-Total-Pages"),
            }
            if page > 1:
                time.sleep(1.05)
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        frame["source_page"] = page
        frames.append(frame)
        page_manifest.append(
            {
                "page": page,
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
                "csv_rows": len(frame),
                "source": source,
                **response_headers,
            }
        )

    raw = pd.concat(frames, ignore_index=True)
    required = {"EntryID", "ECNumber", "EnzymeType", "UniprotIDs", "PubMedID", "Journal", "Reaction", "Organism"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"SABIO export missing columns: {missing}")

    raw["EnzymeType_norm"] = raw["EnzymeType"].str.strip().str.lower()
    raw["ECNumber_norm"] = raw["ECNumber"].str.strip()
    parameter_rows = len(raw)
    entries_all = raw["EntryID"].nunique()
    core = raw.loc[
        raw["EnzymeType_norm"].eq("wildtype")
        & raw["ECNumber_norm"].map(lambda value: bool(FULL_EC.match(value)))
        & raw["PubMedID"].str.strip().ne("")
        & raw["Journal"].str.strip().ne("")
    ].copy()
    entry_columns = [
        "EntryID", "ECNumber_norm", "UniprotIDs", "PubMedID", "Journal", "Year", "Organism",
        "NCBITaxonomyID", "Reaction", "EnzymeName", "EnzymeType", "IsRecombinant", "Specification",
    ]
    entries = core[entry_columns].drop_duplicates()
    expanded_rows = []
    for row in entries.itertuples(index=False):
        for accession in split_accessions(row.UniprotIDs):
            expanded_rows.append(
                {
                    "sabio_entry_id": str(row.EntryID),
                    "uniprot_accession": accession,
                    "ec_l4": str(row.ECNumber_norm),
                    "ec_l3": ".".join(str(row.ECNumber_norm).split(".")[:3]),
                    "pubmed_id": str(row.PubMedID),
                    "journal": str(row.Journal),
                    "year": str(row.Year),
                    "organism": str(row.Organism),
                    "taxonomy_id": str(row.NCBITaxonomyID),
                    "reaction_text": str(row.Reaction),
                    "enzyme_name": str(row.EnzymeName),
                    "enzyme_type": str(row.EnzymeType),
                    "is_recombinant": str(row.IsRecombinant),
                    "specification": str(row.Specification),
                }
            )
    expanded = pd.DataFrame(expanded_rows)
    if expanded.empty:
        raise RuntimeError("No SABIO wild-type UniProt/complete-EC records survived parsing")

    protein = pd.read_parquet(
        ROOT / "data/processed/protein_table.parquet",
        columns=["protein_id", "uniprot_accession"],
    )
    known_accessions = set(protein["protein_id"].astype(str)) | set(protein["uniprot_accession"].astype(str))
    reference = pd.read_parquet(
        ROOT / "data/reference/activity_reference_library.parquet",
        columns=["reference_protein_id"],
    )
    known_accessions |= set(reference["reference_protein_id"].astype(str))
    expanded["accession_in_frozen_core"] = expanded["uniprot_accession"].isin(known_accessions)

    accession_summary = []
    for accession, group in expanded.groupby("uniprot_accession", observed=True):
        ec3 = sorted(group["ec_l3"].unique().tolist())
        ec4 = sorted(group["ec_l4"].unique().tolist())
        accession_summary.append(
            {
                "uniprot_accession": accession,
                "ec_l3": ec3[0] if len(ec3) == 1 else None,
                "ec_l4": ec4[0] if len(ec4) == 1 else None,
                "unique_ec_l3_count": len(ec3),
                "unique_ec_l4_count": len(ec4),
                "ec_l3_set_json": json.dumps(ec3),
                "ec_l4_set_json": json.dumps(ec4),
                "sabio_entry_count": group["sabio_entry_id"].nunique(),
                "pubmed_count": group["pubmed_id"].nunique(),
                "pubmed_ids_json": json.dumps(sorted(group["pubmed_id"].unique().tolist())),
                "organisms_json": json.dumps(sorted(group["organism"].unique().tolist())),
                "taxonomy_ids_json": json.dumps(sorted(group["taxonomy_id"].unique().tolist())),
                "accession_in_frozen_core": bool(group["accession_in_frozen_core"].any()),
                "ec_l3_label_eligible": len(ec3) == 1,
                "ec_l4_label_eligible": len(ec4) == 1,
                "sabio_license_scope": "NON_COMMERCIAL_ACADEMIC_ONLY",
                "prediction_use_policy": "EXTERNAL_BLIND_EVALUATION_ONLY",
            }
        )
    cohort = pd.DataFrame(accession_summary)
    cohort["presequence_candidate"] = cohort["ec_l3_label_eligible"] & ~cohort["accession_in_frozen_core"]
    cohort.to_parquet(RESULTS / "sabio_candidate_cohort.parquet", index=False, compression="zstd")
    expanded.to_parquet(RESULTS / "sabio_entry_accession_records.parquet", index=False, compression="zstd")

    manifest = pd.DataFrame(page_manifest)
    manifest.to_csv(DATA / "manifest.tsv", sep="\t", index=False)
    manifest[["path", "bytes", "sha256"]].to_csv(DATA / "raw_sha256.tsv", sep="\t", index=False)
    acquisition = {
        "phase": "26A0",
        "stage": "sabio_rk_acquisition_and_presequence_filter",
        "status": "PASS",
        "acquired_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "openapi_title": openapi_json.get("info", {}).get("title"),
        "openapi_version": openapi_json.get("info", {}).get("version"),
        "query": QUERY,
        "page_size": PAGE_SIZE,
        "total_entries_reported": total_count,
        "total_pages": total_pages,
        "parameter_rows": parameter_rows,
        "unique_entries_exported": int(entries_all),
        "eligible_entry_accession_rows": int(len(expanded)),
        "unique_accessions": int(len(cohort)),
        "accessions_in_frozen_core": int(cohort["accession_in_frozen_core"].sum()),
        "presequence_candidates": int(cohort["presequence_candidate"].sum()),
        "single_ec_l3_candidates": int((cohort["presequence_candidate"] & cohort["ec_l3_label_eligible"]).sum()),
        "single_ec_l4_candidates": int((cohort["presequence_candidate"] & cohort["ec_l4_label_eligible"]).sum()),
        "raw_data_redistribution": "PROHIBITED_BY_PROJECT_POLICY; SABIO non-commercial terms retained",
        "phase26_predictions_read": False,
    }
    (REPORTS / "phase26_sabio_acquisition_summary.json").write_text(json.dumps(acquisition, indent=2) + "\n", encoding="utf-8")
    checks = [
        ("protocol_present", PROTOCOL.is_file(), str(PROTOCOL)),
        ("official_openapi_retrieved", openapi.status_code == 200, OPENAPI_URL),
        ("all_pages_retrieved", len(manifest) == total_pages, f"pages={len(manifest)};expected={total_pages}"),
        ("page_sha256_complete", manifest["sha256"].str.len().eq(64).all(), len(manifest)),
        ("required_columns_present", not missing, missing),
        ("only_wildtype_after_filter", expanded["enzyme_type"].str.lower().eq("wildtype").all(), sorted(expanded["enzyme_type"].unique())),
        ("complete_ec_only", expanded["ec_l4"].map(lambda value: bool(FULL_EC.match(str(value)))).all(), len(expanded)),
        ("no_phase26_predictions_read", not any(path.exists() for path in forbidden), [str(path) for path in forbidden]),
        ("raw_redistribution_block_recorded", cohort["sabio_license_scope"].eq("NON_COMMERCIAL_ACADEMIC_ONLY").all(), len(cohort)),
    ]
    qc = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    qc.to_csv(REPORTS / "phase26_sabio_acquisition_qc.tsv", sep="\t", index=False)
    failures = qc.loc[~qc["passed"].astype(bool), "check"].tolist()
    if failures:
        acquisition["status"] = "FAIL"
        acquisition["failures"] = failures
        (REPORTS / "phase26_sabio_acquisition_summary.json").write_text(json.dumps(acquisition, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Phase 26 SABIO acquisition failed: {failures}")
    (CHECKPOINTS / "CHECKPOINT_26A0_SABIO_ACQUISITION_PASS").write_text(json.dumps(acquisition, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acquisition, indent=2))


if __name__ == "__main__":
    main()
