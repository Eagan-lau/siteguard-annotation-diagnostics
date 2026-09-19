#!/usr/bin/env python3
"""Fetch label-blind UniProt sequences for the frozen Phase 26 SABIO cohort.

Only sequence and administrative fields are requested.  EC, Rhea, catalytic
activity, active-site, domain, and protein-name fields are deliberately absent.
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
RESULTS = ROOT / "results/phase26"
DATA = ROOT / "data/external/uniprot_phase26_sequences"
RAW = DATA / "raw_sequence_only"
REPORTS = ROOT / "reports"
CHECKPOINTS = ROOT / "checkpoints"
ENDPOINT = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = [
    "accession",
    "id",
    "reviewed",
    "fragment",
    "organism_id",
    "length",
    "sequence",
    "sequence_version",
    "protein_existence",
]
FORBIDDEN_FIELD_TOKENS = {
    "ec", "rhea", "catalytic", "function", "active_site", "binding", "domain", "protein_name"
}
AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZUO]+$")
BATCH_SIZE = 40
USER_AGENT = "SiteGuard-academic-external-validation/4.0 (sequence-only external validation)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_with_retry(session: requests.Session, params: dict[str, object]) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(7):
        try:
            response = session.get(ENDPOINT, params=params, timeout=(20, 180))
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(min(65 if response.status_code == 429 else 2**attempt, 65))
                continue
            if 400 <= response.status_code < 500:
                raise RuntimeError(
                    f"UniProt rejected sequence-only request: status={response.status_code};body={response.text[:1000]}"
                )
            response.raise_for_status()
            return response
        except (requests.RequestException, OSError) as error:
            last_error = error
            time.sleep(min(2**attempt, 30))
    raise RuntimeError("UniProt sequence-only request failed after retries") from last_error


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in frame[["uniprot_accession", "sequence"]].itertuples(index=False):
            handle.write(f">{row.uniprot_accession}\n")
            sequence = str(row.sequence)
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start:start + 80] + "\n")
    temporary.replace(path)


def main() -> None:
    for directory in (RESULTS, DATA, RAW, REPORTS, CHECKPOINTS):
        directory.mkdir(parents=True, exist_ok=True)
    acquisition_checkpoint = CHECKPOINTS / "CHECKPOINT_26A0_SABIO_ACQUISITION_PASS"
    if not acquisition_checkpoint.is_file():
        raise FileNotFoundError(acquisition_checkpoint)
    forbidden_predictions = [
        RESULTS / "external_predictions.parquet",
        RESULTS / "external_operating_points.tsv",
    ]
    if any(path.exists() for path in forbidden_predictions):
        raise RuntimeError("Phase 26 prediction outputs exist; sequence acquisition must remain prediction-blind")
    if any(any(token in field.lower() for token in FORBIDDEN_FIELD_TOKENS) for field in FIELDS):
        raise RuntimeError(f"Forbidden functional field requested: {FIELDS}")

    candidate = pd.read_parquet(RESULTS / "sabio_candidate_cohort.parquet")
    accessions = sorted(candidate.loc[candidate["presequence_candidate"], "uniprot_accession"].astype(str).unique())
    if not accessions:
        raise RuntimeError("No pre-sequence SABIO candidates")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/plain"})
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []
    releases: set[str] = set()
    release_dates: set[str] = set()
    for batch_index, start in enumerate(range(0, len(accessions), BATCH_SIZE), start=1):
        batch = accessions[start:start + BATCH_SIZE]
        query = "(" + " OR ".join(f"accession:{accession}" for accession in batch) + ")"
        params: dict[str, object] = {
            "query": query,
            "format": "tsv",
            "fields": ",".join(FIELDS),
            "size": 500,
        }
        response = request_with_retry(session, params)
        release = response.headers.get("X-UniProt-Release", "")
        release_date = response.headers.get("X-UniProt-Release-Date", "")
        releases.add(release)
        release_dates.add(release_date)
        target = RAW / f"uniprot_sequence_only_batch_{batch_index:04d}.tsv.gz"
        with gzip.open(target, "wt", encoding="utf-8", newline="") as handle:
            handle.write(response.text)
        frame = pd.read_csv(io.StringIO(response.text), sep="\t", dtype=str, keep_default_na=False)
        expected_headers = {
            "Entry", "Entry Name", "Reviewed", "Fragment", "Organism (ID)", "Length",
            "Sequence", "Sequence version", "Protein existence",
        }
        missing = sorted(expected_headers - set(frame.columns))
        if missing:
            raise RuntimeError(f"UniProt response missing sequence-only fields: {missing}")
        frame["request_batch"] = batch_index
        frames.append(frame)
        manifests.append({
            "batch": batch_index,
            "requested_accessions": len(batch),
            "returned_rows": len(frame),
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "http_status": response.status_code,
            "response_date": response.headers.get("Date", ""),
            "uniprot_release": release,
            "uniprot_release_date": release_date,
            "requested_fields": ",".join(FIELDS),
        })
        time.sleep(1.0)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={
        "Entry": "uniprot_accession",
        "Entry Name": "entry_name",
        "Reviewed": "reviewed_status",
        "Fragment": "fragment_status_raw",
        "Organism (ID)": "taxonomy_id",
        "Length": "length_api",
        "Sequence": "sequence",
        "Sequence version": "sequence_version",
        "Protein existence": "protein_existence",
    })
    if raw["uniprot_accession"].duplicated().any():
        raise RuntimeError("UniProt sequence-only response contains duplicate primary accessions")
    returned = set(raw["uniprot_accession"].astype(str))
    missing_accessions = sorted(set(accessions) - returned)

    cohort = candidate.merge(raw, on="uniprot_accession", how="left", validate="one_to_one")
    cohort["sequence_found"] = cohort["sequence"].notna() & cohort["sequence"].astype(str).ne("")
    cohort["length_api"] = pd.to_numeric(cohort["length_api"], errors="coerce")
    cohort["sequence_length_observed"] = cohort["sequence"].fillna("").astype(str).str.len()
    cohort["sequence_valid"] = cohort["sequence"].fillna("").astype(str).map(lambda value: bool(AA.fullmatch(value)))
    cohort["length_consistent"] = cohort["length_api"].eq(cohort["sequence_length_observed"])
    cohort["fragment_free"] = cohort["fragment_status_raw"].fillna("").astype(str).str.strip().eq("")
    cohort["length_in_scope"] = cohort["length_api"].between(50, 5000, inclusive="both")
    cohort["sequence_eligible"] = (
        cohort["presequence_candidate"]
        & cohort["sequence_found"]
        & cohort["sequence_valid"]
        & cohort["length_consistent"]
        & cohort["fragment_free"]
        & cohort["length_in_scope"]
    )
    cohort["sequence_exclusion_reason"] = "ELIGIBLE"
    reason_rules = [
        (~cohort["presequence_candidate"], "KNOWN_ACCESSION_OR_LABEL_INELIGIBLE"),
        (~cohort["sequence_found"], "UNIPROT_SEQUENCE_NOT_FOUND"),
        (cohort["sequence_found"] & ~cohort["sequence_valid"], "INVALID_AMINO_ACID_SEQUENCE"),
        (cohort["sequence_found"] & ~cohort["length_consistent"], "SEQUENCE_LENGTH_MISMATCH"),
        (cohort["sequence_found"] & ~cohort["fragment_free"], "UNIPROT_FRAGMENT"),
        (cohort["sequence_found"] & ~cohort["length_in_scope"], "LENGTH_OUT_OF_SCOPE"),
    ]
    for mask, reason in reason_rules:
        cohort.loc[~cohort["sequence_eligible"] & mask & cohort["sequence_exclusion_reason"].eq("ELIGIBLE"), "sequence_exclusion_reason"] = reason
    cohort.to_parquet(RESULTS / "sabio_sequence_cohort.parquet", index=False, compression="zstd")
    eligible = cohort.loc[cohort["sequence_eligible"]].sort_values("uniprot_accession")
    write_fasta(eligible, DATA / "sabio_sequence_candidates.fasta")

    manifest = pd.DataFrame(manifests)
    manifest.to_csv(DATA / "sequence_request_manifest.tsv", sep="\t", index=False)
    pd.DataFrame({"uniprot_accession": missing_accessions}).to_csv(
        DATA / "uniprot_missing_accessions.tsv", sep="\t", index=False
    )
    summary = {
        "phase": "26A1",
        "stage": "uniprot_sequence_only_acquisition",
        "status": "PASS",
        "uniprot_release": sorted(releases),
        "uniprot_release_date": sorted(release_dates),
        "requested_fields": FIELDS,
        "functional_fields_requested": False,
        "presequence_candidates": len(accessions),
        "sequence_records_returned": int(len(raw)),
        "missing_accessions": len(missing_accessions),
        "sequence_eligible": int(cohort["sequence_eligible"].sum()),
        "ec_l3_sequence_eligible": int((cohort["sequence_eligible"] & cohort["ec_l3_label_eligible"]).sum()),
        "ec_l4_sequence_eligible": int((cohort["sequence_eligible"] & cohort["ec_l4_label_eligible"]).sum()),
        "exclusion_counts": cohort["sequence_exclusion_reason"].value_counts().to_dict(),
        "phase26_predictions_read": False,
    }
    checks = [
        ("acquisition_checkpoint_present", acquisition_checkpoint.is_file(), str(acquisition_checkpoint)),
        ("no_functional_fields_requested", not summary["functional_fields_requested"], FIELDS),
        ("no_phase26_predictions_read", not any(path.exists() for path in forbidden_predictions), forbidden_predictions),
        ("single_uniprot_release", len(releases) == 1 and "" not in releases, sorted(releases)),
        ("single_uniprot_release_date", len(release_dates) == 1 and "" not in release_dates, sorted(release_dates)),
        ("unique_returned_accessions", not raw["uniprot_accession"].duplicated().any(), len(raw)),
        ("eligible_sequence_integrity", bool(eligible["sequence_valid"].all() and eligible["length_consistent"].all()), len(eligible)),
        ("eligible_nonempty", len(eligible) > 0, len(eligible)),
    ]
    qc = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    qc.to_csv(REPORTS / "phase26_uniprot_sequence_qc.tsv", sep="\t", index=False)
    failures = qc.loc[~qc["passed"].astype(bool), "check"].tolist()
    if failures:
        summary["status"] = "FAIL"
        summary["failures"] = failures
    (REPORTS / "phase26_uniprot_sequence_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"Phase 26 sequence acquisition QC failed: {failures}")
    (CHECKPOINTS / "CHECKPOINT_26A1_UNIPROT_SEQUENCE_PASS").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
