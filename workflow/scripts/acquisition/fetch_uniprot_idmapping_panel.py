#!/usr/bin/env python3
"""Retrieve a bounded UniProt panel through the official ID-mapping API."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE = "https://rest.uniprot.org"
FIELDS = ",".join(
    [
        "accession", "id", "reviewed", "protein_name", "gene_names", "organism_name", "organism_id",
        "lineage", "lineage_ids", "length", "ec", "rhea", "cc_catalytic_activity", "ft_act_site",
        "ft_binding", "protein_existence", "annotation_score", "sequence_version", "version", "date_created",
        "date_modified", "date_sequence_modified", "xref_pfam", "xref_interpro", "xref_proteomes", "fragment", "sequence",
    ]
)


def client() -> requests.Session:
    retry = Retry(
        total=15,
        connect=15,
        read=15,
        status=15,
        backoff_factor=1.0,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
    )
    value = requests.Session()
    value.headers["User-Agent"] = "SiteGuard/3.0 academic bounded UniProt ID mapping"
    value.mount("https://", HTTPAdapter(max_retries=retry))
    return value


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def poll(session: requests.Session, job_id: str) -> dict[str, Any]:
    while True:
        response = session.get(f"{BASE}/idmapping/status/{job_id}", timeout=(30, 180))
        response.raise_for_status()
        payload = response.json()
        status = payload.get("jobStatus")
        if status in (None, "FINISHED"):
            return payload
        if status in ("FAILED", "ERROR"):
            raise RuntimeError(f"UniProt ID mapping job {job_id} ended as {status}: {payload}")
        time.sleep(float(response.headers.get("Retry-After", 3)))


def download_gzip(session: requests.Session, url: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    response = session.get(url, stream=True, timeout=(30, 900))
    response.raise_for_status()
    with gzip.open(temporary, "wb") as handle:
        for block in response.iter_content(chunk_size=1024 * 1024):
            if block:
                handle.write(block)
    os.replace(temporary, path)


def returned_inventory(path: Path) -> tuple[set[str], set[str], dict[str, str]]:
    """Return all response IDs, sequence-bearing IDs, and unavailable statuses.

    The UniProt ID-mapping TSV can contain a row for a deleted entry.  Those rows
    have an empty Sequence field (and commonly ``Protein names=deleted``), while
    the FASTA endpoint correctly omits them.  Treating every TSV row as a current
    sequence therefore creates a false FASTA shortfall.
    """
    returned: set[str] = set()
    sequence_ready: set[str] = set()
    unavailable: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        if not {"From", "Sequence"}.issubset(fields):
            raise ValueError(f"UniProt mapping TSV lacks From/Sequence columns: {path}")
        for row in reader:
            accession = row.get("From", "").upper()
            if not accession:
                continue
            returned.add(accession)
            if row.get("Sequence", "").strip():
                sequence_ready.add(accession)
            else:
                deleted = row.get("Protein names", "").strip().lower() == "deleted"
                unavailable[accession] = "DELETED_ENTRY_NO_SEQUENCE" if deleted else "RETURNED_WITHOUT_SEQUENCE"
    return returned, sequence_ready, unavailable


def combine_text_gzip(inputs: list[Path], output: Path, keep_one_header: bool) -> tuple[int, int]:
    temporary = output.with_suffix(output.suffix + ".tmp")
    lines = 0
    records = 0
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as target:
        header: str | None = None
        for path in inputs:
            with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
                for index, line in enumerate(source):
                    if keep_one_header and index == 0:
                        if header is None:
                            header = line
                            target.write(line)
                            lines += 1
                        elif line != header:
                            raise ValueError(f"inconsistent UniProt batch header in {path}")
                        continue
                    target.write(line)
                    lines += 1
                    if (keep_one_header and index > 0) or (not keep_one_header and line.startswith(">")):
                        records += 1
    os.replace(temporary, output)
    return lines, records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accessions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args()
    accessions = sorted({line.strip().upper() for line in args.accessions.read_text(encoding="utf-8").splitlines() if line.strip()})
    if not accessions or len(accessions) > 100_000:
        raise SystemExit(f"invalid bounded UniProt panel size: {len(accessions)}")
    raw_dir = args.output_dir / "raw_idmapping_batches"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = client()
    batches = chunks(accessions, args.batch_size)
    jobs: list[dict[str, Any]] = []
    unavailable_status: dict[str, str] = {}
    tsv_paths: list[Path] = []
    fasta_paths: list[Path] = []
    for index, batch in enumerate(batches, start=1):
        stem = f"batch_{index:04d}"
        request_path = raw_dir / f"{stem}.request.json"
        tsv_path = raw_dir / f"{stem}.metadata.tsv.gz"
        fasta_path = raw_dir / f"{stem}.fasta.gz"
        if tsv_path.is_file() and fasta_path.is_file() and request_path.is_file():
            request_record = json.loads(request_path.read_text(encoding="utf-8"))
            job_id = request_record["job_id"]
        else:
            response = session.post(
                f"{BASE}/idmapping/run",
                data={"from": "UniProtKB_AC-ID", "to": "UniProtKB", "ids": ",".join(batch)},
                timeout=(30, 300),
            )
            response.raise_for_status()
            job_id = response.json()["jobId"]
            request_record = {
                "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "job_id": job_id,
                "from": "UniProtKB_AC-ID",
                "to": "UniProtKB",
                "requested_accessions": batch,
            }
            request_path.write_text(json.dumps(request_record, indent=2) + "\n", encoding="utf-8")
            status_payload = poll(session, job_id)
            (raw_dir / f"{stem}.status.json").write_text(json.dumps(status_payload, indent=2) + "\n", encoding="utf-8")
            download_gzip(session, f"{BASE}/idmapping/uniprotkb/results/stream/{job_id}?format=tsv&fields={FIELDS}", tsv_path)
            download_gzip(session, f"{BASE}/idmapping/uniprotkb/results/stream/{job_id}?format=fasta", fasta_path)
        returned, sequence_ready, returned_unavailable = returned_inventory(tsv_path)
        not_mapped = set(batch) - returned
        unavailable_status.update({accession: "NOT_MAPPED_BY_UNIPROT_ID_MAPPING" for accession in not_mapped})
        unavailable_status.update(returned_unavailable)
        jobs.append({
            "batch": index,
            "job_id": job_id,
            "requested": len(batch),
            "metadata_rows": len(returned),
            "sequence_available": len(sequence_ready),
            "unavailable": len(not_mapped) + len(returned_unavailable),
        })
        tsv_paths.append(tsv_path)
        fasta_paths.append(fasta_path)
        print(json.dumps(jobs[-1]))

    metadata_path = args.output_dir / "trembl_audit_preliminary_metadata.tsv.gz"
    fasta_path = args.output_dir / "trembl_audit_preliminary.fasta.gz"
    _, metadata_records = combine_text_gzip(tsv_paths, metadata_path, keep_one_header=True)
    _, fasta_records = combine_text_gzip(fasta_paths, fasta_path, keep_one_header=False)
    with (args.output_dir / "trembl_audit_preliminary_unmapped.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["accession", "status"], delimiter="\t")
        writer.writeheader()
        writer.writerows(
            {"accession": accession, "status": unavailable_status[accession]}
            for accession in sorted(unavailable_status)
        )
    sequence_available = len(accessions) - len(unavailable_status)
    unavailable_counts: dict[str, int] = {}
    for status in unavailable_status.values():
        unavailable_counts[status] = unavailable_counts.get(status, 0) + 1
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "requested_accessions": len(accessions),
        "batches": jobs,
        "metadata_records": metadata_records,
        "fasta_records": fasta_records,
        "sequence_available_accessions": sequence_available,
        "unavailable_accessions": len(unavailable_status),
        "unavailable_status_counts": unavailable_counts,
        # Retained as a compatibility alias for the preliminary unavailable list.
        "unmapped_accessions": len(unavailable_status),
        "api": "UniProt official ID mapping from UniProtKB_AC-ID to UniProtKB",
        "binding_site_note": "current API exposes metal/cofactor ligand positions through ft_binding; obsolete ft_metal return field is not requested",
        "status": "PASS" if sequence_available + len(unavailable_status) == len(accessions) and fasta_records == sequence_available else "FAIL",
    }
    (args.output_dir / "uniprot_idmapping_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("requested_accessions", "metadata_records", "fasta_records", "unmapped_accessions", "status")}))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
