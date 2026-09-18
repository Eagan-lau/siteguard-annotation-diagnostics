#!/usr/bin/env python3
"""Download a bounded AlphaFold API supplement and selected PAE collection."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"


def read_ids(path: Path) -> set[str]:
    """Read one or more UniProt accessions per line without treating groups as IDs."""
    accessions: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        for token in re.split(r"[,;\s]+", line.strip().upper()):
            if not token:
                continue
            if not re.fullmatch(r"[A-Z0-9]+(?:-\d+)?", token):
                raise ValueError(f"invalid UniProt accession token {token!r} in {path}")
            accessions.add(token)
    return accessions


def client() -> requests.Session:
    retry = Retry(
        total=12,
        connect=12,
        read=12,
        status=12,
        backoff_factor=1.0,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    value = requests.Session()
    value.headers["User-Agent"] = "SiteGuard/3.0 academic selected AlphaFold snapshot"
    value.mount("https://", HTTPAdapter(max_retries=retry))
    return value


def atomic_download(session: requests.Session, url: str, path: Path, kind: str) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        original = path.read_bytes()
        content = normalize_content(original)
        validate_content(content, kind)
        if content != original:
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(content)
            os.replace(temporary, path)
        return len(content), hashlib.sha256(content).hexdigest()
    response = session.get(url, timeout=(30, 600))
    response.raise_for_status()
    content = normalize_content(response.content)
    validate_content(content, kind)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)
    return len(content), hashlib.sha256(content).hexdigest()


def normalize_content(content: bytes) -> bytes:
    """Normalize files that AlphaFold serves gzip-compressed under plain suffixes."""
    if content.startswith(b"\x1f\x8b"):
        return gzip.decompress(content)
    return content


def validate_content(content: bytes, kind: str) -> None:
    if len(content) < 100:
        raise ValueError(f"{kind} response is too small")
    if content.lstrip().lower().startswith((b"<html", b"<!doctype html")):
        raise ValueError(f"{kind} response is HTML")
    if kind == "pae":
        json.loads(content)
    elif kind == "cif" and b"data_" not in content[:4096]:
        raise ValueError("CIF response lacks a data_ block")


def url_name(url: str, accession: str, kind: str, index: int) -> str:
    name = Path(urlparse(url).path).name
    if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        suffix = ".json" if kind == "pae" else ".cif"
        name = f"AF-{accession}-F{index}{suffix}"
    return name


def process(accession: str, need_cif: bool, need_pae: bool, cif_dir: Path, pae_dir: Path) -> dict[str, Any]:
    session = client()
    api_url = API.format(accession=accession)
    response = session.get(api_url, timeout=(30, 300))
    if response.status_code == 404:
        return {"accession": accession, "api_url": api_url, "api_status": 404, "raw": response.text, "files": [], "status": "UNAVAILABLE", "error": ""}
    response.raise_for_status()
    payload = response.json()
    records = payload if isinstance(payload, list) else [payload]
    files: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        for kind, enabled, key, destination_dir in (
            ("cif", need_cif, "cifUrl", cif_dir),
            ("pae", need_pae, "paeDocUrl", pae_dir),
        ):
            url = record.get(key)
            if not enabled or not url:
                continue
            destination = destination_dir / url_name(str(url), accession, kind, index)
            try:
                size, sha256 = atomic_download(session, str(url), destination, kind)
                files.append({"kind": kind, "url": url, "path": str(destination), "size": size, "sha256": sha256, "status": "PASS", "error": ""})
            except Exception as exc:  # noqa: BLE001 - preserve complete per-file failures
                files.append({"kind": kind, "url": url, "path": str(destination), "size": 0, "sha256": "", "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
    expected = {kind for kind, enabled in (("cif", need_cif), ("pae", need_pae)) if enabled}
    present = {row["kind"] for row in files if row["status"] == "PASS"}
    status = "PASS" if expected.issubset(present) and all(row["status"] == "PASS" for row in files) else "FAIL"
    return {"accession": accession, "api_url": api_url, "api_status": response.status_code, "raw": response.text, "files": files, "status": status, "error": "" if status == "PASS" else "missing or failed requested file"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--pae", type=Path, required=True)
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--cif-dir", type=Path, required=True)
    parser.add_argument("--pae-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unavailable", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    supplement = read_ids(args.supplement)
    pae = read_ids(args.pae)
    accessions = sorted(supplement | pae)
    if not accessions:
        raise SystemExit("no selected AlphaFold accessions")
    args.raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_accession = {
            executor.submit(process, accession, accession in supplement, accession in pae, args.cif_dir, args.pae_dir): accession
            for accession in accessions
        }
        for completed, future in enumerate(as_completed(future_to_accession), start=1):
            accession = future_to_accession[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {"accession": accession, "api_url": API.format(accession=accession), "api_status": "", "raw": "", "files": [], "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            if completed % 100 == 0:
                print(json.dumps({"completed": completed, "total": len(accessions), "unavailable": sum(row["status"] == "UNAVAILABLE" for row in results), "failed": sum(row["status"] == "FAIL" for row in results)}))
            time.sleep(0.02)
    results.sort(key=lambda row: row["accession"])
    raw_temporary = args.raw_jsonl.with_suffix(args.raw_jsonl.suffix + ".tmp")
    with gzip.open(raw_temporary, "wt", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps({"accession": row["accession"], "api_url": row["api_url"], "api_status": row["api_status"], "raw_response": row["raw"]}, ensure_ascii=False) + "\n")
    os.replace(raw_temporary, args.raw_jsonl)
    manifest_rows: list[dict[str, Any]] = []
    for result in results:
        if result["files"]:
            for item in result["files"]:
                manifest_rows.append({"accession": result["accession"], "api_status": str(result["api_status"]), **item})
        else:
            manifest_rows.append({"accession": result["accession"], "api_status": str(result["api_status"]), "kind": "", "url": "", "path": "", "size": 0, "sha256": "", "status": result["status"], "error": result["error"]})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_parquet(args.manifest, index=False)
    pd.DataFrame(manifest_rows).to_csv(args.manifest.with_suffix(".tsv"), sep="\t", index=False)
    unavailable = [row for row in results if row["status"] == "UNAVAILABLE"]
    with args.unavailable.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["accession", "api_url", "api_status", "reason"], delimiter="\t")
        writer.writeheader()
        writer.writerows({"accession": row["accession"], "api_url": row["api_url"], "api_status": row["api_status"], "reason": "no AlphaFold prediction returned"} for row in unavailable)
    failed = [row for row in results if row["status"] == "FAIL"]
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "requested_accessions": len(accessions),
        "supplement_accessions": len(supplement),
        "pae_accessions": len(pae),
        "api_available": sum(row["status"] != "UNAVAILABLE" for row in results),
        "api_unavailable": len(unavailable),
        "downloaded_cif_files": sum(item["kind"] == "cif" and item["status"] == "PASS" for row in results for item in row["files"]),
        "downloaded_pae_files": sum(item["kind"] == "pae" and item["status"] == "PASS" for row in results for item in row["files"]),
        "failed_accessions": len(failed),
        "status": "PASS" if not failed else "FAIL",
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
