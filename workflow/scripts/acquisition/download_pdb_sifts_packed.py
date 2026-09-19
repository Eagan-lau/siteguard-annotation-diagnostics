#!/usr/bin/env python3
"""Download selected PDB mmCIF and SIFTS XML directly into inode-safe tar shards."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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
    value.headers["User-Agent"] = "SiteGuard/3.0 academic selected PDB/SIFTS snapshot"
    value.mount("https://", HTTPAdapter(max_retries=retry))
    return value


def validate_gzip(content: bytes) -> None:
    if len(content) < 100 or content.lstrip().lower().startswith((b"<html", b"<!doctype html")):
        raise ValueError("response is too small or HTML")
    with gzip.GzipFile(fileobj=io.BytesIO(content), mode="rb") as handle:
        while handle.read(1024 * 1024):
            pass


def resources(pdb_id: str) -> list[tuple[str, str, str]]:
    middle = pdb_id[1:3]
    return [
        ("pdb_mmcif", f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz", f"pdb/archive_mmcif/{pdb_id}.cif.gz"),
        ("sifts_xml", f"https://ftp.ebi.ac.uk/pub/databases/msd/sifts/split_xml/{middle}/{pdb_id}.xml.gz", f"sifts/selected_xml/{middle}/{pdb_id}.xml.gz"),
    ]


def download_resource(session: requests.Session, pdb_id: str, kind: str, url: str, member: str) -> tuple[dict[str, Any], bytes | None]:
    response = session.get(url, timeout=(30, 600))
    if response.status_code == 404:
        return ({"pdb_id": pdb_id, "kind": kind, "url": url, "archive_member": member, "http_status": 404, "size": 0, "sha256": "", "status": "UNAVAILABLE_AT_SOURCE", "error": "verified HTTP 404"}, None)
    response.raise_for_status()
    content = response.content
    validate_gzip(content)
    return ({"pdb_id": pdb_id, "kind": kind, "url": url, "archive_member": member, "http_status": response.status_code, "size": len(content), "sha256": hashlib.sha256(content).hexdigest(), "status": "PASS", "error": ""}, content)


def process_shard(index: int, ids: list[str], output_dir: Path) -> dict[str, Any]:
    archive_path = output_dir / f"selected_pdb_sifts_{index:04d}.tar"
    manifest_path = output_dir / f"selected_pdb_sifts_{index:04d}.manifest.jsonl.gz"
    if archive_path.is_file() and manifest_path.is_file():
        return {"shard": index, "pdb_ids": len(ids), "archive": str(archive_path), "manifest": str(manifest_path), "status": "REUSED"}
    temporary_archive = archive_path.with_suffix(".tar.partial")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".partial")
    rows: list[dict[str, Any]] = []
    session = client()
    failed = False
    with tarfile.open(temporary_archive, mode="w") as archive:
        for pdb_id in ids:
            for kind, url, member_name in resources(pdb_id):
                try:
                    row, content = download_resource(session, pdb_id, kind, url, member_name)
                except Exception as exc:  # noqa: BLE001 - every selected resource is accounted for
                    failed = True
                    row = {"pdb_id": pdb_id, "kind": kind, "url": url, "archive_member": member_name, "http_status": "", "size": 0, "sha256": "", "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
                    content = None
                rows.append(row)
                if content is not None:
                    info = tarfile.TarInfo(member_name)
                    info.size = len(content)
                    info.mtime = 0
                    info.mode = 0o444
                    archive.addfile(info, io.BytesIO(content))
    with gzip.open(temporary_manifest, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    if failed:
        temporary_archive.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        return {"shard": index, "pdb_ids": len(ids), "archive": str(archive_path), "manifest": str(manifest_path), "status": "FAIL"}
    os.replace(temporary_archive, archive_path)
    os.replace(temporary_manifest, manifest_path)
    return {"shard": index, "pdb_ids": len(ids), "archive": str(archive_path), "manifest": str(manifest_path), "status": "PASS"}


def read_shard_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    ids = sorted({line.strip().lower() for line in args.ids.read_text(encoding="utf-8").splitlines() if line.strip()})
    if not ids or len(ids) > 250_000:
        raise SystemExit(f"invalid selected PDB count: {len(ids)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards = [ids[index : index + args.shard_size] for index in range(0, len(ids), args.shard_size)]
    statuses: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_shard, index, shard, args.output_dir): index for index, shard in enumerate(shards)}
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {"shard": futures[future], "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
            statuses.append(result)
            print(json.dumps({"completed_shards": completed, "total_shards": len(shards), "last": result}))
    failed_shards = [row for row in statuses if row["status"] == "FAIL"]
    rows: list[dict[str, Any]] = []
    if not failed_shards:
        for path in sorted(args.output_dir.glob("selected_pdb_sifts_*.manifest.jsonl.gz")):
            rows.extend(read_shard_rows(path))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_parquet(args.manifest, index=False)
        frame.to_csv(args.manifest.with_suffix(".tsv"), sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    counts = Counter(row["status"] for row in rows)
    kind_counts = Counter((row["kind"], row["status"]) for row in rows)
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selected_pdb_ids": len(ids),
        "shards": len(shards),
        "shard_size": args.shard_size,
        "failed_shards": len(failed_shards),
        "resource_status_counts": dict(counts),
        "kind_status_counts": {f"{kind}:{status}": count for (kind, status), count in sorted(kind_counts.items())},
        "storage_mode": "uncompressed tar shards containing already-gzipped source files",
        "status": "PASS" if not failed_shards and rows else "FAIL",
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
