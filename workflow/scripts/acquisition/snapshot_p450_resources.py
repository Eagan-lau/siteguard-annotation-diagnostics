#!/usr/bin/env python3
"""Snapshot public P450Rdb and PlantP450 resources with bounded crawling."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


USER_AGENT = "SiteGuard/3.0 academic P450 snapshot (rate-limited public pages)"


def session() -> requests.Session:
    retry = Retry(
        total=15,
        connect=15,
        read=15,
        status=15,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        respect_retry_after_header=True,
        allowed_methods=frozenset({"GET"}),
    )
    value = requests.Session()
    value.headers["User-Agent"] = USER_AGENT
    value.mount("https://", HTTPAdapter(max_retries=retry))
    return value


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def fetch(client: requests.Session, url: str, path: Path, delay: float = 1.0) -> dict[str, Any]:
    started = time.monotonic()
    response = client.get(url, timeout=(30, 180))
    response.raise_for_status()
    atomic_write(path, response.content)
    elapsed = time.monotonic() - started
    if elapsed < delay:
        time.sleep(delay - elapsed)
    return {
        "url": url,
        "resolved_url": response.url,
        "path": str(path),
        "status_code": response.status_code,
        "content_type": response.headers.get("Content-Type", ""),
        "etag": response.headers.get("ETag", ""),
        "last_modified": response.headers.get("Last-Modified", ""),
        "size": len(response.content),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "status": "PASS",
        "error": "",
    }


def validate_csv(path: Path) -> tuple[int, list[str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for row in reader if any(cell.strip() for cell in row))
    if not header or rows == 0:
        raise ValueError(f"empty CSV dataset: {path}")
    return rows, header


def snapshot_p450rdb(client: requests.Session, root: Path, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    base = "https://www.cellknowledge.com.cn/p450rdb_v2/"
    root.mkdir(parents=True, exist_ok=True)
    pages: dict[str, bytes] = {}
    for name in ("download.html", "help.html", "statistics.html"):
        metadata = fetch(client, urljoin(base, name), root / name)
        manifest.append({"database": "p450rdb_v2", **metadata})
        pages[name] = (root / name).read_bytes()
    soup = BeautifulSoup(pages["download.html"], "html.parser")
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        url = urljoin(base, anchor["href"])
        parsed = urlparse(url)
        if parsed.netloc == "www.cellknowledge.com.cn" and "/p450rdb_v2/download/" in parsed.path:
            links.append(url)
    links = list(dict.fromkeys(links))
    if len(links) != 5:
        raise RuntimeError(f"expected five P450Rdb downloads parsed from page, found {len(links)}")
    normalized = {
        "1. Reactions.CSV": "Reactions.csv",
        "2. P450s.CSV": "P450s.csv",
        "3. Compounds.CSV": "Compounds.csv",
        "Sequence.fasta": "Sequence.fasta",
        "4. Reaction_Cascades.CSV": "Reaction_Cascades.csv",
    }
    counts: dict[str, int] = {}
    columns: dict[str, list[str]] = {}
    for url in links:
        original_name = unquote(Path(urlparse(url).path).name)
        destination = root / original_name
        metadata = fetch(client, url, destination)
        manifest.append({"database": "p450rdb_v2", **metadata})
        normalized_name = normalized.get(original_name)
        if not normalized_name:
            raise RuntimeError(f"unrecognized P450Rdb resource: {original_name}")
        link = root / normalized_name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(destination.name)
        if destination.suffix.lower() == ".csv":
            row_count, header = validate_csv(destination)
            counts[normalized_name] = row_count
            columns[normalized_name] = header
        else:
            fasta_count = sum(1 for line in destination.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith(">"))
            if fasta_count == 0:
                raise ValueError("P450Rdb FASTA contains no records")
            counts[normalized_name] = fasta_count
    return {"download_links": links, "counts": counts, "columns": columns, "status": "PASS"}


def plant_rows(table_html: bytes) -> list[dict[str, str]]:
    soup = BeautifulSoup(table_html, "html.parser")
    rows: list[dict[str, str]] = []
    for row in soup.select("tbody tr[onclick]"):
        match = re.search(r"document\.location\s*=\s*['\"]([^'\"]+)['\"]", row.get("onclick", ""))
        cells = row.find_all("td")
        if not match or len(cells) < 6:
            continue
        rows.append(
            {
                "clan": cells[0].get_text(" ", strip=True),
                "family": cells[1].get_text(" ", strip=True),
                "cyp_name": cells[2].get_text(" ", strip=True),
                "other_names": cells[3].get_text(" ", strip=True),
                "species": cells[4].get_text(" ", strip=True),
                "taxa": cells[5].get_text(" ", strip=True),
                "detail_relative_url": match.group(1),
            }
        )
    return rows


def snapshot_plantp450(client: requests.Session, root: Path, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    base = "https://erda.dk/public/vgrid/PlantP450/"
    root.mkdir(parents=True, exist_ok=True)
    for remote, local in (("index.html", "index.html"), ("table.html", "table.html"), ("structure.html", "structure.html"), ("herbicide.html", "herbicide.html")):
        metadata = fetch(client, urljoin(base, remote), root / local)
        manifest.append({"database": "plantp450_erda", **metadata})
    table_content = (root / "table.html").read_bytes()
    rows = plant_rows(table_content)
    if not rows:
        raise RuntimeError("PlantP450 table contained no detail rows")
    detail_root = root / "entries"
    url_rows: list[dict[str, str]] = []
    failures = 0
    unavailable_source_details: list[dict[str, str | int]] = []
    for index, row in enumerate(rows, start=1):
        url = urljoin(base, row["detail_relative_url"])
        if urlparse(url).netloc != "erda.dk":
            raise RuntimeError(f"refusing off-domain PlantP450 detail URL: {url}")
        relative_name = Path(urlparse(url).path).name
        destination = detail_root / relative_name
        try:
            if destination.is_file() and destination.stat().st_size > 0:
                content = destination.read_bytes()
                metadata = {
                    "url": url,
                    "resolved_url": url,
                    "path": str(destination),
                    "status_code": 200,
                    "content_type": "text/html",
                    "etag": "",
                    "last_modified": "",
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "status": "PASS_REUSED",
                    "error": "",
                }
            else:
                metadata = fetch(client, url, destination)
            manifest.append({"database": "plantp450_erda", **metadata})
            row["detail_local_path"] = str(destination)
            row["detail_sha256"] = metadata["sha256"]
            row["detail_status"] = metadata["status"]
        except Exception as exc:  # noqa: BLE001 - complete failure inventory required
            failures += 1
            status_code = exc.response.status_code if isinstance(exc, requests.HTTPError) and exc.response is not None else ""
            row["detail_local_path"] = str(destination)
            row["detail_sha256"] = ""
            row["detail_status"] = "FAIL"
            manifest.append(
                {
                    "database": "plantp450_erda",
                    "url": url,
                    "resolved_url": "",
                    "path": str(destination),
                    "status_code": status_code,
                    "content_type": "",
                    "etag": "",
                    "last_modified": "",
                    "size": 0,
                    "sha256": "",
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            unavailable_source_details.append(
                {
                    "cyp_name": row["cyp_name"],
                    "url": url,
                    "status_code": status_code,
                    "source_evidence": "URL is linked directly by the preserved PlantP450 table snapshot",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        url_rows.append({"cyp_name": row["cyp_name"], "url": url, "local_path": row["detail_local_path"], "status": row["detail_status"]})
        if index % 100 == 0:
            print(json.dumps({"plant_details_processed": index, "total": len(rows), "failures": failures}))
    frame = pd.DataFrame(rows)
    staging = root.parents[2] / "staging" / "cyp450"
    staging.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(staging / "plantp450_table_catalog.parquet", index=False)
    with (root / "plantp450_entry_urls.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cyp_name", "url", "local_path", "status"], delimiter="\t")
        writer.writeheader()
        writer.writerows(url_rows)
    permanent_source_failures = bool(unavailable_source_details) and all(
        row["status_code"] == 404 for row in unavailable_source_details
    )
    status = "PASS" if failures == 0 else "PASS_WITH_SOURCE_LIMITATION" if permanent_source_failures else "FAIL"
    return {
        "table_entries": len(rows),
        "detail_pages_saved": len(rows) - failures,
        "detail_failures": failures,
        "unavailable_source_details": unavailable_source_details,
        "status": status,
    }


def write_manifest(root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fields = ["database", "url", "resolved_url", "path", "status_code", "content_type", "etag", "last_modified", "size", "sha256", "status", "error"]
    with (root / "p450_snapshot_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (root / "p450_snapshot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    raw = args.root / "data" / "raw" / "cyp450"
    manifests = args.root / "data" / "manifests"
    client = session()
    rows: list[dict[str, Any]] = []
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "snapshot_date": "2026-08-18",
        "p450rdb": snapshot_p450rdb(client, raw / "p450rdb_v2", rows),
        "plantp450": snapshot_plantp450(client, raw / "plantp450_erda", rows),
    }
    component_statuses = [summary[key]["status"] for key in ("p450rdb", "plantp450")]
    if all(status.startswith("PASS") for status in component_statuses):
        summary["status"] = "PASS_WITH_SOURCE_LIMITATION" if any(status != "PASS" for status in component_statuses) else "PASS"
    else:
        summary["status"] = "FAIL"
    write_manifest(manifests, rows, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
