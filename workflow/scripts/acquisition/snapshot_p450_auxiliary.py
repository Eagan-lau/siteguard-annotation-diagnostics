#!/usr/bin/env python3
"""Snapshot public FunP450 assets and CYP nomenclature resources."""

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
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


USER_AGENT = "SiteGuard/3.0 academic P450 snapshot (rate-limited public pages)"


def build_session() -> requests.Session:
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


def safe_relative_path(url: str) -> Path:
    parsed = urlparse(url)
    raw = parsed.path.strip("/") or "index.html"
    if raw.endswith("/"):
        raw += "index.html"
    parts = [part for part in Path(raw).parts if part not in {"", ".", ".."}]
    return Path(*parts)


def fetch(client: requests.Session, url: str, root: Path) -> dict[str, Any]:
    started = time.monotonic()
    response = client.get(url, timeout=(30, 180))
    response.raise_for_status()
    destination = root / safe_relative_path(response.url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(response.content)
    temporary.replace(destination)
    elapsed = time.monotonic() - started
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    return {
        "url": url,
        "resolved_url": response.url,
        "local_path": str(destination),
        "http_status": response.status_code,
        "content_type": response.headers.get("Content-Type", ""),
        "last_modified": response.headers.get("Last-Modified", ""),
        "etag": response.headers.get("ETag", ""),
        "size": len(response.content),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "status": "PASS",
        "error": "",
    }


def snapshot_funp450(client: requests.Session, root: Path) -> dict[str, Any]:
    base = "https://p450.biodesign.ac.cn/"
    inventory: list[dict[str, Any]] = []
    main = fetch(client, base, root)
    inventory.append(main)
    html = Path(main["local_path"]).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    assets: list[str] = []
    for tag, attribute in (("a", "href"), ("link", "href"), ("script", "src"), ("img", "src")):
        for node in soup.find_all(tag):
            value = node.get(attribute)
            if not value:
                continue
            url = urljoin(base, value)
            if urlparse(url).netloc == "p450.biodesign.ac.cn":
                assets.append(url)
    assets = list(dict.fromkeys(assets))
    explicit_data_links = [url for url in assets if re.search(r"(download|api|\.csv$|\.tsv$|\.xlsx?$|\.fasta$)", url, re.I)]
    for url in assets:
        try:
            inventory.append(fetch(client, url, root))
        except Exception as exc:  # noqa: BLE001 - public asset failures are inventoried
            inventory.append({"url": url, "resolved_url": "", "local_path": "", "http_status": "", "content_type": "", "last_modified": "", "etag": "", "size": 0, "sha256": "", "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
    with (root / "funp450_link_inventory.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(inventory[0])
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(inventory)
    limitation = root / "FUNP450_ACCESS_LIMITATION.txt"
    if not explicit_data_links:
        limitation.write_text(
            "Snapshot date: 2026-08-18\n"
            "The public root and its same-origin linked assets were saved. The rendered application "
            "exposed no explicit public bulk-download, documented API schema, or static entry links in "
            "the root HTML. SiteGuard did not guess hidden endpoints, bypass access controls, or scrape "
            "undocumented services. The preserved JavaScript assets allow later reproducibility review.\n",
            encoding="utf-8",
        )
    return {
        "asset_count": len(inventory),
        "asset_failures": sum(row["status"] == "FAIL" for row in inventory),
        "explicit_data_links": explicit_data_links,
        "limitation_recorded": limitation.exists(),
        "status": "PASS_WITH_LIMITATION" if limitation.exists() else "PASS",
    }


def snapshot_nomenclature(client: requests.Session, root: Path) -> dict[str, Any]:
    base = "https://drnelson.uthsc.edu/nomenclature/"
    inventory: list[dict[str, Any]] = []
    main = fetch(client, base, root)
    inventory.append(main)
    html = Path(main["local_path"]).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    selected: list[str] = []
    pattern = re.compile(r"(?:/resources/biblio(?:A|B|C|D|E|_Fish)\.html$|/(?:animals|plants|fungal-genomes|bacteria)/$)", re.I)
    for anchor in soup.find_all("a", href=True):
        url = urljoin(base, anchor["href"])
        parsed = urlparse(url)
        if parsed.netloc == "drnelson.uthsc.edu" and pattern.search(parsed.path):
            selected.append(url)
    selected = list(dict.fromkeys(selected))
    for url in selected:
        try:
            inventory.append(fetch(client, url, root))
        except Exception as exc:  # noqa: BLE001 - complete public link status is required
            inventory.append({"url": url, "resolved_url": "", "local_path": "", "http_status": "", "content_type": "", "last_modified": "", "etag": "", "size": 0, "sha256": "", "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
    with (root / "nomenclature_link_inventory.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(inventory[0])
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(inventory)
    required_parts = {f"biblio{letter}.html" for letter in "ABCDE"}
    downloaded_names = {Path(row["local_path"]).name for row in inventory if row["status"] == "PASS" and row["local_path"]}
    failures = sum(row["status"] == "FAIL" for row in inventory)
    return {
        "selected_links": len(selected),
        "failures": failures,
        "required_parts_present": sorted(required_parts & downloaded_names),
        "status": "PASS" if failures == 0 and required_parts <= downloaded_names else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    raw = args.root / "data" / "raw" / "cyp450"
    client = build_session()
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "snapshot_date": "2026-08-18",
        "funp450": snapshot_funp450(client, raw / "funp450"),
        "nomenclature": snapshot_nomenclature(client, raw / "nomenclature"),
    }
    payload["status"] = "PASS" if payload["nomenclature"]["status"] == "PASS" and payload["funp450"]["status"] in {"PASS", "PASS_WITH_LIMITATION"} else "FAIL"
    output = args.root / "data" / "manifests" / "p450_auxiliary_snapshot_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
