#!/usr/bin/env python3
"""Fetch all pages from a public JSON API while preserving raw responses."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def make_session(user_agent: str) -> requests.Session:
    retry = Retry(
        total=20,
        connect=20,
        read=20,
        status=20,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        respect_retry_after_header=True,
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.headers["User-Agent"] = user_agent
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def records_from_page(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("page has no supported record list")


def next_url(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("next")
    return str(value) if value else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--requests-per-second", type=float, default=3.0)
    parser.add_argument("--user-agent", default="SiteGuard/3.0 scientific-data-freeze")
    args = parser.parse_args()

    raw_dir = args.output_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = make_session(args.user_agent)
    url: str | None = args.url
    page_number = 0
    records: list[Any] = []
    pages: list[dict[str, Any]] = []
    minimum_interval = 1.0 / args.requests_per_second
    while url:
        page_number += 1
        started = time.monotonic()
        response = session.get(url, timeout=(30, 120))
        response.raise_for_status()
        payload = response.json()
        raw_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        page_path = raw_dir / f"page_{page_number:06d}.json"
        atomic_write(page_path, raw_bytes + b"\n")
        page_records = records_from_page(payload)
        records.extend(page_records)
        following = next_url(payload)
        pages.append(
            {
                "page": page_number,
                "requested_url": url,
                "resolved_url": response.url,
                "status_code": response.status_code,
                "etag": response.headers.get("ETag", ""),
                "last_modified": response.headers.get("Last-Modified", ""),
                "record_count": len(page_records),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "next": following,
            }
        )
        url = following
        elapsed = time.monotonic() - started
        if url and elapsed < minimum_interval:
            time.sleep(minimum_interval - elapsed)

    args.merged.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.merged, json.dumps(records, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "initial_url": args.url,
        "page_count": page_number,
        "record_count": len(records),
        "pagination_terminated_at_null": bool(pages) and pages[-1]["next"] is None,
        "pages": pages,
        "status": "PASS" if records and pages[-1]["next"] is None else "FAIL",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.summary, json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")
    print(json.dumps({key: summary[key] for key in ("page_count", "record_count", "status")}))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
