#!/usr/bin/env python3
"""Parse preserved PlantP450 detail HTML into a source-linked staging table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup


def normalize_label(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    aliases = {
        "name": "cyp_name",
        "other_name_s": "other_names",
        "accession_number": "accession",
        "reference_s": "references",
    }
    return aliases.get(value, value)


def parse(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    soup = BeautifulSoup(content, "html.parser")
    fields: dict[str, str] = {}
    links: dict[str, list[str]] = {}
    for row in soup.find_all("tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) != 2:
            continue
        label = normalize_label(cells[0].get_text(" ", strip=True))
        if not label:
            continue
        fields[label] = cells[1].get_text(" ", strip=True)
        links[label] = [anchor.get("href", "") for anchor in cells[1].find_all("a", href=True)]
    return {
        **fields,
        "references_urls": json.dumps(links.get("references", []), ensure_ascii=False),
        "accession_urls": json.dumps(links.get("accession", []), ensure_ascii=False),
        "all_detail_fields_json": json.dumps(fields, ensure_ascii=False, sort_keys=True),
        "all_detail_links_json": json.dumps(links, ensure_ascii=False, sort_keys=True),
        "detail_local_path": str(path),
        "detail_size": len(content),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.entries.glob("*.html"))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            rows.append(parse(path))
        except Exception as exc:  # noqa: BLE001 - every preserved page is accounted for
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    if not rows:
        raise SystemExit("no PlantP450 detail pages parsed")
    detail_frame = pd.DataFrame(rows)
    catalog = pd.read_parquet(args.catalog)
    if "detail_local_path" not in catalog.columns:
        raise SystemExit("PlantP450 catalog lacks detail_local_path")
    frame = catalog.merge(
        detail_frame,
        on="detail_local_path",
        how="left",
        suffixes=("_catalog", "_detail"),
        validate="one_to_one",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    summary = {
        "input_pages": len(paths),
        "parsed_pages": len(rows),
        "failed_pages": len(failures),
        "catalog_entries": len(catalog),
        "catalog_rows_without_detail_page": int(frame["detail_size"].isna().sum()),
        "entries_with_accession": int(frame.get("accession", pd.Series(index=frame.index, dtype=str)).fillna("").ne("").sum()),
        "entries_with_function": int(frame.get("function", pd.Series(index=frame.index, dtype=str)).fillna("").ne("").sum()),
        "entries_with_references": int(frame.get("references", pd.Series(index=frame.index, dtype=str)).fillna("").ne("").sum()),
        "columns": list(frame.columns),
        "failures": failures,
        "status": (
            "PASS"
            if not failures and len(rows) == len(catalog)
            else "PASS_WITH_SOURCE_LIMITATION"
            if not failures and len(rows) + 1 == len(catalog) and int(frame["detail_size"].isna().sum()) == 1
            else "FAIL"
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
