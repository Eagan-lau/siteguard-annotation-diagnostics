#!/usr/bin/env python3
"""Stage selected indexed AlphaFold members into job-local scratch."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def load_accessions(path: Path) -> set[str]:
    values: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            value = raw.strip().split("\t", 1)[0]
            if value and not value.startswith("#"):
                values.add(value)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--accessions", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--maximum-files", type=int, default=100_000)
    args = parser.parse_args()

    requested = load_accessions(args.accessions)
    if len(requested) > args.maximum_files:
        raise SystemExit(f"requested {len(requested)} accessions exceeds --maximum-files")
    table = pq.read_table(args.index)
    selected = table.filter(
        pc.is_in(table["uniprot_accession"], value_set=pa.array(sorted(requested), type=pa.string()))
    )
    # Arrow's array class is backend-specific; convert the small selected set
    # after filtering to keep the large index columnar.
    rows = selected.to_pylist()
    args.target.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, object]] = []
    found_accessions: set[str] = set()
    with tarfile.open(args.tar, mode="r:") as archive:
        for row in rows:
            destination = args.target / row["filename"]
            result = {"accession": row["uniprot_accession"], "member": row["relative_path"], "path": str(destination), "status": "FAIL", "error": ""}
            try:
                member = archive.getmember(row["relative_path"])
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError("tar member has no file stream")
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                with source, temporary.open("wb") as output:
                    while chunk := source.read(8 * 1024 * 1024):
                        output.write(chunk)
                with gzip.open(temporary, "rb") as handle:
                    while handle.read(8 * 1024 * 1024):
                        pass
                temporary.replace(destination)
                result["status"] = "PASS"
                found_accessions.add(row["uniprot_accession"])
            except Exception as exc:  # noqa: BLE001 - every staged member is audited
                result["error"] = f"{type(exc).__name__}: {exc}"
            report_rows.append(result)
    for accession in sorted(requested - found_accessions):
        report_rows.append({"accession": accession, "member": "", "path": "", "status": "UNAVAILABLE", "error": "not present in bulk v6 index"})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8", newline="") as handle:
        fields = ["accession", "member", "path", "status", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(report_rows)
    summary = {
        "requested": len(requested),
        "staged_members": sum(row["status"] == "PASS" for row in report_rows),
        "unavailable_accessions": len(requested - found_accessions),
        "failures": sum(row["status"] == "FAIL" for row in report_rows),
    }
    print(json.dumps(summary))
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
