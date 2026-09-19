#!/usr/bin/env python3
"""Index AlphaFold v6 tar members without exceeding the cluster file quota."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import tarfile
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq


FILENAME_RE = re.compile(
    r"(?:^|/)AF-(?P<accession>.+)-F(?P<fragment>[0-9]+)-model_v(?P<model_version>[0-9]+)\.cif\.gz$"
)
SCHEMA = pa.schema(
    [
        ("filename", pa.string()),
        ("uniprot_accession", pa.string()),
        ("isoform", pa.string()),
        ("fragment", pa.int16()),
        ("model_version", pa.int16()),
        ("relative_path", pa.string()),
        ("tar_header_offset", pa.int64()),
        ("tar_data_offset", pa.int64()),
        ("compressed_size", pa.int64()),
    ]
)


def parse_member(member: tarfile.TarInfo) -> dict[str, Any] | None:
    if not member.isfile():
        return None
    match = FILENAME_RE.search(member.name)
    if not match:
        return None
    accession = match.group("accession")
    isoform = accession if "-" in accession else ""
    return {
        "filename": Path(member.name).name,
        "uniprot_accession": accession,
        "isoform": isoform,
        "fragment": int(match.group("fragment")),
        "model_version": int(match.group("model_version")),
        "relative_path": member.name,
        "tar_header_offset": member.offset,
        "tar_data_offset": member.offset_data,
        "compressed_size": member.size,
    }


def metadata_numeric_values(value: Any, prefix: str = "") -> Iterator[tuple[str, int]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from metadata_numeric_values(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from metadata_numeric_values(child, f"{prefix}[{index}]")
    elif isinstance(value, int):
        yield prefix, value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    total_members = 0
    indexed_members = 0
    unmatched_files = 0
    fragments: dict[int, int] = {}
    batch: list[dict[str, Any]] = []
    writer = pq.ParquetWriter(temporary, SCHEMA, compression="zstd")
    try:
        with tarfile.open(args.tar, mode="r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                total_members += 1
                row = parse_member(member)
                if row is None:
                    unmatched_files += 1
                    continue
                indexed_members += 1
                fragments[row["fragment"]] = fragments.get(row["fragment"], 0) + 1
                batch.append(row)
                if len(batch) >= args.batch_size:
                    writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                    batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
    finally:
        writer.close()
    temporary.replace(args.output)

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    numeric_metadata = dict(metadata_numeric_values(metadata))
    plausible_expected_counts = {
        key: value
        for key, value in numeric_metadata.items()
        if 100_000 <= value <= 2_000_000
    }
    status = "PASS" if indexed_members > 0 and unmatched_files == 0 else "WARN"
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "archive": str(args.tar.resolve()),
        "archive_size": args.tar.stat().st_size,
        "index": str(args.output.resolve()),
        "total_file_members": total_members,
        "indexed_structure_members": indexed_members,
        "unmatched_file_members": unmatched_files,
        "fragment_counts": fragments,
        "metadata_candidate_expected_counts": plausible_expected_counts,
        "storage_mode": "indexed_tar_on_demand",
        "status": status,
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if indexed_members > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

