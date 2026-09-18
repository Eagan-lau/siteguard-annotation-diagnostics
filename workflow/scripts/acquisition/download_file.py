#!/usr/bin/env python3
"""Resumable, metadata-rich downloader for immutable SiteGuard raw inputs."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin


UTC = dt.timezone.utc


@dataclass(frozen=True)
class Resource:
    resource_id: str
    database: str
    version: str
    url: str
    relative_path: str
    required: bool
    large: bool


@dataclass
class Result:
    resource_name: str
    database: str
    version: str
    source_url: str
    resolved_url: str = ""
    local_path: str = ""
    download_started: str = ""
    download_finished: str = ""
    http_status: str = ""
    etag: str = ""
    last_modified: str = ""
    content_length: str = ""
    actual_size: int = 0
    supplier_checksum: str = ""
    local_sha256: str = ""
    archive_test: str = "NOT_APPLICABLE"
    parse_test: str = "NOT_RUN"
    record_count: str = ""
    required: bool = True
    status: str = "FAIL"
    error_message: str = ""


def now() -> str:
    return dt.datetime.now(UTC).isoformat()


def load_json_yaml(path: Path) -> dict[str, Any]:
    # The checked-in YAML files intentionally use the JSON subset so the
    # bootstrap downloader has no third-party parser dependency.
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expand_manifest(payload: dict[str, Any]) -> list[Resource]:
    resources: list[Resource] = []
    for group in payload.get("groups", []):
        for entry in group["files"]:
            item = {"name": entry} if isinstance(entry, str) else entry
            remote_path = item.get("path", item["name"])
            local_name = item["name"]
            resources.append(
                Resource(
                    resource_id=f"{group['id']}::{local_name}",
                    database=group["database"],
                    version=str(group["version"]),
                    url=urljoin(group["base_url"], remote_path),
                    relative_path=str(Path(group["relative_dir"]) / local_name),
                    required=bool(item.get("required", group.get("required", True))),
                    large=bool(item.get("large", False)),
                )
            )
    for item in payload.get("singletons", []):
        resources.append(
            Resource(
                resource_id=item["id"],
                database=item["database"],
                version=str(item["version"]),
                url=item["url"],
                relative_path=item["relative_path"],
                required=bool(item.get("required", True)),
                large=bool(item.get("large", False)),
            )
        )
    ids = [resource.resource_id for resource in resources]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate resource_id in download manifest")
    return resources


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def run_checked(command: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def archive_test(path: Path) -> str:
    lower = path.name.lower()
    try:
        if lower.endswith((".tar.gz", ".tgz")):
            with tarfile.open(path, "r:gz") as archive:
                next(iter(archive), None)
            return "PASS"
        if lower.endswith(".tar.bz2"):
            with tarfile.open(path, "r:bz2") as archive:
                next(iter(archive), None)
            return "PASS"
        if lower.endswith(".tar"):
            with tarfile.open(path, "r:") as archive:
                next(iter(archive), None)
            return "PASS"
        if lower.endswith(".gz"):
            with gzip.open(path, "rb") as handle:
                while handle.read(8 * 1024 * 1024):
                    pass
            return "PASS"
        if lower.endswith(".bz2"):
            run_checked(["bzip2", "-t", str(path)])
            return "PASS"
        return "NOT_APPLICABLE"
    except Exception as exc:  # noqa: BLE001 - surfaced in a structured result
        return f"FAIL: {exc}"


def fetch_headers(url: str, user_agent: str) -> dict[str, str]:
    with tempfile.NamedTemporaryFile(prefix="siteguard_headers_", delete=False) as tmp:
        header_path = Path(tmp.name)
    try:
        completed = run_checked(
            [
                "curl",
                "--silent",
                "--show-error",
                "--location",
                "--head",
                "--max-time",
                "60",
                "--user-agent",
                user_agent,
                "--dump-header",
                str(header_path),
                "--output",
                "/dev/null",
                "--write-out",
                "%{http_code}\n%{url_effective}\n",
                url,
            ]
        )
        output_lines = completed.stdout.splitlines()
        result = {
            "http_status": output_lines[0] if output_lines else "",
            "resolved_url": output_lines[1] if len(output_lines) > 1 else url,
        }
        for raw in header_path.read_text(encoding="iso-8859-1").splitlines():
            if ":" not in raw:
                continue
            key, value = raw.split(":", 1)
            key = key.strip().lower()
            if key in {"etag", "last-modified", "content-length"}:
                result[key.replace("-", "_")] = value.strip()
        return result
    finally:
        header_path.unlink(missing_ok=True)


def transfer(resource: Resource, target: Path, settings: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if resource.large and shutil.which("aria2c"):
        command = [
            "aria2c",
            "--continue=true",
            "--max-connection-per-server=8",
            "--split=8",
            "--min-split-size=20M",
            f"--max-tries={settings['max_retries']}",
            f"--retry-wait={settings['retry_wait_seconds']}",
            f"--timeout={settings['transfer_timeout_seconds']}",
            f"--connect-timeout={settings['connect_timeout_seconds']}",
            "--file-allocation=none",
            "--auto-file-renaming=false",
            "--allow-overwrite=false",
            "--console-log-level=warn",
            "--summary-interval=0",
            "--show-console-readout=false",
            f"--user-agent={settings['user_agent']}",
            f"--dir={target.parent}",
            f"--out={target.name}",
            resource.url,
        ]
    else:
        command = [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--continue-at",
            "-",
            "--retry",
            str(settings["max_retries"]),
            "--retry-all-errors",
            "--retry-delay",
            str(settings["retry_wait_seconds"]),
            "--connect-timeout",
            str(settings["connect_timeout_seconds"]),
            "--max-time",
            "0",
            "--user-agent",
            settings["user_agent"],
            "--output",
            str(target),
            resource.url,
        ]
    run_checked(command)


def process(resource: Resource, raw_root: Path, settings: dict[str, Any]) -> Result:
    target = raw_root / resource.relative_path
    result = Result(
        resource_name=resource.resource_id,
        database=resource.database,
        version=resource.version,
        source_url=resource.url,
        local_path=str(target),
        required=resource.required,
        download_started=now(),
    )
    try:
        headers = fetch_headers(resource.url, settings["user_agent"])
        for key, value in headers.items():
            setattr(result, key, value)
        reusable = target.is_file() and target.stat().st_size > 0
        if reusable:
            prior_test = archive_test(target)
            reusable = not prior_test.startswith("FAIL")
        if not reusable:
            transfer(resource, target, settings)
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError("download did not create a non-empty regular file")
        result.actual_size = target.stat().st_size
        result.archive_test = archive_test(target)
        if result.archive_test.startswith("FAIL"):
            raise RuntimeError(result.archive_test)
        result.local_sha256 = sha256(target)
        result.parse_test = "PENDING_DOMAIN_VALIDATION"
        result.status = "PASS"
    except Exception as exc:  # noqa: BLE001 - recorded for targeted retry
        result.error_message = f"{type(exc).__name__}: {exc}"
        result.status = "FAIL"
        if target.exists():
            result.actual_size = target.stat().st_size
    finally:
        result.download_finished = now()
    return result


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("SITEGUARD_ROOT", "")))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--large-only", action="store_true")
    parser.add_argument("--exclude-large", action="store_true")
    parser.add_argument("--retry-failed", type=Path)
    parser.add_argument("--result-prefix", default="static")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not str(args.root):
        raise SystemExit("SITEGUARD_ROOT or --root is required")
    root = args.root.resolve()
    settings = load_json_yaml(args.settings)
    resources = expand_manifest(load_json_yaml(args.manifest))
    if args.large_only:
        resources = [resource for resource in resources if resource.large]
    if args.exclude_large:
        resources = [resource for resource in resources if not resource.large]
    if args.retry_failed:
        with args.retry_failed.open("r", encoding="utf-8", newline="") as handle:
            failed_ids = {row["resource_name"] for row in csv.DictReader(handle, delimiter="\t")}
        resources = [resource for resource in resources if resource.resource_id in failed_ids]
    workers = args.workers or int(settings["static_workers"])
    raw_root = root / "data" / "raw"
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process, resource, raw_root, settings) for resource in resources]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    results.sort(key=lambda item: item.resource_name)
    rows = [asdict(result) for result in results]
    manifest_dir = root / "data" / "manifests"
    atomic_json(manifest_dir / f"{args.result_prefix}_download_results.json", rows)
    fields = list(Result.__dataclass_fields__)
    write_tsv(manifest_dir / f"{args.result_prefix}_download_results.tsv", rows, fields)
    failures = [row for row in rows if row["status"] == "FAIL"]
    write_tsv(root / "logs" / f"FAILED_{args.result_prefix.upper()}_RESOURCES.tsv", failures, fields)
    print(json.dumps({"resources": len(rows), "pass": len(rows) - len(failures), "fail": len(failures)}))
    return 1 if any(row["required"] and row["status"] == "FAIL" for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
