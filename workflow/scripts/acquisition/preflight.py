#!/usr/bin/env python3
"""Cluster storage, inode, filesystem, tool, and network preflight."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shutil
import socket
import subprocess
from pathlib import Path
from urllib.parse import urlparse


DOMAINS = [
    "https://ftp.uniprot.org",
    "https://ftp.expasy.org",
    "https://www.rhea-db.org",
    "https://ftp.ebi.ac.uk",
    "https://www.ebi.ac.uk",
    "https://alphafold.ebi.ac.uk",
    "https://files.wwpdb.org",
    "https://files.rcsb.org",
    "https://download.cathdb.info",
    "https://www.cellknowledge.com.cn",
    "https://erda.dk",
    "https://p450.biodesign.ac.cn",
    "https://drnelson.uthsc.edu",
    "https://rest.uniprot.org",
    "https://pubchem.ncbi.nlm.nih.gov",
    "https://huggingface.co",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(root)
    stat = os.statvfs(root)
    free_inodes = stat.f_favail
    report: dict[str, object] = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": str(root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "free_bytes": disk.free,
        "free_inodes": free_inodes,
        "minimum_free_bytes": settings["minimum_free_bytes"],
        "minimum_free_inodes": settings["minimum_free_inodes"],
        "proxy_present": {
            key: bool(os.environ.get(key))
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
        },
        "tools": {name: shutil.which(name) for name in ("curl", "aria2c", "git", "python", "sha256sum", "tar", "gzip", "bzip2")},
        "network": {},
    }
    failures: list[str] = []
    warnings: list[str] = []
    quota_file_remaining: int | None = None
    exception = settings.get("ceci_inode_exception", {})
    if shutil.which("ceci-quota"):
        quota_result = subprocess.run(["ceci-quota"], capture_output=True, text=True)
        quota_text = re.sub(r"\x1b\[[0-9;]*m", "", quota_result.stdout + quota_result.stderr)
        report["ceci_quota"] = quota_text
        for line in quota_text.splitlines():
            fields = line.split()
            if fields and fields[0] == "$GLOBALSCRATCH" and len(fields) >= 7:
                try:
                    file_used, file_limit = int(fields[-2]), int(fields[-1])
                except ValueError:
                    continue
                quota_file_remaining = file_limit - file_used
                report["ceci_globalscratch_file_used"] = file_used
                report["ceci_globalscratch_file_limit"] = file_limit
                report["ceci_globalscratch_file_remaining"] = quota_file_remaining
                break
    if quota_file_remaining is None and int(exception.get("known_file_limit", 0)) > 0:
        project_entries = 0
        for _, directory_names, file_names in os.walk(root):
            project_entries += len(directory_names) + len(file_names)
        quota_file_remaining = int(exception["known_file_limit"]) - project_entries
        report["configured_file_limit"] = int(exception["known_file_limit"])
        report["counted_project_entries"] = project_entries
        report["configured_file_remaining"] = quota_file_remaining
    if disk.free < int(settings["minimum_free_bytes"]):
        failures.append("free_bytes_below_minimum")
    effective_free_inodes = min(free_inodes, quota_file_remaining) if quota_file_remaining is not None else free_inodes
    if free_inodes < 0 and quota_file_remaining is not None:
        effective_free_inodes = quota_file_remaining
    report["effective_free_inodes"] = effective_free_inodes
    if effective_free_inodes < int(settings["minimum_free_inodes"]):
        if exception.get("enabled") and effective_free_inodes >= int(exception["minimum_remaining_user_files"]):
            warnings.append("free_inodes_below_taskbook_minimum_indexed_tar_strategy_required")
            report["inode_strategy"] = exception["strategy"]
        else:
            failures.append("free_inodes_below_minimum")
    for url in DOMAINS:
        host = urlparse(url).hostname or url
        try:
            # Preflight checks routing/DNS/TCP only. Application-level 403/405
            # responses are common for scientific endpoints that reject HEAD;
            # each real transfer still performs its own HTTP validation.
            with socket.create_connection((host, 443), timeout=20):
                report["network"][host] = {"ok": True, "port": 443}
        except Exception as exc:  # noqa: BLE001 - preflight records every endpoint
            report["network"][host] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(f"network:{host}")
    report["df_h"] = subprocess.run(["df", "-h", str(root)], capture_output=True, text=True).stdout
    report["df_i"] = subprocess.run(["df", "-i", str(root)], capture_output=True, text=True).stdout
    report["failures"] = failures
    report["warnings"] = warnings
    report["status"] = "PASS" if not failures else "FAIL"
    output = root / "logs" / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "failures": failures}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
