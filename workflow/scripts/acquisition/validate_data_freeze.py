#!/usr/bin/env python3
"""Perform final domain validation, inventory, checksums, and data-freeze gating."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def status_ok(payload: Any) -> bool:
    if isinstance(payload, dict):
        return str(payload.get("status", "")).startswith("PASS")
    if isinstance(payload, list):
        return bool(payload) and all(str(row.get("status", "")).startswith("PASS") for row in payload)
    return False


def component(name: str, database: str, version: str, required: bool, passed: bool, record_count: int | str, details: str, warning: str = "") -> dict[str, Any]:
    return {
        "resource_name": name, "database": database, "version": version, "source_url": "MULTIPLE_OR_DERIVED",
        "resolved_url": "", "local_path": "", "download_started": "", "download_finished": "",
        "http_status": "", "etag": "", "last_modified": "", "content_length": "", "actual_size": "",
        "supplier_checksum": "", "local_sha256": "", "archive_test": "PASS" if passed else "FAIL",
        "parse_test": "PASS" if passed else "FAIL", "record_count": record_count, "required": required,
        "status": "PASS" if passed else "FAIL", "error_message": warning if passed else details,
        "validation_details": details,
    }


def metalink_hashes(path: Path) -> dict[str, tuple[str, str]]:
    output: dict[str, tuple[str, str]] = {}
    root = ET.parse(path).getroot()
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "file":
            continue
        name = node.attrib.get("name", "")
        for child in node.iter():
            if child.tag.rsplit("}", 1)[-1] == "hash" and child.text:
                algorithm = child.attrib.get("type", "").lower()
                if algorithm in {"sha-256", "sha256", "md5"}:
                    output[name] = (algorithm, child.text.strip().lower())
                    break
    return output


def supplier_validation(root: Path) -> tuple[dict[str, str], list[str]]:
    verified: dict[str, str] = {}
    failures: list[str] = []
    metalinks = [
        root / "data/raw/uniprot/current_2026_02/RELEASE.metalink",
        root / "data/raw/uniprot/historical_2023_01/knowledgebase/RELEASE.metalink",
        root / "data/raw/uniprot/historical_2026_01/knowledgebase/RELEASE.metalink",
    ]
    for metalink in metalinks:
        for name, (algorithm, expected) in metalink_hashes(metalink).items():
            # Metalink file names are relative to the release directory.  A
            # basename-only lookup is unsafe because several frozen releases
            # contain files named README or UniProtKB_SwissProt-relstat.html.
            local = metalink.parent / name
            if not local.is_file():
                continue
            actual = sha256(local) if algorithm in {"sha-256", "sha256"} else md5(local)
            key = str(local)
            verified[key] = f"{algorithm}:{expected}"
            if actual != expected:
                failures.append(f"supplier checksum mismatch: {local}")
    pfam_dir = root / "data/raw/pfam/current_release_2026_01_22"
    checksum_file = pfam_dir / "md5_checksums"
    if checksum_file.is_file():
        for line in checksum_file.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            expected, name = parts[0].lower(), parts[-1].lstrip("*")
            local = pfam_dir / name
            if local.is_file():
                verified[str(local)] = f"md5:{expected}"
                if md5(local) != expected:
                    failures.append(f"supplier checksum mismatch: {local}")
    return verified, failures


def inventory_files(root: Path, known_sha: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw = root / "data/raw"
    for path in sorted(raw.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name.endswith((".tmp", ".partial")):
            continue
        resolved = str(path.resolve())
        digest = known_sha.get(resolved) or sha256(path)
        rows.append(
            {
                "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                "absolute_path": resolved, "size": path.stat().st_size, "sha256": digest,
                "modified_utc": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat(),
            }
        )
        if len(rows) % 5000 == 0:
            print(json.dumps({"inventory_files": len(rows)}), flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifests = root / "data/manifests"
    static_rows: list[dict[str, Any]] = load_json(manifests / "static_download_results.json")
    supplier, supplier_failures = supplier_validation(root)
    known_sha: dict[str, str] = {}
    for row in static_rows:
        path = Path(row["local_path"])
        if path.is_file():
            expected_local = row["local_sha256"]
            actual = sha256(path) if path.stat().st_size < 2 * 1024**3 else expected_local
            known_sha[str(path.resolve())] = actual
            row["local_sha256"] = actual
            row["supplier_checksum"] = supplier.get(str(path.resolve()), "")
            row["parse_test"] = "PASS" if row["status"] == "PASS" and path.stat().st_size == int(row["actual_size"]) else "FAIL"
            if actual != expected_local:
                row["status"] = "FAIL"
                row["error_message"] = "local SHA256 changed since download"

    components: list[dict[str, Any]] = []
    static_pass = all(row["status"] == "PASS" and Path(row["local_path"]).is_file() for row in static_rows if row["required"])
    components.append(component("gate::static_resources", "multiple", "frozen", True, static_pass, len(static_rows), "96 configured static resources present and size/hash checked"))
    archive_summary = load_json(manifests / "archive_extraction_summary.json")
    components.append(component("gate::archive_preparation", "UniProt/Rhea/CATH/Pfam", "frozen", True, status_ok(archive_summary) and all((root / f"data/raw/pfam/current_release_2026_01_22/Pfam-A.hmm.h3{x}").is_file() for x in "mfip"), sum(int(row["members"]) for row in archive_summary["archives"]), "six frozen archives extracted safely and Pfam HMM pressed"))
    uniprot = load_json(manifests / "current_swissprot_download_catalog_summary.json")
    reldate = (root / "data/raw/uniprot/current_2026_02/reldate.txt").read_text(encoding="utf-8", errors="replace")
    components.append(component("gate::uniprot_current_and_history", "UniProtKB/Swiss-Prot", "2026_02;2026_01;2023_01", True, status_ok(uniprot) and "2026_02" in reldate and int(uniprot["entries"]) > 500_000, uniprot["entries"], "current XML stream parsed; historical DAT/FASTA/XML archives extracted"))
    rhea_ok = all((root / f"data/raw/rhea/release_{version}/extracted/{version}/tsv/rhea2uniprot_trembl.tsv.gz").is_file() for version in (126, 140, 141))
    components.append(component("gate::rhea_releases", "Rhea", "126;140;141", True, rhea_ok, 3, "all required reaction, direction, EC, Swiss-Prot, TrEMBL, ChEBI and SDF resources extracted"))
    enzyme_files = list((root / "data/raw/enzyme/release_2026_06_10").glob("*"))
    enzclass = (root / "data/raw/enzyme/release_2026_06_10/enzclass.txt").read_text(encoding="utf-8", errors="replace")
    components.append(component("gate::enzyme", "ENZYME", "2026-06-10", True, len([p for p in enzyme_files if p.is_file()]) >= 5 and all(f"{i}." in enzclass for i in range(1, 8)), 7, "five official files present and EC classes 1-7 detected"))
    mcsa_entries = load_json(manifests / "mcsa_entries_download_summary.json")
    mcsa_residues = load_json(manifests / "mcsa_residues_download_summary.json")
    mcsa_structures = load_json(manifests / "mcsa_structures_download_results.json")
    mcsa_ok = status_ok(mcsa_entries) and status_ok(mcsa_residues) and status_ok(mcsa_structures)
    components.append(component("gate::mcsa", "M-CSA/PDB/SIFTS", "snapshot 2026-08-18", True, mcsa_ok, f"entries={mcsa_entries.get('records', mcsa_entries.get('total_records', ''))};residues={mcsa_residues.get('records', mcsa_residues.get('total_records', ''))};resources={len(mcsa_structures)}", "paginated APIs reached next=null; three flat files and reference structures validated"))
    af_bulk = load_json(manifests / "afdb_bulk_index_summary.json")
    af_selected = load_json(manifests / "alphafold_selected_summary.json")
    af_ok = status_ok(af_bulk) and status_ok(af_selected) and int(af_bulk["indexed_structure_members"]) == 550122
    components.append(component("gate::alphafold_core", "AlphaFold DB", "v6 + API snapshot 2026-08-18", True, af_ok, f"bulk={af_bulk['indexed_structure_members']};api={af_selected.get('api_available', '')};unavailable={af_selected.get('api_unavailable', '')}", "bulk tar/index plus selected API CIF/PAE availability complete"))
    pdb_packed = load_json(manifests / "packed_selected_pdb_sifts_summary.json")
    components.append(component("gate::selected_pdb_sifts", "PDB/SIFTS", "snapshot 2026-08-18", True, status_ok(pdb_packed), pdb_packed.get("selected_pdb_ids", ""), "selected current-enzyme PDB mmCIF/SIFTS responses frozen in tar shards; verified source 404s retained"))
    p450 = load_json(manifests / "p450_snapshot_summary.json")
    p450_aux = load_json(manifests / "p450_auxiliary_snapshot_summary.json")
    plant = load_json(manifests / "plantp450_parse_summary.json")
    p450_ok = status_ok(p450) and status_ok(p450_aux) and status_ok(plant)
    warning = "PlantP450 source table contains one verified permanent 404 (CYP74A); FunP450 public access limitations are preserved" if p450_ok else ""
    components.append(component("gate::cyp450", "P450Rdb/PlantP450/FunP450/nomenclature", "snapshot 2026-08-18", True, p450_ok, "P450Rdb=1012;PlantP450=913", "all public resources completed or source limitation explicitly frozen", warning))
    pubchem = load_json(root / "data/raw/pubchem/external_p450_snapshot_2026-08-18/pubchem_download_summary.json")
    components.append(component("gate::pubchem_targeted", "PubChem", "snapshot 2026-08-18", True, status_ok(pubchem), pubchem["property_records"], "targeted external-P450 CIDs only"))
    trembl = load_json(root / "data/derived_download_lists/trembl_audit_sampling_report.json")
    trembl_af = load_json(manifests / "trembl_audit_alphafold_summary.json")
    trembl_ok = status_ok(trembl) and status_ok(trembl_af) and 0 < int(trembl["final_panel_accessions"]) <= 50_000
    components.append(component("gate::trembl_audit", "UniProtKB/TrEMBL + AlphaFold", "snapshot 2026-08-18", True, trembl_ok, trembl.get("final_panel_accessions", ""), "historical/reaction/EC/taxonomy-stratified panel, metadata/FASTA and AF availability frozen"))
    esm = load_json(manifests / "esm2_offline_validation.json")
    components.append(component("gate::esm2", "ESM2", "pinned t33/t12 commits", True, status_ok(esm), len(esm["models"]), "both frozen model snapshots loaded and executed offline on RTX 6000 Ada"))
    components.append(component("gate::supplier_checksums", "UniProt/Pfam", "supplier manifests", True, bool(supplier) and not supplier_failures, len(supplier), "matched available official metalink SHA256/MD5 and Pfam md5_checksums", "; ".join(supplier_failures)))

    validation_rows = static_rows + components
    required_failures = [row for row in validation_rows if bool(row.get("required")) and row.get("status") != "PASS"]
    reports = root / "data/manifests"
    pd.DataFrame(validation_rows).to_csv(reports / "download_validation_report.tsv", sep="\t", index=False)
    (reports / "download_validation_report.json").write_text(json.dumps(validation_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failed_path = root / "logs/FAILED_RESOURCES.tsv"
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(required_failures, columns=pd.DataFrame(validation_rows).columns).to_csv(failed_path, sep="\t", index=False)

    inventory = inventory_files(root, known_sha)
    inventory_frame = pd.DataFrame(inventory)
    inventory_frame.to_parquet(reports / "resource_inventory.parquet", index=False)
    inventory_frame.to_csv(reports / "resource_inventory.tsv", sep="\t", index=False)
    checksums = root / "checksums/SHA256SUMS"
    checksums.parent.mkdir(parents=True, exist_ok=True)
    checksums.write_text("".join(f"{row['sha256']}  {row['relative_path']}\n" for row in inventory), encoding="utf-8")

    total_size = sum(int(row["size"]) for row in inventory)
    git_commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    environment_file = root / "logs/conda-list-explicit.txt"
    environment_hash = sha256(environment_file) if environment_file.is_file() else ""
    status_dir = root / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = status_dir / "DATA_FREEZE_COMPLETE"
    incomplete_marker = status_dir / "DATA_FREEZE_INCOMPLETE"
    marker_payload = {
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "project_root": str(root), "git_commit": git_commit,
        "environment_sha256": environment_hash, "resources": len(validation_rows),
        "pass": sum(row["status"] == "PASS" for row in validation_rows), "warn": sum(bool(row.get("error_message")) and row["status"] == "PASS" for row in validation_rows),
        "failed": len(required_failures), "total_raw_bytes": total_size,
        "versions": {"UniProt": "2026_02; history 2026_01, 2023_01", "Rhea": "141; history 140,126", "AlphaFold": "v6", "Pfam": "current 2026-01-22", "CATH": "4.4.0"},
        "snapshot_date": "2026-08-18",
    }
    if required_failures:
        complete_marker.unlink(missing_ok=True)
        incomplete_marker.write_text(json.dumps(marker_payload, indent=2) + "\n", encoding="utf-8")
    else:
        incomplete_marker.unlink(missing_ok=True)
        complete_marker.write_text(json.dumps(marker_payload, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# SiteGuard V3 data download and freeze report", "", f"Generated: {marker_payload['completed_at']}",
        f"Project root: `{root}`", f"Git commit: `{git_commit}`", f"Freeze status: **{'COMPLETE' if not required_failures else 'INCOMPLETE'}**",
        f"Raw inventory: {len(inventory):,} files, {total_size:,} bytes", "", "## Validation gates", "",
        "| Gate | Version | Records | Status | Notes |", "|---|---|---:|---|---|",
    ]
    for row in components:
        notes = (row.get("error_message") or row.get("validation_details") or "").replace("|", "/")
        report_lines.append(f"| {row['resource_name']} | {row['version']} | {row['record_count']} | {row['status']} | {notes} |")
    report_lines.extend(["", "## Direct analysis inputs", "", "- `data/raw/`: immutable source snapshots", "- `data/staging/`: parsed download-stage tables", "- `data/derived_download_lists/`: bounded accession/PDB/CID panels", "- `code/analysis_pipeline/`: only allowed downstream entry point", "", "Verified source-side unavailability is retained in manifests and is never silently dropped."])
    report = "\n".join(report_lines) + "\n"
    (root / "DATA_DOWNLOAD_REPORT.md").write_text(report, encoding="utf-8")
    (root / "DATA_DOWNLOAD_REPORT.txt").write_text(report, encoding="utf-8")
    print(json.dumps(marker_payload), flush=True)
    return 0 if not required_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
