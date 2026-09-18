#!/usr/bin/env python3
"""Prepare label-blind Phase 28 experimental structures, Pfam and CATH metadata."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
SOURCE = Path(os.environ.get("SITEGUARD_SOURCE", "/globalsc/ulg/plgen/yugenliu/SiteGuard"))
PHASE_ID = int(os.environ.get("RCSB_EXTERNAL_PHASE", "33"))
if PHASE_ID not in {28, 30, 31, 32, 33}:
    raise RuntimeError(f"Unsupported structural external phase: {PHASE_ID}")
RESULTS = ROOT / f"results/phase{PHASE_ID}"
WORK = ROOT / f"data/interim/phase{PHASE_ID}_inference"
STRUCTURES = WORK / "structures"
CHECKPOINTS = ROOT / "checkpoints"
REPORTS = ROOT / f"reports/phase{PHASE_ID}_external_blind"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request(session: requests.Session, url: str, binary: bool = False):
    last_error: Exception | None = None
    for attempt in range(7):
        try:
            response = session.get(url, timeout=(20, 180))
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(min(2**attempt, 60))
                continue
            response.raise_for_status()
            return response.content if binary else response.json()
        except (requests.RequestException, OSError, ValueError) as error:
            last_error = error
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"RCSB request failed: {url}") from last_error


def clan_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields and fields[0]:
                mapping[fields[0]] = fields[1] if len(fields) > 1 and fields[1] else "NO_CLAN"
    return mapping


def pfam_calls(path: Path) -> dict[str, set[str]]:
    calls: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 23:
                raise RuntimeError(f"Malformed HMMER domtblout row: {line[:200]}")
            accession = fields[1].split(".")[0]
            query = fields[3]
            if accession.startswith("PF"):
                calls[query].add(accession)
    return calls


def cath_superfamilies(payload: dict[str, object]) -> list[str]:
    values: set[str] = set()
    for item in payload.get("rcsb_polymer_instance_annotation", []) or []:
        if str(item.get("type", "")).upper() != "CATH" and str(item.get("provenance_source", "")).upper() != "CATH":
            continue
        lineage = item.get("annotation_lineage") or []
        depth4 = [str(node.get("id")) for node in lineage if int(node.get("depth", -1)) == 4 and node.get("id")]
        identifier = depth4[0] if depth4 else str(item.get("annotation_id") or "")
        if identifier and identifier.count(".") >= 3:
            values.add(identifier)
    return sorted(values)


def main() -> None:
    required = CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}A_EXTERNAL_COHORT_LOCKED"
    if not required.is_file():
        raise FileNotFoundError(required)
    if (RESULTS / "external_predictions.parquet").exists() or (RESULTS / "external_blind_predictions.parquet").exists():
        raise RuntimeError("Phase 28 prediction stage has started; evidence preparation is closed")
    for directory in (WORK, STRUCTURES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)

    allowed_columns = [
        "query_id", "representative_pdb_entry_id", "representative_asym_ids_json",
        "sequence_length", "taxonomy_ids_json",
    ]
    cohort = pd.read_parquet(RESULTS / "rcsb_strict_blind_cohort.parquet", columns=allowed_columns)
    calls = pfam_calls(WORK / "external_pfam.domtblout")
    clans = clan_map(SOURCE / "data/raw/pfam/current_release_2026_01_22/Pfam-A.clans.tsv.gz")
    lock = json.loads((ROOT / "models/phase25/evidencejudge/errorjudge/errorjudge_operating_point_lock.json").read_text())
    fit_pfam = set(lock["gate_state"]["fit_pfam"])
    fit_clan = set(lock["gate_state"]["fit_pfam_clan"])
    fit_cath = set(lock["gate_state"]["fit_cath"])

    session = requests.Session()
    session.headers.update({"User-Agent": "SiteGuard-academic-external-validation/4.0"})
    rows = []
    manifest = []
    for ordinal, row in enumerate(cohort.sort_values("query_id").itertuples(index=False), start=1):
        asym_ids = json.loads(row.representative_asym_ids_json)
        if not asym_ids:
            raise RuntimeError(f"No representative asym ID: {row.query_id}")
        asym_id = str(asym_ids[0])
        token = f"RCSB{PHASE_ID}Q{ordinal:06d}"
        coordinate = STRUCTURES / f"{token}.cif.gz"
        content = request(session, f"https://files.rcsb.org/download/{row.representative_pdb_entry_id}.cif.gz", binary=True)
        coordinate.write_bytes(content)
        instance = request(
            session,
            f"https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{row.representative_pdb_entry_id}/{asym_id}",
        )
        cath = cath_superfamilies(instance)
        pfam = sorted(calls.get(str(row.query_id), set()))
        primary_pfam = pfam[0] if pfam else "__MISSING__"
        primary_clan = clans.get(primary_pfam, "UNASSIGNED") if pfam else "__MISSING__"
        primary_cath = cath[0] if cath else "__MISSING__"
        taxonomy = json.loads(row.taxonomy_ids_json)
        rows.append({
            "query_protein_id": str(row.query_id),
            "taxonomy_id": str(taxonomy[0]) if taxonomy else "__MISSING__",
            "sequence_length_observed": int(row.sequence_length),
            "pfam_ids": ";".join(pfam),
            "pfam_ids_json": json.dumps(pfam),
            "primary_pfam": primary_pfam,
            "primary_pfam_clan": primary_clan,
            "pfam_domain_count": len(pfam),
            "pfam_seen_in_fit": primary_pfam in fit_pfam,
            "pfam_clan_seen_in_fit": primary_clan in fit_clan,
            "cath_superfamilies_json": json.dumps(cath),
            "primary_cath_superfamily": primary_cath,
            "cath_domain_count": len(cath),
            "cath_seen_in_fit": primary_cath in fit_cath,
            "query_structure_available": True,
            "pdb_entry_id": str(row.representative_pdb_entry_id),
            "asym_id": asym_id,
            "structure_token": token,
            "foldseek_query_name": f"{token}_{asym_id}",
            "domain_source": "Pfam_2026_01_22_HMMER_cut_ga;RCSB_CATH_instance_annotation",
        })
        manifest.append({
            "query_protein_id": str(row.query_id), "pdb_entry_id": str(row.representative_pdb_entry_id),
            "asym_id": asym_id, "structure_token": token, "coordinate_path": str(coordinate),
            "coordinate_bytes": coordinate.stat().st_size, "coordinate_sha256": sha256(coordinate),
        })
        if ordinal % 25 == 0:
            print(json.dumps({"prepared_structures": ordinal, "total": len(cohort)}), flush=True)

    metadata = pd.DataFrame(rows)
    structure_manifest = pd.DataFrame(manifest)
    if len(metadata) != len(cohort) or metadata["query_protein_id"].duplicated().any():
        raise RuntimeError("Phase 28 query metadata grain mismatch")
    if not metadata["query_structure_available"].all():
        raise RuntimeError("A locked Phase 28 query lacks its experimental structure")
    metadata.to_parquet(RESULTS / "external_query_metadata.parquet", index=False, compression="zstd")
    structure_manifest.to_parquet(RESULTS / "external_structure_manifest.parquet", index=False, compression="zstd")
    metadata[["query_protein_id", "pfam_ids"]].rename(columns={"query_protein_id": "protein_id"}).to_csv(
        WORK / "external_query_pfam.tsv", sep="\t", index=False
    )
    summary = {
        "phase": f"{PHASE_ID}A4", "status": "PASS", "queries": len(metadata),
        "structures_downloaded": len(structure_manifest),
        "queries_with_pfam": int(metadata["pfam_domain_count"].gt(0).sum()),
        "queries_with_cath": int(metadata["cath_domain_count"].gt(0).sum()),
        "primary_pfam_seen_in_fit": int(metadata["pfam_seen_in_fit"].sum()),
        "primary_cath_seen_in_fit": int(metadata["cath_seen_in_fit"].sum()),
        "cohort_columns_read": allowed_columns, "truth_columns_read": [],
    }
    (REPORTS / f"phase{PHASE_ID}_structure_domain_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}A4_EXTERNAL_STRUCTURE_DOMAIN_PASS").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

