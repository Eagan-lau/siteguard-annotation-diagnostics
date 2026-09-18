#!/usr/bin/env python3
"""Parse label-blind Pfam HMM calls for the locked external cohort."""

from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
SOURCE = Path(os.environ.get("SITEGUARD_SOURCE", "/globalsc/ulg/plgen/yugenliu/SiteGuard"))
WORK = ROOT / "data/interim/phase26_inference"
RESULTS = ROOT / "results/phase26"
CHECKPOINTS = ROOT / "checkpoints"


def clan_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields and fields[0]:
                mapping[fields[0]] = fields[1] if len(fields) > 1 and fields[1] else "NO_CLAN"
    return mapping


def main() -> None:
    checkpoint = CHECKPOINTS / "CHECKPOINT_26A_EXTERNAL_COHORT_LOCKED"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if (RESULTS / "external_predictions.parquet").exists():
        raise RuntimeError("External predictions already exist; domain extraction is frozen")
    domtbl = WORK / "external_pfam.domtblout"
    calls: dict[str, set[str]] = defaultdict(set)
    with domtbl.open("r", encoding="utf-8") as handle:
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

    cohort = pd.read_parquet(RESULTS / "sabio_strict_blind_cohort.parquet")
    clans = clan_map(SOURCE / "data/raw/pfam/current_release_2026_01_22/Pfam-A.clans.tsv.gz")
    lock = json.loads((ROOT / "models/phase25/evidencejudge/errorjudge/errorjudge_operating_point_lock.json").read_text())
    fit_pfam = set(lock["gate_state"]["fit_pfam"])
    fit_clan = set(lock["gate_state"]["fit_pfam_clan"])
    rows = []
    for accession in cohort["uniprot_accession"].astype(str):
        values = sorted(calls.get(accession, set()))
        primary = values[0] if values else "__MISSING__"
        clan = clans.get(primary, "UNASSIGNED") if values else "__MISSING__"
        rows.append({
            "query_protein_id": accession,
            "pfam_ids": ";".join(values),
            "pfam_ids_json": json.dumps(values),
            "primary_pfam": primary,
            "primary_pfam_clan": clan,
            "pfam_domain_count": len(values),
            "pfam_seen_in_fit": primary in fit_pfam,
            "pfam_clan_seen_in_fit": clan in fit_clan,
            "primary_cath_superfamily": "__MISSING__",
            "cath_seen_in_fit": False,
            "query_structure_available": False,
            "domain_source": "Pfam_current_release_2026_01_22_HMMER3_cut_ga",
        })
    metadata = pd.DataFrame(rows)
    if metadata["query_protein_id"].duplicated().any() or len(metadata) != len(cohort):
        raise RuntimeError("External Pfam metadata grain mismatch")
    metadata.to_parquet(RESULTS / "external_query_metadata.parquet", index=False, compression="zstd")
    metadata[["query_protein_id", "pfam_ids"]].rename(columns={"query_protein_id": "protein_id"}).to_csv(
        WORK / "external_query_pfam.tsv", sep="\t", index=False
    )
    summary = {
        "phase": "26A4_PFAM",
        "status": "PASS",
        "queries": len(metadata),
        "queries_with_pfam": int(metadata["pfam_domain_count"].gt(0).sum()),
        "queries_primary_pfam_seen_in_fit": int(metadata["pfam_seen_in_fit"].sum()),
        "queries_primary_pfam_clan_seen_in_fit": int(metadata["pfam_clan_seen_in_fit"].sum()),
        "cath_and_structure_policy": "MISSING_NOT_IMPUTED",
        "phase26_labels_read": False,
        "phase26_predictions_read": False,
    }
    (ROOT / "reports/phase26_external_pfam_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (CHECKPOINTS / "CHECKPOINT_26A4_EXTERNAL_PFAM_PASS").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
