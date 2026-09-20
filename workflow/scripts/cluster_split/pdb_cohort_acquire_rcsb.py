#!/usr/bin/env python3
"""Acquire the frozen post-2026-02 RCSB protein-EC temporal cohort."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "/globalsc/ulg/plgen/yugenliu/SiteGuard/V4"))
PHASE_ID = int(os.environ.get("RCSB_EXTERNAL_PHASE", "27"))
if PHASE_ID == 27:
    PROTOCOL = ROOT / "PROJECT/Phase27_RCSB_temporal_external_blind_protocol.md"
    LOWER_DATE = "2026-06-10"
    UPPER_DATE: str | None = None
    QUERY_PREFIX = "RCSBSEQ::"
elif PHASE_ID == 28:
    PROTOCOL = ROOT / "PROJECT/Phase28_RCSB_structural_external_blind_protocol.md"
    LOWER_DATE = "2025-01-01"
    UPPER_DATE = "2026-06-10"
    QUERY_PREFIX = "RCSB28::"
elif PHASE_ID == 30:
    PROTOCOL = ROOT / "PROJECT/Phase30_RCSB_retrospective_external_blind_protocol.md"
    LOWER_DATE = "2022-01-01"
    UPPER_DATE = "2025-01-01"
    QUERY_PREFIX = "RCSB30::"
elif PHASE_ID == 31:
    PROTOCOL = ROOT / "PROJECT/Phase31_HIT_anchored_external_blind_protocol.md"
    LOWER_DATE = "2019-01-01"
    UPPER_DATE = "2022-01-01"
    QUERY_PREFIX = "RCSB31::"
elif PHASE_ID == 32:
    PROTOCOL = ROOT / "PROJECT/Phase32_tool_training_clean_external_blind_protocol.md"
    LOWER_DATE = "2016-01-01"
    UPPER_DATE = "2019-01-01"
    QUERY_PREFIX = "RCSB32::"
else:
    raise RuntimeError(f"Unsupported RCSB external phase: {PHASE_ID}")
DATA = ROOT / f"data/external/rcsb_phase{PHASE_ID}"
RAW = DATA / "raw_metadata"
RESULTS = ROOT / f"results/phase{PHASE_ID}"
REPORTS = ROOT / f"reports/phase{PHASE_ID}_external_blind"
CHECKPOINTS = ROOT / "checkpoints"
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
LOWER_CUTOFF = pd.Timestamp(LOWER_DATE, tz="UTC")
UPPER_CUTOFF = pd.Timestamp(UPPER_DATE, tz="UTC") if UPPER_DATE else None
FULL_EC = re.compile(r"^[1-9]\d*\.\d+\.\d+\.\d+$")
AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZUO]+$")
USER_AGENT = "SiteGuard-academic-temporal-validation/4.0 (RCSB external blind study)"
BATCH = 100 if PHASE_ID == 27 else 200


def fasta_sequence_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    sequence: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sequence:
                    hashes.add(hashlib.sha256("".join(sequence).upper().encode()).hexdigest())
                sequence = []
            else:
                sequence.append(line)
    if sequence:
        hashes.add(hashlib.sha256("".join(sequence).upper().encode()).hexdigest())
    return hashes


def clean_training_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    for chunk in pd.read_csv(path, sep="\t", usecols=["Sequence"], chunksize=50000):
        hashes.update(hashlib.sha256(str(sequence).upper().encode()).hexdigest() for sequence in chunk["Sequence"])
    return hashes


ENTITY_QUERY = """query($ids:[String!]!){polymer_entities(entity_ids:$ids){
rcsb_id
entity_poly{pdbx_seq_one_letter_code_can rcsb_entity_polymer_type rcsb_sample_sequence_length}
rcsb_polymer_entity{pdbx_ec pdbx_mutation rcsb_enzyme_class_combined{depth ec provenance_source}}
rcsb_polymer_entity_container_identifiers{entry_id entity_id asym_ids auth_asym_ids uniprot_ids}
rcsb_polymer_entity_annotation{annotation_id assignment_version provenance_source type}
rcsb_entity_source_organism{ncbi_taxonomy_id scientific_name}
}}"""

ENTRY_QUERY = """query($ids:[String!]!){entries(entry_ids:$ids){
rcsb_id
rcsb_accession_info{initial_release_date}
exptl{method}
rcsb_primary_citation{pdbx_database_id_PubMed pdbx_database_id_DOI title}
}}"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_json(session: requests.Session, url: str, payload: dict[str, object]) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(7):
        try:
            response = session.post(url, json=payload, timeout=(20, 240))
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(min(65 if response.status_code == 429 else 2**attempt, 65))
                continue
            response.raise_for_status()
            body = response.json()
            if body.get("errors"):
                raise RuntimeError(f"RCSB GraphQL errors: {body['errors']}")
            return response
        except (requests.RequestException, OSError, ValueError) as error:
            last_error = error
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"RCSB request failed: {url}") from last_error


def write_gzip_json(path: Path, payload: object) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in frame[["query_id", "sequence"]].itertuples(index=False):
            handle.write(f">{row.query_id}\n")
            for start in range(0, len(row.sequence), 80):
                handle.write(row.sequence[start:start + 80] + "\n")


def main() -> None:
    for directory in (DATA, RAW, RESULTS, REPORTS, CHECKPOINTS):
        directory.mkdir(parents=True, exist_ok=True)
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    if PHASE_ID == 32:
        protocol_lock_path = ROOT / "models/phase32_protocol_lock.json"
        if not protocol_lock_path.is_file():
            raise FileNotFoundError(protocol_lock_path)
        protocol_lock = json.loads(protocol_lock_path.read_text(encoding="utf-8"))
        if protocol_lock.get("status") != "FROZEN_BEFORE_ACQUISITION":
            raise RuntimeError("Phase32 protocol lock is not valid")
        if protocol_lock.get("protocol_sha256") != sha256(PROTOCOL):
            raise RuntimeError("Phase32 protocol changed after freeze")
        expected_script = protocol_lock.get("script_sha256", {}).get("scripts/pdb_cohort_acquire_rcsb.py")
        if expected_script != sha256(Path(__file__)):
            raise RuntimeError("Phase32 acquisition script changed after freeze")
    forbidden = [RESULTS / "external_blind_predictions.parquet", RESULTS / "external_predictions.parquet"]
    if any(path.exists() for path in forbidden):
        raise RuntimeError(f"Phase {PHASE_ID} prediction outputs exist; acquisition must remain blind")
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    search_nodes = [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_accession_info.initial_release_date", "operator": "greater", "value": LOWER_DATE
        }},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "entity_poly.rcsb_entity_polymer_type", "operator": "exact_match", "value": "Protein"
        }},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_polymer_entity.rcsb_enzyme_class_combined.ec", "operator": "exists"
        }},
    ]
    if UPPER_DATE:
        search_nodes.insert(1, {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_accession_info.initial_release_date", "operator": "less", "value": UPPER_DATE
        }})
    search_payload = {
        "query": {
            "type": "group", "logical_operator": "and", "nodes": search_nodes,
        },
        "request_options": {"return_all_hits": True, "results_verbosity": "minimal"},
        "return_type": "polymer_entity",
    }
    search_response = request_json(session, SEARCH_URL, search_payload)
    search_json = search_response.json()
    search_path = RAW / "search_response.json.gz"
    write_gzip_json(search_path, search_json)
    entity_ids = sorted({str(row["identifier"]) for row in search_json.get("result_set", [])})
    if int(search_json.get("total_count", -1)) != len(entity_ids) or not entity_ids:
        raise RuntimeError("RCSB search result count mismatch")

    manifest_rows = [{
        "resource": "search", "batch": 0, "records": len(entity_ids), "path": str(search_path),
        "bytes": search_path.stat().st_size, "sha256": sha256(search_path),
        "response_date": search_response.headers.get("Date", ""), "http_status": search_response.status_code,
    }]
    entities: list[dict[str, object]] = []
    for batch_index, start in enumerate(range(0, len(entity_ids), BATCH), start=1):
        values = entity_ids[start:start + BATCH]
        response = request_json(session, GRAPHQL_URL, {"query": ENTITY_QUERY, "variables": {"ids": values}})
        body = response.json()
        rows = body.get("data", {}).get("polymer_entities", []) or []
        target = RAW / f"polymer_entities_{batch_index:04d}.json.gz"
        write_gzip_json(target, body)
        entities.extend(rows)
        manifest_rows.append({
            "resource": "polymer_entities", "batch": batch_index, "records": len(rows), "path": str(target),
            "bytes": target.stat().st_size, "sha256": sha256(target),
            "response_date": response.headers.get("Date", ""), "http_status": response.status_code,
        })
        time.sleep(0.35)
    if {str(row.get("rcsb_id")) for row in entities} != set(entity_ids):
        raise RuntimeError("RCSB GraphQL entity inventory differs from frozen search identifiers")

    entry_ids = sorted({str(row["rcsb_polymer_entity_container_identifiers"]["entry_id"]) for row in entities})
    entries: list[dict[str, object]] = []
    for batch_index, start in enumerate(range(0, len(entry_ids), BATCH), start=1):
        values = entry_ids[start:start + BATCH]
        response = request_json(session, GRAPHQL_URL, {"query": ENTRY_QUERY, "variables": {"ids": values}})
        body = response.json()
        rows = body.get("data", {}).get("entries", []) or []
        target = RAW / f"entries_{batch_index:04d}.json.gz"
        write_gzip_json(target, body)
        entries.extend(rows)
        manifest_rows.append({
            "resource": "entries", "batch": batch_index, "records": len(rows), "path": str(target),
            "bytes": target.stat().st_size, "sha256": sha256(target),
            "response_date": response.headers.get("Date", ""), "http_status": response.status_code,
        })
        time.sleep(0.35)
    entry_lookup = {str(row["rcsb_id"]): row for row in entries}
    if set(entry_lookup) != set(entry_ids):
        raise RuntimeError("RCSB GraphQL entry inventory incomplete")

    known = set(pd.read_parquet(
        ROOT / "data/processed/protein_table.parquet", columns=["protein_id", "uniprot_accession"]
    ).astype(str).stack())
    known |= set(pd.read_parquet(
        ROOT / "data/reference/activity_reference_library.parquet", columns=["reference_protein_id"]
    )["reference_protein_id"].astype(str))
    phase26_path = ROOT / "results/phase26/sabio_strict_blind_cohort.parquet"
    if phase26_path.exists():
        known |= set(pd.read_parquet(phase26_path, columns=["uniprot_accession"])["uniprot_accession"].astype(str))
    prior_rcsb_entities: set[str] = set()
    prior_rcsb_sequences: set[str] = set()
    prior_phases = (
        [27] if PHASE_ID == 28 else
        [27, 28] if PHASE_ID == 30 else
        [27, 28, 30] if PHASE_ID == 31 else
        [27, 28, 30, 31] if PHASE_ID == 32 else []
    )
    for prior_phase in prior_phases:
        prior_path = ROOT / f"results/phase{prior_phase}/rcsb_candidate_entities.parquet"
        if not prior_path.exists():
            raise FileNotFoundError(f"Phase {PHASE_ID} requires Phase {prior_phase} entity inventory")
        prior = pd.read_parquet(prior_path, columns=["rcsb_entity_id", "sequence_sha256"])
        prior_rcsb_entities |= set(prior["rcsb_entity_id"].astype(str))
        prior_rcsb_sequences |= set(prior["sequence_sha256"].astype(str))
    clean_hashes: set[str] = set()
    hit_hashes: set[str] = set()
    if PHASE_ID in {30, 31, 32}:
        clean_hashes = clean_training_hashes(ROOT / "external_tools/CLEAN/app/data/split100.csv")
        hit_hashes = fasta_sequence_hashes(ROOT / "external_tools/HIT-EC/data/new-28245.fasta")

    rows = []
    for entity in entities:
        entity_id = str(entity["rcsb_id"])
        polymer = entity.get("entity_poly") or {}
        annotation = entity.get("rcsb_polymer_entity") or {}
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        entry_id = str(identifiers.get("entry_id", ""))
        entry = entry_lookup.get(entry_id, {})
        citation = entry.get("rcsb_primary_citation") or {}
        ec_records = annotation.get("rcsb_enzyme_class_combined") or []
        pdb_ec = sorted({
            str(item.get("ec", "")) for item in ec_records
            if item.get("provenance_source") == "PDB Primary Data" and FULL_EC.fullmatch(str(item.get("ec", "")))
        })
        sequence = re.sub(r"\s+", "", str(polymer.get("pdbx_seq_one_letter_code_can") or "").upper())
        uniprot_ids = sorted({str(value) for value in identifiers.get("uniprot_ids") or [] if value})
        pfam = sorted({
            str(item.get("annotation_id")) for item in entity.get("rcsb_polymer_entity_annotation") or []
            if str(item.get("type")) == "Pfam" and str(item.get("annotation_id", "")).startswith("PF")
        })
        organisms = entity.get("rcsb_entity_source_organism") or []
        release = pd.to_datetime((entry.get("rcsb_accession_info") or {}).get("initial_release_date"), utc=True, errors="coerce")
        methods = sorted({str(item.get("method")) for item in entry.get("exptl") or [] if item.get("method")})
        mutation = str(annotation.get("pdbx_mutation") or "").strip()
        in_window = bool(
            pd.notna(release) and release > LOWER_CUTOFF
            and (UPPER_CUTOFF is None or release < UPPER_CUTOFF)
        )
        sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
        rows.append({
            "rcsb_entity_id": entity_id,
            "pdb_entry_id": entry_id,
            "entity_id": str(identifiers.get("entity_id", "")),
            "asym_ids_json": json.dumps(identifiers.get("asym_ids") or []),
            "auth_asym_ids_json": json.dumps(identifiers.get("auth_asym_ids") or []),
            "uniprot_ids_json": json.dumps(uniprot_ids),
            "uniprot_ids": ";".join(uniprot_ids),
            "sequence": sequence,
            "sequence_length": len(sequence),
            "sequence_sha256": sequence_hash,
            "pdb_primary_ec_set_json": json.dumps(pdb_ec),
            "ec_l3_set_json": json.dumps(sorted({".".join(value.split(".")[:3]) for value in pdb_ec})),
            "ec_l4_set_json": json.dumps(pdb_ec),
            "pfam_ids_json": json.dumps(pfam),
            "primary_pfam": pfam[0] if pfam else "__MISSING__",
            "taxonomy_ids_json": json.dumps(sorted({str(item.get("ncbi_taxonomy_id")) for item in organisms if item.get("ncbi_taxonomy_id") is not None})),
            "organisms_json": json.dumps(sorted({str(item.get("scientific_name")) for item in organisms if item.get("scientific_name")})),
            "initial_release_date": release,
            "experimental_methods_json": json.dumps(methods),
            "pubmed_id": str(citation.get("pdbx_database_id_PubMed") or ""),
            "doi": str(citation.get("pdbx_database_id_DOI") or ""),
            "citation_title": str(citation.get("title") or ""),
            "pdbx_mutation": mutation,
            "protein_entity": polymer.get("rcsb_entity_polymer_type") == "Protein",
            "complete_pdb_primary_ec": bool(pdb_ec),
            "sequence_valid": bool(AA.fullmatch(sequence)),
            "length_in_scope": 50 <= len(sequence) <= 5000,
            "mutation_free": mutation == "",
            "citation_present": bool(citation.get("pdbx_database_id_PubMed") or citation.get("pdbx_database_id_DOI")),
            "experimental_method_present": bool(methods),
            "strictly_post_t0": in_window,
            "uniprot_in_frozen_or_prior_external": any(value in known for value in uniprot_ids),
            "phase27_entity_or_sequence": PHASE_ID == 28 and (entity_id in prior_rcsb_entities or sequence_hash in prior_rcsb_sequences),
            "prior_rcsb_entity_or_sequence": entity_id in prior_rcsb_entities or sequence_hash in prior_rcsb_sequences,
            "exact_sequence_in_clean_train": sequence_hash in clean_hashes,
            "exact_sequence_in_hit_ec_disclosed_corpus": sequence_hash in hit_hashes,
        })
    entities_frame = pd.DataFrame(rows)
    entities_frame["presequence_eligible"] = (
        entities_frame["protein_entity"] & entities_frame["complete_pdb_primary_ec"]
        & entities_frame["sequence_valid"] & entities_frame["length_in_scope"]
        & entities_frame["mutation_free"] & entities_frame["citation_present"]
        & entities_frame["experimental_method_present"] & entities_frame["strictly_post_t0"]
        & ~entities_frame["uniprot_in_frozen_or_prior_external"]
        & ~entities_frame["prior_rcsb_entity_or_sequence"]
        & ~entities_frame["exact_sequence_in_clean_train"]
        & ~entities_frame["exact_sequence_in_hit_ec_disclosed_corpus"]
    )
    entities_frame.to_parquet(RESULTS / "rcsb_candidate_entities.parquet", index=False, compression="zstd")

    collapsed_rows = []
    for sequence_hash, group in entities_frame.loc[entities_frame["presequence_eligible"]].groupby("sequence_sha256", observed=True):
        ec4 = sorted({value for payload in group["ec_l4_set_json"] for value in json.loads(payload)})
        ec3 = sorted({value for payload in group["ec_l3_set_json"] for value in json.loads(payload)})
        representative = group.sort_values(["initial_release_date", "rcsb_entity_id"]).iloc[0]
        collapsed_rows.append({
            "query_id": QUERY_PREFIX + str(sequence_hash)[:16],
            "sequence_sha256": sequence_hash,
            "sequence": representative["sequence"],
            "sequence_length": representative["sequence_length"],
            "representative_rcsb_entity_id": representative["rcsb_entity_id"],
            "representative_pdb_entry_id": representative["pdb_entry_id"],
            "representative_entity_id": representative["entity_id"],
            "representative_asym_ids_json": representative["asym_ids_json"],
            "rcsb_entity_ids_json": json.dumps(sorted(group["rcsb_entity_id"].astype(str).unique().tolist())),
            "pdb_entry_ids_json": json.dumps(sorted(group["pdb_entry_id"].astype(str).unique().tolist())),
            "ec_l3": ec3[0] if len(ec3) == 1 else None,
            "ec_l4": ec4[0] if len(ec4) == 1 else None,
            "ec_l3_set_json": json.dumps(ec3),
            "ec_l4_set_json": json.dumps(ec4),
            "ec_l3_label_eligible": len(ec3) == 1,
            "ec_l4_label_eligible": len(ec4) == 1,
            "primary_pfam": representative["primary_pfam"],
            "pfam_ids_json": representative["pfam_ids_json"],
            "taxonomy_ids_json": representative["taxonomy_ids_json"],
            "initial_release_date": group["initial_release_date"].min(),
            "entity_count": group["rcsb_entity_id"].nunique(),
            "entry_count": group["pdb_entry_id"].nunique(),
            "prediction_use_policy": f"PHASE{PHASE_ID}_EXTERNAL_BLIND_EVALUATION_ONLY",
        })
    collapsed = pd.DataFrame(collapsed_rows).sort_values("query_id")
    if collapsed.empty:
        raise RuntimeError(f"No Phase {PHASE_ID} sequence candidates survived frozen metadata filters")
    collapsed.to_parquet(RESULTS / "rcsb_sequence_candidates.parquet", index=False, compression="zstd")
    write_fasta(collapsed, DATA / "rcsb_sequence_candidates.fasta")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(DATA / "raw_sha256.tsv", sep="\t", index=False)
    search_manifest = {
        "phase": f"{PHASE_ID}A0", "status": "PASS", "search_url": SEARCH_URL,
        "data_url": GRAPHQL_URL, "search_payload": search_payload,
        "search_total_count": len(entity_ids), "unique_entries": len(entry_ids),
        "protocol_sha256": sha256(PROTOCOL), "search_response_date": search_response.headers.get("Date", ""),
        "acquired_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    (DATA / "search_manifest.json").write_text(json.dumps(search_manifest, indent=2) + "\n")
    summary = {
        **search_manifest,
        "candidate_entities": len(entities_frame),
        "pdb_primary_complete_ec_entities": int(entities_frame["complete_pdb_primary_ec"].sum()),
        "mutation_free_entities": int(entities_frame["mutation_free"].sum()),
        "citation_present_entities": int(entities_frame["citation_present"].sum()),
        "uniprot_core_or_prior_external_excluded": int(entities_frame["uniprot_in_frozen_or_prior_external"].sum()),
        "prior_rcsb_entity_or_sequence_excluded": int(entities_frame["prior_rcsb_entity_or_sequence"].sum()),
        "exact_clean_training_sequence_excluded": int(entities_frame["exact_sequence_in_clean_train"].sum()),
        "exact_hit_ec_disclosed_sequence_excluded": int(entities_frame["exact_sequence_in_hit_ec_disclosed_corpus"].sum()),
        "clean_training_sequence_hashes": len(clean_hashes),
        "hit_ec_disclosed_sequence_hashes": len(hit_hashes),
        "presequence_eligible_entities": int(entities_frame["presequence_eligible"].sum()),
        "unique_sequence_candidates": len(collapsed),
        "ec_l3_sequence_candidates": int(collapsed["ec_l3_label_eligible"].sum()),
        "ec_l4_sequence_candidates": int(collapsed["ec_l4_label_eligible"].sum()),
        f"phase{PHASE_ID}_predictions_read": False,
    }
    checks = [
        ("protocol_precedes_metadata", PROTOCOL.stat().st_mtime_ns < search_path.stat().st_mtime_ns, f"protocol={PROTOCOL.stat().st_mtime_ns};search={search_path.stat().st_mtime_ns}"),
        ("search_count_complete", len(entity_ids) == int(search_json["total_count"]), len(entity_ids)),
        ("entity_inventory_complete", len(entities) == len(entity_ids), len(entities)),
        ("entry_inventory_complete", len(entries) == len(entry_ids), len(entries)),
        ("frozen_release_window", bool(entities_frame.loc[entities_frame["presequence_eligible"], "strictly_post_t0"].all()), f"{LOWER_DATE}<{UPPER_DATE or 'open'}"),
        ("pdb_primary_ec_only", bool(entities_frame.loc[entities_frame["presequence_eligible"], "complete_pdb_primary_ec"].all()), "PDB Primary Data"),
        ("mutation_free", bool(entities_frame.loc[entities_frame["presequence_eligible"], "mutation_free"].all()), int(entities_frame["presequence_eligible"].sum())),
        ("no_known_uniprot_accessions", not bool(entities_frame.loc[entities_frame["presequence_eligible"], "uniprot_in_frozen_or_prior_external"].any()), "accession exclusion"),
        ("prior_rcsb_entities_excluded", PHASE_ID == 27 or not bool(entities_frame.loc[entities_frame["presequence_eligible"], "prior_rcsb_entity_or_sequence"].any()), len(prior_rcsb_entities)),
        ("clean_exact_training_sequences_excluded", not bool(entities_frame.loc[entities_frame["presequence_eligible"], "exact_sequence_in_clean_train"].any()), len(clean_hashes)),
        ("hit_ec_exact_disclosed_sequences_excluded", not bool(entities_frame.loc[entities_frame["presequence_eligible"], "exact_sequence_in_hit_ec_disclosed_corpus"].any()), len(hit_hashes)),
        (f"no_phase{PHASE_ID}_predictions_read", not any(path.exists() for path in forbidden), forbidden),
    ]
    qc = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    qc.to_csv(REPORTS / f"phase{PHASE_ID}_acquisition_qc.tsv", sep="\t", index=False)
    failures = qc.loc[~qc["passed"].astype(bool), "check"].tolist()
    if failures:
        summary["status"] = "FAIL"
        summary["failures"] = failures
    (REPORTS / f"phase{PHASE_ID}_acquisition_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    if failures:
        raise RuntimeError(f"Phase {PHASE_ID} acquisition failed: {failures}")
    (CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}A0_RCSB_ACQUISITION_PASS").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
