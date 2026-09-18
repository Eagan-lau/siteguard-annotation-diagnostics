#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from lxml import etree


NS_URI = "https://uniprot.org/uniprot"
N = f"{{{NS_URI}}}"
COMPLETE_EC_RE = re.compile(r"^[1-7]\.[0-9]+\.[0-9]+\.[0-9]+$")
VALID_SEQUENCE_RE = re.compile(r"^[A-Z]+$")
GOLD_ECO = {"ECO:0000269", "ECO:0000303"}
SILVER_ECO = {"ECO:0000305", "ECO:0007744", "ECO:0000312", "ECO:0000250", "ECO:0000255", "ECO:0000256"}
SITE_TYPES = {"active site", "binding site"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def text_of(element: etree._Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    text = " ".join(element.text.split())
    return text or None


def evidence_keys(value: str | None) -> list[str]:
    if not value:
        return []
    return [token for token in value.split() if token]


def evidence_tier(codes: set[str]) -> str:
    if codes & GOLD_ECO:
        return "GOLD"
    if codes & SILVER_ECO:
        return "SILVER"
    if codes:
        return "OTHER"
    return "UNATTRIBUTED"


def ec_levels(ec: str | None) -> tuple[str | None, str | None, str | None, str | None]:
    if not ec or not COMPLETE_EC_RE.match(ec):
        return None, None, None, None
    parts = ec.split(".")
    return parts[0], ".".join(parts[:2]), ".".join(parts[:3]), ec


def collect_evidence(entry: etree._Element) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    reference_pubmed: dict[str, list[str]] = {}
    for reference in entry.findall(N + "reference"):
        key = reference.get("key", "")
        pmids = [node.get("id") for node in reference.findall(f".//{N}dbReference") if node.get("type") == "PubMed" and node.get("id")]
        reference_pubmed[key] = sorted(set(pmids))

    evidence: dict[str, dict[str, Any]] = {}
    for node in entry.findall(N + "evidence"):
        key = node.get("key", "")
        code = node.get("type", "")
        sources: list[str] = []
        pmids: list[str] = []
        source = node.find(N + "source")
        if source is not None:
            ref = source.get("ref")
            if ref:
                sources.append(f"entry_reference:{ref}")
                pmids.extend(reference_pubmed.get(ref, []))
            dbref = source.find(N + "dbReference")
            if dbref is not None:
                db_type, db_id = dbref.get("type"), dbref.get("id")
                if db_type and db_id:
                    sources.append(f"{db_type}:{db_id}")
                    if db_type == "PubMed":
                        pmids.append(db_id)
        imported = node.find(N + "importedFrom")
        if imported is not None:
            dbref = imported.find(N + "dbReference")
            if dbref is not None and dbref.get("type") and dbref.get("id"):
                sources.append(f"imported:{dbref.get('type')}:{dbref.get('id')}")
        evidence[key] = {"eco": code, "sources": sorted(set(sources)), "pubmed": sorted(set(pmids))}
    return evidence, reference_pubmed


def resolve_evidence(keys: list[str], evidence: dict[str, dict[str, Any]]) -> tuple[set[str], list[str], list[str]]:
    codes: set[str] = set()
    sources: set[str] = set()
    pmids: set[str] = set()
    for key in keys:
        item = evidence.get(key)
        if not item:
            continue
        if item["eco"]:
            codes.add(item["eco"])
        sources.update(item["sources"])
        pmids.update(item["pubmed"])
    return codes, sorted(sources), sorted(pmids)


def direct_dbrefs(entry: etree._Element, db_type: str) -> list[etree._Element]:
    return [node for node in entry.findall(N + "dbReference") if node.get("type") == db_type]


def dbref_payload(nodes: list[etree._Element]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for node in nodes:
        properties = {prop.get("type", ""): prop.get("value", "") for prop in node.findall(N + "property")}
        payload.append({"id": node.get("id"), "properties": properties})
    return payload


def parse_location(feature: etree._Element) -> tuple[int | None, int | None, str, str | None]:
    location = feature.find(N + "location")
    if location is None:
        return None, None, "missing", None
    sequence_ref = location.get("sequence")
    position = location.find(N + "position")
    if position is not None:
        raw = position.get("position")
        return (int(raw) if raw else None), (int(raw) if raw else None), position.get("status", "certain"), sequence_ref
    begin, end = location.find(N + "begin"), location.find(N + "end")
    begin_value = int(begin.get("position")) if begin is not None and begin.get("position") else None
    end_value = int(end.get("position")) if end is not None and end.get("position") else None
    statuses = [node.get("status", "certain") for node in (begin, end) if node is not None]
    status = ";".join(sorted(set(statuses))) if statuses else "missing"
    return begin_value, end_value, status, sequence_ref


def parse_entry(entry: etree._Element, af_accessions: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    accessions = [text_of(node) for node in entry.findall(N + "accession")]
    accessions = [value for value in accessions if value]
    accession = accessions[0]
    entry_name = text_of(entry.find(N + "name"))
    evidence, _ = collect_evidence(entry)

    sequence_node = entry.find(N + "sequence")
    sequence = "".join((sequence_node.text or "").split()) if sequence_node is not None else ""
    sequence_length = int(sequence_node.get("length", "0")) if sequence_node is not None else 0
    sequence_valid = bool(sequence) and bool(VALID_SEQUENCE_RE.match(sequence)) and len(sequence) == sequence_length
    fragment_status = sequence_node.get("fragment") if sequence_node is not None else None

    protein = entry.find(N + "protein")
    full_name = None
    if protein is not None:
        full_name = text_of(protein.find(f"{N}recommendedName/{N}fullName")) or text_of(protein.find(f"{N}submittedName/{N}fullName"))

    organism = entry.find(N + "organism")
    species = None
    taxonomy_id = None
    lineage: list[str] = []
    if organism is not None:
        for name in organism.findall(N + "name"):
            if name.get("type") == "scientific":
                species = text_of(name)
                break
        for dbref in organism.findall(N + "dbReference"):
            if dbref.get("type") == "NCBI Taxonomy":
                taxonomy_id = dbref.get("id")
        lineage = [text_of(node) for node in organism.findall(f"{N}lineage/{N}taxon") if text_of(node)]

    gene_names: list[dict[str, str]] = []
    for gene in entry.findall(N + "gene"):
        for name in gene.findall(N + "name"):
            if text_of(name):
                gene_names.append({"type": name.get("type", ""), "name": text_of(name) or ""})

    ec_records: list[dict[str, Any]] = []
    for ec_node in entry.findall(f".//{N}ecNumber"):
        ec = text_of(ec_node)
        if not ec:
            continue
        keys = evidence_keys(ec_node.get("evidence"))
        codes, sources, pmids = resolve_evidence(keys, evidence)
        ec_records.append({"ec": ec, "evidence_keys": keys, "eco": sorted(codes), "sources": sources, "pubmed": pmids})
    ec_by_id: dict[str, dict[str, set[str]]] = {}
    for record in ec_records:
        bucket = ec_by_id.setdefault(record["ec"], {"eco": set(), "sources": set(), "pubmed": set()})
        bucket["eco"].update(record["eco"])
        bucket["sources"].update(record["sources"])
        bucket["pubmed"].update(record["pubmed"])
    entry_ecs = sorted(ec_by_id)
    complete_ecs = [ec for ec in entry_ecs if COMPLETE_EC_RE.match(ec)]

    cofactors: list[dict[str, Any]] = []
    for comment in entry.findall(N + "comment"):
        if comment.get("type") != "cofactor":
            continue
        for cofactor in comment.findall(N + "cofactor"):
            name = text_of(cofactor.find(N + "name"))
            refs = [{"type": node.get("type"), "id": node.get("id")} for node in cofactor.findall(N + "dbReference")]
            codes, sources, pmids = resolve_evidence(evidence_keys(cofactor.get("evidence")), evidence)
            cofactors.append({"name": name, "references": refs, "eco": sorted(codes), "sources": sources, "pubmed": pmids})

    activities: list[dict[str, Any]] = []
    catalytic_comments = [node for node in entry.findall(N + "comment") if node.get("type") == "catalytic activity"]
    for index, comment in enumerate(catalytic_comments, start=1):
        reaction = comment.find(N + "reaction")
        if reaction is None:
            continue
        reaction_text = text_of(reaction.find(N + "text"))
        refs = reaction.findall(N + "dbReference")
        rhea_ids = sorted({node.get("id") for node in refs if node.get("type") == "Rhea" and node.get("id")})
        reaction_ecs = sorted({node.get("id") for node in refs if node.get("type") == "EC" and node.get("id")})
        ec_candidates = reaction_ecs or (entry_ecs if len(entry_ecs) == 1 else [])
        raw_ec = ec_candidates[0] if len(ec_candidates) == 1 else None
        assignment_basis = "reaction_db_reference" if len(reaction_ecs) == 1 else ("single_entry_ec_fallback" if raw_ec else "ambiguous_or_missing")
        keys = evidence_keys(reaction.get("evidence"))
        codes, sources, pmids = resolve_evidence(keys, evidence)
        if raw_ec and raw_ec in ec_by_id:
            codes.update(ec_by_id[raw_ec]["eco"])
            sources = sorted(set(sources) | ec_by_id[raw_ec]["sources"])
            pmids = sorted(set(pmids) | ec_by_id[raw_ec]["pubmed"])
        directions = [node.get("direction") for node in comment.findall(N + "physiologicalReaction") if node.get("direction")]
        ec_l1, ec_l2, ec_l3, ec_l4 = ec_levels(raw_ec)
        signature = rhea_ids[0] if rhea_ids else (raw_ec or f"REACTION_{index}")
        activities.append({
            "activity_id": f"{accession}::RAW::{signature}::{index}",
            "protein_id": accession,
            "activity_index": index,
            "activity_source": "UNIPROT_CATALYTIC_ACTIVITY",
            "raw_ec": raw_ec,
            "ec_l1": ec_l1,
            "ec_l2": ec_l2,
            "ec_l3": ec_l3,
            "ec_l4": ec_l4,
            "entry_ec_candidates_json": json_text(entry_ecs),
            "reaction_ec_candidates_json": json_text(reaction_ecs),
            "ec_assignment_basis": assignment_basis,
            "raw_rhea_ids_json": json_text(rhea_ids),
            "canonical_rhea": None,
            "rhea_direction_group": None,
            "rhea_relationship_status": "PENDING_PHASE_2",
            "reaction_text": reaction_text,
            "physiological_directions_json": json_text(sorted(set(directions))),
            "substrates_json": None,
            "products_json": None,
            "reaction_smiles": None,
            "cofactors_json": json_text(cofactors),
            "evidence_keys_json": json_text(keys),
            "evidence_eco_json": json_text(sorted(codes)),
            "evidence_sources_json": json_text(sources),
            "pubmed_ids_json": json_text(pmids),
            "evidence_tier": evidence_tier(codes),
            "source_release": "UniProtKB/Swiss-Prot 2026_02",
            "annotation_version": entry.get("version"),
            "annotation_modified": entry.get("modified"),
            "obsolete_replacement_status": "PENDING_PHASE_2",
            "query_use_policy": "GROUND_TRUTH_ONLY",
        })

    if not activities:
        for index, ec in enumerate(entry_ecs, start=1):
            bucket = ec_by_id[ec]
            codes = set(bucket["eco"])
            ec_l1, ec_l2, ec_l3, ec_l4 = ec_levels(ec)
            activities.append({
                "activity_id": f"{accession}::EC_ONLY::{ec}::{index}",
                "protein_id": accession,
                "activity_index": index,
                "activity_source": "UNIPROT_EC_ONLY",
                "raw_ec": ec,
                "ec_l1": ec_l1,
                "ec_l2": ec_l2,
                "ec_l3": ec_l3,
                "ec_l4": ec_l4,
                "entry_ec_candidates_json": json_text(entry_ecs),
                "reaction_ec_candidates_json": "[]",
                "ec_assignment_basis": "entry_ec",
                "raw_rhea_ids_json": "[]",
                "canonical_rhea": None,
                "rhea_direction_group": None,
                "rhea_relationship_status": "NO_RAW_RHEA",
                "reaction_text": None,
                "physiological_directions_json": "[]",
                "substrates_json": None,
                "products_json": None,
                "reaction_smiles": None,
                "cofactors_json": json_text(cofactors),
                "evidence_keys_json": "[]",
                "evidence_eco_json": json_text(sorted(codes)),
                "evidence_sources_json": json_text(sorted(bucket["sources"])),
                "pubmed_ids_json": json_text(sorted(bucket["pubmed"])),
                "evidence_tier": evidence_tier(codes),
                "source_release": "UniProtKB/Swiss-Prot 2026_02",
                "annotation_version": entry.get("version"),
                "annotation_modified": entry.get("modified"),
                "obsolete_replacement_status": "PENDING_PHASE_2",
                "query_use_policy": "GROUND_TRUTH_ONLY",
            })

    uniprot_sites: list[dict[str, Any]] = []
    single_activity_id = activities[0]["activity_id"] if len(activities) == 1 else None
    for feature_index, feature in enumerate(entry.findall(N + "feature"), start=1):
        feature_type = feature.get("type")
        if feature_type not in SITE_TYPES:
            continue
        begin, end, location_status, sequence_ref = parse_location(feature)
        if begin is None or begin != end or sequence_ref:
            continue
        keys = evidence_keys(feature.get("evidence"))
        codes, sources, pmids = resolve_evidence(keys, evidence)
        tier = evidence_tier(codes)
        ligand = feature.find(N + "ligand")
        ligand_name = text_of(ligand.find(N + "name")) if ligand is not None else None
        ligand_ref = ligand.find(N + "dbReference") if ligand is not None else None
        ligand_db = f"{ligand_ref.get('type')}:{ligand_ref.get('id')}" if ligand_ref is not None and ligand_ref.get("type") and ligand_ref.get("id") else None
        if feature_type == "active site" and tier == "GOLD":
            confidence = "TIER_2_UNIPROT_EXPERIMENTAL_ACTIVE_SITE"
            in_site_cohort = True
        elif feature_type == "binding site" and tier == "GOLD" and (ligand_name or ligand_db):
            confidence = "TIER_3_UNIPROT_EXPERIMENTAL_MECHANISTIC_BINDING"
            in_site_cohort = True
        else:
            confidence = "TIER_4_REVIEWED_WEAK_OR_INFERRED"
            in_site_cohort = False
        residue_type = sequence[begin - 1] if 1 <= begin <= len(sequence) else None
        uniprot_sites.append({
            "catalytic_site_id": f"UP::{accession}::{feature_type.replace(' ', '_')}::{begin}::{feature_index}",
            "protein_id": accession,
            "activity_id": single_activity_id,
            "activity_link_status": "UNIQUE_PROTEIN_ACTIVITY" if single_activity_id else "AMBIGUOUS_OR_NO_ACTIVITY",
            "site_source": "UniProtKB/Swiss-Prot",
            "mcsa_entry_id": None,
            "mcsa_residue_id": None,
            "uniprot_feature_type": feature_type,
            "residue_number": begin,
            "residue_type": residue_type,
            "catalytic_role": feature.get("description"),
            "roles_json": "[]",
            "functional_location": None,
            "metal_cofactor_role": ligand_name,
            "ligand_database_id": ligand_db,
            "pdb_id": None,
            "pdb_chain": None,
            "pdb_residue_number": None,
            "sifts_uniprot_residue": begin,
            "location_status": location_status,
            "evidence_eco_json": json_text(sorted(codes)),
            "evidence_sources_json": json_text(sources),
            "pubmed_ids_json": json_text(pmids),
            "site_confidence": confidence,
            "included_in_site_resolved_cohort": in_site_cohort,
            "source_version": "UniProtKB/Swiss-Prot 2026_02",
            "query_use_policy": "GROUND_TRUTH_ONLY",
        })

    pdb_payload = dbref_payload(direct_dbrefs(entry, "PDB"))
    pfam_payload = dbref_payload(direct_dbrefs(entry, "Pfam"))
    interpro_payload = dbref_payload(direct_dbrefs(entry, "InterPro"))
    protein_row = {
        "protein_id": accession,
        "uniprot_accession": accession,
        "secondary_accessions_json": json_text(accessions[1:]),
        "entry_name": entry_name,
        "protein_name": full_name,
        "gene_names_json": json_text(gene_names),
        "sequence": sequence,
        "sequence_version": int(sequence_node.get("version")) if sequence_node is not None and sequence_node.get("version") else None,
        "sequence_checksum": sequence_node.get("checksum") if sequence_node is not None else None,
        "taxonomy_id": taxonomy_id,
        "species": species,
        "lineage_json": json_text(lineage),
        "reviewed_status": "Swiss-Prot",
        "fragment_status": fragment_status or "complete",
        "length": sequence_length,
        "protein_existence": entry.find(N + "proteinExistence").get("type") if entry.find(N + "proteinExistence") is not None else None,
        "annotation_score": None,
        "sequence_valid": sequence_valid,
        "alphafold_availability": accession in af_accessions,
        "alphafold_model_version": 6 if accession in af_accessions else None,
        "pdb_availability": bool(pdb_payload),
        "pdb_entries_json": json_text(pdb_payload),
        "pfam_domains_json": json_text(pfam_payload),
        "pfam_positions_status": "PENDING_PHASE_6_SEQUENCE_DOMAIN_MAPPING",
        "interpro_domains_json": json_text(interpro_payload),
        "cath_assignment_json": "[]",
        "cath_assignment_status": "PENDING_PHASE_6_STRUCTURE_MAPPING",
        "global_structure_quality": None,
        "raw_ec_annotations_json": json_text(ec_records),
        "has_complete_ec": bool(complete_ecs),
        "source_database_version": "UniProtKB/Swiss-Prot 2026_02",
        "entry_version": int(entry.get("version")) if entry.get("version") else None,
        "entry_created": entry.get("created"),
        "entry_modified": entry.get("modified"),
        "sequence_provenance": "QUERY_DERIVED",
    }
    return protein_row, activities, uniprot_sites


def load_mcsa_sites(mcsa_path: Path, activities: pd.DataFrame, protein_ids: set[str]) -> tuple[list[dict[str, Any]], Counter]:
    entries = json.loads((mcsa_path / "mcsa_entries_full.json").read_text(encoding="utf-8"))
    residues = json.loads((mcsa_path / "mcsa_residues_full.json").read_text(encoding="utf-8"))
    residue_to_entry: dict[int, dict[str, Any]] = {}
    for entry in entries:
        for residue in entry.get("residues", []):
            residue_to_entry[int(residue["mcsa_id"])] = entry

    protein_activities: dict[str, list[dict[str, Any]]] = defaultdict(list)
    protein_ec_activities: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in activities.to_dict("records"):
        protein_activities[row["protein_id"]].append(row)
        if row.get("raw_ec"):
            protein_ec_activities[(row["protein_id"], row["raw_ec"])].append(row["activity_id"])

    rows: list[dict[str, Any]] = []
    counts: Counter = Counter()
    seen: set[tuple[Any, ...]] = set()
    for residue in residues:
        residue_id = int(residue["mcsa_id"])
        entry = residue_to_entry.get(residue_id, {})
        entry_id = entry.get("mcsa_id")
        entry_ecs = [str(ec) for ec in entry.get("all_ecs", [])]
        reference_sequences = [item for item in residue.get("residue_sequences", []) if item.get("is_reference")]
        reference_chains = [item for item in residue.get("residue_chains", []) if item.get("is_reference")]
        chain = reference_chains[0] if reference_chains else (residue.get("residue_chains", [None])[0] if residue.get("residue_chains") else None)
        for sequence_site in reference_sequences:
            accession = sequence_site.get("uniprot_id")
            position = sequence_site.get("resid")
            residue_type = sequence_site.get("code")
            if not accession or position is None:
                continue
            counts["mcsa_reference_residue_mappings"] += 1
            if accession not in protein_ids:
                counts["mcsa_accession_absent_current_swissprot"] += 1
                continue
            activity_candidates: set[str] = set()
            for ec in entry_ecs:
                activity_candidates.update(protein_ec_activities.get((accession, ec), []))
            if len(activity_candidates) == 1:
                activity_id = next(iter(activity_candidates))
                link_status = "UNIQUE_EC_MATCH"
            elif len(protein_activities.get(accession, [])) == 1:
                activity_id = protein_activities[accession][0]["activity_id"]
                link_status = "UNIQUE_PROTEIN_ACTIVITY_FALLBACK"
            else:
                activity_id = None
                link_status = "AMBIGUOUS_OR_NO_ACTIVITY"
            key = (entry_id, residue_id, accession, int(position), residue_type)
            if key in seen:
                counts["mcsa_duplicate_reference_mapping_removed"] += 1
                continue
            seen.add(key)
            roles = residue.get("roles", [])
            rows.append({
                "catalytic_site_id": f"MCSA::{entry_id}::{residue_id}::{accession}::{position}",
                "protein_id": accession,
                "activity_id": activity_id,
                "activity_link_status": link_status,
                "site_source": "M-CSA",
                "mcsa_entry_id": f"M{int(entry_id):04d}" if entry_id is not None else None,
                "mcsa_residue_id": residue_id,
                "uniprot_feature_type": None,
                "residue_number": int(position),
                "residue_type": residue_type,
                "catalytic_role": residue.get("roles_summary"),
                "roles_json": json_text(roles),
                "functional_location": residue.get("function_location_abv") or residue.get("main_annotation"),
                "metal_cofactor_role": None,
                "ligand_database_id": None,
                "pdb_id": chain.get("pdb_id") if chain else None,
                "pdb_chain": chain.get("chain_name") if chain else None,
                "pdb_residue_number": chain.get("auth_resid") if chain else None,
                "sifts_uniprot_residue": int(position),
                "location_status": "certain",
                "evidence_eco_json": "[]",
                "evidence_sources_json": json_text(["M-CSA manually curated reference residue"]),
                "pubmed_ids_json": "[]",
                "site_confidence": "TIER_1_MCSA_MANUALLY_CURATED",
                "included_in_site_resolved_cohort": True,
                "source_version": "M-CSA snapshot 2026-08-18",
                "query_use_policy": "GROUND_TRUTH_ONLY",
            })
            counts["mcsa_rows_retained"] += 1
    counts["mcsa_entries"] = len(entries)
    counts["mcsa_residue_objects"] = len(residues)
    return rows, counts


def independent_audit(xml_path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    targets = set(sample["protein_id"])
    observed: dict[str, dict[str, set[str]]] = {}
    with gzip.open(xml_path, "rb") as handle:
        context = etree.iterparse(handle, events=("end",), tag=N + "entry", huge_tree=True)
        for _, entry in context:
            accession_node = entry.find(N + "accession")
            accession = text_of(accession_node)
            if accession in targets:
                ecs = {text_of(node) for node in entry.findall(f".//{N}ecNumber") if text_of(node)}
                rheas = {node.get("id") for node in entry.findall(f".//{N}comment[@type='catalytic activity']/{N}reaction/{N}dbReference") if node.get("type") == "Rhea" and node.get("id")}
                ecos = {node.get("type") for node in entry.findall(N + "evidence") if node.get("type")}
                observed[accession] = {"ec": ecs, "rhea": rheas, "eco": ecos}
            entry.clear()
            parent = entry.getparent()
            if parent is not None:
                while entry.getprevious() is not None:
                    del parent[0]
            if len(observed) == len(targets):
                break

    audit_rows: list[dict[str, Any]] = []
    for row in sample.to_dict("records"):
        raw = observed.get(row["protein_id"], {"ec": set(), "rhea": set(), "eco": set()})
        expected_rhea = set(json.loads(row["raw_rhea_ids_json"]))
        expected_eco = set(json.loads(row["evidence_eco_json"]))
        ec_pass = bool(row.get("raw_ec")) and row["raw_ec"] in raw["ec"]
        rhea_pass = bool(expected_rhea) and expected_rhea <= raw["rhea"]
        evidence_pass = bool(expected_eco & GOLD_ECO) and bool(expected_eco <= raw["eco"])
        audit_rows.append({
            "activity_id": row["activity_id"],
            "protein_id": row["protein_id"],
            "raw_ec": row["raw_ec"],
            "raw_rhea_ids_json": row["raw_rhea_ids_json"],
            "evidence_eco_json": row["evidence_eco_json"],
            "raw_entry_found": row["protein_id"] in observed,
            "ec_reparse_pass": ec_pass,
            "rhea_reparse_pass": rhea_pass,
            "evidence_reparse_pass": evidence_pass,
            "audit_pass": row["protein_id"] in observed and ec_pass and rhea_pass and evidence_pass,
        })
    return pd.DataFrame(audit_rows)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False, engine="pyarrow", compression="zstd")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-proteins", type=int, required=True)
    parser.add_argument("--audit-sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    started = utc_now()
    project = args.project_root.resolve()
    source = args.source_root.resolve()
    processed = project / "data" / "processed"
    reports = project / "reports"
    checkpoints = project / "checkpoints"
    for path in [processed, reports, checkpoints]:
        path.mkdir(parents=True, exist_ok=True)

    xml_path = source / "data" / "raw" / "uniprot" / "current_2026_02" / "uniprot_sprot.xml.gz"
    af_index = source / "data" / "raw" / "alphafold" / "bulk_swissprot_v6" / "afdb_bulk_file_index.parquet"
    mcsa_path = source / "data" / "raw" / "mcsa" / "snapshot_2026-08-18"
    raw_inventory = source / "data" / "manifests" / "resource_inventory.tsv"
    rules_path = project / "configs" / "evidence_rules_v4.yaml"
    raw_hash_before = sha256_file(raw_inventory)
    rules_hash = sha256_file(rules_path)

    af_table = pq.read_table(af_index, columns=["uniprot_accession", "model_version"])
    af_accessions = set(af_table.column("uniprot_accession").to_pylist())

    proteins: list[dict[str, Any]] = []
    activities: list[dict[str, Any]] = []
    uniprot_sites: list[dict[str, Any]] = []
    with gzip.open(xml_path, "rb") as handle:
        context = etree.iterparse(handle, events=("end",), tag=N + "entry", huge_tree=True)
        for index, (_, entry) in enumerate(context, start=1):
            protein, entry_activities, sites = parse_entry(entry, af_accessions)
            proteins.append(protein)
            activities.extend(entry_activities)
            uniprot_sites.extend(sites)
            if index % 50000 == 0:
                print(f"parsed_entries={index} activities={len(activities)} uniprot_sites={len(uniprot_sites)}", flush=True)
            entry.clear()
            parent = entry.getparent()
            if parent is not None:
                while entry.getprevious() is not None:
                    del parent[0]

    protein_df = pd.DataFrame(proteins)
    activity_df = pd.DataFrame(activities)
    protein_ids = set(protein_df["protein_id"])
    mcsa_sites, mcsa_counts = load_mcsa_sites(mcsa_path, activity_df, protein_ids)
    site_df = pd.DataFrame(uniprot_sites + mcsa_sites)

    activity_counts = activity_df.groupby("protein_id")["activity_id"].nunique().to_dict()
    gold_counts = activity_df[activity_df["evidence_tier"] == "GOLD"].groupby("protein_id")["activity_id"].nunique().to_dict()
    protein_df["documented_activity_count"] = protein_df["protein_id"].map(activity_counts).fillna(0).astype("int32")
    protein_df["gold_activity_count"] = protein_df["protein_id"].map(gold_counts).fillna(0).astype("int32")

    eligible_proteins = protein_df[
        protein_df["sequence_valid"]
        & (protein_df["fragment_status"] == "complete")
        & protein_df["has_complete_ec"]
    ][["protein_id", "documented_activity_count", "gold_activity_count"]]
    eligible_ids = set(eligible_proteins["protein_id"])
    has_raw_rhea = activity_df["raw_rhea_ids_json"] != "[]"
    has_complete_ec = activity_df["ec_l4"].notna()
    expanded_df = activity_df[
        activity_df["protein_id"].isin(eligible_ids)
        & activity_df["evidence_tier"].isin(["GOLD", "SILVER"])
        & has_complete_ec
    ].copy()
    single_ids = set(eligible_proteins[eligible_proteins["documented_activity_count"] == 1]["protein_id"])
    single_gold_df = activity_df[
        activity_df["protein_id"].isin(single_ids)
        & (activity_df["evidence_tier"] == "GOLD")
        & has_complete_ec
        & has_raw_rhea
    ].copy()
    multifunctional_ids = set(eligible_proteins[eligible_proteins["documented_activity_count"] >= 2]["protein_id"])
    multifunctional_df = activity_df[activity_df["protein_id"].isin(multifunctional_ids)].copy()
    multifunctional_df["documented_activity_count"] = multifunctional_df["protein_id"].map(activity_counts).astype("int32")

    output_paths = {
        "protein_table": processed / "protein_table.parquet",
        "activity_table": processed / "activity_table.parquet",
        "catalytic_site_table": processed / "catalytic_site_table.parquet",
        "single_documented_activity_gold": processed / "single_documented_activity_gold.parquet",
        "expanded_gold_silver": processed / "expanded_gold_silver.parquet",
        "multifunctional_challenge": processed / "multifunctional_challenge.parquet",
    }
    write_parquet(protein_df, output_paths["protein_table"])
    write_parquet(activity_df, output_paths["activity_table"])
    write_parquet(site_df, output_paths["catalytic_site_table"])
    write_parquet(single_gold_df, output_paths["single_documented_activity_gold"])
    write_parquet(expanded_df, output_paths["expanded_gold_silver"])
    write_parquet(multifunctional_df, output_paths["multifunctional_challenge"])

    sample_pool = single_gold_df.copy()
    if len(sample_pool) >= args.audit_sample_size:
        sample = sample_pool.sample(n=args.audit_sample_size, random_state=args.seed).sort_values("activity_id")
    else:
        sample = sample_pool.copy()
    audit_df = independent_audit(xml_path, sample)
    audit_path = reports / "phase01_activity_audit_sample.tsv"
    audit_df.to_csv(audit_path, sep="\t", index=False)

    metric_rows = [
        ("protein", "swissprot_entries", len(protein_df), "all 2026_02 XML entries"),
        ("protein", "sequence_valid", int(protein_df["sequence_valid"].sum()), "uppercase sequence and XML length agreement"),
        ("protein", "fragments", int((protein_df["fragment_status"] != "complete").sum()), "excluded from Gold cohorts"),
        ("protein", "complete_ec", int(protein_df["has_complete_ec"].sum()), "at least one four-level EC"),
        ("activity", "all_rows", len(activity_df), "catalytic-activity comments or EC-only records"),
        ("activity", "gold_rows", int((activity_df["evidence_tier"] == "GOLD").sum()), "ECO:0000269 or ECO:0000303"),
        ("activity", "silver_rows", int((activity_df["evidence_tier"] == "SILVER").sum()), "predefined curator/import/homology evidence"),
        ("activity", "raw_rhea_rows", int(has_raw_rhea.sum()), "contains at least one raw Rhea identifier"),
        ("cohort", "expanded_gold_silver", len(expanded_df), "eligible proteins and Gold/Silver complete EC"),
        ("cohort", "single_documented_activity_gold", len(single_gold_df), "one documented activity, Gold evidence, complete EC and raw Rhea"),
        ("cohort", "multifunctional_activity_rows", len(multifunctional_df), "proteins with at least two documented activity rows"),
        ("cohort", "multifunctional_proteins", len(multifunctional_ids), "eligible proteins with at least two documented activity rows"),
        ("site", "uniprot_site_rows", len(uniprot_sites), "single-position active/binding features"),
        ("site", "mcsa_site_rows", len(mcsa_sites), "manually curated reference residues mapped to current Swiss-Prot"),
        ("site", "site_resolved_rows", int(site_df["included_in_site_resolved_cohort"].sum()), "predefined Tier 1-3 rows"),
    ]
    for key, value in sorted(mcsa_counts.items()):
        metric_rows.append(("mcsa_qc", key, int(value), "M-CSA raw mapping accounting"))
    filtering_df = pd.DataFrame(metric_rows, columns=["section", "metric", "count", "definition"])
    filtering_path = processed / "evidence_filtering_report.tsv"
    filtering_df.to_csv(filtering_path, sep="\t", index=False)

    provenance_rows = [
        ("protein_table", "sequence and sequence-derived metadata", "QUERY_DERIVED", True),
        ("protein_table", "AlphaFold/PDB/Pfam availability", "QUERY_DERIVED", True),
        ("activity_table", "EC/Rhea/reaction/cofactor/evidence", "GROUND_TRUTH_ONLY", False),
        ("catalytic_site_table", "query true ACT_SITE/M-CSA/binding positions", "GROUND_TRUTH_ONLY", False),
        ("catalytic_site_table", "reference catalytic sites when used on reference side", "REFERENCE_DERIVED", True),
    ]
    pd.DataFrame(provenance_rows, columns=["table", "field_group", "provenance", "allowed_as_query_model_input"]).to_csv(reports / "phase01_feature_provenance.tsv", sep="\t", index=False)

    raw_hash_after = sha256_file(raw_inventory)
    checks = [
        ("checkpoint_00_present", (project / "checkpoints" / "CHECKPOINT_00_PASS").exists(), "Phase 0 prerequisite"),
        ("protein_count_matches_validated_xml", len(protein_df) == args.expected_proteins, f"observed={len(protein_df)} expected={args.expected_proteins}"),
        ("protein_primary_key_unique", protein_df["protein_id"].is_unique, f"duplicates={protein_df['protein_id'].duplicated().sum()}"),
        ("activity_primary_key_unique", activity_df["activity_id"].is_unique, f"duplicates={activity_df['activity_id'].duplicated().sum()}"),
        ("activity_foreign_keys", activity_df["protein_id"].isin(protein_ids).all(), "all activity protein IDs present"),
        ("site_foreign_keys", site_df["protein_id"].isin(protein_ids).all(), "all retained site protein IDs present"),
        ("gold_activity_nonempty", int((activity_df["evidence_tier"] == "GOLD").sum()) >= 5000, f"observed={(activity_df['evidence_tier'] == 'GOLD').sum()}"),
        ("single_gold_nonempty", len(single_gold_df) >= 1000, f"observed={len(single_gold_df)}"),
        ("expanded_training_nonempty", len(expanded_df) >= 10000, f"observed={len(expanded_df)}"),
        ("mcsa_reference_mapping", len(mcsa_sites) >= 3000, f"observed={len(mcsa_sites)}"),
        ("independent_activity_audit_size", len(audit_df) == args.audit_sample_size, f"observed={len(audit_df)} expected={args.audit_sample_size}"),
        ("independent_activity_audit_pass", len(audit_df) == args.audit_sample_size and bool(audit_df["audit_pass"].all()), f"passed={int(audit_df['audit_pass'].sum()) if len(audit_df) else 0}"),
        ("raw_inventory_unchanged", raw_hash_before == raw_hash_after, raw_hash_after),
        ("all_required_outputs_exist", all(path.exists() and path.stat().st_size > 0 for path in output_paths.values()) and filtering_path.exists(), "six Parquet tables plus filtering report"),
    ]
    qc_df = pd.DataFrame([(name, "PASS" if passed else "FAIL", details) for name, passed, details in checks], columns=["check", "status", "details"])
    qc_df.to_csv(reports / "phase01_qc.tsv", sep="\t", index=False)
    failures = qc_df[qc_df["status"] != "PASS"]

    report = [
        "# SiteGuard V4 Phase 1 Report",
        "",
        f"- Started: `{started}`",
        f"- Completed: `{utc_now()}`",
        f"- Decision: `{'PASS' if failures.empty else 'FAIL'}`",
        f"- Evidence-rule SHA-256: `{rules_hash}`",
        f"- Raw-inventory SHA-256 before/after: `{raw_hash_before}` / `{raw_hash_after}`",
        "",
        "## Truth construction",
        "",
        "All V4 tables were rebuilt from the frozen UniProtKB/Swiss-Prot 2026_02 XML, the frozen AlphaFold index and the frozen M-CSA JSON snapshot. No legacy V1-V3 derived truth table was used as input.",
        "",
        "| Metric | Count |",
        "|---|---:|",
    ]
    for _, row in filtering_df.iterrows():
        report.append(f"| {row['section']}::{row['metric']} | {int(row['count']):,} |")
    report += [
        "",
        "## Evidence semantics",
        "",
        "Gold and Silver are evidence qualifications for documented annotations. They are not claims that the database captures the complete biochemical activity spectrum. Raw Rhea identifiers remain uncanonicalized until Phase 2.",
        "",
        "## Independent audit",
        "",
        f"A deterministic sample of {len(audit_df)} Gold activity rows was independently reparsed from the raw XML. Passed: {int(audit_df['audit_pass'].sum()) if len(audit_df) else 0}/{len(audit_df)}.",
        "",
        "## Automated QC",
        "",
        "| Check | Status | Details |",
        "|---|---|---|",
    ]
    for _, row in qc_df.iterrows():
        report.append(f"| {row['check']} | {row['status']} | {row['details']} |")
    report += [
        "",
        "## Leakage status",
        "",
        "Activity labels and true catalytic-site fields are marked `GROUND_TRUTH_ONLY`. They are retained for truth construction and evaluation but are prohibited as query-side model inputs.",
        "",
    ]
    (reports / "PHASE_01_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    summary = {
        "phase": 1,
        "project_version": "V4",
        "status": "PASS" if failures.empty else "FAIL",
        "started_at": started,
        "completed_at": utc_now(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "seed": args.seed,
        "counts": {row[1]: int(row[2]) for row in metric_rows},
        "raw_inventory_sha256": raw_hash_after,
        "evidence_rules_sha256": rules_hash,
        "qc_failures": failures.to_dict("records"),
        "outputs": {key: str(path) for key, path in output_paths.items()},
    }
    (reports / "phase01_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checkpoint = checkpoints / "CHECKPOINT_01_PASS"
    if not failures.empty:
        checkpoint.unlink(missing_ok=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2
    checkpoint.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
