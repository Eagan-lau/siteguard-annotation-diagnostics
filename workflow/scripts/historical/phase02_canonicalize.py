#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator


RHEA_ID_RE = re.compile(r"(?:RHEA:)?([0-9]+)")
EC_RE = re.compile(r"[1-7]\.[0-9]+\.[0-9]+\.[0-9]+")
CHEBI_RE = re.compile(r"CHEBI:[0-9]+")
ARROW_RE = re.compile(r"\s+(?:<=>|=>|=)\s+")
RELEASES = (126, 140, 141)


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


def rhea_int(value: Any) -> int | None:
    match = RHEA_ID_RE.fullmatch(str(value).strip())
    return int(match.group(1)) if match else None


def parse_reaction_flatfile(path: Path) -> dict[int, dict[str, str]]:
    records: dict[int, dict[str, str]] = {}
    current: dict[str, list[str]] = defaultdict(list)
    last_key: str | None = None

    def finish() -> None:
        nonlocal current, last_key
        if not current.get("ENTRY"):
            current = defaultdict(list)
            last_key = None
            return
        rid = rhea_int(current["ENTRY"][0])
        if rid is not None:
            records[rid] = {key: " ".join(values).strip() for key, values in current.items()}
        current = defaultdict(list)
        last_key = None

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line == "///":
                finish()
                continue
            key = line[:12].strip()
            value = line[12:].strip()
            if key:
                current[key].append(value)
                last_key = key
            elif last_key and value:
                current[last_key].append(value)
    finish()
    return records


def equation_sides(equation: str | None) -> tuple[list[str], list[str]]:
    if not equation:
        return [], []
    parts = ARROW_RE.split(equation, maxsplit=1)
    if len(parts) != 2:
        return [], []
    return CHEBI_RE.findall(parts[0]), CHEBI_RE.findall(parts[1])


def chemistry_hash(left: list[str], right: list[str], directionless: bool) -> str | None:
    if not left and not right:
        return None
    left_text, right_text = ",".join(sorted(left)), ",".join(sorted(right))
    if directionless:
        signature = "||".join(sorted([left_text, right_text]))
    else:
        signature = left_text + ">>" + right_text
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def read_two_column(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t", 1)
            if len(fields) == 2:
                values[fields[0].strip()] = fields[1].strip()
    return values


def parse_release(root: Path, release: int) -> tuple[pd.DataFrame, dict[int, dict[str, Any]], set[int]]:
    base = root / f"release_{release}" / "extracted" / str(release)
    tsv = base / "tsv"
    directions = pd.read_csv(tsv / "rhea-directions.tsv", sep="\t", dtype="int64")
    reaction_records = parse_reaction_flatfile(base / "txt" / "rhea-reactions.txt.gz")
    smiles = read_two_column(tsv / "rhea-reaction-smiles.tsv")
    chebi_smiles = read_two_column(tsv / "rhea-chebi-smiles.tsv")
    chebi_names = read_two_column(tsv / "chebiId_name.tsv")
    rhea_ec = pd.read_csv(tsv / "rhea2ec.tsv", sep="\t", dtype=str)
    ec_by_master = rhea_ec.groupby("MASTER_ID")["ID"].apply(lambda values: sorted(set(values.dropna()))).to_dict()
    obsolete = set(pd.read_csv(tsv / "rhea-obsoletes.tsv", sep="\t", dtype="int64")["RHEA_ID"].tolist())

    direction_map: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for row in directions.itertuples(index=False):
        master = int(row.RHEA_ID_MASTER)
        lr, rl, bi = int(row.RHEA_ID_LR), int(row.RHEA_ID_RL), int(row.RHEA_ID_BI)
        for rid, direction in [(master, "MASTER"), (lr, "LR"), (rl, "RL"), (bi, "BI")]:
            if rid in direction_map and direction_map[rid]["master"] != master:
                raise ValueError(f"Rhea direction ID collision in release {release}: {rid}")
            direction_map[rid] = {"master": master, "direction": direction}
        master_record = reaction_records.get(master, {})
        lr_record = reaction_records.get(lr, {})
        equation = master_record.get("EQUATION") or lr_record.get("EQUATION")
        definition = master_record.get("DEFINITION") or lr_record.get("DEFINITION")
        left, right = equation_sides(lr_record.get("EQUATION") or equation)
        participant_ids = sorted(set(left + right))
        rows.append({
            "release": release,
            "canonical_rhea": f"RHEA:{master}",
            "master_id": master,
            "lr_id": lr,
            "rl_id": rl,
            "bi_id": bi,
            "definition": definition,
            "equation": equation,
            "lr_equation": lr_record.get("EQUATION"),
            "reaction_smiles_lr": smiles.get(str(lr)),
            "reaction_smiles_rl": smiles.get(str(rl)),
            "ec_ids_json": json_text(ec_by_master.get(str(master), [])),
            "substrate_chebi_ids_json": json_text(left),
            "product_chebi_ids_json": json_text(right),
            "participant_chebi_ids_json": json_text(participant_ids),
            "participant_names_json": json_text({chebi: chebi_names.get(chebi) for chebi in participant_ids}),
            "participant_smiles_json": json_text({chebi: chebi_smiles.get(chebi) for chebi in participant_ids}),
            "directionless_chemistry_hash": chemistry_hash(left, right, True),
            "directional_chemistry_hash": chemistry_hash(left, right, False),
            "is_obsolete_in_release": master in obsolete,
            "source_version": f"Rhea {release}",
        })
    return pd.DataFrame(rows), direction_map, obsolete


def cross_release_map(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    source_release = int(source["release"].iloc[0])
    target_release = int(target["release"].iloc[0])
    source_records = {int(row.master_id): row._asdict() for row in source.itertuples(index=False)}
    target_records = {int(row.master_id): row._asdict() for row in target.itertuples(index=False)}
    source_by_hash: dict[str, list[int]] = defaultdict(list)
    target_by_hash: dict[str, list[int]] = defaultdict(list)
    for master, row in source_records.items():
        if row["directionless_chemistry_hash"]:
            source_by_hash[row["directionless_chemistry_hash"]].append(master)
    for master, row in target_records.items():
        if row["directionless_chemistry_hash"]:
            target_by_hash[row["directionless_chemistry_hash"]].append(master)

    rows: list[dict[str, Any]] = []
    mapped_targets: set[int] = set()
    for source_master, source_row in sorted(source_records.items()):
        source_hash = source_row["directionless_chemistry_hash"]
        if source_master in target_records:
            target_row = target_records[source_master]
            mapped_targets.add(source_master)
            if source_hash == target_row["directionless_chemistry_hash"]:
                relation = "direction_change" if source_row["directional_chemistry_hash"] != target_row["directional_chemistry_hash"] else "stable_id_stable_chemistry"
            else:
                relation = "stable_id_chemistry_revision"
            candidates = [source_master]
        else:
            candidates = sorted(target_by_hash.get(source_hash, [])) if source_hash else []
            if candidates:
                mapped_targets.update(candidates)
                if len(candidates) > 1:
                    relation = "split"
                elif len(source_by_hash.get(source_hash, [])) > 1:
                    relation = "merge"
                else:
                    relation = "chemistry_equivalent_id_change"
            else:
                relation = "obsolete_without_exact_replacement"
                candidates = [None]
        for target_master in candidates:
            target_row = target_records.get(target_master) if target_master is not None else None
            rows.append({
                "source_release": source_release,
                "target_release": target_release,
                "source_canonical_rhea": f"RHEA:{source_master}",
                "target_canonical_rhea": f"RHEA:{target_master}" if target_master is not None else None,
                "relationship": relation,
                "source_directionless_hash": source_hash,
                "target_directionless_hash": target_row["directionless_chemistry_hash"] if target_row else None,
                "source_definition": source_row["definition"],
                "target_definition": target_row["definition"] if target_row else None,
            })
    for target_master, target_row in sorted(target_records.items()):
        if target_master in mapped_targets:
            continue
        target_hash = target_row["directionless_chemistry_hash"]
        source_candidates = source_by_hash.get(target_hash, []) if target_hash else []
        relation = "merge" if len(source_candidates) > 1 else ("chemistry_equivalent_id_change" if source_candidates else "genuinely_novel_chemistry")
        rows.append({
            "source_release": source_release,
            "target_release": target_release,
            "source_canonical_rhea": None,
            "target_canonical_rhea": f"RHEA:{target_master}",
            "relationship": relation,
            "source_directionless_hash": target_hash if source_candidates else None,
            "target_directionless_hash": target_hash,
            "source_definition": None,
            "target_definition": target_row["definition"],
        })
    return pd.DataFrame(rows)


def parse_enzyme(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current: dict[str, list[str]] = defaultdict(list)

    def finish() -> None:
        nonlocal current
        if not current.get("ID"):
            current = defaultdict(list)
            return
        ec = " ".join(current["ID"]).strip()
        description = " ".join(current.get("DE", [])).strip()
        lower = description.lower()
        replacements = sorted(set(EC_RE.findall(description)))
        if "transferred entry" in lower:
            status = "TRANSFERRED"
        elif "deleted entry" in lower:
            status = "DELETED"
        else:
            status = "ACTIVE"
        rows.append({
            "raw_ec": ec,
            "status": status,
            "description": description,
            "replacement_ecs_json": json_text([value for value in replacements if value != ec]),
            "canonical_ec": ([value for value in replacements if value != ec][0] if status == "TRANSFERRED" and len([value for value in replacements if value != ec]) == 1 else (ec if status == "ACTIVE" else None)),
            "source_version": "ENZYME 2026-06-10",
        })
        current = defaultdict(list)

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line == "//":
                finish()
                continue
            if len(line) >= 5 and line[:2].strip():
                current[line[:2]].append(line[5:].strip())
    finish()
    return pd.DataFrame(rows)


def canonicalize_activities(activity: pd.DataFrame, direction_map: dict[int, dict[str, Any]], ec_status: pd.DataFrame, current_reactions: pd.DataFrame) -> pd.DataFrame:
    ec_lookup = ec_status.set_index("raw_ec").to_dict("index")
    reaction_lookup = current_reactions.set_index("canonical_rhea").to_dict("index")
    records: list[dict[str, Any]] = []
    for row in activity.to_dict("records"):
        raw_ec = row.get("raw_ec")
        ec_record = ec_lookup.get(raw_ec)
        if ec_record:
            canonical_ec = ec_record["canonical_ec"]
            if not isinstance(canonical_ec, str) or not canonical_ec:
                canonical_ec = None
            ec_mapping_status = ec_record["status"]
        else:
            canonical_ec = raw_ec if isinstance(raw_ec, str) and EC_RE.fullmatch(raw_ec) else None
            ec_mapping_status = "NOT_IN_ENZYME" if isinstance(raw_ec, str) and raw_ec else "MISSING"
        if canonical_ec:
            parts = canonical_ec.split(".")
            row["ec_l1"] = parts[0]
            row["ec_l2"] = ".".join(parts[:2])
            row["ec_l3"] = ".".join(parts[:3])
            row["ec_l4"] = canonical_ec
        else:
            row["ec_l1"] = row["ec_l2"] = row["ec_l3"] = row["ec_l4"] = None
        row["canonical_ec"] = canonical_ec
        row["ec_mapping_status"] = ec_mapping_status

        raw_rheas = json.loads(row.get("raw_rhea_ids_json") or "[]")
        mapped = []
        unmapped = []
        for value in raw_rheas:
            rid = rhea_int(value)
            if rid is not None and rid in direction_map:
                mapped.append(direction_map[rid])
            else:
                unmapped.append(value)
        masters = sorted({item["master"] for item in mapped})
        directions = sorted({item["direction"] for item in mapped})
        if len(masters) == 1:
            canonical_rhea = f"RHEA:{masters[0]}"
            rhea_status = "CANONICALIZED_CURRENT_RELEASE"
        elif len(masters) > 1:
            canonical_rhea = None
            rhea_status = "MULTIPLE_CANONICAL_MASTERS"
        elif raw_rheas:
            canonical_rhea = None
            rhea_status = "UNMAPPED_OR_OBSOLETE_CURRENT_RELEASE"
        else:
            canonical_rhea = None
            rhea_status = "NO_RAW_RHEA"
        row["canonical_rhea"] = canonical_rhea
        row["rhea_direction_group"] = ";".join(directions) if directions else None
        row["rhea_relationship_status"] = rhea_status
        row["unmapped_raw_rhea_ids_json"] = json_text(unmapped)
        reaction = reaction_lookup.get(canonical_rhea) if canonical_rhea else None
        row["canonical_reaction_definition"] = reaction["definition"] if reaction else None
        row["canonical_reaction_equation"] = reaction["equation"] if reaction else None
        if reaction:
            row["substrates_json"] = reaction["substrate_chebi_ids_json"]
            row["products_json"] = reaction["product_chebi_ids_json"]
            row["reaction_smiles"] = reaction["reaction_smiles_lr"]
        records.append(row)
    return pd.DataFrame(records)


def molecular_features(smiles: str | None, generator: Any, bits: int) -> dict[str, Any]:
    empty_bits = np.zeros(bits, dtype=np.uint8)
    empty_counts = np.zeros(bits, dtype=np.int16)
    if not isinstance(smiles, str) or not smiles:
        return {"valid": False, "molecules": [], "bits": empty_bits, "counts": empty_counts, "mw": 0.0, "logp": 0.0, "tpsa": 0.0, "heavy_atoms": 0}
    molecules = []
    aggregate_bits = empty_bits.copy()
    aggregate_counts = empty_counts.copy()
    mw = logp = tpsa = 0.0
    heavy_atoms = 0
    for component in smiles.split("."):
        molecule = Chem.MolFromSmiles(component)
        if molecule is None:
            continue
        molecules.append(molecule)
        fingerprint = generator.GetFingerprint(molecule)
        array = np.zeros(bits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        aggregate_bits = np.maximum(aggregate_bits, array)
        for index, count in generator.GetCountFingerprint(molecule).GetNonzeroElements().items():
            aggregate_counts[index] += int(count)
        mw += Descriptors.MolWt(molecule)
        logp += Crippen.MolLogP(molecule)
        tpsa += Descriptors.TPSA(molecule)
        heavy_atoms += Lipinski.HeavyAtomCount(molecule)
    valid = bool(molecules)
    return {"valid": valid, "molecules": molecules, "bits": aggregate_bits, "counts": aggregate_counts, "mw": mw, "logp": logp, "tpsa": tpsa, "heavy_atoms": heavy_atoms}


def build_reaction_features(current: pd.DataFrame, currency: set[str], bits: int = 2048) -> pd.DataFrame:
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits)
    rows: list[dict[str, Any]] = []
    for row in current.to_dict("records"):
        reaction_smiles = row.get("reaction_smiles_lr")
        if isinstance(reaction_smiles, str) and ">>" in reaction_smiles:
            substrate_smiles, product_smiles = reaction_smiles.split(">>", 1)
        else:
            substrate_smiles = product_smiles = None
        substrate = molecular_features(substrate_smiles, generator, bits)
        product = molecular_features(product_smiles, generator, bits)
        transform = np.clip(product["counts"] - substrate["counts"], -127, 127).astype(np.int8)
        substrates = json.loads(row["substrate_chebi_ids_json"])
        products = json.loads(row["product_chebi_ids_json"])
        participants = substrates + products
        names = json.loads(row["participant_names_json"])
        joined_names = " ".join(value or "" for value in names.values()).lower()
        if "nad" in joined_names:
            cofactor_class = "NAD_OR_NADP"
        elif "flavin" in joined_names or "fad" in joined_names or "fmn" in joined_names:
            cofactor_class = "FLAVIN"
        elif "coenzyme a" in joined_names or "coa" in joined_names:
            cofactor_class = "COENZYME_A"
        elif "atp" in joined_names or "adenosine triphosphate" in joined_names:
            cofactor_class = "ATP"
        elif any(token in joined_names for token in ["iron", "zinc", "magnesium", "manganese", "copper", "cobalt"]):
            cofactor_class = "METAL"
        else:
            cofactor_class = "NONE_OR_OTHER"
        currency_count = sum(item in currency for item in participants)
        rows.append({
            "canonical_rhea": row["canonical_rhea"],
            "release": 141,
            "substrate_fingerprint_packed": np.packbits(substrate["bits"]).tobytes(),
            "product_fingerprint_packed": np.packbits(product["bits"]).tobytes(),
            "signed_transform_fingerprint_int8": transform.tobytes(),
            "fingerprint_bits": bits,
            "fingerprint_radius": 2,
            "smiles_parse_valid": bool(substrate["valid"] and product["valid"]),
            "substrate_component_count": len(substrate["molecules"]),
            "product_component_count": len(product["molecules"]),
            "participant_count": len(participants),
            "unique_participant_count": len(set(participants)),
            "currency_participant_count": currency_count,
            "currency_fraction": currency_count / len(participants) if participants else 0.0,
            "substrate_mw_sum": substrate["mw"],
            "product_mw_sum": product["mw"],
            "substrate_logp_sum": substrate["logp"],
            "product_logp_sum": product["logp"],
            "substrate_tpsa_sum": substrate["tpsa"],
            "product_tpsa_sum": product["tpsa"],
            "substrate_heavy_atoms": substrate["heavy_atoms"],
            "product_heavy_atoms": product["heavy_atoms"],
            "heavy_atom_change": product["heavy_atoms"] - substrate["heavy_atoms"],
            "cofactor_class": cofactor_class,
            "reaction_complexity": len(set(participants) - currency) + 0.1 * abs(product["heavy_atoms"] - substrate["heavy_atoms"]),
            "directionless_chemistry_hash": row["directionless_chemistry_hash"],
            "key_use_policy": "REFERENCE_KEY_ONLY_NOT_MODEL_FEATURE",
            "feature_provenance": "REFERENCE_DERIVED",
        })
    return pd.DataFrame(rows)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False, compression="zstd")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    started = utc_now()
    project = args.project_root.resolve()
    source = args.source_root.resolve()
    processed = project / "data" / "processed"
    reports = project / "reports"
    checkpoints = project / "checkpoints"
    rules = project / "configs" / "reaction_rules_v4.yaml"
    inventory = source / "data" / "manifests" / "resource_inventory.tsv"
    inventory_hash_before = sha256_file(inventory)
    rules_hash = sha256_file(rules)

    release_frames: dict[int, pd.DataFrame] = {}
    direction_maps: dict[int, dict[int, dict[str, Any]]] = {}
    obsolete_sets: dict[int, set[int]] = {}
    rhea_root = source / "data" / "raw" / "rhea"
    for release in RELEASES:
        frame, direction_map, obsolete = parse_release(rhea_root, release)
        release_frames[release] = frame
        direction_maps[release] = direction_map
        obsolete_sets[release] = obsolete
        print(f"release={release} masters={len(frame)} direction_ids={len(direction_map)} obsolete_ids={len(obsolete)}", flush=True)

    reaction_table = pd.concat([release_frames[value] for value in RELEASES], ignore_index=True)
    cross_map = pd.concat([
        cross_release_map(release_frames[126], release_frames[140]),
        cross_release_map(release_frames[140], release_frames[141]),
        cross_release_map(release_frames[126], release_frames[141]),
    ], ignore_index=True)

    enzyme_table = parse_enzyme(source / "data" / "raw" / "enzyme" / "release_2026_06_10" / "enzyme.dat")
    activity = pd.read_parquet(processed / "activity_table.parquet")
    activity_canonical = canonicalize_activities(activity, direction_maps[141], enzyme_table, release_frames[141])

    currency = {"CHEBI:15377", "CHEBI:15378", "CHEBI:15422", "CHEBI:16761", "CHEBI:16027", "CHEBI:18367", "CHEBI:33019", "CHEBI:57540", "CHEBI:57945", "CHEBI:58349", "CHEBI:57783", "CHEBI:57287"}
    reaction_features = build_reaction_features(release_frames[141], currency)

    phase1_single = pd.read_parquet(processed / "single_documented_activity_gold.parquet", columns=["activity_id"])
    phase1_single_activity_ids = set(phase1_single["activity_id"])
    single_canonical = activity_canonical[
        activity_canonical["activity_id"].isin(phase1_single_activity_ids)
        & (activity_canonical["evidence_tier"] == "GOLD")
        & activity_canonical["ec_l4"].notna()
        & activity_canonical["canonical_rhea"].notna()
    ].copy()

    obsolete_rows: list[dict[str, Any]] = []
    for row in enzyme_table[enzyme_table["status"] != "ACTIVE"].to_dict("records"):
        obsolete_rows.append({"database": "ENZYME", "release": "2026-06-10", "identifier": row["raw_ec"], "status": row["status"], "replacement_identifiers": row["replacement_ecs_json"], "mapping_basis": "official ENZYME description"})
    for release, identifiers in obsolete_sets.items():
        for identifier in sorted(identifiers):
            obsolete_rows.append({"database": "Rhea", "release": str(release), "identifier": f"RHEA:{identifier}", "status": "OBSOLETE", "replacement_identifiers": "[]", "mapping_basis": "official rhea-obsoletes.tsv; exact replacement resolved only when chemistry evidence supports it"})
    obsolete_report = pd.DataFrame(obsolete_rows)

    outputs = {
        "reaction_table": processed / "reaction_table.parquet",
        "rhea_cross_release_map": processed / "rhea_cross_release_map.parquet",
        "reaction_features": processed / "reaction_features.parquet",
        "ec_status_table": processed / "ec_status_table.parquet",
        "activity_table_canonical": processed / "activity_table_canonical.parquet",
        "single_documented_activity_gold_canonical": processed / "single_documented_activity_gold_canonical.parquet",
    }
    write_parquet(reaction_table, outputs["reaction_table"])
    write_parquet(cross_map, outputs["rhea_cross_release_map"])
    write_parquet(reaction_features, outputs["reaction_features"])
    write_parquet(enzyme_table, outputs["ec_status_table"])
    write_parquet(activity_canonical, outputs["activity_table_canonical"])
    write_parquet(single_canonical, outputs["single_documented_activity_gold_canonical"])
    obsolete_path = processed / "obsolete_mapping_report.tsv"
    obsolete_report.to_csv(obsolete_path, sep="\t", index=False)

    raw_rhea_mask = activity_canonical["raw_rhea_ids_json"] != "[]"
    mapped_rhea_mask = activity_canonical["canonical_rhea"].notna()
    rhea_mapping_rate = float(mapped_rhea_mask[raw_rhea_mask].mean()) if raw_rhea_mask.any() else 0.0
    smiles_valid_rate = float(reaction_features["smiles_parse_valid"].mean()) if len(reaction_features) else 0.0
    relation_counts = cross_map["relationship"].value_counts().to_dict()
    inventory_hash_after = sha256_file(inventory)
    checks = [
        ("checkpoint_01_present", (checkpoints / "CHECKPOINT_01_PASS").exists(), "Phase 1 prerequisite"),
        ("three_rhea_releases_parsed", set(reaction_table["release"]) == set(RELEASES), f"releases={sorted(reaction_table['release'].unique())}"),
        ("current_reaction_count", len(release_frames[141]) >= 10000, f"observed={len(release_frames[141])}"),
        ("reaction_primary_key_unique", not reaction_table.duplicated(["release", "canonical_rhea"]).any(), f"duplicates={reaction_table.duplicated(['release','canonical_rhea']).sum()}"),
        ("cross_release_map_nonempty", len(cross_map) >= 30000, f"observed={len(cross_map)}"),
        ("enzyme_table_nonempty", len(enzyme_table) >= 7000, f"observed={len(enzyme_table)}"),
        ("activity_primary_key_preserved", len(activity_canonical) == len(activity) and activity_canonical["activity_id"].is_unique, f"before={len(activity)} after={len(activity_canonical)}"),
        ("raw_rhea_mapping_rate", rhea_mapping_rate >= 0.95, f"rate={rhea_mapping_rate:.6f}"),
        ("canonical_single_gold_nonempty", len(single_canonical) >= 1000, f"observed={len(single_canonical)}"),
        ("canonical_single_gold_subset_of_phase1", set(single_canonical["activity_id"]) <= phase1_single_activity_ids and len(single_canonical) <= len(phase1_single), f"phase1={len(phase1_single)} canonical={len(single_canonical)}"),
        ("reaction_features_complete", len(reaction_features) == len(release_frames[141]), f"features={len(reaction_features)} reactions={len(release_frames[141])}"),
        ("reaction_smiles_parse_rate", smiles_valid_rate >= 0.80, f"rate={smiles_valid_rate:.6f}"),
        ("no_raw_rhea_shortcut_feature", "raw_rhea_id" not in reaction_features.columns and (reaction_features["key_use_policy"] == "REFERENCE_KEY_ONLY_NOT_MODEL_FEATURE").all(), "canonical key retained only for joins"),
        ("raw_inventory_unchanged", inventory_hash_before == inventory_hash_after, inventory_hash_after),
        ("outputs_exist", all(path.exists() and path.stat().st_size > 0 for path in outputs.values()) and obsolete_path.exists(), "all Phase 2 outputs"),
    ]
    qc = pd.DataFrame([(name, "PASS" if passed else "FAIL", details) for name, passed, details in checks], columns=["check", "status", "details"])
    qc.to_csv(reports / "phase02_qc.tsv", sep="\t", index=False)
    failures = qc[qc["status"] == "FAIL"]

    report = [
        "# SiteGuard V4 Phase 2 Report",
        "",
        f"- Started: `{started}`",
        f"- Completed: `{utc_now()}`",
        f"- Decision: `{'PASS' if failures.empty else 'FAIL'}`",
        f"- Reaction-rule SHA-256: `{rules_hash}`",
        "",
        "## Canonicalization summary",
        "",
        "| Item | Value |",
        "|---|---:|",
    ]
    for release in RELEASES:
        report.append(f"| Rhea {release} master reactions | {len(release_frames[release]):,} |")
    report += [
        f"| Cross-release mapping rows | {len(cross_map):,} |",
        f"| Current activity rows canonicalized | {len(activity_canonical):,} |",
        f"| Raw-Rhea mapping rate | {rhea_mapping_rate:.4%} |",
        f"| Canonical single-documented Gold | {len(single_canonical):,} |",
        f"| Reaction fingerprint rows | {len(reaction_features):,} |",
        f"| Reaction SMILES parse rate | {smiles_valid_rate:.4%} |",
        "",
        "## Cross-release relationships",
        "",
    ]
    for key, value in sorted(relation_counts.items()):
        report.append(f"- `{key}`: {value:,}")
    report += [
        "",
        "A new Rhea identifier is never automatically treated as novel chemistry. Exact directionless chemistry, stable master identifiers and direction changes are assessed separately. Raw Rhea IDs are retained as audit keys but are prohibited as categorical model features.",
        "",
        "## Automated QC",
        "",
        "| Check | Status | Details |",
        "|---|---|---|",
    ]
    for _, row in qc.iterrows():
        report.append(f"| {row['check']} | {row['status']} | {row['details']} |")
    report.append("")
    (reports / "PHASE_02_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    summary = {
        "phase": 2,
        "project_version": "V4",
        "status": "PASS" if failures.empty else "FAIL",
        "started_at": started,
        "completed_at": utc_now(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "seed": args.seed,
        "rhea_master_counts": {str(release): len(release_frames[release]) for release in RELEASES},
        "cross_release_relationship_counts": relation_counts,
        "raw_rhea_mapping_rate": rhea_mapping_rate,
        "smiles_parse_rate": smiles_valid_rate,
        "canonical_single_gold": len(single_canonical),
        "qc_failures": failures.to_dict("records"),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    (reports / "phase02_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checkpoint = checkpoints / "CHECKPOINT_02_PASS"
    if not failures.empty:
        checkpoint.unlink(missing_ok=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2
    checkpoint.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
