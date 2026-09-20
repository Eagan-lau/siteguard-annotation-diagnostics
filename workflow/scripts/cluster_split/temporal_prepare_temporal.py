#!/usr/bin/env python3
"""Build a function-time-frozen T0/T1 benchmark and model matrices.

T0 annotations (UniProt 2023_01 and Rhea 126) are the only functional
knowledge used for training, candidate eligibility, calibration, and model
selection. T1 annotations (UniProt 2026_01 and Rhea 140) are written only to
the temporal evaluation labels.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lxml import etree
from sklearn.metrics import average_precision_score


SEED = 20260821
ROW_KEY = ["pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id"]
LEVELS = ["EC_L3", "EC_L4", "EXACT_RHEA"]
CANDIDATE_COLUMNS = {"EC_L3": "ec_l3", "EC_L4": "ec_l4", "EXACT_RHEA": "canonical_rhea"}
COMPLETE_EC = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$")
ACCEPTED_RHEA_RELATIONSHIPS = {
    "stable_id_stable_chemistry", "direction_change", "chemistry_equivalent_id_change", "merge",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def json_text(values: Any) -> str:
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def rhea_direction_map(source: Path, release: int) -> dict[int, str]:
    path = source / "data/raw/rhea" / f"release_{release}" / "extracted" / str(release) / "tsv/rhea-directions.tsv"
    frame = pd.read_csv(path, sep="\t")
    output: dict[int, str] = {}
    for row in frame.itertuples(index=False):
        master = int(row.RHEA_ID_MASTER)
        canonical = f"RHEA:{master}"
        for identifier in [master, int(row.RHEA_ID_LR), int(row.RHEA_ID_RL), int(row.RHEA_ID_BI)]:
            output[identifier] = canonical
    return output


def cross_release_lookup(cross: pd.DataFrame, release: int) -> tuple[dict[str, str], dict[str, str]]:
    selected = cross.loc[(cross["source_release"] == release) & (cross["target_release"] == 141)].copy()
    mapping: dict[str, str] = {}
    status: dict[str, str] = {}
    for source_id, group in selected.groupby("source_canonical_rhea", dropna=True):
        relationships = sorted(set(group["relationship"].dropna().astype(str)))
        targets = sorted(set(group["target_canonical_rhea"].dropna().astype(str)))
        if len(targets) == 1 and set(relationships).issubset(ACCEPTED_RHEA_RELATIONSHIPS):
            mapping[str(source_id)] = targets[0]
            status[str(source_id)] = relationships[0] if len(relationships) == 1 else "+".join(relationships)
        else:
            status[str(source_id)] = "UNRESOLVED:" + "+".join(relationships or ["NO_TARGET"])
    return mapping, status


def parse_uniprot_release(
    xml_path: Path,
    release_name: str,
    target_ids: set[str],
    direction_map: dict[int, str],
    current_map: dict[str, str],
    current_status: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    protein_rows: list[dict[str, Any]] = []
    activity_rows: list[dict[str, Any]] = []
    with gzip.open(xml_path, "rb") as handle:
        context = etree.iterparse(
            handle, events=("end",),
            tag=("{http://uniprot.org/uniprot}entry", "{https://uniprot.org/uniprot}entry"),
            huge_tree=True,
        )
        for count, (_, entry) in enumerate(context, start=1):
            namespace = entry.tag.split("}", 1)[0][1:]
            tag = lambda name: f"{{{namespace}}}{name}"  # noqa: E731
            accessions = [node.text.strip().upper() for node in entry.findall(tag("accession")) if node.text]
            protein_id = accessions[0] if accessions else ""
            if protein_id in target_ids:
                sequence_node = entry.find(tag("sequence"))
                sequence = "" if sequence_node is None or sequence_node.text is None else "".join(sequence_node.text.split()).upper()
                protein_rows.append({
                    "protein_id": protein_id,
                    "source_release": release_name,
                    "sequence_length": len(sequence),
                    "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest() if sequence else None,
                })
                entry_ecs: set[str] = set()
                for node in entry.findall(f".//{tag('protein')}//{tag('ecNumber')}"):
                    if node.text and COMPLETE_EC.match(node.text.strip()):
                        entry_ecs.add(node.text.strip())
                signatures: set[tuple[str, str, str, str]] = set()
                represented_ecs: set[str] = set()
                for comment in entry.findall(tag("comment")):
                    if comment.get("type") != "catalytic activity":
                        continue
                    for reaction in comment.findall(tag("reaction")):
                        ecs: set[str] = set()
                        raw_rhea_ids: set[int] = set()
                        for reference in reaction.findall(tag("dbReference")):
                            identifier = reference.get("id", "")
                            if reference.get("type") == "EC" and COMPLETE_EC.match(identifier):
                                ecs.add(identifier)
                                entry_ecs.add(identifier)
                            elif reference.get("type") == "Rhea":
                                match = re.search(r"([0-9]+)$", identifier)
                                if match:
                                    raw_rhea_ids.add(int(match.group(1)))
                        represented_ecs |= ecs
                        source_rheas = {direction_map[value] for value in raw_rhea_ids if value in direction_map}
                        if source_rheas:
                            for source_rhea in source_rheas:
                                current_rhea = current_map.get(source_rhea, "")
                                rhea_state = current_status.get(source_rhea, "UNRESOLVED:NO_CROSS_RELEASE_RECORD")
                                for ec in ecs or {""}:
                                    signatures.add((ec, current_rhea, source_rhea, rhea_state))
                        else:
                            for ec in ecs or {""}:
                                signatures.add((ec, "", "", "NO_RHEA" if not raw_rhea_ids else "UNRESOLVED_RAW_RHEA"))
                for ec in entry_ecs - represented_ecs:
                    signatures.add((ec, "", "", "NO_RHEA"))
                for index, (ec4, current_rhea, source_rhea, rhea_state) in enumerate(sorted(signatures), start=1):
                    activity_rows.append({
                        "activity_id": f"{release_name}:{protein_id}:{index}",
                        "protein_id": protein_id,
                        "ec_l3": ".".join(ec4.split(".")[:3]) if ec4 else "",
                        "ec_l4": ec4,
                        "canonical_rhea": current_rhea,
                        "source_release_rhea": source_rhea,
                        "rhea_cross_release_status": rhea_state,
                        "source_release": release_name,
                    })
            entry.clear()
            while entry.getprevious() is not None:
                del entry.getparent()[0]
            if count % 100_000 == 0:
                print(json.dumps({"release": release_name, "entries_scanned": count, "target_proteins": len(protein_rows)}), flush=True)
    proteins = pd.DataFrame(protein_rows).drop_duplicates("protein_id")
    activities = pd.DataFrame(activity_rows)
    return proteins, activities


def build_labels(proteins: pd.DataFrame, activities: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, dict[str, dict[str, set[str]]], dict[str, set[tuple[str, str]]]]:
    labels: dict[str, dict[str, set[str]]] = defaultdict(lambda: {level: set() for level in LEVELS})
    signatures: dict[str, set[tuple[str, str]]] = defaultdict(set)
    unresolved: dict[str, int] = defaultdict(int)
    for row in activities.itertuples(index=False):
        if row.ec_l3:
            labels[row.protein_id]["EC_L3"].add(str(row.ec_l3))
        if row.ec_l4:
            labels[row.protein_id]["EC_L4"].add(str(row.ec_l4))
        if row.canonical_rhea:
            labels[row.protein_id]["EXACT_RHEA"].add(str(row.canonical_rhea))
        if row.ec_l4 or row.canonical_rhea:
            signatures[row.protein_id].add((str(row.ec_l4 or ""), str(row.canonical_rhea or "")))
        if str(row.rhea_cross_release_status).startswith("UNRESOLVED"):
            unresolved[row.protein_id] += 1
    rows: list[dict[str, Any]] = []
    for protein_id in proteins["protein_id"].astype(str):
        record: dict[str, Any] = {"protein_id": protein_id}
        for level in LEVELS:
            record[f"{prefix}_{level.lower()}_json"] = json_text(sorted(labels[protein_id][level]))
            record[f"{prefix}_{level.lower()}_count"] = len(labels[protein_id][level])
        record[f"{prefix}_resolution"] = max(
            (index + 1 for index, level in enumerate(LEVELS) if labels[protein_id][level]), default=0
        )
        record[f"{prefix}_activity_signatures_json"] = json_text(sorted([list(value) for value in signatures[protein_id]]))
        record[f"{prefix}_unresolved_rhea_activities"] = unresolved[protein_id]
        rows.append(record)
    return pd.DataFrame(rows), labels, signatures


def candidate_signature_known(protein_id: str, ec4: Any, rhea: Any, signatures: dict[str, set[tuple[str, str]]]) -> bool:
    ec_value = "" if pd.isna(ec4) else str(ec4)
    rhea_value = "" if pd.isna(rhea) else str(rhea)
    if not ec_value and not rhea_value:
        return False
    known = signatures.get(str(protein_id), set())
    if ec_value and rhea_value:
        return (ec_value, rhea_value) in known
    if ec_value:
        return any(item_ec == ec_value for item_ec, _ in known)
    return any(item_rhea == rhea_value for _, item_rhea in known)


def truth_arrays(
    frame: pd.DataFrame,
    labels: dict[str, dict[str, set[str]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outcomes = np.zeros((len(frame), len(LEVELS)), dtype=np.float32)
    query_available = np.zeros_like(outcomes, dtype=bool)
    candidate_available = np.zeros_like(outcomes, dtype=bool)
    query_ids = frame["query_protein_id"].astype(str).tolist()
    for index, level in enumerate(LEVELS):
        values = frame[CANDIDATE_COLUMNS[level]].fillna("").astype(str).tolist()
        for row_index, (protein_id, value) in enumerate(zip(query_ids, values, strict=True)):
            truth = labels.get(protein_id, {}).get(level, set())
            query_available[row_index, index] = bool(truth)
            candidate_available[row_index, index] = bool(value)
            outcomes[row_index, index] = float(bool(value) and value in truth)
    return outcomes, query_available, candidate_available


def add_truth_metadata(frame: pd.DataFrame, outcomes: np.ndarray, query_available: np.ndarray, candidate_available: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    for index, level in enumerate(LEVELS):
        output[f"correct_{level}"] = outcomes[:, index].astype(bool)
        output[f"query_truth_available_{level}"] = query_available[:, index]
        output[f"candidate_label_available_{level}"] = candidate_available[:, index]
        output[f"evaluation_mask_{level}"] = query_available[:, index] & candidate_available[:, index]
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    started = utc_now()
    root = args.project_root.resolve()
    source = args.source_root.resolve()
    processed = root / "data/processed"
    splits = root / "data/splits"
    work = root / "data/interim/phase13"
    results = root / "results/phase13"
    models = root / "models/phase13"
    reports = root / "reports"
    for path in [work, results, models, reports]:
        path.mkdir(parents=True, exist_ok=True)
    if not (root / "checkpoints/CHECKPOINT_12_PASS").is_file():
        raise RuntimeError("CHECKPOINT_12_PASS is required")

    temporal_split = pd.read_parquet(splits / "split_temporal.parquet")
    target_ids = set(temporal_split["protein_id"].astype(str))
    cross = pd.read_parquet(processed / "rhea_cross_release_map.parquet")
    maps: dict[int, dict[str, str]] = {}
    states: dict[int, dict[str, str]] = {}
    for release in [126, 140]:
        maps[release], states[release] = cross_release_lookup(cross, release)
    directions = {release: rhea_direction_map(source, release) for release in [126, 140]}
    t0_proteins, t0_activities = parse_uniprot_release(
        source / "data/raw/uniprot/historical_2023_01/extracted/uniprot_sprot.xml.gz",
        "T0_2023_01", target_ids, directions[126], maps[126], states[126],
    )
    t1_proteins, t1_activities = parse_uniprot_release(
        source / "data/raw/uniprot/historical_2026_01/extracted/uniprot_sprot.xml.gz",
        "T1_2026_01", target_ids, directions[140], maps[140], states[140],
    )
    t0_label_frame, t0_labels, t0_signatures = build_labels(t0_proteins, t0_activities, "t0")
    t1_label_frame, t1_labels, _ = build_labels(t1_proteins, t1_activities, "t1")
    write_parquet(t0_proteins, work / "t0_proteins.parquet")
    write_parquet(t0_activities, work / "t0_activities.parquet")
    write_parquet(t1_proteins, work / "t1_proteins.parquet")
    write_parquet(t1_activities, work / "t1_activities.parquet")
    write_parquet(t0_label_frame, work / "t0_labels.parquet")
    write_parquet(t1_label_frame, work / "t1_labels.parquet")

    direct_126_141 = cross.loc[(cross["source_release"] == 126) & (cross["target_release"] == 141)].copy()
    known_t0_chemistry = set(
        direct_126_141.loc[
            direct_126_141["source_canonical_rhea"].notna()
            & direct_126_141["source_directionless_hash"].notna()
            & direct_126_141["source_directionless_hash"].eq(direct_126_141["target_directionless_hash"]),
            "target_canonical_rhea",
        ].dropna().astype(str)
    )
    novel_chemistry = set(
        direct_126_141.loc[direct_126_141["relationship"].eq("genuinely_novel_chemistry"), "target_canonical_rhea"]
        .dropna().astype(str)
    )
    t0_reference_rhea = set().union(*(values["EXACT_RHEA"] for values in t0_labels.values())) if t0_labels else set()

    temporal = temporal_split.merge(t0_label_frame, on="protein_id", how="left").merge(t1_label_frame, on="protein_id", how="left")
    t0_ids, t1_ids = set(t0_proteins["protein_id"]), set(t1_proteins["protein_id"])
    task_rows: list[dict[str, Any]] = []
    for row in temporal.itertuples(index=False):
        protein_id = str(row.protein_id)
        old = t0_labels.get(protein_id, {level: set() for level in LEVELS})
        new = t1_labels.get(protein_id, {level: set() for level in LEVELS})
        old_resolution = max((i + 1 for i, level in enumerate(LEVELS) if old[level]), default=0)
        new_resolution = max((i + 1 for i, level in enumerate(LEVELS) if new[level]), default=0)
        stable_shared = protein_id in t0_ids and protein_id in t1_ids and bool(row.T0_T1_sequence_stable)
        new_t1 = protein_id not in t0_ids and protein_id in t1_ids
        new_rheas = new["EXACT_RHEA"]
        novel_values = sorted(new_rheas & novel_chemistry)
        known_values = sorted(new_rheas & known_t0_chemistry)
        known_library_values = sorted(new_rheas & t0_reference_rhea)
        limited = stable_shared and old_resolution > 0 and new_resolution > old_resolution
        is_new_known = new_t1 and bool(known_library_values)
        is_novel = bool(novel_values)
        if is_novel:
            primary_task = "GENUINE_NOVEL_CHEMISTRY"
        elif limited:
            primary_task = "LIMITED_KNOWLEDGE_REFINEMENT"
        elif is_new_known:
            primary_task = "NEW_PROTEIN_KNOWN_REACTION"
        elif new_t1 and new_resolution > 0:
            primary_task = "NEW_PROTEIN_OTHER_OR_UNRESOLVED"
        elif stable_shared and new_resolution > 0:
            primary_task = "SEQUENCE_STABLE_CONTEXT"
        elif row.temporal_membership == "T0_T1_SEQUENCE_CHANGED_SENSITIVITY":
            primary_task = "SEQUENCE_UPDATE_SENSITIVITY"
        else:
            primary_task = "NOT_EVALUABLE"
        revision = stable_shared and any(old[level] != new[level] for level in LEVELS)
        task_rows.append({
            "protein_id": protein_id,
            "present_in_parsed_T0": protein_id in t0_ids,
            "present_in_parsed_T1": protein_id in t1_ids,
            "old_resolution": old_resolution,
            "new_resolution": new_resolution,
            "is_new_protein_known_reaction": is_new_known,
            "is_limited_knowledge_refinement": limited,
            "is_genuine_novel_chemistry": is_novel,
            "annotation_revision": revision,
            "novel_t1_rhea_json": json_text(novel_values),
            "known_t0_chemistry_t1_rhea_json": json_text(known_values),
            "known_t0_reference_library_t1_rhea_json": json_text(known_library_values),
            "primary_task": primary_task,
            "primary_evaluation_eligible": bool(
                row.T1_current_sequence_stable and new_resolution > 0
                and primary_task in {"GENUINE_NOVEL_CHEMISTRY", "LIMITED_KNOWLEDGE_REFINEMENT", "NEW_PROTEIN_KNOWN_REACTION"}
            ),
        })
    temporal = temporal.merge(pd.DataFrame(task_rows), on="protein_id", how="left", validate="one_to_one")
    write_parquet(temporal, results / "temporal_sequence_stable_set.parquet")

    pair_columns = [
        "query_protein_id", "reference_protein_id", "reference_activity_id",
        "ec_l3", "ec_l4", "canonical_rhea", "query_split_expected", "query_cluster_id_30",
    ]
    pairs = pd.read_parquet(processed / "population_pairs.parquet", columns=pair_columns).assign(pair_set="population")
    ref_temporal = temporal_split[["protein_id", "T0_T1_sequence_stable", "T1_current_sequence_stable"]].rename(
        columns={
            "protein_id": "reference_protein_id",
            "T0_T1_sequence_stable": "reference_T0_T1_sequence_stable",
            "T1_current_sequence_stable": "reference_T1_current_sequence_stable",
        }
    )
    pairs = pairs.merge(ref_temporal, on="reference_protein_id", how="left", validate="many_to_one")
    stable_reference = pairs["reference_T0_T1_sequence_stable"].fillna(False) & pairs["reference_T1_current_sequence_stable"].fillna(False)
    signature_known = np.fromiter(
        (
            candidate_signature_known(row.reference_protein_id, row.ec_l4, row.canonical_rhea, t0_signatures)
            for row in pairs[["reference_protein_id", "ec_l4", "canonical_rhea"]].itertuples(index=False)
        ), dtype=bool, count=len(pairs),
    )
    pairs = pairs.loc[stable_reference.to_numpy() & signature_known].copy()
    temporal_lookup_columns = [
        "protein_id", "temporal_membership", "T0_T1_sequence_stable", "T1_current_sequence_stable",
        "present_in_parsed_T0", "present_in_parsed_T1",
        "primary_task", "primary_evaluation_eligible", "is_new_protein_known_reaction",
        "is_limited_knowledge_refinement", "is_genuine_novel_chemistry", "annotation_revision",
        "old_resolution", "new_resolution", "novel_t1_rhea_json",
    ]
    pairs = pairs.merge(
        temporal[temporal_lookup_columns].rename(columns={"protein_id": "query_protein_id"}),
        on="query_protein_id", how="left", validate="many_to_one",
    )

    feature_path = processed / "global_features.parquet"
    feature_columns = pq.ParquetFile(feature_path).schema_arrow.names
    input_columns = [column for column in feature_columns if column not in ROW_KEY + ["query_split"]]
    forbidden = sorted(column for column in input_columns if "ground_truth" in column.lower() or column.startswith("same_"))
    if forbidden:
        raise RuntimeError(f"Future/ground-truth feature leakage: {forbidden}")
    features = pd.read_parquet(feature_path, columns=ROW_KEY + input_columns, filters=[("pair_set", "==", "population")])
    frame = pairs.merge(features, on=ROW_KEY, how="inner", validate="one_to_one")

    t0_outcome, t0_query_available, candidate_available = truth_arrays(frame, t0_labels)
    t1_outcome, t1_query_available, _ = truth_arrays(frame, t1_labels)
    train_query = (
        frame["query_split_expected"].eq("train") & frame["T0_T1_sequence_stable"].fillna(False)
        & frame["T1_current_sequence_stable"].fillna(False)
    ).to_numpy(copy=True)
    validation_query = (
        frame["query_split_expected"].eq("validation") & frame["T0_T1_sequence_stable"].fillna(False)
        & frame["T1_current_sequence_stable"].fillna(False)
    ).to_numpy(copy=True)
    temporal_query = (
        frame["query_split_expected"].eq("test") & frame["present_in_parsed_T1"].fillna(False)
        & frame["T1_current_sequence_stable"].fillna(False) & (frame["new_resolution"].fillna(0) > 0)
    ).to_numpy(copy=True)
    t0_mask = t0_query_available & candidate_available
    t1_mask = t1_query_available & candidate_available
    train_query &= t0_mask.any(axis=1)
    validation_query &= t0_mask.any(axis=1)
    temporal_query &= t1_mask.any(axis=1)
    train = frame.loc[train_query].reset_index(drop=True)
    validation = frame.loc[validation_query].reset_index(drop=True)
    temporal_eval = frame.loc[temporal_query].reset_index(drop=True)
    train_y, train_truth, train_candidate = truth_arrays(train, t0_labels)
    validation_y, validation_truth, validation_candidate = truth_arrays(validation, t0_labels)
    temporal_y, temporal_truth, temporal_candidate = truth_arrays(temporal_eval, t1_labels)
    train_mask = train_truth & train_candidate
    validation_mask = validation_truth & validation_candidate
    temporal_mask = temporal_truth & temporal_candidate

    categorical = [column for column in ["reference_ec_l1", "reference_cofactor_class"] if column in input_columns]
    numeric = [column for column in input_columns if column not in categorical]
    train_categories = pd.get_dummies(train[categorical].fillna("MISSING").astype(str), prefix=categorical, dtype=np.float32)
    validation_categories = pd.get_dummies(validation[categorical].fillna("MISSING").astype(str), prefix=categorical, dtype=np.float32)
    temporal_categories = pd.get_dummies(temporal_eval[categorical].fillna("MISSING").astype(str), prefix=categorical, dtype=np.float32)
    validation_categories = validation_categories.reindex(columns=train_categories.columns, fill_value=0.0)
    temporal_categories = temporal_categories.reindex(columns=train_categories.columns, fill_value=0.0)
    train_numeric = train[numeric].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    validation_numeric = validation[numeric].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    temporal_numeric = temporal_eval[numeric].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    medians = train_numeric.median(axis=0).fillna(0.0)
    train_numeric = train_numeric.fillna(medians)
    validation_numeric = validation_numeric.fillna(medians)
    temporal_numeric = temporal_numeric.fillna(medians)
    means = train_numeric.mean(axis=0)
    scales = train_numeric.std(axis=0).replace(0, 1).fillna(1.0)
    train_x = np.column_stack([((train_numeric - means) / scales).to_numpy(np.float32), train_categories.to_numpy(np.float32)])
    validation_x = np.column_stack([((validation_numeric - means) / scales).to_numpy(np.float32), validation_categories.to_numpy(np.float32)])
    temporal_x = np.column_stack([((temporal_numeric - means) / scales).to_numpy(np.float32), temporal_categories.to_numpy(np.float32)])
    np.save(work / "train_X.npy", train_x)
    np.save(work / "train_y.npy", train_y)
    np.save(work / "train_mask.npy", train_mask)
    np.save(work / "validation_X.npy", validation_x)
    np.save(work / "validation_y.npy", validation_y)
    np.save(work / "validation_mask.npy", validation_mask)
    np.save(work / "temporal_X.npy", temporal_x)
    np.save(work / "temporal_y.npy", temporal_y)
    np.save(work / "temporal_mask.npy", temporal_mask)

    metadata_columns = ROW_KEY + [
        "query_split_expected", "query_cluster_id_30", "ec_l3", "ec_l4", "canonical_rhea",
        "temporal_membership", "T0_T1_sequence_stable", "T1_current_sequence_stable", "primary_task",
        "primary_evaluation_eligible", "is_new_protein_known_reaction", "is_limited_knowledge_refinement",
        "is_genuine_novel_chemistry", "annotation_revision", "old_resolution", "new_resolution", "novel_t1_rhea_json",
    ]
    validation_metadata = add_truth_metadata(validation[metadata_columns], validation_y, validation_truth, validation_candidate)
    temporal_metadata = add_truth_metadata(temporal_eval[metadata_columns], temporal_y, temporal_truth, temporal_candidate)
    write_parquet(validation_metadata, work / "validation_metadata.parquet")
    write_parquet(temporal_metadata, work / "temporal_metadata.parquet")

    tree_validation = np.zeros((len(validation), len(LEVELS)), dtype=np.float32)
    tree_temporal = np.zeros((len(temporal_eval), len(LEVELS)), dtype=np.float32)
    lightgbm_summary: dict[str, Any] = {}
    for index, level in enumerate(LEVELS):
        fit_mask = train_mask[:, index]
        valid_mask = validation_mask[:, index]
        y_fit = train_y[fit_mask, index].astype(int)
        positive = int(y_fit.sum())
        negative = len(y_fit) - positive
        weights = np.ones(len(y_fit), dtype=np.float32)
        if positive and negative:
            weights[y_fit == 1] = np.sqrt(negative / positive)
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=600, learning_rate=0.04, num_leaves=31,
            min_child_samples=100, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=SEED, n_jobs=args.threads, verbosity=-1,
        )
        model.fit(
            train_x[fit_mask], y_fit, sample_weight=weights,
            eval_set=[(validation_x[valid_mask], validation_y[valid_mask, index].astype(int))],
            eval_metric="average_precision", callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        tree_validation[:, index] = model.predict_proba(validation_x)[:, 1].astype(np.float32)
        tree_temporal[:, index] = model.predict_proba(temporal_x)[:, 1].astype(np.float32)
        model.booster_.save_model(str(models / f"lightgbm_t0_{level}.txt"))
        lightgbm_summary[level] = {
            "training_rows": int(fit_mask.sum()), "training_positives": positive,
            "validation_rows": int(valid_mask.sum()), "best_iteration": int(model.best_iteration_),
            "validation_auprc": float(average_precision_score(
                validation_y[valid_mask, index], tree_validation[valid_mask, index]
            )),
        }
    np.save(work / "tree_validation_predictions.npy", tree_validation)
    np.save(work / "tree_temporal_predictions.npy", tree_temporal)
    preprocessing = {
        "model_columns": numeric + train_categories.columns.tolist(),
        "numeric_columns": numeric, "categorical_columns": categorical,
        "categorical_dummy_columns": train_categories.columns.tolist(),
        "numeric_medians": {key: float(value) for key, value in medians.items()},
        "numeric_means": {key: float(value) for key, value in means.items()},
        "numeric_scales": {key: float(value) for key, value in scales.items()},
        "functional_training_snapshot": "UniProt 2023_01 + Rhea 126 mapped chemically to Rhea 141",
        "future_function_fields_in_inputs": False,
        "t1_used_for_training_or_validation": False,
        "structure_note": "structure/domain features are current sequence-derived evidence; functional annotations are time-frozen",
    }
    (models / "temporal_preprocessing.json").write_text(json.dumps(preprocessing, indent=2) + "\n", encoding="utf-8")
    task_counts = temporal.loc[temporal["primary_evaluation_eligible"], "primary_task"].value_counts().to_dict()
    summary = {
        "phase": 13, "stage": "prepare_function_time_frozen", "status": "PASS",
        "started_at": started, "completed_at": utc_now(), "slurm_job_id": os.getenv("SLURM_JOB_ID", "NA"),
        "t0_proteins": len(t0_proteins), "t1_proteins": len(t1_proteins),
        "t0_activities": len(t0_activities), "t1_activities": len(t1_activities),
        "known_t0_chemistry_ids": len(known_t0_chemistry), "genuine_novel_chemistry_ids": len(novel_chemistry),
        "t0_reference_filtered_pairs": len(frame), "training_rows": len(train),
        "validation_rows": len(validation), "temporal_test_rows": len(temporal_eval),
        "training_queries": train["query_protein_id"].nunique(),
        "validation_queries": validation["query_protein_id"].nunique(),
        "temporal_test_queries": temporal_eval["query_protein_id"].nunique(),
        "primary_task_counts": task_counts, "model_features": train_x.shape[1],
        "lightgbm": lightgbm_summary,
        "functional_knowledge_time_frozen": True, "t1_used_for_training_or_selection": False,
    }
    (reports / "phase13_prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
