#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


RATIOS = (("train", 0.70), ("validation", 0.15), ("test", 0.15))


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


def stable_split(group: str | None, seed: int, namespace: str) -> str:
    if not group or group == "UNASSIGNED":
        return "UNASSIGNED"
    digest = hashlib.sha256(f"{seed}|{namespace}|{group}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "validation"
    return "test"


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def aggregate_unique(values: pd.Series) -> str:
    return json_text(sorted({str(value) for value in values.dropna() if str(value)}))


def prepare(project: Path, source: Path, seed: int) -> None:
    processed = project / "data" / "processed"
    work = project / "data" / "interim" / "phase03"
    work.mkdir(parents=True, exist_ok=True)
    activity = pd.read_parquet(processed / "activity_table_canonical.parquet", columns=["activity_id", "protein_id", "canonical_ec", "ec_l1", "ec_l3", "canonical_rhea", "evidence_tier"])
    proteins = pd.read_parquet(processed / "protein_table.parquet", columns=["protein_id", "sequence", "sequence_valid", "fragment_status", "taxonomy_id", "pfam_domains_json"])
    eligible_activity = activity[
        activity["evidence_tier"].isin(["GOLD", "SILVER"])
        & activity["canonical_ec"].notna()
    ].copy()
    eligible_protein_ids = set(eligible_activity["protein_id"])
    benchmark = proteins[
        proteins["protein_id"].isin(eligible_protein_ids)
        & proteins["sequence_valid"]
        & (proteins["fragment_status"] == "complete")
    ].copy()
    benchmark_ids = set(benchmark["protein_id"])
    eligible_activity = eligible_activity[eligible_activity["protein_id"].isin(benchmark_ids)]
    labels = eligible_activity.groupby("protein_id").agg(
        documented_activity_count=("activity_id", "nunique"),
        ec_l1_json=("ec_l1", aggregate_unique),
        ec_l3_json=("ec_l3", aggregate_unique),
        ec_l4_json=("canonical_ec", aggregate_unique),
        canonical_rhea_json=("canonical_rhea", aggregate_unique),
        evidence_tiers_json=("evidence_tier", aggregate_unique),
    ).reset_index()
    benchmark = benchmark.merge(labels, on="protein_id", how="inner", validate="one_to_one")
    benchmark = benchmark.sort_values("protein_id").reset_index(drop=True)
    write_parquet(benchmark, work / "benchmark_proteins.parquet")

    fasta_path = work / "benchmark_sequences.fasta"
    temporary = fasta_path.with_suffix(".fasta.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in benchmark[["protein_id", "sequence"]].itertuples(index=False):
            handle.write(f">{row.protein_id}\n")
            sequence = row.sequence
            for index in range(0, len(sequence), 80):
                handle.write(sequence[index:index + 80] + "\n")
    temporary.replace(fasta_path)
    summary = {
        "stage": "prepare",
        "created_at": utc_now(),
        "seed": seed,
        "benchmark_proteins": len(benchmark),
        "benchmark_activities": len(eligible_activity),
        "fasta_sha256": sha256_file(fasta_path),
    }
    (work / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def read_cluster_map(path: Path, expected_ids: set[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            representative, member = fields[0], fields[1]
            if member in mapping and mapping[member] != representative:
                raise ValueError(f"Member assigned to multiple clusters in {path}: {member}")
            mapping[member] = representative
    for protein_id in expected_ids - set(mapping):
        mapping[protein_id] = protein_id
    extras = set(mapping) - expected_ids
    if extras:
        raise ValueError(f"Unexpected MMseqs members in {path}: {len(extras)}")
    return mapping


def pfam_clan_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields and fields[0]:
                mapping[fields[0]] = fields[1] if len(fields) > 1 and fields[1] else "NO_CLAN"
    return mapping


def primary_pfam(payload: str) -> str:
    try:
        records = json.loads(payload)
    except Exception:
        return "UNASSIGNED"
    identifiers = sorted({str(record.get("id")) for record in records if record.get("id")})
    return identifiers[0] if identifiers else "UNASSIGNED"


def cath_superfamilies(source: Path, benchmark_ids: set[str]) -> dict[str, list[str]]:
    sifts_path = source / "data" / "raw" / "sifts" / "flatfiles" / "pdb_chain_cath_uniprot.tsv.gz"
    sifts = pd.read_csv(sifts_path, sep="\t", comment="#", dtype=str)
    sifts = sifts[sifts["SP_PRIMARY"].isin(benchmark_ids)].dropna(subset=["SP_PRIMARY", "CATH_ID"])
    requested_domains = set(sifts["CATH_ID"])
    domain_to_superfamily: dict[str, str] = {}
    cath_path = source / "data" / "raw" / "cath" / "v4_4_0" / "classification" / "cath-domain-list-v4_4_0.txt"
    with cath_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) >= 5 and fields[0] in requested_domains:
                domain_to_superfamily[fields[0]] = ".".join(fields[1:5])
    protein_supers: dict[str, set[str]] = defaultdict(set)
    for row in sifts[["SP_PRIMARY", "CATH_ID"]].itertuples(index=False):
        superfamily = domain_to_superfamily.get(row.CATH_ID)
        if superfamily:
            protein_supers[row.SP_PRIMARY].add(superfamily)
    return {protein: sorted(values) for protein, values in protein_supers.items()}


def fasta_hashes(path: Path, targets: set[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    accession: str | None = None
    sequence_parts: list[str] = []

    def finish() -> None:
        nonlocal accession, sequence_parts
        if accession in targets:
            hashes[accession] = hashlib.sha256("".join(sequence_parts).encode("ascii")).hexdigest()
        accession = None
        sequence_parts = []

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                finish()
                header = line[1:].strip().split()[0]
                parts = header.split("|")
                accession = parts[1] if len(parts) >= 3 else header
            elif accession in targets:
                sequence_parts.append(line.strip())
    finish()
    return hashes


def leakage_row(scheme: str, unit: str, group_series: pd.Series, split_series: pd.Series) -> dict[str, Any]:
    frame = pd.DataFrame({"group": group_series, "split": split_series})
    frame = frame[(frame["group"] != "UNASSIGNED") & (frame["split"] != "UNASSIGNED")]
    overlaps = int((frame.groupby("group")["split"].nunique() > 1).sum()) if len(frame) else 0
    return {"scheme": scheme, "unit": unit, "overlap_count": overlaps, "status": "PASS" if overlaps == 0 else "FAIL", "details": f"groups_checked={frame['group'].nunique()}"}


def finalize(project: Path, source: Path, seed: int) -> int:
    started = utc_now()
    processed = project / "data" / "processed"
    work = project / "data" / "interim" / "phase03"
    split_dir = project / "data" / "splits"
    reports = project / "reports"
    checkpoints = project / "checkpoints"
    split_dir.mkdir(parents=True, exist_ok=True)
    rules = project / "configs" / "split_rules_v4.yaml"
    raw_inventory = source / "data" / "manifests" / "resource_inventory.tsv"
    raw_hash_before = sha256_file(raw_inventory)

    benchmark = pd.read_parquet(work / "benchmark_proteins.parquet")
    benchmark_ids = set(benchmark["protein_id"])
    cluster_maps = {
        "cluster_id_30": read_cluster_map(work / "mmseqs_id30_cluster.tsv", benchmark_ids),
        "cluster_id_40": read_cluster_map(work / "mmseqs_id40_cluster.tsv", benchmark_ids),
        "cluster_id_50": read_cluster_map(work / "mmseqs_id50_cluster.tsv", benchmark_ids),
    }
    sequence_split = benchmark[["protein_id", "documented_activity_count", "ec_l1_json", "ec_l3_json", "ec_l4_json", "canonical_rhea_json"]].copy()
    for column, mapping in cluster_maps.items():
        sequence_split[column] = sequence_split["protein_id"].map(mapping)
    # MMseqs easy-cluster is representative/greedy and cluster assignments from
    # separate identity runs are not guaranteed to be nested.  The 30% result is
    # the main split, while 40% and 50% are independent sensitivity splits.
    sequence_split["split_30"] = sequence_split["cluster_id_30"].map(lambda value: stable_split(value, seed, "sequence30"))
    sequence_split["split_40"] = sequence_split["cluster_id_40"].map(lambda value: stable_split(value, seed, "sequence40"))
    sequence_split["split_50"] = sequence_split["cluster_id_50"].map(lambda value: stable_split(value, seed, "sequence50"))
    sequence_split["split"] = sequence_split["split_30"]
    sequence_split["split_seed"] = seed
    sequence_split["split_provenance"] = "QUERY_DERIVED_CLUSTER_MEMBERSHIP"

    clans = pfam_clan_map(source / "data" / "raw" / "pfam" / "current_release_2026_01_22" / "Pfam-A.clans.tsv.gz")
    family_split = benchmark[["protein_id", "pfam_domains_json"]].copy()
    family_split["primary_pfam"] = family_split["pfam_domains_json"].map(primary_pfam)
    family_split["primary_pfam_clan"] = family_split["primary_pfam"].map(lambda value: clans.get(value, "UNASSIGNED") if value != "UNASSIGNED" else "UNASSIGNED")
    family_split["pfam_family_split"] = family_split["primary_pfam"].map(lambda value: stable_split(value, seed, "pfam_family"))
    family_split["pfam_clan_split"] = family_split["primary_pfam_clan"].map(lambda value: stable_split(value, seed, "pfam_clan"))

    protein_supers = cath_superfamilies(source, benchmark_ids)
    structure_split = pd.DataFrame({"protein_id": sorted(benchmark_ids)})
    structure_split["cath_superfamilies_json"] = structure_split["protein_id"].map(lambda value: json_text(protein_supers.get(value, [])))
    structure_split["primary_cath_superfamily"] = structure_split["protein_id"].map(lambda value: protein_supers.get(value, ["UNASSIGNED"])[0])
    structure_split["cath_superfamily_split"] = structure_split["primary_cath_superfamily"].map(lambda value: stable_split(value, seed, "cath_superfamily"))
    structure_split["foldseek_structure_cluster"] = "PENDING_PHASE_4"

    taxonomy_split = benchmark[["protein_id", "taxonomy_id"]].copy()
    taxonomy_split["taxonomy_group"] = taxonomy_split["taxonomy_id"].fillna("UNASSIGNED").astype(str)
    taxonomy_split["taxonomy_species_split"] = taxonomy_split["taxonomy_group"].map(lambda value: stable_split(value, seed, "taxonomy_species"))

    activity = pd.read_parquet(processed / "activity_table_canonical.parquet", columns=["protein_id", "canonical_rhea", "evidence_tier"])
    activity = activity[activity["protein_id"].isin(benchmark_ids) & activity["canonical_rhea"].notna() & activity["evidence_tier"].isin(["GOLD", "SILVER"])]
    reaction_groups = activity.groupby("protein_id")["canonical_rhea"].apply(lambda values: sorted(set(values))).to_dict()
    reaction_rows: list[dict[str, Any]] = []
    for protein_id in sorted(benchmark_ids):
        groups = reaction_groups.get(protein_id, [])
        group_splits = sorted({stable_split(group, seed, "reaction_holdout") for group in groups})
        if not groups:
            membership = "UNASSIGNED_NO_CANONICAL_RHEA"
        elif len(group_splits) == 1:
            membership = group_splits[0]
        else:
            membership = "EXCLUDED_CROSS_SPLIT_MULTIFUNCTIONAL"
        reaction_rows.append({
            "protein_id": protein_id,
            "canonical_rhea_groups_json": json_text(groups),
            "reaction_group_splits_json": json_text(group_splits),
            "reaction_holdout_split": membership,
        })
    reaction_split = pd.DataFrame(reaction_rows)

    t0_path = source / "data" / "raw" / "uniprot" / "historical_2023_01" / "extracted" / "uniprot_sprot.fasta.gz"
    t1_path = source / "data" / "raw" / "uniprot" / "historical_2026_01" / "extracted" / "uniprot_sprot.fasta.gz"
    t0_hashes = fasta_hashes(t0_path, benchmark_ids)
    t1_hashes = fasta_hashes(t1_path, benchmark_ids)
    current_hashes = {row.protein_id: hashlib.sha256(row.sequence.encode("ascii")).hexdigest() for row in benchmark[["protein_id", "sequence"]].itertuples(index=False)}
    temporal_rows: list[dict[str, Any]] = []
    for protein_id in sorted(benchmark_ids):
        t0_hash, t1_hash, current_hash = t0_hashes.get(protein_id), t1_hashes.get(protein_id), current_hashes[protein_id]
        if t0_hash and t1_hash and t0_hash == t1_hash:
            temporal_class = "T0_T1_SEQUENCE_STABLE"
        elif t0_hash and t1_hash:
            temporal_class = "T0_T1_SEQUENCE_CHANGED_SENSITIVITY"
        elif not t0_hash and t1_hash:
            temporal_class = "NEW_IN_T1"
        elif t0_hash and not t1_hash:
            temporal_class = "ABSENT_IN_T1"
        else:
            temporal_class = "NEW_AFTER_T1_OR_UNRESOLVED"
        temporal_rows.append({
            "protein_id": protein_id,
            "present_T0_2023_01": bool(t0_hash),
            "present_T1_2026_01": bool(t1_hash),
            "present_current_2026_02": True,
            "sequence_hash_T0": t0_hash,
            "sequence_hash_T1": t1_hash,
            "sequence_hash_current": current_hash,
            "T0_T1_sequence_stable": bool(t0_hash and t1_hash and t0_hash == t1_hash),
            "T1_current_sequence_stable": bool(t1_hash and t1_hash == current_hash),
            "temporal_membership": temporal_class,
        })
    temporal_split = pd.DataFrame(temporal_rows)

    leakage_rows = [
        leakage_row("sequence_30_main", "protein_id", sequence_split["protein_id"], sequence_split["split_30"]),
        leakage_row("sequence_30_main", "cluster_id_30", sequence_split["cluster_id_30"], sequence_split["split_30"]),
        leakage_row("sequence_40_sensitivity", "cluster_id_40", sequence_split["cluster_id_40"], sequence_split["split_40"]),
        leakage_row("sequence_50_sensitivity", "cluster_id_50", sequence_split["cluster_id_50"], sequence_split["split_50"]),
        leakage_row("family", "primary_pfam", family_split["primary_pfam"], family_split["pfam_family_split"]),
        leakage_row("family", "primary_pfam_clan", family_split["primary_pfam_clan"], family_split["pfam_clan_split"]),
        leakage_row("structure", "primary_cath_superfamily", structure_split["primary_cath_superfamily"], structure_split["cath_superfamily_split"]),
        leakage_row("taxonomy", "taxonomy_group", taxonomy_split["taxonomy_group"], taxonomy_split["taxonomy_species_split"]),
    ]
    reaction_exploded = reaction_split[reaction_split["reaction_holdout_split"].isin(["train", "validation", "test"])].copy()
    reaction_exploded["group"] = reaction_exploded["canonical_rhea_groups_json"].map(json.loads)
    reaction_exploded = reaction_exploded.explode("group")
    leakage_rows.append(leakage_row("reaction", "canonical_rhea", reaction_exploded["group"], reaction_exploded["reaction_holdout_split"]))
    leakage = pd.DataFrame(leakage_rows)

    outputs = {
        "split_sequence": split_dir / "split_sequence.parquet",
        "split_family": split_dir / "split_family.parquet",
        "split_structure": split_dir / "split_structure.parquet",
        "split_taxonomy": split_dir / "split_taxonomy.parquet",
        "split_reaction_holdout": split_dir / "split_reaction_holdout.parquet",
        "split_temporal": split_dir / "split_temporal.parquet",
    }
    for name, frame in [
        ("split_sequence", sequence_split),
        ("split_family", family_split),
        ("split_structure", structure_split),
        ("split_taxonomy", taxonomy_split),
        ("split_reaction_holdout", reaction_split),
        ("split_temporal", temporal_split),
    ]:
        write_parquet(frame, outputs[name])
    leakage_path = split_dir / "split_leakage_report.tsv"
    leakage.to_csv(leakage_path, sep="\t", index=False)

    split_counts = sequence_split["split"].value_counts().to_dict()
    split_fractions = {key: value / len(sequence_split) for key, value in split_counts.items()}
    sensitivity_split_counts = {
        identity: sequence_split[f"split_{identity}"].value_counts().to_dict()
        for identity in (40, 50)
    }
    sensitivity_split_fractions = {
        identity: {key: value / len(sequence_split) for key, value in counts.items()}
        for identity, counts in sensitivity_split_counts.items()
    }
    raw_hash_after = sha256_file(raw_inventory)
    checks = [
        ("checkpoint_02_present", (checkpoints / "CHECKPOINT_02_PASS").exists(), "Phase 2 prerequisite"),
        ("benchmark_nonempty", len(benchmark) >= 100000, f"observed={len(benchmark)}"),
        ("mmseqs_30_complete", len(cluster_maps["cluster_id_30"]) == len(benchmark), f"mapped={len(cluster_maps['cluster_id_30'])}"),
        ("mmseqs_40_complete", len(cluster_maps["cluster_id_40"]) == len(benchmark), f"mapped={len(cluster_maps['cluster_id_40'])}"),
        ("mmseqs_50_complete", len(cluster_maps["cluster_id_50"]) == len(benchmark), f"mapped={len(cluster_maps['cluster_id_50'])}"),
        ("all_leakage_checks_pass", bool((leakage["status"] == "PASS").all()), f"failures={(leakage['status'] != 'PASS').sum()}"),
        ("sequence_split_ratio_tolerance", all(abs(split_fractions.get(name, 0.0) - target) <= 0.05 for name, target in RATIOS), json_text(split_fractions)),
        ("sequence_40_split_ratio_tolerance", all(abs(sensitivity_split_fractions[40].get(name, 0.0) - target) <= 0.05 for name, target in RATIOS), json_text(sensitivity_split_fractions[40])),
        ("sequence_50_split_ratio_tolerance", all(abs(sensitivity_split_fractions[50].get(name, 0.0) - target) <= 0.05 for name, target in RATIOS), json_text(sensitivity_split_fractions[50])),
        ("temporal_sequence_stable_nonempty", int(temporal_split["T0_T1_sequence_stable"].sum()) >= 50000, f"observed={temporal_split['T0_T1_sequence_stable'].sum()}"),
        ("reaction_holdout_nonempty", int(reaction_split["reaction_holdout_split"].isin(["train", "validation", "test"]).sum()) >= 50000, f"observed={reaction_split['reaction_holdout_split'].isin(['train','validation','test']).sum()}"),
        ("no_pairs_generated", not (project / "data" / "processed" / "population_pairs.parquet").exists() and not (project / "data" / "processed" / "training_pairs.parquet").exists(), "Phase 3 split-before-pairs guard"),
        ("raw_inventory_unchanged", raw_hash_before == raw_hash_after, raw_hash_after),
        ("all_outputs_exist", all(path.exists() and path.stat().st_size > 0 for path in outputs.values()) and leakage_path.exists(), "six split tables plus leakage report"),
    ]
    qc = pd.DataFrame([(name, "PASS" if passed else "FAIL", details) for name, passed, details in checks], columns=["check", "status", "details"])
    qc.to_csv(reports / "phase03_qc.tsv", sep="\t", index=False)
    failures = qc[qc["status"] == "FAIL"]

    report = [
        "# SiteGuard V4 Phase 3 Report",
        "",
        f"- Started: `{started}`",
        f"- Completed: `{utc_now()}`",
        f"- Decision: `{'PASS' if failures.empty else 'FAIL'}`",
        f"- Split-rule SHA-256: `{sha256_file(rules)}`",
        f"- Benchmark proteins: **{len(benchmark):,}**",
        "",
        "## Sequence clustering",
        "",
        "| Identity | Clusters | Coverage rule |",
        "|---:|---:|---|",
        f"| 30% | {sequence_split['cluster_id_30'].nunique():,} | >=70% bidirectional |",
        f"| 40% | {sequence_split['cluster_id_40'].nunique():,} | >=70% bidirectional |",
        f"| 50% | {sequence_split['cluster_id_50'].nunique():,} | >=70% bidirectional |",
        "",
        "## Main split",
        "",
    ]
    for split, count in sorted(split_counts.items()):
        report.append(f"- `{split}`: {count:,} proteins ({count / len(sequence_split):.2%})")
    report += [
        "",
        "The 40% and 50% sensitivity splits are assigned independently at their own cluster level because separate MMseqs greedy cluster runs are not mathematically nested. The `split` column is an alias of the frozen 30% main split.",
        "",
        "## Sensitivity split sizes",
        "",
    ]
    for identity in (40, 50):
        for split, count in sorted(sensitivity_split_counts[identity].items()):
            report.append(f"- `{identity}% {split}`: {count:,} proteins ({count / len(sequence_split):.2%})")
    report += [
        "",
        "## Holdout coverage",
        "",
        f"- Pfam assigned: {(family_split['primary_pfam'] != 'UNASSIGNED').sum():,}",
        f"- CATH superfamily assigned: {(structure_split['primary_cath_superfamily'] != 'UNASSIGNED').sum():,}",
        f"- Strict reaction-holdout assigned: {reaction_split['reaction_holdout_split'].isin(['train','validation','test']).sum():,}",
        f"- T0–T1 sequence-stable: {temporal_split['T0_T1_sequence_stable'].sum():,}",
        "",
        "Foldseek structure-cluster membership is explicitly deferred to Phase 4 and will be appended without changing the already frozen protein sequence split.",
        "",
        "## Leakage checks",
        "",
        "| Scheme | Unit | Overlap | Status |",
        "|---|---|---:|---|",
    ]
    for _, row in leakage.iterrows():
        report.append(f"| {row['scheme']} | {row['unit']} | {int(row['overlap_count'])} | {row['status']} |")
    report += ["", "No pair table was generated in Phase 3.", ""]
    (reports / "PHASE_03_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    summary = {
        "phase": 3,
        "project_version": "V4",
        "status": "PASS" if failures.empty else "FAIL",
        "started_at": started,
        "completed_at": utc_now(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "seed": seed,
        "benchmark_proteins": len(benchmark),
        "cluster_counts": {"30": int(sequence_split["cluster_id_30"].nunique()), "40": int(sequence_split["cluster_id_40"].nunique()), "50": int(sequence_split["cluster_id_50"].nunique())},
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "sensitivity_split_counts": {
            str(identity): {key: int(value) for key, value in counts.items()}
            for identity, counts in sensitivity_split_counts.items()
        },
        "leakage_failures": leakage[leakage["status"] != "PASS"].to_dict("records"),
        "qc_failures": failures.to_dict("records"),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    (reports / "phase03_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checkpoint = checkpoints / "CHECKPOINT_03_PASS"
    if not failures.empty:
        checkpoint.unlink(missing_ok=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2
    checkpoint.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "finalize"])
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.project_root.resolve(), args.source_root.resolve(), args.seed)
        return 0
    return finalize(args.project_root.resolve(), args.source_root.resolve(), args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
