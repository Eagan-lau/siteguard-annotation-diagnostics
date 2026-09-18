#!/usr/bin/env python3
"""Freeze Phase 15 family and P450 external-validation cohorts before scoring."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


SEED = 20260819
COMPLETE_EC = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$")
EC_L3_PREFIX = re.compile(r"^([0-9]+\.[0-9]+\.[0-9]+)(?:\.[0-9-]+)?$")
UNIPROT_ACCESSION = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]|"
    r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){2})$"
)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cyp_family(value: object) -> str:
    match = re.search(r"CYP(\d+)", str(value).upper())
    return f"CYP{match.group(1)}" if match else "UNKNOWN_CYP_FAMILY"


def species_group(value: object) -> str:
    text = str(value).strip().lower()
    if text == "plant" or any(token in text for token in ["viridiplantae", "streptophyta", "plant"]):
        return "Plant"
    if "fung" in text:
        return "Fungi"
    if any(token in text for token in ["animal", "metazoa"]):
        return "Animal"
    if "bacter" in text:
        return "Bacteria"
    if "archaea" in text:
        return "Archaea"
    return "Other"


def rhea_direction_map(path: Path) -> dict[int, str]:
    frame = pd.read_csv(path, sep="\t")
    output: dict[int, str] = {}
    for row in frame.itertuples(index=False):
        master = f"RHEA:{int(row.RHEA_ID_MASTER)}"
        for identifier in [row.RHEA_ID_MASTER, row.RHEA_ID_LR, row.RHEA_ID_RL, row.RHEA_ID_BI]:
            output[int(identifier)] = master
    return output


def old_sequences(path: Path, wanted: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            protein_id, sequence = line.rstrip("\n").split("\t", 1)
            if protein_id in wanted:
                found[protein_id] = sequence
    return found


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in frame[["accession", "sequence"]].sort_values("accession").itertuples(index=False):
            handle.write(f">{row.accession}\n")
            for start in range(0, len(row.sequence), 80):
                handle.write(row.sequence[start:start + 80] + "\n")
    temporary.replace(path)


def select_general_families(root: Path, results: Path) -> pd.DataFrame:
    gold = pd.read_parquet(
        root / "data/processed/single_documented_activity_gold_canonical.parquet",
        columns=["protein_id", "ec_l1", "ec_l4", "canonical_rhea"],
    )
    split = pd.read_parquet(
        root / "data/splits/split_sequence.parquet", columns=["protein_id", "split"],
    )
    families = pd.read_parquet(
        root / "data/splits/split_family.parquet", columns=["protein_id", "primary_pfam"],
    )
    site_ids = set(pd.read_parquet(
        root / "data/processed/catalytic_site_table.parquet", columns=["protein_id"],
    )["protein_id"].dropna().astype(str))
    frame = gold.merge(split, on="protein_id", how="inner", validate="many_to_one")
    frame = frame.merge(families, on="protein_id", how="left", validate="many_to_one")
    frame["primary_pfam"] = frame["primary_pfam"].fillna("UNASSIGNED")
    frame["has_site"] = frame["protein_id"].isin(site_ids)
    selection = frame.loc[frame["split"].isin(["train", "validation"]) & frame["primary_pfam"].ne("UNASSIGNED")].copy()
    rows: list[dict[str, Any]] = []
    for family, group in selection.groupby("primary_pfam", sort=True):
        counts = group["ec_l1"].astype(str).value_counts()
        proteins = group["protein_id"].nunique()
        ec4 = group["ec_l4"].dropna().astype(str).nunique()
        rhea = group["canonical_rhea"].dropna().astype(str).nunique()
        site_coverage = float(group.groupby("protein_id")["has_site"].max().mean())
        rows.append({
            "family": family,
            "train_validation_gold_proteins": proteins,
            "train_validation_distinct_ec4": ec4,
            "train_validation_distinct_rhea": rhea,
            "train_validation_site_coverage": site_coverage,
            "dominant_ec_l1": str(counts.index[0]) if len(counts) else "",
            "selection_score": proteins * math.log1p(ec4 + rhea) * (0.5 + site_coverage),
        })
    protocol = pd.DataFrame(rows)
    protocol["eligible"] = (
        protocol["train_validation_gold_proteins"].ge(20)
        & protocol["train_validation_distinct_ec4"].ge(3)
        & protocol["train_validation_distinct_rhea"].ge(3)
    )
    protocol["rank_within_ec_l1"] = (
        protocol.loc[protocol["eligible"]]
        .groupby("dominant_ec_l1")["selection_score"]
        .rank(method="first", ascending=False)
    )
    chosen: list[str] = []
    for ec_l1 in sorted(protocol.loc[protocol["eligible"], "dominant_ec_l1"].unique()):
        group = protocol.loc[protocol["eligible"] & protocol["dominant_ec_l1"].eq(ec_l1)].sort_values(
            ["selection_score", "family"], ascending=[False, True],
        )
        if len(group):
            chosen.append(str(group.iloc[0]["family"]))
        if len(chosen) == 6:
            break
    if len(chosen) < 6:
        remaining = protocol.loc[protocol["eligible"] & ~protocol["family"].isin(chosen)].sort_values(
            ["selection_score", "family"], ascending=[False, True],
        )
        chosen.extend(remaining["family"].astype(str).head(6 - len(chosen)).tolist())
    protocol["selected"] = protocol["family"].isin(chosen)
    protocol["selection_rule"] = (
        "train/validation-only canonical Gold: >=20 proteins, >=3 EC-L4, >=3 canonical Rhea; "
        "top score per dominant EC-L1, then highest remaining; maximum 6 families"
    )
    protocol["test_performance_used_for_selection"] = False
    protocol["scientific_go7_rule"] = (
        ">=3 selected families with >=20 test queries, EC-L3 coverage >=0.05 and selective accuracy >=0.75"
    )
    protocol = protocol.sort_values(["selected", "selection_score", "family"], ascending=[False, False, True])
    protocol.to_csv(results / "selected_family_protocol.tsv", sep="\t", index=False)
    return protocol.loc[protocol["selected"]].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root, source = args.project_root.resolve(), args.source_root.resolve()
    work = root / "data/interim/phase15"
    results = root / "results/phase15"
    reports = root / "reports"
    for path in [work, results, reports]:
        path.mkdir(parents=True, exist_ok=True)
    if not (root / "checkpoints/CHECKPOINT_14_PASS").is_file():
        raise RuntimeError("CHECKPOINT_14_PASS is required")

    selected = select_general_families(root, results)
    p450 = pd.read_csv(source / "data/raw/cyp450/p450rdb_v2/2. P450s.CSV", encoding="latin1").fillna("")
    p450["source_uniprot_id"] = p450["Uniprot ID"].astype(str).str.strip().str.upper()
    p450["accession"] = p450["source_uniprot_id"].str.split(r"[,;\s]+", regex=True)
    p450 = p450.explode("accession")
    p450["sequence"] = p450["Sequence"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    p450["cyp_family"] = p450["Symbol"].map(cyp_family)
    p450["species_group"] = p450["Species"].map(species_group)
    valid = p450["sequence"].ne("") & p450["accession"].map(lambda value: bool(UNIPROT_ACCESSION.fullmatch(str(value))))
    excluded = p450.loc[p450["sequence"].ne("") & ~valid].copy()
    excluded["exclusion_reason"] = excluded["accession"].map(
        lambda value: "MISSING_UNIPROT_ACCESSION" if not str(value) else "INVALID_UNIPROT_ACCESSION_FORMAT"
    )
    excluded[["Symbol", "Species", "source_uniprot_id", "accession", "exclusion_reason"]].to_csv(
        results / "p450rdb_excluded_records.tsv", sep="\t", index=False,
    )
    p450 = p450.loc[valid].drop_duplicates("accession").copy()

    benchmark = set(pd.read_parquet(
        root / "data/splits/split_sequence.parquet", columns=["protein_id"],
    )["protein_id"].astype(str))
    protein = pd.read_parquet(
        root / "data/processed/protein_table.parquet",
        columns=["protein_id", "pfam_domains_json", "lineage_json"],
    )
    family = pd.read_parquet(
        root / "data/splits/split_family.parquet",
        columns=["protein_id", "primary_pfam", "primary_pfam_clan"],
    )
    metadata = protein.merge(family, on="protein_id", how="left", validate="one_to_one").set_index("protein_id")
    membership = pd.read_parquet(
        root / "data/interim/phase04/afdb_membership.parquet", columns=["protein_id", "has_structure"],
    ).set_index("protein_id")["has_structure"].to_dict()
    index = pd.read_csv(source / "05_features/esm2_t33_index.tsv", sep="\t", dtype={"protein_id": str})
    embedding_row = index.set_index("protein_id")["row_index"].astype(int).to_dict()
    wanted = set(p450["accession"])
    sequences = old_sequences(source / "05_features/esm2_required_sequences.tsv.gz", wanted)
    exact = p450.apply(lambda row: sequences.get(row["accession"]) == row["sequence"], axis=1)
    if not exact.all() or not wanted.issubset(embedding_row):
        raise RuntimeError(
            f"P450 ESM2 reuse failed: exact={int(exact.sum())}/{len(exact)} "
            f"missing_rows={len(wanted - set(embedding_row))}"
        )
    p450["sequence_length"] = p450["sequence"].str.len().astype(int)
    p450["sequence_sha256"] = p450["sequence"].map(lambda value: hashlib.sha256(value.encode("ascii")).hexdigest())
    p450["esm2_source_row"] = p450["accession"].map(embedding_row).astype(int)
    p450["benchmark_overlap"] = p450["accession"].isin(benchmark)
    p450["evaluation_cohort"] = p450["benchmark_overlap"].map({
        False: "STRICT_EXTERNAL_ACCESSION", True: "FROZEN_BENCHMARK_OVERLAP_SENSITIVITY",
    })
    p450["pfam_ids_json"] = p450["accession"].map(metadata["pfam_domains_json"]).fillna("[]")
    p450["primary_pfam"] = p450["accession"].map(metadata["primary_pfam"]).fillna("")
    p450["primary_pfam_clan"] = p450["accession"].map(metadata["primary_pfam_clan"]).fillna("")
    p450["query_structure_available"] = p450["accession"].map(membership).fillna(False).astype(bool)
    query_columns = [
        "accession", "Symbol", "Name", "Species name", "Species", "Txid", "cyp_family", "species_group",
        "sequence", "sequence_length", "sequence_sha256", "esm2_source_row", "benchmark_overlap",
        "evaluation_cohort", "pfam_ids_json", "primary_pfam", "primary_pfam_clan", "query_structure_available",
    ]
    query = p450[query_columns].copy()
    query.to_parquet(work / "p450_query_metadata.parquet", index=False, compression="zstd")
    write_fasta(query, work / "p450_queries.fasta")

    rhea_map = rhea_direction_map(source / "data/raw/rhea/release_141/extracted/141/tsv/rhea-directions.tsv")
    reaction_catalog = pd.read_parquet(root / "data/processed/reaction_table.parquet")
    reaction_catalog = reaction_catalog.loc[reaction_catalog["release"].eq(141)].drop_duplicates("canonical_rhea")
    reaction_ec = {
        str(row.canonical_rhea): {
            value for value in json.loads(row.ec_ids_json) if COMPLETE_EC.fullmatch(str(value))
        }
        for row in reaction_catalog.itertuples(index=False)
    }
    reactions = pd.read_csv(source / "data/raw/cyp450/p450rdb_v2/1. Reactions.CSV", encoding="latin1").fillna("")
    reaction_rows: list[dict[str, Any]] = []
    wanted = set(query["accession"])
    for source_index, row in enumerate(reactions.to_dict("records")):
        accessions = {value for value in re.split(r"[,;\s]+", str(row.get("Uniprot ID", "")).strip().upper()) if value in wanted}
        if not accessions:
            continue
        direct_ec = {
            value.strip() for value in re.split(r"[;,]", str(row.get("EC number", "")))
            if COMPLETE_EC.fullmatch(value.strip())
        }
        direct_ec3 = {
            match.group(1) for value in re.split(r"[;,]", str(row.get("EC number", "")))
            if (match := EC_L3_PREFIX.fullmatch(value.strip())) is not None
        }
        rheas = set()
        for token in re.findall(r"[0-9]+", str(row.get("RheaID", ""))):
            if int(token) in rhea_map:
                rheas.add(rhea_map[int(token)])
        mapped_ec = set(direct_ec)
        for rhea in rheas:
            mapped_ec.update(reaction_ec.get(rhea, set()))
        for accession in accessions:
            reaction_rows.append({
                "accession": accession, "source_reaction_row": source_index,
                "ec_l3_json": json_text(sorted(direct_ec3 | {".".join(value.split(".")[:3]) for value in mapped_ec})),
                "ec_l4_json": json_text(sorted(mapped_ec)),
                "canonical_rhea_json": json_text(sorted(rheas)),
                "substrates_json": json_text([
                    {"name": str(row.get(f"Substrate{i}", "")).strip(), "smiles": str(row.get(f"sub_Smiles{i}", "")).strip()}
                    for i in range(1, 5) if str(row.get(f"Substrate{i}", "")).strip()
                ]),
                "products_json": json_text([
                    {"name": str(row.get(f"Product{i}", "")).strip(), "smiles": str(row.get(f"pro_Smiles{i}", "")).strip()}
                    for i in range(1, 7) if str(row.get(f"Product{i}", "")).strip()
                ]),
                "transformation": str(row.get("Transformations", "")).strip(),
                "pmid": str(row.get("PMID", "")).strip(),
                "doi": str(row.get("DOI", "")).strip(),
                "truth_provenance": "P450RDB_V2_INDEPENDENT_REACTION_RECORD",
            })
    reaction_truth = pd.DataFrame(reaction_rows)
    reaction_truth.to_parquet(work / "p450_reaction_truth.parquet", index=False, compression="zstd")
    p450_summary_ec = p450.set_index("accession")["EC number"].to_dict()
    truth_rows: list[dict[str, Any]] = []
    for accession in sorted(wanted):
        group = reaction_truth.loc[reaction_truth["accession"].eq(accession)]
        rheas: set[str] = set()
        for value in group["canonical_rhea_json"]:
            rheas.update(json.loads(value))
        ec4: set[str] = {
            value.strip() for value in re.split(r"[;,]", str(p450_summary_ec.get(accession, "")))
            if COMPLETE_EC.fullmatch(value.strip())
        }
        ec3: set[str] = {
            match.group(1) for value in re.split(r"[;,]", str(p450_summary_ec.get(accession, "")))
            if (match := EC_L3_PREFIX.fullmatch(value.strip())) is not None
        }
        for value in group["ec_l4_json"]:
            ec4.update(json.loads(value))
        for value in group["ec_l3_json"]:
            ec3.update(json.loads(value))
        truth_rows.append({
            "accession": accession,
            "truth_ec_l3_json": json_text(sorted(ec3)),
            "truth_ec_l4_json": json_text(sorted(ec4)),
            "truth_rhea_json": json_text(sorted(rheas)),
            "independent_reaction_records": len(group),
        })
    truth = query[["accession", "evaluation_cohort", "cyp_family", "species_group"]].merge(
        pd.DataFrame(truth_rows), on="accession", how="left", validate="one_to_one",
    )
    for column in ["truth_ec_l3_json", "truth_ec_l4_json", "truth_rhea_json"]:
        truth[column] = truth[column].fillna("[]")
    truth["independent_reaction_records"] = truth["independent_reaction_records"].fillna(0).astype(int)
    truth.to_parquet(work / "p450_query_truth.parquet", index=False, compression="zstd")

    plant = pd.read_parquet(source / "data/staging/cyp450/plantp450_entries.parquet")
    plant_output = plant[[
        "cyp_name_catalog", "family_catalog", "species_catalog", "taxa_catalog", "compound_class",
        "biosynthetic_pathway", "function", "accession", "references", "detail_status",
    ]].copy()
    plant_output["has_function_description"] = plant_output["function"].astype(str).str.strip().ne("")
    plant_accession = plant_output["accession"].fillna("").map(lambda value: str(value).strip().upper())
    plant_output["uniprot_accession_format"] = plant_accession.map(
        lambda value: bool(UNIPROT_ACCESSION.fullmatch(str(value)))
    )
    plant_output["model_evaluable_by_frozen_sequence"] = plant_accession.isin(wanted)
    plant_output["interpretation"] = "descriptive external catalog; missing sequence is not a negative prediction"
    plant_output.to_csv(results / "plant_p450_catalog_coverage.tsv", sep="\t", index=False)

    strict = truth.loc[truth["evaluation_cohort"].eq("STRICT_EXTERNAL_ACCESSION")]
    counts = {
        "phase": 15, "stage": "protocol_and_external_cohort_freeze", "status": "PASS",
        "slurm_job_id": os.getenv("SLURM_JOB_ID", "NA"), "seed": SEED,
        "selected_general_families": selected["family"].astype(str).tolist(),
        "p450_unique_exact_sequence_queries": len(query),
        "p450_strict_external_accessions": len(strict),
        "p450_strict_external_with_ec3": int(strict["truth_ec_l3_json"].ne("[]").sum()),
        "p450_strict_external_with_ec4": int(strict["truth_ec_l4_json"].ne("[]").sum()),
        "p450_strict_external_with_rhea": int(strict["truth_rhea_json"].ne("[]").sum()),
        "p450_benchmark_overlap_sensitivity": int(query["benchmark_overlap"].sum()),
        "plantp450_catalog_entries": len(plant_output),
        "plantp450_function_descriptions": int(plant_output["has_function_description"].sum()),
        "query_truth_used_as_model_input": False,
        "test_performance_used_for_family_selection": False,
        "primary_external_cohort": "STRICT_EXTERNAL_ACCESSION",
        "scenarios_frozen": ["GENERAL", "LEAVE_ONE_CYP_FAMILY_OUT", "PLANT_COLD_START"],
        "go7_rule_frozen_before_scoring": True,
        "cyp_go7_rule": "strict-external EC-L3: >=50 labeled queries, >=5 accepted, retrieval recall >=0.50, coverage >=0.05, selective accuracy >=0.75",
    }
    if (
        len(selected) < 3 or len(query) != 821 or len(strict) < 300
        or int(strict["truth_ec_l3_json"].ne("[]").sum()) < 50
        or int(strict["truth_rhea_json"].ne("[]").sum()) < 150
    ):
        raise RuntimeError(f"Unexpected P450 external cohort: {counts}")
    (reports / "phase15_prepare_summary.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
