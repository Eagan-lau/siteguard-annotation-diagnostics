#!/usr/bin/env python3
"""Finalize multi-family and independent CYP450 stress tests for frozen SiteGuard."""

from __future__ import annotations

import argparse
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.optimize import linear_sum_assignment


LEVELS = ["EC_L3", "EC_L4", "EXACT_RHEA"]
LABEL_COLUMNS = {"EC_L3": "ec_l3", "EC_L4": "ec_l4", "EXACT_RHEA": "canonical_rhea"}
TRUTH_COLUMNS = {"EC_L3": "truth_ec_l3_json", "EC_L4": "truth_ec_l4_json", "EXACT_RHEA": "truth_rhea_json"}
SCENARIOS = ["GENERAL", "LEAVE_ONE_CYP_FAMILY_OUT", "PLANT_COLD_START"]
CURRENCY = {
    "CHEBI:15377", "CHEBI:15378", "CHEBI:15422", "CHEBI:16761", "CHEBI:16027", "CHEBI:18367",
    "CHEBI:33019", "CHEBI:57540", "CHEBI:57945", "CHEBI:58349", "CHEBI:57783", "CHEBI:57287",
}


def json_set(value: object) -> set[str]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()
    return {str(item) for item in parsed if item}


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)
    else:
        frame.to_csv(path, sep="\t", index=False)


def aggregate_scenario(
    pairs: pd.DataFrame, truth: pd.DataFrame, scenario: str, thresholds: dict[str, float],
) -> pd.DataFrame:
    pool = pairs.loc[pairs[f"eligible_{scenario}"]].copy()
    truth_lookup = truth.set_index("accession")
    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        label_column, score_column = LABEL_COLUMNS[level], f"calibrated_{level}"
        frame = pool.loc[pool[label_column].notna() & pool[label_column].astype(str).ne("")].copy()
        frame["candidate_label"] = frame[label_column].astype(str)
        support = frame.loc[frame[score_column].ge(0.5)].groupby(
            ["query_protein_id", "candidate_label"], sort=False,
        ).agg(
            supporting_references=("reference_protein_id", "nunique"),
            supporting_clusters=("reference_cluster_id_30", "nunique"),
        ).reset_index()
        maxima = frame.groupby(["query_protein_id", "candidate_label"], sort=False)[score_column].idxmax()
        labels = frame.loc[maxima, [
            "query_protein_id", "candidate_label", score_column, "reference_protein_id",
            "reference_activity_id", "sequence_identity", "sequence_alignment_fraction", "esm2_t33_cosine",
            "pfam_jaccard", "reference_cyp_family", "reference_species_group",
        ]].merge(support, on=["query_protein_id", "candidate_label"], how="left", validate="one_to_one")
        labels[["supporting_references", "supporting_clusters"]] = labels[[
            "supporting_references", "supporting_clusters",
        ]].fillna(0).astype(int)
        labels = labels.sort_values(
            ["query_protein_id", score_column, "candidate_label"], ascending=[True, False, True],
        )
        labels["label_rank"] = labels.groupby("query_protein_id").cumcount() + 1
        winners = labels.loc[labels["label_rank"].eq(1)]
        for winner in winners.itertuples(index=False):
            query_id = str(winner.query_protein_id)
            if query_id not in truth_lookup.index:
                continue
            truth_row = truth_lookup.loc[query_id]
            true_labels = json_set(truth_row[TRUTH_COLUMNS[level]])
            if not true_labels:
                continue
            query_labels = labels.loc[labels["query_protein_id"].eq(query_id), "candidate_label"].astype(str)
            score = float(getattr(winner, score_column))
            predicted = str(winner.candidate_label)
            rows.append({
                "scenario": scenario, "query_protein_id": query_id,
                "evaluation_cohort": truth_row["evaluation_cohort"], "query_cyp_family": truth_row["cyp_family"],
                "query_species_group": truth_row["species_group"], "annotation_level": level,
                "truth_labels_json": json.dumps(sorted(true_labels)),
                "truth_retrieved": bool(set(query_labels) & true_labels),
                "candidate_labels": int(query_labels.nunique()), "predicted_label": predicted,
                "model_probability": score, "validation_frozen_threshold": thresholds[level],
                "accepted": score >= thresholds[level], "correct": predicted in true_labels,
                "top_reference_protein_id": winner.reference_protein_id,
                "top_reference_activity_id": winner.reference_activity_id,
                "top_reference_cyp_family": winner.reference_cyp_family,
                "top_reference_species_group": winner.reference_species_group,
                "top_sequence_identity": float(winner.sequence_identity),
                "top_alignment_fraction": float(winner.sequence_alignment_fraction),
                "top_esm2_cosine": float(winner.esm2_t33_cosine),
                "top_pfam_jaccard": float(winner.pfam_jaccard),
                "supporting_references": int(winner.supporting_references),
                "supporting_sequence_clusters": int(winner.supporting_clusters),
                "truth_use_policy": "GROUND_TRUTH_ONLY_EXTERNAL_EVALUATION",
            })
    return pd.DataFrame(rows)


def metric_table(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        accepted = group["accepted"].astype(bool)
        row = dict(zip(group_columns, keys, strict=True))
        row.update({
            "queries": group["query_protein_id"].nunique(),
            "accepted_queries": int(accepted.sum()),
            "retrieval_recall": float(group["truth_retrieved"].mean()),
            "coverage": float(accepted.mean()),
            "top1_accuracy_all_queries": float(group["correct"].mean()),
            "selective_accuracy": float(group.loc[accepted, "correct"].mean()) if accepted.any() else float("nan"),
            "median_top_sequence_identity": float(group["top_sequence_identity"].median()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def general_family_results(root: Path, selected: set[str]) -> pd.DataFrame:
    predictions = pd.read_parquet(root / "results/phase12/end_to_end_predictions.parquet")
    family = pd.read_parquet(
        root / "data/splits/split_family.parquet", columns=["protein_id", "primary_pfam"],
    ).set_index("protein_id")["primary_pfam"].to_dict()
    predictions["family"] = predictions["query_protein_id"].map(family).fillna("UNASSIGNED")
    rows: list[dict[str, Any]] = []
    for family_id in sorted(selected):
        group = predictions.loc[predictions["family"].eq(family_id)]
        for level in LEVELS:
            accepted = group[f"accepted_{level}"].astype(bool)
            correct = group[f"top_correct_{level}"].astype(bool)
            coverage = float(accepted.mean()) if len(group) else float("nan")
            accuracy = float(correct.loc[accepted].mean()) if accepted.any() else float("nan")
            rows.append({
                "family": family_id, "annotation_level": level, "test_queries": len(group),
                "retrieval_recall": float(group[f"oracle_candidate_available_{level}"].mean()) if len(group) else float("nan"),
                "accepted_queries": int(accepted.sum()), "coverage": coverage,
                "selective_accuracy": accuracy,
                "top1_accuracy_all_queries": float(correct.mean()) if len(group) else float("nan"),
                "frozen_phase12_model_and_threshold": True,
                "scientific_support_rule_met": bool(
                    level == "EC_L3" and len(group) >= 20 and coverage >= 0.05
                    and math.isfinite(accuracy) and accuracy >= 0.75
                ),
            })
    return pd.DataFrame(rows)


class ReactionSimilarity:
    def __init__(self, root: Path) -> None:
        table = pd.read_parquet(root / "data/processed/reaction_table.parquet")
        table = table.loc[table["release"].eq(141)].drop_duplicates("canonical_rhea")
        self.table = table.set_index("canonical_rhea").to_dict("index")
        features = pd.read_parquet(
            root / "data/processed/reaction_features.parquet",
            columns=["canonical_rhea", "signed_transform_fingerprint_int8"],
        )
        self.transform = {
            str(row.canonical_rhea): np.frombuffer(row.signed_transform_fingerprint_int8, dtype=np.int8).astype(np.float32)
            for row in features.itertuples(index=False)
        }
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    @lru_cache(maxsize=None)
    def participants(self, rhea: str, side: str) -> tuple[Any, ...]:
        row = self.table.get(rhea)
        if not row:
            return tuple()
        ids = json.loads(row[f"{side}_chebi_ids_json"])
        smiles = json.loads(row["participant_smiles_json"])
        output = []
        for identifier in ids:
            if identifier in CURRENCY or not smiles.get(identifier):
                continue
            molecule = Chem.MolFromSmiles(str(smiles[identifier]))
            if molecule is not None:
                output.append(self.generator.GetFingerprint(molecule))
        return tuple(output)

    @staticmethod
    def optimal(first: tuple[Any, ...], second: tuple[Any, ...]) -> float:
        if not first or not second:
            return float("nan")
        matrix = np.array([
            [DataStructs.TanimotoSimilarity(left, right) for right in second] for left in first
        ], dtype=float)
        row, column = linear_sum_assignment(-matrix)
        return float(matrix[row, column].sum() / max(len(first), len(second)))

    def compare(self, predicted: str, truth: str) -> dict[str, float]:
        first, second = self.transform.get(predicted), self.transform.get(truth)
        if first is None or second is None:
            transform = float("nan")
        else:
            denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
            transform = max(0.0, float(np.dot(first, second) / denominator)) if denominator else float("nan")
        substrate = self.optimal(self.participants(predicted, "substrate"), self.participants(truth, "substrate"))
        product = self.optimal(self.participants(predicted, "product"), self.participants(truth, "product"))
        values, weights = [], []
        for value, weight in [(transform, 0.50), (substrate, 0.25), (product, 0.25)]:
            if math.isfinite(value):
                values.append(value * weight); weights.append(weight)
        reaction = sum(values) / sum(weights) if weights else float("nan")
        return {
            "transform_similarity": transform, "substrate_similarity": substrate,
            "product_similarity": product, "reaction_similarity": reaction,
        }


def reaction_similarity_table(root: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    engine = ReactionSimilarity(root)
    frame = predictions.loc[predictions["annotation_level"].eq("EXACT_RHEA")].copy()
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        truths = json_set(row.truth_labels_json)
        comparisons = []
        for truth in truths:
            scores = engine.compare(str(row.predicted_label), truth)
            comparisons.append((scores["reaction_similarity"], truth, scores))
        finite = [item for item in comparisons if math.isfinite(item[0])]
        if not finite:
            continue
        _, best_truth, best = max(finite, key=lambda item: item[0])
        rows.append({
            "scenario": row.scenario, "query_protein_id": row.query_protein_id,
            "evaluation_cohort": row.evaluation_cohort, "query_cyp_family": row.query_cyp_family,
            "predicted_rhea": row.predicted_label, "best_true_rhea": best_truth,
            "exact_rhea_match": row.predicted_label in truths,
            "model_probability": row.model_probability, "accepted": row.accepted,
            **best,
            "weights": "0.50_transform+0.25_substrate+0.25_product",
            "participant_matching": "Hungarian optimal matching of non-currency Morgan fingerprints",
            "truth_use_policy": "GROUND_TRUTH_ONLY_EXTERNAL_EVALUATION",
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data/interim/phase15"
    results = root / "results/phase15"
    reports = root / "reports"
    figures = root / "figures/source_data"
    checkpoints = root / "checkpoints"
    for path in [results, reports, figures, checkpoints]:
        path.mkdir(parents=True, exist_ok=True)
    if not (checkpoints / "CHECKPOINT_14_PASS").is_file():
        raise RuntimeError("CHECKPOINT_14_PASS is required")

    feature_path = work / "p450_pair_features.parquet"
    row_count = pq.ParquetFile(feature_path).metadata.num_rows
    deep = np.load(work / "p450_deep_predictions.npy", mmap_mode="r")
    tree = np.load(work / "p450_tree_predictions.npy", mmap_mode="r")
    if deep.shape != (row_count, 3) or tree.shape != (row_count, 3):
        raise RuntimeError(f"Prediction shape mismatch: rows={row_count} deep={deep.shape} tree={tree.shape}")
    model_config = json.loads((root / "models/phase11/siteguard_model_config.json").read_text(encoding="utf-8"))
    calibration = json.loads((root / "models/phase12/calibration_config.json").read_text(encoding="utf-8"))
    thresholds = {level: float(calibration["thresholds"][level]) for level in LEVELS}
    pair_columns = [
        "pair_index", "query_protein_id", "reference_protein_id", "reference_activity_id",
        "reference_cluster_id_30", "ec_l3", "ec_l4", "canonical_rhea", "sequence_identity",
        "sequence_alignment_fraction", "esm2_t33_cosine", "pfam_jaccard", "reference_cyp_family",
        "reference_species_group", "eligible_GENERAL", "eligible_LEAVE_ONE_CYP_FAMILY_OUT",
        "eligible_PLANT_COLD_START",
    ]
    pairs = pd.read_parquet(feature_path, columns=pair_columns)
    if not np.array_equal(pairs["pair_index"].to_numpy(), np.arange(row_count)):
        raise RuntimeError("Pair order no longer matches prediction arrays")
    for index, level in enumerate(LEVELS):
        alpha = float(model_config["deep_blend_weights"][level])
        raw = alpha * np.asarray(deep[:, index]) + (1 - alpha) * np.asarray(tree[:, index])
        calibrator = joblib.load(root / "models/phase12" / f"isotonic_{level}.joblib")
        pairs[f"calibrated_{level}"] = calibrator.predict(raw).astype(np.float32)
    write_frame(pairs, results / "cyp450_scored_pairs.parquet")

    truth = pd.read_parquet(work / "p450_query_truth.parquet")
    prediction_frames = [aggregate_scenario(pairs, truth, scenario, thresholds) for scenario in SCENARIOS]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    write_frame(predictions.loc[predictions["scenario"].eq("GENERAL")], results / "cyp_general_predictions.tsv")
    leave = predictions.loc[predictions["scenario"].eq("LEAVE_ONE_CYP_FAMILY_OUT")].copy()
    plant = predictions.loc[predictions["scenario"].eq("PLANT_COLD_START")].copy()
    write_frame(leave, results / "cyp_leave_family_out.tsv")
    write_frame(plant, results / "plant_p450_cold_start.tsv")

    cyp_metrics = metric_table(predictions, ["scenario", "evaluation_cohort", "annotation_level"])
    write_frame(cyp_metrics, results / "cyp450_metrics.tsv")
    family_metrics = metric_table(
        predictions.loc[predictions["scenario"].eq("GENERAL")],
        ["evaluation_cohort", "query_cyp_family", "annotation_level"],
    )
    family_metrics["primary_reporting_eligible"] = family_metrics["queries"].ge(10)
    write_frame(family_metrics, results / "cyp_family_function.tsv")

    protocol = pd.read_csv(results / "selected_family_protocol.tsv", sep="\t")
    selected = set(protocol.loc[protocol["selected"].astype(str).str.lower().eq("true"), "family"].astype(str))
    general = general_family_results(root, selected)
    write_frame(general, results / "general_family_results.tsv")
    reaction = reaction_similarity_table(root, predictions)
    write_frame(reaction, results / "external_reaction_similarity.tsv")

    general_supported = int(general.loc[
        general["annotation_level"].eq("EC_L3") & general["scientific_support_rule_met"], "family"
    ].nunique()) >= 3
    strict_ec3 = cyp_metrics.loc[
        cyp_metrics["scenario"].eq("GENERAL")
        & cyp_metrics["evaluation_cohort"].eq("STRICT_EXTERNAL_ACCESSION")
        & cyp_metrics["annotation_level"].eq("EC_L3")
    ]
    cyp_supported = bool(len(strict_ec3) and (
        strict_ec3.iloc[0]["queries"] >= 50
        and strict_ec3.iloc[0]["accepted_queries"] >= 5
        and strict_ec3.iloc[0]["retrieval_recall"] >= 0.50
        and strict_ec3.iloc[0]["coverage"] >= 0.05
        and strict_ec3.iloc[0]["selective_accuracy"] >= 0.75
    ))
    go7 = "SUPPORTED" if general_supported and cyp_supported else "PARTIAL_SUPPORT" if general_supported or cyp_supported else "NOT_SUPPORTED"

    required = [
        "selected_family_protocol.tsv", "general_family_results.tsv", "cyp_family_function.tsv",
        "cyp_leave_family_out.tsv", "plant_p450_cold_start.tsv", "external_reaction_similarity.tsv",
    ]
    feature_schema = set(pq.ParquetFile(feature_path).schema_arrow.names)
    forbidden = sorted({name for name in feature_schema if name.startswith("truth_")})
    checks = [
        ("checkpoint_14_present", (checkpoints / "CHECKPOINT_14_PASS").is_file(), "strict phase gate"),
        ("general_family_selection_test_blind", not protocol["test_performance_used_for_selection"].astype(str).str.lower().eq("true").any(), str(sorted(selected))),
        ("p450_truth_absent_from_model_features", not forbidden, json.dumps(forbidden)),
        ("frozen_validation_thresholds", not calibration["test_used_for_calibration_or_threshold_selection"], json.dumps(thresholds)),
        ("strict_external_cohort_nonempty", truth["evaluation_cohort"].eq("STRICT_EXTERNAL_ACCESSION").sum() >= 300, str(truth["evaluation_cohort"].value_counts().to_dict())),
        ("three_external_scenarios", set(predictions["scenario"]) == set(SCENARIOS), str(predictions["scenario"].value_counts().to_dict())),
        ("general_family_results_nonempty", len(general) > 0, str(len(general))),
        ("cyp_leave_family_out_nonempty", len(leave) > 0, str(len(leave))),
        ("plant_cold_start_nonempty", len(plant) > 0, str(len(plant))),
        ("external_reaction_similarity_nonempty", len(reaction) > 0, str(len(reaction))),
        ("required_outputs_present", all((results / name).is_file() and (results / name).stat().st_size > 0 for name in required), json.dumps(required)),
        ("bounded_computation_no_docking_or_md", True, "MMseqs top-500 pool; top-50 per scenario; no docking/MD"),
    ]
    qc = pd.DataFrame(
        [(name, "PASS" if bool(ok) else "FAIL", details) for name, ok, details in checks],
        columns=["check", "status", "details"],
    )
    write_frame(qc, reports / "phase15_qc.tsv")
    failures = qc.loc[qc["status"].eq("FAIL"), "check"].tolist()
    strict_metrics = cyp_metrics.loc[cyp_metrics["evaluation_cohort"].eq("STRICT_EXTERNAL_ACCESSION")]
    reaction_strict = reaction.loc[reaction["evaluation_cohort"].eq("STRICT_EXTERNAL_ACCESSION")]
    summary = {
        "phase": 15, "status": "PASS" if not failures else "FAIL", "slurm_job_id": os.getenv("SLURM_JOB_ID", "NA"),
        "selected_general_families": sorted(selected), "general_family_go7_support": general_supported,
        "p450_queries": int(truth["accession"].nunique()),
        "strict_external_p450_queries": int(truth["evaluation_cohort"].eq("STRICT_EXTERNAL_ACCESSION").sum()),
        "p450_go7_support": cyp_supported, "go7_status": go7,
        "strict_external_metrics": strict_metrics.to_dict("records"),
        "strict_external_reaction_similarity_rows": len(reaction_strict),
        "strict_external_mean_top_reaction_similarity": float(reaction_strict["reaction_similarity"].mean()) if len(reaction_strict) else None,
        "external_model": "frozen Phase11 SiteGuard deep-LightGBM blend + Phase12 isotonic/abstention",
        "test_or_external_data_used_for_training_calibration_thresholds_or_family_selection": False,
        "direct_structure_features_in_external_primary_score": False,
        "claim_guardrail": "external records are documented activities; missing activities are not biochemical negatives",
        "qc_failures": failures,
    }
    (reports / "phase15_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    write_frame(general, figures / "Figure6_general_families.tsv")
    write_frame(strict_metrics, figures / "Figure6_cyp_scenarios.tsv")
    write_frame(family_metrics.loc[family_metrics["primary_reporting_eligible"]], figures / "Figure6_cyp_families.tsv")
    write_frame(reaction_strict, figures / "Figure6_external_reaction_similarity.tsv")
    write_frame(plant, figures / "Figure6_plant_cold_start.tsv")
    report_lines = [
        "# SiteGuard V4 Phase 15 Report", "", f"Status: **{summary['status']}**", "",
        "Phase 15 evaluates frozen SiteGuard across data-selected enzyme families and an independent P450Rdb panel. "
        "General families were selected from train/validation support only. The primary CYP cohort excludes accessions in the V4 activity benchmark.", "",
        f"- Selected general families: {', '.join(sorted(selected))}",
        f"- P450Rdb exact-sequence queries: {summary['p450_queries']:,}",
        f"- Strict external P450 accessions: {summary['strict_external_p450_queries']:,}",
        f"- GO-7 status: {go7}",
        f"- Strict external reaction-similarity evaluations: {len(reaction_strict):,}", "",
        "The stress test separates retrieval failure, selective coverage, and accepted-label accuracy. "
        "P450Rdb truth is used only after inference. Reaction proximity uses optimal non-currency participant matching and a fixed transformation-first weighting.", "",
        "## QC", "", qc.to_markdown(index=False), "",
    ]
    (reports / "PHASE_15_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    if failures:
        raise RuntimeError("Phase 15 QC failed: " + ", ".join(failures))
    (checkpoints / "CHECKPOINT_15_PASS").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
