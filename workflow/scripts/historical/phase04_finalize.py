#!/usr/bin/env python3
"""Compute retrieval ceilings and finalize leakage-safe SiteGuard V4 Phase 4."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


KS = (10, 20, 50, 100)
LEVELS = ("EC_L3", "EC_L4", "EXACT_RHEA")
FORBIDDEN_CANDIDATE_COLUMNS = {
    "canonical_ec", "canonical_rhea", "ec_l3_json", "ec_l4_json", "canonical_rhea_json",
    "query_ec", "query_rhea", "query_activity_id",
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_json_set(value: Any) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    try:
        return {str(item) for item in json.loads(value) if item is not None and str(item)}
    except Exception:
        return set()


def value_set(values: pd.Series) -> set[str]:
    return {str(value) for value in values.dropna() if str(value)}


def candidate_lists(path: Path) -> tuple[dict[str, list[str]], pd.DataFrame, set[str]]:
    schema_names = set(pq.read_schema(path).names)
    frame = pd.read_parquet(
        path,
        columns=["query_protein_id", "reference_protein_id", "modality_rank", "query_partition", "reference_partition"],
    )
    frame = frame.sort_values(["query_protein_id", "modality_rank"], kind="stable")
    lists = frame.groupby("query_protein_id", sort=False)["reference_protein_id"].agg(list).to_dict()
    return lists, frame, schema_names


def cumulative_label_sets(refs: list[str], reference_labels: dict[str, dict[str, set[str]]]) -> dict[int, dict[str, set[str]]]:
    result: dict[int, dict[str, set[str]]] = {}
    accumulated = {level: set() for level in LEVELS}
    cursor = 0
    for k in KS:
        while cursor < min(k, len(refs)):
            labels = reference_labels.get(refs[cursor], {})
            for level in LEVELS:
                accumulated[level].update(labels.get(level, set()))
            cursor += 1
        result[k] = {level: set(values) for level, values in accumulated.items()}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    started = now()
    root = args.project_root.resolve()
    source = args.source_root.resolve()
    processed = root / "data" / "processed"
    work = root / "data" / "interim" / "phase04"
    reports = root / "reports"
    checkpoints = root / "checkpoints"
    reports.mkdir(parents=True, exist_ok=True)

    mm_path = processed / "mmseqs_candidates.parquet"
    fs_path = processed / "foldseek_candidates.parquet"
    mm_lists, mm_frame, mm_schema = candidate_lists(mm_path)
    fs_lists, fs_frame, fs_schema = candidate_lists(fs_path)
    query_truth = pd.read_parquet(work / "retrieval_query_truth.parquet")
    membership = pd.read_parquet(work / "afdb_membership.parquet", columns=["protein_id", "has_structure"])
    has_structure = membership.set_index("protein_id")["has_structure"].to_dict()
    library = pd.read_parquet(
        root / "data" / "reference" / "activity_reference_library.parquet",
        columns=["reference_protein_id", "ec_l3", "canonical_ec", "canonical_rhea", "reference_partition"],
    )

    reference_labels: dict[str, dict[str, set[str]]] = {}
    for protein_id, group in library.groupby("reference_protein_id", sort=False):
        reference_labels[str(protein_id)] = {
            "EC_L3": value_set(group["ec_l3"]),
            "EC_L4": value_set(group["canonical_ec"]),
            "EXACT_RHEA": value_set(group["canonical_rhea"]),
        }

    aggregates: dict[tuple[str, str, str, int, str], dict[str, float]] = defaultdict(
        lambda: {"total_queries": 0, "truth_queries": 0, "queries_with_candidate": 0, "correct_queries": 0, "label_recall_sum": 0.0}
    )
    truth_columns = {"EC_L3": "ec_l3_json", "EC_L4": "ec_l4_json", "EXACT_RHEA": "canonical_rhea_json"}
    for row in query_truth.itertuples(index=False):
        query_id = str(row.protein_id)
        partition = str(row.split)
        truths = {level: parse_json_set(getattr(row, column)) for level, column in truth_columns.items()}
        modality_refs = {
            "mmseqs": mm_lists.get(query_id, []),
            "foldseek": fs_lists.get(query_id, []),
        }
        modality_sets = {
            modality: cumulative_label_sets(refs, reference_labels)
            for modality, refs in modality_refs.items()
        }
        cohorts = ["all_queries"]
        if bool(has_structure.get(query_id, False)):
            cohorts.append("structure_available")
        for k in KS:
            retrieved_by_modality = {
                "mmseqs": modality_sets["mmseqs"][k],
                "foldseek": modality_sets["foldseek"][k],
                "union": {
                    level: modality_sets["mmseqs"][k][level] | modality_sets["foldseek"][k][level]
                    for level in LEVELS
                },
            }
            candidate_available = {
                "mmseqs": bool(modality_refs["mmseqs"][:k]),
                "foldseek": bool(modality_refs["foldseek"][:k]),
                "union": bool(modality_refs["mmseqs"][:k] or modality_refs["foldseek"][:k]),
            }
            for split_name in (partition, "all"):
                for cohort in cohorts:
                    for modality in ("mmseqs", "foldseek", "union"):
                        for level in LEVELS:
                            values = aggregates[(split_name, cohort, modality, k, level)]
                            values["total_queries"] += 1
                            values["queries_with_candidate"] += int(candidate_available[modality])
                            truth = truths[level]
                            if truth:
                                values["truth_queries"] += 1
                                intersection = truth & retrieved_by_modality[modality][level]
                                values["correct_queries"] += int(bool(intersection))
                                values["label_recall_sum"] += len(intersection) / len(truth)

    metric_rows: list[dict[str, Any]] = []
    for key, values in sorted(aggregates.items()):
        split_name, cohort, modality, k, level = key
        total = int(values["total_queries"])
        truth_n = int(values["truth_queries"])
        covered = int(values["queries_with_candidate"])
        correct = int(values["correct_queries"])
        metric_rows.append({
            "split": split_name,
            "cohort": cohort,
            "modality": modality,
            "top_k": k,
            "annotation_level": level,
            "total_queries": total,
            "truth_queries": truth_n,
            "queries_with_candidate": covered,
            "query_candidate_coverage": covered / total if total else None,
            "correct_queries": correct,
            "query_any_correct_recall": correct / truth_n if truth_n else None,
            "macro_label_set_recall": values["label_recall_sum"] / truth_n if truth_n else None,
            "candidate_definition": "top_k_cluster_deduplicated" if modality != "union" else "union_of_top_k_per_modality",
        })
    metrics = pd.DataFrame(metric_rows)
    metrics_path = processed / "retrieval_ceiling_metrics.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)

    structure_summary = json.loads((reports / "phase04_structure_split_summary.json").read_text(encoding="utf-8"))
    raw_inventory = source / "data" / "manifests" / "resource_inventory.tsv"
    candidate_checks = []
    for modality, frame, candidate_schema in (("mmseqs", mm_frame, mm_schema), ("foldseek", fs_frame, fs_schema)):
        candidate_checks.extend([
            (f"{modality}_candidate_rows_nonempty", len(frame) > 0, f"rows={len(frame)}"),
            (f"{modality}_references_train_only", set(frame["reference_partition"]) == {"train"}, str(sorted(frame["reference_partition"].unique()))),
            (f"{modality}_queries_nontrain_only", set(frame["query_partition"]).issubset({"validation", "test"}), str(sorted(frame["query_partition"].unique()))),
            (f"{modality}_rank_bounded", int(frame["modality_rank"].max()) <= 100, f"max={frame['modality_rank'].max()}"),
            (f"{modality}_candidate_schema_no_query_truth", not (FORBIDDEN_CANDIDATE_COLUMNS & candidate_schema), str(sorted(FORBIDDEN_CANDIDATE_COLUMNS & candidate_schema))),
        ])
    checks = [
        ("checkpoint_03_present", (checkpoints / "CHECKPOINT_03_PASS").is_file(), "Phase 3 prerequisite"),
        ("reference_library_train_only", set(library["reference_partition"]) == {"train"}, str(sorted(library["reference_partition"].unique()))),
        ("query_truth_isolated", set(query_truth["provenance"]) == {"GROUND_TRUTH_ONLY_EVALUATION_DO_NOT_JOIN_TO_MODEL_FEATURES"}, "evaluation-only provenance"),
        ("structure_split_pass", structure_summary.get("status") == "PASS" and structure_summary.get("cluster_overlap") == 0, json.dumps(structure_summary)),
        ("retrieval_metrics_complete", len(metrics) >= 200 and set(metrics["annotation_level"]) == set(LEVELS), f"rows={len(metrics)}"),
        ("raw_inventory_unchanged", sha256_file(raw_inventory) == "3cf819b53ecfffed0113a89116496f6561b3771f29d97960111bc397eec67fc2", sha256_file(raw_inventory)),
    ] + candidate_checks
    qc = pd.DataFrame([(name, "PASS" if passed else "FAIL", detail) for name, passed, detail in checks], columns=["check", "status", "details"])
    qc.to_csv(reports / "phase04_qc.tsv", sep="\t", index=False)
    failures = qc[qc["status"] == "FAIL"]

    focus = metrics[
        (metrics["split"] == "test") & (metrics["cohort"] == "all_queries")
        & (metrics["modality"] == "union") & metrics["top_k"].isin(KS)
    ].sort_values(["annotation_level", "top_k"])
    report = [
        "# SiteGuard V4 Phase 4 Report", "",
        f"- Started: `{started}`", f"- Completed: `{now()}`",
        f"- Decision: `{'PASS' if failures.empty else 'FAIL'}`",
        f"- MMseqs candidate rows: **{len(mm_frame):,}**",
        f"- Foldseek candidate rows: **{len(fs_frame):,}**",
        f"- Structure clusters: **{structure_summary['structure_clusters']:,}**",
        "", "## Strict test retrieval ceiling (union)", "",
        "| Level | K | Candidate coverage | Any-correct recall | Macro label-set recall | Truth queries |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in focus.itertuples(index=False):
        report.append(
            f"| {row.annotation_level} | {row.top_k} | {row.query_candidate_coverage:.4f} | "
            f"{row.query_any_correct_recall:.4f} | {row.macro_label_set_recall:.4f} | {row.truth_queries:,} |"
        )
    exact50 = focus[(focus["annotation_level"] == "EXACT_RHEA") & (focus["top_k"] == 50)]
    exact50_value = float(exact50.iloc[0]["query_any_correct_recall"]) if len(exact50) else None
    report += [
        "", "## Interpretation guard", "",
        "Checkpoint PASS certifies database construction, split integrity, provenance separation, and metric completeness; it does not automatically certify that Exact Rhea is a strong end-to-end claim.",
        f"Observed strict-test union Exact Rhea any-correct Recall@50: **{exact50_value:.4f}**." if exact50_value is not None else "Exact Rhea Recall@50 unavailable.",
        "The manuscript claim will follow the pre-registered GO-3 rule: weak Exact Rhea retrieval shifts the end-to-end emphasis to EC-L4 and reaction proximity while retaining oracle/pairwise Rhea analyses.",
        "", "Candidate files contain no query EC/Rhea truth fields; those remain isolated for evaluation only.", "",
    ]
    (reports / "PHASE_04_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    summary = {
        "phase": 4,
        "project_version": "V4",
        "status": "PASS" if failures.empty else "FAIL",
        "started_at": started,
        "completed_at": now(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "mmseqs_candidate_rows": len(mm_frame),
        "foldseek_candidate_rows": len(fs_frame),
        "structure_clusters": structure_summary["structure_clusters"],
        "exact_rhea_union_any_correct_recall_at_50_strict_test": exact50_value,
        "qc_failures": failures.to_dict("records"),
        "outputs": {
            "mmseqs_candidates": str(mm_path),
            "foldseek_candidates": str(fs_path),
            "retrieval_ceiling_metrics": str(metrics_path),
            "activity_reference_library": str(root / "data" / "reference" / "activity_reference_library.parquet"),
        },
    }
    (reports / "phase04_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    checkpoint = checkpoints / "CHECKPOINT_04_PASS"
    if not failures.empty:
        checkpoint.unlink(missing_ok=True)
        print(json.dumps(summary, indent=2), flush=True)
        return 2
    checkpoint.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
