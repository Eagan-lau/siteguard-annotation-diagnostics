#!/usr/bin/env python3
"""Create Figure 10 and the canonical Phase 24 EvidenceJudge technical report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEVEL_LABEL = {"EC_L3": "EC level 3", "EC_L4": "EC level 4", "EXACT_RHEA": "Exact Rhea"}
LEVEL_COLORS = {"EC_L3": "#1B6CA8", "EC_L4": "#D99A2B", "EXACT_RHEA": "#D66B4D"}
SYSTEM_LABEL = {"EVIDENCEJUDGE": "EvidenceJudge", "BEST_SINGLE_TOOL": "Best single tool"}
MODEL_LABEL = {
    "LOGISTIC_STACK": "Logistic",
    "LIGHTGBM_ROUTER": "LightGBM",
    "DEEPSETS_ROUTER": "DeepSets",
}


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, float_format="%.10g")


def source(source_id: str, label: str, path: str, filters: list[str]) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb-file",
            "sql": f"SELECT * FROM read_csv_auto('{path}', delim='\\t', header=true)",
            "description": f"Direct read of the frozen SiteGuard Phase 24 export: {path}",
            "executed_at": "2026-08-21T00:00:00Z",
            "tables_used": [path],
            "filters": filters,
        },
    }


def make_figure(root: Path, screen: pd.DataFrame, test: pd.DataFrame, comparison: pd.DataFrame, calibration: pd.DataFrame) -> dict[str, pd.DataFrame]:
    source_dir = root / "figures/source_data"
    output_dir = root / "figures/main"
    source_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_a = screen[["annotation_level", "model", "coverage", "documented_precision", "cluster_ci_low", "cluster_ci_high", "qualifies_95", "selected_for_level"]].copy()
    panel_a["annotation_label"] = panel_a["annotation_level"].map(LEVEL_LABEL)
    panel_a["model_label"] = panel_a["model"].map(MODEL_LABEL)
    panel_b = test[["system", "annotation_level", "model", "coverage", "documented_precision", "cluster_ci_low", "cluster_ci_high", "qualifies_95", "safe_coverage_at_95"]].copy()
    panel_b["annotation_label"] = panel_b["annotation_level"].map(LEVEL_LABEL)
    panel_b["system_label"] = panel_b["system"].map(SYSTEM_LABEL)
    panel_c = comparison.copy()
    panel_c["annotation_label"] = panel_c["annotation_level"].map(LEVEL_LABEL)
    panel_d = calibration.copy()
    panel_d["annotation_label"] = panel_d["annotation_level"].map(LEVEL_LABEL)
    panels = {"A": panel_a, "B": panel_b, "C": panel_c, "D": panel_d}
    for name, frame in panels.items():
        write_frame(frame, source_dir / f"Figure10{name}_EvidenceJudge.tsv")

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.5))
    levels = list(LEVEL_LABEL)
    models = list(MODEL_LABEL)
    positions = np.arange(len(levels))
    width = 0.24
    model_colors = {"LOGISTIC_STACK": "#1B6CA8", "LIGHTGBM_ROUTER": "#D99A2B", "DEEPSETS_ROUTER": "#D66B4D"}
    for index, model in enumerate(models):
        frame = panel_a.loc[panel_a["model"].eq(model)].set_index("annotation_level")
        values = [float(frame.loc[level, "coverage"]) if level in frame.index else 0.0 for level in levels]
        bars = axes[0, 0].bar(positions + (index - 1) * width, values, width, color=model_colors[model], label=MODEL_LABEL[model])
        for bar, level in zip(bars, levels):
            selected = bool(frame.loc[level, "selected_for_level"]) if level in frame.index else False
            if selected:
                bar.set_edgecolor("#202020"); bar.set_linewidth(1.8)
    axes[0, 0].set_xticks(positions, [LEVEL_LABEL[level] for level in levels])
    axes[0, 0].set_ylabel("SELECT coverage at 95% precision-CI")
    axes[0, 0].set_title("A  Validation-only model screening")
    axes[0, 0].legend(frameon=False, fontsize=8)

    marker = {"EVIDENCEJUDGE": "o", "BEST_SINGLE_TOOL": "s"}
    for _, row in panel_b.iterrows():
        if not np.isfinite(row["documented_precision"]):
            continue
        low = row["documented_precision"] - row["cluster_ci_low"] if np.isfinite(row["cluster_ci_low"]) else 0.0
        high = row["cluster_ci_high"] - row["documented_precision"] if np.isfinite(row["cluster_ci_high"]) else 0.0
        axes[0, 1].errorbar(
            row["coverage"], row["documented_precision"], yerr=[[low], [high]],
            fmt=marker[row["system"]], color=LEVEL_COLORS[row["annotation_level"]],
            markerfacecolor=LEVEL_COLORS[row["annotation_level"]] if row["system"] == "EVIDENCEJUDGE" else "white",
            markersize=7, capsize=3, linewidth=1.2,
        )
        axes[0, 1].annotate(f"{LEVEL_LABEL[row['annotation_level']]}\n{SYSTEM_LABEL[row['system']]}",
                            (row["coverage"], row["documented_precision"]), xytext=(5, 4), textcoords="offset points", fontsize=6.8)
    axes[0, 1].axhline(0.95, color="#555555", linestyle="--", linewidth=1)
    axes[0, 1].set_xlim(left=0); axes[0, 1].set_ylim(0.75, 1.01)
    axes[0, 1].set_xlabel("Locked test coverage"); axes[0, 1].set_ylabel("Documented precision (95% cluster CI)")
    axes[0, 1].set_title("B  Locked operating points on the exposed test set")

    bar_width = 0.36
    comparison_indexed = panel_c.set_index("annotation_level")
    for offset, field, label, color in [
        (-bar_width / 2, "locked_coverage_gain", "Locked coverage gain", "#B8BDC5"),
        (bar_width / 2, "incremental_safe_coverage", "Incremental safe coverage@95", "#1B6CA8"),
    ]:
        values = [float(comparison_indexed.loc[level, field]) for level in levels]
        bars = axes[1, 0].bar(
            positions + offset, values, bar_width,
            color=color, edgecolor="#333333", linewidth=0.6, label=label,
        )
        for bar, value in zip(bars, values):
            axes[1, 0].annotate(
                f"{value:+.3f}", (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4 if value >= 0 else -13), textcoords="offset points",
                ha="center", va="bottom" if value >= 0 else "top", fontsize=7,
            )
    axes[1, 0].set_xticks(positions, [LEVEL_LABEL[level] for level in levels])
    axes[1, 0].axhline(0, color="#555555", linewidth=1)
    axes[1, 0].set_ylim(-0.055, 0.02)
    axes[1, 0].set_ylabel("EvidenceJudge minus best single coverage")
    axes[1, 0].set_title("C  No positive incremental safe coverage at 95% precision-CI")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1, label="Ideal")
    for level in levels:
        frame = panel_d.loc[panel_d["annotation_level"].eq(level)].sort_values("mean_predicted_probability")
        axes[1, 1].plot(
            frame["mean_predicted_probability"], frame["observed_documented_precision"],
            marker="o", linewidth=1.6, color=LEVEL_COLORS[level], label=LEVEL_LABEL[level],
        )
    axes[1, 1].set_xlim(0, 1); axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel("Mean calibrated probability"); axes[1, 1].set_ylabel("Observed documented precision")
    axes[1, 1].set_title("D  Test calibration is descriptive, not confirmatory")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.grid(axis="y", color="#E3E6EA", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "EvidenceJudge: validation-locked selective annotation across eight evidence channels\n"
        "FIT/CAL/SELECT split by 30% sequence cluster; population test is post-hoc feasibility only",
        fontsize=14.0, fontweight="bold",
    )
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            output_dir / f"Figure10_EvidenceJudge_precision_coverage.{suffix}",
            dpi=300 if suffix == "png" else None, bbox_inches="tight", pad_inches=0.15,
        )
    plt.close(fig)
    return panels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "checkpoints/CHECKPOINT_24D_PASS").is_file():
        raise RuntimeError("CHECKPOINT_24D_PASS is required")
    screen = pd.read_csv(root / "results/phase24/evidencejudge_model_screen.tsv", sep="\t")
    validation = pd.read_csv(root / "results/phase24/evidencejudge_validation_operating_points.tsv", sep="\t")
    test = pd.read_csv(root / "results/phase24/evidencejudge_test_operating_points.tsv", sep="\t")
    comparison = pd.read_csv(root / "results/phase24/evidencejudge_incremental_safe_coverage.tsv", sep="\t")
    calibration = pd.read_csv(root / "results/phase24/evidencejudge_test_calibration_bins.tsv", sep="\t")
    qc = pd.read_csv(root / "reports/phase24_evidencejudge_final_qc.tsv", sep="\t")
    summary_json = json.loads((root / "reports/phase24_evidencejudge_final_summary.json").read_text(encoding="utf-8"))
    panels = make_figure(root, screen, test, comparison, calibration)

    report_dir = root / "reports/phase24_evidencejudge"
    report_dir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    title = "SiteGuard V4 Phase 24: EvidenceJudge precision–coverage evaluation"
    endpoint_rows = []
    for level in LEVEL_LABEL:
        for system in ("EVIDENCEJUDGE", "BEST_SINGLE_TOOL"):
            row = test.loc[(test["annotation_level"] == level) & (test["system"] == system)].iloc[0]
            endpoint_rows.append({
                "annotation_level": LEVEL_LABEL[level],
                "system": SYSTEM_LABEL[system],
                "model": row["model"],
                "locked_coverage": float(row["coverage"]),
                "documented_precision": float(row["documented_precision"]) if pd.notna(row["documented_precision"]) else None,
                "cluster_ci_low": float(row["cluster_ci_low"]) if pd.notna(row["cluster_ci_low"]) else None,
                "cluster_ci_high": float(row["cluster_ci_high"]) if pd.notna(row["cluster_ci_high"]) else None,
                "safe_coverage_at_95": float(row["safe_coverage_at_95"]),
                "qualifies_95": bool(row["qualifies_95"]),
            })
    endpoint_table = pd.DataFrame(endpoint_rows)
    model_table = screen[["annotation_level", "model", "coverage", "documented_precision", "cluster_ci_low", "cluster_ci_high", "qualifies_95", "selected_for_level"]].copy()
    model_table["annotation_level"] = model_table["annotation_level"].map(LEVEL_LABEL)
    model_table["model"] = model_table["model"].map(MODEL_LABEL)

    comparison_by_level = comparison.set_index("annotation_level")
    summary_record: dict[str, Any] = {"qc_passed": int(qc["passed"].astype(str).str.lower().eq("true").sum())}
    for level, short in [("EC_L3", "ec3"), ("EC_L4", "ec4"), ("EXACT_RHEA", "rhea")]:
        row = comparison_by_level.loc[level]
        summary_record[f"safe_{short}"] = float(row["evidencejudge_safe_coverage_at_95"])
        summary_record[f"gain_{short}"] = float(row["incremental_safe_coverage"])
    any_gain = bool((comparison["incremental_safe_coverage"] > 0).any())
    deep_promoted = bool(summary_json["deep_model_promoted"])
    selected_text = ", ".join(f"{LEVEL_LABEL[level]}={MODEL_LABEL.get(model, model)}" for level, model in summary_json["selected_models"].items())
    if any_gain:
        headline = "EvidenceJudge achieved non-zero incremental safe coverage at the frozen 95% precision-CI endpoint on at least one annotation level."
    else:
        headline = "EvidenceJudge did not establish positive incremental safe coverage at the frozen 95% precision-CI endpoint on the exposed test set."
    deep_text = "The DeepSets router was promoted on validation." if deep_promoted else "The DeepSets router was not promoted because a simpler calibrated model matched or exceeded its validation safe coverage."

    sources = [
        source("model_screen", "Validation-only EvidenceJudge model screen", "results/phase24/evidencejudge_model_screen.tsv", ["SELECT clusters only", "95% cluster-bootstrap CI lower bound >= 0.95"]),
        source("validation_lock", "Validation operating-point lock", "results/phase24/evidencejudge_validation_operating_points.tsv", ["Hierarchical probabilities", "thresholds locked before test evaluation"]),
        source("test_endpoints", "Locked population-test operating points", "results/phase24/evidencejudge_test_operating_points.tsv", ["POST_HOC_FEASIBILITY", "27,639 query proteins"]),
        source("incremental", "Incremental safe-coverage comparison", "results/phase24/evidencejudge_incremental_safe_coverage.tsv", ["Same 95% precision-CI requirement for both systems"]),
        source("calibration", "Test calibration bins", "results/phase24/evidencejudge_test_calibration_bins.tsv", ["Equal-mass deciles", "descriptive test analysis"]),
        source("final_qc", "EvidenceJudge final quality controls", "reports/phase24_evidencejudge_final_qc.tsv", ["Phase 24D final audit"]),
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Technical report for the validation-only EvidenceJudge model ladder and the post-hoc population-test precision–coverage endpoints.",
            "generatedAt": generated,
            "cards": [
                {"id": "safe_ec3", "description": "Realized test coverage only when both point precision and the 95% sequence-cluster bootstrap lower bound reach 0.95.", "dataset": "summary", "sourceId": "incremental", "metrics": [{"label": "EvidenceJudge EC-L3 safe coverage", "field": "safe_ec3", "format": "percent"}, {"label": "Incremental vs best single", "field": "gain_ec3", "format": "percent", "signed": True}]},
                {"id": "safe_ec4", "description": "Conservative EC-L4 endpoint under the frozen 95% precision-CI criterion.", "dataset": "summary", "sourceId": "incremental", "metrics": [{"label": "EvidenceJudge EC-L4 safe coverage", "field": "safe_ec4", "format": "percent"}, {"label": "Incremental vs best single", "field": "gain_ec4", "format": "percent", "signed": True}]},
                {"id": "safe_rhea", "description": "Conservative exact-Rhea endpoint; broad EC agreement is not counted as exact reaction support.", "dataset": "summary", "sourceId": "incremental", "metrics": [{"label": "EvidenceJudge exact-Rhea safe coverage", "field": "safe_rhea", "format": "percent"}, {"label": "Incremental vs best single", "field": "gain_rhea", "format": "percent", "signed": True}]},
                {"id": "qc", "description": "Final checks for lock order, model inventory, hierarchy, output grain and finite probabilities.", "dataset": "summary", "sourceId": "final_qc", "metrics": [{"label": "Final EvidenceJudge checks passed", "field": "qc_passed", "format": "number"}]},
            ],
            "charts": [
                {"id": "model_screen_chart", "title": "Validation-only safe coverage by model family", "subtitle": "FIT/CAL/SELECT are disjoint 30%-sequence-cluster partitions; outlined selections are locked per annotation level.", "type": "bar", "dataset": "model_screen", "sourceId": "model_screen", "encodings": {"x": {"field": "annotation_level", "type": "nominal", "label": "Annotation level"}, "y": {"field": "coverage", "type": "quantitative", "label": "Coverage at 95% precision-CI", "format": "percent"}, "color": {"field": "model", "type": "nominal", "label": "Model"}}, "yAxisTitle": "Coverage at 95% precision-CI", "valueFormat": "percent", "layout": "full"},
                {"id": "safe_coverage_chart", "title": "Realized safe coverage at the frozen 95% precision-CI endpoint", "subtitle": "A zero value means the locked test operating point failed the point-precision or sequence-cluster CI requirement.", "type": "bar", "dataset": "endpoints", "sourceId": "test_endpoints", "encodings": {"x": {"field": "annotation_level", "type": "nominal", "label": "Annotation level"}, "y": {"field": "safe_coverage_at_95", "type": "quantitative", "label": "Safe coverage at 95%", "format": "percent"}, "color": {"field": "system", "type": "nominal", "label": "System"}}, "yAxisTitle": "Safe coverage at 95% precision-CI", "valueFormat": "percent", "layout": "full"},
                {"id": "calibration_chart", "title": "EvidenceJudge calibration on the exposed population test", "subtitle": "Equal-mass deciles; this is a descriptive robustness view rather than a new confirmatory evaluation.", "type": "line", "dataset": "calibration", "sourceId": "calibration", "encodings": {"x": {"field": "mean_predicted_probability", "type": "quantitative", "label": "Mean calibrated probability"}, "y": {"field": "observed_documented_precision", "type": "quantitative", "label": "Observed documented precision", "format": "percent"}, "color": {"field": "annotation_level", "type": "nominal", "label": "Annotation level"}}, "yAxisTitle": "Observed documented precision", "valueFormat": "percent", "layout": "full"},
            ],
            "tables": [
                {"id": "endpoint_table", "title": "Locked test operating points", "subtitle": "Coverage, documented precision and 95% sequence-cluster bootstrap intervals for the selected router and best single-tool comparator.", "dataset": "endpoints", "sourceId": "test_endpoints", "defaultSort": {"field": "annotation_level", "direction": "asc"}, "density": "dense", "layout": "full", "columns": [
                    {"field": "annotation_level", "label": "Annotation level", "type": "text"}, {"field": "system", "label": "System", "type": "text"}, {"field": "model", "label": "Model/tool", "type": "text"}, {"field": "locked_coverage", "label": "Locked coverage", "format": "percent"}, {"field": "documented_precision", "label": "Precision", "format": "percent"}, {"field": "cluster_ci_low", "label": "Cluster CI low", "format": "percent"}, {"field": "cluster_ci_high", "label": "Cluster CI high", "format": "percent"}, {"field": "safe_coverage_at_95", "label": "Safe coverage@95", "format": "percent"}, {"field": "qualifies_95", "label": "95% endpoint met", "type": "text"},
                ]},
                {"id": "model_table", "title": "Validation-only model ladder", "subtitle": "DeepSets is retained only if it produces greater conservative SELECT coverage than Logistic/LightGBM.", "dataset": "model_screen", "sourceId": "model_screen", "defaultSort": {"field": "coverage", "direction": "desc"}, "density": "dense", "layout": "full", "columns": [
                    {"field": "annotation_level", "label": "Annotation level", "type": "text"}, {"field": "model", "label": "Model", "type": "text"}, {"field": "coverage", "label": "SELECT coverage@95", "format": "percent"}, {"field": "documented_precision", "label": "Precision", "format": "percent"}, {"field": "cluster_ci_low", "label": "CI low", "format": "percent"}, {"field": "cluster_ci_high", "label": "CI high", "format": "percent"}, {"field": "qualifies_95", "label": "Qualifies", "type": "text"}, {"field": "selected_for_level", "label": "Selected", "type": "text"},
                ]},
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {"id": "summary", "type": "markdown", "sourceId": "incremental", "body": f"## Technical summary\n\n**{headline}** The selected model family was {selected_text}. {deep_text} All population-test numbers in this report are explicitly **POST_HOC FEASIBILITY** because the component predictions had been viewed before the Phase 24 fusion protocol was frozen."},
                {"id": "metrics", "type": "metric-strip", "cardIds": ["safe_ec3", "safe_ec4", "safe_rhea", "qc"]},
                {"id": "scope", "type": "markdown", "body": "## Scope, data and frozen endpoint definitions\n\nEvidenceJudge operates only on reaction-resolved enzyme catalytic-function annotation. It receives top predictions and confidence/margin/availability/agreement features from eight frozen evidence channels; query EC, Rhea, substrates, products, activities, catalytic sites and truth-derived chemistry are prohibited inputs. Validation clusters were deterministically divided into FIT (model fitting), CAL (isotonic calibration) and SELECT (model/threshold selection). The primary endpoint is accepted-query coverage when documented precision is at least 95% and the 95% sequence-cluster bootstrap CI lower bound is also at least 95%, with at least 50 accepted queries from 20 clusters."},
                {"id": "screen_intro", "type": "markdown", "sourceId": "model_screen", "body": "## Model selection was driven by conservative validation coverage\n\nLogistic stacking, LightGBM and a three-seed masked DeepSets router were evaluated under the same fixed feature matrix and SELECT clusters. Model complexity was not rewarded by itself; the simplest model wins ties, and the deep model is promoted only for higher conservative coverage."},
                {"id": "screen_chart", "type": "chart", "chartId": "model_screen_chart", "layout": "full"},
                {"id": "model_table_block", "type": "table", "tableId": "model_table", "layout": "full"},
                {"id": "endpoint_intro", "type": "markdown", "sourceId": "test_endpoints", "body": "## The 95% precision-CI endpoint determines whether coverage is safe\n\nThe locked threshold is applied unchanged to the population test. Locked coverage and point precision are shown even when the conservative endpoint fails, but only `safe coverage@95` is used for the primary claim. This prevents a high point estimate with a wide cluster interval from being presented as reliable annotation coverage."},
                {"id": "endpoint_chart", "type": "chart", "chartId": "safe_coverage_chart", "layout": "full"},
                {"id": "endpoint_table_block", "type": "table", "tableId": "endpoint_table", "layout": "full"},
                {"id": "calibration_intro", "type": "markdown", "sourceId": "calibration", "body": "## Calibration remains distribution-specific\n\nThe decile plot tests whether predicted reliability tracks observed documented concordance across the exposed population test. Departures from the diagonal identify calibration error, but they do not establish biochemical absence for undocumented activities and cannot replace a new external or future temporal holdout."},
                {"id": "calibration_chart_block", "type": "chart", "chartId": "calibration_chart", "layout": "full"},
                {"id": "method", "type": "markdown", "sourceId": "validation_lock", "body": "## Model specification and lock order\n\nEach query contributes the union of the eight tools' top candidate labels. Candidate features encode per-tool support, raw score, margin, missingness, evidence-channel diversity, vote fraction and disagreement entropy; label identifiers are output keys, never model features. Models fit on FIT clusters, isotonic calibrators fit only on CAL, and SELECT fixes the model and threshold. Probabilities are clipped to satisfy P(Exact Rhea) ≤ P(EC-L4) ≤ P(EC-L3). The threshold lock is written before any TEST outcome calculation."},
                {"id": "limitations", "type": "markdown", "body": "## Limitations, uncertainty and robustness\n\n1. The population test is post-hoc feasibility evidence, not a confirmatory benchmark.\n2. Public external pretrained tools are not yet fused because training-set overlap with Swiss-Prot/Rhea must be audited or the tools retrained on frozen splits.\n3. Documented concordance is not biochemical absence; promiscuous or incompletely curated activities may be counted as unsupported.\n4. Exact-Rhea evidence is not inferred from broad EC-only tools.\n5. ChemBridge rescue is not available on this population query table and is therefore reported as unavailable rather than imputed.\n6. Very high precision with cluster-level uncertainty can produce zero safe coverage even when point precision exceeds 95%; that is an intended conservative outcome."},
                {"id": "next", "type": "markdown", "body": "## Recommended next steps\n\n1. Freeze a new temporal or external holdout after the current model/threshold lock and open it once.\n2. Add task-compatible CLEAN, eggNOG/InterPro and reaction-pair adapters only after license, catalog and training-overlap audits.\n3. Recompute the same two primary endpoints without changing thresholds: safe coverage@95 and incremental safe coverage versus the best single tool.\n4. Release `siteguard judge` only for annotation levels whose confirmatory CI lower bound meets the target; all other outputs remain REVIEW or ABSTAIN."},
                {"id": "questions", "type": "markdown", "body": "## Further questions\n\n- Which independent evidence combinations create the largest incremental safe coverage rather than merely correlated votes?\n- Does calibration transfer across enzyme families, taxa, structure availability and nearest-training identity?\n- Can reaction-rescue candidates add non-zero exact-Rhea safe coverage without increasing unsupported fine-grained annotation?\n- Does a new temporal holdout reproduce the validation-selected model hierarchy?"},
                {"id": "validation", "type": "markdown", "sourceId": "final_qc", "body": f"## Reproducibility and validation\n\nPhase 24D passed {int(qc['passed'].astype(str).str.lower().eq('true').sum())}/{len(qc)} final checks. The candidate matrix, development partitions, serialized models, isotonic calibrators, operating-point lock, query-level decisions, risk–coverage tables, calibration bins and Figure 10 source tables are retained. Figure 10 is exported as PDF, SVG and 300-dpi PNG."},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "summary": [summary_record],
                "model_screen": records(model_table),
                "endpoints": records(endpoint_table),
                "calibration": records(panels["D"][["annotation_label", "bin", "queries", "mean_predicted_probability", "observed_documented_precision"]].rename(columns={"annotation_label": "annotation_level"})),
            },
        },
        "sources": sources,
        "package_info": {"root": "reports/phase24_evidencejudge", "manifestPath": "artifact.json", "snapshotPath": "artifact.json"},
    }
    (report_dir / "artifact.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    chart_map = pd.DataFrame([
        ("Figure10A", "Which model family maximizes conservative SELECT coverage?", "grouped bar", "annotation_level,model,coverage", "Model selection without rewarding complexity"),
        ("Figure10B", "Where do locked test operating points lie in precision–coverage space?", "scatter with interval", "system,annotation_level,coverage,precision,cluster_ci", "Point precision and cluster uncertainty shown together"),
        ("Figure10C", "Does EvidenceJudge add safe coverage at the 95% endpoint?", "grouped bar", "system,annotation_level,safe_coverage", "Primary paper endpoint"),
        ("Figure10D", "Does calibrated probability track observed documented precision?", "multi-series line", "annotation_level,predicted,observed", "Distribution-specific calibration diagnostic"),
    ], columns=["visual", "analytical_question", "chart_type", "fields", "supported_claim"])
    write_frame(chart_map, report_dir / "chart_map.tsv")
    checks = [
        ("figure_formats", all((root / f"figures/main/Figure10_EvidenceJudge_precision_coverage.{suffix}").is_file() for suffix in ("pdf", "svg", "png")), "PDF/SVG/PNG"),
        ("figure_source_tables", all((root / f"figures/source_data/Figure10{panel}_EvidenceJudge.tsv").is_file() for panel in "ABCD"), "A/B/C/D"),
        ("artifact_json", (report_dir / "artifact.json").is_file(), str(report_dir / "artifact.json")),
        ("chart_map", (report_dir / "chart_map.tsv").is_file(), str(report_dir / "chart_map.tsv")),
    ]
    qc_artifact = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    qc_artifact.to_csv(root / "reports/phase24_evidencejudge_artifact_qc.tsv", sep="\t", index=False)
    failures = qc_artifact.loc[~qc_artifact["passed"].astype(bool), "check"].tolist()
    if failures:
        raise RuntimeError(f"Phase 24 artifact generation failed: {failures}")
    print(json.dumps({"status": "PASS", "figure": "Figure10", "artifact": str(report_dir / "artifact.json")}, indent=2))


if __name__ == "__main__":
    main()
