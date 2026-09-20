#!/usr/bin/env python3
"""Independent integrity and endpoint audit for augmented EvidenceJudge development results."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
OUT = ROOT / "results/phase29_augmented"
REPORTS = ROOT / "reports/phase29_augmented"
SEED = 20260829


def cluster_bootstrap(frame: pd.DataFrame, context: str, draws: int = 10000) -> tuple[float, float]:
    groups = frame.groupby("query_cluster_id_30", observed=True)["correct"].agg(["sum", "count"])
    values = groups[["sum", "count"]].to_numpy(float)
    if not len(values):
        return float("nan"), float("nan")
    if np.all(values[:, 0] == values[:, 1]):
        return 1.0, 1.0
    seed = int.from_bytes(hashlib.sha256(f"AUDIT|{SEED}|{context}".encode()).digest()[:8], "big") % (2**32 - 1)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for start in range(0, draws, 250):
        n = min(250, draws - start)
        sampled = values[rng.integers(0, len(values), size=(n, len(values)))]
        estimates[start : start + n] = sampled[:, :, 0].sum(1) / sampled[:, :, 1].sum(1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def metrics(frame: pd.DataFrame, total: int, context: str) -> dict[str, object]:
    n = len(frame)
    successes = int(frame["correct"].sum()) if n else 0
    clusters = int(frame["query_cluster_id_30"].nunique()) if n else 0
    low, high = cluster_bootstrap(frame, context) if n else (float("nan"), float("nan"))
    precision = successes / n if n else float("nan")
    cp_low = float(beta.ppf(0.025, successes, n - successes + 1)) if successes else 0.0
    safe = bool(n >= 50 and clusters >= 20 and precision >= 0.95 and low >= 0.95) if n else False
    return {
        "accepted": n, "clusters": clusters, "coverage": n / total,
        "precision": precision, "cluster_bootstrap_low": low,
        "cluster_bootstrap_high": high, "clopper_pearson_two_sided_low": cp_low,
        "primary_safe": safe,
    }


def truth_tables() -> dict[tuple[int, int], pd.DataFrame]:
    result = {}
    for phase, file_name, id_name in [
        (26, "sabio_strict_blind_cohort.parquet", "uniprot_accession"),
        (28, "rcsb_strict_blind_cohort.parquet", "query_id"),
    ]:
        truth = pd.read_parquet(ROOT / f"results/phase{phase}/{file_name}").rename(columns={id_name: "query_protein_id"})
        for level in (3, 4):
            result[(phase, level)] = truth.loc[truth[f"ec_l{level}_label_eligible"].astype(bool), [
                "query_protein_id", f"ec_l{level}", "external_cluster_id_30",
            ]].rename(columns={f"ec_l{level}": "truth_label", "external_cluster_id_30": "query_cluster_id_30"})
    return result


def single_tool_rows(truth: dict[tuple[int, int], pd.DataFrame]) -> pd.DataFrame:
    records = []
    for phase in (26, 28):
        old = pd.read_parquet(ROOT / f"results/phase{phase}/external_tool_top_predictions_blind.parquet")
        hit = pd.read_csv(ROOT / f"data/interim/phase29_hit_ec/phase{phase}_hit_ec_predictions.tsv", sep="\t").rename(
            columns={"query_id": "query_protein_id"}
        )
        clean = pd.read_csv(ROOT / f"data/interim/phase29_clean/phase{phase}_clean_predictions.tsv", sep="\t").rename(
            columns={"query_id": "query_protein_id"}
        )
        clean["clean_ec3"] = clean["clean_top1_ec4"].str.rsplit(".", n=1).str[0]
        for level in (3, 4):
            base = truth[(phase, level)]
            current = old.loc[old["annotation_level"].eq(f"EC_L{level}")].merge(
                base, on="query_protein_id", validate="many_to_one"
            )
            for score_column in ("raw_score", "top1_margin"):
                for row in current.itertuples(index=False):
                    score = getattr(row, score_column)
                    if pd.notna(score):
                        records.append({
                            "phase": phase, "annotation_level": f"EC_L{level}",
                            "query_protein_id": row.query_protein_id,
                            "query_cluster_id_30": row.query_cluster_id_30,
                            "method": row.method, "score_variant": score_column,
                            "direction": "HIGH", "score": float(score),
                            "candidate_label": str(row.candidate_label),
                            "truth_label": str(row.truth_label),
                            "correct": str(row.candidate_label) == str(row.truth_label),
                        })
            hit_current = base.merge(hit, on="query_protein_id", validate="one_to_one")
            hit_variants = [f"ec{level}_top1_softmax", f"ec{level}_top1_logit", f"ec{level}_softmax_margin12"]
            if level == 4:
                hit_variants += ["ec4_top1_sigmoid", "ec4_sigmoid_margin12"]
            for variant in hit_variants:
                for row in hit_current.itertuples(index=False):
                    candidate = str(getattr(row, f"ec{level}_top1"))
                    records.append({
                        "phase": phase, "annotation_level": f"EC_L{level}",
                        "query_protein_id": row.query_protein_id,
                        "query_cluster_id_30": row.query_cluster_id_30,
                        "method": "HIT_EC", "score_variant": variant,
                        "direction": "HIGH", "score": float(getattr(row, variant)),
                        "candidate_label": candidate, "truth_label": str(row.truth_label),
                        "correct": candidate == str(row.truth_label),
                    })
            clean_current = base.merge(clean, on="query_protein_id", validate="one_to_one")
            clean_pred = "clean_ec3" if level == 3 else "clean_top1_ec4"
            for variant, direction in [
                ("clean_top1_distance", "LOW"),
                ("clean_top1_gmm_confidence", "HIGH"),
                ("clean_distance_margin12", "HIGH"),
            ]:
                for row in clean_current.itertuples(index=False):
                    candidate = str(getattr(row, clean_pred))
                    records.append({
                        "phase": phase, "annotation_level": f"EC_L{level}",
                        "query_protein_id": row.query_protein_id,
                        "query_cluster_id_30": row.query_cluster_id_30,
                        "method": "CLEAN", "score_variant": variant,
                        "direction": direction, "score": float(getattr(row, variant)),
                        "candidate_label": candidate, "truth_label": str(row.truth_label),
                        "correct": candidate == str(row.truth_label),
                    })
    return pd.DataFrame(records)


def scan_single_tools(frame: pd.DataFrame, totals: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = []
    best = []
    keys = ["annotation_level", "method", "score_variant", "direction"]
    for key, group in frame.groupby(keys, observed=True):
        level, method, variant, direction = key
        ascending = direction == "LOW"
        ordered = group.sort_values("score", ascending=ascending, kind="mergesort")
        thresholds = ordered["score"].drop_duplicates().to_numpy()
        qualified = []
        point_only = []
        for threshold in thresholds:
            accepted = group.loc[group["score"].le(threshold) if ascending else group["score"].ge(threshold)]
            n = len(accepted)
            clusters = accepted["query_cluster_id_30"].nunique()
            precision = float(accepted["correct"].mean())
            base = {
                "annotation_level": level, "method": method, "score_variant": variant,
                "direction": direction, "threshold": float(threshold), "accepted": n,
                "clusters": int(clusters), "coverage": n / totals[level], "precision": precision,
            }
            if n >= 50 and clusters >= 20 and precision >= 0.95:
                point_only.append(base)
                low, high = cluster_bootstrap(accepted, f"SINGLE|{'|'.join(key)}|{threshold:.12g}", draws=4000)
                base = {**base, "cluster_bootstrap_low": low, "cluster_bootstrap_high": high, "primary_safe": low >= 0.95}
                curves.append(base)
                if low >= 0.95:
                    qualified.append(base)
        if qualified:
            choice = max(qualified, key=lambda row: (row["accepted"], row["precision"]))
            best.append({"qualified": True, **choice})
        elif point_only:
            choice = max(point_only, key=lambda row: (row["accepted"], row["precision"]))
            best.append({"qualified": False, **choice, "cluster_bootstrap_low": np.nan, "cluster_bootstrap_high": np.nan, "primary_safe": False})
        else:
            best.append({
                "qualified": False, "annotation_level": level, "method": method,
                "score_variant": variant, "direction": direction, "threshold": np.nan,
                "accepted": 0, "clusters": 0, "coverage": 0.0, "precision": np.nan,
                "cluster_bootstrap_low": np.nan, "cluster_bootstrap_high": np.nan, "primary_safe": False,
            })
    return pd.DataFrame(curves), pd.DataFrame(best)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    checks: dict[str, bool] = {}
    truth = truth_tables()
    expected = {(26, 3): 123, (26, 4): 118, (28, 3): 165, (28, 4): 162}
    checks["eligible_query_counts_match"] = all(len(truth[key]) == value for key, value in expected.items())

    candidates = pd.read_parquet(OUT / "augmented_development_candidates.parquet")
    protected = {"truth_label", "correct", "candidate_label", "query_protein_id", "query_cluster_id_30"}
    freeze = json.loads((OUT / "augmented_evidencejudge_freeze.json").read_text(encoding="utf-8"))
    checks["truth_not_in_feature_schema"] = protected.isdisjoint(freeze["feature_columns"])
    checks["candidate_labels_reconstruct_correct"] = bool(
        candidates["correct"].astype(bool).eq(candidates["candidate_label"].astype(str).eq(candidates["truth_label"].astype(str))).all()
    )
    checks["one_fold_per_cluster"] = bool(candidates.groupby("query_cluster_id_30")["cv_fold"].nunique().eq(1).all())
    checks["no_duplicate_candidates"] = not candidates.duplicated([
        "source_dataset", "annotation_level", "query_protein_id", "candidate_label"
    ]).any()
    checks["no_missing_hit_or_clean_support"] = bool(
        candidates.groupby(["source_dataset", "annotation_level", "query_protein_id"])[["support__HIT_EC", "support__CLEAN"]].max().eq(1).all().all()
    )
    manifests = list((ROOT / "data/interim/phase29_hit_ec").glob("*.manifest.json")) + list(
        (ROOT / "data/interim/phase29_clean").glob("*.manifest.json")
    )
    checks["all_tool_manifests_truth_free"] = len(manifests) == 4 and all(
        json.loads(path.read_text(encoding="utf-8"))["truth_inputs_used"] is False for path in manifests
    )

    oof = pd.read_parquet(OUT / "augmented_judge_oof_predictions.parquet")
    checks["one_oof_winner_per_model_query"] = not oof.duplicated([
        "model", "annotation_level", "source_dataset", "query_protein_id"
    ]).any()
    endpoint_rows = []
    for (level, model), group in oof.groupby(["annotation_level", "model"], observed=True):
        accepted = group.loc[group["accepted_at_locked_threshold"].astype(bool)]
        endpoint_rows.append({
            "annotation_level": level, "model": model,
            **metrics(accepted, group["query_protein_id"].nunique(), f"AUDIT|{level}|{model}"),
        })
    judge = pd.DataFrame(endpoint_rows)
    judge.to_csv(OUT / "audited_judge_operating_points.tsv", sep="\t", index=False)

    single = single_tool_rows(truth)
    totals = {"EC_L3": len(truth[(26, 3)]) + len(truth[(28, 3)]), "EC_L4": len(truth[(26, 4)]) + len(truth[(28, 4)])}
    curves, best_variants = scan_single_tools(single, totals)
    curves.to_csv(OUT / "single_tool_risk_coverage_development.tsv", sep="\t", index=False)
    best_variants.to_csv(OUT / "single_tool_best_operating_points.tsv", sep="\t", index=False)
    best_safe = best_variants.loc[best_variants["qualified"].astype(bool)].sort_values(
        ["annotation_level", "coverage", "precision"], ascending=[True, False, False], kind="mergesort"
    ).drop_duplicates("annotation_level")

    incremental = []
    for level in ("EC_L3", "EC_L4"):
        selected_name = freeze.get("selected", {}).get(level, {}).get("model")
        selected = judge.loc[(judge["annotation_level"].eq(level)) & (judge["model"].eq(selected_name))]
        judge_coverage = float(selected.iloc[0]["coverage"]) if len(selected) and bool(selected.iloc[0]["primary_safe"]) else 0.0
        baseline = best_safe.loc[best_safe["annotation_level"].eq(level)]
        baseline_coverage = float(baseline.iloc[0]["coverage"]) if len(baseline) else 0.0
        incremental.append({
            "annotation_level": level,
            "selected_evidencejudge": selected_name,
            "evidencejudge_safe_coverage": judge_coverage,
            "best_single_tool": str(baseline.iloc[0]["method"]) if len(baseline) else "NONE_QUALIFIED",
            "best_single_score_variant": str(baseline.iloc[0]["score_variant"]) if len(baseline) else "NONE",
            "best_single_safe_coverage": baseline_coverage,
            "incremental_safe_coverage": judge_coverage - baseline_coverage,
            "interpretation": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        })
    incremental_frame = pd.DataFrame(incremental)
    incremental_frame.to_csv(OUT / "incremental_safe_coverage_development.tsv", sep="\t", index=False)

    checks["selected_ec3_recomputes_safe"] = bool(
        len(judge.loc[(judge.annotation_level.eq("EC_L3")) & (judge.model.eq("LOGISTIC_L2")) & judge.primary_safe]) == 1
    )
    checks["ec4_not_misrepresented_as_safe"] = not bool(judge.loc[judge.annotation_level.eq("EC_L4"), "primary_safe"].any())
    status = "PASS" if all(checks.values()) else "FAIL"
    audit = {
        "status": status,
        "checks": checks,
        "development_only": True,
        "confirmatory_claim_allowed": False,
        "eligible_totals": {str(key): value for key, value in expected.items()},
        "single_tool_variants_audited": len(best_variants),
        "endpoint_definition": "precision>=0.95; cluster-bootstrap 95% lower>=0.95; accepted>=50; clusters>=20",
    }
    (REPORTS / "phase29_augmented_independent_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(judge.to_string(index=False))
    print(best_safe.to_string(index=False))
    print(incremental_frame.to_string(index=False))
    print(json.dumps(audit, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)
    print("CHECKPOINT_29D_AUGMENTED_DEVELOPMENT_AUDIT_PASS")


if __name__ == "__main__":
    main()
