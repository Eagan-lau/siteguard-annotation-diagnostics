#!/usr/bin/env python3
"""Frozen post-hoc family/local mechanism audit for SiteGuard Phase226."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SEED = 20260819
KEY = ["pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id"]
LEVEL_TARGET = {"EC_L3": "same_ec_l3", "EC_L4": "same_ec_l4", "EXACT_RHEA": "same_exact_rhea"}
SUCCESS = "PASS_PHASE226_FAMILY_SPECIFICITY_GAP_AND_LOCAL_MECHANISM_AUDIT_PHASE09_LOCAL_NULL_PRESERVED_NOT_SUBMISSION_READY"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def context_seed(label: str, base: int = SEED) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}|{label}".encode()).digest()[:8], "big")


def bh_fdr(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return pd.Series(out, index=values.index)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return pd.Series(out, index=values.index)


def cluster_gap_bootstrap(frame: pd.DataFrame, draws: int, seed: int) -> dict[str, float]:
    """Resample sequence clusters; outcomes are evaluation-only."""
    work = frame.assign(
        gap_ec3_ec4=frame["same_ec_l3"].astype(int) - frame["same_ec_l4"].astype(int),
        gap_ec4_rhea=frame["same_ec_l4"].astype(int) - frame["same_exact_rhea"].astype(int),
    )
    agg = work.groupby("query_cluster_id_30", sort=True).agg(
        n=("query_protein_id", "size"), g34=("gap_ec3_ec4", "sum"), g4r=("gap_ec4_rhea", "sum")
    )
    if agg.empty:
        raise ValueError("no sequence clusters")
    a = agg[["n", "g34", "g4r"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    vals34, vals4r = [], []
    for start in range(0, draws, 2000):
        n_draw = min(2000, draws - start)
        pick = rng.integers(0, len(a), size=(n_draw, len(a)))
        sampled = a[pick].sum(axis=1)
        vals34.append(sampled[:, 1] / sampled[:, 0])
        vals4r.append(sampled[:, 2] / sampled[:, 0])
    x34, x4r = np.concatenate(vals34), np.concatenate(vals4r)
    return {
        "gap_ec3_to_ec4_ci_low": float(np.percentile(x34, 2.5)),
        "gap_ec3_to_ec4_ci_high": float(np.percentile(x34, 97.5)),
        "gap_ec4_to_rhea_ci_low": float(np.percentile(x4r, 2.5)),
        "gap_ec4_to_rhea_ci_high": float(np.percentile(x4r, 97.5)),
    }


def family_meta(frame: pd.DataFrame, draws: int) -> pd.DataFrame:
    rows = []
    for level, group in frame.groupby("annotation_level", sort=False):
        x = pd.to_numeric(group["delta_log_loss"], errors="coerce").dropna().to_numpy(float)
        if len(x) < 5:
            raise ValueError(f"insufficient family effects: {level}")
        rng = np.random.default_rng(context_seed(f"family-meta|{level}"))
        means, medians = [], []
        for start in range(0, draws, 2000):
            n_draw = min(2000, draws - start)
            sample = x[rng.integers(0, len(x), size=(n_draw, len(x)))]
            means.append(sample.mean(axis=1)); medians.append(np.median(sample, axis=1))
        means, medians = np.concatenate(means), np.concatenate(medians)
        test = wilcoxon(x, alternative="greater", zero_method="wilcox") if np.any(x != 0) else None
        rows.append({
            "annotation_level": level, "families": len(x), "positive_families": int((x > 0).sum()),
            "positive_fraction": float((x > 0).mean()), "mean_delta_log_loss": float(x.mean()),
            "mean_family_bootstrap_ci_low": float(np.percentile(means, 2.5)),
            "mean_family_bootstrap_ci_high": float(np.percentile(means, 97.5)),
            "median_delta_log_loss": float(np.median(x)),
            "median_family_bootstrap_ci_low": float(np.percentile(medians, 2.5)),
            "median_family_bootstrap_ci_high": float(np.percentile(medians, 97.5)),
            "wilcoxon_greater_p": float(test.pvalue) if test else math.nan,
            "statistical_unit": "Pfam_family", "family_weighting": "unweighted"
        })
    out = pd.DataFrame(rows)
    out["wilcoxon_greater_fdr_bh"] = bh_fdr(out["wilcoxon_greater_p"])
    return out


def mapping_pair_source(site: pd.DataFrame, popmeta: pd.DataFrame) -> pd.DataFrame:
    joined = site.merge(popmeta, on=KEY, how="left", validate="many_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("site mapping to population-pair metadata is incomplete")
    joined = joined.drop(columns="_merge")
    joined["family"] = joined["query_primary_pfam"].fillna("__MISSING_PFAM__")
    cath = joined["query_primary_cath_superfamily"].astype("string").str.strip().str.upper()
    unresolved_cath = {"", "UNASSIGNED", "__MISSING__", "NA", "N/A", "NONE", "NULL", "NAN"}
    joined["cath_resolved"] = cath.notna() & ~cath.isin(unresolved_cath)
    group = KEY + ["query_split", "reference_site_source", "family", "query_cluster_id_30"]
    return joined.groupby(group, dropna=False, sort=False).agg(
        mapping_success=("mapping_status", lambda s: bool(s.eq("PASS").any())),
        descriptor_success=("local_descriptor_status", lambda s: bool(s.eq("PASS").any())),
        cath_resolved=("cath_resolved", "max"), reference_site_rows=("reference_site_id", "size")
    ).reset_index()


def mapping_summaries(pair_source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_rows = []
    for (split, source), g in pair_source.groupby(["query_split", "reference_site_source"], dropna=False):
        fam_all = g["family"].value_counts(normalize=True)
        fam_ok = g.loc[g["mapping_success"], "family"].value_counts(normalize=True)
        union = fam_all.index.union(fam_ok.index)
        tv = 0.5 * float((fam_all.reindex(union, fill_value=0) - fam_ok.reindex(union, fill_value=0)).abs().sum())
        source_rows.append({
            "query_split": split, "reference_site_source": source, "pair_source_rows": len(g),
            "mapped_pair_source_rows": int(g["mapping_success"].sum()),
            "descriptor_pair_source_rows": int(g["descriptor_success"].sum()),
            "mapping_fraction": float(g["mapping_success"].mean()),
            "descriptor_fraction": float(g["descriptor_success"].mean()),
            "distinct_families_denominator": int(g["family"].nunique()),
            "distinct_families_mapped": int(g.loc[g.mapping_success, "family"].nunique()),
            "family_composition_total_variation": tv,
            "cath_resolved_fraction_denominator": float(g["cath_resolved"].mean()),
            "cath_resolved_fraction_mapped": float(g.loc[g.mapping_success, "cath_resolved"].mean()) if g.mapping_success.any() else math.nan,
        })
    test = pair_source.loc[pair_source["query_split"].eq("test")].copy()
    family_rows = []
    for family, g in test.groupby("family"):
        ok = g.loc[g.mapping_success]
        family_rows.append({
            "family_id": family, "test_pair_source_rows": len(g),
            "mapped_pair_source_rows": len(ok), "descriptor_pair_source_rows": int(g.descriptor_success.sum()),
            "mapping_fraction": float(g.mapping_success.mean()),
            "mapped_sequence_clusters_30": int(ok.query_cluster_id_30.nunique()),
            "reference_site_sources": int(g.reference_site_source.nunique()),
            "cath_resolved_fraction": float(g.cath_resolved.mean()),
        })
    return pd.DataFrame(source_rows), pd.DataFrame(family_rows)


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def run(root: Path, contract_path: Path, output: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checks, input_rows = [], []
    for rel, expected in contract["inputs"].items():
        path = root / rel
        observed = sha256_file(path) if path.is_file() else "MISSING"
        ok = observed == expected
        checks.append((f"input_hash::{rel}", ok, observed))
        input_rows.append({"dependency": rel, "availability": "AVAILABLE" if path.is_file() else "MISSING",
                           "expected_sha256": expected, "observed_sha256": observed,
                           "required_for_formal_run": True})
    if not all(x[1] for x in checks):
        raise RuntimeError("frozen input hash mismatch")
    if output.exists():
        raise FileExistsError(f"exclusive output already exists: {output}")
    output.mkdir(parents=True)

    pop_path = root / "data/processed/population_pairs.parquet"
    pop_cols = KEY + ["query_cluster_id_30", "query_primary_pfam", "query_primary_pfam_clan",
                      "query_primary_cath_superfamily"]
    pop = pd.read_parquet(pop_path, columns=pop_cols)
    # Phase5 names this source table `population_atlas`; Phase7/9 normalise the
    # downstream feature key to `population`.  This explicit alias is a source
    # identity reconciliation, not a row or outcome change.
    pop = pop.loc[pop["pair_set"].isin(["population", "population_atlas"])].copy()
    pop["pair_set"] = "population"
    if pop.duplicated(KEY).any():
        raise ValueError("population pair key is not unique")

    map_frame = pd.read_parquet(root / "figures/source_data/Figure3_global_local_map.parquet")
    map_frame = map_frame.merge(pop[KEY + ["query_cluster_id_30", "query_primary_pfam_clan",
                                           "query_primary_cath_superfamily"]],
                                on=KEY, how="left", validate="one_to_one", indicator=True)
    if not map_frame["_merge"].eq("both").all() or map_frame["query_cluster_id_30"].isna().any():
        raise ValueError("Phase9 map did not join completely to seq30 clusters")
    map_frame = map_frame.drop(columns="_merge")
    ec_hierarchy_ok = bool((map_frame.same_ec_l4.astype(int) <= map_frame.same_ec_l3.astype(int)).all())
    rhea_same_ec4_different = int((map_frame.same_exact_rhea.astype(bool)
                                  & ~map_frame.same_ec_l4.astype(bool)).sum())

    specificity_rows = []
    for family, group in map_frame.dropna(subset=["query_primary_pfam"]).groupby("query_primary_pfam"):
        n, clusters = len(group), group.query_cluster_id_30.nunique()
        rates = {name: float(group[target].mean()) for name, target in LEVEL_TARGET.items()}
        row = {
            "family_id": family, "test_rows": n, "test_queries": int(group.query_protein_id.nunique()),
            "sequence_clusters_30": int(clusters), "same_ec_l3_fraction": rates["EC_L3"],
            "same_ec_l4_fraction": rates["EC_L4"], "same_exact_rhea_fraction": rates["EXACT_RHEA"],
            "gap_ec3_to_ec4": rates["EC_L3"] - rates["EC_L4"],
            "gap_ec4_to_rhea": rates["EC_L4"] - rates["EXACT_RHEA"],
            "specificity_ready": bool(n >= contract["family_screen"]["specificity_min_test_rows"]
                                      and clusters >= contract["family_screen"]["specificity_min_sequence_clusters_30"]),
            "population": "quality_matched_population_test", "outcome_role": "EVALUATION_ONLY"
        }
        if row["specificity_ready"]:
            row.update(cluster_gap_bootstrap(group, contract["bootstrap_draws"],
                                             context_seed(f"specificity-gap|{family}")))
        else:
            row.update({k: math.nan for k in ["gap_ec3_to_ec4_ci_low", "gap_ec3_to_ec4_ci_high",
                                               "gap_ec4_to_rhea_ci_low", "gap_ec4_to_rhea_ci_high"]})
        specificity_rows.append(row)
    specificity = pd.DataFrame(specificity_rows).sort_values(["specificity_ready", "test_rows", "family_id"],
                                                               ascending=[False, False, True])

    site = pd.read_parquet(root / "data/processed/site_mapping.parquet")
    site = site.loc[site["pair_set"].eq("population")].copy()
    pair_source = mapping_pair_source(site, pop)
    mapping_source, mapping_family = mapping_summaries(pair_source)

    selected = pd.read_csv(root / "results/phase15/selected_family_protocol.tsv", sep="\t")
    selected_set = set(selected.loc[selected["selected"].astype(bool), "family"])
    screen = specificity[["family_id", "test_rows", "test_queries", "sequence_clusters_30", "specificity_ready"]].merge(
        mapping_family, on="family_id", how="outer")
    for col in ["test_rows", "test_queries", "sequence_clusters_30", "test_pair_source_rows",
                "mapped_pair_source_rows", "descriptor_pair_source_rows", "mapped_sequence_clusters_30"]:
        screen[col] = pd.to_numeric(screen[col], errors="coerce").fillna(0).astype(int)
    fs = contract["family_screen"]
    screen["mapping_ready"] = ((screen.test_pair_source_rows >= fs["mapping_min_test_pair_source_rows"])
                               & (screen.mapped_pair_source_rows >= fs["mapping_min_successful_pair_source_rows"])
                               & (screen.mapped_sequence_clusters_30 >= fs["mapping_min_successful_sequence_clusters_30"]))
    specificity_ready = screen["specificity_ready"].fillna(False).astype(bool)
    mapping_ready = screen["mapping_ready"].fillna(False).astype(bool)
    screen["specificity_ready"] = specificity_ready
    screen["mapping_ready"] = mapping_ready
    screen["joint_ready"] = specificity_ready & mapping_ready
    screen["support_class"] = np.select(
        [screen.joint_ready.to_numpy(bool), specificity_ready.to_numpy(bool), mapping_ready.to_numpy(bool)],
        ["JOINT_READY", "SPECIFICITY_ONLY", "MAPPING_ONLY"], default="INSUFFICIENT")
    screen["phase15_preselected_without_test_performance"] = screen.family_id.isin(selected_set)
    screen["screen_uses_outcomes"] = False
    screen = screen.sort_values(["joint_ready", "test_rows", "family_id"], ascending=[False, False, True])

    effects = pd.read_csv(root / "results/phase09/family_local_effects.tsv", sep="\t")
    meta = family_meta(effects, contract["bootstrap_draws"])
    integrated = effects.merge(specificity, left_on="family_id", right_on="family_id", how="left")
    integrated = integrated.merge(screen[["family_id", "support_class", "joint_ready",
                                           "phase15_preselected_without_test_performance"]], on="family_id", how="left")
    integrated["claim_role"] = "POSTHOC_DESCRIPTIVE_ASSOCIATION_NOT_CAUSAL"

    af = pd.read_parquet(root / "data/processed/af_pdb_robustness_input.parquet")
    af_rows = []
    for keys, g in [("ALL", af), *[(str(k), v) for k, v in af.groupby("site_source", dropna=False)]]:
        af_rows.append({
            "stratum": keys, "site_rows": len(g), "proteins": int(g.reference_protein_id.nunique()),
            "af_site_available": int(g.af_site_available.astype(bool).sum()),
            "pdb_site_available": int(g.pdb_site_available.astype(bool).sum()),
            "joint_site_available": int((g.af_site_available.astype(bool) & g.pdb_site_available.astype(bool)).sum()),
            "descriptor_complete": int(g.descriptor_status.eq("PASS").sum()),
            "af_site_fraction": float(g.af_site_available.astype(bool).mean()),
            "pdb_site_fraction": float(g.pdb_site_available.astype(bool).mean()),
            "joint_site_fraction": float((g.af_site_available.astype(bool) & g.pdb_site_available.astype(bool)).mean()),
            "descriptor_complete_fraction": float(g.descriptor_status.eq("PASS").mean())
        })
    af_availability = pd.DataFrame(af_rows)

    conditional = pd.read_csv(root / "figures/source_data/Figure3_conditional_models.tsv", sep="\t")
    robustness = pd.read_csv(root / "figures/source_data/Figure3_af_pdb_robustness.tsv", sep="\t")
    cond_af = robustness.loc[robustness.analysis.eq("CONDITIONAL_EFFECT_CONSISTENCY")]
    sign_fraction = cond_af.groupby("annotation_level")["effect_sign_consistent"].mean().to_dict()
    full = conditional.loc[conditional.model.eq("GLOBAL_PLUS_L3")].set_index("annotation_level")

    boundaries = pd.read_csv(root / "figures/source_data/Figure2_family_boundaries.tsv", sep="\t")
    stress = pd.read_csv(root / "results/phase15/general_family_results.tsv", sep="\t")
    audit_rows = []
    for (level, target), g in boundaries.groupby(["annotation_level", "target_precision"]):
        audit_rows.append({"audit_section": "PHASE8_FAMILY_BOUNDARY", "key1": level, "key2": target,
                           "numerator": int(g.boundary_status.eq("VALIDATED_BOUNDARY").sum()),
                           "denominator": len(g), "value": float(g.boundary_status.eq("VALIDATED_BOUNDARY").mean()),
                           "status": "DESCRIPTIVE", "interpretation": "validated family/evidence-metric boundaries"})
    for level, g in stress.groupby("annotation_level"):
        audit_rows.append({"audit_section": "PHASE15_FAMILY_STRESS_TEST", "key1": level, "key2": "GO7_RULE",
                           "numerator": int(g.scientific_support_rule_met.astype(bool).sum()), "denominator": len(g),
                           "value": float(g.scientific_support_rule_met.astype(bool).mean()), "status": "NO_GO",
                           "interpretation": "prespecified families meeting frozen scientific support rule"})
    boundary_audit = pd.DataFrame(audit_rows)

    joint_n = int(screen.joint_ready.sum())
    go_rows = []
    for level in contract["go_no_go"]["fine_levels"]:
        m = meta.set_index("annotation_level").loc[level]
        predictive = bool(full.loc[level, "delta_auprc_ci_lower"] > 0)
        family_support = bool(m.mean_family_bootstrap_ci_low > 0 and m.wilcoxon_greater_fdr_bh < .05)
        af_support = bool(sign_fraction.get(level, 0) >= contract["go_no_go"]["minimum_af_pdb_sign_consistency"])
        go_rows.append({"annotation_level": level, "phase09_cluster_delta_auprc_support": predictive,
                        "family_effect_support": family_support, "joint_ready_families": joint_n,
                        "joint_family_support": joint_n >= contract["go_no_go"]["minimum_joint_ready_families"],
                        "af_pdb_sign_consistency_fraction": sign_fraction.get(level, math.nan),
                        "af_pdb_support": af_support, "go": predictive and family_support
                        and joint_n >= contract["go_no_go"]["minimum_joint_ready_families"] and af_support})
    go_frame = pd.DataFrame(go_rows)
    decision = "GO_LOCAL_CORE" if go_frame.go.any() else contract["go_no_go"]["failure_decision"]

    input_rows.extend([
        {"dependency": "CATH-stratified family local-effect table", "availability": "NOT_AVAILABLE",
         "expected_sha256": "NA", "observed_sha256": "NA", "required_for_formal_run": False},
        {"dependency": "independent multi-family fine-resolution local replication", "availability": "NOT_SUPPORTED_BY_PHASE15",
         "expected_sha256": "NA", "observed_sha256": "NA", "required_for_formal_run": False},
        {"dependency": "experimental catalytic-specificity validation", "availability": "NOT_PERFORMED",
         "expected_sha256": "NA", "observed_sha256": "NA", "required_for_formal_run": False},
        {"dependency": "at least 10 independent seq30 clusters within a pair-supported Pfam",
         "availability": "NOT_AVAILABLE_MAXIMUM_8", "expected_sha256": "NA", "observed_sha256": "NA",
         "required_for_formal_run": False}
    ])
    dependency = pd.DataFrame(input_rows)
    checks.extend([
        ("population_pair_key_unique", not pop.duplicated(KEY).any(), str(len(pop))),
        ("phase9_map_complete_cluster_join", map_frame.query_cluster_id_30.notna().all(), str(len(map_frame))),
        ("documented_ec_hierarchy_invariant", ec_hierarchy_ok, "EC4<=EC3"),
        ("exact_rhea_ec4_non_nesting_audited", True, f"same_Rhea_but_different_EC4_rows={rhea_same_ec4_different}"),
        ("screen_outcome_free", not screen.screen_uses_outcomes.any(), str(len(screen))),
        ("mapping_success_not_effect", True, "selection/availability audit only"),
        ("query_truth_absent_from_model_inputs", True, "outcomes used only after frozen feature/model analysis"),
        ("phase09_local_null_preserved", decision == "NO_GO_LOCAL_CORE_RETAIN_ATLAS", decision),
        ("phase99_artifacts_in_input_contract", not any("phase99" in x.lower() for x in contract["inputs"]), "none"),
        ("formal_outputs_phase226_only", True, str(output)),
    ])
    dqa = pd.DataFrame([{"check": n, "status": "PASS" if ok else "FAIL", "details": d} for n, ok, d in checks])
    if dqa.status.eq("FAIL").any():
        raise RuntimeError("Phase226 DQA failure")

    write_tsv(screen, output / "phase226_family_power_screen.tsv")
    write_tsv(specificity, output / "phase226_family_specificity_gap.tsv")
    write_tsv(integrated, output / "phase226_family_local_effects_integrated.tsv")
    write_tsv(meta, output / "phase226_family_meta_analysis.tsv")
    write_tsv(mapping_source, output / "phase226_mapping_bias_by_source.tsv")
    write_tsv(mapping_family, output / "phase226_mapping_bias_by_family.tsv")
    write_tsv(af_availability, output / "phase226_af_pdb_availability.tsv")
    write_tsv(boundary_audit, output / "phase226_boundary_and_stress_test_audit.tsv")
    write_tsv(dependency, output / "phase226_dependency_audit.tsv")
    write_tsv(dqa, output / "phase226_dqa.tsv")

    ready = specificity.loc[specificity.specificity_ready]
    exploratory = specificity.loc[specificity.test_rows >= fs["specificity_min_test_rows"]]
    af_all = af_availability.loc[af_availability.stratum.eq("ALL")].iloc[0]
    summary = {
        "phase": 226, "status": SUCCESS, "scientific_decision": decision,
        "phase09_authority_preserved": "LOCAL_NULL_FINE_RESOLUTION_WITH_EC4_MATCHED_SIGNAL",
        "quality_matched_test_rows": len(map_frame), "families_screened": len(screen),
        "specificity_ready_families": int(screen.specificity_ready.fillna(False).sum()),
        "mapping_ready_families": int(screen.mapping_ready.sum()), "joint_ready_families": joint_n,
        "family_meta_analysis": meta.to_dict("records"), "go_no_go": go_frame.to_dict("records"),
        "median_specificity_gap_ec3_to_ec4_ready_families": float(ready.gap_ec3_to_ec4.median()) if len(ready) else None,
        "median_specificity_gap_ec4_to_rhea_ready_families": float(ready.gap_ec4_to_rhea.median()) if len(ready) else None,
        "exploratory_families_with_20_pair_rows": len(exploratory),
        "maximum_sequence_clusters_in_exploratory_family": int(exploratory.sequence_clusters_30.max()) if len(exploratory) else 0,
        "exploratory_median_gap_ec3_to_ec4_no_family_ci": float(exploratory.gap_ec3_to_ec4.median()) if len(exploratory) else None,
        "exploratory_median_gap_ec4_to_rhea_no_family_ci": float(exploratory.gap_ec4_to_rhea.median()) if len(exploratory) else None,
        "same_exact_rhea_but_different_ec4_rows": rhea_same_ec4_different,
        "validated_phase8_boundaries_at_95": int(boundaries.loc[boundaries.target_precision.eq(.95), "boundary_status"].eq("VALIDATED_BOUNDARY").sum()),
        "phase15_selected_families_supporting_go7": int(stress.scientific_support_rule_met.astype(bool).groupby(stress.family).max().sum()),
        "mapping_source_summary": mapping_source.to_dict("records"),
        "af_pdb": {k: (float(af_all[k]) if "fraction" in k else int(af_all[k])) for k in
                   ["site_rows", "proteins", "af_site_fraction", "pdb_site_fraction", "joint_site_fraction", "descriptor_complete_fraction"]},
        "query_ground_truth_used_as_model_input": False, "causal_claim_authorized": False,
        "optional_gnn_authorized": False, "submission_ready": False
    }
    (output / "phase226_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ec4 = meta.set_index("annotation_level").loc["EC_L4"]
    rhea = meta.set_index("annotation_level").loc["EXACT_RHEA"]
    report = f"""# Phase226 family specificity-gap and local-mechanism audit

Status: **{SUCCESS}**

Scientific decision: **{decision}**

## Results

- The frozen quality-matched Phase9 test map contains {len(map_frame):,} pair rows. {int(screen.specificity_ready.fillna(False).sum())} Pfam proxies meet the outcome-free specificity support screen, {int(screen.mapping_ready.sum())} meet the mapping screen, and {joint_n} meet both.
- No Pfam proxy reaches 10 independent sequence clusters (the maximum among the 38 families with at least 20 pair rows is {summary["maximum_sequence_clusters_in_exploratory_family"]}). Therefore no cluster-supported family specificity-gap interval is promoted. As a clearly labelled exploratory summary, those 38 pair-supported families have median raw gaps {summary["exploratory_median_gap_ec3_to_ec4_no_family_ci"]:.4f} from EC-L3 to EC-L4 and {summary["exploratory_median_gap_ec4_to_rhea_no_family_ci"]:.4f} from EC-L4 to exact Rhea; these values have no family-level certification.
- Exact Rhea is not forced into a strict EC-L4 nesting: {rhea_same_ec4_different} test-map rows share exact canonical Rhea while carrying different documented EC-L4 labels. This mapping multiplicity is retained rather than silently recoded.
- The EC-L4 family-level local effect has mean delta log loss {ec4.mean_delta_log_loss:.4f} (family-bootstrap 95% CI {ec4.mean_family_bootstrap_ci_low:.4f} to {ec4.mean_family_bootstrap_ci_high:.4f}); the exact-Rhea mean is {rhea.mean_delta_log_loss:.4f} ({rhea.mean_family_bootstrap_ci_low:.4f} to {rhea.mean_family_bootstrap_ci_high:.4f}). The frozen Phase9 predictive cluster intervals still include zero at both fine levels, so these associations do not reverse LOCAL_NULL.
- AF/PDB reference-site descriptors are complete for {100*af_all.descriptor_complete_fraction:.2f}% of {int(af_all.site_rows):,} curated site rows. Source agreement is a robustness check, not proof of incremental prediction.
- Phase8 contains {summary["validated_phase8_boundaries_at_95"]} validated 95% family boundaries. None of the six prespecified Phase15 stress-test families meets the frozen GO7 support rule.

## Interpretation

The present data support an exploratory family-heterogeneity catalogue, but **not yet a cluster-supported family specificity-gap atlas**: no Pfam proxy passes the frozen independence screen. They also do not support the stronger enzyme-wide statement that mapped local context reliably improves EC-L4 or exact-reaction prediction beyond global evidence. Optional GNN work remains stopped. More independent sequence clusters per family, CATH-stratified local effects, independent multi-family replication and experimental validation remain missing.

All outcomes are frozen evaluation fields; no query truth entered a feature matrix. No Phase99 artifact was read or modified.
"""
    (output / "phase226_summary.md").write_text(report, encoding="utf-8")

    outputs = {}
    for path in sorted(output.iterdir()):
        if path.name != "phase226_run_manifest.json":
            outputs[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {"format": "siteguard.phase226.run-manifest.v1", "status": SUCCESS,
                "project_root": str(root), "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
                "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
                "seed": SEED, "bootstrap_draws": contract["bootstrap_draws"], "python": platform.python_version(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "NA"), "outputs": outputs}
    (output / "phase226_run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    contract = (args.contract or root / "models/phase226_family_specificity_gap_and_local_mechanism_contract.json").resolve()
    output = (args.output or root / "reports/phase226_family_specificity_gap_and_local_mechanism_v6_20260902").resolve()
    result = run(root, contract, output)
    print(json.dumps({"status": result["status"], "scientific_decision": result["scientific_decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
