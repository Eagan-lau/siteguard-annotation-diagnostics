#!/usr/bin/env python3
"""Apply frozen HIT-anchored EvidenceJudge V2 without reading external truth."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import phase26_finalize_blind_inference as base
import phase30_finalize_augmented_blind as phase30
from phase31_train_hit_evidencejudge import canonical_full, parse_enzyme_transfers
from siteguard.model import blend_scores


ROOT = Path(os.environ.get("SITEGUARD_ROOT", "workspace/V4"))
PHASE_ID = int(os.environ.get("EVIDENCEJUDGE_EXTERNAL_PHASE", "33"))
if PHASE_ID not in {31, 33}:
    raise RuntimeError(f"Unsupported EvidenceJudge external phase: {PHASE_ID}")
RPHASE = ROOT / f"results/phase{PHASE_ID}"
WORK = ROOT / f"data/interim/phase{PHASE_ID}_inference"
CHECKPOINTS = ROOT / "checkpoints"
REPORTS = ROOT / f"reports/phase{PHASE_ID}_external_blind"
MODEL_DIR = ROOT / "results/phase31_development"
ENZYME = ROOT / "data/external/enzyme_release_2026_06_10/enzyme.dat"
PROTOCOL = ROOT / {
    31: "PROJECT/Phase31_HIT_anchored_external_blind_protocol.md",
    33: "PROJECT/Phase33_blinded_sample_size_adapted_external_protocol.md",
}[PHASE_ID]
LEVELS = ("EC_L3", "EC_L4", "EXACT_RHEA")
FIXED_OLD_METHODS = phase30.FIXED_OLD_METHODS
AVAILABLE_METHOD_SCORE = phase30.AVAILABLE_METHOD_SCORE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def engineer(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    frame["hit_old_agreement_fraction"] = frame["hit_old_agreement_count"] / frame[
        "available_old_tools"
    ].clip(lower=1)
    frame["hit_old_disagreement_count"] = frame["available_old_tools"] - frame[
        "hit_old_agreement_count"
    ]
    frame["hit_clean_disagree"] = 1 - frame["hit_clean_agree"]
    frame["clean_neg_distance"] = -frame["clean_top1_distance"]
    frame["clean_log_gmm"] = np.log10(frame["clean_top1_gmm_confidence"].clip(lower=1e-12))
    frame["hit_logit_abs"] = frame["hit_top1_logit"].abs()
    for minimum in range(1, 8):
        frame[f"hit_old_agree_ge{minimum}"] = frame["hit_old_agreement_count"].ge(minimum).astype(int)
    missing = sorted(set(features) - set(frame.columns))
    for feature in missing:
        frame[feature] = np.nan
    frame[features] = frame[features].apply(pd.to_numeric, errors="coerce")
    return frame


def canonicalize_hit_labels(
    anchored: pd.DataFrame, hit: pd.DataFrame, transfers: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    hit_labels = hit[["query_id", "ec3_top1", "ec4_top1", "ec3_ec4_hierarchy_consistent"]].rename(
        columns={"query_id": "query_protein_id"}
    )
    output = anchored.merge(hit_labels, on="query_protein_id", how="left", validate="many_to_one")
    mapped_l4 = {
        query: canonical_full(label, transfers)[0] or str(label)
        for query, label in hit_labels[["query_protein_id", "ec4_top1"]].itertuples(index=False, name=None)
    }
    ec4 = output["annotation_level"].eq("EC_L4")
    output.loc[ec4, "candidate_label"] = output.loc[ec4, "query_protein_id"].map(mapped_l4)
    ec3 = output["annotation_level"].eq("EC_L3")
    output.loc[ec3, "candidate_label"] = output.loc[ec3, "query_protein_id"].map(
        {query: label.rsplit(".", 1)[0] for query, label in mapped_l4.items()}
    )
    output["candidate_label"] = output["candidate_label"].astype(str)
    return output.drop(columns=["ec3_top1", "ec4_top1", "ec3_ec4_hierarchy_consistent"])


def canonicalize_baselines(
    baselines: pd.DataFrame, hit: pd.DataFrame, clean: pd.DataFrame,
    transfers: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    output = baselines.copy()
    hit_l4 = {str(row.query_id): canonical_full(row.ec4_top1, transfers)[0] or str(row.ec4_top1)
              for row in hit.itertuples(index=False)}
    clean_l4 = {str(row.query_id): canonical_full(row.clean_top1_ec4, transfers)[0] or str(row.clean_top1_ec4)
                for row in clean.itertuples(index=False)}
    for method, mapping in (("HIT_EC", hit_l4), ("CLEAN", clean_l4)):
        method_rows = output["method"].eq(method)
        l4 = method_rows & output["annotation_level"].eq("EC_L4")
        l3 = method_rows & output["annotation_level"].eq("EC_L3")
        output.loc[l4, "candidate_label"] = output.loc[l4, "query_protein_id"].map(mapping)
        output.loc[l3, "candidate_label"] = output.loc[l3, "query_protein_id"].map(
            {query: label.rsplit(".", 1)[0] for query, label in mapping.items()}
        )
    # Exact EC-L4 legacy labels from other tools are also canonicalized. EC-L3
    # is left unchanged when no exact fourth-level source is available.
    old_l4 = output["annotation_level"].eq("EC_L4") & ~output["method"].isin(["HIT_EC", "CLEAN"])
    output.loc[old_l4, "candidate_label"] = [
        canonical_full(value, transfers)[0] or str(value)
        for value in output.loc[old_l4, "candidate_label"]
    ]
    return output


def main() -> None:
    protocol_lock_path = ROOT / "models/phase33_protocol_lock.json"
    protocol_lock = None
    frozen_scripts_unchanged = True
    tool_training_resource_unchanged = True
    if PHASE_ID == 33:
        if not protocol_lock_path.is_file():
            raise FileNotFoundError(protocol_lock_path)
        protocol_lock = json.loads(protocol_lock_path.read_text(encoding="utf-8"))
        frozen_scripts_unchanged = all(
            (ROOT / path).is_file() and sha256(ROOT / path) == expected
            for path, expected in protocol_lock["script_sha256"].items()
        )
        tool_training_resource_unchanged = sha256(
            ROOT / "data/interim/phase31_tool_training_homology/tool_training_unique.fasta"
        ) == protocol_lock["resource_sha256"]["tool_training_unique"]
        if (
            sha256(PROTOCOL) != protocol_lock["protocol_sha256"]
            or not frozen_scripts_unchanged
            or not tool_training_resource_unchanged
        ):
            raise RuntimeError("Phase33 protocol, scripts or tool-training inventory changed after freeze")
    required = [
        CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}A5A_EXTERNAL_PAIR_EVIDENCE_PASS",
        CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}A5B_EXTERNAL_DEEP_SCORE_PASS",
        ROOT / f"data/interim/phase{PHASE_ID}_hit_ec/phase{PHASE_ID}_hit_ec_predictions.tsv",
        ROOT / f"data/interim/phase{PHASE_ID}_clean/phase{PHASE_ID}_clean_predictions.tsv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if (RPHASE / "external_blind_predictions.parquet").exists() or (RPHASE / "external_predictions.parquet").exists():
        raise RuntimeError(f"Phase{PHASE_ID} inference/evaluation output already exists")
    REPORTS.mkdir(parents=True, exist_ok=True)
    transfers, _ = parse_enzyme_transfers(ENZYME)

    cohort_columns = ["query_id", "external_cluster_id_30", "nearest_frozen_identity_fraction"]
    if PHASE_ID == 33:
        cohort_columns.extend([
            "tool_training_homology_independent", "tool_training_homology_preexcluded",
        ])
    locked = pd.read_parquet(RPHASE / "rcsb_strict_blind_cohort.parquet", columns=cohort_columns)
    if PHASE_ID == 33 and (
        not locked["tool_training_homology_independent"].astype(bool).all()
        or not locked["tool_training_homology_preexcluded"].astype(bool).all()
    ):
        raise RuntimeError("Phase33 cohort is not preprediction tool-training-homology clean")
    metadata = pd.read_parquet(RPHASE / "external_query_metadata.parquet")
    hit = pd.read_csv(required[2], sep="\t")
    clean = pd.read_csv(required[3], sep="\t")

    pairs = pd.read_parquet(RPHASE / "external_pair_evidence_blind.parquet")
    deep = np.load(WORK / "external_deep_global_scores.npy")
    if len(pairs) != len(deep):
        raise RuntimeError(f"pair/deep mismatch: {len(pairs)} vs {deep.shape}")
    for index, level in enumerate(LEVELS):
        pairs[f"score_deep_global_{level}"] = deep[:, index].astype(np.float32)
    config = json.loads((ROOT / "models/phase11/siteguard_model_config.json").read_text())
    tree = pairs[[f"score_lightgbm_global_{level}" for level in LEVELS]].to_numpy(np.float32)
    blended = blend_scores(deep.astype(np.float32), tree, config["deep_blend_weights"], LEVELS)
    for index, level in enumerate(LEVELS):
        pairs[f"score_siteguard_{level}"] = blended[:, index]
    pairs.to_parquet(RPHASE / "external_pair_scores_blind.parquet", index=False, compression="zstd")

    base.AVAILABLE_METHOD_SCORE = AVAILABLE_METHOD_SCORE
    top_all = base.make_top_predictions(pairs, metadata)
    top_all.to_parquet(RPHASE / "external_tool_top_predictions_all_blind.parquet", index=False, compression="zstd")
    top = top_all.loc[top_all["method"].isin(FIXED_OLD_METHODS)].copy()
    top.to_parquet(RPHASE / "external_tool_top_predictions_blind.parquet", index=False, compression="zstd")

    candidates = phase30.augmented_candidates(locked, metadata, top, hit, clean)
    anchored = candidates.loc[candidates["support__HIT_EC"].eq(1)].copy()
    if anchored.duplicated(["annotation_level", "query_protein_id"]).any():
        raise RuntimeError("HIT-anchored rows are not unique")
    anchored = canonicalize_hit_labels(anchored, hit, transfers)

    freeze_path = MODEL_DIR / "hit_evidencejudge_v2_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    features = list(freeze["feature_columns"])
    anchored = engineer(anchored, features)
    prediction_rows: list[pd.DataFrame] = []
    for level in ("EC_L3", "EC_L4"):
        info = freeze["selected"][level]
        model_path = Path(info["model_path"])
        if sha256(model_path) != info["model_sha256"]:
            raise RuntimeError(f"{level} frozen model hash mismatch")
        subset = anchored.loc[anchored["annotation_level"].eq(level)].copy()
        model = joblib.load(model_path)
        subset["selection_score"] = model.predict_proba(subset[features])[:, 1]
        subset["system_model"] = str(info["model"])
        subset["frozen_threshold"] = float(info["threshold"])
        subset["accepted"] = subset["selection_score"].ge(subset["frozen_threshold"])
        prediction_rows.append(subset[[
            "annotation_level", "query_protein_id", "query_cluster_id_30", "candidate_label",
            "selection_score", "system_model", "frozen_threshold", "accepted",
        ]])
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions["interpretation"] = f"PHASE{PHASE_ID}_BLIND_HIT_TOP1_RELIABILITY_TRUTH_NOT_OPENED"
    predictions.to_parquet(RPHASE / "external_blind_predictions.parquet", index=False, compression="zstd")
    anchored.to_parquet(RPHASE / "external_hit_anchored_feature_matrix_blind.parquet", index=False, compression="zstd")

    policies_path = MODEL_DIR / "single_tool_policy_freeze_v2.tsv"
    policies = pd.read_csv(policies_path, sep="\t")
    raw_policies = policies.copy()
    hit_score_mapping = {
        ("EC_L3", "hit_top1_logit"): "ec3_top1_logit",
        ("EC_L3", "hit_top1_softmax"): "ec3_top1_softmax",
        ("EC_L3", "hit_softmax_margin12"): "ec3_softmax_margin12",
        ("EC_L4", "hit_top1_logit"): "ec4_top1_logit",
        ("EC_L4", "hit_top1_softmax"): "ec4_top1_softmax",
        ("EC_L4", "hit_softmax_margin12"): "ec4_softmax_margin12",
    }
    for (level, generic), raw_name in hit_score_mapping.items():
        mask = (
            raw_policies["method"].eq("HIT_EC")
            & raw_policies["annotation_level"].eq(level)
            & raw_policies["score_variant"].eq(generic)
        )
        raw_policies.loc[mask, "score_variant"] = raw_name
    baselines = phase30.single_tool_predictions(locked, top, hit, clean, raw_policies)
    baselines = canonicalize_baselines(baselines, hit, clean, transfers)
    baselines.to_parquet(RPHASE / "external_blind_single_tool_predictions.parquet", index=False, compression="zstd")

    counts = predictions.groupby("annotation_level")["accepted"].sum().astype(int).to_dict()
    lock = {
        "phase": f"{PHASE_ID}B", "status": "PASS", "truth_opened": False,
        "strict_cohort_queries": len(locked), "blind_prediction_rows": len(predictions),
        "accepted_before_truth": counts, "feature_count": len(features),
        "fixed_old_methods": sorted(FIXED_OLD_METHODS), "available_old_methods": sorted(set(top["method"])),
        "features_missing": sorted(set(features) - set(anchored.columns)),
        "freeze_sha256": sha256(freeze_path),
        "ec_l3_model_sha256": sha256(Path(freeze["selected"]["EC_L3"]["model_path"])),
        "ec_l4_model_sha256": sha256(Path(freeze["selected"]["EC_L4"]["model_path"])),
        "single_tool_policy_sha256": sha256(MODEL_DIR / "single_tool_policy_freeze_v2.json"),
        "enzyme_sha256": sha256(ENZYME), "protocol_sha256": sha256(PROTOCOL),
        "cohort_lock_sha256": sha256(CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}A_EXTERNAL_COHORT_LOCKED"),
        "inference_script_sha256": sha256(Path(__file__)),
        "hit_manifest_truth_free": json.loads(Path(str(required[2]) + ".manifest.json").read_text())["truth_inputs_used"] is False,
        "clean_manifest_truth_free": json.loads(Path(str(ROOT / f"data/interim/phase{PHASE_ID}_clean/phase{PHASE_ID}_clean_predictions_raw.tsv") + ".manifest.json").read_text())["truth_inputs_used"] is False,
        "truth_columns_read": [], "cohort_columns_read": cohort_columns,
        "ec_canonicalization_applied_to_predictions": True,
        "tool_training_fasta_sha256": (
            "0b04c34acc91371b30f31077237870257af698aa859481f1c24e4fbf0c5ec1e7"
            if PHASE_ID == 33 else None
        ),
        "protocol_lock_sha256": sha256(protocol_lock_path) if PHASE_ID == 33 else None,
    }
    checks = {
        "truth_columns_not_read": not lock["truth_columns_read"],
        "feature_schema_complete": not lock["features_missing"],
        "candidate_label_not_a_feature": "candidate_label" not in features,
        "one_prediction_per_query_and_level": len(predictions) == 2 * len(locked)
        and not predictions.duplicated(["annotation_level", "query_protein_id"]).any(),
        "both_tool_manifests_truth_free": lock["hit_manifest_truth_free"] and lock["clean_manifest_truth_free"],
        "phase33_tool_training_homology_preexcluded": PHASE_ID != 33 or bool(
            locked["tool_training_homology_independent"].astype(bool).all()
            and locked["tool_training_homology_preexcluded"].astype(bool).all()
        ),
        "phase33_frozen_scripts_unchanged": PHASE_ID != 33 or frozen_scripts_unchanged,
        "phase33_tool_training_resource_unchanged": PHASE_ID != 33 or tool_training_resource_unchanged,
        "frozen_hashes_match_protocol": lock["freeze_sha256"] == "9ba0745a3035ca69b67f2acf7a3fdf0f52fe05c3812841db54a9f3fdeab61448"
        and lock["single_tool_policy_sha256"] == "3eacc6d40c127133dd1da79a73f2e624aadec8de2876b948ce0ba25f78424e91"
        and lock["enzyme_sha256"] == "e3cf02778ecea7c5b3e2813c8120625b5626d779a3657e60b8d27f063082c34a",
    }
    (REPORTS / f"phase{PHASE_ID}_blind_inference_qc.json").write_text(json.dumps(checks, indent=2) + "\n")
    failures = [key for key, value in checks.items() if not value]
    if failures:
        lock["status"] = "FAIL"; lock["failures"] = failures
        (ROOT / f"models/phase{PHASE_ID}_external_inference_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"Phase{PHASE_ID} blind inference QC failed: {failures}")
    lock_path = ROOT / f"models/phase{PHASE_ID}_external_inference_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    (CHECKPOINTS / f"CHECKPOINT_{PHASE_ID}B_BLIND_PREDICTIONS_LOCKED").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(lock, indent=2, sort_keys=True))
    print(f"CHECKPOINT_{PHASE_ID}B_BLIND_PREDICTIONS_LOCKED")


if __name__ == "__main__":
    main()


