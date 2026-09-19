#!/usr/bin/env python3
"""Freeze fair native-score abstention policies for every single-tool baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase29_audit_augmented_evidencejudge import ROOT, cluster_bootstrap, single_tool_rows, truth_tables


OUT = ROOT / "results/phase29_augmented"


def main() -> None:
    truth = truth_tables()
    data = single_tool_rows(truth)
    policies = []
    for (level, method), method_frame in data.groupby(["annotation_level", "method"], observed=True):
        candidates = []
        for (variant, direction), frame in method_frame.groupby(["score_variant", "direction"], observed=True):
            low_direction = direction == "LOW"
            for threshold in np.unique(frame["score"]):
                accepted = frame.loc[frame["score"].le(threshold) if low_direction else frame["score"].ge(threshold)]
                if len(accepted) < 20 or accepted["query_cluster_id_30"].nunique() < 10:
                    continue
                precision = float(accepted["correct"].mean())
                if precision < 0.95:
                    continue
                low, high = cluster_bootstrap(
                    accepted, f"SINGLE_FREEZE|{level}|{method}|{variant}|{threshold:.12g}", draws=10000
                )
                candidates.append({
                    "annotation_level": level, "method": method,
                    "score_variant": variant, "direction": direction,
                    "threshold": float(threshold), "development_accepted": len(accepted),
                    "development_clusters": int(accepted["query_cluster_id_30"].nunique()),
                    "development_precision": precision,
                    "development_cluster_bootstrap_low": low,
                    "development_cluster_bootstrap_high": high,
                })
        if candidates:
            choice = max(candidates, key=lambda row: (
                row["development_accepted"], row["development_precision"],
                row["development_cluster_bootstrap_low"], -abs(row["threshold"]),
            ))
            choice["policy_status"] = "FROZEN_NATIVE_SCORE_THRESHOLD"
            policies.append(choice)
        else:
            policies.append({
                "annotation_level": level, "method": method,
                "score_variant": "NONE", "direction": "NONE", "threshold": np.nan,
                "development_accepted": 0, "development_clusters": 0,
                "development_precision": np.nan,
                "development_cluster_bootstrap_low": np.nan,
                "development_cluster_bootstrap_high": np.nan,
                "policy_status": "FROZEN_ABSTAIN_ALL_NO_DEV_POLICY_AT_95_PRECISION",
            })
    frame = pd.DataFrame(policies).sort_values(["annotation_level", "method"], kind="mergesort")
    target = OUT / "single_tool_policy_freeze.tsv"
    frame.to_csv(target, sep="\t", index=False)
    manifest = {
        "status": "FROZEN_BEFORE_PHASE30_ACQUISITION",
        "selection_data": ["PHASE26_DEVELOPMENT", "PHASE28_DEVELOPMENT"],
        "selection_rule": "maximize accepted queries among native scalar thresholds with development precision>=0.95, accepted>=20, clusters>=10",
        "phase30_evaluation_rule": "apply thresholds unchanged; compute primary safe endpoint independently for each tool; best qualifying tool defines baseline",
        "policies": frame.replace({np.nan: None}).to_dict(orient="records"),
        "tsv_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    manifest_path = OUT / "single_tool_policy_freeze.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("CHECKPOINT_29E_SINGLE_TOOL_POLICIES_FROZEN")


if __name__ == "__main__":
    main()
