"""Frozen hierarchical selection and abstention logic."""

from __future__ import annotations

from typing import Mapping


LEVELS = ("EC_L3", "EC_L4", "EXACT_RHEA")


def decide_highest_supported_resolution(
    probabilities: Mapping[str, float],
    labels: Mapping[str, str],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    missing = [level for level in LEVELS if level not in probabilities or level not in thresholds]
    if missing:
        raise ValueError(f"Missing decision inputs: {missing}")
    accepted = {level: float(probabilities[level]) >= float(thresholds[level]) for level in LEVELS}
    if accepted["EXACT_RHEA"] and accepted["EC_L3"] and labels.get("EXACT_RHEA"):
        resolution = "EXACT_RHEA"
    elif accepted["EC_L4"] and accepted["EC_L3"] and labels.get("EC_L4"):
        resolution = "EC_L4"
    elif accepted["EC_L3"] and labels.get("EC_L3"):
        resolution = "EC_L3"
    else:
        resolution = "ABSTAIN"
    if resolution == "ABSTAIN":
        return {
            "final_resolution": resolution,
            "final_label": "",
            "final_probability": None,
            "abstention_reason": "NO_LEVEL_REACHED_FROZEN_VALIDATION_THRESHOLD",
            **{f"accepted_{level}": accepted[level] for level in LEVELS},
        }
    return {
        "final_resolution": resolution,
        "final_label": str(labels[resolution]),
        "final_probability": float(probabilities[resolution]),
        "abstention_reason": "",
        **{f"accepted_{level}": accepted[level] for level in LEVELS},
    }
