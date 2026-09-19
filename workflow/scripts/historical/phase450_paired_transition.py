#!/usr/bin/env python3
"""Post-hoc paired transitions from three immutable, already evaluated inputs.

This producer does not fit, score, calibrate, search, or read an unused cohort.
It does not create a scientific pass checkpoint; the runner may do so only after
the independent auditor passes.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ("EC_L3", "EC_L4", "EXACT_RHEA")
TARGETS = (0.90, 0.95)
DATA_PATHS = (
    "reports/phase445_full_candidate_inference_20260907/phase445_query_results.parquet",
    "reports/phase444_1_candidate_lineage_20260907/phase444_query_lineage.parquet",
    "results/phase12/abstention_thresholds.tsv",
)
LINEAGE = (
    "NO_DOCUMENTED_TRUTH", "NO_TRAIN_LIBRARY_SUPPORT",
    "LIBRARY_SUPPORTED_UNION50_MISS", "UNION50_POSITIVE_SCORED_SUBSET_MISS",
    "SCORED_SUBSET_POSITIVE",
)
STATES = (
    "NO_DOCUMENTED_TRUTH", "CANDIDATE_ABSENT_REJECTED",
    "CANDIDATE_ABSENT_ACCEPTED_DISAGREEMENT",
    "CANDIDATE_PRESENT_TOP1_DISAGREEMENT_REJECTED",
    "CANDIDATE_PRESENT_TOP1_DISAGREEMENT_ACCEPTED",
    "TOP1_CONCORDANT_REJECTED", "TOP1_CONCORDANT_ACCEPTED",
)
METRICS = (
    "coverage_delta", "accepted_concordant_fraction_delta",
    "accepted_disagreement_fraction_delta", "gain_concordant_fraction",
    "loss_concordant_fraction", "accepted_precision_delta",
)
OUTPUTS = (
    "phase450_query_transitions.parquet", "phase450_state_transitions.tsv",
    "phase450_sampling_loss_outcomes.tsv", "phase450_acceptance_changes.tsv",
    "phase450_paired_bootstrap.tsv", "phase450_summary.json",
)


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_table(path, records):
    require(bool(records), "Empty output table: " + path.name)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


def within(root, relative):
    require(not Path(relative).is_absolute(), "Input/output paths must be root-relative")
    require(".." not in Path(relative).parts, "Parent traversal is forbidden")
    result = root / relative
    require(result.resolve().is_relative_to(root), "Path escaped physical root")
    return result


def identities(root, contract):
    for item in contract["inputs"]:
        path = within(root, item["path"])
        require(path.is_file(), "Missing locked input: " + item["path"])
        require(path.stat().st_size == item["bytes"], "Input size mismatch: " + item["path"])
        require(digest(path) == item["sha256"], "Input hash mismatch: " + item["path"])


def configuration(root_arg, contract_path):
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    root = root_arg.resolve()
    require(root == Path(contract["physical_root"]).resolve(), "Physical root identity mismatch")
    require(Path(contract["root"]).resolve() == root, "Logical root identity mismatch")
    require(contract["seed"] == 20260819, "Frozen bootstrap seed mismatch")
    require(contract["bootstrap_replicates"] == 2000, "Frozen bootstrap count mismatch")
    require(tuple(x["path"] for x in contract["inputs"][:3]) == DATA_PATHS, "Three-data input order mismatch")
    require(len({x["path"] for x in contract["inputs"]}) == len(contract["inputs"]), "Duplicate input identity")
    out = within(root, contract["output_dir"])
    require(out != root and not out.exists(), "Output directory must not exist")
    require(not within(root, contract["checkpoint"]).exists(), "Final checkpoint must not exist")
    identities(root, contract)
    return root, contract, out


def boolean_column(frame, name):
    require(frame[name].notna().all(), "Null boolean: " + name)
    require(pd.api.types.is_bool_dtype(frame[name]), "Non-boolean schema: " + name)


def thresholds(path):
    # Parse decimal strings directly to Python float; pandas' C CSV parser can
    # round a boundary by one ULP and change >= decisions at an isotonic plateau.
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    result = {}
    for row in rows:
        key = (row["annotation_level"], float(row["target_precision"]))
        if key[0] not in LEVELS or key[1] not in TARGETS:
            continue
        require(key not in result, "Duplicate threshold cell")
        value = float(row["threshold"])
        require(math.isfinite(value) and 0 <= value <= 1, "Invalid frozen threshold")
        result[key] = value
    require(set(result) == {(level, target) for level in LEVELS for target in TARGETS}, "Missing threshold cell")
    require(result[("EXACT_RHEA", .9)] == result[("EXACT_RHEA", .95)], "Exact Rhea policy thresholds diverged")
    return result


def load_and_validate(root):
    predictions = pd.read_parquet(root / DATA_PATHS[0])
    lineage = pd.read_parquet(root / DATA_PATHS[1])
    policy = thresholds(root / DATA_PATHS[2])
    key = ["query_protein_id", "level"]
    require(len(predictions) == 27639 * 3 * 2, "Prediction row count mismatch")
    require(len(lineage) == 27639 * 3, "Lineage row count mismatch")
    require(set(predictions.cohort) == {"sampled", "full"}, "Unexpected cohort labels")
    require(set(predictions.level) == set(LEVELS) == set(lineage.level), "Unexpected levels")
    require(not predictions.duplicated(key + ["cohort"]).any(), "Prediction key is not unique")
    require(not lineage.duplicated(key).any(), "Lineage key is not unique")
    require(not predictions.isna().any().any(), "Missing prediction value")
    require(not lineage.isna().any().any(), "Missing lineage value")
    for field in ("correct", "candidate_available"):
        boolean_column(predictions, field)
    for field in ("documented_truth", "library_supported", "scored_available", "union_at_50"):
        boolean_column(lineage, field)
    require(np.isfinite(predictions.probability).all() and predictions.probability.between(0, 1).all(), "Probability out of range")
    require((predictions.candidate_activities > 0).all(), "Empty query candidate set")
    require((~predictions.correct | predictions.candidate_available).all(), "Concordant Top-1 without candidate")
    parts = []
    for cohort in ("sampled", "full"):
        part = predictions.loc[predictions.cohort.eq(cohort)].drop(columns="cohort")
        part = part.rename(columns={x: cohort + "_" + x for x in part.columns if x not in key})
        parts.append(part)
    combined = parts[0].merge(parts[1], on=key, how="outer", validate="one_to_one", indicator=True)
    require(combined._merge.eq("both").all(), "Sampled/full key sets differ")
    combined = combined.drop(columns="_merge").merge(lineage, on=key, how="outer", validate="one_to_one", indicator=True)
    require(combined._merge.eq("both").all(), "Prediction/lineage key sets differ")
    combined = combined.drop(columns="_merge").rename(columns={"state": "baseline_lineage_state"})
    require((combined.sampled_cluster_id_30 == combined.cluster_id_30).all(), "Sampled cluster mismatch")
    require((combined.full_cluster_id_30 == combined.cluster_id_30).all(), "Full cluster mismatch")
    require((combined.sampled_candidate_available == combined.scored_available).all(), "Sampled availability replay mismatch")
    require((combined.full_candidate_available == combined.union_at_50).all(), "Full availability replay mismatch")
    require((combined.sampled_candidate_activities == combined.scored_activity_pairs).all(), "Sampled candidate count mismatch")
    require((combined.full_candidate_activities >= combined.sampled_candidate_activities).all(), "Candidate superset count violated")
    require((~combined.sampled_candidate_available | combined.full_candidate_available).all(), "Candidate superset availability violated")
    require((combined.full_probability >= combined.sampled_probability).all(), "Maximum probability decreased")
    require((~combined.full_candidate_available | combined.library_supported).all(), "Retrieved candidate without library support")
    require((~combined.library_supported | combined.documented_truth).all(), "Library support without documented truth")
    require((combined.documented_truth | ~(combined.sampled_correct | combined.full_correct)).all(), "Concordance with undocumented truth")
    expected_state = np.select(
        [~combined.documented_truth, ~combined.library_supported, ~combined.full_candidate_available, ~combined.sampled_candidate_available],
        list(LINEAGE[:4]), default=LINEAGE[4],
    )
    require(np.array_equal(expected_state, combined.baseline_lineage_state.to_numpy()), "Five-state replay mismatch")
    reference_ids = None
    for level in LEVELS:
        subset = combined.loc[combined.level.eq(level)].sort_values("query_protein_id", kind="stable")
        require(len(subset) == 27639 and subset.query_protein_id.is_unique, "Per-level query identity mismatch")
        require(subset.cluster_id_30.nunique() == 1220, "Frozen cluster count mismatch")
        ids = list(zip(subset.query_protein_id, subset.cluster_id_30))
        require(reference_ids is None or ids == reference_ids, "Cross-level query/cluster identity mismatch")
        reference_ids = ids
        for target in TARGETS:
            threshold = policy[(level, target)]
            sa = subset.sampled_probability.to_numpy() >= threshold
            fa = subset.full_probability.to_numpy() >= threshold
            require((~sa | fa).all(), "Previously accepted query became rejected")
    exact = combined.loc[combined.level.eq("EXACT_RHEA")]
    require([int(exact.baseline_lineage_state.eq(x).sum()) for x in LINEAGE] == [1208, 9426, 3396, 9082, 4527], "Exact Rhea five-state identity mismatch")
    return combined, policy


def classify(documented, available, correct, accepted):
    documented, available, correct, accepted = [np.asarray(x, dtype=bool) for x in (documented, available, correct, accepted)]
    require((~correct | available).all(), "Synthetic/real invalid Top-1 truth")
    require((documented | ~(available | correct)).all(), "Evidence attached to unknown truth")
    return np.select(
        [~documented, ~available & ~accepted, ~available & accepted,
         available & ~correct & ~accepted, available & ~correct & accepted,
         correct & ~accepted, correct & accepted], STATES, default="INVALID",
    )


def transition_frame(subset, level, target, threshold):
    frame = subset.sort_values("query_protein_id", kind="stable").copy()
    frame["target"] = target
    frame["threshold"] = threshold
    for mode in ("sampled", "full"):
        frame[mode + "_accepted"] = frame[mode + "_probability"] >= threshold
        frame[mode + "_state"] = classify(frame.documented_truth, frame[mode + "_candidate_available"], frame[mode + "_correct"], frame[mode + "_accepted"])
    names = ["query_protein_id", "cluster_id_30", "level", "target", "threshold", "baseline_lineage_state", "documented_truth"]
    for mode in ("sampled", "full"):
        names += [mode + "_" + field for field in ("probability", "accepted", "correct", "reference_protein_id", "reference_activity_id", "candidate_available", "candidate_activities", "state")]
    return frame[names].reset_index(drop=True)


def event_arrays(frame):
    truth = frame.documented_truth.to_numpy(bool)
    sa, fa = frame.sampled_accepted.to_numpy(bool), frame.full_accepted.to_numpy(bool)
    sc, fc = frame.sampled_correct.to_numpy(bool), frame.full_correct.to_numpy(bool)
    sac, fac = sa & sc & truth, fa & fc & truth
    sad, fad = sa & ~sc & truth, fa & ~fc & truth
    return {
        "sampled_accepted": sa, "full_accepted": fa,
        "sampled_accepted_evaluable": sa & truth, "full_accepted_evaluable": fa & truth,
        "sampled_accepted_concordant": sac, "full_accepted_concordant": fac,
        "sampled_accepted_disagreement": sad, "full_accepted_disagreement": fad,
        "sampled_accepted_unevaluable": sa & ~truth, "full_accepted_unevaluable": fa & ~truth,
        "accepted_concordant_gain": ~sac & fac,
        "accepted_concordant_retained": sac & fac,
        "accepted_concordant_loss": sac & ~fac,
        "accepted_disagreement_new": ~sad & fad,
        "accepted_disagreement_repaired": sad & ~fad,
        "accepted_disagreement_retained": sad & fad,
        "full_top1_concordant": fc & truth,
        "reference_activity_changed": frame.sampled_reference_activity_id.to_numpy() != frame.full_reference_activity_id.to_numpy(),
    }


def metric_values(counts, denominator):
    shape = np.asarray(denominator).shape
    result = []
    with np.errstate(divide="ignore", invalid="ignore"):
        result.append((counts["full_accepted"] - counts["sampled_accepted"]) / denominator)
        result.append((counts["full_accepted_concordant"] - counts["sampled_accepted_concordant"]) / denominator)
        result.append((counts["full_accepted_disagreement"] - counts["sampled_accepted_disagreement"]) / denominator)
        result.append(counts["accepted_concordant_gain"] / denominator)
        result.append(counts["accepted_concordant_loss"] / denominator)
        result.append(counts["full_accepted_concordant"] / counts["full_accepted_evaluable"] - counts["sampled_accepted_concordant"] / counts["sampled_accepted_evaluable"])
    return np.stack([np.broadcast_to(np.asarray(x, dtype=float), shape) for x in result], axis=-1)


def bootstrap(frame, mask, seed, replicates):
    mask = np.asarray(mask, dtype=bool)
    cluster_names, inv = np.unique(frame.cluster_id_30.to_numpy(), return_inverse=True)
    nclusters = len(cluster_names)
    events = event_arrays(frame)
    totals = {name: np.bincount(inv, weights=(value & mask).astype(np.int64), minlength=nclusters) for name, value in events.items()}
    denominators = np.bincount(inv, weights=mask.astype(np.int64), minlength=nclusters)
    point_counts = {name: np.asarray(value.sum(), dtype=float) for name, value in totals.items()}
    point = metric_values(point_counts, np.asarray(denominators.sum(), dtype=float))
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, replicates, 25):
        size = min(25, replicates - start)
        draw = rng.integers(0, nclusters, size=(size, nclusters))
        counts = {name: value[draw].sum(axis=1) for name, value in totals.items()}
        values.append(metric_values(counts, denominators[draw].sum(axis=1)))
    values = np.concatenate(values, axis=0)
    rows = []
    for column, name in enumerate(METRICS):
        valid = np.isfinite(values[:, column])
        undefined = int((~valid).sum())
        # Deliberately do not condition on only defined replicates. If any draw
        # has an undefined denominator, no percentile interval is issued.
        interval = np.quantile(values[:, column], [.025, .975]) if not undefined else [None, None]
        rows.append({"metric": name, "estimate": float(point[column]) if np.isfinite(point[column]) else None,
                     "ci95_low": None if interval[0] is None else float(interval[0]),
                     "ci95_high": None if interval[1] is None else float(interval[1]),
                     "denominator_queries": int(mask.sum()), "cohort_clusters": nclusters,
                     "scope_clusters": int(np.count_nonzero(denominators)), "replicates": replicates,
                     "seed": seed, "valid_draws": int(valid.sum()), "undefined_draws": undefined,
                     "interval_status": "ALL_DRAWS_DEFINED" if not undefined else "NOT_ISSUED_UNDEFINED_DRAWS"})
    return rows


def analyze(combined, policy, contract):
    all_frames, transitions, outcomes, changes, contrasts = [], [], [], [], []
    for level in LEVELS:
        subset = combined.loc[combined.level.eq(level)]
        for target in TARGETS:
            frame = transition_frame(subset, level, target, policy[(level, target)])
            all_frames.append(frame)
            common = {"level": level, "target": target, "threshold": policy[(level, target)]}
            masks = [("ALL_QUERIES", "ALL", np.ones(len(frame), dtype=bool))]
            masks += [("BASELINE_LINEAGE_STATE", state, frame.baseline_lineage_state.eq(state).to_numpy()) for state in LINEAGE]
            for scope, state, mask in masks:
                denominator = int(mask.sum())
                counts = frame.loc[mask].groupby(["sampled_state", "full_state"], observed=True).size().to_dict()
                for old in STATES:
                    for new in STATES:
                        count = int(counts.get((old, new), 0))
                        transitions.append({**common, "scope": scope, "baseline_lineage_state": state,
                                            "sampled_state": old, "full_state": new, "queries": count,
                                            "denominator_queries": denominator, "fraction": count / denominator if denominator else None})
            for scope, mask in (("ALL_QUERIES", np.ones(len(frame), dtype=bool)),
                                ("BASELINE_SAMPLING_LOSS", frame.baseline_lineage_state.eq(LINEAGE[3]).to_numpy())):
                events = event_arrays(frame)
                for name, flags in events.items():
                    count = int((flags & mask).sum())
                    changes.append({**common, "scope": scope, "event": name, "queries": count,
                                    "denominator_queries": int(mask.sum()), "fraction": count / int(mask.sum()) if mask.any() else None})
                for row in bootstrap(frame, mask, contract["seed"], contract["bootstrap_replicates"]):
                    contrasts.append({**common, "scope": scope, **row})
            loss = frame.loc[frame.baseline_lineage_state.eq(LINEAGE[3])]
            require((~loss.sampled_candidate_available & loss.full_candidate_available).all(), "Sampling-loss subset identity violated")
            for state in STATES:
                count = int(loss.full_state.eq(state).sum())
                outcomes.append({**common, "scope": "BASELINE_SAMPLING_LOSS", "full_state": state,
                                 "queries": count, "denominator_queries": len(loss), "fraction": count / len(loss) if len(loss) else None})
    ledger = pd.concat(all_frames, ignore_index=True)
    require(len(ledger) == 27639 * 3 * 2 and not ledger.duplicated(["query_protein_id", "level", "target"]).any(), "Transition output key/count mismatch")
    return ledger, transitions, outcomes, changes, contrasts


def self_test():
    truth = np.array([False, True, True, True, True, True, True])
    available = np.array([False, False, False, True, True, True, True])
    correct = np.array([False, False, False, False, False, True, True])
    accepted = np.array([True, False, True, False, True, False, True])
    require(classify(truth, available, correct, accepted).tolist() == list(STATES), "Seven-state synthetic partition failed")
    try:
        classify([True], [False], [True], [True])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid correct-without-candidate was accepted")
    toy = pd.DataFrame({"cluster_id_30": ["a", "a", "b", "b"], "documented_truth": [True] * 4,
                        "sampled_accepted": [True, False, True, False], "full_accepted": [True] * 4,
                        "sampled_correct": [True, False, False, False], "full_correct": [False, True, True, False],
                        "sampled_reference_activity_id": ["x"] * 4, "full_reference_activity_id": ["y"] * 4})
    event = event_arrays(toy)
    require(int(event["accepted_concordant_gain"].sum()) == 2 and int(event["accepted_concordant_loss"].sum()) == 1, "Synthetic gain/loss failed")
    ci = bootstrap(toy, np.ones(4, dtype=bool), 20260819, 2000)
    expected = [.5, .25, .25, .5, .25, 0.0]
    require([x["estimate"] for x in ci] == expected, "Synthetic metric definitions failed")
    require(all(x["valid_draws"] == 2000 for x in ci), "Synthetic defined interval failed")
    undefined = bootstrap(toy, np.array([False, True, False, True]), 20260819, 2000)[-1]
    require(undefined["estimate"] is None and undefined["undefined_draws"] == 2000 and undefined["ci95_low"] is None, "Undefined precision was silently filtered")
    threshold = float("0.9416666626930236")
    require(np.array([threshold, np.nextafter(threshold, -np.inf)]).__ge__(threshold).tolist() == [True, False], "Threshold boundary semantics failed")
    print("PHASE450_SELF_TEST_PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        require(not args.preflight and args.root is None and args.contract is None, "Self-test must be isolated from real inputs")
        self_test()
        return
    require(args.root is not None and args.contract is not None, "Root and contract are required")
    root, contract, out = configuration(args.root, args.contract)
    combined, policy = load_and_validate(root)
    identities(root, contract)
    if args.preflight:
        print("PHASE450_PREFLIGHT_PASS")
        return
    out.mkdir(exist_ok=False)
    write_json(out / "reservation.json", {"status": "STARTED", "contract_sha256": digest(args.contract),
                                         "input_identities": contract["inputs"], "retry_authorized": False})
    try:
        ledger, transitions, outcomes, changes, contrasts = analyze(combined, policy, contract)
        with (out / OUTPUTS[0]).open("xb") as stream:
            ledger.to_parquet(stream, index=False, compression="zstd")
        for name, records in zip(OUTPUTS[1:5], (transitions, outcomes, changes, contrasts)):
            write_table(out / name, records)
        identities(root, contract)
        summary = {"status": "PRODUCER_PASS_POSTHOC_DESCRIPTIVE_NOT_INDEPENDENT_VALIDATION", "queries": 27639,
                   "clusters": 1220, "ledger_rows": len(ledger), "levels": list(LEVELS), "targets": list(TARGETS),
                   "exact_rhea_policy_thresholds_identical": True, "baseline_exact_rhea_sampling_loss_queries": 9082,
                   "state_order": list(STATES), "lineage_state_order": list(LINEAGE),
                   "table_rows": {name: len(records) for name, records in zip(OUTPUTS[1:5], (transitions, outcomes, changes, contrasts))},
                   "bootstrap_seed": contract["seed"], "bootstrap_replicates": contract["bootstrap_replicates"],
                   "precision_denominator": "accepted_queries_with_documented_truth_in_scope",
                   "fraction_denominator": "all_queries_in_scope_including_undocumented_truth",
                   "unknown_truth_policy": "NO_DOCUMENTED_TRUTH separate; never called accepted disagreement",
                   "undefined_bootstrap_policy": "record every undefined draw; do not issue interval if any draw undefined",
                   "scientific_scope": "frozen-policy paired transitions on previously evaluated benchmark; no new inference or calibration",
                   "all_inputs_unchanged": True, "new_external_validation": False, "checkpoint_created_by_producer": False,
                   "outputs": [{"path": name, "bytes": (out / name).stat().st_size, "sha256": digest(out / name)} for name in OUTPUTS[:-1]]}
        write_json(out / OUTPUTS[-1], summary)
        print("PHASE450_PRODUCER_PASS")
    except Exception as exc:
        write_json(out / "FAILED_producer.json", {"status": "FAIL_CLOSED", "type": type(exc).__name__, "error": str(exc), "automatic_retry": False})
        raise


if __name__ == "__main__":
    main()
