"""Pure deterministic S4K balanced TRAIN sampling rules."""
import hashlib
import math
from collections import Counter, defaultdict


SEED = 20260819
TARGET_PAIRS = 1_500_000
MINIMUM_PAIRS = 500_000
MAX_PER_EXACT_RHEA = 5_000
MAX_PER_EC_L4 = 10_000
DIFFICULT_TARGET_FRACTION = 0.5
KEY_FIELDS = ("query_protein_id", "reference_protein_id", "reference_activity_id")


def require(value, message):
    if not value: raise ValueError(message)


def priority(row, stage, seed=SEED):
    text = "|".join([str(seed), stage] + [str(row[name]) for name in KEY_FIELDS])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def take_uniform(rows, limit, stage):
    require(type(limit) is int and not isinstance(limit, bool) and limit >= 0, "sampling limit")
    ordered = sorted(rows, key=lambda row: (priority(row, stage),) + tuple(row[name] for name in KEY_FIELDS))
    count = min(limit, len(ordered)); probability = count / len(ordered) if ordered else 0.0
    return ordered[:count], probability


def validate(row):
    required = {
        *KEY_FIELDS, "query_role", "query_node", "reference_node", "reference_component_id",
        "canonical_rhea", "ec_l4", "observed_same_ec_l4", "exact_rhea_outcome_evaluable",
        "observed_same_exact_rhea", "ground_truth_only", "outcome_semantics",
        "query_primary_pfam", "retrieval_present", "mmseqs_fident", "mmseqs_qcov",
        "mmseqs_tcov", "foldseek_rank", "hard_case_flags_available_before_features",
        "sampling_applied", "sample_weight",
    }
    require(required <= set(row), "S4J row schema")
    require(row["query_role"] == "TRAIN" and row["query_node"] != row["reference_node"], "TRAIN pair scope")
    require(row["ground_truth_only"] is True and row["outcome_semantics"] == "DOCUMENTED_CONCORDANCE_NOT_BIOCHEMICAL_NEGATIVE", "outcome semantics")
    require(row["sampling_applied"] is False and row["sample_weight"] is None, "S4J unsampled input")
    require(type(row["observed_same_ec_l4"]) is bool and type(row["exact_rhea_outcome_evaluable"]) is bool, "outcome types")
    if row["exact_rhea_outcome_evaluable"]: require(type(row["observed_same_exact_rhea"]) is bool, "evaluable Rhea outcome")
    else: require(row["observed_same_exact_rhea"] is None, "nullable Rhea outcome")


def available_flags(row):
    existing = row["hard_case_flags_available_before_features"]
    flags = set() if existing == "NONE_AT_PAIR_UNION_STAGE" else set(existing.split("+"))
    fine_divergence = not row["observed_same_ec_l4"] or (row["exact_rhea_outcome_evaluable"] and not row["observed_same_exact_rhea"])
    identity, qcov, tcov = row["mmseqs_fident"], row["mmseqs_qcov"], row["mmseqs_tcov"]
    adequate = identity is not None and qcov is not None and tcov is not None and qcov >= 0.70 and tcov >= 0.70
    if adequate and identity >= 0.40 and fine_divergence: flags.add("H3_HIGH_SEQUENCE_IDENTITY_FINE_DIVERGENCE")
    if row["foldseek_rank"] is not None and row["foldseek_rank"] <= 10 and fine_divergence: flags.add("H4_FOLDSEEK_TOP10_FINE_DIVERGENCE_PROXY")
    if adequate and identity < 0.30 and row["exact_rhea_outcome_evaluable"] and row["observed_same_exact_rhea"]: flags.add("H6_LOW_SEQUENCE_IDENTITY_SAME_EXACT_RHEA")
    return sorted(flags)


def select(rows, target_pairs=TARGET_PAIRS, minimum_pairs=MINIMUM_PAIRS, max_per_rhea=MAX_PER_EXACT_RHEA, max_per_ec4=MAX_PER_EC_L4):
    require(type(target_pairs) is int and target_pairs > 0 and type(minimum_pairs) is int and 0 <= minimum_pairs <= target_pairs, "size guard")
    require(type(max_per_rhea) is int and max_per_rhea > 0 and type(max_per_ec4) is int and max_per_ec4 > 0, "activity caps")
    seen, source = set(), []
    for row in rows:
        validate(row); key = tuple(row[name] for name in KEY_FIELDS); require(key not in seen, "duplicate S4J pair"); seen.add(key); source.append(dict(row))
    require(source, "empty S4J union")

    cluster_groups = defaultdict(list)
    for row in source: cluster_groups[(row["query_protein_id"], row["reference_component_id"], row["reference_activity_id"])].append(row)
    stage = []
    for group, values in sorted(cluster_groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        chosen, probability = take_uniform(values, 1, "REFERENCE_CLUSTER|" + "|".join(map(str, group)))
        item = chosen[0]; item["reference_cluster_dedup_probability"] = probability; stage.append(item)

    rhea_groups, missing_rhea = defaultdict(list), []
    for row in stage:
        (rhea_groups[row["canonical_rhea"]] if row["canonical_rhea"] is not None else missing_rhea).append(row)
    stage = []
    for label, values in sorted(rhea_groups.items()):
        chosen, probability = take_uniform(values, max_per_rhea, "RHEA_CAP|" + label)
        for row in chosen: row["exact_rhea_cap_probability"] = probability
        stage.extend(chosen)
    for row in missing_rhea: row["exact_rhea_cap_probability"] = 1.0
    stage.extend(missing_rhea)

    ec4_groups = defaultdict(list)
    for row in stage: ec4_groups[row["ec_l4"]].append(row)
    stage = []
    for label, values in sorted(ec4_groups.items()):
        chosen, probability = take_uniform(values, max_per_ec4, "EC4_CAP|" + label)
        for row in chosen: row["ec_l4_cap_probability"] = probability
        stage.extend(chosen)

    difficult, ordinary = [], []
    for row in stage:
        row["hard_case_flags_for_sampling"] = "+".join(available_flags(row)) or "NONE_AVAILABLE_BEFORE_GLOBAL_FEATURES"
        (difficult if row["hard_case_flags_for_sampling"] != "NONE_AVAILABLE_BEFORE_GLOBAL_FEATURES" else ordinary).append(row)
    if len(stage) > target_pairs:
        difficult_quota = min(len(difficult), math.ceil(target_pairs * DIFFICULT_TARGET_FRACTION))
        ordinary_quota = min(len(ordinary), target_pairs - difficult_quota)
        remainder = target_pairs - difficult_quota - ordinary_quota
        if remainder:
            extra_difficult = min(len(difficult) - difficult_quota, remainder); difficult_quota += extra_difficult; remainder -= extra_difficult
        if remainder: ordinary_quota += min(len(ordinary) - ordinary_quota, remainder)
    else: difficult_quota, ordinary_quota = len(difficult), len(ordinary)
    selected = []
    for label, values, quota in (("DIFFICULT", difficult, difficult_quota), ("ORDINARY", ordinary, ordinary_quota)):
        chosen, probability = take_uniform(values, quota, "TARGET_STRATUM|" + label)
        for row in chosen: row["difficulty_stratum_probability"] = probability; row["sampling_stratum"] = label
        selected.extend(chosen)
    require(len(selected) >= minimum_pairs, "minimum balanced TRAIN pair guard")

    activity_counts = Counter(row["reference_activity_id"] for row in selected); family_counts = Counter(row["query_primary_pfam"] for row in selected); component_counts = Counter(row["reference_component_id"] for row in selected)
    raw_weights = []
    for row in selected:
        activity_weight = 1 / activity_counts[row["reference_activity_id"]]; family_weight = 1 / family_counts[row["query_primary_pfam"]]; cluster_weight = 1 / component_counts[row["reference_component_id"]]
        raw = (activity_weight * family_weight * cluster_weight) ** (1 / 3); raw_weights.append(raw)
        row.update({"activity_balance_weight": activity_weight, "family_balance_weight": family_weight, "reference_cluster_weight": cluster_weight, "unscaled_training_weight": raw})
    scale = len(selected) / sum(raw_weights)
    for row in selected:
        row["sampling_probability"] = row["reference_cluster_dedup_probability"] * row["exact_rhea_cap_probability"] * row["ec_l4_cap_probability"] * row["difficulty_stratum_probability"]
        row["selection_probability_is_population_ipw"] = False; row["sampling_applied"] = True; row["sample_weight"] = row.pop("unscaled_training_weight") * scale
    selected.sort(key=lambda row: tuple(row[name] for name in KEY_FIELDS))
    manifest = {"input_pairs": len(source), "after_reference_cluster_dedup": len(cluster_groups), "after_activity_caps": len(stage), "selected_pairs": len(selected), "difficult_selected": sum(row["sampling_stratum"] == "DIFFICULT" for row in selected), "ordinary_selected": sum(row["sampling_stratum"] == "ORDINARY" for row in selected), "target_pairs": target_pairs, "minimum_pairs": minimum_pairs, "max_per_exact_rhea": max_per_rhea, "max_per_ec_l4": max_per_ec4, "sampling_probability_is_population_ipw": False, "mean_sample_weight": sum(row["sample_weight"] for row in selected) / len(selected)}
    return selected, manifest
