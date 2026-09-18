"""Pure TRAIN pair-union and pre-sampling census rules."""
from collections import Counter


ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")
COVARIATE_FIELDS = {"protein_id", "node_id", "component_id", "role", "primary_pfam"}
IDENTITY_FIELDS = (
    "query_protein_id", "query_node", "query_component_id", "query_role",
    "reference_node", "reference_component_id", "reference_protein_id",
    "reference_activity_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3",
    "ec_l4", "canonical_rhea", "evidence_tier",
)
OUTCOME_FIELDS = (
    "query_documented_activity_count", "query_documented_ec_l3_count",
    "query_documented_ec_l4_count", "query_documented_rhea_count",
    "observed_same_ec_l3", "observed_same_ec_l4",
    "exact_rhea_outcome_evaluable", "observed_same_exact_rhea",
    "deepest_shared_recorded_ec_level", "ground_truth_only",
)
RETRIEVAL_FIELDS = {
    "rrf60_score", "mmseqs_rank", "foldseek_rank", "candidate_union_provenance",
}
AUGMENTATION_FIELDS = {
    "augmentation_mechanisms", "maximum_mechanism_inclusion_probability",
    "augmentation_probability_is_population_weight", "candidate_origin",
}


def require(value, message):
    if not value: raise ValueError(message)


def validate_covariates(rows):
    covariates = {}
    for row in rows:
        require(COVARIATE_FIELDS <= set(row), "covariate schema")
        protein = row["protein_id"]
        require(isinstance(protein, str) and protein and protein not in covariates and row["role"] in ROLES, "covariate identity")
        covariates[protein] = {name: row[name] for name in COVARIATE_FIELDS}
    require(covariates and set(row["role"] for row in covariates.values()) == set(ROLES), "covariate role universe")
    return covariates


def validate_common(row, covariates):
    require(set(IDENTITY_FIELDS) <= set(row) and set(OUTCOME_FIELDS) <= set(row), "pair common schema")
    query, reference = covariates.get(row["query_protein_id"]), covariates.get(row["reference_protein_id"])
    require(query is not None and reference is not None and query["role"] == reference["role"] == row["query_role"] == "TRAIN", "TRAIN pair scope")
    require(row["query_node"] == query["node_id"] and row["query_component_id"] == query["component_id"], "query identity")
    require(row["reference_node"] == reference["node_id"] and row["reference_component_id"] == reference["component_id"], "reference identity")
    require(row["query_node"] != row["reference_node"], "same exact-sequence node")
    require(all(type(row[name]) is int and row[name] > 0 for name in ("query_documented_activity_count", "query_documented_ec_l3_count", "query_documented_ec_l4_count")), "query outcome counts")
    require(type(row["query_documented_rhea_count"]) is int and row["query_documented_rhea_count"] >= 0, "query Rhea count")
    require(type(row["observed_same_ec_l3"]) is bool and type(row["observed_same_ec_l4"]) is bool and type(row["exact_rhea_outcome_evaluable"]) is bool, "outcome types")
    require(not row["observed_same_ec_l4"] or row["observed_same_ec_l3"], "EC hierarchy outcome")
    require(row["deepest_shared_recorded_ec_level"] == (4 if row["observed_same_ec_l4"] else (3 if row["observed_same_ec_l3"] else 0)), "deepest EC outcome")
    require(row["ground_truth_only"] is True, "outcome provenance")
    if row["exact_rhea_outcome_evaluable"]:
        require(type(row["observed_same_exact_rhea"]) is bool and row["canonical_rhea"] is not None and row["query_documented_rhea_count"] > 0, "evaluable Rhea outcome")
    else:
        require(row["observed_same_exact_rhea"] is None, "nullable Rhea outcome")
    canonical = row["canonical_ec"]; parts = canonical.split(".") if isinstance(canonical, str) else []
    require(len(parts) == 4 and row["ec_l1"] == parts[0] and row["ec_l2"] == ".".join(parts[:2]) and row["ec_l3"] == ".".join(parts[:3]) and row["ec_l4"] == canonical, "reference EC hierarchy")
    return query, reference


def shared_payload(row):
    return {name: row[name] for name in IDENTITY_FIELDS + OUTCOME_FIELDS}


def build_union(retrieval_rows, augmentation_rows, covariate_rows):
    covariates = validate_covariates(covariate_rows); records = {}
    for origin, rows in (("RETRIEVAL", retrieval_rows), ("AUGMENTATION", augmentation_rows)):
        seen = set()
        for row in rows:
            query, reference = validate_common(row, covariates)
            key = row["query_protein_id"], row["reference_protein_id"], row["reference_activity_id"]
            require(key not in seen, "duplicate within origin"); seen.add(key)
            if origin == "RETRIEVAL":
                require(RETRIEVAL_FIELDS <= set(row) and row["candidate_union_provenance"] == "TRUTH_FREE_AUDITED_MODALITY_UNION_NO_POLICY_SELECTION", "retrieval provenance")
                retrieval = {"rrf60_score": row["rrf60_score"], "mmseqs_rank": row["mmseqs_rank"], "foldseek_rank": row["foldseek_rank"]}
                augmentation = None
            else:
                require(AUGMENTATION_FIELDS <= set(row) and row["candidate_origin"] == "SUPERVISED_TRAIN_AUGMENTATION_NOT_DEPLOYMENT_CANDIDATE", "augmentation provenance")
                require(row["augmentation_probability_is_population_weight"] is False and 0 < row["maximum_mechanism_inclusion_probability"] <= 1, "augmentation probability semantics")
                retrieval = None
                augmentation = {"augmentation_mechanisms": row["augmentation_mechanisms"], "augmentation_design_inclusion_probability": row["maximum_mechanism_inclusion_probability"]}
            if key not in records:
                records[key] = {"shared": shared_payload(row), "query_primary_pfam": query["primary_pfam"], "reference_primary_pfam": reference["primary_pfam"], "retrieval": retrieval, "augmentation": augmentation}
            else:
                item = records[key]
                require(item["shared"] == shared_payload(row), "cross-origin identity/outcome disagreement")
                require(item[origin.lower()] is None, "duplicate origin state")
                item[origin.lower()] = retrieval if origin == "RETRIEVAL" else augmentation
    require(records, "empty TRAIN pair union")
    activity_frequency = Counter(key[2] for key in records)
    family_frequency = Counter(item["query_primary_pfam"] for item in records.values())
    component_frequency = Counter(item["shared"]["reference_component_id"] for item in records.values())
    output = []
    for key in sorted(records):
        item, shared = records[key], records[key]["shared"]
        retrieval, augmentation = item["retrieval"], item["augmentation"]
        flags = []
        if shared["observed_same_ec_l3"] and not shared["observed_same_ec_l4"]: flags.append("H1_SAME_EC3_DIFFERENT_EC4")
        if shared["observed_same_ec_l4"] and shared["exact_rhea_outcome_evaluable"] and not shared["observed_same_exact_rhea"]: flags.append("H2_SAME_EC4_DIFFERENT_RHEA")
        fine_divergent = not shared["observed_same_ec_l4"] or (shared["exact_rhea_outcome_evaluable"] and not shared["observed_same_exact_rhea"])
        if item["query_primary_pfam"] is not None and item["query_primary_pfam"] == item["reference_primary_pfam"] and fine_divergent: flags.append("H5_SAME_PFAM_FINE_DIVERGENCE")
        output.append({**shared, "query_primary_pfam": item["query_primary_pfam"], "reference_primary_pfam": item["reference_primary_pfam"], "pair_origin": "BOTH" if retrieval and augmentation else ("RETRIEVAL" if retrieval else "AUGMENTATION"), "retrieval_present": retrieval is not None, "rrf60_score": retrieval["rrf60_score"] if retrieval else None, "mmseqs_rank": retrieval["mmseqs_rank"] if retrieval else None, "foldseek_rank": retrieval["foldseek_rank"] if retrieval else None, "augmentation_present": augmentation is not None, "augmentation_mechanisms": augmentation["augmentation_mechanisms"] if augmentation else None, "augmentation_design_inclusion_probability": augmentation["augmentation_design_inclusion_probability"] if augmentation else None, "augmentation_probability_is_population_weight": False, "hard_case_flags_available_before_features": "+".join(flags) if flags else "NONE_AT_PAIR_UNION_STAGE", "reference_activity_frequency": activity_frequency[key[2]], "query_family_frequency": family_frequency[item["query_primary_pfam"]], "reference_component_frequency": component_frequency[shared["reference_component_id"]], "sampling_applied": False, "sample_weight": None})
    return output
