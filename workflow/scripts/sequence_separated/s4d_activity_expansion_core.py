"""Pure TRAIN-reference activity library and node-to-activity expansion rules."""
import importlib.util
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("s4c_candidate_union_core_for_activity", HERE / "s4c_candidate_union_core.py")
union_core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(union_core)


ROLES = union_core.ROLES
EVIDENCE_TIERS = ("GOLD", "SILVER")
PROTEIN_FIELDS = {"protein_id", "node_id", "component_id", "role"}
ACTIVITY_FIELDS = {
    "activity_id", "protein_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3",
    "ec_l4", "canonical_rhea", "evidence_tier",
}
LIBRARY_FIELDS = {
    "reference_node", "reference_component_id", "reference_protein_id",
    "reference_activity_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3",
    "ec_l4", "canonical_rhea", "evidence_tier", "reference_role",
    "activity_provenance",
}
UNION_FIELDS = {
    "query_node", "reference_node", "query_role", "reference_role",
    "reference_component_id", "mmseqs_rank", "foldseek_rank", "mmseqs_raw_rank",
    "foldseek_raw_rank", "source_class", "rrf60_score", "rrf_constant",
    "candidate_union_provenance", "query_truth_read",
    *{f"{modality}_{name}" for modality in ("mmseqs", "foldseek") for name in union_core.RAW_METRICS},
}


def require(value, message):
    if not value:
        raise ValueError(message)


def build_train_library(activity_rows, protein_rows):
    proteins = {}
    for row in protein_rows:
        require(set(row) == PROTEIN_FIELDS, "protein ledger schema")
        protein = row["protein_id"]
        require(isinstance(protein, str) and protein and protein not in proteins, "protein identity")
        require(row["role"] in ROLES and type(row["node_id"]) is int and type(row["component_id"]) is int, "protein ledger row")
        proteins[protein] = row
    require(proteins, "empty protein ledger")
    library, activity_ids = [], set()
    for row in activity_rows:
        require(set(row) == ACTIVITY_FIELDS, "activity projection schema")
        activity, protein = row["activity_id"], row["protein_id"]
        require(isinstance(activity, str) and activity and activity not in activity_ids, "activity identity")
        require(protein in proteins and proteins[protein]["role"] == "TRAIN", "activity is not TRAIN reference")
        require(row["evidence_tier"] in EVIDENCE_TIERS, "activity evidence tier")
        require(isinstance(row["canonical_ec"], str) and row["canonical_ec"], "canonical EC")
        require(row["ec_l4"] == row["canonical_ec"], "EC L4 identity")
        parts = row["canonical_ec"].split(".")
        require(len(parts) == 4 and row["ec_l1"] == parts[0] and row["ec_l2"] == ".".join(parts[:2]) and row["ec_l3"] == ".".join(parts[:3]), "EC hierarchy")
        require(row["canonical_rhea"] is None or (isinstance(row["canonical_rhea"], str) and row["canonical_rhea"].startswith("RHEA:")), "canonical Rhea")
        activity_ids.add(activity)
        meta = proteins[protein]
        library.append(
            {
                "reference_node": meta["node_id"],
                "reference_component_id": meta["component_id"],
                "reference_protein_id": protein,
                "reference_activity_id": activity,
                "canonical_ec": row["canonical_ec"],
                "ec_l1": row["ec_l1"],
                "ec_l2": row["ec_l2"],
                "ec_l3": row["ec_l3"],
                "ec_l4": row["ec_l4"],
                "canonical_rhea": row["canonical_rhea"],
                "evidence_tier": row["evidence_tier"],
                "reference_role": "TRAIN",
                "activity_provenance": "TRAIN_REFERENCE_SOURCE_ACTIVITY_PRESERVED",
            }
        )
    return sorted(library, key=lambda row: (row["reference_node"], row["reference_protein_id"], row["reference_activity_id"]))


def expand_union(union_rows, library_rows, query_nodes, query_role):
    require(query_role in ROLES, "query role")
    query_nodes = list(query_nodes)
    require(query_nodes and all(type(node) is int for node in query_nodes) and len(query_nodes) == len(set(query_nodes)), "query node universe")
    query_node_set = set(query_nodes)
    by_node = defaultdict(list)
    activity_keys = set()
    for row in library_rows:
        require(set(row) == LIBRARY_FIELDS, "library schema")
        require(row["reference_role"] == "TRAIN" and row["activity_provenance"] == "TRAIN_REFERENCE_SOURCE_ACTIVITY_PRESERVED", "library provenance")
        key = row["reference_protein_id"], row["reference_activity_id"]
        require(key not in activity_keys, "duplicate library activity")
        activity_keys.add(key)
        by_node[row["reference_node"]].append(row)
    for values in by_node.values():
        values.sort(key=lambda row: (row["reference_protein_id"], row["reference_activity_id"]))

    union_by_query = defaultdict(int)
    activity_by_query = defaultdict(int)
    seen_union = set()
    expanded = []
    for candidate in union_rows:
        require(set(candidate) == UNION_FIELDS, "union schema")
        require(candidate["query_role"] == query_role and candidate["reference_role"] == "TRAIN", "union role")
        require(candidate["query_node"] in query_node_set, "union query universe")
        require(not candidate["query_truth_read"] and candidate["candidate_union_provenance"] == "TRUTH_FREE_AUDITED_MODALITY_UNION_NO_POLICY_SELECTION", "union truth/provenance")
        key = candidate["query_node"], candidate["reference_node"]
        require(key not in seen_union, "duplicate union candidate")
        seen_union.add(key)
        union_by_query[candidate["query_node"]] += 1
        for activity in by_node.get(candidate["reference_node"], []):
            require(activity["reference_component_id"] == candidate["reference_component_id"], "candidate/activity component")
            expanded.append(
                {
                    **candidate,
                    **{name: activity[name] for name in (
                        "reference_protein_id", "reference_activity_id", "canonical_ec", "ec_l1",
                        "ec_l2", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier",
                    )},
                    "activity_provenance": "TRAIN_REFERENCE_ACTIVITY_EXPANDED_AFTER_TRUTH_FREE_RETRIEVAL",
                }
            )
            activity_by_query[candidate["query_node"]] += 1
    status = []
    for query in sorted(query_nodes):
        if activity_by_query[query]:
            state = "ACTIVITY_CANDIDATES_EXPANDED"
        elif union_by_query[query]:
            state = "NO_ELIGIBLE_REFERENCE_ACTIVITY"
        else:
            state = "NO_RETRIEVAL_CANDIDATE"
        status.append(
            {
                "query_node": query,
                "query_role": query_role,
                "status": state,
                "retrieved_reference_nodes": union_by_query[query],
                "expanded_reference_activities": activity_by_query[query],
                "query_truth_read": False,
            }
        )
    return expanded, status
