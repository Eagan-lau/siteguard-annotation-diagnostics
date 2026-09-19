"""Pure truth-free candidate retention rules for S4B."""
import math


ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")
MODALITIES = ("mmseqs", "foldseek")
NODE_FIELDS = {
    "node_id", "component_id", "role", "sequence_sha256", "sequence_length",
    "representative_protein_id", "protein_id_count",
}
RAW_FIELDS = {
    "query_node", "reference_node", "fident", "alnlen", "qstart", "qend", "qlen",
    "tstart", "tend", "tlen", "qcov", "tcov", "evalue", "bits",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_nodes(rows):
    require(isinstance(rows, list) and rows, "nonempty node ledger")
    by_id, role_by_component = {}, {}
    for row in rows:
        require(set(row) == NODE_FIELDS, "node ledger schema")
        node, component, role = row["node_id"], row["component_id"], row["role"]
        require(type(node) is int and node >= 0 and node not in by_id, "node identity")
        require(type(component) is int and component >= 0 and role in ROLES, "component/role")
        require(component not in role_by_component or role_by_component[component] == role, "component split across roles")
        require(type(row["sequence_length"]) is int and 0 < row["sequence_length"] <= 65535, "sequence length")
        require(isinstance(row["sequence_sha256"], str) and len(row["sequence_sha256"]) == 64, "sequence hash")
        require(isinstance(row["representative_protein_id"], str) and row["representative_protein_id"], "representative")
        require(type(row["protein_id_count"]) is int and row["protein_id_count"] >= 1, "protein count")
        by_id[node] = row
        role_by_component[component] = role
    require(set(row["role"] for row in rows) == set(ROLES), "role coverage")
    return by_id


def validate_alignment(row, nodes, modality):
    require(set(row) == RAW_FIELDS, "raw retrieval schema")
    query, reference = row["query_node"], row["reference_node"]
    require(query in nodes and reference in nodes, "raw node universe")
    integer_fields = ("alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen")
    require(all(type(row[name]) is int for name in integer_fields), "alignment integer fields")
    require(row["qlen"] == nodes[query]["sequence_length"] and row["tlen"] == nodes[reference]["sequence_length"], "alignment length identity")
    require(1 <= row["qstart"] <= row["qend"] <= row["qlen"], "query coordinates")
    require(1 <= row["tstart"] <= row["tend"] <= row["tlen"], "target coordinates")
    qspan = row["qend"] - row["qstart"] + 1
    tspan = row["tend"] - row["tstart"] + 1
    require(row["alnlen"] >= max(qspan, tspan) > 0, "alignment length")
    for name in ("fident", "qcov", "tcov", "evalue", "bits"):
        require(type(row[name]) in (int, float) and not isinstance(row[name], bool) and math.isfinite(row[name]), "finite " + name)
    require(0 <= row["fident"] <= 1 and 0 <= row["qcov"] <= 1 and 0 <= row["tcov"] <= 1, "alignment ratio")
    require(row["evalue"] >= 0, "alignment evalue")
    if modality == "mmseqs":
        require(row["bits"] >= 0, "alignment bits")
    require(abs(row["qcov"] - qspan / row["qlen"]) <= 0.00101, "query coverage serialization")
    require(abs(row["tcov"] - tspan / row["tlen"]) <= 0.00101, "target coverage serialization")


def retain_candidates(raw_rows, node_rows, query_role, modality, top_k):
    require(query_role in ROLES, "query role")
    require(modality in MODALITIES, "modality")
    require(type(top_k) is int and not isinstance(top_k, bool) and top_k > 0, "top K")
    nodes = validate_nodes(node_rows)
    expected_queries = sorted(node for node, row in nodes.items() if row["role"] == query_role)
    require(expected_queries, "empty query role")
    raw_count = {node: 0 for node in expected_queries}
    self_removed = {node: 0 for node in expected_queries}
    component_filtered = {node: 0 for node in expected_queries}
    retained = {node: [] for node in expected_queries}
    seen_components = {node: set() for node in expected_queries}
    completed_queries, current_query = set(), None

    for row in raw_rows:
        validate_alignment(row, nodes, modality)
        query, reference = row["query_node"], row["reference_node"]
        require(nodes[query]["role"] == query_role, "raw query role")
        require(nodes[reference]["role"] == "TRAIN", "reference is not TRAIN")
        if query != current_query:
            if current_query is not None:
                completed_queries.add(current_query)
            require(query not in completed_queries, "raw output is not query grouped")
            current_query = query
        raw_count[query] += 1
        raw_rank = raw_count[query]
        if query == reference:
            require(query_role == "TRAIN", "non-TRAIN self hit")
            self_removed[query] += 1
            continue
        reference_component = nodes[reference]["component_id"]
        if query_role != "TRAIN":
            require(nodes[query]["component_id"] != reference_component, "cross-role component leakage")
        if reference_component in seen_components[query] or len(retained[query]) >= top_k:
            component_filtered[query] += 1
            continue
        seen_components[query].add(reference_component)
        retained[query].append(
            {
                **row,
                "query_role": query_role,
                "reference_role": "TRAIN",
                "reference_component_id": reference_component,
                "raw_rank": raw_rank,
                "modality_rank": len(retained[query]) + 1,
                "modality": modality,
                "candidate_provenance": "QUERY_DERIVED_RETRIEVAL_PLUS_TRAIN_REFERENCE_METADATA",
            }
        )

    candidates = [row for query in expected_queries for row in retained[query]]
    status = []
    for query in expected_queries:
        if retained[query]:
            state = "CANDIDATES_RETAINED"
        elif raw_count[query] == 0:
            state = "ZERO_RAW_HITS"
        else:
            state = "NO_ELIGIBLE_REFERENCE_AFTER_FILTER"
        status.append(
            {
                "query_node": query,
                "query_role": query_role,
                "modality": modality,
                "status": state,
                "raw_hits": raw_count[query],
                "self_hits_removed": self_removed[query],
                "component_duplicate_or_beyond_top_k": component_filtered[query],
                "retained_candidates": len(retained[query]),
                "query_truth_read": False,
            }
        )
    return candidates, status
