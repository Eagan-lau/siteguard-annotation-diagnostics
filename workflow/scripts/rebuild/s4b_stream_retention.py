"""Bounded-memory S4B candidate retention with sink-based output."""
import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("s4b_candidate_core_stream", HERE / "s4b_candidate_core.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


UNAVAILABILITY_REASONS = {"NO_REGISTERED_STRUCTURE"}


def stream_retain(
    raw_rows,
    node_rows,
    query_role,
    modality,
    top_k,
    candidate_sink,
    status_sink,
    unavailable_queries=None,
):
    core.require(query_role in core.ROLES, "query role")
    core.require(modality in core.MODALITIES, "modality")
    core.require(type(top_k) is int and not isinstance(top_k, bool) and top_k > 0, "top K")
    core.require(callable(candidate_sink) and callable(status_sink), "output sinks")
    nodes = core.validate_nodes(node_rows)
    expected = sorted(node for node, row in nodes.items() if row["role"] == query_role)
    expected_set = set(expected)
    unavailable = {} if unavailable_queries is None else dict(unavailable_queries)
    core.require(set(unavailable) <= expected_set, "unavailable query universe")
    core.require(all(reason in UNAVAILABILITY_REASONS for reason in unavailable.values()), "unavailability reason")
    core.require(modality == "foldseek" or not unavailable, "sequence modality cannot lack sequence input")

    cursor = 0
    current = None
    raw_count = self_removed = filtered = retained_count = 0
    seen_components = set()
    totals = {
        "queries": len(expected),
        "raw_hits": 0,
        "self_hits_removed": 0,
        "component_duplicate_or_beyond_top_k": 0,
        "retained_candidates": 0,
        "zero_raw_hit_queries": 0,
        "unavailable_queries": len(unavailable),
    }

    def emit_status(query, raw_hits, self_hits, removed, retained):
        reason = unavailable.get(query)
        if reason is not None:
            core.require(raw_hits == retained == 0, "unavailable query counts")
            state, available = "MODALITY_INPUT_UNAVAILABLE", False
        elif retained:
            state, available = "CANDIDATES_RETAINED", True
        elif raw_hits == 0:
            state, available = "ZERO_RAW_HITS", True
            totals["zero_raw_hit_queries"] += 1
        else:
            state, available = "NO_ELIGIBLE_REFERENCE_AFTER_FILTER", True
        status_sink(
            {
                "query_node": query,
                "query_role": query_role,
                "modality": modality,
                "status": state,
                "raw_hits": raw_hits,
                "self_hits_removed": self_hits,
                "component_duplicate_or_beyond_top_k": removed,
                "retained_candidates": retained,
                "query_truth_read": False,
                "modality_input_available": available,
                "unavailability_reason": reason,
            }
        )

    for row in raw_rows:
        core.validate_alignment(row, nodes, modality)
        query, reference = row["query_node"], row["reference_node"]
        core.require(query in expected_set and nodes[query]["role"] == query_role, "raw query role")
        core.require(nodes[reference]["role"] == "TRAIN", "reference is not TRAIN")
        core.require(query not in unavailable, "raw hit exists for unavailable query")
        if query != current:
            if current is not None:
                emit_status(current, raw_count, self_removed, filtered, retained_count)
                cursor += 1
            while cursor < len(expected) and expected[cursor] < query:
                emit_status(expected[cursor], 0, 0, 0, 0)
                cursor += 1
            core.require(cursor < len(expected) and expected[cursor] == query, "raw queries not in registered increasing order")
            current = query
            raw_count = self_removed = filtered = retained_count = 0
            seen_components = set()
        raw_count += 1
        totals["raw_hits"] += 1
        if query == reference:
            core.require(query_role == "TRAIN", "non-TRAIN self hit")
            self_removed += 1
            totals["self_hits_removed"] += 1
            continue
        component = nodes[reference]["component_id"]
        if query_role != "TRAIN":
            core.require(nodes[query]["component_id"] != component, "cross-role component leakage")
        if component in seen_components or retained_count >= top_k:
            filtered += 1
            totals["component_duplicate_or_beyond_top_k"] += 1
            continue
        seen_components.add(component)
        retained_count += 1
        totals["retained_candidates"] += 1
        candidate_sink(
            {
                **row,
                "query_role": query_role,
                "reference_role": "TRAIN",
                "reference_component_id": component,
                "raw_rank": raw_count,
                "modality_rank": retained_count,
                "modality": modality,
                "candidate_provenance": "QUERY_DERIVED_RETRIEVAL_PLUS_TRAIN_REFERENCE_METADATA",
            }
        )

    if current is not None:
        emit_status(current, raw_count, self_removed, filtered, retained_count)
        cursor += 1
    while cursor < len(expected):
        emit_status(expected[cursor], 0, 0, 0, 0)
        cursor += 1
    return totals
