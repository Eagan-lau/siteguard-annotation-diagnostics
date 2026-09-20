"""Versioned S4B.1 layer separating absent modality inputs from zero hits."""
import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("s4b_candidate_core_v1", HERE / "s4b_candidate_core.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


UNAVAILABILITY_REASONS = {"NO_REGISTERED_STRUCTURE"}


def retain_candidates(raw_rows, node_rows, query_role, modality, top_k, unavailable_queries=None):
    unavailable_queries = {} if unavailable_queries is None else dict(unavailable_queries)
    nodes = core.validate_nodes(node_rows)
    expected = {node for node, row in nodes.items() if row["role"] == query_role}
    core.require(modality in core.MODALITIES, "modality")
    core.require(set(unavailable_queries) <= expected, "unavailable query universe")
    core.require(
        all(reason in UNAVAILABILITY_REASONS for reason in unavailable_queries.values()),
        "unavailability reason",
    )
    core.require(modality == "foldseek" or not unavailable_queries, "sequence modality cannot lack sequence input")
    core.require(
        all(row.get("query_node") not in unavailable_queries for row in raw_rows),
        "raw hit exists for unavailable query",
    )
    candidates, status_rows = core.retain_candidates(raw_rows, node_rows, query_role, modality, top_k)
    enriched = []
    for row in status_rows:
        value = dict(row)
        reason = unavailable_queries.get(row["query_node"])
        if reason is not None:
            core.require(row["raw_hits"] == row["retained_candidates"] == 0, "unavailable query counts")
            value["status"] = "MODALITY_INPUT_UNAVAILABLE"
            value["modality_input_available"] = False
            value["unavailability_reason"] = reason
        else:
            value["modality_input_available"] = True
            value["unavailability_reason"] = None
        enriched.append(value)
    return candidates, enriched

