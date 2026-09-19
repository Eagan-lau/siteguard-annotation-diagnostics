"""Truth-free cross-modality union of audited S4B candidate rows."""
import math
from collections import defaultdict


ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")
RAW_METRICS = (
    "fident", "alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen",
    "qcov", "tcov", "evalue", "bits",
)
CANDIDATE_FIELDS = {
    "query_node", "reference_node", *RAW_METRICS, "query_role", "reference_role",
    "reference_component_id", "raw_rank", "modality_rank", "modality",
    "candidate_provenance",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_modality(rows, modality, query_role):
    require(modality in ("mmseqs", "foldseek") and query_role in ROLES, "unit identity")
    by_query = defaultdict(list)
    seen = set()
    for row in rows:
        require(set(row) == CANDIDATE_FIELDS, "candidate schema")
        require(row["modality"] == modality and row["query_role"] == query_role, "candidate modality/role")
        require(row["reference_role"] == "TRAIN", "reference role")
        require(row["candidate_provenance"] == "QUERY_DERIVED_RETRIEVAL_PLUS_TRAIN_REFERENCE_METADATA", "candidate provenance")
        key = row["query_node"], row["reference_node"]
        require(key not in seen and row["query_node"] != row["reference_node"], "candidate identity")
        seen.add(key)
        require(type(row["reference_component_id"]) is int and row["reference_component_id"] >= 0, "reference component")
        require(type(row["raw_rank"]) is int and type(row["modality_rank"]) is int, "candidate ranks")
        require(row["raw_rank"] >= row["modality_rank"] >= 1, "candidate rank order")
        require(all(type(row[name]) in (int, float) and not isinstance(row[name], bool) and math.isfinite(row[name]) for name in RAW_METRICS), "finite retrieval metrics")
        by_query[row["query_node"]].append(row)
    for query, values in by_query.items():
        ordered = sorted(values, key=lambda row: row["modality_rank"])
        require([row["modality_rank"] for row in ordered] == list(range(1, len(ordered) + 1)), "contiguous modality ranks")
        require(all(left["raw_rank"] < right["raw_rank"] for left, right in zip(ordered, ordered[1:])), "raw rank order")
        components = [row["reference_component_id"] for row in ordered]
        require(len(components) == len(set(components)), "component deduplication")
    return seen


def candidate_union(mmseqs_rows, foldseek_rows, query_role, rrf_constant=60):
    require(type(rrf_constant) is int and not isinstance(rrf_constant, bool) and rrf_constant > 0, "RRF constant")
    validate_modality(mmseqs_rows, "mmseqs", query_role)
    validate_modality(foldseek_rows, "foldseek", query_role)
    combined = {}

    def add(row, modality):
        key = row["query_node"], row["reference_node"]
        if key not in combined:
            combined[key] = {
                "query_node": row["query_node"],
                "reference_node": row["reference_node"],
                "query_role": query_role,
                "reference_role": "TRAIN",
                "reference_component_id": row["reference_component_id"],
                "mmseqs_rank": None,
                "foldseek_rank": None,
                "mmseqs_raw_rank": None,
                "foldseek_raw_rank": None,
                **{f"mmseqs_{name}": None for name in RAW_METRICS},
                **{f"foldseek_{name}": None for name in RAW_METRICS},
            }
        target = combined[key]
        require(
            target["query_role"] == row["query_role"]
            and target["reference_role"] == row["reference_role"]
            and target["reference_component_id"] == row["reference_component_id"],
            "cross-modality identity disagreement",
        )
        target[f"{modality}_rank"] = row["modality_rank"]
        target[f"{modality}_raw_rank"] = row["raw_rank"]
        for name in RAW_METRICS:
            target[f"{modality}_{name}"] = row[name]

    for row in mmseqs_rows:
        add(row, "mmseqs")
    for row in foldseek_rows:
        add(row, "foldseek")
    output = []
    for key in sorted(combined):
        row = combined[key]
        present = [name for name in ("mmseqs", "foldseek") if row[f"{name}_rank"] is not None]
        require(present, "empty modality union")
        row["source_class"] = "both" if len(present) == 2 else present[0] + "_only"
        row["rrf60_score"] = sum(1.0 / (rrf_constant + row[f"{name}_rank"]) for name in present)
        row["rrf_constant"] = rrf_constant
        row["candidate_union_provenance"] = "TRUTH_FREE_AUDITED_MODALITY_UNION_NO_POLICY_SELECTION"
        row["query_truth_read"] = False
        output.append(row)
    return output

