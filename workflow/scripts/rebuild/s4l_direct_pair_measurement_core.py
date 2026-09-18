"""Pure deterministic S4L direct-pair measurement planning rules."""
import hashlib


SEED = 20260819
DEFAULT_SHARDS = 64
PAIR_PROJECTION = (
    "query_protein_id", "reference_protein_id", "query_node", "reference_node",
    "query_component_id", "reference_component_id", "query_role", "reference_role",
    "sampling_applied",
)
PROTEIN_PROJECTION = ("protein_id", "node_id", "component_id", "role")
PLAN_COLUMNS = (
    "query_protein_id", "reference_protein_id", "query_node", "reference_node",
    "query_component_id", "reference_component_id", "activity_pair_rows",
    "mmseqs_query_db_key", "mmseqs_reference_db_key", "mmseqs_shard",
    "foldseek_query_db_key", "foldseek_reference_db_key", "foldseek_shard",
    "structure_availability",
)
FORBIDDEN_NAMES = {
    "canonical_ec", "ec_l1", "ec_l2", "ec_l3", "ec_l4", "canonical_rhea",
    "observed_same_ec_l3", "observed_same_ec_l4", "observed_same_exact_rhea",
    "exact_rhea_outcome_evaluable", "deepest_shared_recorded_ec_level",
    "hard_case_flags_available_before_features", "hard_case_flags_for_sampling",
    "sampling_probability", "sample_weight", "sampling_stratum", "pair_origin",
    "retrieval_present",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def shard(modality, query_node, reference_node, shards=DEFAULT_SHARDS, seed=SEED):
    require(modality in {"mmseqs", "foldseek"}, "modality")
    require(type(shards) is int and not isinstance(shards, bool) and shards > 0, "shards")
    payload = f"{seed}|{modality}|{query_node}|{reference_node}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % shards


def unique_mapping(rows, key, value, label):
    mapping = {}
    for row in rows:
        require(set(row) == {key, value}, label + " projection")
        source, target = row[key], row[value]
        require(type(source) is int and type(target) is int and source >= 0 and target >= 0, label + " values")
        require(source not in mapping and target not in mapping.values(), label + " one-to-one mapping")
        mapping[source] = target
    require(mapping, "empty " + label)
    return mapping


def protein_mapping(rows):
    result = {}
    for row in rows:
        require(set(row) == set(PROTEIN_PROJECTION), "protein projection")
        protein = row["protein_id"]
        require(isinstance(protein, str) and protein and protein not in result, "protein identity")
        require(type(row["node_id"]) is int and type(row["component_id"]) is int and row["role"] == "TRAIN", "TRAIN protein mapping")
        result[protein] = dict(row)
    require(result, "empty protein mapping")
    return result


def structure_state(query_node, reference_node, structures):
    query_available, reference_available = query_node in structures, reference_node in structures
    if query_available and reference_available:
        return "BOTH_AVAILABLE"
    if query_available:
        return "REFERENCE_UNAVAILABLE"
    if reference_available:
        return "QUERY_UNAVAILABLE"
    return "BOTH_UNAVAILABLE"


def plan(pair_rows, protein_rows, mmseqs_rows, foldseek_rows, shards=DEFAULT_SHARDS):
    proteins = protein_mapping(protein_rows)
    mmseqs = unique_mapping(mmseqs_rows, "node_id", "db_key", "MMseqs")
    foldseek = unique_mapping(foldseek_rows, "node_id", "db_key", "Foldseek") if foldseek_rows else {}
    pairs = {}
    for row in pair_rows:
        require(set(row) == set(PAIR_PROJECTION), "S4K direct-measurement projection")
        require(not (set(row) & FORBIDDEN_NAMES), "functional or sampling metadata in measurement projection")
        require(row["query_role"] == row["reference_role"] == "TRAIN" and row["sampling_applied"] is True, "sampled TRAIN pair")
        query, reference = row["query_protein_id"], row["reference_protein_id"]
        require(isinstance(query, str) and query and isinstance(reference, str) and reference, "protein pair identity")
        qmeta, rmeta = proteins.get(query), proteins.get(reference)
        require(qmeta is not None and rmeta is not None, "protein pair absent from S4A mapping")
        expected = (qmeta["node_id"], rmeta["node_id"], qmeta["component_id"], rmeta["component_id"])
        observed = (row["query_node"], row["reference_node"], row["query_component_id"], row["reference_component_id"])
        require(observed == expected and observed[0] != observed[1], "pair node/component identity")
        key = (query, reference)
        if key in pairs:
            require(pairs[key]["identity"] == observed, "inconsistent repeated activity pair")
            pairs[key]["count"] += 1
        else:
            pairs[key] = {"identity": observed, "count": 1}
    require(pairs, "empty S4K pair projection")
    result = []
    for (query, reference), metadata in sorted(pairs.items()):
        query_node, reference_node, query_component, reference_component = metadata["identity"]
        require(query_node in mmseqs and reference_node in mmseqs, "pair node absent from MMseqs reference database")
        state = structure_state(query_node, reference_node, foldseek)
        row = {
            "query_protein_id": query, "reference_protein_id": reference,
            "query_node": query_node, "reference_node": reference_node,
            "query_component_id": query_component, "reference_component_id": reference_component,
            "activity_pair_rows": metadata["count"], "mmseqs_query_db_key": mmseqs[query_node],
            "mmseqs_reference_db_key": mmseqs[reference_node],
            "mmseqs_shard": shard("mmseqs", query_node, reference_node, shards),
            "foldseek_query_db_key": foldseek.get(query_node),
            "foldseek_reference_db_key": foldseek.get(reference_node),
            "foldseek_shard": shard("foldseek", query_node, reference_node, shards) if state == "BOTH_AVAILABLE" else None,
            "structure_availability": state,
        }
        require(tuple(row) == PLAN_COLUMNS, "measurement plan schema")
        result.append(row)
    return result


def prefilter_rows(plan_rows, modality, shard_index):
    require(modality in {"mmseqs", "foldseek"}, "prefilter modality")
    require(type(shard_index) is int and shard_index >= 0, "prefilter shard")
    prefix = modality
    rows = set()
    for row in plan_rows:
        if row[prefix + "_shard"] == shard_index:
            query = row[prefix + "_query_db_key"]
            reference = row[prefix + "_reference_db_key"]
            require(type(query) is int and type(reference) is int, "prefilter database keys")
            rows.add((query, reference, 2000, 0))
    return sorted(rows)
