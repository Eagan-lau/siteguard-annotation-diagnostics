"""Pure node-native structure planning and Foldseek identifier remapping."""
import re


ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")
PROTEIN_FIELDS = {"protein_id", "node_id", "component_id", "role"}
AFDB_FIELDS = {
    "protein_id", "filename", "model_version", "compressed_size",
    "has_structure", "structure_source",
}
LOOKUP_FIELDS = {"db_key", "structure_name", "file_number"}
SELECTED_FIELDS = {
    "node_id", "component_id", "role", "protein_id", "db_key",
    "structure_name",
}
AF_NAME = re.compile(r"AF-([A-Z0-9]+)-F\d+-model_v\d+", re.IGNORECASE)


def require(value, message):
    if not value:
        raise ValueError(message)


def accession(value):
    require(isinstance(value, str) and value, "structure identifier")
    match = AF_NAME.search(value)
    return match.group(1).upper() if match else value.split()[0].upper()


def select_node_structures(protein_rows, afdb_rows, lookup_rows):
    """Select one deterministic structure-bearing protein for every node."""
    require(isinstance(protein_rows, list) and protein_rows, "protein ledger")
    require(isinstance(afdb_rows, list) and afdb_rows, "AFDB ledger")
    require(isinstance(lookup_rows, list) and lookup_rows, "Foldseek lookup")

    proteins, node_meta = {}, {}
    for row in protein_rows:
        require(set(row) == PROTEIN_FIELDS, "protein ledger schema")
        protein, node, component, role = (
            row["protein_id"], row["node_id"], row["component_id"], row["role"]
        )
        require(isinstance(protein, str) and protein and protein not in proteins, "protein identity")
        require(type(node) is int and node >= 0 and type(component) is int and component >= 0, "node/component")
        require(role in ROLES, "role")
        meta = (component, role)
        require(node not in node_meta or node_meta[node] == meta, "node metadata conflict")
        node_meta[node] = meta
        proteins[protein.upper()] = row
    require(set(row["role"] for row in protein_rows) == set(ROLES), "role coverage")

    afdb = {}
    for row in afdb_rows:
        require(set(row) == AFDB_FIELDS, "AFDB schema")
        protein = row["protein_id"]
        require(isinstance(protein, str) and protein and protein.upper() not in afdb, "AFDB protein identity")
        require(type(row["has_structure"]) is bool, "AFDB availability type")
        if row["has_structure"]:
            require(isinstance(row["filename"], str) and row["filename"], "available AFDB filename")
            require(row["structure_source"] == "AlphaFoldDB_SwissProt_v6", "AFDB source")
        else:
            require(row["filename"] is None, "unavailable AFDB filename")
            require(row["structure_source"] == "UNAVAILABLE_IN_FROZEN_BULK_INDEX", "AFDB unavailable source")
        afdb[protein.upper()] = row
    require(set(afdb) == set(proteins), "AFDB/protein universe")

    lookup_by_protein = {}
    seen_keys = set()
    for row in lookup_rows:
        require(set(row) == LOOKUP_FIELDS, "Foldseek lookup schema")
        key, name = row["db_key"], row["structure_name"]
        require(type(key) is int and key >= 0 and key not in seen_keys, "Foldseek key identity")
        require(type(row["file_number"]) is int and row["file_number"] >= 0, "Foldseek file number")
        seen_keys.add(key)
        protein = accession(name)
        lookup_by_protein.setdefault(protein, []).append(row)

    candidates = {}
    for protein in sorted(proteins):
        if not afdb[protein]["has_structure"]:
            continue
        require(protein in lookup_by_protein, "registered structure absent from Foldseek lookup")
        lookup = min(lookup_by_protein[protein], key=lambda row: (row["db_key"], row["structure_name"]))
        ledger = proteins[protein]
        value = {
            "node_id": ledger["node_id"],
            "component_id": ledger["component_id"],
            "role": ledger["role"],
            "protein_id": ledger["protein_id"],
            "db_key": lookup["db_key"],
            "structure_name": lookup["structure_name"],
        }
        candidates.setdefault(ledger["node_id"], []).append(value)

    selected, unavailable = [], []
    for node in sorted(node_meta):
        values = candidates.get(node, [])
        if values:
            selected.append(min(values, key=lambda row: (row["protein_id"], row["db_key"], row["structure_name"])))
        else:
            component, role = node_meta[node]
            unavailable.append(
                {"query_node": node, "component_id": component, "role": role, "reason": "NO_REGISTERED_STRUCTURE"}
            )
    require(len(selected) + len(unavailable) == len(node_meta), "node availability conservation")
    require(all(set(row) == SELECTED_FIELDS for row in selected), "selected schema")
    return selected, unavailable


def key_plan(selected_rows):
    """Return deterministic TRAIN reference and five role-specific query keys."""
    require(all(set(row) == SELECTED_FIELDS for row in selected_rows), "selected schema")
    by_node = {}
    for row in selected_rows:
        require(row["node_id"] not in by_node and row["role"] in ROLES, "selected node identity")
        by_node[row["node_id"]] = row
    queries = {
        role: [row["db_key"] for row in sorted(selected_rows, key=lambda value: value["node_id"]) if row["role"] == role]
        for role in ROLES
    }
    reference = list(queries["TRAIN"])
    require(len(reference) == len(set(reference)), "reference key uniqueness")
    return reference, queries


def identifier_maps(selected_rows, query_role):
    require(query_role in ROLES, "query role")
    query, reference = {}, {}
    for row in selected_rows:
        require(set(row) == SELECTED_FIELDS, "selected schema")
        aliases = {row["structure_name"].upper(), accession(row["structure_name"]), row["protein_id"].upper()}
        target = query if row["role"] == query_role else reference if row["role"] == "TRAIN" else None
        if target is not None:
            for alias in aliases:
                require(alias not in target or target[alias] == row["node_id"], "ambiguous structure alias")
                target[alias] = row["node_id"]
    return query, reference


def remap_foldseek_identifiers(query_identifier, reference_identifier, query_map, reference_map):
    q_aliases = (str(query_identifier).upper(), accession(str(query_identifier)))
    r_aliases = (str(reference_identifier).upper(), accession(str(reference_identifier)))
    q_values = {query_map[value] for value in q_aliases if value in query_map}
    r_values = {reference_map[value] for value in r_aliases if value in reference_map}
    require(len(q_values) == len(r_values) == 1, "unknown or ambiguous Foldseek identifier")
    return next(iter(q_values)), next(iter(r_values))

