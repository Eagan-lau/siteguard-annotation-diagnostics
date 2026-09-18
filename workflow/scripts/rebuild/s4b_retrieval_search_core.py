"""Pure S4B raw-search unit plan and alignment normalization rules."""
import math
import re


ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")
MODALITIES = ("mmseqs", "foldseek")
RAW_NAMES = (
    "query_node", "reference_node", "fident", "alnlen", "qstart", "qend", "qlen",
    "tstart", "tend", "tlen", "qcov", "tcov", "evalue", "bits",
)
INTEGER_NAMES = {"query_node", "reference_node", "alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen"}
AF_NAME = re.compile(r"AF-([A-Z0-9]+)-F\d+-model_v\d+", re.IGNORECASE)


def require(value, message):
    if not value:
        raise ValueError(message)


def units():
    return [(role, modality) for role in ROLES for modality in MODALITIES]


def unit(index):
    require(type(index) is int and 0 <= index < len(units()), "unit index")
    role, modality = units()[index]
    return {"unit_index": index, "query_role": role, "modality": modality}


def search_parameters(modality):
    require(modality in MODALITIES, "modality")
    if modality == "mmseqs":
        return {
            "sensitivity": 7.5, "evalue": 1000.0, "max_seqs": 1000,
            "search_type": 1, "top_k_distinct_components": 100,
        }
    return {
        "sensitivity": 9.5, "evalue": 1000.0, "max_seqs": 500,
        "alignment_type": 2, "sort_by_structure_bits": 1,
        "top_k_distinct_components": 100,
    }


def accession(value):
    require(isinstance(value, str) and value, "structure identifier")
    match = AF_NAME.search(value)
    return match.group(1).upper() if match else value.split()[0].upper()


def selected_identifier_maps(selected_rows, query_role):
    require(query_role in ROLES, "query role")
    query, reference = {}, {}
    for row in selected_rows:
        require(set(row) == {"node_id", "component_id", "role", "protein_id", "db_key", "structure_name"}, "selection schema")
        aliases = {row["structure_name"].upper(), accession(row["structure_name"]), row["protein_id"].upper()}
        targets = []
        if row["role"] == query_role:
            targets.append(query)
        if row["role"] == "TRAIN":
            targets.append(reference)
        for target in targets:
            for alias in aliases:
                require(alias not in target or target[alias] == row["node_id"], "ambiguous structure alias")
                target[alias] = row["node_id"]
    return query, reference


def resolve_identifier(value, mapping):
    aliases = {str(value).upper(), accession(str(value))}
    nodes = {mapping[alias] for alias in aliases if alias in mapping}
    require(len(nodes) == 1, "unknown or ambiguous structure identifier")
    return next(iter(nodes))


def parse_raw_fields(fields, modality, query_map=None, reference_map=None, identifiers_are_nodes=False):
    require(modality in MODALITIES and len(fields) == len(RAW_NAMES), "raw alignment fields")
    require(all(isinstance(value, str) and value != "" for value in fields), "empty raw alignment field")
    values = list(fields)
    if modality == "mmseqs" or identifiers_are_nodes:
        require(re.fullmatch(r"\d+", values[0]) and re.fullmatch(r"\d+", values[1]), "MMseqs node identifiers")
        values[0], values[1] = int(values[0]), int(values[1])
    else:
        require(query_map is not None and reference_map is not None, "Foldseek identifier maps")
        values[0] = resolve_identifier(values[0], query_map)
        values[1] = resolve_identifier(values[1], reference_map)
    row = {"query_node": values[0], "reference_node": values[1]}
    for name, value in zip(RAW_NAMES[2:], values[2:]):
        if name in INTEGER_NAMES:
            require(re.fullmatch(r"\d+", value) is not None, "alignment integer " + name)
            row[name] = int(value)
        else:
            parsed = float(value)
            require(math.isfinite(parsed), "alignment finite " + name)
            row[name] = parsed
    require(0 <= row["fident"] <= 1 and 0 <= row["qcov"] <= 1 and 0 <= row["tcov"] <= 1, "alignment ratios")
    require(row["evalue"] >= 0, "alignment evalue")
    # Foldseek 10.941cd33 emits finite negative bit scores for weak alignments
    # retained by the frozen permissive E-value search.  Their ordering remains
    # meaningful; only MMseqs is constrained to non-negative scores here.
    if modality == "mmseqs":
        require(row["bits"] >= 0, "alignment bits")
    require(1 <= row["qstart"] <= row["qend"] <= row["qlen"], "query coordinates")
    require(1 <= row["tstart"] <= row["tend"] <= row["tlen"], "target coordinates")
    require(row["alnlen"] >= max(row["qend"] - row["qstart"] + 1, row["tend"] - row["tstart"] + 1), "alignment length")
    return row


def serialize_row(row):
    require(set(row) == set(RAW_NAMES), "normalized row schema")
    return "\t".join(str(row[name]) for name in RAW_NAMES) + "\n"


def validate_query_order(rows, query_nodes, reference_nodes, query_role):
    require(query_role in ROLES, "query role")
    query_nodes, reference_nodes = set(query_nodes), set(reference_nodes)
    current, completed = None, set()
    count = 0
    for row in rows:
        require(set(row) == set(RAW_NAMES), "normalized row schema")
        query, reference = row["query_node"], row["reference_node"]
        require(query in query_nodes and reference in reference_nodes, "role node universe")
        if query != current:
            if current is not None:
                completed.add(current)
                require(query > current, "query blocks not strictly increasing")
            require(query not in completed, "query block repeated")
            current = query
        count += 1
    return count
