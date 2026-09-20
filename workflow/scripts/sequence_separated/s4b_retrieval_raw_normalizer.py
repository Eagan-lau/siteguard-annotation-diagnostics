"""Normalize native MMseqs2/Foldseek TSV into ordered node-native S4B rows."""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("s4b_retrieval_search_core_normalizer", HERE / "s4b_retrieval_search_core.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)

NODE_COLUMNS = [
    "node_id", "component_id", "role", "sequence_sha256", "sequence_length",
    "representative_protein_id", "protein_id_count",
]
SELECTED_COLUMNS = ["node_id", "component_id", "role", "protein_id", "db_key", "structure_name"]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_fields(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").rstrip("\r").split("\t")
            require(len(fields) == len(core.RAW_NAMES) and all(fields), f"engine raw fields line {line_number}")
            yield fields


def iter_normalized(path, modality):
    for fields in iter_fields(path):
        # Mapping has already replaced engine-specific identifiers by node IDs.
        # Keep the original modality so its score-domain rule is preserved.
        yield core.parse_raw_fields(fields, modality, identifiers_are_nodes=True)


def execute(engine_raw, modality, query_role, node_ledger, output, scratch, sort_binary, selection=None):
    import pyarrow.parquet as pq

    engine_raw, node_ledger, output, scratch, sort_binary = map(
        lambda value: Path(value).resolve(),
        (engine_raw, node_ledger, output, scratch, sort_binary),
    )
    selection = Path(selection).resolve() if selection is not None else None
    require(modality in core.MODALITIES and query_role in core.ROLES, "unit identity")
    require(engine_raw.is_file() and node_ledger.is_file() and sort_binary.is_file(), "normalizer inputs")
    require(scratch.is_dir() and output.parent.is_dir() and not output.exists(), "normalizer output identity")
    node_table = pq.read_table(node_ledger)
    require(node_table.column_names == NODE_COLUMNS, "node ledger schema")
    nodes = node_table.to_pylist()
    node_by_id = {row["node_id"]: row for row in nodes}
    require(len(node_by_id) == len(nodes), "node identity")
    query_nodes = {node for node, row in node_by_id.items() if row["role"] == query_role}
    reference_nodes = {node for node, row in node_by_id.items() if row["role"] == "TRAIN"}
    require(query_nodes and reference_nodes, "role nodes")
    query_map = reference_map = None
    if modality == "foldseek":
        require(selection is not None and selection.is_file(), "Foldseek selection")
        selected_table = pq.read_table(selection)
        require(selected_table.column_names == SELECTED_COLUMNS, "selection schema")
        query_map, reference_map = core.selected_identifier_maps(selected_table.to_pylist(), query_role)
    else:
        require(selection is None, "MMseqs selection forbidden")

    mapped = scratch / ("s4b_mapped_" + query_role + "_" + modality + ".tsv")
    require(not mapped.exists(), "mapped scratch exists")
    mapped_rows = 0
    with mapped.open("x", encoding="ascii", newline="\n") as handle:
        for fields in iter_fields(engine_raw):
            row = core.parse_raw_fields(fields, modality, query_map, reference_map)
            require(row["query_node"] in query_nodes, "query role universe")
            require(row["reference_node"] in reference_nodes, "reference TRAIN universe")
            if query_role != "TRAIN":
                require(node_by_id[row["query_node"]]["component_id"] != node_by_id[row["reference_node"]]["component_id"], "cross-role component leakage")
            handle.write(core.serialize_row(row))
            mapped_rows += 1

    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    environment["TMPDIR"] = str(scratch)
    command = [
        str(sort_binary), "--stable", "--numeric-sort", "--field-separator=\t",
        "--key=1,1", "--output=" + str(output), str(mapped),
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=environment)
    require(result.returncode == 0 and not result.stderr, "stable query sort")
    normalized_rows = core.validate_query_order(iter_normalized(output, modality), query_nodes, reference_nodes, query_role)
    require(normalized_rows == mapped_rows, "normalization row conservation")
    return {
        "status": "PASS_S4B_RAW_HIT_NORMALIZATION",
        "query_role": query_role,
        "modality": modality,
        "rows": normalized_rows,
        "engine_raw_sha256": sha(engine_raw),
        "normalized_sha256": sha(output),
        "sort_binary": str(sort_binary),
        "sort_binary_sha256": sha(sort_binary),
        "stable_within_query_native_order": True,
        "functional_labels_read": False,
        "query_truth_read": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-raw", type=Path, required=True)
    parser.add_argument("--modality", choices=core.MODALITIES, required=True)
    parser.add_argument("--query-role", choices=core.ROLES, required=True)
    parser.add_argument("--node-ledger", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--sort-binary", type=Path, default=Path("/usr/bin/sort"))
    args = parser.parse_args()
    print(json.dumps(execute(args.engine_raw, args.modality, args.query_role, args.node_ledger, args.output, args.scratch, args.sort_binary, args.selection), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
