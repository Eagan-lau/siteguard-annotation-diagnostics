"""Prepare node-native Foldseek keys and availability ledgers for S4B."""
import argparse
import csv
import hashlib
import importlib.util
import json
import traceback
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("s4b_retrieval_core_prepare", HERE / "s4b_retrieval_core.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)

PROTEIN_COLUMNS = ["protein_id", "node_id", "component_id", "role"]
NODE_COLUMNS = [
    "node_id", "component_id", "role", "sequence_sha256", "sequence_length",
    "representative_protein_id", "protein_id_count",
]
AFDB_COLUMNS = [
    "protein_id", "filename", "model_version", "compressed_size",
    "has_structure", "structure_source",
]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_json(path):
    require(path.is_file() and not path.is_symlink(), "required JSON " + path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_record(root, record, label):
    path = (root / record["path"]).resolve()
    require(path.is_file() and not path.is_symlink() and sha(path) == record["sha256"], "input identity " + label)
    return path


def read_lookup(path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, fields in enumerate(reader, 1):
            require(len(fields) == 3 and all(fields), f"Foldseek lookup fields line {line_number}")
            try:
                rows.append({"db_key": int(fields[0]), "structure_name": fields[1], "file_number": int(fields[2])})
            except ValueError as exc:
                raise RuntimeError(f"Foldseek lookup integer line {line_number}") from exc
    require(rows, "empty Foldseek lookup")
    return rows


def write_keys(path, values):
    with path.open("x", encoding="ascii", newline="\n") as handle:
        for value in values:
            handle.write(str(value) + "\n")


def execute(protein_ledger, node_ledger, afdb, foldseek_lookup, output):
    import pyarrow as pa
    import pyarrow.parquet as pq

    protein_ledger, node_ledger, afdb, foldseek_lookup = map(
        lambda value: Path(value).resolve(),
        (protein_ledger, node_ledger, afdb, foldseek_lookup),
    )
    output = Path(output).resolve()
    require(all(path.is_file() for path in (protein_ledger, node_ledger, afdb, foldseek_lookup)), "preparer inputs")
    require(output.parent.is_dir() and not output.exists(), "exclusive output")
    output.mkdir(exist_ok=False)
    inputs = {
        name: {"path": str(path), "sha256": sha(path)}
        for name, path in {
            "protein_ledger": protein_ledger,
            "node_ledger": node_ledger,
            "afdb": afdb,
            "foldseek_lookup": foldseek_lookup,
        }.items()
    }
    emit(output / "reservation.json", {"status": "ONE_S4B_RETRIEVAL_PREPARE_ATTEMPT_RESERVED", "inputs": inputs, "automatic_retry": False})
    state, error = "FAIL_CLOSED", None
    try:
        protein_table = pq.read_table(protein_ledger)
        node_table = pq.read_table(node_ledger)
        afdb_table = pq.read_table(afdb, columns=AFDB_COLUMNS)
        require(protein_table.column_names == PROTEIN_COLUMNS, "protein ledger schema")
        require(node_table.column_names == NODE_COLUMNS, "node ledger schema")
        require(afdb_table.column_names == AFDB_COLUMNS, "AFDB projection")
        proteins, nodes = protein_table.to_pylist(), node_table.to_pylist()
        require(len(proteins) == sum(row["protein_id_count"] for row in nodes), "protein/node census")
        node_by_id = {row["node_id"]: row for row in nodes}
        require(len(node_by_id) == len(nodes), "node identity")
        selected, unavailable = core.select_node_structures(proteins, afdb_table.to_pylist(), read_lookup(foldseek_lookup))
        require({row["node_id"] for row in selected} | {row["query_node"] for row in unavailable} == set(node_by_id), "node availability universe")
        for row in selected:
            node = node_by_id[row["node_id"]]
            require((row["component_id"], row["role"]) == (node["component_id"], node["role"]), "selected node metadata")
        for row in unavailable:
            node = node_by_id[row["query_node"]]
            require((row["component_id"], row["role"]) == (node["component_id"], node["role"]), "unavailable node metadata")

        selected_schema = pa.schema([
            ("node_id", pa.int32()), ("component_id", pa.int32()), ("role", pa.string()),
            ("protein_id", pa.string()), ("db_key", pa.int64()), ("structure_name", pa.string()),
        ])
        unavailable_schema = pa.schema([
            ("query_node", pa.int32()), ("component_id", pa.int32()),
            ("role", pa.string()), ("reason", pa.string()),
        ])
        writer_unavailable_schema = pa.schema([("query_node", pa.int32()), ("reason", pa.string())])
        pq.write_table(pa.Table.from_pylist(selected, schema=selected_schema), output / "foldseek_node_selection.parquet", compression="zstd")
        pq.write_table(pa.Table.from_pylist(unavailable, schema=unavailable_schema), output / "foldseek_unavailability.parquet", compression="zstd")
        reference, queries = core.key_plan(selected)
        write_keys(output / "foldseek_train_reference.keys", reference)
        for role in core.ROLES:
            write_keys(output / f"foldseek_query_{role}.keys", queries[role])
            role_unavailable = [
                {"query_node": row["query_node"], "reason": row["reason"]}
                for row in unavailable if row["role"] == role
            ]
            pq.write_table(
                pa.Table.from_pylist(role_unavailable, schema=writer_unavailable_schema),
                output / f"foldseek_unavailable_{role}.parquet",
                compression="zstd",
            )
        available_counts = Counter(row["role"] for row in selected)
        unavailable_counts = Counter(row["role"] for row in unavailable)
        role_nodes = Counter(row["role"] for row in nodes)
        summary = {
            "status": "PASS_S4B_RETRIEVAL_PREPARE_PENDING_INDEPENDENT_AUDIT",
            "inputs": inputs,
            "nodes": len(nodes),
            "proteins": len(proteins),
            "structure_available_nodes": len(selected),
            "structure_unavailable_nodes": len(unavailable),
            "role_nodes": {role: role_nodes[role] for role in core.ROLES},
            "role_structure_available_nodes": {role: available_counts[role] for role in core.ROLES},
            "role_structure_unavailable_nodes": {role: unavailable_counts[role] for role in core.ROLES},
            "train_reference_structure_nodes": len(reference),
            "structure_representative_rule": "minimum protein_id then minimum db_key per exact-sequence node",
            "functional_labels_read": False,
            "query_truth_read": False,
            "retrieval_started": False,
            "training_started": False,
        }
        emit(output / "producer_summary.json", summary)
        state = summary["status"]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    emit(output / "terminal.json", {"status": state, "error": error, "automatic_retry": False})
    if error:
        raise RuntimeError(error["message"])
    return summary


def execute_contract(contract_path):
    """Run the preparer under an immutable, truth-free formal contract."""
    import sys

    contract_path = Path(contract_path).resolve()
    contract = read_json(contract_path)
    require(contract["status"] == "ARMED_S4B_NODE_NATIVE_RETRIEVAL_PREPARE", "contract status")
    require(contract["attempt"] == 1 and contract["attempt_limit"] == 3 and not contract["automatic_retry"], "attempt policy")
    require(list(sys.version_info[:3]) == contract["python_version"], "Python identity")
    root = Path(contract["root"])
    require(str(root.resolve()) == contract["physical_root"], "root identity")
    authorization = contract["authorization"]
    require(authorization["prepare"] and authorization["independent_prepare_audit"], "prepare authorization")
    require(not any(authorization[name] for name in (
        "functional_labels", "query_truth", "full_search", "activity_expansion",
        "pair_generation", "training", "evaluation", "phase99", "protected_cohort",
    )), "prepare scope")
    for name, expected in contract["prepare_scripts"].items():
        path = HERE / name
        require(path.is_file() and sha(path) == expected, "script identity " + name)
    inputs = contract["inputs"]
    protein_ledger = validate_record(root, inputs["protein_ledger"], "protein ledger")
    node_ledger = validate_record(root, inputs["node_ledger"], "node ledger")
    afdb = validate_record(root, inputs["afdb"], "AFDB ledger")
    foldseek_lookup = validate_record(root, inputs["foldseek_lookup"], "Foldseek lookup")
    s4a_audit_path = validate_record(root, inputs["s4a_audit"], "S4A audit")
    s4a_pass_path = validate_record(root, inputs["s4a_pass"], "S4A checkpoint")
    s4a_audit, s4a_pass = read_json(s4a_audit_path), read_json(s4a_pass_path)
    require(s4a_audit["status"] == "PASS_S4A_TRUTH_FREE_LEDGER_INDEPENDENT_AUDIT", "S4A audit status")
    require(s4a_pass["status"] == "PASS_S4A_TRUTH_FREE_LEDGER", "S4A checkpoint status")
    require(s4a_pass["audit_sha256"] == sha(s4a_audit_path), "S4A audit binding")
    require(not s4a_audit["functional_labels_read"] and not s4a_audit["retest_labels_read"], "S4A scope")
    output = (root / contract["prepare_output"]).resolve()
    summary = execute(protein_ledger, node_ledger, afdb, foldseek_lookup, output)
    receipt = {
        "status": "PASS_S4B_RETRIEVAL_PREPARE_CONTRACT_BINDING",
        "contract_sha256": sha(contract_path),
        "producer_summary_sha256": sha(output / "producer_summary.json"),
        "terminal_sha256": sha(output / "terminal.json"),
        "functional_labels_read": False,
        "query_truth_read": False,
    }
    emit(output / "prepare_contract_receipt.json", receipt)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--protein-ledger", type=Path)
    parser.add_argument("--node-ledger", type=Path)
    parser.add_argument("--afdb", type=Path)
    parser.add_argument("--foldseek-lookup", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.contract is not None:
        require(not any((args.protein_ledger, args.node_ledger, args.afdb, args.foldseek_lookup, args.output)), "contract mode arguments")
        execute_contract(args.contract)
    else:
        require(all((args.protein_ledger, args.node_ledger, args.afdb, args.foldseek_lookup, args.output)), "direct mode arguments")
        execute(args.protein_ledger, args.node_ledger, args.afdb, args.foldseek_lookup, args.output)


if __name__ == "__main__":
    main()
