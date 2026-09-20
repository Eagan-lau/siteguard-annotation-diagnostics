"""Create sequence-only data-role tables from validated sequence components."""
import hashlib
import json
import sys
import traceback
from collections import Counter
from pathlib import Path


ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")
ALPHABET = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


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


def normalize(sequence):
    require(isinstance(sequence, str) and sequence, "missing sequence")
    value = sequence.upper()
    require(set(value) <= ALPHABET and len(value) <= 65535, "invalid sequence")
    return value


def write_fasta(path, rows):
    with path.open("x", encoding="ascii", newline="\n") as handle:
        for row in rows:
            handle.write(f">{row['node_id']}\n{row['sequence']}\n")


def validate_upstream(root, contract, contract_sha):
    for name, expected in contract["scripts"].items():
        require(sha(Path(__file__).resolve().parent / name) == expected, "script identity " + name)
    for name, record in contract["inputs"].items():
        path = root / record["path"]
        require(path.is_file() and sha(path) == record["sha256"], "input identity " + name)
    source = root / contract["source"]["path"]
    require(
        source.is_file()
        and source.stat().st_size == contract["source"]["bytes"]
        and sha(source) == contract["source"]["sha256"],
        "source identity",
    )
    audit_path = root / contract["inputs"]["s3_audit"]["path"]
    pass_path = root / contract["inputs"]["s3_pass"]["path"]
    audit = json.loads(audit_path.read_text())
    checkpoint = json.loads(pass_path.read_text())
    require(audit["status"] == "PASS_S3_FINAL_CROSS_ROLE_RESULT_INDEPENDENT_AUDIT", "S3 audit state")
    require(checkpoint["status"] == "PASS_S3_SEQUENCE_ISOLATION", "S3 checkpoint state")
    require(checkpoint["audit_sha256"] == sha(audit_path), "S3 checkpoint audit identity")
    require(checkpoint["contract_sha256"] == audit["contract_sha256"], "S3 contract identity")
    require(audit["qualifying_cross_role_edges"] == 0, "S3 cross-role leakage")
    require(not audit["functional_labels_read"], "S3 label scope")
    return source


def build_records(nodes, memberships, node_split, protein_split, source_rows):
    require(len(nodes) == len(node_split), "node census")
    node_by_id = {row["node_id"]: row for row in nodes}
    split_by_node = {row["node_id"]: row for row in node_split}
    require(len(node_by_id) == len(nodes) and set(node_by_id) == set(split_by_node), "node universe")
    role_by_component = {}
    for row in node_split:
        component, role = row["component_id"], row["role"]
        require(role in ROLES, "node role")
        require(component not in role_by_component or role_by_component[component] == role, "component split across roles")
        role_by_component[component] = role

    sequence_by_protein = {}
    for row in source_rows:
        protein = row["protein_id"]
        require(isinstance(protein, str) and protein and protein not in sequence_by_protein, "source protein identity")
        sequence_by_protein[protein] = normalize(row["sequence"])

    node_of_protein = {}
    members_by_node = Counter()
    for row in memberships:
        protein, node = row["protein_id"], row["node_id"]
        require(protein in sequence_by_protein and protein not in node_of_protein and node in node_by_id, "membership identity")
        node_of_protein[protein] = node
        members_by_node[node] += 1
        sequence_hash = hashlib.sha256(sequence_by_protein[protein].encode("ascii")).hexdigest()
        require(sequence_hash == node_by_id[node]["sequence_sha256"], "protein exact-sequence node")
    require(set(node_of_protein) == set(sequence_by_protein), "source/membership universe")
    require(set(members_by_node) == set(node_by_id), "membership node coverage")

    split_by_protein = {}
    for row in protein_split:
        protein = row["protein_id"]
        require(protein in node_of_protein and protein not in split_by_protein, "protein split identity")
        node = node_of_protein[protein]
        require(row["node_id"] == node and row["role"] == split_by_node[node]["role"], "protein split expansion")
        split_by_protein[protein] = row["role"]
    require(set(split_by_protein) == set(node_of_protein), "protein split universe")

    node_records = []
    sequence_records = []
    for node in sorted(node_by_id):
        record, split = node_by_id[node], split_by_node[node]
        role = split["role"]
        require(role in ROLES and type(split["component_id"]) is int, "node role/component")
        representative = record["representative_protein_id"]
        require(node_of_protein.get(representative) == node, "representative membership")
        sequence = sequence_by_protein[representative]
        require(len(sequence) == record["sequence_length"], "representative length")
        require(members_by_node[node] == record["protein_id_count"], "node member count")
        node_records.append(
            {
                "node_id": node,
                "component_id": split["component_id"],
                "role": role,
                "sequence_sha256": record["sequence_sha256"],
                "sequence_length": record["sequence_length"],
                "representative_protein_id": representative,
                "protein_id_count": record["protein_id_count"],
            }
        )
        sequence_records.append({"node_id": node, "component_id": split["component_id"], "role": role, "sequence": sequence})
    require(set(row["role"] for row in node_records) == set(ROLES), "role coverage")

    protein_records = [
        {
            "protein_id": protein,
            "node_id": node_of_protein[protein],
            "component_id": split_by_node[node_of_protein[protein]]["component_id"],
            "role": split_by_protein[protein],
        }
        for protein in sorted(node_of_protein)
    ]
    return node_records, protein_records, sequence_records


def main():
    contract_path = Path(sys.argv[1]).resolve()
    contract = json.loads(contract_path.read_text())
    require(contract["status"] == "ARMED_S4A_TRUTH_FREE_LEDGER", "unarmed S4A")
    authorization = contract["authorization"]
    require(authorization["truth_free_prepare"], "S4A authorization")
    require(
        not any(authorization[name] for name in ("functional_labels", "retrieval", "activity_expansion", "pair_generation", "training", "evaluation")),
        "S4A scope",
    )
    require(list(sys.version_info[:3]) == contract["python_version"], "Python identity")
    contract_sha = sha(contract_path)
    root = Path(contract["root"])
    require(str(root.resolve()) == contract["physical_root"], "root identity")
    source = validate_upstream(root, contract, contract_sha)
    output = root / contract["output"]
    require(output.parent.is_dir() and not output.exists(), "output identity")
    output.mkdir(exist_ok=False)
    emit(output / "reservation.json", {"status": "ONE_S4A_ATTEMPT_RESERVED", "contract_sha256": contract_sha})
    status, error = "FAIL_CLOSED", None
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        nodes_table = pq.read_table(root / contract["inputs"]["node_manifest"]["path"])
        memberships_table = pq.read_table(root / contract["inputs"]["membership"]["path"])
        node_split_table = pq.read_table(root / contract["inputs"]["node_split"]["path"])
        protein_split_table = pq.read_table(root / contract["inputs"]["protein_split"]["path"])
        require(
            nodes_table.column_names == ["node_id", "sequence_sha256", "sequence_length", "representative_protein_id", "protein_id_count"],
            "node manifest schema",
        )
        require(memberships_table.column_names == ["protein_id", "node_id"], "membership schema")
        require(node_split_table.column_names == ["node_id", "component_id", "role"], "node split schema")
        require(protein_split_table.column_names == ["protein_id", "node_id", "role"], "protein split schema")
        nodes = nodes_table.to_pylist()
        memberships = memberships_table.to_pylist()
        node_split = node_split_table.to_pylist()
        protein_split = protein_split_table.to_pylist()
        source_table = pq.read_table(source, columns=contract["source"]["allowed_columns"])
        require(source_table.column_names == ["protein_id", "sequence"], "sequence-only source projection")
        node_records, protein_records, sequence_records = build_records(
            nodes, memberships, node_split, protein_split, source_table.to_pylist()
        )
        require(
            len(node_records) == contract["expected"]["nodes"]
            and len(protein_records) == contract["expected"]["proteins"],
            "locked census",
        )

        node_schema = pa.schema(
            [
                ("node_id", pa.int32()), ("component_id", pa.int32()), ("role", pa.string()),
                ("sequence_sha256", pa.string()), ("sequence_length", pa.int32()),
                ("representative_protein_id", pa.string()), ("protein_id_count", pa.int32()),
            ]
        )
        protein_schema = pa.schema(
            [("protein_id", pa.string()), ("node_id", pa.int32()), ("component_id", pa.int32()), ("role", pa.string())]
        )
        query_schema = pa.schema([("node_id", pa.int32()), ("component_id", pa.int32()), ("role", pa.string())])
        status_schema = pa.schema(
            [("node_id", pa.int32()), ("component_id", pa.int32()), ("role", pa.string()), ("status", pa.string())]
        )
        pq.write_table(pa.Table.from_pylist(node_records, schema=node_schema), output / "node_role_ledger.parquet", compression="zstd")
        pq.write_table(pa.Table.from_pylist(protein_records, schema=protein_schema), output / "protein_role_ledger.parquet", compression="zstd")
        counts = {}
        for role in ROLES:
            role_nodes = [row for row in node_records if row["role"] == role]
            role_sequences = [row for row in sequence_records if row["role"] == role]
            query_rows = [{"node_id": row["node_id"], "component_id": row["component_id"], "role": role} for row in role_nodes]
            pq.write_table(pa.Table.from_pylist(query_rows, schema=query_schema), output / f"query_nodes_{role}.parquet", compression="zstd")
            write_fasta(output / f"query_{role}.fasta", role_sequences)
            counts[role] = len(role_nodes)
        train_nodes = [row for row in node_records if row["role"] == "TRAIN"]
        train_sequences = [row for row in sequence_records if row["role"] == "TRAIN"]
        train_rows = [{"node_id": row["node_id"], "component_id": row["component_id"], "role": "TRAIN"} for row in train_nodes]
        pq.write_table(pa.Table.from_pylist(train_rows, schema=query_schema), output / "train_reference_nodes.parquet", compression="zstd")
        write_fasta(output / "train_reference.fasta", train_sequences)
        initial = [
            {"node_id": row["node_id"], "component_id": row["component_id"], "role": row["role"], "status": "REGISTERED_PENDING_RETRIEVAL"}
            for row in node_records
        ]
        pq.write_table(pa.Table.from_pylist(initial, schema=status_schema), output / "initial_query_status.parquet", compression="zstd")
        summary = {
            "status": "PASS_S4A_PRODUCER_PENDING_INDEPENDENT_AUDIT",
            "contract_sha256": contract_sha,
            "nodes": len(node_records),
            "proteins": len(protein_records),
            "role_nodes": counts,
            "reference_role": "TRAIN",
            "source_projection": ["protein_id", "sequence"],
            "functional_labels_read": False,
            "retest_labels_read": False,
            "retrieval_started": False,
            "activity_expansion_started": False,
            "pair_generation_started": False,
            "training_started": False,
        }
        emit(output / "S4A_PRODUCER_PASS.json", summary)
        status = summary["status"]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(error["traceback"], file=sys.stderr)
    emit(output / "terminal.json", {"status": status, "error": error, "contract_sha256": contract_sha, "automatic_retry": False})
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

