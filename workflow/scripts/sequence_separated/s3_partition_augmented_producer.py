"""S3 partition augmented producer."""
import hashlib
import importlib.util
import json
import sys
import traceback
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assignment_core = load("s3_partition_core")
augmentation_core = load("s3_partition_augmented_core")


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


def component_inputs(nodes, memberships, augmented_rows):
    node_proteins = {row["node_id"]: row["protein_id_count"] for row in nodes}
    require(len(node_proteins) == len(nodes) and set(node_proteins) == set(range(len(nodes))), "node manifest")
    member_counts = Counter(row["node_id"] for row in memberships)
    require(set(member_counts) == set(node_proteins), "membership node coverage")
    require(all(member_counts[node] == count for node, count in node_proteins.items()), "membership counts")
    augmented_by_node = {row["node_id"]: row["augmented_component_id"] for row in augmented_rows}
    require(len(augmented_by_node) == len(augmented_rows) and set(augmented_by_node) == set(node_proteins), "augmented node universe")
    node_counts = Counter(augmented_by_node.values())
    protein_counts = Counter()
    for node, component in augmented_by_node.items():
        protein_counts[component] += node_proteins[node]
    components = [
        {"component_id": component, "node_count": node_counts[component], "protein_count": protein_counts[component]}
        for component in sorted(node_counts)
    ]
    return components, augmented_by_node


def main():
    contract_path = Path(sys.argv[1]).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(contract["status"] == "ARMED_S3_AUGMENTED_PARTITION_ATTEMPT03", "contract status")
    require(contract["attempt"] == contract["attempt_limit"] == 3 and contract["automatic_retry"] is False, "attempt policy")
    require(contract["partition_target_unit"] == "exact_sequence_node", "partition target unit")
    require(contract["authorization"]["partition"] and contract["authorization"]["audited_edge_closure"], "partition authorization")
    require(not any(contract["authorization"][k] for k in ("functional_labels", "pair_generation", "training", "phase99", "protected_cohort")), "scope")
    require(list(sys.version_info[:3]) == contract["python_version"], "Python identity")
    contract_sha = sha(contract_path)
    root = Path(contract["root"])
    require(str(root.resolve()) == contract["physical_root"], "root identity")
    for name, expected in contract["scripts"].items():
        require(sha(HERE / name) == expected, "script identity " + name)
    for item in contract["evidence"]:
        path = Path(item["path"])
        require(path.is_file() and not path.is_symlink() and sha(path) == item["sha256"], "evidence identity")
    output = root / contract["output"]
    require(output.parent.is_dir() and not output.exists(), "output identity")
    output.mkdir(exist_ok=False)
    emit(output / "reservation.json", {
        "status": "ONE_S3_AUGMENTED_PARTITION_ATTEMPT03_RESERVED",
        "contract_sha256": contract_sha,
    })
    status = "FAIL_CLOSED"
    error = None
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        nodes = pq.read_table(root / contract["inputs"]["node_manifest"], columns=["node_id", "protein_id_count"]).to_pylist()
        memberships = pq.read_table(root / contract["inputs"]["membership"], columns=["protein_id", "node_id"]).to_pylist()
        original_components = pq.read_table(root / contract["inputs"]["components"], columns=["node_id", "component_id", "component_size"]).to_pylist()
        edges = pq.read_table(root / contract["inputs"]["audited_cross_role_edges"]).to_pylist()
        require(len(nodes) == contract["expected"]["nodes"] and len(memberships) == contract["expected"]["proteins"], "locked census")
        require(len(edges) == contract["expected"]["audited_cross_role_edges"] > 0, "audited edge census")
        augmented_rows, closure = augmentation_core.augment_components(original_components, edges)
        require(closure["original_components"] == contract["expected"]["original_components"], "original component census")
        require(closure["augmented_components"] == contract["expected"]["augmented_components"], "augmented component census")
        component_rows, component_by_node = component_inputs(nodes, memberships, augmented_rows)
        require(len(component_rows) == closure["augmented_components"], "assignment component census")
        assignments, role_summary = assignment_core.assign_components(
            component_rows,
            seed=contract["seed"],
            minimum_components=contract["feasibility"]["minimum_components"],
        )
        feasibility_rule = contract["feasibility"]
        largest_component_nodes = max(row["node_count"] for row in component_rows)
        deviations = {
            role: abs(role_summary[role]["observed_fraction"] - role_summary[role]["target_fraction"])
            for role in assignment_core.ROLES
        }
        feasibility = {
            "status": "PASS_S3_AUGMENTED_PARTITION_FEASIBILITY"
            if largest_component_nodes / len(nodes) <= feasibility_rule["largest_component_fraction_max_closed"]
            and all(value <= feasibility_rule["role_fraction_absolute_deviation_max_closed"] for value in deviations.values())
            and all(role_summary[role]["components"] >= minimum for role, minimum in feasibility_rule["minimum_components"].items())
            else "FAIL_S3_AUGMENTED_PARTITION_FEASIBILITY",
            "partition_target_unit": "exact_sequence_node",
            "largest_component_nodes": largest_component_nodes,
            "largest_component_fraction": largest_component_nodes / len(nodes),
            "role_absolute_deviation": deviations,
            "role_components": {role: role_summary[role]["components"] for role in assignment_core.ROLES},
            "registered_rule": feasibility_rule,
        }
        emit(output / "partition_feasibility.json", feasibility)
        require(feasibility["status"] == "PASS_S3_AUGMENTED_PARTITION_FEASIBILITY", "registered partition feasibility")
        node_rows = [{"node_id": node, "component_id": component_by_node[node]} for node in sorted(component_by_node)]
        node_split, protein_split = assignment_core.expand_assignments(assignments, node_rows, memberships)

        augmented_schema = pa.schema([
            ("node_id", pa.int32()), ("original_component_id", pa.int32()),
            ("augmented_component_id", pa.int32()), ("augmented_component_size", pa.int32()),
        ])
        component_schema = pa.schema([
            ("component_id", pa.int32()), ("role", pa.string()), ("node_count", pa.int32()),
            ("protein_count", pa.int32()), ("order_hash", pa.string()),
        ])
        node_schema = pa.schema([("node_id", pa.int32()), ("component_id", pa.int32()), ("role", pa.string())])
        protein_schema = pa.schema([("protein_id", pa.string()), ("node_id", pa.int32()), ("role", pa.string())])
        pq.write_table(pa.Table.from_pylist(augmented_rows, schema=augmented_schema), output / "augmented_components.parquet", compression="zstd")
        pq.write_table(pa.Table.from_pylist(assignments, schema=component_schema), output / "component_assignment.parquet", compression="zstd")
        pq.write_table(pa.Table.from_pylist(node_split, schema=node_schema), output / "node_split.parquet", compression="zstd")
        pq.write_table(pa.Table.from_pylist(protein_split, schema=protein_schema), output / "protein_split.parquet", compression="zstd")
        result = {
            "status": "PASS_S3_AUGMENTED_PARTITION_PRODUCER_AUDIT_PENDING",
            "contract_sha256": contract_sha,
            "attempt": 3,
            "attempt_limit_reached": True,
            "seed": contract["seed"],
            "partition_target_unit": "exact_sequence_node",
            "nodes": len(node_split),
            "proteins": len(protein_split),
            "components": len(assignments),
            "audited_cross_role_edges_consumed": len(edges),
            "closure_summary": closure,
            "role_summary": role_summary,
            "feasibility": feasibility,
            "functional_labels_read": False,
            "pair_generation_started": False,
            "training_started": False,
            "direct_cross_role_search_completed_for_attempt03": False,
        }
        emit(output / "partition_summary.json", result)
        emit(output / "S3_AUGMENTED_PARTITION_PRODUCER_PASS.json", result)
        status = result["status"]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(error["traceback"], file=sys.stderr)
    emit(output / "terminal.json", {
        "status": status, "error": error, "contract_sha256": contract_sha,
        "automatic_retry": False,
    })
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
