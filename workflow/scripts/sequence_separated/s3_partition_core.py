"""Pure, label-free assignment of complete sequence components to fixed roles."""
import hashlib
import json


SEED = 20260819
ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")
FRACTIONS = {"TRAIN": 0.70, "DEV": 0.10, "CAL_FIT": 0.05, "CAL_RULE": 0.05, "RETEST": 0.10}
MINIMUM_COMPONENTS = {"DEV": 50, "CAL_FIT": 50, "CAL_RULE": 50, "RETEST": 50}


def require(value, message):
    if not value:
        raise ValueError(message)


def stable_hash(seed, component_id):
    payload = json.dumps([seed, component_id], separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_components(components):
    require(isinstance(components, list) and len(components) >= len(ROLES), "at least five components required")
    ids = []
    for row in components:
        require(set(row) == {"component_id", "node_count", "protein_count"}, "component schema")
        require(type(row["component_id"]) is int and row["component_id"] >= 0, "component ID")
        require(type(row["node_count"]) is int and row["node_count"] > 0, "node count")
        require(type(row["protein_count"]) is int and row["protein_count"] >= row["node_count"], "protein count")
        ids.append(row["component_id"])
    require(len(ids) == len(set(ids)), "duplicate component ID")
    return True


def objective(counts, targets):
    return sum(((counts[role] - targets[role]) / targets[role]) ** 2 for role in ROLES)


def assign_components(components, seed=SEED, minimum_components=None):
    validate_components(components)
    require(seed == SEED, "production seed is fixed")
    minimum_components = dict(MINIMUM_COMPONENTS if minimum_components is None else minimum_components)
    require(set(minimum_components) == set(MINIMUM_COMPONENTS), "minimum-component role contract")
    require(all(type(value) is int and not isinstance(value, bool) and value >= 1 for value in minimum_components.values()), "minimum-component counts")
    require(sum(minimum_components.values()) + 1 <= len(components), "insufficient components for constrained roles")
    total_nodes = sum(row["node_count"] for row in components)
    total_proteins = sum(row["protein_count"] for row in components)
    targets = {role: total_nodes * FRACTIONS[role] for role in ROLES}
    node_counts = {role: 0 for role in ROLES}
    protein_counts = {role: 0 for role in ROLES}
    component_counts = {role: 0 for role in ROLES}
    assignments = []

    def place(row, role):
        node_counts[role] += row["node_count"]
        protein_counts[role] += row["protein_count"]
        component_counts[role] += 1
        assignments.append(
            {
                "component_id": row["component_id"], "role": role,
                "node_count": row["node_count"], "protein_count": row["protein_count"],
                "order_hash": stable_hash(seed, row["component_id"]),
            }
        )

    # Reserve the smallest components round-robin so every non-TRAIN estimate
    # has the preregistered number of independent components without spending
    # appreciable node mass. The remaining allocation still optimizes nodes.
    smallest = sorted(components, key=lambda row: (row["node_count"], stable_hash(seed, row["component_id"]), row["component_id"]))
    reserved = set()
    cursor = 0
    reserve_roles = tuple(role for role in ROLES if role in minimum_components)
    for round_index in range(max(minimum_components.values())):
        for role in reserve_roles:
            if round_index < minimum_components[role]:
                row = smallest[cursor]; cursor += 1; reserved.add(row["component_id"]); place(row, role)

    remaining = [row for row in components if row["component_id"] not in reserved]
    ordered = sorted(remaining, key=lambda row: (-row["node_count"], stable_hash(seed, row["component_id"]), row["component_id"]))
    for row in ordered:
        choices = []
        for role_index, role in enumerate(ROLES):
            candidate = dict(node_counts)
            candidate[role] += row["node_count"]
            choices.append((objective(candidate, targets), role_index, role))
        _, _, role = min(choices)
        place(row, role)
    require(all(component_counts[role] > 0 for role in ROLES), "empty role")
    require(sum(protein_counts.values()) == total_proteins and sum(node_counts.values()) == total_nodes, "assignment conservation")
    summary = {
        role: {
            "components": component_counts[role],
            "nodes": node_counts[role],
            "proteins": protein_counts[role],
            "target_fraction": FRACTIONS[role],
            "observed_fraction": node_counts[role] / total_nodes,
            "target_nodes": targets[role],
            "deviation_nodes": node_counts[role] - targets[role],
            "observed_protein_fraction": protein_counts[role] / total_proteins,
            "minimum_components": minimum_components.get(role, 0),
        }
        for role in ROLES
    }
    return sorted(assignments, key=lambda row: row["component_id"]), summary


def expand_assignments(component_assignments, node_rows, membership_rows):
    role_by_component = {row["component_id"]: row["role"] for row in component_assignments}
    require(len(role_by_component) == len(component_assignments), "duplicate component assignment")
    role_by_node = {}
    node_output = []
    for row in node_rows:
        require(set(row) >= {"node_id", "component_id"}, "node row schema")
        require(row["component_id"] in role_by_component and row["node_id"] not in role_by_node, "node/component coverage")
        role = role_by_component[row["component_id"]]
        role_by_node[row["node_id"]] = role
        node_output.append({"node_id": row["node_id"], "component_id": row["component_id"], "role": role})
    protein_output = []
    seen = set()
    for row in membership_rows:
        require(set(row) >= {"protein_id", "node_id"}, "membership row schema")
        require(row["protein_id"] not in seen and row["node_id"] in role_by_node, "protein membership coverage")
        seen.add(row["protein_id"])
        protein_output.append({"protein_id": row["protein_id"], "node_id": row["node_id"], "role": role_by_node[row["node_id"]]})
    require(seen and set(role_by_node.values()) == set(ROLES), "expanded role coverage")
    return sorted(node_output, key=lambda row: row["node_id"]), sorted(protein_output, key=lambda row: row["protein_id"])
