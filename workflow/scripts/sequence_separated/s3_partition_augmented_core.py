"""Pure union closure of frozen S2 components with audited S3 edges."""
from collections import Counter, defaultdict


def require(value, message):
    if not value:
        raise ValueError(message)


def augment_components(component_rows, edge_rows):
    require(isinstance(component_rows, list) and component_rows, "component rows")
    require(isinstance(edge_rows, list) and edge_rows, "nonzero audited edge rows")
    component_by_node = {}
    declared_sizes = {}
    for row in component_rows:
        require(set(row) >= {"node_id", "component_id", "component_size"}, "component row schema")
        node = row["node_id"]
        component = row["component_id"]
        size = row["component_size"]
        require(type(node) is int and node >= 0 and node not in component_by_node, "node identity")
        require(type(component) is int and component >= 0 and type(size) is int and size > 0, "component identity")
        component_by_node[node] = component
        require(component not in declared_sizes or declared_sizes[component] == size, "component size disagreement")
        declared_sizes[component] = size
    observed = Counter(component_by_node.values())
    require(all(observed[component] == size for component, size in declared_sizes.items()), "component size census")

    parent = {component: component for component in declared_sizes}

    def find(value):
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            nxt = parent[value]
            parent[value] = root
            value = nxt
        return root

    def union(left, right):
        a, b = find(left), find(right)
        if a == b:
            return
        # The minimum original component ID is the canonical augmented ID.
        if a > b:
            a, b = b, a
        parent[b] = a

    affected_original = set()
    unique_edges = set()
    for row in edge_rows:
        require(set(row) >= {"query_node", "target_node", "query_role", "target_role"}, "edge row schema")
        query, target = row["query_node"], row["target_node"]
        require(query in component_by_node and target in component_by_node and query != target, "edge node universe")
        require(row["query_role"] != row["target_role"], "edge is not cross-role")
        left, right = component_by_node[query], component_by_node[target]
        require(left != right, "edge already within an S2 component")
        affected_original.update((left, right))
        unique_edges.add((min(query, target), max(query, target)))
        union(left, right)

    groups = defaultdict(list)
    for component in sorted(parent):
        groups[find(component)].append(component)
    canonical = {}
    for members in groups.values():
        identity = min(members)
        for component in members:
            canonical[component] = identity
    augmented_sizes = Counter(canonical[component] for component in component_by_node.values())
    output = [
        {
            "node_id": node,
            "original_component_id": component_by_node[node],
            "augmented_component_id": canonical[component_by_node[node]],
            "augmented_component_size": augmented_sizes[canonical[component_by_node[node]]],
        }
        for node in sorted(component_by_node)
    ]
    merged_groups = [members for members in groups.values() if len(members) > 1]
    summary = {
        "nodes": len(component_by_node),
        "original_components": len(declared_sizes),
        "augmented_components": len(groups),
        "collapsed_original_components": len(declared_sizes) - len(groups),
        "input_edge_rows": len(edge_rows),
        "unique_undirected_node_edges": len(unique_edges),
        "affected_original_components": len(affected_original),
        "affected_nodes": sum(declared_sizes[component] for component in affected_original),
        "merged_component_groups": len(merged_groups),
        "largest_merged_group_original_components": max((len(members) for members in merged_groups), default=1),
        "canonical_rule": "minimum_original_component_id",
    }
    require(summary["augmented_components"] == summary["original_components"] - summary["collapsed_original_components"], "component conservation")
    require(sum(augmented_sizes.values()) == len(component_by_node), "node conservation")
    return output, summary
