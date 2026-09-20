"""Pure scheduling and receipt audit for direct cross-role sequence searches."""
from collections import Counter


ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")
PASSES = ("masked", "unmasked")


def require(value, message):
    if not value:
        raise ValueError(message)


def role_shards(node_rows, shard_size):
    require(type(shard_size) is int and shard_size > 0, "shard size")
    by_role = {role: [] for role in ROLES}
    seen = set()
    for row in node_rows:
        require(set(row) >= {"node_id", "role"}, "node split schema")
        require(type(row["node_id"]) is int and row["node_id"] >= 0 and row["node_id"] not in seen, "node identity")
        require(row["role"] in by_role, "role")
        seen.add(row["node_id"])
        by_role[row["role"]].append(row["node_id"])
    require(seen and all(by_role[role] for role in ROLES), "role coverage")
    return {
        role: [sorted(by_role[role])[start : start + shard_size] for start in range(0, len(by_role[role]), shard_size)]
        for role in ROLES
    }


def worker_plan(node_rows, query_shard_size=2048, target_shard_size=2000):
    query = role_shards(node_rows, query_shard_size)
    target = role_shards(node_rows, target_shard_size)
    workers = []
    tasks = []
    worker_index = 0
    for query_role in ROLES:
        for query_shard, query_nodes in enumerate(query[query_role]):
            local_tasks = 0
            for target_role in ROLES:
                if target_role == query_role:
                    continue
                for target_shard, target_nodes in enumerate(target[target_role]):
                    for search_pass in PASSES:
                        tasks.append(
                            {
                                "worker_index": worker_index,
                                "query_role": query_role,
                                "query_shard": query_shard,
                                "query_count": len(query_nodes),
                                "target_role": target_role,
                                "target_shard": target_shard,
                                "target_count": len(target_nodes),
                                "search_pass": search_pass,
                            }
                        )
                        local_tasks += 1
            workers.append(
                {
                    "worker_index": worker_index,
                    "query_role": query_role,
                    "query_shard": query_shard,
                    "query_count": len(query_nodes),
                    "task_count": local_tasks,
                }
            )
            worker_index += 1
    summary = {
        "roles": {role: sum(len(shard) for shard in query[role]) for role in ROLES},
        "query_shards": {role: len(query[role]) for role in ROLES},
        "target_shards": {role: len(target[role]) for role in ROLES},
        "workers": len(workers),
        "task_pass_units": len(tasks),
        "ordered_role_pairs": len(ROLES) * (len(ROLES) - 1),
        "search_passes": len(PASSES),
    }
    return workers, tasks, summary


def audit_receipts(tasks, receipts):
    expected = {
        (
            row["worker_index"],
            row["query_role"],
            row["query_shard"],
            row["target_role"],
            row["target_shard"],
            row["search_pass"],
        ): row
        for row in tasks
    }
    require(len(expected) == len(tasks), "duplicate planned task")
    observed = {}
    edge_total = 0
    for row in receipts:
        key = (
            row["worker_index"],
            row["query_role"],
            row["query_shard"],
            row["target_role"],
            row["target_shard"],
            row["search_pass"],
        )
        require(key in expected and key not in observed, "unknown or duplicate receipt")
        plan = expected[key]
        require(row["status"] == "PASS" and row["prefilter_complete"] and not row["unknown_truncation"], "incomplete task")
        require(row["query_count"] == plan["query_count"] and row["target_count"] == plan["target_count"], "task census")
        require(row["cap"] > row["target_count"], "target cap")
        require(row["prefilter_pairs"] == row["alignments_calculated"], "alignment completeness")
        require(0 <= row["qualifying_cross_role_edges"] <= row["alignments_returned"] <= row["alignments_calculated"], "task counts")
        edge_total += row["qualifying_cross_role_edges"]
        observed[key] = row
    require(set(observed) == set(expected), "missing task receipts")
    return {
        "status": "PASS_S3_DIRECT_CROSS_ROLE_SEARCH" if edge_total == 0 else "FAIL_S3_CROSS_ROLE_LEAKAGE_DETECTED",
        "task_pass_units": len(observed),
        "qualifying_cross_role_edges": edge_total,
        "all_twenty_ordered_role_pairs": {
            (query_role, target_role)
            for query_role in ROLES
            for target_role in ROLES
            if query_role != target_role
        }
        == {(row["query_role"], row["target_role"]) for row in receipts},
        "both_search_passes": set(PASSES) == {row["search_pass"] for row in receipts},
    }


def directed_pair_coverage(node_rows, query_shard_size=2048, target_shard_size=2000):
    query = role_shards(node_rows, query_shard_size)
    target = role_shards(node_rows, target_shard_size)
    counts = Counter()
    for query_role in ROLES:
        for query_nodes in query[query_role]:
            for target_role in ROLES:
                if target_role == query_role:
                    continue
                for target_nodes in target[target_role]:
                    for search_pass in PASSES:
                        for query_node in query_nodes:
                            for target_node in target_nodes:
                                counts[(query_node, target_node, search_pass)] += 1
    return counts
