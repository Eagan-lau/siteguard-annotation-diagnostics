"""Stream deterministic TRAIN-only supervised augmentation pairs.

The semantic core used SHA-256 sorting to specify an unbiased priority sample.
For production scale, this writer realizes the same sampling estimand with a
query-keyed pseudorandom permutation (PRP).  It never materializes the pair
universe and retains only input indexes plus one query's selected rows.
"""
import argparse
import hashlib
import json
import traceback
from collections import Counter, defaultdict
from pathlib import Path


SEED = 20260819
ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")
COUNTS = {
    "same_exact_rhea": 2,
    "same_ec_l4": 1,
    "same_ec_l3": 1,
    "same_family_divergent": 2,
    "matched_control": 2,
    "global_control": 2,
}
PROTEIN_COLUMNS = ["protein_id", "node_id", "component_id", "role"]
NODE_COLUMNS = [
    "node_id", "component_id", "role", "sequence_sha256", "sequence_length",
    "representative_protein_id", "protein_id_count",
]
COVARIATE_COLUMNS = PROTEIN_COLUMNS + [
    "primary_pfam", "primary_pfam_clan", "primary_cath_superfamily",
    "foldseek_structure_cluster", "taxonomy_id", "taxonomy_group", "filename",
    "model_version", "compressed_size", "has_structure", "structure_source",
]
LIBRARY_COLUMNS = [
    "reference_node", "reference_component_id", "reference_protein_id",
    "reference_activity_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3",
    "ec_l4", "canonical_rhea", "evidence_tier", "reference_role",
    "activity_provenance",
]
PAIR_COLUMNS = [
    "query_protein_id", "query_node", "query_component_id", "query_role",
    "reference_node", "reference_component_id", "reference_protein_id",
    "reference_activity_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3",
    "ec_l4", "canonical_rhea", "evidence_tier", "candidate_origin",
    "augmentation_mechanisms", "maximum_mechanism_inclusion_probability",
    "augmentation_probability_is_population_weight", "query_truth_provenance",
    "outcome_semantics",
]
STATUS_COLUMNS = [
    "query_protein_id", "query_node", "query_component_id", "query_role",
    "status", "documented_train_activities", "selected_pairs",
    "selected_mechanism_occurrences",
]
SELECTION_ENGINE = "SHA256_KEYED_SIX_ROUND_FEISTEL_PRP_V1"


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


def length_bin(length):
    require(type(length) is int and not isinstance(length, bool) and length > 0, "sequence length")
    for boundary, label in (
        (150, "0001-0150"), (250, "0151-0250"), (400, "0251-0400"),
        (600, "0401-0600"), (1000, "0601-1000"),
    ):
        if length <= boundary:
            return label
    return "1001+"


def _feistel(value, bits, key):
    half = bits // 2
    mask = (1 << half) - 1
    left, right = value >> half, value & mask
    for round_index in range(6):
        block = f"{key}|{round_index}|{right}".encode("utf-8")
        function = int.from_bytes(hashlib.sha256(block).digest()[:8], "big") & mask
        left, right = right, left ^ function
    return (left << half) | right


def permuted_index(position, size, key):
    require(type(size) is int and size > 0 and 0 <= position < size, "PRP index domain")
    if size == 1:
        return 0
    bits = (size - 1).bit_length()
    if bits % 2:
        bits += 1
    value = position
    while True:
        value = _feistel(value, bits, key)
        if value < size:
            return value


def prp_pick(rows, query, source, k):
    size = len(rows)
    if not size or not k:
        return [], 0.0, 0
    key = f"{SEED}|{query}|{source}"
    take = min(k, size)
    chosen = [rows[permuted_index(position, size, key)] for position in range(take)]
    return chosen, take / size, take


def pair_schema():
    import pyarrow as pa

    string_fields = {
        "query_protein_id", "query_role", "reference_protein_id",
        "reference_activity_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3",
        "ec_l4", "canonical_rhea", "evidence_tier", "candidate_origin",
        "augmentation_mechanisms", "query_truth_provenance", "outcome_semantics",
    }
    int_fields = {"query_node", "query_component_id", "reference_node", "reference_component_id"}
    bool_fields = {"augmentation_probability_is_population_weight"}
    types = {name: pa.string() for name in string_fields}
    types.update({name: pa.int32() for name in int_fields})
    types.update({name: pa.bool_() for name in bool_fields})
    types["maximum_mechanism_inclusion_probability"] = pa.float64()
    return pa.schema([(name, types[name]) for name in PAIR_COLUMNS])


def status_schema():
    import pyarrow as pa

    return pa.schema([
        ("query_protein_id", pa.string()), ("query_node", pa.int32()),
        ("query_component_id", pa.int32()), ("query_role", pa.string()),
        ("status", pa.string()), ("documented_train_activities", pa.int32()),
        ("selected_pairs", pa.int32()), ("selected_mechanism_occurrences", pa.int32()),
    ])


class BufferedSink:
    def __init__(self, path, schema, batch_rows):
        import pyarrow.parquet as pq

        require(type(batch_rows) is int and not isinstance(batch_rows, bool) and batch_rows > 0, "batch rows")
        self.schema, self.batch_rows, self.rows, self.count = schema, batch_rows, [], 0
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.batch_rows:
            self.flush()

    def flush(self):
        if self.rows:
            import pyarrow as pa

            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.count += len(self.rows)
            self.rows = []

    def close(self):
        self.flush()
        self.writer.close()


def read_table(path, expected, name):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    require(table.column_names == expected, name + " schema")
    return table.to_pylist()


def load_inputs(protein_path, node_path, covariate_path, library_path):
    protein_rows = read_table(protein_path, PROTEIN_COLUMNS, "protein ledger")
    node_rows = read_table(node_path, NODE_COLUMNS, "node ledger")
    covariate_rows = read_table(covariate_path, COVARIATE_COLUMNS, "covariate ledger")
    library_rows = read_table(library_path, LIBRARY_COLUMNS, "TRAIN activity library")

    proteins, by_node = {}, defaultdict(list)
    component_roles = {}
    for row in protein_rows:
        protein, node, component, role = row["protein_id"], row["node_id"], row["component_id"], row["role"]
        require(isinstance(protein, str) and protein and protein not in proteins, "protein identity")
        require(type(node) is int and type(component) is int and role in ROLES, "protein ledger row")
        require(component not in component_roles or component_roles[component] == role, "component split across roles")
        proteins[protein], component_roles[component] = row, role
        by_node[node].append(row)
    require(proteins and set(row["role"] for row in proteins.values()) == set(ROLES), "protein role universe")

    nodes = {}
    for row in node_rows:
        node = row["node_id"]
        require(type(node) is int and node not in nodes and row["role"] in ROLES, "node identity")
        require(isinstance(row["sequence_sha256"], str) and len(row["sequence_sha256"]) == 64, "node sequence identity")
        require(type(row["protein_id_count"]) is int and row["protein_id_count"] > 0, "node protein count")
        require(row["representative_protein_id"] in proteins, "node representative")
        nodes[node] = row
    require(set(nodes) == set(by_node), "node universe")
    for node, rows in by_node.items():
        meta = nodes[node]
        require(len(rows) == meta["protein_id_count"], "node protein census")
        require(all(row["component_id"] == meta["component_id"] and row["role"] == meta["role"] for row in rows), "node role/component")
        require(proteins[meta["representative_protein_id"]]["node_id"] == node, "node representative mapping")

    covariates = {}
    for row in covariate_rows:
        protein = row["protein_id"]
        require(protein in proteins and protein not in covariates, "covariate protein identity")
        require(all(row[name] == proteins[protein][name] for name in ("node_id", "component_id", "role")), "covariate SC1 identity")
        require(type(row["has_structure"]) is bool, "covariate structure availability")
        covariates[protein] = row
    require(set(covariates) == set(proteins), "covariate universe")

    activities = []
    activity_ids = set()
    truth = defaultdict(lambda: {"activities": set(), "ec_l3": set(), "ec_l4": set(), "rhea": set()})
    by_rhea, by_ec4, by_ec3 = defaultdict(list), defaultdict(list), defaultdict(list)
    by_family, by_match, by_activity_node = defaultdict(list), defaultdict(list), defaultdict(list)
    global_ec4_counts, global_rhea_counts, global_ec4_rhea_counts = Counter(), Counter(), Counter()
    previous = None
    for row in library_rows:
        key = (row["reference_node"], row["reference_protein_id"], row["reference_activity_id"])
        require(previous is None or key > previous, "TRAIN activity library order")
        previous = key
        activity, protein = row["reference_activity_id"], row["reference_protein_id"]
        require(isinstance(activity, str) and activity and activity not in activity_ids, "TRAIN activity identity")
        meta = proteins.get(protein)
        require(meta is not None and meta["role"] == "TRAIN", "TRAIN activity role")
        require(row["reference_node"] == meta["node_id"] and row["reference_component_id"] == meta["component_id"], "TRAIN activity protein mapping")
        require(row["reference_role"] == "TRAIN" and row["activity_provenance"] == "TRAIN_REFERENCE_SOURCE_ACTIVITY_PRESERVED", "TRAIN activity provenance")
        canonical = row["canonical_ec"]
        parts = canonical.split(".") if isinstance(canonical, str) else []
        require(len(parts) == 4 and row["ec_l1"] == parts[0] and row["ec_l2"] == ".".join(parts[:2]) and row["ec_l3"] == ".".join(parts[:3]) and row["ec_l4"] == canonical, "EC hierarchy")
        require(row["canonical_rhea"] is None or (isinstance(row["canonical_rhea"], str) and row["canonical_rhea"].startswith("RHEA:")), "canonical Rhea")
        activity_ids.add(activity)
        activities.append(row)
        by_ec3[row["ec_l3"]].append(row)
        by_ec4[row["ec_l4"]].append(row)
        by_activity_node[row["reference_node"]].append(row)
        global_ec4_counts[row["ec_l4"]] += 1
        if row["canonical_rhea"] is not None:
            by_rhea[row["canonical_rhea"]].append(row)
            global_rhea_counts[row["canonical_rhea"]] += 1
            global_ec4_rhea_counts[(row["ec_l4"], row["canonical_rhea"])] += 1
        by_family[covariates[protein]["primary_pfam"]].append(row)
        match = (length_bin(nodes[meta["node_id"]]["sequence_length"]), covariates[protein]["taxonomy_group"])
        by_match[match].append(row)
        target = truth[protein]
        target["activities"].add(activity)
        target["ec_l3"].add(row["ec_l3"])
        target["ec_l4"].add(row["ec_l4"])
        if row["canonical_rhea"] is not None:
            target["rhea"].add(row["canonical_rhea"])
    require(activities, "empty TRAIN activity library")
    activities.sort(key=lambda row: (row["reference_protein_id"], row["reference_activity_id"]))
    for index in (by_rhea, by_ec4, by_ec3, by_family, by_match, by_activity_node):
        for rows in index.values():
            rows.sort(key=lambda row: (row["reference_protein_id"], row["reference_activity_id"]))
    indexes = {
        "by_rhea": by_rhea, "by_ec4": by_ec4, "by_ec3": by_ec3,
        "by_family": by_family, "by_match": by_match,
        "by_activity_node": by_activity_node,
        "global_ec4_counts": global_ec4_counts,
        "global_rhea_counts": global_rhea_counts,
        "global_ec4_rhea_counts": global_ec4_rhea_counts,
    }
    return proteins, nodes, covariates, activities, truth, indexes


def divergent(row, query_truth):
    return row["ec_l4"] not in query_truth["ec_l4"] and (
        row["canonical_rhea"] is None or row["canonical_rhea"] not in query_truth["rhea"]
    )


def eligible(row, query_node, query_truth, require_divergence=False):
    return row["reference_node"] != query_node and (not require_divergence or divergent(row, query_truth))


def select_materialized(query, source, pool, k, query_node, query_truth, require_divergence=False):
    rows = [row for row in pool if eligible(row, query_node, query_truth, require_divergence)]
    return prp_pick(rows, query, source, k)


def global_eligible_count(activities, query_node, query_truth, indexes):
    ec4 = query_truth["ec_l4"]
    rhea = query_truth["rhea"]
    label_excluded = sum(indexes["global_ec4_counts"][label] for label in ec4)
    label_excluded += sum(indexes["global_rhea_counts"][label] for label in rhea)
    label_excluded -= sum(indexes["global_ec4_rhea_counts"][(ec_label, rhea_label)] for ec_label in ec4 for rhea_label in rhea)
    own_node = indexes["by_activity_node"][query_node]
    own_node_label_excluded = sum(
        row["ec_l4"] in ec4 or (row["canonical_rhea"] is not None and row["canonical_rhea"] in rhea)
        for row in own_node
    )
    count = len(activities) - label_excluded - len(own_node) + own_node_label_excluded
    require(0 <= count <= len(activities), "global eligibility count")
    return count


def select_global(query, source, activities, k, query_node, query_truth, indexes):
    eligible_count = global_eligible_count(activities, query_node, query_truth, indexes)
    if not eligible_count or not k:
        return [], 0.0, 0
    take = min(k, eligible_count)
    key = f"{SEED}|{query}|{source}"
    selected, examined = [], 0
    for position in range(len(activities)):
        row = activities[permuted_index(position, len(activities), key)]
        examined += 1
        if eligible(row, query_node, query_truth, True):
            selected.append(row)
            if len(selected) == take:
                break
    require(len(selected) == take, "global PRP eligible census")
    return selected, take / eligible_count, examined


def iter_query_rows(query, proteins, nodes, covariates, activities, truth, indexes):
    meta, qtruth, cov = proteins[query], truth[query], covariates[query]
    selected = {}
    mechanism_occurrences = Counter()
    global_positions = 0

    def add(source, label, pool, k, require_divergence=False, global_pool=False):
        nonlocal global_positions
        source_key = f"{source}:{label}" if label is not None else source
        if global_pool:
            rows, probability, examined = select_global(query, source_key, pool, k, meta["node_id"], qtruth, indexes)
            global_positions += examined
        else:
            rows, probability, _ = select_materialized(query, source_key, pool, k, meta["node_id"], qtruth, require_divergence)
        for row in rows:
            key = (row["reference_protein_id"], row["reference_activity_id"])
            item = selected.setdefault(key, {"activity": row, "mechanisms": [], "probabilities": []})
            item["mechanisms"].append(source_key)
            item["probabilities"].append(probability)
            mechanism_occurrences[source] += 1

    for label in sorted(qtruth["rhea"]):
        add("same_exact_rhea", label, indexes["by_rhea"][label], COUNTS["same_exact_rhea"])
    for label in sorted(qtruth["ec_l4"]):
        add("same_ec_l4", label, indexes["by_ec4"][label], COUNTS["same_ec_l4"])
    for label in sorted(qtruth["ec_l3"]):
        add("same_ec_l3", label, indexes["by_ec3"][label], COUNTS["same_ec_l3"])
    add("same_family_divergent", None, indexes["by_family"][cov["primary_pfam"]], COUNTS["same_family_divergent"], True)
    match = (length_bin(nodes[meta["node_id"]]["sequence_length"]), cov["taxonomy_group"])
    add("matched_control", None, indexes["by_match"][match], COUNTS["matched_control"], True)
    add("global_control", None, activities, COUNTS["global_control"], True, True)

    rows = []
    for key in sorted(selected):
        item, row = selected[key], selected[key]["activity"]
        rows.append({
            "query_protein_id": query,
            "query_node": meta["node_id"],
            "query_component_id": meta["component_id"],
            "query_role": "TRAIN",
            **{name: row[name] for name in (
                "reference_node", "reference_component_id", "reference_protein_id",
                "reference_activity_id", "canonical_ec", "ec_l1", "ec_l2",
                "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier",
            )},
            "candidate_origin": "SUPERVISED_TRAIN_AUGMENTATION_NOT_DEPLOYMENT_CANDIDATE",
            "augmentation_mechanisms": "+".join(sorted(item["mechanisms"])),
            "maximum_mechanism_inclusion_probability": max(item["probabilities"]),
            "augmentation_probability_is_population_weight": False,
            "query_truth_provenance": "TRAIN_SAMPLING_ONLY_NOT_MODEL_INPUT",
            "outcome_semantics": "DOCUMENTED_CONCORDANCE_NOT_BIOCHEMICAL_NEGATIVE",
        })
    return rows, mechanism_occurrences, global_positions


def execute(protein_ledger, node_ledger, covariate_ledger, train_activity_library, output, batch_rows=100000):
    paths = {
        "protein_ledger": Path(protein_ledger).resolve(),
        "node_ledger": Path(node_ledger).resolve(),
        "covariate_ledger": Path(covariate_ledger).resolve(),
        "train_activity_library": Path(train_activity_library).resolve(),
    }
    output = Path(output).resolve()
    require(all(path.is_file() for path in paths.values()), "augmentation inputs")
    require(output.parent.is_dir() and not output.exists(), "exclusive output")
    output.mkdir(exist_ok=False)
    inputs = {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}
    emit(output / "reservation.json", {
        "status": "ONE_S4H_TRAIN_AUGMENTATION_STREAM_ATTEMPT_RESERVED",
        "inputs": inputs, "automatic_retry": False,
    })
    pair_sink = status_sink = None
    state, error = "FAIL_CLOSED", None
    try:
        proteins, nodes, covariates, activities, truth, indexes = load_inputs(**{
            "protein_path": paths["protein_ledger"],
            "node_path": paths["node_ledger"],
            "covariate_path": paths["covariate_ledger"],
            "library_path": paths["train_activity_library"],
        })
        pair_sink = BufferedSink(output / "train_supervised_augmentation_pairs.parquet", pair_schema(), batch_rows)
        status_sink = BufferedSink(output / "train_augmentation_query_status.parquet", status_schema(), batch_rows)
        status_counts, mechanism_counts = Counter(), Counter()
        merged_pairs = global_positions = selected_occurrences = 0
        train = sorted(protein for protein, row in proteins.items() if row["role"] == "TRAIN")
        for query in train:
            qtruth = truth.get(query)
            if not qtruth or not qtruth["activities"]:
                rows, occurrences, examined = [], Counter(), 0
                status = "NO_ELIGIBLE_TRAIN_QUERY_ACTIVITY"
            else:
                rows, occurrences, examined = iter_query_rows(query, proteins, nodes, covariates, activities, truth, indexes)
                status = "TRAIN_SUPERVISED_AUGMENTATION_WRITTEN" if rows else "NO_AUGMENTATION_PAIR_SELECTED"
            occurrence_count = sum(occurrences.values())
            selected_occurrences += occurrence_count
            merged_pairs += occurrence_count - len(rows)
            global_positions += examined
            mechanism_counts.update(occurrences)
            status_counts[status] += 1
            for row in rows:
                pair_sink.add(row)
            meta = proteins[query]
            status_sink.add({
                "query_protein_id": query, "query_node": meta["node_id"],
                "query_component_id": meta["component_id"], "query_role": "TRAIN",
                "status": status,
                "documented_train_activities": len(qtruth["activities"]) if qtruth else 0,
                "selected_pairs": len(rows),
                "selected_mechanism_occurrences": occurrence_count,
            })
        pair_sink.close(); status_sink.close()
        require(status_sink.count == len(train), "augmentation status census")
        require(pair_sink.count == selected_occurrences - merged_pairs, "augmentation pair census")
        summary = {
            "status": "PASS_S4H_TRAIN_AUGMENTATION_STREAM_PENDING_INDEPENDENT_AUDIT",
            "inputs": inputs,
            "seed": SEED,
            "counts_per_mechanism": COUNTS,
            "selection_engine": SELECTION_ENGINE,
            "global_eligibility_count_engine": "INCLUSION_EXCLUSION_LABEL_COUNTS_PLUS_SAME_NODE_REPLAY",
            "train_proteins": len(train),
            "train_activity_library_rows": len(activities),
            "augmentation_pair_rows": pair_sink.count,
            "selected_mechanism_occurrences": selected_occurrences,
            "duplicate_pair_mechanism_merges": merged_pairs,
            "mechanism_occurrence_counts": {name: mechanism_counts[name] for name in COUNTS},
            "status_counts": {name: status_counts[name] for name in (
                "TRAIN_SUPERVISED_AUGMENTATION_WRITTEN",
                "NO_AUGMENTATION_PAIR_SELECTED",
                "NO_ELIGIBLE_TRAIN_QUERY_ACTIVITY",
            )},
            "global_prp_positions_examined": global_positions,
            "maximum_output_rows_held_per_query": True,
            "full_pair_universe_materialized": False,
            "query_truth_scope": "TRAIN_SAMPLING_ONLY_NOT_MODEL_INPUT",
            "nontrain_truth_read": False,
            "candidate_membership_or_rank_changed": False,
            "augmentation_probability_is_population_weight": False,
            "balanced_training_sampler_started": False,
            "feature_matrix_created": False,
            "training_started": False,
        }
        emit(output / "producer_summary.json", summary)
        state = summary["status"]
    except BaseException as exc:
        for sink in (pair_sink, status_sink):
            if sink is not None:
                try:
                    sink.close()
                except BaseException:
                    pass
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    emit(output / "terminal.json", {"status": state, "error": error, "automatic_retry": False})
    if error:
        raise RuntimeError(error["message"])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein-ledger", type=Path, required=True)
    parser.add_argument("--node-ledger", type=Path, required=True)
    parser.add_argument("--covariate-ledger", type=Path, required=True)
    parser.add_argument("--train-activity-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=100000)
    args = parser.parse_args()
    execute(args.protein_ledger, args.node_ledger, args.covariate_ledger, args.train_activity_library, args.output, args.batch_rows)


if __name__ == "__main__":
    main()
