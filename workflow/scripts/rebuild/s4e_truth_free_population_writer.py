"""Build the label-blind SC1 population pair sample from S4D outputs."""
import argparse
import hashlib
import json
import math
import traceback
from collections import Counter, defaultdict
from pathlib import Path


ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")
SEED = 20260819
RAW_METRICS = (
    "fident", "alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen",
    "qcov", "tcov", "evalue", "bits",
)
PROTEIN_COLUMNS = ["protein_id", "node_id", "component_id", "role"]
UNION_COLUMNS = [
    "query_node", "reference_node", "query_role", "reference_role",
    "reference_component_id", "mmseqs_rank", "foldseek_rank", "mmseqs_raw_rank",
    "foldseek_raw_rank",
] + [f"{modality}_{name}" for modality in ("mmseqs", "foldseek") for name in RAW_METRICS] + [
    "source_class", "rrf60_score", "rrf_constant", "candidate_union_provenance",
    "query_truth_read",
]
ACTIVITY_FIELDS = [
    "reference_protein_id", "reference_activity_id", "canonical_ec", "ec_l1",
    "ec_l2", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier",
]
EXPANDED_COLUMNS = UNION_COLUMNS + ACTIVITY_FIELDS + ["activity_provenance"]
POPULATION_COLUMNS = ["query_protein_id", "query_component_id"] + EXPANDED_COLUMNS + [
    "pair_set", "sampling_probability", "sample_weight", "sampling_design",
    "sampling_seed", "pair_provenance",
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


def selected(pair_key, probability, seed=SEED):
    require(0 < probability <= 1 and type(seed) is int and not isinstance(seed, bool), "sampling parameters")
    value = int.from_bytes(hashlib.sha256(f"{seed}|{pair_key}".encode("utf-8")).digest()[:8], "big")
    threshold = math.floor(probability * ((1 << 64) - 1))
    return value <= threshold


def population_schema():
    import pyarrow as pa

    types = {
        "query_protein_id": pa.string(), "query_component_id": pa.int32(),
        "query_node": pa.int32(), "reference_node": pa.int32(),
        "query_role": pa.string(), "reference_role": pa.string(),
        "reference_component_id": pa.int32(), "mmseqs_rank": pa.int16(),
        "foldseek_rank": pa.int16(), "mmseqs_raw_rank": pa.int32(),
        "foldseek_raw_rank": pa.int32(), "source_class": pa.string(),
        "rrf60_score": pa.float64(), "rrf_constant": pa.int16(),
        "candidate_union_provenance": pa.string(), "query_truth_read": pa.bool_(),
        "reference_protein_id": pa.string(), "reference_activity_id": pa.string(),
        "canonical_ec": pa.string(), "ec_l1": pa.string(), "ec_l2": pa.string(),
        "ec_l3": pa.string(), "ec_l4": pa.string(), "canonical_rhea": pa.string(),
        "evidence_tier": pa.string(), "activity_provenance": pa.string(),
        "pair_set": pa.string(), "sampling_probability": pa.float64(),
        "sample_weight": pa.float64(), "sampling_design": pa.string(),
        "sampling_seed": pa.int64(), "pair_provenance": pa.string(),
    }
    integer_metrics = {"alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen"}
    for modality in ("mmseqs", "foldseek"):
        for name in RAW_METRICS:
            types[f"{modality}_{name}"] = pa.int32() if name in integer_metrics else pa.float64()
    return pa.schema([(name, types[name]) for name in POPULATION_COLUMNS])


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


def load_proteins(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    require(table.column_names == PROTEIN_COLUMNS, "protein ledger schema")
    by_node, by_protein, component_roles = defaultdict(list), {}, {}
    for row in table.to_pylist():
        protein, node, component, role = row["protein_id"], row["node_id"], row["component_id"], row["role"]
        require(isinstance(protein, str) and protein and protein not in by_protein, "protein identity")
        require(type(node) is int and type(component) is int and role in ROLES, "protein ledger row")
        require(component not in component_roles or component_roles[component] == role, "component split across roles")
        by_protein[protein] = row
        by_node[node].append(row)
        component_roles[component] = role
    require(by_protein and set(row["role"] for row in by_protein.values()) == set(ROLES), "protein role universe")
    for rows in by_node.values():
        require(len({(row["component_id"], row["role"]) for row in rows}) == 1, "node split identity")
        rows.sort(key=lambda row: row["protein_id"])
    return by_node, by_protein


def iter_candidates(path, role, batch_rows=100000):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    require(parquet.schema_arrow.names == EXPANDED_COLUMNS, f"{role} activity candidate schema")
    previous = None
    for batch in parquet.iter_batches(batch_size=batch_rows):
        for row in batch.to_pylist():
            key = row["query_node"], row["reference_node"], row["reference_protein_id"], row["reference_activity_id"]
            require(previous is None or key > previous, f"{role} candidate order")
            previous = key
            require(row["query_role"] == role and row["reference_role"] == "TRAIN", f"{role} candidate roles")
            require(not row["query_truth_read"], "query truth flag")
            require(row["candidate_union_provenance"] == "TRUTH_FREE_AUDITED_MODALITY_UNION_NO_POLICY_SELECTION", "union provenance")
            require(row["activity_provenance"] == "TRAIN_REFERENCE_ACTIVITY_EXPANDED_AFTER_TRUTH_FREE_RETRIEVAL", "activity provenance")
            yield row


def validate_candidate_identity(row, role, proteins_by_node, proteins_by_id):
    require(row["query_node"] in proteins_by_node, "query node universe")
    qmeta = proteins_by_node[row["query_node"]][0]
    require(qmeta["role"] == role, "query ledger role")
    reference = proteins_by_id.get(row["reference_protein_id"])
    require(reference is not None and reference["role"] == "TRAIN", "reference protein role")
    require(reference["node_id"] == row["reference_node"] and reference["component_id"] == row["reference_component_id"], "reference protein identity")
    require(row["query_node"] != row["reference_node"], "self-node candidate")
    if role != "TRAIN":
        require(qmeta["component_id"] != row["reference_component_id"], "cross-role component leakage")


def pair_key(query_protein, row):
    return "|".join(
        map(str, (
            query_protein, row["query_node"], row["reference_node"],
            row["reference_protein_id"], row["reference_activity_id"],
        ))
    )


def execute(protein_ledger, candidates_by_role, target_pairs, output, seed=SEED, batch_rows=100000):
    protein_ledger, output = Path(protein_ledger).resolve(), Path(output).resolve()
    require(type(target_pairs) is int and not isinstance(target_pairs, bool) and target_pairs > 0, "target pairs")
    require(seed == SEED, "sampling seed")
    require(set(candidates_by_role) == set(ROLES), "candidate role inputs")
    paths = {role: Path(candidates_by_role[role]).resolve() for role in ROLES}
    require(protein_ledger.is_file() and all(path.is_file() for path in paths.values()), "population inputs")
    require(output.parent.is_dir() and not output.exists(), "exclusive output")
    output.mkdir(exist_ok=False)
    inputs = {"protein_ledger": {"path": str(protein_ledger), "sha256": sha(protein_ledger)}}
    inputs["activity_candidates"] = {role: {"path": str(paths[role]), "sha256": sha(paths[role])} for role in ROLES}
    emit(
        output / "reservation.json",
        {"status": "ONE_S4E_POPULATION_ATTEMPT_RESERVED", "target_pairs": target_pairs, "seed": seed, "inputs": inputs, "automatic_retry": False},
    )
    sink, state, error = None, "FAIL_CLOSED", None
    try:
        by_node, by_protein = load_proteins(protein_ledger)
        universe_by_role = Counter()
        for role in ROLES:
            for row in iter_candidates(paths[role], role, batch_rows):
                validate_candidate_identity(row, role, by_node, by_protein)
                universe_by_role[role] += len(by_node[row["query_node"]])
        universe = sum(universe_by_role.values())
        require(universe > 0, "empty population universe")
        probability = min(1.0, target_pairs / universe)
        weight = 1.0 / probability
        sink = BufferedSink(output / "population_pairs.parquet", population_schema(), batch_rows)
        sampled_by_role = Counter()
        for role in ROLES:
            for row in iter_candidates(paths[role], role, batch_rows):
                for query in by_node[row["query_node"]]:
                    key = pair_key(query["protein_id"], row)
                    if selected(key, probability, seed):
                        sink.add(
                            {
                                "query_protein_id": query["protein_id"],
                                "query_component_id": query["component_id"],
                                **row,
                                "pair_set": "population_atlas",
                                "sampling_probability": probability,
                                "sample_weight": weight,
                                "sampling_design": "LABEL_BLIND_SHA256_BERNOULLI_FROM_COMPLETE_RETRIEVED_ACTIVITY_UNIVERSE",
                                "sampling_seed": seed,
                                "pair_provenance": "QUERY_NODE_EXPANDED_TO_SOURCE_PROTEIN_AFTER_TRUTH_FREE_CANDIDATE_GENERATION",
                            }
                        )
                        sampled_by_role[role] += 1
        sink.close()
        require(sink.count == sum(sampled_by_role.values()), "sample writer census")
        require(sink.count > 0, "empty realized population sample")
        summary = {
            "status": "PASS_S4E_TRUTH_FREE_POPULATION_PENDING_INDEPENDENT_AUDIT",
            "seed": seed, "target_pairs": target_pairs, "inputs": inputs,
            "universe_pairs": universe, "universe_pairs_by_role": {role: universe_by_role[role] for role in ROLES},
            "sampling_probability": probability, "sample_weight": weight,
            "realized_pairs": sink.count, "realized_pairs_by_role": {role: sampled_by_role[role] for role in ROLES},
            "sampling_design": "LABEL_BLIND_SHA256_BERNOULLI_FROM_COMPLETE_RETRIEVED_ACTIVITY_UNIVERSE",
            "query_node_expanded_to_all_source_proteins": True,
            "query_truth_read": False, "retest_evaluation_labels_read": False,
            "outcome_balancing_used": False, "difficult_case_enrichment_used": False,
            "pair_features_started": False, "preprocessing_started": False,
            "training_started": False, "evaluation_started": False,
        }
        emit(output / "producer_summary.json", summary)
        state = summary["status"]
    except BaseException as exc:
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
    for role in ROLES:
        parser.add_argument(
            f"--{role.lower().replace('_', '-')}-candidates",
            dest=f"{role.lower()}_candidates",
            type=Path,
            required=True,
        )
    parser.add_argument("--target-pairs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-rows", type=int, default=100000)
    args = parser.parse_args()
    mapping = {role: getattr(args, f"{role.lower()}_candidates") for role in ROLES}
    execute(args.protein_ledger, mapping, args.target_pairs, args.output, args.seed, args.batch_rows)


if __name__ == "__main__":
    main()
