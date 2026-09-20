"""Stream S4C node candidates into source-preserving TRAIN activities."""
import argparse
import hashlib
import importlib.util
import json
import traceback
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("s4d_activity_core_for_stream_writer", HERE / "s4d_activity_expansion_core.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)

NODE_COLUMNS = [
    "node_id", "component_id", "role", "sequence_sha256", "sequence_length",
    "representative_protein_id", "protein_id_count",
]
UNION_COLUMNS = [
    "query_node", "reference_node", "query_role", "reference_role",
    "reference_component_id", "mmseqs_rank", "foldseek_rank", "mmseqs_raw_rank",
    "foldseek_raw_rank",
] + [f"{modality}_{name}" for modality in ("mmseqs", "foldseek") for name in core.union_core.RAW_METRICS] + [
    "source_class", "rrf60_score", "rrf_constant", "candidate_union_provenance",
    "query_truth_read",
]
LIBRARY_COLUMNS = [
    "reference_node", "reference_component_id", "reference_protein_id",
    "reference_activity_id", "canonical_ec", "ec_l1", "ec_l2", "ec_l3",
    "ec_l4", "canonical_rhea", "evidence_tier", "reference_role",
    "activity_provenance",
]
ACTIVITY_FIELDS = [
    "reference_protein_id", "reference_activity_id", "canonical_ec", "ec_l1",
    "ec_l2", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier",
]
EXPANDED_COLUMNS = UNION_COLUMNS + ACTIVITY_FIELDS + ["activity_provenance"]
STATUS_COLUMNS = [
    "query_node", "query_role", "status", "retrieved_reference_nodes",
    "expanded_reference_activities", "query_truth_read",
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


def schemas():
    import pyarrow as pa

    union_types = {
        "query_node": pa.int32(), "reference_node": pa.int32(), "query_role": pa.string(),
        "reference_role": pa.string(), "reference_component_id": pa.int32(),
        "mmseqs_rank": pa.int16(), "foldseek_rank": pa.int16(),
        "mmseqs_raw_rank": pa.int32(), "foldseek_raw_rank": pa.int32(),
        "source_class": pa.string(), "rrf60_score": pa.float64(),
        "rrf_constant": pa.int16(), "candidate_union_provenance": pa.string(),
        "query_truth_read": pa.bool_(),
    }
    integer_metrics = {"alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen"}
    for modality in ("mmseqs", "foldseek"):
        for name in core.union_core.RAW_METRICS:
            union_types[f"{modality}_{name}"] = pa.int32() if name in integer_metrics else pa.float64()
    expanded_types = dict(union_types)
    for name in ACTIVITY_FIELDS + ["activity_provenance"]:
        expanded_types[name] = pa.string()
    expanded = pa.schema([(name, expanded_types[name]) for name in EXPANDED_COLUMNS])
    status = pa.schema(
        [
            ("query_node", pa.int32()), ("query_role", pa.string()), ("status", pa.string()),
            ("retrieved_reference_nodes", pa.int32()),
            ("expanded_reference_activities", pa.int32()), ("query_truth_read", pa.bool_()),
        ]
    )
    return expanded, status


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


def load_query_nodes(node_ledger, role):
    import pyarrow.parquet as pq

    table = pq.read_table(node_ledger)
    require(table.column_names == NODE_COLUMNS, "node ledger schema")
    seen, queries = set(), []
    for row in table.to_pylist():
        node = row["node_id"]
        require(type(node) is int and not isinstance(node, bool) and node >= 0 and node not in seen, "node identity")
        require(row["role"] in core.ROLES, "node role")
        seen.add(node)
        if row["role"] == role:
            queries.append(node)
    require(queries, "empty query role")
    return sorted(queries)


def load_library(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    require(table.column_names == LIBRARY_COLUMNS, "activity library schema")
    by_node, seen = defaultdict(list), set()
    previous = None
    for row in table.to_pylist():
        key = row["reference_node"], row["reference_protein_id"], row["reference_activity_id"]
        require(previous is None or key > previous, "activity library order")
        previous = key
        require(row["reference_role"] == "TRAIN" and row["activity_provenance"] == "TRAIN_REFERENCE_SOURCE_ACTIVITY_PRESERVED", "activity library provenance")
        identity = row["reference_protein_id"], row["reference_activity_id"]
        require(identity not in seen, "duplicate activity library identity")
        seen.add(identity)
        by_node[row["reference_node"]].append(row)
    return by_node, len(seen)


def iter_union_groups(path, role, batch_rows):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    require(parquet.schema_arrow.names == UNION_COLUMNS, "union schema")
    current, rows, previous_key = None, [], None
    for batch in parquet.iter_batches(batch_size=batch_rows):
        for row in batch.to_pylist():
            key = row["query_node"], row["reference_node"]
            require(previous_key is None or key > previous_key, "union order")
            previous_key = key
            require(row["query_role"] == role, "union query role")
            if current is None:
                current = row["query_node"]
            if row["query_node"] != current:
                yield current, rows
                current, rows = row["query_node"], []
            rows.append(row)
    if current is not None:
        yield current, rows


def execute(node_ledger, union_candidates, train_activity_library, query_role, output, batch_rows=100000):
    node_ledger, union_candidates = Path(node_ledger).resolve(), Path(union_candidates).resolve()
    train_activity_library, output = Path(train_activity_library).resolve(), Path(output).resolve()
    require(query_role in core.ROLES, "query role")
    require(node_ledger.is_file() and union_candidates.is_file() and train_activity_library.is_file(), "expansion inputs")
    require(output.parent.is_dir() and not output.exists(), "exclusive output")
    output.mkdir(exist_ok=False)
    inputs = {
        "node_ledger": {"path": str(node_ledger), "sha256": sha(node_ledger)},
        "union_candidates": {"path": str(union_candidates), "sha256": sha(union_candidates)},
        "train_activity_library": {"path": str(train_activity_library), "sha256": sha(train_activity_library)},
    }
    emit(
        output / "reservation.json",
        {"status": "ONE_S4D_STREAMING_EXPANSION_ATTEMPT_RESERVED", "query_role": query_role, "inputs": inputs, "automatic_retry": False},
    )
    expanded_sink = status_sink = None
    state, error = "FAIL_CLOSED", None
    try:
        query_nodes = load_query_nodes(node_ledger, query_role)
        query_set = set(query_nodes)
        library, library_rows = load_library(train_activity_library)
        expanded_schema, status_schema = schemas()
        expanded_sink = BufferedSink(output / "activity_candidates.parquet", expanded_schema, batch_rows)
        status_sink = BufferedSink(output / "query_activity_status.parquet", status_schema, batch_rows)
        totals = {
            "queries": len(query_nodes), "union_rows": 0, "expanded_activity_rows": 0,
            "activity_library_rows": library_rows, "activity_candidates_expanded_queries": 0,
            "no_eligible_reference_activity_queries": 0, "no_retrieval_candidate_queries": 0,
        }
        cursor = 0

        def write_status(row):
            totals[{"ACTIVITY_CANDIDATES_EXPANDED": "activity_candidates_expanded_queries", "NO_ELIGIBLE_REFERENCE_ACTIVITY": "no_eligible_reference_activity_queries", "NO_RETRIEVAL_CANDIDATE": "no_retrieval_candidate_queries"}[row["status"]]] += 1
            status_sink.add(row)

        for query, union_rows in iter_union_groups(union_candidates, query_role, batch_rows):
            while cursor < len(query_nodes) and query_nodes[cursor] < query:
                write_status(core.expand_union([], [], [query_nodes[cursor]], query_role)[1][0])
                cursor += 1
            require(cursor < len(query_nodes) and query_nodes[cursor] == query and query in query_set, "union query universe/order")
            activity_subset = []
            for row in union_rows:
                activity_subset.extend(library.get(row["reference_node"], []))
            expanded, statuses = core.expand_union(union_rows, activity_subset, [query], query_role)
            require(len(statuses) == 1, "one query status")
            totals["union_rows"] += len(union_rows)
            totals["expanded_activity_rows"] += len(expanded)
            for row in expanded:
                expanded_sink.add(row)
            write_status(statuses[0])
            cursor += 1
        while cursor < len(query_nodes):
            write_status(core.expand_union([], [], [query_nodes[cursor]], query_role)[1][0])
            cursor += 1
        expanded_sink.close()
        status_sink.close()
        require(expanded_sink.count == totals["expanded_activity_rows"] and status_sink.count == totals["queries"], "writer census")
        require(
            totals["queries"] == totals["activity_candidates_expanded_queries"] + totals["no_eligible_reference_activity_queries"] + totals["no_retrieval_candidate_queries"],
            "status census",
        )
        summary = {
            "status": "PASS_S4D_STREAMING_EXPANSION_PENDING_INDEPENDENT_AUDIT",
            "query_role": query_role, "batch_rows": batch_rows, "inputs": inputs,
            "totals": totals,
            "bounded_live_state": "TRAIN_ACTIVITY_LIBRARY_PLUS_ONE_QUERY_UNION_AND_MATCHING_ACTIVITIES",
            "functional_input_scope": "TRAIN_REFERENCE_ACTIVITY_LIBRARY_ONLY",
            "query_truth_read": False, "retest_evaluation_labels_read": False,
            "synthetic_consensus_labels_created": False, "policy_selection_started": False,
            "pair_generation_started": False, "training_started": False,
        }
        emit(output / "producer_summary.json", summary)
        state = summary["status"]
    except BaseException as exc:
        for sink in (expanded_sink, status_sink):
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
    parser.add_argument("--node-ledger", type=Path, required=True)
    parser.add_argument("--union-candidates", type=Path, required=True)
    parser.add_argument("--train-activity-library", type=Path, required=True)
    parser.add_argument("--query-role", choices=core.ROLES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=100000)
    args = parser.parse_args()
    execute(args.node_ledger, args.union_candidates, args.train_activity_library, args.query_role, args.output, args.batch_rows)


if __name__ == "__main__":
    main()

