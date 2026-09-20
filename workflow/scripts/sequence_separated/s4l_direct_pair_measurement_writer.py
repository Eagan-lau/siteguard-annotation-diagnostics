"""Production S4L manifest and direct-alignment prefilter shard writer."""
import argparse
import csv
import hashlib
import importlib.util
import json
import sqlite3
import traceback
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("s4l_direct_pair_measurement_core_writer", HERE / "s4l_direct_pair_measurement_core.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)

SEED = core.SEED
DEFAULT_SHARDS = core.DEFAULT_SHARDS
S4K_PROJECTION = list(core.PAIR_PROJECTION)
PROTEIN_COLUMNS = ["protein_id", "node_id", "component_id", "role"]
FOLDSEEK_COLUMNS = ["node_id", "component_id", "role", "protein_id", "db_key", "structure_name"]
MANIFEST_COLUMNS = list(core.PLAN_COLUMNS)
ROLES = {"TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST"}


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


def input_record(path):
    require(path.is_file() and not path.is_symlink(), "required regular input " + path.name)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def read_proteins(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=PROTEIN_COLUMNS)
    require(table.column_names == PROTEIN_COLUMNS, "protein ledger projection")
    proteins, seen = {}, set()
    for row in table.to_pylist():
        protein = row["protein_id"]
        require(isinstance(protein, str) and protein and protein not in seen, "protein ledger identity")
        seen.add(protein)
        require(
            type(row["node_id"]) is int and row["node_id"] >= 0
            and type(row["component_id"]) is int and row["component_id"] >= 0
            and row["role"] in ROLES,
            "protein ledger metadata",
        )
        if row["role"] == "TRAIN":
            proteins[protein] = (row["node_id"], row["component_id"])
    require(proteins, "empty TRAIN protein ledger")
    return proteins, len(seen)


def read_mmseqs_lookup(path):
    mapping, keys = {}, set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, fields in enumerate(csv.reader(handle, delimiter="\t"), 1):
            require(len(fields) == 3 and all(fields), f"MMseqs lookup fields line {line_number}")
            try:
                db_key, node_id, file_number = int(fields[0]), int(fields[1]), int(fields[2])
            except ValueError as exc:
                raise RuntimeError(f"MMseqs lookup integer line {line_number}") from exc
            require(db_key >= 0 and node_id >= 0 and file_number >= 0, f"MMseqs lookup range line {line_number}")
            require(node_id not in mapping and db_key not in keys, "MMseqs one-to-one node/key mapping")
            mapping[node_id], _ = db_key, keys.add(db_key)
    require(mapping, "empty MMseqs lookup")
    return mapping


def read_foldseek_mapping(path):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    require(parquet.schema_arrow.names == FOLDSEEK_COLUMNS, "Foldseek selection schema")
    mapping, keys = {}, set()
    for batch in parquet.iter_batches(columns=FOLDSEEK_COLUMNS):
        for row in batch.to_pylist():
            require(
                type(row["node_id"]) is int and row["node_id"] >= 0
                and type(row["component_id"]) is int and row["component_id"] >= 0
                and row["role"] in ROLES and isinstance(row["protein_id"], str) and row["protein_id"]
                and type(row["db_key"]) is int and row["db_key"] >= 0
                and isinstance(row["structure_name"], str) and row["structure_name"],
                "Foldseek selection metadata",
            )
            node, key = row["node_id"], row["db_key"]
            require(node not in mapping and key not in keys, "Foldseek one-to-one node/key mapping")
            mapping[node], _ = key, keys.add(key)
    return mapping


def connect_database(path):
    require(not path.exists(), "pair index exists")
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-262144")
    connection.execute("""
        CREATE TABLE activity_rows (
            query_protein_id TEXT NOT NULL,
            reference_protein_id TEXT NOT NULL,
            query_node INTEGER NOT NULL,
            reference_node INTEGER NOT NULL,
            query_component_id INTEGER NOT NULL,
            reference_component_id INTEGER NOT NULL
        )
    """)
    return connection


def validate_pair(row, proteins):
    require(set(row) == set(S4K_PROJECTION), "S4K measurement projection")
    require(row["query_role"] == row["reference_role"] == "TRAIN", "sampled TRAIN pair")
    require(row["sampling_applied"] is True, "S4K sampling applied")
    query, reference = row["query_protein_id"], row["reference_protein_id"]
    require(isinstance(query, str) and query and isinstance(reference, str) and reference, "protein pair identity")
    require(query in proteins and reference in proteins, "protein pair absent from S4A ledger")
    observed = (
        row["query_node"], row["reference_node"],
        row["query_component_id"], row["reference_component_id"],
    )
    expected = (*proteins[query], *proteins[reference])
    expected = (expected[0], expected[2], expected[1], expected[3])
    require(all(type(value) is int and value >= 0 for value in observed), "pair node/component types")
    require(observed == expected and observed[0] != observed[1], "pair node/component identity")
    return (query, reference, *observed)


def ingest_pairs(source, connection, proteins, batch_rows):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(source)
    require(len(parquet.schema_arrow.names) == len(set(parquet.schema_arrow.names)), "unique S4K schema")
    require(set(S4K_PROJECTION) <= set(parquet.schema_arrow.names), "S4K projection availability")
    pending, source_rows = [], 0
    insert = "INSERT INTO activity_rows VALUES (?,?,?,?,?,?)"
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=S4K_PROJECTION):
        for row in batch.to_pylist():
            pending.append(validate_pair(row, proteins))
            source_rows += 1
            if len(pending) >= batch_rows:
                connection.executemany(insert, pending)
                connection.commit()
                pending = []
    if pending:
        connection.executemany(insert, pending)
        connection.commit()
    require(source_rows > 0, "empty S4K pair table")
    connection.execute("""
        CREATE TABLE protein_pairs AS
        SELECT query_protein_id,reference_protein_id,
               MIN(query_node) AS query_node,MIN(reference_node) AS reference_node,
               MIN(query_component_id) AS query_component_id,
               MIN(reference_component_id) AS reference_component_id,
               COUNT(*) AS activity_pair_rows
        FROM activity_rows GROUP BY query_protein_id,reference_protein_id
    """)
    inconsistent = connection.execute("""
        SELECT COUNT(*) FROM (
            SELECT query_protein_id,reference_protein_id FROM activity_rows
            GROUP BY query_protein_id,reference_protein_id
            HAVING MIN(query_node)<>MAX(query_node) OR MIN(reference_node)<>MAX(reference_node)
                OR MIN(query_component_id)<>MAX(query_component_id)
                OR MIN(reference_component_id)<>MAX(reference_component_id)
        )
    """).fetchone()[0]
    require(inconsistent == 0, "inconsistent repeated activity pair")
    connection.execute("CREATE UNIQUE INDEX protein_pair_key ON protein_pairs(query_protein_id,reference_protein_id)")
    connection.commit()
    return source_rows, connection.execute("SELECT COUNT(*) FROM protein_pairs").fetchone()[0]


def build_plan(connection, mmseqs, foldseek, shards):
    connection.execute("""
        CREATE TABLE plan (
            query_protein_id TEXT NOT NULL,
            reference_protein_id TEXT NOT NULL,
            query_node INTEGER NOT NULL,
            reference_node INTEGER NOT NULL,
            query_component_id INTEGER NOT NULL,
            reference_component_id INTEGER NOT NULL,
            activity_pair_rows INTEGER NOT NULL,
            mmseqs_query_db_key INTEGER NOT NULL,
            mmseqs_reference_db_key INTEGER NOT NULL,
            mmseqs_shard INTEGER NOT NULL,
            foldseek_query_db_key INTEGER,
            foldseek_reference_db_key INTEGER,
            foldseek_shard INTEGER,
            structure_availability TEXT NOT NULL
        )
    """)
    insert = "INSERT INTO plan VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    pending = []
    for row in connection.execute("""
        SELECT query_protein_id,reference_protein_id,query_node,reference_node,
               query_component_id,reference_component_id,activity_pair_rows
        FROM protein_pairs ORDER BY query_protein_id,reference_protein_id
    """):
        query, reference, query_node, reference_node, query_component, reference_component, count = row
        require(query_node in mmseqs and reference_node in mmseqs, "pair node absent from MMseqs reference database")
        state = core.structure_state(query_node, reference_node, foldseek)
        pending.append((
            query, reference, query_node, reference_node, query_component, reference_component, count,
            mmseqs[query_node], mmseqs[reference_node], core.shard("mmseqs", query_node, reference_node, shards),
            foldseek.get(query_node), foldseek.get(reference_node),
            core.shard("foldseek", query_node, reference_node, shards) if state == "BOTH_AVAILABLE" else None,
            state,
        ))
        if len(pending) >= 100000:
            connection.executemany(insert, pending)
            connection.commit()
            pending = []
    if pending:
        connection.executemany(insert, pending)
        connection.commit()
    planned = connection.execute("SELECT COUNT(*) FROM plan").fetchone()[0]
    require(planned > 0, "empty measurement plan")
    connection.execute("CREATE UNIQUE INDEX plan_protein_key ON plan(query_protein_id,reference_protein_id)")
    connection.execute("CREATE INDEX plan_mmseqs_shard ON plan(mmseqs_shard,mmseqs_query_db_key,mmseqs_reference_db_key)")
    connection.execute("CREATE INDEX plan_foldseek_shard ON plan(foldseek_shard,foldseek_query_db_key,foldseek_reference_db_key)")
    connection.commit()
    return planned


def manifest_schema():
    import pyarrow as pa
    return pa.schema([
        ("query_protein_id", pa.string()), ("reference_protein_id", pa.string()),
        ("query_node", pa.int32()), ("reference_node", pa.int32()),
        ("query_component_id", pa.int32()), ("reference_component_id", pa.int32()),
        ("activity_pair_rows", pa.int64()),
        ("mmseqs_query_db_key", pa.int64()), ("mmseqs_reference_db_key", pa.int64()),
        ("mmseqs_shard", pa.int16()),
        ("foldseek_query_db_key", pa.int64()), ("foldseek_reference_db_key", pa.int64()),
        ("foldseek_shard", pa.int16()), ("structure_availability", pa.string()),
    ])


class ManifestSink:
    def __init__(self, path, batch_rows):
        import pyarrow.parquet as pq
        self.schema, self.limit, self.rows, self.count = manifest_schema(), batch_rows, [], 0
        self.digest = hashlib.sha256()
        self.writer = pq.ParquetWriter(path, self.schema, compression="zstd")

    def add(self, row):
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
        self.digest.update(len(encoded).to_bytes(8, "big")); self.digest.update(encoded)
        self.rows.append(row)
        if len(self.rows) >= self.limit:
            self.flush()

    def flush(self):
        if self.rows:
            import pyarrow as pa
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.count += len(self.rows); self.rows = []

    def close(self):
        self.flush(); self.writer.close()


def write_manifest(connection, path, batch_rows):
    sink = ManifestSink(path, batch_rows)
    try:
        for values in connection.execute("SELECT " + ",".join(MANIFEST_COLUMNS) + " FROM plan ORDER BY query_protein_id,reference_protein_id"):
            sink.add(dict(zip(MANIFEST_COLUMNS, values)))
        sink.close()
    except BaseException:
        try:
            sink.writer.close()
        except BaseException:
            pass
        raise
    return sink.count, sink.digest.hexdigest()


def write_prefilter_shards(connection, output, modality, shards):
    directory = output / (modality + "_prefilter")
    directory.mkdir(exist_ok=False)
    query_field, reference_field, shard_field = (
        modality + "_query_db_key", modality + "_reference_db_key", modality + "_shard"
    )
    records, total = [], 0
    for shard_index in range(shards):
        path = directory / f"pairs_{shard_index:03d}.tsv"
        digest, count = hashlib.sha256(), 0
        sql = (
            f"SELECT DISTINCT {query_field},{reference_field} FROM plan WHERE {shard_field}=? "
            f"ORDER BY {query_field},{reference_field}"
        )
        with path.open("x", encoding="ascii", newline="\n") as handle:
            for query_key, reference_key in connection.execute(sql, (shard_index,)):
                require(type(query_key) is int and type(reference_key) is int, modality + " prefilter database keys")
                line = f"{query_key}\t{reference_key}\t2000\t0\n"
                handle.write(line); digest.update(line.encode("ascii")); count += 1
        records.append({"shard": shard_index, "rows": count, "sha256": digest.hexdigest(), "path": str(path)})
        total += count
    return records, total


def execute(s4k_pairs, protein_ledger, mmseqs_lookup, foldseek_selection, output, batch_rows=100000, shards=DEFAULT_SHARDS):
    s4k_pairs, protein_ledger, mmseqs_lookup, foldseek_selection = map(
        lambda value: Path(value).resolve(), (s4k_pairs, protein_ledger, mmseqs_lookup, foldseek_selection)
    )
    output = Path(output).resolve()
    require(output.parent.is_dir() and not output.exists(), "exclusive output")
    require(type(batch_rows) is int and not isinstance(batch_rows, bool) and batch_rows > 0, "batch rows")
    require(type(shards) is int and not isinstance(shards, bool) and shards > 0, "shards")
    paths = {
        "s4k_pairs": s4k_pairs, "protein_ledger": protein_ledger,
        "mmseqs_lookup": mmseqs_lookup, "foldseek_selection": foldseek_selection,
    }
    inputs = {name: input_record(path) for name, path in paths.items()}
    output.mkdir(exist_ok=False)
    emit(output / "reservation.json", {
        "status": "ONE_S4L_DIRECT_PAIR_MEASUREMENT_PLAN_ATTEMPT_RESERVED",
        "inputs": inputs, "parameters": {"seed": SEED, "shards_per_modality": shards},
        "automatic_retry": False,
    })
    connection, error, state, summary = None, None, "FAIL_CLOSED", None
    try:
        proteins, ledger_proteins = read_proteins(protein_ledger)
        mmseqs = read_mmseqs_lookup(mmseqs_lookup)
        foldseek = read_foldseek_mapping(foldseek_selection)
        database = output / "direct_pair_measurement_index.sqlite"
        connection = connect_database(database)
        activity_rows, protein_pairs = ingest_pairs(s4k_pairs, connection, proteins, batch_rows)
        require(build_plan(connection, mmseqs, foldseek, shards) == protein_pairs, "measurement plan census")
        manifest = output / "direct_pair_measurement_manifest.parquet"
        written, row_digest = write_manifest(connection, manifest, batch_rows)
        require(written == protein_pairs, "manifest census")
        mm_records, mm_pairs = write_prefilter_shards(connection, output, "mmseqs", shards)
        fs_records, fs_pairs = write_prefilter_shards(connection, output, "foldseek", shards)
        node_pair_counts = {
            "mmseqs": connection.execute("SELECT COUNT(*) FROM (SELECT DISTINCT query_node,reference_node FROM plan)").fetchone()[0],
            "foldseek": connection.execute("SELECT COUNT(*) FROM (SELECT DISTINCT query_node,reference_node FROM plan WHERE foldseek_shard IS NOT NULL)").fetchone()[0],
        }
        require(mm_pairs == node_pair_counts["mmseqs"] and fs_pairs == node_pair_counts["foldseek"], "unique node-pair prefilter census")
        states = dict(connection.execute("SELECT structure_availability,COUNT(*) FROM plan GROUP BY structure_availability").fetchall())
        require(sum(states.values()) == protein_pairs, "structure availability census")
        require(all(sha(paths[name]) == record["sha256"] for name, record in inputs.items()), "input stability")
        summary = {
            "status": "PASS_S4L_DIRECT_PAIR_MEASUREMENT_PLAN_PENDING_INDEPENDENT_AUDIT",
            "inputs": inputs, "seed": SEED, "shards_per_modality": shards,
            "s4a_ledger_proteins": ledger_proteins, "s4a_train_proteins": len(proteins),
            "s4k_activity_pair_rows": activity_rows, "unique_protein_pairs": protein_pairs,
            "unique_mmseqs_node_pairs": mm_pairs, "unique_foldseek_node_pairs": fs_pairs,
            "structure_availability": dict(sorted(states.items())),
            "manifest_row_sha256": row_digest, "manifest_file_sha256": sha(manifest),
            "prefilter_shards": {"mmseqs": mm_records, "foldseek": fs_records},
            "protein_pair_manifest_complete": True, "exact_node_pair_measured_once_per_modality": True,
            "s4k_columns_read": S4K_PROJECTION, "functional_labels_read": False,
            "query_truth_read": False, "sampling_probabilities_read": False,
            "training_weights_read": False, "retrieval_metrics_read": False,
            "direct_alignment_started": False, "feature_matrix_created": False,
            "preprocessing_started": False, "training_started": False,
        }
        emit(output / "producer_summary.json", summary)
        state = summary["status"]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if connection is not None:
            connection.close()
    emit(output / "terminal.json", {"status": state, "error": error, "automatic_retry": False})
    if error:
        raise RuntimeError(error["message"])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s4k-pairs", type=Path, required=True)
    parser.add_argument("--protein-ledger", type=Path, required=True)
    parser.add_argument("--mmseqs-lookup", type=Path, required=True)
    parser.add_argument("--foldseek-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=100000)
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    args = parser.parse_args()
    execute(args.s4k_pairs, args.protein_ledger, args.mmseqs_lookup, args.foldseek_selection, args.output, args.batch_rows, args.shards)


if __name__ == "__main__":
    main()
