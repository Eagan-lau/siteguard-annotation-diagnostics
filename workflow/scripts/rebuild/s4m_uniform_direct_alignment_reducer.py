"""Production S4M reducer with complete S4L node-pair reconciliation."""
import argparse
import hashlib
import json
import math
import sqlite3
import traceback
from collections import Counter
from pathlib import Path


SEED = 20260819
S4L_MANIFEST_COLUMNS = [
    "query_protein_id", "reference_protein_id", "query_node", "reference_node",
    "query_component_id", "reference_component_id", "activity_pair_rows",
    "mmseqs_query_db_key", "mmseqs_reference_db_key", "mmseqs_shard",
    "foldseek_query_db_key", "foldseek_reference_db_key", "foldseek_shard",
    "structure_availability",
]
BASE_METRICS = (
    "fident", "alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen",
    "qcov", "tcov", "evalue", "bits",
)
FOLD_METRICS = ("lddt", "qtmscore", "ttmscore", "alntmscore", "rmsd", "prob")
INTEGER_METRICS = {"alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen"}
FRACTION_METRICS = {"fident", "qcov", "tcov", "lddt", "qtmscore", "ttmscore", "prob"}


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
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")


def read_json(path):
    require(path.is_file() and not path.is_symlink(), "required JSON " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def record(path):
    require(path.is_file() and not path.is_symlink(), "required regular file " + path.name)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def stable_shard(modality, query_node, reference_node, shards):
    payload = f"{SEED}|{modality}|{query_node}|{reference_node}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % shards


def worker_name(task, modality, shard):
    return f"{task:03d}_{modality}_{shard:03d}"


def connect(path):
    require(not path.exists(), "reducer database exists")
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-262144")
    db.execute("CREATE TABLE expected_mm(query_node INTEGER,reference_node INTEGER,shard INTEGER,PRIMARY KEY(query_node,reference_node))")
    db.execute("CREATE TABLE expected_fs(query_node INTEGER,reference_node INTEGER,shard INTEGER,PRIMARY KEY(query_node,reference_node))")
    db.execute("""
        CREATE TABLE observed_mm(
            query_node INTEGER,reference_node INTEGER,query_db_key INTEGER,reference_db_key INTEGER,shard INTEGER,
            fident REAL,alnlen INTEGER,qstart INTEGER,qend INTEGER,qlen INTEGER,tstart INTEGER,tend INTEGER,tlen INTEGER,
            qcov REAL,tcov REAL,evalue REAL,bits REAL,PRIMARY KEY(query_node,reference_node)
        )
    """)
    db.execute("""
        CREATE TABLE observed_fs(
            query_node INTEGER,reference_node INTEGER,query_db_key INTEGER,reference_db_key INTEGER,shard INTEGER,
            fident REAL,alnlen INTEGER,qstart INTEGER,qend INTEGER,qlen INTEGER,tstart INTEGER,tend INTEGER,tlen INTEGER,
            qcov REAL,tcov REAL,evalue REAL,bits REAL,lddt REAL,qtmscore REAL,ttmscore REAL,alntmscore REAL,rmsd REAL,prob REAL,
            PRIMARY KEY(query_node,reference_node)
        )
    """)
    return db


def index_manifest(plan, db, shards, batch_rows):
    import pyarrow.parquet as pq
    manifest = plan / "direct_pair_measurement_manifest.parquet"
    summary = read_json(plan / "producer_summary.json")
    terminal = read_json(plan / "terminal.json")
    require(summary["status"] == terminal["status"] == "PASS_S4L_DIRECT_PAIR_MEASUREMENT_PLAN_PENDING_INDEPENDENT_AUDIT", "S4L producer status")
    require(terminal["error"] is None and not terminal["automatic_retry"], "S4L producer terminal")
    require(summary["shards_per_modality"] == shards and summary["seed"] == SEED, "S4L shard parameters")
    parquet = pq.ParquetFile(manifest)
    require(parquet.schema_arrow.names == S4L_MANIFEST_COLUMNS, "S4L manifest schema")
    mm_rows, fs_rows, pair_rows, states = [], [], 0, Counter()
    columns = ["query_node", "reference_node", "mmseqs_shard", "foldseek_shard", "structure_availability"]
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
        for row in batch.to_pylist():
            qnode, rnode = row["query_node"], row["reference_node"]
            require(type(qnode) is int and type(rnode) is int and qnode != rnode, "S4L node pair")
            require(row["mmseqs_shard"] == stable_shard("mmseqs", qnode, rnode, shards), "S4L MMseqs shard")
            mm_rows.append((qnode, rnode, row["mmseqs_shard"]))
            state = row["structure_availability"]
            require(state in {"BOTH_AVAILABLE", "QUERY_UNAVAILABLE", "REFERENCE_UNAVAILABLE", "BOTH_UNAVAILABLE"}, "structure availability")
            states[state] += 1
            if state == "BOTH_AVAILABLE":
                require(row["foldseek_shard"] == stable_shard("foldseek", qnode, rnode, shards), "S4L Foldseek shard")
                fs_rows.append((qnode, rnode, row["foldseek_shard"]))
            else:
                require(row["foldseek_shard"] is None, "unavailable Foldseek shard")
            pair_rows += 1
            if len(mm_rows) >= batch_rows:
                db.executemany("INSERT OR IGNORE INTO expected_mm VALUES (?,?,?)", mm_rows); mm_rows = []
            if len(fs_rows) >= batch_rows:
                db.executemany("INSERT OR IGNORE INTO expected_fs VALUES (?,?,?)", fs_rows); fs_rows = []
    if mm_rows:
        db.executemany("INSERT OR IGNORE INTO expected_mm VALUES (?,?,?)", mm_rows)
    if fs_rows:
        db.executemany("INSERT OR IGNORE INTO expected_fs VALUES (?,?,?)", fs_rows)
    db.commit()
    mm_count = db.execute("SELECT COUNT(*) FROM expected_mm").fetchone()[0]
    fs_count = db.execute("SELECT COUNT(*) FROM expected_fs").fetchone()[0]
    require(pair_rows == summary["unique_protein_pairs"], "S4L protein-pair census")
    require(mm_count == summary["unique_mmseqs_node_pairs"] and fs_count == summary["unique_foldseek_node_pairs"], "S4L node-pair census")
    require(dict(sorted(states.items())) == summary["structure_availability"], "S4L structure census")
    require(sha(manifest) == summary["manifest_file_sha256"], "S4L manifest file binding")
    return summary, record(manifest), pair_rows, mm_count, fs_count, dict(sorted(states.items()))


def result_schema(modality):
    import pyarrow as pa
    fields = [
        ("modality", pa.string()), ("shard", pa.int16()),
        ("query_node", pa.int32()), ("reference_node", pa.int32()),
        ("query_db_key", pa.int64()), ("reference_db_key", pa.int64()),
        ("fident", pa.float64()), ("alnlen", pa.int64()),
        ("qstart", pa.int64()), ("qend", pa.int64()), ("qlen", pa.int64()),
        ("tstart", pa.int64()), ("tend", pa.int64()), ("tlen", pa.int64()),
        ("qcov", pa.float64()), ("tcov", pa.float64()),
        ("evalue", pa.float64()), ("bits", pa.float64()),
    ]
    if modality == "foldseek":
        fields.extend((name, pa.float64()) for name in FOLD_METRICS)
    return pa.schema(fields)


def validate_metric(name, value, modality="mmseqs"):
    if name in INTEGER_METRICS:
        require(type(value) is int and value >= 0, "integer metric " + name)
    else:
        require(isinstance(value, (int, float)) and math.isfinite(value), "finite metric " + name)
        if name in FRACTION_METRICS:
            require(0 <= value <= 1, "fraction metric " + name)
        if name in {"evalue", "rmsd"} or (name == "bits" and modality != "foldseek"):
            require(value >= 0, "nonnegative metric " + name)


def validate_worker(worker, audit, plan, plan_summary, modality, shard, task):
    unit = {"modality": modality, "shard": shard}
    require(worker.is_dir() and not worker.is_symlink() and audit.is_dir() and not audit.is_symlink(), "worker/audit directories")
    summary = read_json(worker / "worker_summary.json")
    terminal = read_json(worker / "terminal.json")
    report = read_json(audit / "independent_audit.json")
    audit_terminal = read_json(audit / "terminal.json")
    checkpoint = read_json(audit / "S4M_DIRECT_ALIGNMENT_WORKER_PASS.json")
    require(summary["unit"] == report["unit"] == checkpoint["unit"] == unit, "worker unit identity")
    require(summary["status"] == terminal["status"] == "PASS_S4M_DIRECT_ALIGNMENT_WORKER_PENDING_INDEPENDENT_AUDIT", "worker status")
    require(terminal["error"] is None and not terminal["automatic_retry"], "worker terminal")
    require(report["status"] == audit_terminal["status"] == "PASS_S4M_DIRECT_ALIGNMENT_WORKER_INDEPENDENT_AUDIT", "worker audit status")
    require(audit_terminal["error"] is None and not audit_terminal["automatic_retry"], "worker audit terminal")
    require(checkpoint["status"] == "PASS_S4M_DIRECT_ALIGNMENT_WORKER_AUDIT" and checkpoint["audit_sha256"] == sha(audit / "independent_audit.json"), "worker audit checkpoint")
    require(checkpoint["producer_summary_sha256"] == sha(worker / "worker_summary.json") == report["producer"]["summary_sha256"], "worker summary binding")
    require(Path(report["producer"]["path"]).resolve() == worker, "worker audit producer path")
    prefilter = plan / (modality + "_prefilter") / f"pairs_{shard:03d}.tsv"
    prefilter_record = record(prefilter)
    require(summary["inputs"]["prefilter"] == report["inputs"]["prefilter"] == prefilter_record, "worker prefilter binding")
    metadata = plan_summary["prefilter_shards"][modality][shard]
    require(metadata["shard"] == shard and metadata["sha256"] == prefilter_record["sha256"], "S4L prefilter metadata binding")
    require(metadata["rows"] == summary["requested_node_pairs"], "S4L prefilter row binding")
    require(summary["normalized_alignment_rows"] == report["normalized_alignment_rows"] == summary["requested_node_pairs"] == report["requested_node_pairs"], "worker row census")
    require(summary["normalized_row_sha256"] == report["normalized_row_sha256"], "worker row digest")
    normalized = worker / "normalized_alignments.parquet"
    require(sha(normalized) == summary["normalized_file_sha256"] == report["producer_normalized_file_sha256"], "worker normalized file binding")
    require(report["full_raw_to_parquet_reconstruction"] and report["command_receipt_reconstructed"] and not report["imports_producer_or_semantic_core"], "worker audit independence")
    require(summary["direct_known_pair_measurement"] and not summary["candidate_search_performed"] and not report["candidate_search_performed"], "direct measurement semantics")
    for owner in (summary, report):
        for name in ("functional_labels_read", "query_truth_read", "sampling_probabilities_read", "training_weights_read", "retrieval_metrics_read", "feature_matrix_created", "preprocessing_started", "training_started", "evaluation_started"):
            require(owner[name] is False, "worker prohibited state " + name)
    return summary, report, normalized


def ingest_worker(path, db, modality, shard, batch_rows):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    require(parquet.schema_arrow == result_schema(modality), "worker result schema")
    metrics = BASE_METRICS + (FOLD_METRICS if modality == "foldseek" else ())
    columns = ["query_node", "reference_node", "query_db_key", "reference_db_key", "shard", *metrics]
    table = "observed_mm" if modality == "mmseqs" else "observed_fs"
    insert_columns = ["query_node", "reference_node", "query_db_key", "reference_db_key", "shard", *metrics]
    insert = f"INSERT INTO {table}({','.join(insert_columns)}) VALUES ({','.join('?' for _ in insert_columns)})"
    pending, count = [], 0
    for batch in parquet.iter_batches(batch_size=batch_rows):
        for row in batch.to_pylist():
            require(row["modality"] == modality and row["shard"] == shard, "worker result unit")
            require(type(row["query_node"]) is int and type(row["reference_node"]) is int and row["query_node"] != row["reference_node"], "worker node pair")
            require(type(row["query_db_key"]) is int and type(row["reference_db_key"]) is int, "worker database keys")
            for name in metrics:
                validate_metric(name, row[name], modality)
            if modality == "foldseek":
                validate_aligned_tm(row)
            pending.append(tuple(row[name] for name in columns)); count += 1
            if len(pending) >= batch_rows:
                try:
                    db.executemany(insert, pending)
                except sqlite3.IntegrityError as exc:
                    raise RuntimeError("duplicate observed node pair") from exc
                db.commit(); pending = []
    if pending:
        try:
            db.executemany(insert, pending)
        except sqlite3.IntegrityError as exc:
            raise RuntimeError("duplicate observed node pair") from exc
        db.commit()
    return count


def reconcile(db, modality):
    expected, observed = ("expected_mm", "observed_mm") if modality == "mmseqs" else ("expected_fs", "observed_fs")
    missing = db.execute(f"SELECT COUNT(*) FROM {expected} e LEFT JOIN {observed} o USING(query_node,reference_node) WHERE o.query_node IS NULL").fetchone()[0]
    unexpected = db.execute(f"SELECT COUNT(*) FROM {observed} o LEFT JOIN {expected} e USING(query_node,reference_node) WHERE e.query_node IS NULL").fetchone()[0]
    wrong_shard = db.execute(f"SELECT COUNT(*) FROM {observed} o JOIN {expected} e USING(query_node,reference_node) WHERE o.shard<>e.shard").fetchone()[0]
    require(missing == unexpected == wrong_shard == 0, modality + " complete node-pair reconciliation")
    return db.execute(f"SELECT COUNT(*) FROM {observed}").fetchone()[0]


def write_consolidated(db, modality, path, batch_rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = "observed_mm" if modality == "mmseqs" else "observed_fs"
    metrics = BASE_METRICS + (FOLD_METRICS if modality == "foldseek" else ())
    select_columns = ["shard", "query_node", "reference_node", "query_db_key", "reference_db_key", *metrics]
    schema = result_schema(modality)
    writer = pq.ParquetWriter(path, schema, compression="zstd")
    rows, count, digest = [], 0, hashlib.sha256()
    try:
        for values in db.execute(f"SELECT {','.join(select_columns)} FROM {table} ORDER BY query_node,reference_node"):
            source = dict(zip(select_columns, values))
            row = {"modality": modality, **source}
            encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big")); digest.update(encoded)
            rows.append(row)
            if len(rows) >= batch_rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema)); count += len(rows); rows = []
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema)); count += len(rows)
        writer.close()
    except BaseException:
        try:
            writer.close()
        except BaseException:
            pass
        raise
    return count, digest.hexdigest(), sha(path)


def execute(s4l_plan, workers, audits, output, shards=64, batch_rows=100000):
    s4l_plan, workers, audits, output = map(lambda value: Path(value).resolve(), (s4l_plan, workers, audits, output))
    require(s4l_plan.is_dir() and workers.is_dir() and audits.is_dir(), "reducer input directories")
    require(output.parent.is_dir() and not output.exists(), "exclusive reducer output")
    require(type(shards) is int and not isinstance(shards, bool) and shards > 0, "shards")
    require(type(batch_rows) is int and not isinstance(batch_rows, bool) and batch_rows > 0, "batch rows")
    output.mkdir(exist_ok=False)
    emit(output / "reservation.json", {
        "status": "ONE_S4M_DIRECT_ALIGNMENT_REDUCER_ATTEMPT_RESERVED",
        "s4l_plan": {"path": str(s4l_plan)}, "workers": str(workers), "audits": str(audits),
        "parameters": {"seed": SEED, "shards_per_modality": shards}, "automatic_retry": False,
    })
    db, state, error, summary = None, "FAIL_CLOSED", None, None
    try:
        db = connect(output / "reducer_index.sqlite")
        plan_summary, manifest_record, protein_pairs, expected_mm, expected_fs, states = index_manifest(s4l_plan, db, shards, batch_rows)
        units, identity = [], {"mmseqs": set(), "foldseek": set()}
        expected_counts = {"mmseqs": expected_mm, "foldseek": expected_fs}
        ingested_counts = {"mmseqs": 0, "foldseek": 0}
        expected_worker_names, expected_audit_names = set(), set()
        for task in range(2 * shards):
            modality = "mmseqs" if task < shards else "foldseek"
            shard = task if modality == "mmseqs" else task - shards
            name = worker_name(task, modality, shard)
            expected_worker_names.add("worker_" + name); expected_audit_names.add("audit_" + name)
        require({path.name for path in workers.iterdir() if path.is_dir()} == {n for n in expected_worker_names if "_foldseek_" in n}, "exact recovery Foldseek worker directory set")
        require({path.name for path in audits.iterdir() if path.is_dir()} == {n for n in expected_audit_names if "_foldseek_" in n}, "exact recovery Foldseek audit directory set")
        for task in range(2 * shards):
            modality = "mmseqs" if task < shards else "foldseek"
            shard = task if modality == "mmseqs" else task - shards
            name = worker_name(task, modality, shard)
            worker, audit = selected_worker(workers, name, modality), selected_audit(audits, name, modality)
            worker_summary, report, normalized = validate_worker(worker, audit, s4l_plan, plan_summary, modality, shard, task)
            count = ingest_worker(normalized, db, modality, shard, batch_rows)
            require(count == worker_summary["normalized_alignment_rows"], "worker ingestion census")
            ingested_counts[modality] += count
            identity[modality].add(tuple(worker_summary["inputs"][field]["sha256"] for field in ("mapping", "database_dbtype", "tool")))
            units.append({
                "task": task, "modality": modality, "shard": shard, "rows": count,
                "worker_summary_sha256": sha(worker / "worker_summary.json"),
                "audit_sha256": sha(audit / "independent_audit.json"),
                "normalized_file_sha256": sha(normalized),
            })
        require(all(len(identity[name]) == 1 for name in identity), "one frozen mapping/database/tool identity per modality")
        mm_count, fs_count = reconcile(db, "mmseqs"), reconcile(db, "foldseek")
        require(mm_count == ingested_counts["mmseqs"] == expected_counts["mmseqs"], "MMseqs total census")
        require(fs_count == ingested_counts["foldseek"] == expected_counts["foldseek"], "Foldseek total census")
        mm = output / "direct_mmseqs_node_alignments.parquet"
        fs = output / "direct_foldseek_node_alignments.parquet"
        mm_written, mm_digest, mm_sha = write_consolidated(db, "mmseqs", mm, batch_rows)
        fs_written, fs_digest, fs_sha = write_consolidated(db, "foldseek", fs, batch_rows)
        require(mm_written == mm_count and fs_written == fs_count, "consolidated census")
        summary = {
            "status": "PASS_S4M_UNIFORM_DIRECT_ALIGNMENT_REDUCER_PENDING_INDEPENDENT_AUDIT",
            "s4l_plan": {"path": str(s4l_plan), "summary_sha256": sha(s4l_plan / "producer_summary.json"), "manifest": manifest_record},
            "shards_per_modality": shards, "unit_count": len(units), "units": units,
            "protein_pair_manifest_rows": protein_pairs, "structure_availability": states,
            "direct_mmseqs_node_pairs": mm_count, "direct_foldseek_node_pairs": fs_count,
            "mmseqs_row_sha256": mm_digest, "foldseek_row_sha256": fs_digest,
            "mmseqs_file_sha256": mm_sha, "foldseek_file_sha256": fs_sha,
            "complete_node_pair_reconciliation": True, "exact_node_pair_measured_once_per_modality": True,
            "structure_unavailable_imputed_as_zero": False, "candidate_search_performed": False,
            "functional_labels_read": False, "query_truth_read": False,
            "sampling_probabilities_read": False, "training_weights_read": False,
            "retrieval_metrics_read": False, "feature_matrix_created": False,
            "preprocessing_started": False, "training_started": False, "evaluation_started": False,
        }
        emit(output / "reducer_summary.json", summary); state = summary["status"]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if db is not None:
            db.close()
    emit(output / "terminal.json", {"status": state, "error": error, "automatic_retry": False})
    if error:
        raise RuntimeError(error["message"])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s4l-plan", type=Path, required=True)
    parser.add_argument("--workers", type=Path, required=True)
    parser.add_argument("--audits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--batch-rows", type=int, default=100000)
    args = parser.parse_args()
    execute(args.s4l_plan, args.workers, args.audits, args.output, args.shards, args.batch_rows)



# The original passed MMseqs evidence is read in place, never copied or relabeled.
LEGACY_ROOT = Path("/globalsc/ulg/plgen/yugenliu/SiteGuard/V4/revisions/s4_20260910/uniform_direct_alignment_attempt_01").resolve()

def selected_worker(workers, name, modality):
    return (LEGACY_ROOT / "workers" if modality == "mmseqs" else workers) / ("worker_" + name)

def selected_audit(audits, name, modality):
    return (LEGACY_ROOT / "worker_audits" if modality == "mmseqs" else audits) / ("audit_" + name)

def validate_aligned_tm(row):
    """Foldseek 10-941cd33 endpoint-span normalization, retaining raw value."""
    span = min(row["qend"] - row["qstart"], row["tend"] - row["tstart"])
    require(span > 0, "positive aligned-TM normalization span")
    # SSTR exports four significant digits; values in [1,2] round by at most 0.0005.
    require(0 <= row["alntmscore"] <= (span + 1) / span + 0.000500000001,
             "version-specific aligned-TM bound")

if __name__ == "__main__":
    main()

