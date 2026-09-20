"""Production S4K balanced TRAIN sampler with retained SQLite spill state."""
import argparse
import hashlib
import json
import math
import sqlite3
import traceback
from pathlib import Path


SEED = 20260819
TARGET_PAIRS = 1_500_000
MINIMUM_PAIRS = 500_000
MAX_PER_EXACT_RHEA = 5_000
MAX_PER_EC_L4 = 10_000
DIFFICULT_TARGET_FRACTION = 0.5
KEY_FIELDS = ("query_protein_id", "reference_protein_id", "reference_activity_id")
RAW_METRICS = ("fident", "alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen", "qcov", "tcov", "evalue", "bits")
COMMON_FIELDS = [
    "query_protein_id", "query_node", "query_component_id", "query_role", "reference_node",
    "reference_component_id", "reference_protein_id", "reference_activity_id", "canonical_ec", "ec_l1",
    "ec_l2", "ec_l3", "ec_l4", "canonical_rhea", "evidence_tier",
    "query_documented_activity_count", "query_documented_ec_l3_count", "query_documented_ec_l4_count",
    "query_documented_rhea_count", "observed_same_ec_l3", "observed_same_ec_l4",
    "exact_rhea_outcome_evaluable", "observed_same_exact_rhea", "deepest_shared_recorded_ec_level",
    "outcome_semantics", "ground_truth_only",
]
RETRIEVAL_FEATURE_FIELDS = [
    "mmseqs_rank", "foldseek_rank", "mmseqs_raw_rank", "foldseek_raw_rank",
] + [f"{modality}_{name}" for modality in ("mmseqs", "foldseek") for name in RAW_METRICS] + [
    "source_class", "rrf60_score", "rrf_constant",
]
INPUT_COLUMNS = COMMON_FIELDS + [
    "reference_role", "query_primary_pfam", "reference_primary_pfam", "pair_origin", "retrieval_present",
] + RETRIEVAL_FEATURE_FIELDS + [
    "retrieval_candidate_union_provenance", "retrieval_query_truth_read", "retrieval_activity_provenance",
    "retrieval_outcome_truth_provenance", "augmentation_present", "augmentation_mechanisms",
    "augmentation_design_inclusion_probability", "augmentation_probability_is_population_weight",
    "augmentation_candidate_origin", "augmentation_sampling_truth_provenance",
    "augmentation_outcome_truth_provenance", "hard_case_flags_available_before_features",
    "reference_activity_frequency", "query_family_frequency", "reference_component_frequency",
    "sampling_applied", "sample_weight",
]
SAMPLING_COLUMNS = [
    "reference_cluster_dedup_probability", "exact_rhea_cap_probability", "ec_l4_cap_probability",
    "difficulty_stratum_probability", "sampling_probability", "selection_probability_is_population_ipw",
    "sampling_stratum", "hard_case_flags_for_sampling", "activity_balance_weight", "family_balance_weight",
    "reference_cluster_weight",
]
METADATA_COLUMNS = list(dict.fromkeys([
    *KEY_FIELDS, "query_role", "reference_role", "query_node", "reference_node", "reference_component_id",
    "canonical_rhea", "ec_l4", "observed_same_ec_l4", "exact_rhea_outcome_evaluable",
    "observed_same_exact_rhea", "ground_truth_only", "outcome_semantics", "query_primary_pfam",
    "mmseqs_fident", "mmseqs_qcov", "mmseqs_tcov", "foldseek_rank",
    "hard_case_flags_available_before_features", "sampling_applied", "sample_weight",
    "augmentation_probability_is_population_weight",
]))


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


def priority(row, stage, seed=SEED):
    payload = "|".join([str(seed), stage] + [str(row[name]) for name in KEY_FIELDS])
    return hashlib.sha256(payload.encode("utf-8")).digest()


def available_flags(row):
    existing = row["hard_case_flags_available_before_features"]
    require(isinstance(existing, str) and existing, "S4J hard-case flag encoding")
    flags = set() if existing == "NONE_AT_PAIR_UNION_STAGE" else set(existing.split("+"))
    require(all(flag.startswith(("H1_", "H2_", "H5_")) for flag in flags), "unexpected pre-feature hard-case flag")
    fine_divergence = not row["observed_same_ec_l4"] or (
        row["exact_rhea_outcome_evaluable"] and not row["observed_same_exact_rhea"]
    )
    identity, qcov, tcov = row["mmseqs_fident"], row["mmseqs_qcov"], row["mmseqs_tcov"]
    metrics = (identity, qcov, tcov)
    require(all(value is None or (isinstance(value, (int, float)) and math.isfinite(value)) for value in metrics), "finite MMseqs metrics")
    require(identity is None or 0 <= identity <= 1, "MMseqs identity range")
    require(qcov is None or 0 <= qcov <= 1, "MMseqs qcov range")
    require(tcov is None or 0 <= tcov <= 1, "MMseqs tcov range")
    adequate = identity is not None and qcov is not None and tcov is not None and qcov >= 0.70 and tcov >= 0.70
    if adequate and identity >= 0.40 and fine_divergence:
        flags.add("H3_HIGH_SEQUENCE_IDENTITY_FINE_DIVERGENCE")
    foldseek_rank = row["foldseek_rank"]
    require(foldseek_rank is None or (type(foldseek_rank) is int and foldseek_rank > 0), "Foldseek rank")
    if foldseek_rank is not None and foldseek_rank <= 10 and fine_divergence:
        flags.add("H4_FOLDSEEK_TOP10_FINE_DIVERGENCE_PROXY")
    if adequate and identity < 0.30 and row["exact_rhea_outcome_evaluable"] and row["observed_same_exact_rhea"]:
        flags.add("H6_LOW_SEQUENCE_IDENTITY_SAME_EXACT_RHEA")
    return "+".join(sorted(flags)) or "NONE_AVAILABLE_BEFORE_GLOBAL_FEATURES"


def validate_metadata(row, previous_key):
    key = tuple(row[name] for name in KEY_FIELDS)
    require(all(isinstance(value, str) and value for value in key), "pair identity")
    require(previous_key is None or key > previous_key, "strict S4J pair order")
    require(row["query_role"] == row["reference_role"] == "TRAIN", "TRAIN pair scope")
    require(type(row["query_node"]) is int and type(row["reference_node"]) is int and row["query_node"] != row["reference_node"], "sequence-node separation")
    require(type(row["reference_component_id"]) is int, "reference component")
    require(isinstance(row["ec_l4"], str) and row["ec_l4"], "EC-L4 group")
    require(type(row["observed_same_ec_l4"]) is bool and type(row["exact_rhea_outcome_evaluable"]) is bool, "outcome types")
    if row["exact_rhea_outcome_evaluable"]:
        require(type(row["observed_same_exact_rhea"]) is bool and isinstance(row["canonical_rhea"], str) and row["canonical_rhea"], "evaluable exact Rhea")
    else:
        require(row["observed_same_exact_rhea"] is None, "nullable exact Rhea outcome")
    require(row["ground_truth_only"] is True and row["outcome_semantics"] == "DOCUMENTED_CONCORDANCE_NOT_BIOCHEMICAL_NEGATIVE", "outcome semantics")
    require(row["sampling_applied"] is False and row["sample_weight"] is None, "unsampled S4J input")
    require(row["augmentation_probability_is_population_weight"] is False, "augmentation probability semantics")
    return key


def output_schema(input_schema):
    import pyarrow as pa
    require(input_schema.names == INPUT_COLUMNS, "S4J input schema")
    extra_types = {
        "reference_cluster_dedup_probability": pa.float64(), "exact_rhea_cap_probability": pa.float64(),
        "ec_l4_cap_probability": pa.float64(), "difficulty_stratum_probability": pa.float64(),
        "sampling_probability": pa.float64(), "selection_probability_is_population_ipw": pa.bool_(),
        "sampling_stratum": pa.string(), "hard_case_flags_for_sampling": pa.string(),
        "activity_balance_weight": pa.float64(), "family_balance_weight": pa.float64(),
        "reference_cluster_weight": pa.float64(),
    }
    return pa.schema(list(input_schema) + [pa.field(name, extra_types[name]) for name in SAMPLING_COLUMNS])


def connect_database(path):
    require(not path.exists(), "selection database exists")
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-262144")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("""
        CREATE TABLE candidates (
            row_id INTEGER PRIMARY KEY,
            query_protein_id TEXT NOT NULL,
            reference_protein_id TEXT NOT NULL,
            reference_activity_id TEXT NOT NULL,
            reference_component_id INTEGER NOT NULL,
            canonical_rhea TEXT,
            ec_l4 TEXT NOT NULL,
            query_family TEXT,
            hard_flags TEXT NOT NULL,
            stratum TEXT NOT NULL CHECK(stratum IN ('DIFFICULT','ORDINARY')),
            priority_cluster BLOB NOT NULL,
            priority_rhea BLOB NOT NULL,
            priority_ec4 BLOB NOT NULL,
            priority_target BLOB NOT NULL
        )
    """)
    return connection


def ingest_metadata(source, connection, batch_rows):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(source)
    require(parquet.schema_arrow.names == INPUT_COLUMNS, "S4J input schema")
    insert = "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    previous, row_id, pending = None, 0, []
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=METADATA_COLUMNS):
        for row in batch.to_pylist():
            key = validate_metadata(row, previous)
            previous = key
            flags = available_flags(row)
            stratum = "ORDINARY" if flags == "NONE_AVAILABLE_BEFORE_GLOBAL_FEATURES" else "DIFFICULT"
            cluster_stage = "REFERENCE_CLUSTER|" + "|".join(map(str, (row["query_protein_id"], row["reference_component_id"], row["reference_activity_id"])))
            pending.append((
                row_id, *key, row["reference_component_id"], row["canonical_rhea"], row["ec_l4"],
                row["query_primary_pfam"], flags, stratum, priority(row, cluster_stage),
                priority(row, "RHEA_CAP|" + str(row["canonical_rhea"])),
                priority(row, "EC4_CAP|" + row["ec_l4"]), priority(row, "TARGET_STRATUM|" + stratum),
            ))
            row_id += 1
            if len(pending) >= batch_rows:
                connection.executemany(insert, pending)
                connection.commit()
                pending = []
    if pending:
        connection.executemany(insert, pending)
        connection.commit()
    require(row_id > 0, "empty S4J union")
    return row_id


BASE_SQL_COLUMNS = "row_id,query_protein_id,reference_protein_id,reference_activity_id,reference_component_id,canonical_rhea,ec_l4,query_family,hard_flags,stratum,priority_cluster,priority_rhea,priority_ec4,priority_target"


def count_table(connection, table):
    return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def build_selection(connection, target_pairs, minimum_pairs, max_per_rhea, max_per_ec4):
    require(type(target_pairs) is int and not isinstance(target_pairs, bool) and target_pairs > 0, "target pairs")
    require(type(minimum_pairs) is int and not isinstance(minimum_pairs, bool) and 0 <= minimum_pairs <= target_pairs, "minimum pairs")
    require(type(max_per_rhea) is int and not isinstance(max_per_rhea, bool) and max_per_rhea > 0, "Rhea cap")
    require(type(max_per_ec4) is int and not isinstance(max_per_ec4, bool) and max_per_ec4 > 0, "EC-L4 cap")
    connection.execute("""
        CREATE TABLE cluster_stage AS
        SELECT row_id,query_protein_id,reference_protein_id,reference_activity_id,reference_component_id,
               canonical_rhea,ec_l4,query_family,hard_flags,stratum,priority_cluster,priority_rhea,
               priority_ec4,priority_target,1.0/group_n AS p_cluster
        FROM (
            SELECT candidates.*,
                   COUNT(*) OVER (PARTITION BY query_protein_id,reference_component_id,reference_activity_id) AS group_n,
                   ROW_NUMBER() OVER (
                       PARTITION BY query_protein_id,reference_component_id,reference_activity_id
                       ORDER BY priority_cluster,query_protein_id,reference_protein_id,reference_activity_id
                   ) AS selection_rank
            FROM candidates
        ) WHERE selection_rank=1
    """)
    connection.execute("""
        CREATE TABLE rhea_stage AS
        SELECT row_id,query_protein_id,reference_protein_id,reference_activity_id,reference_component_id,
               canonical_rhea,ec_l4,query_family,hard_flags,stratum,priority_cluster,priority_rhea,
               priority_ec4,priority_target,p_cluster,
               CASE WHEN canonical_rhea IS NULL THEN 1.0 ELSE MIN(1.0,? * 1.0/group_n) END AS p_rhea
        FROM (
            SELECT cluster_stage.*,
                   COUNT(*) OVER (PARTITION BY canonical_rhea) AS group_n,
                   ROW_NUMBER() OVER (
                       PARTITION BY canonical_rhea
                       ORDER BY priority_rhea,query_protein_id,reference_protein_id,reference_activity_id
                   ) AS selection_rank
            FROM cluster_stage
        ) WHERE canonical_rhea IS NULL OR selection_rank<=?
    """, (max_per_rhea, max_per_rhea))
    connection.execute("""
        CREATE TABLE ec4_stage AS
        SELECT row_id,query_protein_id,reference_protein_id,reference_activity_id,reference_component_id,
               canonical_rhea,ec_l4,query_family,hard_flags,stratum,priority_target,p_cluster,p_rhea,
               MIN(1.0,? * 1.0/group_n) AS p_ec4
        FROM (
            SELECT rhea_stage.*,
                   COUNT(*) OVER (PARTITION BY ec_l4) AS group_n,
                   ROW_NUMBER() OVER (
                       PARTITION BY ec_l4
                       ORDER BY priority_ec4,query_protein_id,reference_protein_id,reference_activity_id
                   ) AS selection_rank
            FROM rhea_stage
        ) WHERE selection_rank<=?
    """, (max_per_ec4, max_per_ec4))
    connection.commit()
    after_cluster = count_table(connection, "cluster_stage")
    after_rhea = count_table(connection, "rhea_stage")
    after_ec4 = count_table(connection, "ec4_stage")
    strata = dict(connection.execute("SELECT stratum,COUNT(*) FROM ec4_stage GROUP BY stratum").fetchall())
    difficult, ordinary = strata.get("DIFFICULT", 0), strata.get("ORDINARY", 0)
    if after_ec4 > target_pairs:
        difficult_quota = min(difficult, math.ceil(target_pairs * DIFFICULT_TARGET_FRACTION))
        ordinary_quota = min(ordinary, target_pairs - difficult_quota)
        remainder = target_pairs - difficult_quota - ordinary_quota
        extra_difficult = min(difficult - difficult_quota, remainder)
        difficult_quota += extra_difficult
        remainder -= extra_difficult
        ordinary_quota += min(ordinary - ordinary_quota, remainder)
    else:
        difficult_quota, ordinary_quota = difficult, ordinary
    connection.execute("""
        CREATE TABLE selected AS
        SELECT row_id,query_protein_id,reference_protein_id,reference_activity_id,reference_component_id,
               query_family,hard_flags,stratum,p_cluster,p_rhea,p_ec4,
               CASE stratum
                    WHEN 'DIFFICULT' THEN ? * 1.0/group_n
                    ELSE ? * 1.0/group_n
               END AS p_target
        FROM (
            SELECT ec4_stage.*,
                   COUNT(*) OVER (PARTITION BY stratum) AS group_n,
                   ROW_NUMBER() OVER (
                       PARTITION BY stratum
                       ORDER BY priority_target,query_protein_id,reference_protein_id,reference_activity_id
                   ) AS selection_rank
            FROM ec4_stage
        )
        WHERE (stratum='DIFFICULT' AND selection_rank<=?)
           OR (stratum='ORDINARY' AND selection_rank<=?)
    """, (difficult_quota, ordinary_quota, difficult_quota, ordinary_quota))
    connection.execute("CREATE UNIQUE INDEX selected_row_id ON selected(row_id)")
    connection.execute("CREATE TABLE activity_counts AS SELECT reference_activity_id,COUNT(*) AS n FROM selected GROUP BY reference_activity_id")
    connection.execute("CREATE UNIQUE INDEX activity_counts_key ON activity_counts(reference_activity_id)")
    connection.execute("CREATE TABLE family_counts AS SELECT query_family,COUNT(*) AS n FROM selected GROUP BY query_family")
    connection.execute("CREATE INDEX family_counts_key ON family_counts(query_family)")
    connection.execute("CREATE TABLE component_counts AS SELECT reference_component_id,COUNT(*) AS n FROM selected GROUP BY reference_component_id")
    connection.execute("CREATE UNIQUE INDEX component_counts_key ON component_counts(reference_component_id)")
    connection.commit()
    selected = count_table(connection, "selected")
    require(selected >= minimum_pairs, "minimum balanced TRAIN pair guard")
    require(selected == min(target_pairs, after_ec4), "target pair census")
    return {
        "after_reference_cluster_dedup": after_cluster, "after_exact_rhea_cap": after_rhea,
        "after_ec_l4_cap": after_ec4, "difficult_candidates": difficult, "ordinary_candidates": ordinary,
        "difficult_selected": difficult_quota, "ordinary_selected": ordinary_quota, "selected_pairs": selected,
    }


SELECTED_METADATA_SQL = """
    SELECT s.row_id,s.query_protein_id,s.reference_protein_id,s.reference_activity_id,s.hard_flags,s.stratum,
           s.p_cluster,s.p_rhea,s.p_ec4,s.p_target,a.n,f.n,c.n
    FROM selected AS s
    JOIN activity_counts AS a ON a.reference_activity_id=s.reference_activity_id
    JOIN family_counts AS f ON f.query_family IS s.query_family
    JOIN component_counts AS c ON c.reference_component_id=s.reference_component_id
    ORDER BY s.row_id
"""


def raw_weight(activity_n, family_n, component_n):
    require(activity_n > 0 and family_n > 0 and component_n > 0, "selected-set frequency")
    return ((1 / activity_n) * (1 / family_n) * (1 / component_n)) ** (1 / 3)


def calculate_weight_scale(connection, selected_count):
    total = math.fsum(raw_weight(row[10], row[11], row[12]) for row in connection.execute(SELECTED_METADATA_SQL))
    require(total > 0 and math.isfinite(total), "training weight normalization")
    return selected_count / total


class Sink:
    def __init__(self, path, schema, batch_rows):
        import pyarrow.parquet as pq
        self.schema, self.limit, self.rows, self.count = schema, batch_rows, [], 0
        self.digest = hashlib.sha256()
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")

    def add(self, row):
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
        self.digest.update(len(encoded).to_bytes(8, "big"))
        self.digest.update(encoded)
        self.rows.append(row)
        if len(self.rows) >= self.limit:
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


def write_selected(source, connection, target, schema, batch_rows, selected_count, weight_scale):
    import pyarrow.parquet as pq
    cursor = iter(connection.execute(SELECTED_METADATA_SQL))
    current = next(cursor, None)
    sink = Sink(target, schema, batch_rows)
    try:
        row_id = 0
        for batch in pq.ParquetFile(source).iter_batches(batch_size=batch_rows, columns=INPUT_COLUMNS):
            for row in batch.to_pylist():
                if current is not None and row_id == current[0]:
                    require(tuple(row[name] for name in KEY_FIELDS) == tuple(current[1:4]), "selected row identity")
                    activity_n, family_n, component_n = current[10], current[11], current[12]
                    activity_weight, family_weight, component_weight = 1 / activity_n, 1 / family_n, 1 / component_n
                    sample_weight = raw_weight(activity_n, family_n, component_n) * weight_scale
                    p_cluster, p_rhea, p_ec4, p_target = current[6:10]
                    row.update({
                        "sampling_applied": True, "sample_weight": sample_weight,
                        "reference_cluster_dedup_probability": p_cluster,
                        "exact_rhea_cap_probability": p_rhea, "ec_l4_cap_probability": p_ec4,
                        "difficulty_stratum_probability": p_target,
                        "sampling_probability": p_cluster * p_rhea * p_ec4 * p_target,
                        "selection_probability_is_population_ipw": False, "sampling_stratum": current[5],
                        "hard_case_flags_for_sampling": current[4], "activity_balance_weight": activity_weight,
                        "family_balance_weight": family_weight, "reference_cluster_weight": component_weight,
                    })
                    require(0 < row["sampling_probability"] <= 1 and math.isfinite(sample_weight), "selected pair weights")
                    sink.add(row)
                    current = next(cursor, None)
                row_id += 1
        require(current is None and sink.count + len(sink.rows) == selected_count, "selected output census")
        sink.close()
    except BaseException:
        try:
            sink.writer.close()
        except BaseException:
            pass
        raise
    return sink.count, sink.digest.hexdigest()


def execute(s4j_pairs, output, batch_rows=100000, target_pairs=TARGET_PAIRS, minimum_pairs=MINIMUM_PAIRS,
            max_per_rhea=MAX_PER_EXACT_RHEA, max_per_ec4=MAX_PER_EC_L4):
    import pyarrow.parquet as pq
    source, output = Path(s4j_pairs).resolve(), Path(output).resolve()
    require(source.is_file() and not source.is_symlink(), "S4J input")
    require(output.parent.is_dir() and not output.exists(), "exclusive output")
    require(type(batch_rows) is int and not isinstance(batch_rows, bool) and batch_rows > 0, "batch rows")
    input_sha = sha(source)
    input_record = {"path": str(source), "sha256": input_sha}
    output.mkdir(exist_ok=False)
    emit(output / "reservation.json", {
        "status": "ONE_S4K_BALANCED_TRAIN_SAMPLER_ATTEMPT_RESERVED", "input": input_record,
        "parameters": {"seed": SEED, "target_pairs": target_pairs, "minimum_pairs": minimum_pairs,
                       "max_per_exact_rhea": max_per_rhea, "max_per_ec_l4": max_per_ec4},
        "automatic_retry": False,
    })
    state, error, connection = "FAIL_CLOSED", None, None
    try:
        parquet = pq.ParquetFile(source)
        schema = output_schema(parquet.schema_arrow)
        database = output / "selection_index.sqlite"
        connection = connect_database(database)
        input_pairs = ingest_metadata(source, connection, batch_rows)
        census = build_selection(connection, target_pairs, minimum_pairs, max_per_rhea, max_per_ec4)
        weight_scale = calculate_weight_scale(connection, census["selected_pairs"])
        output_pairs = output / "balanced_train_pairs.parquet"
        written, row_digest = write_selected(source, connection, output_pairs, schema, batch_rows, census["selected_pairs"], weight_scale)
        require(written == census["selected_pairs"] and sha(source) == input_sha, "input/output stability")
        mean_weight = math.fsum(raw_weight(row[10], row[11], row[12]) * weight_scale for row in connection.execute(SELECTED_METADATA_SQL)) / written
        require(abs(mean_weight - 1.0) <= 1e-12, "mean sample weight")
        summary = {
            "status": "PASS_S4K_BALANCED_TRAIN_SAMPLER_PENDING_INDEPENDENT_AUDIT", "input": input_record,
            "input_pairs": input_pairs, **census, "seed": SEED, "target_pairs": target_pairs,
            "minimum_pairs": minimum_pairs, "max_per_exact_rhea": max_per_rhea, "max_per_ec_l4": max_per_ec4,
            "selected_row_sha256": row_digest, "mean_sample_weight": mean_weight,
            "selection_probability_is_population_ipw": False, "population_estimand_modified": False,
            "all_s4j_columns_preserved": True, "sqlite_spill_state_retained": True,
            "h7_h8_fabricated": False, "different_rhea_is_biochemical_negative": False,
            "feature_matrix_created": False, "training_started": False,
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
    parser.add_argument("--s4j-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=100000)
    parser.add_argument("--target-pairs", type=int, default=TARGET_PAIRS)
    parser.add_argument("--minimum-pairs", type=int, default=MINIMUM_PAIRS)
    parser.add_argument("--max-per-rhea", type=int, default=MAX_PER_EXACT_RHEA)
    parser.add_argument("--max-per-ec4", type=int, default=MAX_PER_EC_L4)
    args = parser.parse_args()
    execute(args.s4j_pairs, args.output, args.batch_rows, args.target_pairs, args.minimum_pairs, args.max_per_rhea, args.max_per_ec4)


if __name__ == "__main__":
    main()
