"""Filter normalized Foldseek hits to exact structure/sequence length identity."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


NODE_COLUMNS = ["node_id", "component_id", "role", "sequence_sha256", "sequence_length", "representative_protein_id", "protein_id_count"]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute(source, node_ledger, output, summary_path):
    import pyarrow.parquet as pq

    source, node_ledger, output, summary_path = map(lambda p: Path(p).resolve(), (source, node_ledger, output, summary_path))
    require(source.is_file() and node_ledger.is_file(), "filter inputs")
    require(output.parent.is_dir() and not output.exists() and not summary_path.exists(), "filter outputs")
    table = pq.read_table(node_ledger)
    require(table.column_names == NODE_COLUMNS, "node ledger schema")
    lengths = {row["node_id"]: row["sequence_length"] for row in table.to_pylist()}
    counts = Counter()
    with source.open("r", encoding="ascii", newline="") as reader, output.open("x", encoding="ascii", newline="\n") as writer:
        for line_number, line in enumerate(reader, 1):
            fields = line.rstrip("\r\n").split("\t")
            require(len(fields) == 14 and all(fields), "normalized fields line " + str(line_number))
            query, reference = int(fields[0]), int(fields[1])
            require(query in lengths and reference in lengths, "node identity")
            qlen, tlen = int(fields[6]), int(fields[9])
            qmatch, tmatch = qlen == lengths[query], tlen == lengths[reference]
            counts["source_rows"] += 1
            counts["query_length_mismatch_rows"] += int(not qmatch)
            counts["reference_length_mismatch_rows"] += int(not tmatch)
            counts["any_length_mismatch_rows"] += int(not (qmatch and tmatch))
            if qmatch and tmatch:
                writer.write(line.rstrip("\r\n") + "\n")
                counts["retained_rows"] += 1
    require(counts["source_rows"] > 0 and counts["retained_rows"] > 0, "nonempty length filter")
    require(counts["source_rows"] == counts["retained_rows"] + counts["any_length_mismatch_rows"], "filter conservation")
    summary = {
        "status": "PASS_S4B_FOLDSEEK_LENGTH_IDENTITY_FILTER_PENDING_INDEPENDENT_AUDIT",
        **dict(counts), "source_sha256": sha(source), "output_sha256": sha(output),
        "filter_rule": "foldseek_qlen_equals_sequence_length_and_foldseek_tlen_equals_sequence_length",
        "search_rerun": False, "functional_labels_read": False, "query_truth_read": False,
    }
    with summary_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True); parser.add_argument("--node-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(execute(args.source, args.node_ledger, args.output, args.summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
