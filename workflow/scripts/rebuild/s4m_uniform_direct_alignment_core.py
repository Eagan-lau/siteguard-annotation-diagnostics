"""Pure S4M command and raw-alignment normalization rules."""
import math


BASE_COLUMNS = (
    "query", "target", "fident", "alnlen", "qstart", "qend", "qlen",
    "tstart", "tend", "tlen", "qcov", "tcov", "evalue", "bits",
)
FOLDSEEK_COLUMNS = ("lddt", "qtmscore", "ttmscore", "alntmscore", "rmsd", "prob")
INTEGER_COLUMNS = {"alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen"}
FRACTION_COLUMNS = {"fident", "qcov", "tcov", "lddt", "qtmscore", "ttmscore", "prob"}
COMMON_OUTPUT = (
    "query_node", "reference_node", "query_db_key", "reference_db_key",
    *BASE_COLUMNS[2:],
)


def require(value, message):
    if not value:
        raise ValueError(message)


def format_columns(modality):
    require(modality in {"mmseqs", "foldseek"}, "modality")
    return BASE_COLUMNS + (FOLDSEEK_COLUMNS if modality == "foldseek" else ())


def command_plan(tool, modality, database, prefilter, alignment, raw, threads):
    require(isinstance(tool, str) and tool and isinstance(database, str) and database, "tool/database")
    require(all(isinstance(value, str) and value for value in (prefilter, alignment, raw)), "command paths")
    require(type(threads) is int and not isinstance(threads, bool) and threads > 0, "threads")
    columns = ",".join(format_columns(modality))
    commands = [
        [tool, "tsv2db", prefilter + ".tsv", prefilter, "--output-dbtype", "7", "-v", "1"],
    ]
    if modality == "mmseqs":
        commands.append([tool, "align", database, database, prefilter, alignment, "--alignment-mode", "3", "-e", "1e100", "--threads", str(threads)])
    else:
        commands.append([
            tool, "structurealign", database, database, prefilter, alignment,
            "--alignment-type", "2", "--alignment-mode", "3",
            "--exact-tmscore", "1", "-a", "1", "-e", "1e100",
            "--threads", str(threads),
        ])
    commands.append([tool, "convertalis", database, database, alignment, raw, "--format-output", columns])
    if modality == "foldseek":
        commands[-1].extend(["--exact-tmscore", "1"])
    return commands


def parse_prefilter(lines):
    result, prior = [], None
    for line_number, line in enumerate(lines, 1):
        fields = line.rstrip("\n").split("\t")
        require(len(fields) == 4 and fields[2:] == ["2000", "0"], f"prefilter format line {line_number}")
        try:
            pair = (int(fields[0]), int(fields[1]))
        except ValueError as exc:
            raise ValueError(f"prefilter integer line {line_number}") from exc
        require(min(pair) >= 0 and (prior is None or pair > prior), "strict unique prefilter order")
        prior = pair; result.append(pair)
    return result


def key_metadata(rows):
    by_key, by_name, by_node = {}, {}, {}
    for row in rows:
        require(set(row) == {"db_key", "node_id", "output_name"}, "key metadata projection")
        key, node, name = row["db_key"], row["node_id"], row["output_name"]
        require(type(key) is int and key >= 0 and type(node) is int and node >= 0, "key metadata integer")
        require(isinstance(name, str) and name, "key metadata name")
        require(key not in by_key and name not in by_name and node not in by_node, "one-to-one key/name/node mapping")
        record = (node, name)
        by_key[key], by_name[name], by_node[node] = record, (node, key), (key, name)
    require(by_key, "empty key metadata")
    return by_key, by_name


def expected_requests(prefilter_lines, metadata_rows):
    by_key, by_name = key_metadata(metadata_rows)
    expected = {}
    for query_key, reference_key in parse_prefilter(prefilter_lines):
        require(query_key in by_key and reference_key in by_key, "prefilter key absent from frozen mapping")
        query_node, query_name = by_key[query_key]
        reference_node, reference_name = by_key[reference_key]
        name_pair = (query_name, reference_name)
        require(name_pair not in expected and query_node != reference_node, "unique separated request")
        expected[name_pair] = (query_node, reference_node, query_key, reference_key)
    return expected, by_name


def parse_number(name, value, modality="mmseqs"):
    try:
        parsed = int(value) if name in INTEGER_COLUMNS else float(value)
    except ValueError as exc:
        raise ValueError("alignment numeric field " + name) from exc
    require(type(parsed) is int or math.isfinite(parsed), "finite alignment field " + name)
    if name in INTEGER_COLUMNS:
        require(parsed >= 0, "nonnegative alignment integer " + name)
    if name in FRACTION_COLUMNS:
        require(0 <= parsed <= 1, "fractional alignment field " + name)
    if name in {"evalue", "rmsd"} or (name == "bits" and modality != "foldseek"):
        require(parsed >= 0, "nonnegative alignment field " + name)
    return parsed


def normalize(raw_lines, modality, expected):
    columns = format_columns(modality)
    seen, result = set(), []
    for line_number, line in enumerate(raw_lines, 1):
        fields = line.rstrip("\n").split("\t")
        require(len(fields) == len(columns), f"alignment field count line {line_number}")
        query_name, reference_name = fields[:2]
        pair = (query_name, reference_name)
        require(pair in expected and pair not in seen, "unexpected or duplicate alignment pair")
        seen.add(pair)
        query_node, reference_node, query_key, reference_key = expected[pair]
        row = {
            "query_node": query_node, "reference_node": reference_node,
            "query_db_key": query_key, "reference_db_key": reference_key,
        }
        for name, value in zip(columns[2:], fields[2:]):
            row[name] = parse_number(name, value, modality)
        require(row["qlen"] > 0 and row["tlen"] > 0 and row["alnlen"] > 0, "positive alignment lengths")
        require(0 <= row["qstart"] <= row["qend"] <= row["qlen"], "query coordinate range")
        require(0 <= row["tstart"] <= row["tend"] <= row["tlen"], "target coordinate range")
        if modality == "foldseek":
            validate_aligned_tm(row)
        require(tuple(row) == COMMON_OUTPUT + (FOLDSEEK_COLUMNS if modality == "foldseek" else ()), "normalized schema")
        result.append(row)
    require(seen == set(expected), "complete alignment request census")
    return sorted(result, key=lambda row: (row["query_node"], row["reference_node"]))

def validate_aligned_tm(row):
    """Foldseek 10-941cd33 endpoint-span normalization, retaining raw value."""
    span = min(row["qend"] - row["qstart"], row["tend"] - row["tstart"])
    require(span > 0, "positive aligned-TM normalization span")
    # SSTR exports four significant digits; values in [1,2] round by at most 0.0005.
    require(0 <= row["alntmscore"] <= (span + 1) / span + 0.000500000001,
             "version-specific aligned-TM bound")

