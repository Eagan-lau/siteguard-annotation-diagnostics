"""Remap role-independent historical covariates onto the SC1 protein ledger."""
import argparse
import hashlib
import json
import math
import traceback
from collections import Counter
from pathlib import Path


ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")
PROTEIN_COLUMNS = ["protein_id", "node_id", "component_id", "role"]
PROJECTIONS = {
    "family": ["protein_id", "primary_pfam", "primary_pfam_clan"],
    "structure": ["protein_id", "primary_cath_superfamily", "foldseek_structure_cluster"],
    "taxonomy": ["protein_id", "taxonomy_id", "taxonomy_group"],
    "afdb": ["protein_id", "filename", "model_version", "compressed_size", "has_structure", "structure_source"],
}
OUTPUT_COLUMNS = PROTEIN_COLUMNS + [
    "primary_pfam", "primary_pfam_clan", "primary_cath_superfamily",
    "foldseek_structure_cluster", "taxonomy_id", "taxonomy_group", "filename",
    "model_version", "compressed_size", "has_structure", "structure_source",
]
FORBIDDEN_SOURCE_COLUMNS = [
    "split", "cluster_id_30", "pfam_family_split", "pfam_clan_split",
    "cath_superfamily_split", "foldseek_structure_cluster_split",
    "taxonomy_species_split", "pfam_domains_json", "cath_superfamilies_json",
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


def output_schema():
    import pyarrow as pa

    return pa.schema(
        [("protein_id", pa.string()), ("node_id", pa.int32()), ("component_id", pa.int32()), ("role", pa.string())]
        + [(name, pa.string()) for name in (
            "primary_pfam", "primary_pfam_clan", "primary_cath_superfamily",
            "foldseek_structure_cluster", "taxonomy_id", "taxonomy_group", "filename",
        )]
        + [("model_version", pa.float64()), ("compressed_size", pa.float64()),
           ("has_structure", pa.bool_()), ("structure_source", pa.string())]
    )


def read_unique(path, columns, name):
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=columns)
    require(table.column_names == columns, name + " projected schema")
    rows = {}
    for row in table.to_pylist():
        protein = row["protein_id"]
        require(isinstance(protein, str) and protein and protein not in rows, name + " protein identity")
        rows[protein] = row
    require(rows, "empty " + name)
    return rows


def execute(protein_ledger, family, structure, taxonomy, afdb, output):
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = {
        "protein_ledger": Path(protein_ledger).resolve(), "family": Path(family).resolve(),
        "structure": Path(structure).resolve(), "taxonomy": Path(taxonomy).resolve(),
        "afdb": Path(afdb).resolve(),
    }
    output = Path(output).resolve()
    require(all(path.is_file() for path in paths.values()), "covariate inputs")
    require(output.parent.is_dir() and not output.exists(), "exclusive output")
    output.mkdir(exist_ok=False)
    inputs = {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()}
    emit(output / "reservation.json", {"status": "ONE_S4G_COVARIATE_REMAP_ATTEMPT_RESERVED", "inputs": inputs, "automatic_retry": False})
    state, error = "FAIL_CLOSED", None
    try:
        proteins = read_unique(paths["protein_ledger"], PROTEIN_COLUMNS, "protein ledger")
        require(set(row["role"] for row in proteins.values()) == set(ROLES), "role universe")
        sources = {name: read_unique(paths[name], columns, name) for name, columns in PROJECTIONS.items()}
        universe = set(proteins)
        require(all(set(rows) == universe for rows in sources.values()), "covariate protein universe")
        records = []
        for protein in sorted(universe):
            base = proteins[protein]
            af = sources["afdb"][protein]
            require(type(af["has_structure"]) is bool, "AFDB availability type")
            if af["has_structure"]:
                require(isinstance(af["filename"], str) and af["filename"] and isinstance(af["model_version"], (int, float)) and math.isfinite(af["model_version"]) and isinstance(af["compressed_size"], (int, float)) and math.isfinite(af["compressed_size"]) and af["compressed_size"] > 0 and af["structure_source"] == "AlphaFoldDB_SwissProt_v6", "available AFDB record")
            else:
                require(af["filename"] is None and af["model_version"] is None and af["compressed_size"] is None and af["structure_source"] == "UNAVAILABLE_IN_FROZEN_BULK_INDEX", "unavailable AFDB record")
            records.append(
                {
                    **base,
                    "primary_pfam": sources["family"][protein]["primary_pfam"],
                    "primary_pfam_clan": sources["family"][protein]["primary_pfam_clan"],
                    "primary_cath_superfamily": sources["structure"][protein]["primary_cath_superfamily"],
                    "foldseek_structure_cluster": sources["structure"][protein]["foldseek_structure_cluster"],
                    "taxonomy_id": sources["taxonomy"][protein]["taxonomy_id"],
                    "taxonomy_group": sources["taxonomy"][protein]["taxonomy_group"],
                    **{name: af[name] for name in ("filename", "model_version", "compressed_size", "has_structure", "structure_source")},
                }
            )
        pq.write_table(pa.Table.from_pylist(records, schema=output_schema()), output / "protein_covariate_ledger.parquet", compression="zstd")
        role_counts = Counter(row["role"] for row in records)
        summary = {
            "status": "PASS_S4G_COVARIATE_REMAP_PENDING_INDEPENDENT_AUDIT",
            "inputs": inputs, "proteins": len(records),
            "role_counts": {role: role_counts[role] for role in ROLES},
            "structures_available": sum(row["has_structure"] for row in records),
            "structures_unavailable": sum(not row["has_structure"] for row in records),
            "allowed_source_projections": PROJECTIONS,
            "forbidden_source_columns_not_read": FORBIDDEN_SOURCE_COLUMNS,
            "new_roles_from_s4a_only": True, "functional_labels_read": False,
            "query_truth_read": False, "old_split_columns_read": False,
            "feature_matrix_created": False, "sampling_started": False,
            "training_started": False,
        }
        emit(output / "producer_summary.json", summary)
        state = summary["status"]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    emit(output / "terminal.json", {"status": state, "error": error, "automatic_retry": False})
    if error:
        raise RuntimeError(error["message"])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein-ledger", type=Path, required=True); parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True); parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--afdb", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); execute(args.protein_ledger, args.family, args.structure, args.taxonomy, args.afdb, args.output)


if __name__ == "__main__": main()

