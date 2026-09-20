"""Build shared node-native MMseqs2 and Foldseek databases for S4B."""
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


ROLES = ("TRAIN", "DEV", "CAL_FIT", "CAL_RULE", "RETEST")


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


def run_command(argv, cwd, env):
    result = subprocess.run([str(value) for value in argv], cwd=cwd, env=env, capture_output=True, text=True)
    return {"argv": [str(value) for value in argv], "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def checked(name, argv, cwd, env, commands, runner):
    record = runner(argv, cwd, env)
    record["name"] = name
    commands.append(record)
    require(record["returncode"] == 0, "command failed " + name)


def validate_record(root, record, label):
    path = root / record["path"]
    require(path.is_file() and sha(path) == record["sha256"], "input identity " + label)
    return path


def fasta_count(path):
    count = 0
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def key_count(path):
    values = path.read_text(encoding="ascii").splitlines()
    require(all(value.isdigit() for value in values), "Foldseek key format")
    require(len(values) == len(set(values)), "Foldseek key uniqueness")
    return len(values)


def manifest(directory, base):
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_dir():
            continue
        relative = str(path.relative_to(base))
        if path.is_symlink():
            records.append({
                "path": relative, "kind": "symlink", "target": os.readlink(path),
                "resolved_target": str(path.resolve()), "lstat_bytes": path.lstat().st_size,
            })
        else:
            records.append({"path": relative, "kind": "file", "bytes": path.stat().st_size, "sha256": sha(path)})
    require(records, "empty database manifest")
    return records


def execute(contract_path, runner=run_command):
    contract_path = Path(contract_path).resolve()
    contract = json.loads(contract_path.read_text())
    require(contract["status"] == "ARMED_S4B_NODE_NATIVE_RETRIEVAL", "contract status")
    require(contract["attempt"] == 1 and contract["attempt_limit"] == 3 and not contract["automatic_retry"], "attempt policy")
    require(list(sys.version_info[:3]) == contract["python_version"], "Python identity")
    root = Path(contract["root"])
    require(str(root.resolve()) == contract["physical_root"], "root identity")
    authorization = contract["authorization"]
    require(authorization["database_setup"] and authorization["full_search"], "database authorization")
    require(not any(authorization[name] for name in ("functional_labels", "query_truth", "activity_expansion", "pair_generation", "training", "evaluation", "phase99", "protected_cohort")), "scope")
    here = Path(__file__).resolve().parent
    for name, expected in contract["scripts"].items():
        require((here / name).is_file() and sha(here / name) == expected, "script identity " + name)
    tools = {}
    for name in ("mmseqs", "foldseek"):
        record = contract["tools"][name]
        path = Path(record["path"])
        require(path.is_file() and sha(path) == record["sha256"], "tool identity " + name)
        tools[name] = path
    inputs = contract["inputs"]
    reference_fasta = validate_record(root, inputs["reference_fasta"], "reference FASTA")
    query_fastas = {role: validate_record(root, inputs["query_fastas"][role], "query FASTA " + role) for role in ROLES}
    reference_keys = validate_record(root, inputs["reference_keys"], "reference keys")
    query_keys = {role: validate_record(root, inputs["query_keys"][role], "query keys " + role) for role in ROLES}
    for index, record in enumerate(inputs["foldseek_full_db_evidence"]):
        validate_record(root, record, "Foldseek full DB " + str(index))
    full_foldseek = root / inputs["foldseek_full_db_prefix"]
    require(Path(str(full_foldseek) + ".dbtype").is_file(), "Foldseek full DB prefix")
    expected = contract["expected"]
    require(fasta_count(reference_fasta) == expected["role_nodes"]["TRAIN"], "reference FASTA census")
    require(all(fasta_count(query_fastas[role]) == expected["role_nodes"][role] for role in ROLES), "query FASTA census")
    require(key_count(reference_keys) == expected["role_structure_available_nodes"]["TRAIN"], "reference key census")
    require(all(key_count(query_keys[role]) == expected["role_structure_available_nodes"][role] for role in ROLES), "query key census")

    output = root / contract["database_output"]
    require(output.parent.is_dir() and not output.exists(), "database output identity")
    output.mkdir(exist_ok=False)
    emit(output / "reservation.json", {"status": "ONE_S4B_DATABASE_SETUP_ATTEMPT_RESERVED", "contract_sha256": sha(contract_path), "automatic_retry": False})
    status, error, commands = "FAIL_CLOSED", None, []
    try:
        scratch_root = Path(os.environ.get("LOCALSCRATCH") or os.environ.get("TMPDIR") or "/tmp").resolve()
        require(scratch_root.is_dir(), "scratch root")
        scratch = scratch_root / f"siteguard_s4b_db_{os.environ.get('SLURM_JOB_ID', 'local')}"
        require(not scratch.exists(), "database scratch exists")
        scratch.mkdir(exist_ok=False)
        environment = dict(os.environ)
        environment.update({"OMP_NUM_THREADS": str(contract["threads"]), "OPENBLAS_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"})
        prefixes = {"mmseqs": {"query": {}}, "foldseek": {"query": {}}}
        mm_reference = output / "mmseqs/reference/db"
        mm_reference.parent.mkdir(parents=True)
        prefixes["mmseqs"]["reference"] = mm_reference
        checked("mmseqs_createdb_reference", [tools["mmseqs"], "createdb", reference_fasta, mm_reference, "--dbtype", 1, "--shuffle", 0, "--compressed", 0, "-v", 3], output, environment, commands, runner)
        for role in ROLES:
            prefix = output / f"mmseqs/query_{role}/db"
            prefix.parent.mkdir(parents=True)
            prefixes["mmseqs"]["query"][role] = prefix
            checked("mmseqs_createdb_query_" + role, [tools["mmseqs"], "createdb", query_fastas[role], prefix, "--dbtype", 1, "--shuffle", 0, "--compressed", 0, "-v", 3], output, environment, commands, runner)
        checked("mmseqs_createindex_reference", [tools["mmseqs"], "createindex", mm_reference, scratch / "mmseqs_index", "--threads", contract["threads"], "--search-type", 1], output, environment, commands, runner)

        fs_reference = output / "foldseek/reference/db"
        fs_reference.parent.mkdir(parents=True)
        prefixes["foldseek"]["reference"] = fs_reference
        checked("foldseek_createsubdb_reference", [tools["foldseek"], "createsubdb", reference_keys, full_foldseek, fs_reference, "--subdb-mode", 1], output, environment, commands, runner)
        for role in ROLES:
            prefix = output / f"foldseek/query_{role}/db"
            prefix.parent.mkdir(parents=True)
            prefixes["foldseek"]["query"][role] = prefix
            checked("foldseek_createsubdb_query_" + role, [tools["foldseek"], "createsubdb", query_keys[role], full_foldseek, prefix, "--subdb-mode", 1], output, environment, commands, runner)
        checked("foldseek_createindex_reference", [tools["foldseek"], "createindex", fs_reference, scratch / "foldseek_index", "--threads", contract["threads"]], output, environment, commands, runner)
        for modality in ("mmseqs", "foldseek"):
            require(Path(str(prefixes[modality]["reference"]) + ".dbtype").is_file(), "reference DB " + modality)
            require(all(Path(str(prefixes[modality]["query"][role]) + ".dbtype").is_file() for role in ROLES), "query DB " + modality)
        database_manifest = manifest(output / "mmseqs", output) + manifest(output / "foldseek", output)
        database_manifest.sort(key=lambda row: row["path"])
        emit(output / "database_manifest.json", {"status": "S4B_DATABASE_COMPOUND_MEMBERS_FROZEN", "members": database_manifest})
        summary = {
            "status": "PASS_S4B_RETRIEVAL_DATABASE_SETUP_PENDING_INDEPENDENT_AUDIT",
            "contract_sha256": sha(contract_path), "commands": commands,
            "command_count": len(commands), "database_manifest_members": len(database_manifest),
            "prefixes": {
                modality: {
                    "reference": str(prefixes[modality]["reference"].relative_to(root)),
                    "query": {role: str(prefixes[modality]["query"][role].relative_to(root)) for role in ROLES},
                }
                for modality in ("mmseqs", "foldseek")
            },
            "functional_labels_read": False, "query_truth_read": False,
            "search_started": False, "training_started": False,
        }
        emit(output / "setup_summary.json", summary)
        emit(output / "S4B_RETRIEVAL_DATABASE_SETUP_PASS.json", summary)
        status = summary["status"]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    emit(output / "terminal.json", {"status": status, "error": error, "contract_sha256": sha(contract_path), "automatic_retry": False})
    if error:
        raise RuntimeError(error["message"])
    return summary


if __name__ == "__main__":
    execute(Path(sys.argv[1]))
