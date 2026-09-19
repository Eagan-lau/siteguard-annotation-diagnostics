"""Run one hash-locked S4B node-native MMseqs2 or Foldseek search unit."""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load("s4b_retrieval_search_core")
normalizer = load("s4b_retrieval_raw_normalizer")
FORMAT = "query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,qcov,tcov,evalue,bits"


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


def checked_run(name, argv, cwd, env, commands, runner):
    record = runner(argv, cwd, env)
    record["name"] = name
    commands.append(record)
    require(record["returncode"] == 0, "command failed " + name)


def validate_record(root, record, label):
    path = root / record["path"]
    require(path.is_file() and sha(path) == record["sha256"], "input identity " + label)
    return path


def execute(contract_path, unit_index, runner=run_command):
    contract_path = Path(contract_path).resolve()
    contract = json.loads(contract_path.read_text())
    require(contract["status"] == "ARMED_S4B_NODE_NATIVE_RETRIEVAL", "contract status")
    require(contract["attempt"] == 1 and contract["attempt_limit"] == 3 and not contract["automatic_retry"], "attempt policy")
    require(list(sys.version_info[:3]) == contract["python_version"], "Python identity")
    root = Path(contract["root"])
    require(str(root.resolve()) == contract["physical_root"], "root identity")
    authorization = contract["authorization"]
    require(authorization["full_search"] and authorization["worker_array"], "search authorization")
    require(not any(authorization[name] for name in ("functional_labels", "query_truth", "activity_expansion", "pair_generation", "training", "evaluation", "phase99", "protected_cohort")), "scope")
    for name, expected in contract["scripts"].items():
        require((HERE / name).is_file() and sha(HERE / name) == expected, "script identity " + name)
    chosen = core.unit(unit_index)
    require(contract["plan"]["units"][unit_index] == chosen and len(contract["plan"]["units"]) == 10, "unit plan")
    role, modality = chosen["query_role"], chosen["modality"]
    require(contract["parameters"][modality] == core.search_parameters(modality), "search parameters")
    tool_record = contract["tools"][modality]
    tool = Path(tool_record["path"])
    require(tool.is_file() and sha(tool) == tool_record["sha256"], "search tool identity")
    sort_record = contract["tools"]["sort"]
    sort_binary = Path(sort_record["path"])
    require(sort_binary.is_file() and sha(sort_binary) == sort_record["sha256"], "sort identity")
    node_ledger = validate_record(root, contract["inputs"]["node_ledger"], "node ledger")
    selection = None
    if modality == "foldseek":
        selection = validate_record(root, contract["inputs"]["foldseek_selection"], "Foldseek selection")
    database = contract["databases"][modality]
    reference_prefix = root / database["reference_prefix"]
    query_prefix = root / database["query_prefixes"][role]
    database_audit_path = root / contract["database_audit"]["independent_audit"]
    database_pass_path = root / contract["database_audit"]["checkpoint"]
    require(database_audit_path.is_file() and database_pass_path.is_file(), "database audit gate")
    database_audit = json.loads(database_audit_path.read_text())
    database_pass = json.loads(database_pass_path.read_text())
    require(database_audit["status"] == "PASS_S4B_RETRIEVAL_DATABASE_SETUP_INDEPENDENT_AUDIT", "database audit status")
    require(database_pass["status"] == "PASS_S4B_RETRIEVAL_DATABASE_SETUP_AUDIT", "database checkpoint status")
    require(database_pass["audit_sha256"] == sha(database_audit_path), "database audit checkpoint binding")
    require(database_audit["contract_sha256"] == sha(contract_path), "database audit contract binding")
    require((Path(str(reference_prefix) + ".dbtype")).is_file() and (Path(str(query_prefix) + ".dbtype")).is_file(), "database prefixes")

    parent = root / contract["worker_output"]
    output = parent / f"worker_{unit_index:02d}_{role}_{modality}"
    require(parent.is_dir() and not output.exists(), "worker output identity")
    output.mkdir(exist_ok=False)
    contract_sha = sha(contract_path)
    emit(output / "reservation.json", {
        "status": "ONE_S4B_RETRIEVAL_WORKER_ATTEMPT_RESERVED", "unit": chosen,
        "contract_sha256": contract_sha, "automatic_retry": False,
    })
    state, error, commands = "FAIL_CLOSED", None, []
    try:
        scratch_root = Path(os.environ.get("LOCALSCRATCH") or os.environ.get("TMPDIR") or "/tmp").resolve()
        require(scratch_root.is_dir(), "scratch root")
        scratch = scratch_root / f"siteguard_s4b_{os.environ.get('SLURM_JOB_ID', 'local')}_{unit_index:02d}"
        require(not scratch.exists(), "worker scratch exists")
        scratch.mkdir(exist_ok=False)
        result_prefix = output / "engine_result"
        engine_raw = output / "engine_native_raw.tsv"
        parameters = contract["parameters"][modality]
        threads = contract["threads"]
        environment = dict(os.environ)
        environment.update({"OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"})
        if modality == "mmseqs":
            search = [
                tool, "search", query_prefix, reference_prefix, result_prefix, scratch / "search",
                "-s", parameters["sensitivity"], "-e", parameters["evalue"],
                "--max-seqs", parameters["max_seqs"], "--threads", threads,
                "--search-type", parameters["search_type"],
            ]
        else:
            search = [
                tool, "search", query_prefix, reference_prefix, result_prefix, scratch / "search",
                "-s", parameters["sensitivity"], "-e", parameters["evalue"],
                "--max-seqs", parameters["max_seqs"], "--threads", threads,
                "--alignment-type", parameters["alignment_type"],
                "--sort-by-structure-bits", parameters["sort_by_structure_bits"],
            ]
        checked_run("search", search, output, environment, commands, runner)
        convert = [
            tool, "convertalis", query_prefix, reference_prefix, result_prefix, engine_raw,
            "--format-output", FORMAT, "--threads", threads,
        ]
        checked_run("convertalis", convert, output, environment, commands, runner)
        require(engine_raw.is_file(), "engine raw output")
        normalization = normalizer.execute(
            engine_raw, modality, role, node_ledger, output / "raw_hits.tsv", scratch,
            sort_binary, selection,
        )
        summary = {
            "status": "PASS_S4B_RETRIEVAL_WORKER_PENDING_INDEPENDENT_AUDIT",
            "unit": chosen, "parameters": parameters, "threads": threads,
            "contract_sha256": contract_sha, "commands": commands,
            "normalization": normalization,
            "functional_labels_read": False, "query_truth_read": False,
            "activity_expansion_started": False, "pair_generation_started": False,
            "training_started": False, "evaluation_started": False,
        }
        emit(output / "worker_summary.json", summary)
        emit(output / "S4B_RETRIEVAL_WORKER_PASS.json", summary)
        state = summary["status"]
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    emit(output / "terminal.json", {"status": state, "error": error, "contract_sha256": contract_sha, "automatic_retry": False})
    if error:
        raise RuntimeError(error["message"])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--unit-index", type=int, required=True)
    args = parser.parse_args()
    execute(args.contract, args.unit_index)


if __name__ == "__main__":
    main()
