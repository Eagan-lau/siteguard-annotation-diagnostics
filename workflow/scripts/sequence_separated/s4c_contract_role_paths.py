"""Print hash-validated MMseqs and Foldseek candidate paths for one S4C role."""
import hashlib
import json
import sys
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


contract_path, role = Path(sys.argv[1]).resolve(), sys.argv[2]
contract = json.loads(contract_path.read_text(encoding="utf-8"))
if contract["status"] != "ARMED_S4C_CROSS_MODALITY_UNION" or role not in contract["roles"]:
    raise RuntimeError("contract or role identity")
root = Path(contract["root"])
for modality in ("mmseqs", "foldseek"):
    record = contract["inputs"][role][modality]["candidate"]
    path = (root / record["path"]).resolve()
    if not path.is_file() or path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
        raise RuntimeError("candidate identity " + modality)
    print(path)
