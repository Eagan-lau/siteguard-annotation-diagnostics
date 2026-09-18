#!/usr/bin/env python3
"""Fetch the bounded external-P450 PubChem panel with raw-response retention."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROPERTIES = "CanonicalSMILES,IsomericSMILES,InChI,InChIKey,MolecularFormula,Charge,Title"


def client() -> requests.Session:
    retry = Retry(
        total=12,
        connect=12,
        read=12,
        status=12,
        backoff_factor=1.0,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    value = requests.Session()
    value.headers["User-Agent"] = "SiteGuard/3.0 academic targeted PubChem snapshot"
    value.mount("https://", HTTPAdapter(max_retries=retry))
    return value


def chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def request_json(session: requests.Session, url: str) -> dict[str, Any]:
    response = session.get(url, timeout=(30, 300))
    response.raise_for_status()
    time.sleep(0.25)
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    cids = sorted({int(line.strip()) for line in args.cids.read_text(encoding="utf-8").splitlines() if line.strip()})
    if not cids or len(cids) > 100_000:
        raise SystemExit(f"invalid targeted CID count: {len(cids)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "pubchem_compounds_raw.jsonl.gz"
    rows: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    session = client()
    with gzip.open(raw_path.with_suffix(raw_path.suffix + ".tmp"), "wt", encoding="utf-8") as raw:
        for batch_index, batch in enumerate(chunks(cids, args.batch_size), start=1):
            joined = ",".join(map(str, batch))
            try:
                payload = request_json(session, f"{BASE}/compound/cid/{joined}/property/{PROPERTIES}/JSON")
                raw.write(json.dumps({"kind": "properties", "requested_cids": batch, "response": payload}, ensure_ascii=False) + "\n")
                for item in payload.get("PropertyTable", {}).get("Properties", []):
                    cid = int(item["CID"])
                    rows[cid] = {
                        "cid": cid,
                        "connectivity_smiles": item.get("ConnectivitySMILES", item.get("CanonicalSMILES", "")),
                        "isomeric_smiles": item.get("SMILES", item.get("IsomericSMILES", "")),
                        "inchi": item.get("InChI", ""),
                        "inchikey": item.get("InChIKey", ""),
                        "molecular_formula": item.get("MolecularFormula", ""),
                        "formal_charge": item.get("Charge"),
                        "title": item.get("Title", ""),
                        "synonyms_json": "[]",
                    }
            except Exception as exc:  # noqa: BLE001 - retain a complete targeted failure inventory
                failures.extend({"cid": cid, "stage": "properties", "error": f"{type(exc).__name__}: {exc}"} for cid in batch)
            print(json.dumps({"stage": "properties", "batch": batch_index, "batches": len(chunks(cids, args.batch_size)), "records": len(rows)}))
        synonym_batch_size = min(20, args.batch_size)
        for batch_index, batch in enumerate(chunks(sorted(rows), synonym_batch_size), start=1):
            joined = ",".join(map(str, batch))
            try:
                payload = request_json(session, f"{BASE}/compound/cid/{joined}/synonyms/JSON")
                raw.write(json.dumps({"kind": "synonyms", "requested_cids": batch, "response": payload}, ensure_ascii=False) + "\n")
                for item in payload.get("InformationList", {}).get("Information", []):
                    cid = int(item["CID"])
                    if cid in rows:
                        rows[cid]["synonyms_json"] = json.dumps(item.get("Synonym", []), ensure_ascii=False)
            except Exception as exc:  # noqa: BLE001
                failures.extend({"cid": cid, "stage": "synonyms", "error": f"{type(exc).__name__}: {exc}"} for cid in batch)
            print(json.dumps({"stage": "synonyms", "batch": batch_index, "batches": len(chunks(sorted(rows), synonym_batch_size))}))
    raw_path.with_suffix(raw_path.suffix + ".tmp").replace(raw_path)
    frame = pd.DataFrame([rows[cid] for cid in sorted(rows)])
    frame.to_parquet(args.output_dir / "pubchem_compounds.parquet", index=False)
    missing = sorted(set(cids) - set(rows))
    failure_path = args.output_dir / "pubchem_failed_cids.tsv"
    with failure_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cid", "stage", "error"], delimiter="\t")
        writer.writeheader()
        writer.writerows(failures)
        writer.writerows({"cid": cid, "stage": "missing", "error": "no property record returned"} for cid in missing if not any(row["cid"] == cid for row in failures))
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "requested_cids": len(cids),
        "property_records": len(rows),
        "missing_cids": len(missing),
        "failed_requests_by_cid_stage": len(failures),
        "raw_response": str(raw_path),
        "scope": "external P450 compound normalization only; never used to overwrite Rhea/ChEBI chemistry",
        "status": "PASS" if not missing else "PASS_WITH_UNAVAILABLE_CIDS",
    }
    (args.output_dir / "pubchem_download_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
