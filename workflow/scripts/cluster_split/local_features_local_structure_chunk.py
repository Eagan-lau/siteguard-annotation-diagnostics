#!/usr/bin/env python3
"""Compute structure-derived Local-2/Local-3 site descriptors for one shard."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pandas as pd


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: index for index, aa in enumerate(AA)}
RADII = (6.0, 8.0, 10.0)
PROPERTY_SETS = [set("KRH"), set("DE"), set("AILMFWVY"), set("FWY"), set("STNQKRH"), set("DENQSTYCH")]
KEY = [
    "pair_set", "query_protein_id", "reference_protein_id", "reference_activity_id",
    "reference_site_id",
]


def cosine(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None:
        return float("nan")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator else float("nan")


class StructureStore:
    def __init__(self, source_root: Path, needed: set[str], cache_size: int = 256) -> None:
        self.tar_path = source_root / "data/raw/alphafold/bulk_swissprot_v6/swissprot_cif_v6.tar"
        index = pd.read_parquet(
            source_root / "data/raw/alphafold/bulk_swissprot_v6/afdb_bulk_file_index.parquet",
            columns=["uniprot_accession", "tar_data_offset", "compressed_size"],
        )
        index = index.loc[index["uniprot_accession"].isin(needed)].drop_duplicates("uniprot_accession")
        self.index = index.set_index("uniprot_accession").to_dict("index")
        self.cache_size = cache_size
        self.cache: OrderedDict[str, tuple[dict[int, dict[str, Any]], str]] = OrderedDict()

    @staticmethod
    def parse(content: bytes) -> dict[int, dict[str, Any]]:
        if content.startswith(b"\x1f\x8b"):
            content = gzip.decompress(content)
        document = gemmi.cif.read_string(content.decode("utf-8"))
        structure = gemmi.make_structure_from_block(document.sole_block())
        if not structure:
            return {}
        chains: list[dict[int, dict[str, Any]]] = []
        for chain in structure[0]:
            residues: dict[int, dict[str, Any]] = {}
            for residue in chain:
                atoms = {
                    atom.name.strip(): np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32)
                    for atom in residue
                }
                if "CA" not in atoms:
                    continue
                aa = gemmi.find_tabulated_residue(residue.name).one_letter_code
                ca_atom = next((atom for atom in residue if atom.name.strip() == "CA"), None)
                residues[int(residue.seqid.num)] = {
                    "aa": aa, "atoms": atoms,
                    "plddt": float(ca_atom.b_iso) if ca_atom is not None else float("nan"),
                }
            if residues:
                chains.append(residues)
        return max(chains, key=len) if chains else {}

    def get(self, protein_id: str) -> tuple[dict[int, dict[str, Any]], str]:
        if protein_id in self.cache:
            value = self.cache.pop(protein_id)
            self.cache[protein_id] = value
            return value
        if protein_id not in self.index:
            return {}, "NO_FROZEN_STRUCTURE"
        try:
            record = self.index[protein_id]
            with self.tar_path.open("rb") as handle:
                handle.seek(int(record["tar_data_offset"]))
                content = handle.read(int(record["compressed_size"]))
            value = (self.parse(content), "PASS")
        except Exception as exc:  # noqa: BLE001
            value = ({}, f"{type(exc).__name__}:{exc}")
        self.cache[protein_id] = value
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value


def descriptor(residues: dict[int, dict[str, Any]], position: int, radius: float) -> dict[str, Any] | None:
    center = residues.get(position)
    if not center:
        return None
    atoms = center["atoms"]
    origin = atoms.get("CB", atoms["CA"])
    aa_composition = np.zeros(len(AA), dtype=np.float32)
    properties = np.zeros(len(PROPERTY_SETS), dtype=np.float32)
    radial = np.zeros(5, dtype=np.float32)
    octants = np.zeros(8, dtype=np.float32)
    quality: list[float] = []
    frame: np.ndarray | None = None
    sidechain_vector: np.ndarray | None = None
    if all(name in atoms for name in ["N", "CA", "C"]):
        x_axis = atoms["C"] - atoms["N"]
        x_axis /= max(float(np.linalg.norm(x_axis)), 1e-8)
        guide = atoms.get("CB", atoms["CA"] - atoms["N"]) - atoms["CA"]
        y_axis = guide - np.dot(guide, x_axis) * x_axis
        if np.linalg.norm(y_axis) > 1e-6:
            y_axis /= np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)
            frame = np.vstack([x_axis, y_axis, z_axis])
            if "CB" in atoms:
                vector = atoms["CB"] - atoms["CA"]
                norm = float(np.linalg.norm(vector))
                if norm > 1e-6:
                    sidechain_vector = frame @ (vector / norm)
    count = 0
    for residue in residues.values():
        anchor = residue["atoms"].get("CB", residue["atoms"]["CA"])
        delta = anchor - origin
        distance = float(np.linalg.norm(delta))
        if distance > radius:
            continue
        count += 1
        aa = residue["aa"]
        if aa in AA_INDEX:
            aa_composition[AA_INDEX[aa]] += 1
            for prop_index, values in enumerate(PROPERTY_SETS):
                properties[prop_index] += int(aa in values)
        radial[min(4, int(distance // 2.0))] += 1
        if frame is not None and distance > 1e-6:
            local = frame @ delta
            octant = (1 if local[0] >= 0 else 0) + (2 if local[1] >= 0 else 0) + (4 if local[2] >= 0 else 0)
            octants[octant] += 1
        if math.isfinite(residue["plddt"]):
            quality.append(float(residue["plddt"]))
    for vector in [aa_composition, properties, radial, octants]:
        total = float(vector.sum())
        if total:
            vector /= total
    return {
        "aa": aa_composition, "properties": properties, "radial": radial,
        "octants": octants, "sidechain": sidechain_vector, "residue_count": count,
        "local_plddt": float(np.mean(quality)) if quality else float("nan"),
        "center_plddt": float(center["plddt"]),
    }


def shard_for(query: str, reference: str, shards: int) -> int:
    digest = hashlib.sha256(f"{query}|{reference}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % shards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--chunks", type=int, default=8)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "data/interim/phase07"
    mapping = pd.read_parquet(root / "data/processed/site_mapping.parquet")
    mapping = mapping.loc[mapping["mapping_status"].eq("PASS")].copy()
    mapping["structure_chunk"] = [
        shard_for(query, reference, args.chunks)
        for query, reference in zip(mapping["query_protein_id"], mapping["reference_protein_id"], strict=True)
    ]
    mapping = mapping.loc[mapping["structure_chunk"].eq(args.chunk)].copy()
    needed = set(mapping["query_protein_id"]) | set(mapping["reference_protein_id"])
    store = StructureStore(args.source_root.resolve(), needed)
    rows: list[dict[str, Any]] = []
    for pair_index, ((query_id, reference_id), frame) in enumerate(
        mapping.groupby(["query_protein_id", "reference_protein_id"], sort=True), start=1,
    ):
        query_structure, query_status = store.get(str(query_id))
        reference_structure, reference_status = store.get(str(reference_id))
        cache: dict[tuple[int, int], dict[str, Any]] = {}
        for row in frame.itertuples(index=False):
            query_position = int(row.mapped_query_position)
            reference_position = int(row.reference_site_position)
            position_key = (query_position, reference_position)
            if position_key not in cache:
                values: dict[str, Any] = {
                    "query_structure_status": query_status,
                    "reference_structure_status": reference_status,
                    "local_descriptor_status": "STRUCTURE_UNAVAILABLE",
                }
                if query_structure and reference_structure:
                    all_valid = True
                    for radius in RADII:
                        query_descriptor = descriptor(query_structure, query_position, radius)
                        reference_descriptor = descriptor(reference_structure, reference_position, radius)
                        prefix = f"r{int(radius)}"
                        if query_descriptor is None or reference_descriptor is None:
                            all_valid = False
                            for name in ["aa_composition_cosine", "property_cosine", "radial_cosine", "octant_cosine", "sidechain_direction_cosine", "residue_count_ratio", "query_local_plddt", "reference_local_plddt", "query_center_plddt", "reference_center_plddt"]:
                                values[f"{name}_{prefix}"] = float("nan")
                            continue
                        values[f"aa_composition_cosine_{prefix}"] = cosine(query_descriptor["aa"], reference_descriptor["aa"])
                        values[f"property_cosine_{prefix}"] = cosine(query_descriptor["properties"], reference_descriptor["properties"])
                        values[f"radial_cosine_{prefix}"] = cosine(query_descriptor["radial"], reference_descriptor["radial"])
                        values[f"octant_cosine_{prefix}"] = cosine(query_descriptor["octants"], reference_descriptor["octants"])
                        values[f"sidechain_direction_cosine_{prefix}"] = cosine(query_descriptor["sidechain"], reference_descriptor["sidechain"])
                        values[f"residue_count_ratio_{prefix}"] = min(query_descriptor["residue_count"], reference_descriptor["residue_count"]) / max(query_descriptor["residue_count"], reference_descriptor["residue_count"], 1)
                        values[f"query_local_plddt_{prefix}"] = query_descriptor["local_plddt"]
                        values[f"reference_local_plddt_{prefix}"] = reference_descriptor["local_plddt"]
                        values[f"query_center_plddt_{prefix}"] = query_descriptor["center_plddt"]
                        values[f"reference_center_plddt_{prefix}"] = reference_descriptor["center_plddt"]
                    values["local_descriptor_status"] = "PASS" if all_valid else "MAPPED_RESIDUE_UNAVAILABLE"
                cache[position_key] = values
            rows.append({
                **{name: getattr(row, name) for name in KEY},
                **cache[position_key],
                "feature_provenance": "QUERY_AND_REFERENCE_FROZEN_AFDB_STRUCTURES;NO_QUERY_TRUTH",
            })
        if pair_index % 2000 == 0:
            print(json.dumps({"chunk": args.chunk, "protein_pairs": pair_index, "site_rows": len(rows)}), flush=True)
    output = work / f"local_structure_site_chunk_{args.chunk}.parquet"
    pd.DataFrame(rows).to_parquet(output, index=False, compression="zstd")
    summary = {
        "status": "PASS", "chunk": args.chunk, "site_rows": len(rows),
        "protein_pairs": int(mapping[["query_protein_id", "reference_protein_id"]].drop_duplicates().shape[0]),
        "descriptor_pass": sum(row["local_descriptor_status"] == "PASS" for row in rows),
        "query_ground_truth_used": False,
    }
    (work / f"local_structure_chunk_{args.chunk}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
