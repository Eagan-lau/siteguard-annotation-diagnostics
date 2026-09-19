"""Strict, dependency-free FASTA parsing used by the CLI."""

from __future__ import annotations

import hashlib
from pathlib import Path


VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def read_fasta(path: str | Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current is not None:
                    records[current] = _validate(current, "".join(chunks))
                current = line[1:].split()[0]
                if not current:
                    raise ValueError(f"Empty FASTA identifier at line {line_number}")
                if current in records:
                    raise ValueError(f"Duplicate FASTA identifier: {current}")
                chunks = []
            elif current is None:
                raise ValueError(f"Sequence precedes the first FASTA header at line {line_number}")
            else:
                chunks.append(line)
    if current is not None:
        records[current] = _validate(current, "".join(chunks))
    if not records:
        raise ValueError("Input FASTA contains no sequences")
    return records


def _validate(identifier: str, sequence: str) -> str:
    sequence = sequence.replace(" ", "").upper()
    if not sequence:
        raise ValueError(f"Empty sequence for {identifier}")
    invalid = sorted(set(sequence) - VALID_AMINO_ACIDS)
    if invalid:
        raise ValueError(f"Invalid amino-acid symbols for {identifier}: {''.join(invalid)}")
    return sequence
