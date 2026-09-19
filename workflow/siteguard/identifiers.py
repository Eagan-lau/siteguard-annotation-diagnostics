"""Canonical identifier handling for SiteGuard catalytic annotations.

The helpers intentionally implement only unambiguous presentation-level
normalization.  They do not resolve obsolete identifiers, infer reactions, or
turn a broader EC label into a more specific one.
"""

from __future__ import annotations

import re


_EC_PREFIX = re.compile(r"^EC\s*:\s*", flags=re.IGNORECASE)
_RHEA_PREFIX = re.compile(r"^RHEA\s*:\s*", flags=re.IGNORECASE)


def canonicalize_ec(value: object, *, level: int | None = None) -> str:
    """Return an EC identifier without a presentation prefix or whitespace.

    Empty values remain empty.  Non-empty malformed values fail closed rather
    than being compared as opaque strings.
    """
    original = str(value).strip()
    if not original:
        return ""
    normalized = _EC_PREFIX.sub("", original)
    normalized = re.sub(r"\s+", "", normalized)
    parts = normalized.split(".")
    if level is not None and len(parts) != level:
        raise ValueError(f"Expected EC-L{level} identifier, received: {original!r}")
    if len(parts) not in {3, 4} or any(not re.fullmatch(r"\d+|-", part) for part in parts):
        raise ValueError(f"Malformed EC identifier: {original!r}")
    return ".".join(parts)


def canonicalize_rhea(value: object) -> str:
    """Return the canonical ``RHEA:<digits>`` presentation of a Rhea ID."""
    original = str(value).strip()
    if not original:
        return ""
    normalized = _RHEA_PREFIX.sub("", original)
    normalized = re.sub(r"\s+", "", normalized)
    if not re.fullmatch(r"\d+", normalized):
        raise ValueError(f"Malformed Rhea identifier: {original!r}")
    return f"RHEA:{normalized}"


def ec_l3_from_l4(value: object) -> str:
    """Return the explicit EC-L3 ancestor of a valid EC-L4 identifier."""
    ec_l4 = canonicalize_ec(value, level=4)
    return ".".join(ec_l4.split(".")[:3])
