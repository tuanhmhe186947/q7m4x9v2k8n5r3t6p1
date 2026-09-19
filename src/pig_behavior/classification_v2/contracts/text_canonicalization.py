"""Versioned canonical text bytes for scientific-contract rendering."""

from __future__ import annotations

import hashlib
from typing import Any

TEXT_CANONICALIZATION_ID = (
    "policy.classification_v2.scientific_contract_text"
)
TEXT_CANONICALIZATION_VERSION = (
    "classification_v2.scientific_contract_text.v1"
)
UTF8_BOM = b"\xef\xbb\xbf"


def text_canonicalization_contract() -> dict[str, Any]:
    """Return the exact checkout-independent text authority."""

    return {
        "text_canonicalization_id": TEXT_CANONICALIZATION_ID,
        "text_canonicalization_version": TEXT_CANONICALIZATION_VERSION,
        "encoding": "UTF-8",
        "newline_convention": "LF",
        "terminal_newline_policy": "EXACTLY_ONE",
        "bom_policy": "FORBID_UTF8_BOM",
        "trailing_whitespace_policy": "REJECT_SPACE_OR_TAB",
    }


def canonicalize_contract_text_bytes(payload: bytes) -> bytes:
    """Return canonical UTF-8/LF bytes or reject forbidden text."""

    if payload.startswith(UTF8_BOM):
        raise ValueError("TEXT_CANONICALIZATION_UTF8_BOM_FORBIDDEN")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("TEXT_CANONICALIZATION_INVALID_UTF8") from exc
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for line_number, line in enumerate(normalized.split("\n"), start=1):
        if line.endswith((" ", "\t")):
            raise ValueError(
                "TEXT_CANONICALIZATION_TRAILING_WHITESPACE:"
                f"{line_number}"
            )
    return (normalized.rstrip("\n") + "\n").encode("utf-8")


def canonical_contract_text_sha256(payload: bytes) -> str:
    """Hash text only after applying the declared canonicalization."""

    return hashlib.sha256(
        canonicalize_contract_text_bytes(payload)
    ).hexdigest()


__all__ = [
    "TEXT_CANONICALIZATION_ID",
    "TEXT_CANONICALIZATION_VERSION",
    "canonical_contract_text_sha256",
    "canonicalize_contract_text_bytes",
    "text_canonicalization_contract",
]
