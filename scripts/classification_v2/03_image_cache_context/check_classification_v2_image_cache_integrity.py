from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Check an image-cache checksum sidecar against cache files.")
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=Path("outputs/classification_v2/image_cache_v2_letterbox/manifest.csv"),
    )
    parser.add_argument(
        "--integrity-manifest",
        type=Path,
        default=Path("outputs/classification_v2/image_cache_v2_letterbox/image_cache_integrity_manifest.csv"),
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--full", action="store_true", help="Rehash every cache file for a release gate.")
    parser.add_argument(
        "--output-audit",
        type=Path,
        default=Path("outputs/classification_v2/image_cache_v2_letterbox/check_image_cache_integrity_audit.json"),
    )
    args = parser.parse_args()
    audit = check_integrity_manifest(
        cache_manifest=args.cache_manifest,
        integrity_manifest=args.integrity_manifest,
        sample_size=args.sample_size,
        full=args.full,
    )
    args.output_audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["valid"]:
        raise SystemExit(2)


def check_integrity_manifest(
    *,
    cache_manifest: Path,
    integrity_manifest: Path,
    sample_size: int,
    full: bool,
) -> dict[str, Any]:
    """Validate sidecar lineage, then rehash a deterministic sample or all files."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    cache = pd.read_csv(cache_manifest, usecols=["image_context_id", "cache_path"], low_memory=False)
    integrity = pd.read_csv(integrity_manifest, low_memory=False)
    required = {"image_context_id", "cache_path", "cache_size_bytes", "cache_sha256", "integrity_status"}
    missing_columns = sorted(required.difference(integrity.columns))
    errors: list[str] = []
    if missing_columns:
        return {
            "schema_version": "classification_v2_image_cache_integrity_check_v1",
            "valid": False,
            "errors": [f"missing_integrity_columns={missing_columns}"],
        }
    duplicate_cache = int(cache["image_context_id"].duplicated().sum())
    duplicate_integrity = int(integrity["image_context_id"].duplicated().sum())
    cache_by_id = cache.set_index("image_context_id")["cache_path"].astype(str)
    integrity_by_id = integrity.set_index("image_context_id")
    cache_ids = set(cache_by_id.index.astype(str))
    integrity_ids = set(integrity_by_id.index.astype(str))
    missing_ids = cache_ids.difference(integrity_ids)
    unexpected_ids = integrity_ids.difference(cache_ids)
    common_ids = sorted(cache_ids.intersection(integrity_ids))
    path_mismatch = sum(
        str(cache_by_id.loc[context_id]) != str(integrity_by_id.loc[context_id, "cache_path"])
        for context_id in common_ids
    )
    if duplicate_cache:
        errors.append(f"duplicate_cache_ids={duplicate_cache}")
    if duplicate_integrity:
        errors.append(f"duplicate_integrity_ids={duplicate_integrity}")
    if missing_ids:
        errors.append(f"missing_integrity_ids={len(missing_ids)}")
    if unexpected_ids:
        errors.append(f"unexpected_integrity_ids={len(unexpected_ids)}")
    if path_mismatch:
        errors.append(f"cache_path_mismatch_rows={path_mismatch}")

    verify_ids = common_ids
    verification_mode = "full" if full else "sample"
    if not full and len(verify_ids) > sample_size:
        positions = np.linspace(0, len(verify_ids) - 1, sample_size, dtype=int)
        verify_ids = [verify_ids[int(position)] for position in positions]
    content_mismatches = 0
    for context_id in verify_ids:
        expected = integrity_by_id.loc[context_id]
        actual = _hash_cache_entry(cache_manifest.parent, context_id, str(cache_by_id.loc[context_id]))
        if (
            actual["integrity_status"] != str(expected["integrity_status"])
            or int(actual["cache_size_bytes"]) != int(expected["cache_size_bytes"])
            or str(actual["cache_sha256"]) != str(expected["cache_sha256"])
        ):
            content_mismatches += 1
    if content_mismatches:
        errors.append(f"cache_content_mismatches={content_mismatches}")
    return {
        "schema_version": "classification_v2_image_cache_integrity_check_v1",
        "cache_manifest": str(cache_manifest),
        "integrity_manifest": str(integrity_manifest),
        "cache_rows": int(len(cache)),
        "integrity_rows": int(len(integrity)),
        "verification_mode": verification_mode,
        "verified_file_rows": int(len(verify_ids)),
        "duplicate_cache_ids": duplicate_cache,
        "duplicate_integrity_ids": duplicate_integrity,
        "missing_integrity_ids": int(len(missing_ids)),
        "unexpected_integrity_ids": int(len(unexpected_ids)),
        "cache_path_mismatch_rows": int(path_mismatch),
        "cache_content_mismatches": int(content_mismatches),
        "errors": errors,
        "valid": not errors,
    }


def _hash_cache_entry(base: Path, context_id: str, cache_value: str) -> dict[str, Any]:
    """Hash one cache file using the same stable record contract as the builder."""

    path = Path(cache_value)
    if not path.is_absolute():
        path = base / path
    if not path.exists():
        return {
            "image_context_id": context_id,
            "cache_size_bytes": 0,
            "cache_sha256": "",
            "integrity_status": "missing",
        }
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "image_context_id": context_id,
        "cache_size_bytes": int(path.stat().st_size),
        "cache_sha256": digest.hexdigest(),
        "integrity_status": "ok",
    }


if __name__ == "__main__":
    main()
