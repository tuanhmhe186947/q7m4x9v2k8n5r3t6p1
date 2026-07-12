"""Verify lineage artifact hashes without opening model payloads."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import finish, load_json, sha256_file


def audit(manifest: Path, root: Path | None, required: set[str]) -> dict[str, object]:
    """Validate paths and SHA256 values declared by a lineage manifest."""
    payload = load_json(manifest)
    records = payload.get("artifacts", [])
    base = root.resolve() if root else manifest.parent.resolve()
    errors: list[str] = []
    observed: dict[str, dict[str, object]] = {}
    for record in records:
        name = str(record.get("name", "")).strip()
        raw_path = str(record.get("path", "")).strip()
        expected = str(record.get("sha256", "")).strip().lower()
        if not name or name in observed:
            errors.append(f"invalid_or_duplicate_artifact_name={name}")
            continue
        path = Path(raw_path)
        path = path if path.is_absolute() else base / path
        exists = path.is_file()
        actual = sha256_file(path) if exists else ""
        matches = bool(expected and actual == expected)
        observed[name] = {
            "path": str(path),
            "exists": exists,
            "hash_matches": matches,
        }
        if not exists:
            errors.append(f"missing_artifact={name}")
        elif not matches:
            errors.append(f"hash_mismatch={name}")
    missing_names = sorted(required - set(observed))
    if missing_names:
        errors.append(f"missing_required_artifacts={missing_names}")
    return {
        "check": "artifact_hashes",
        "artifacts": observed,
        "required_names": sorted(required),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--required-names", default="")
    args = parser.parse_args()
    required = {value for value in args.required_names.split(",") if value}
    return finish(audit(args.manifest_json, args.root, required))


if __name__ == "__main__":
    raise SystemExit(main())
