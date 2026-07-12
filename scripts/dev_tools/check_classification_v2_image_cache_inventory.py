from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CANONICAL_CACHE_DIR = Path("outputs/classification_v2/image_cache_v2_letterbox")
ACTIVE_REFERENCE_ROOTS = (
    Path("configs/classification_v2"),
    Path("outputs/classification_v2/training_snapshots"),
)


def main() -> None:
    """Inventory image-cache roots without deleting or rewriting cache data."""

    parser = argparse.ArgumentParser(
        description="Inventory classification_v2 image cache roots."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("outputs/classification_v2"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/image_cache_inventory_audit.json"
        ),
    )
    args = parser.parse_args()

    audit = check_image_cache_inventory(args.root_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_image_cache_inventory(root_dir: Path) -> dict[str, Any]:
    """Report canonical/ad-hoc cache roots and active training references."""

    errors: list[str] = []
    warnings: list[str] = []
    cache_dirs = sorted(
        [path for path in root_dir.glob("image_cache*") if path.is_dir()],
        key=lambda path: path.as_posix().lower(),
    )
    canonical = CANONICAL_CACHE_DIR
    if not canonical.exists():
        errors.append(f"missing_canonical_image_cache_dir={canonical}")
    ad_hoc_dirs = [path for path in cache_dirs if _norm(path) != _norm(canonical)]
    if ad_hoc_dirs:
        warnings.append(
            "non_canonical_image_cache_dirs_present="
            f"{[path.as_posix() for path in ad_hoc_dirs]}"
        )

    active_references = _find_cache_references(
        ad_hoc_dirs,
        roots=ACTIVE_REFERENCE_ROOTS,
    )
    active_reference_count = sum(len(paths) for paths in active_references.values())
    if active_reference_count:
        errors.append(
            "non_canonical_cache_referenced_by_active_training_contracts="
            f"{active_references}"
        )

    return {
        "schema_version": "classification_v2_image_cache_inventory_audit_v2",
        "root_dir": str(root_dir),
        "canonical_cache_dir": str(canonical),
        "canonical_cache_dir_exists": canonical.exists(),
        "cache_dir_count": len(cache_dirs),
        "ad_hoc_cache_dir_count": len(ad_hoc_dirs),
        "cache_dirs": [path.as_posix() for path in cache_dirs],
        "ad_hoc_cache_dirs": [path.as_posix() for path in ad_hoc_dirs],
        "active_reference_roots": [path.as_posix() for path in ACTIVE_REFERENCE_ROOTS],
        "ad_hoc_active_training_reference_count": active_reference_count,
        "ad_hoc_active_training_references": active_references,
        "ad_hoc_cache_policy": (
            "residual_smoke_dirs_allowed_only_when_unreferenced_by_active_"
            "training_configs_or_snapshots"
        ),
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }


def _find_cache_references(
    cache_dirs: list[Path],
    *,
    roots: tuple[Path, ...],
) -> dict[str, list[str]]:
    """Return active config/snapshot files that still point at ad-hoc caches."""

    references: dict[str, list[str]] = {}
    needles = {
        path.as_posix(): {
            path.as_posix(),
            str(path),
            path.as_posix().replace("/", "\\"),
        }
        for path in cache_dirs
    }
    for root in roots:
        if not root.exists():
            continue
        for file_path in _iter_text_files(root):
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for cache_dir, variants in needles.items():
                if any(variant in text for variant in variants):
                    references.setdefault(cache_dir, []).append(file_path.as_posix())
    return {key: sorted(value) for key, value in sorted(references.items())}


def _iter_text_files(root: Path) -> list[Path]:
    """List deterministic text files used by active contracts."""

    suffixes = {".json", ".py", ".md"}
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        ],
        key=lambda path: path.as_posix().lower(),
    )


def _norm(path: Path) -> str:
    return path.as_posix().lower().rstrip("/")


if __name__ == "__main__":
    main()
