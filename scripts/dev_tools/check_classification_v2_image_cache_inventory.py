from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CANONICAL_CACHE_DIR = Path("outputs/classification_v2/image_cache_v2_letterbox")


def main() -> None:
    """Inventory image-cache roots without deleting or rewriting any output."""

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
    """Report canonical and ad hoc cache roots so full runs stay deterministic."""

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

    return {
        "schema_version": "classification_v2_image_cache_inventory_audit_v1",
        "root_dir": str(root_dir),
        "canonical_cache_dir": str(canonical),
        "canonical_cache_dir_exists": canonical.exists(),
        "cache_dir_count": len(cache_dirs),
        "ad_hoc_cache_dir_count": len(ad_hoc_dirs),
        "cache_dirs": [path.as_posix() for path in cache_dirs],
        "ad_hoc_cache_dirs": [path.as_posix() for path in ad_hoc_dirs],
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }


def _norm(path: Path) -> str:
    return path.as_posix().lower().rstrip("/")


if __name__ == "__main__":
    main()
