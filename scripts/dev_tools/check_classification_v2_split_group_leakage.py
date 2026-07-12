from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_SPLIT_MANIFEST = Path(
    "outputs/classification_v2/train_ready_windows/split_manifest.csv"
)
DEFAULT_SPLIT_AUDIT = Path("outputs/classification_v2/train_ready_windows/split_audit.json")
DEFAULT_OUTPUT_JSON = Path(
    "outputs/classification_v2/model_design/split_group_leakage_audit.json"
)
REQUIRED_COLUMNS = ["window_id", "split", "split_group_key"]


def main() -> None:
    """Check split artifacts for video/session-safe grouping and pig_id leakage."""

    parser = argparse.ArgumentParser(
        description="Audit classification_v2 split artifacts for group leakage."
    )
    parser.add_argument("--split-manifest-csv", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--split-audit-json", type=Path, default=DEFAULT_SPLIT_AUDIT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    audit = check_split_group_leakage(args.split_manifest_csv, args.split_audit_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_split_group_leakage(manifest_path: Path, split_audit_path: Path) -> dict[str, Any]:
    """Validate that split artifacts isolate video/session groups across splits."""

    errors: list[str] = []
    warnings: list[str] = []
    if not manifest_path.exists():
        return _base_audit(manifest_path, split_audit_path, [f"missing_manifest={manifest_path}"])

    manifest = pd.read_csv(manifest_path, low_memory=False)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in manifest.columns]
    if missing_columns:
        errors.append(f"missing_required_columns={missing_columns}")

    if "window_uid" in manifest.columns:
        errors.append("forbidden_column_present=window_uid")

    if not missing_columns:
        errors.extend(_check_required_values(manifest))
        errors.extend(_check_group_split_leakage(manifest))
        errors.extend(_check_video_split_leakage(manifest))
        errors.extend(_check_pig_id_group_key_leakage(manifest))

    split_audit = _read_optional_json(split_audit_path, warnings)
    if split_audit.get("leakage_group_count", 0) not in (0, None):
        errors.append(
            f"builder_split_audit_leakage_group_count="
            f"{split_audit.get('leakage_group_count')}"
        )

    return {
        "schema_version": "classification_v2_split_group_leakage_audit_v1",
        "split_manifest_csv": str(manifest_path),
        "split_audit_json": str(split_audit_path),
        "manifest_exists": manifest_path.exists(),
        "split_audit_exists": split_audit_path.exists(),
        "rows": int(len(manifest)),
        "columns": list(manifest.columns),
        "required_columns": REQUIRED_COLUMNS,
        "split_counts": _value_counts(manifest, "split"),
        "group_count": _nunique(manifest, "split_group_key"),
        "video_count": _nunique(manifest, "video_key"),
        "pig_id_count": _nunique(manifest, "pig_id"),
        "group_split_leakage_count": _group_leakage_count(
            manifest, "split_group_key", "split"
        ),
        "video_split_leakage_count": _group_leakage_count(manifest, "video_key", "split"),
        "split_group_key_uses_pig_id_only_count": _pig_id_only_group_key_count(
            manifest
        ),
        "builder_leakage_group_count": split_audit.get("leakage_group_count"),
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }


def _base_audit(
    manifest_path: Path,
    split_audit_path: Path,
    errors: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "classification_v2_split_group_leakage_audit_v1",
        "split_manifest_csv": str(manifest_path),
        "split_audit_json": str(split_audit_path),
        "manifest_exists": manifest_path.exists(),
        "split_audit_exists": split_audit_path.exists(),
        "rows": 0,
        "columns": [],
        "required_columns": REQUIRED_COLUMNS,
        "split_counts": {},
        "group_count": 0,
        "video_count": 0,
        "pig_id_count": 0,
        "group_split_leakage_count": 0,
        "video_split_leakage_count": 0,
        "split_group_key_uses_pig_id_only_count": 0,
        "builder_leakage_group_count": None,
        "warnings": [],
        "errors": errors,
        "valid": False,
    }


def _check_required_values(manifest: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for col in REQUIRED_COLUMNS:
        missing_count = int(manifest[col].isna().sum() + manifest[col].astype(str).eq("").sum())
        if missing_count:
            errors.append(f"missing_values:{col}={missing_count}")
    return errors


def _check_group_split_leakage(manifest: pd.DataFrame) -> list[str]:
    leaked = _leaked_keys(manifest, "split_group_key", "split")
    if leaked:
        return [f"split_group_key_maps_to_multiple_splits={leaked[:20]}"]
    return []


def _check_video_split_leakage(manifest: pd.DataFrame) -> list[str]:
    if "video_key" not in manifest.columns:
        return []
    leaked = _leaked_keys(manifest, "video_key", "split")
    if leaked:
        return [f"video_key_maps_to_multiple_splits={leaked[:20]}"]
    return []


def _check_pig_id_group_key_leakage(manifest: pd.DataFrame) -> list[str]:
    if "pig_id" not in manifest.columns:
        return []
    leaked_rows = manifest.loc[
        manifest.apply(_group_key_is_pig_id_only, axis=1),
        ["split_group_key", "pig_id"],
    ]
    if not leaked_rows.empty:
        examples = leaked_rows.drop_duplicates().head(20).to_dict("records")
        return [f"split_group_key_uses_pig_id_only={examples}"]
    return []


def _group_key_is_pig_id_only(row: pd.Series) -> bool:
    group_key = _norm(row.get("split_group_key"))
    pig_id = _norm(row.get("pig_id"))
    if not group_key or not pig_id:
        return False
    if group_key == pig_id:
        return True
    parts = [part for part in re.split(r"[^a-z0-9]+", group_key) if part]
    ignored = {"pig", "id", "track", "source", "dataset"}
    useful_parts = [part for part in parts if part not in ignored]
    return bool(useful_parts) and all(part == pig_id for part in useful_parts)


def _leaked_keys(manifest: pd.DataFrame, key_col: str, split_col: str) -> list[str]:
    if key_col not in manifest.columns or split_col not in manifest.columns:
        return []
    grouped = manifest.groupby(key_col, dropna=False)[split_col].nunique(dropna=False)
    return sorted(str(key) for key in grouped.loc[grouped > 1].index.tolist())


def _group_leakage_count(manifest: pd.DataFrame, key_col: str, split_col: str) -> int:
    return len(_leaked_keys(manifest, key_col, split_col))


def _pig_id_only_group_key_count(manifest: pd.DataFrame) -> int:
    if "pig_id" not in manifest.columns or "split_group_key" not in manifest.columns:
        return 0
    return int(manifest.apply(_group_key_is_pig_id_only, axis=1).sum())


def _read_optional_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        warnings.append(f"missing_optional_split_audit={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _value_counts(manifest: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in manifest.columns:
        return {}
    return {str(key): int(value) for key, value in manifest[col].value_counts().items()}


def _nunique(manifest: pd.DataFrame, col: str) -> int:
    if col not in manifest.columns:
        return 0
    return int(manifest[col].nunique(dropna=False))


def _norm(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


if __name__ == "__main__":
    main()
