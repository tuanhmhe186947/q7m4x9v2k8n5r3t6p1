"""Audit legacy rows against an explicit source-video exclusion policy.

This utility is a read-only audit of the legacy metadata table.  It does not
create or alter the exclusion policy and it is not the CVAT six-anchor rebuild
authority.  The caller must provide all input and output paths so a new
lineage cannot silently read or overwrite a project-root artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

CANONICAL_KEY_RE = re.compile(
    r"^(?P<day>pigs\d{6}[a-z]?)/(?P<clip>\d{1,6})$",
    re.IGNORECASE,
)
NESTED_VIDEO_RE = re.compile(
    r"/(?P<day>pigs\d{6}[a-z]?)/pigs\d{6}[a-z]?/"
    r"(?P<clip>\d{1,6})/color\.mp4(?:$|[?#])",
    re.IGNORECASE,
)
VIDEO_FILENAME_RE = re.compile(
    r"(?P<day>pigs\d{6}[a-z]?)[_-](?P<clip>\d{1,6})"
    r"(?:[_-]30fps)?\.mp4(?:$|[?#])",
    re.IGNORECASE,
)
SOURCE_COLUMN_CANDIDATES = (
    "video_final",
    "source_video_resolved",
    "color_video_path",
    "video_file",
    "source_video_key",
)


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_source_video_key(value: object) -> str:
    """Return ``pigsDDMMYY/NNNNNN`` or an empty string when unresolved."""
    raw = _text(value).replace("\\", "/").strip().lower()
    if not raw:
        return ""
    raw = re.sub(r"/{2,}", "/", raw)

    match = CANONICAL_KEY_RE.fullmatch(raw)
    if match is None:
        match = NESTED_VIDEO_RE.search(raw)
    if match is None:
        match = VIDEO_FILENAME_RE.search(raw)
    if match is None:
        return ""

    day = match.group("day").lower()
    clip = match.group("clip").zfill(6)
    return f"{day}/{clip}"


def choose_source_column(
    frame: pd.DataFrame,
    requested: str,
) -> str:
    if requested != "auto":
        if requested not in frame.columns:
            raise ValueError(
                f"source column is missing: {requested}; "
                f"available={list(frame.columns)}"
            )
        return requested
    for column in SOURCE_COLUMN_CANDIDATES:
        if column in frame.columns:
            return column
    raise ValueError(
        "could not choose a source column; expected one of "
        f"{SOURCE_COLUMN_CANDIDATES}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_duplicate_videos(
    legacy: pd.DataFrame,
    exclusions: pd.DataFrame,
    *,
    source_column: str,
    allow_unresolved: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a duplicate preview and an auditable, fail-closed summary."""
    if "source_video_key" not in exclusions.columns:
        raise ValueError(
            "exclusion CSV must contain canonical column source_video_key"
        )
    if exclusions["source_video_key"].map(normalize_source_video_key).eq("").any():
        raise ValueError("exclusion CSV contains blank or invalid source_video_key")

    source_keys = legacy[source_column].map(normalize_source_video_key)
    unresolved = source_keys.eq("")
    if unresolved.any() and not allow_unresolved:
        sample = legacy.loc[unresolved, source_column].head(5).tolist()
        raise ValueError(
            f"unresolved source keys={int(unresolved.sum())}; sample={sample}"
        )

    exclusion_keys = set(
        exclusions["source_video_key"].map(normalize_source_video_key)
    )
    duplicate_mask = source_keys.isin(exclusion_keys)
    preview = legacy.loc[duplicate_mask].copy()
    preview["source_video_key_audit"] = source_keys.loc[duplicate_mask].to_numpy()
    preview["duplicate_video"] = True

    counts: dict[str, Any] = {
        "legacy_rows": int(len(legacy)),
        "legacy_unique_source_keys": int(source_keys[source_keys.ne("")].nunique()),
        "resolved_source_rows": int((~unresolved).sum()),
        "unresolved_source_rows": int(unresolved.sum()),
        "excluded_source_keys": int(len(exclusion_keys)),
        "duplicate_rows": int(duplicate_mask.sum()),
        "duplicate_source_keys": int(source_keys[duplicate_mask].nunique()),
    }
    if {"group_id", "pig_id"}.issubset(legacy.columns):
        counts["duplicate_group_pig"] = int(
            legacy.loc[duplicate_mask, ["group_id", "pig_id"]]
            .drop_duplicates()
            .shape[0]
        )

    audit = {
        "schema_version": 2,
        "status": (
            "DUPLICATES_FOUND" if counts["duplicate_rows"] else "PASS_NO_DUPLICATES"
        ),
        "policy": {
            "source_column": source_column,
            "exclusion_column": "source_video_key",
            "unresolved_policy": (
                "allowed_and_reported" if allow_unresolved else "fail_closed"
            ),
            "this_tool_does_not_modify_exclusion_policy": True,
        },
        "counts": counts,
        "duplicate_source_key_counts": {
            str(key): int(value)
            for key, value in source_keys[duplicate_mask].value_counts().items()
        },
        "errors": [],
        "warnings": (
            [f"unresolved_source_rows={int(unresolved.sum())}"]
            if unresolved.any()
            else []
        ),
    }
    return preview.reset_index(drop=True), audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-csv", type=Path, required=True)
    parser.add_argument("--exclude-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument(
        "--source-column",
        default="auto",
        help="Legacy source column; auto selects a known column deterministically.",
    )
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Keep unresolved rows out of the hit set but report them as warnings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and audit inputs without writing preview or audit files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of the explicitly supplied output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy_path = args.legacy_csv.expanduser().resolve()
    exclude_path = args.exclude_csv.expanduser().resolve()
    output_path = args.output_csv.expanduser().resolve()
    audit_path = args.audit_json.expanduser().resolve()

    if output_path in {legacy_path, exclude_path}:
        raise ValueError("output-csv cannot overwrite an input CSV")
    if audit_path in {legacy_path, exclude_path}:
        raise ValueError("audit-json cannot overwrite an input CSV")
    if not args.dry_run and not args.overwrite:
        existing = [path for path in (output_path, audit_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "output exists; use a fresh lineage root or --overwrite: "
                + ", ".join(str(path) for path in existing)
            )

    legacy = pd.read_csv(legacy_path, low_memory=False)
    exclusions = pd.read_csv(exclude_path, low_memory=False)
    source_column = choose_source_column(legacy, args.source_column)
    preview, audit = audit_duplicate_videos(
        legacy,
        exclusions,
        source_column=source_column,
        allow_unresolved=args.allow_unresolved,
    )
    audit["inputs"] = {
        "legacy_csv": str(legacy_path),
        "legacy_sha256": _sha256(legacy_path),
        "exclude_csv": str(exclude_path),
        "exclude_sha256": _sha256(exclude_path),
    }
    audit["outputs"] = {
        "preview_csv": str(output_path),
        "audit_json": str(audit_path),
    }

    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    preview.to_csv(output_path, index=False)
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"preview_csv={output_path}")
    print(f"audit_json={audit_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, FileExistsError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
