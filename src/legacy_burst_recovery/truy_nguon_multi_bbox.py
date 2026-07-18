"""Trace legacy burst rows to their source video and manifest provenance.

This script produces a historical metadata scaffold. It is not behavior or bbox
authority for the current CVAT six-anchor rebuild. Current annotation authority
comes from native CVAT ``k0..k5`` rows; this trace only resolves group/video/path
metadata and preserves the old evidence for audit.

Main fix:
- The old script kept only order == 3 (center keyframe), so the output had only one
  bbox per group_id + pig_id.
- This version also writes an all-keyframe/per-frame bbox CSV that keeps bbox
  annotations for all 6 burst frames.

Recommended output for multi-GT anchor tracking:
    old_burst_all_keyframe_bboxes_combined.csv

Backward-compatible output:
    old_burst_center_keyframes_combined.csv
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from legacy_burst_recovery.path_utils import map_drive_path

# =========================
# DEFAULT CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOOGLE_DRIVE_CANDIDATES = (
    Path("G:/My Drive"),
    Path("G:/"),
)
DEFAULT_SEARCH_ROOT_NAMES = tuple(
    f"pig-selected_frame_attribute_({index})" for index in range(1, 6)
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "legacy_source_trace"
DEFAULT_OUT_CENTER = DEFAULT_OUTPUT_ROOT / "old_burst_center_keyframes_combined.csv"
DEFAULT_OUT_ALL_BBOX = (
    DEFAULT_OUTPUT_ROOT / "old_burst_all_keyframe_bboxes_combined.csv"
)
DEFAULT_OUT_AUDIT = DEFAULT_OUTPUT_ROOT / "legacy_gt_support_audit.csv"
DEFAULT_OUT_MISSING = DEFAULT_OUTPUT_ROOT / "missing_old_burst_groups.csv"
DEFAULT_OUT_LINEAGE = DEFAULT_OUTPUT_ROOT / "legacy_source_trace_lineage.json"

EXPECTED_BEHAVIORS = {
    "drink",
    "eat",
    "explore",
    "fight",
    "lying",
    "move",
    "playwithtoy",
    "sitting",
    "social-nose",
    "stand",
}
EXPECTED_HIDDEN = {"Yes", "No"}
EXPECTED_ORDERS = tuple(range(6))
EXPECTED_BEHAVIOR_AUTHORITY_POLICY = "first_task_frame_per_group_pig"
IMAGE_NAME_PATTERN = re.compile(
    r"^burst_(?P<stream>[^_]+)_(?P<video_code>[0-9a-fA-F]+)_"
    r"(?P<center_ms_img>\d+)_f(?P<frame_idx>\d+)_k(?P<k>\d+)"
    r"\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)
GROUP_ID_PATTERN = re.compile(
    r"^burst_(?P<group_stream>[^_]+)_"
    r"(?P<group_video_code>[0-9a-fA-F]+)_(?P<group_center_ms>\d+)$"
)
TRACE_SCHEMA_VERSION = 3


# =========================
# HELPERS
# =========================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Trace old burst labels to source video and export per-frame "
            "legacy bbox GT."
        )
    )
    parser.add_argument(
        "--behavior-csv",
        required=True,
        help="Fresh behavior CSV created from the current native CVAT export.",
    )
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=None,
        help=(
            "Mounted Google Drive My Drive root. When omitted, the script "
            "discovers the current Windows Google Drive mount."
        ),
    )
    parser.add_argument(
        "--search-roots",
        nargs="+",
        default=None,
        help=(
            "Explicit provenance roots. When omitted, all five project roots "
            "are resolved below --drive-root, including Drive shortcuts."
        ),
    )
    parser.add_argument("--out-center-csv", default=DEFAULT_OUT_CENTER)
    parser.add_argument("--out-all-bbox-csv", default=DEFAULT_OUT_ALL_BBOX)
    parser.add_argument("--out-audit-csv", default=DEFAULT_OUT_AUDIT)
    parser.add_argument("--out-missing-csv", default=DEFAULT_OUT_MISSING)
    parser.add_argument("--out-lineage-json", default=DEFAULT_OUT_LINEAGE)
    parser.add_argument(
        "--allow-incomplete-actor-keys",
        action="store_true",
        help=(
            "Exclude actor keys missing one or more of k0..k5 and record them "
            "in the audit. Without this flag, incomplete actor keys fail."
        ),
    )
    parser.add_argument(
        "--require-video-exists",
        action="store_true",
        help="Fail when a resolved source video path does not exist.",
    )
    parser.add_argument(
        "--allow-unresolved-video",
        action="store_true",
        help=(
            "Keep rows without a manifest/candidate video mapping and record "
            "them in the audit. Without this flag, unresolved groups fail."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run discovery and every validation without writing output files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing outputs after all validations pass.",
    )
    return parser.parse_args(argv)


def md5_code(s: str) -> str:
    return hashlib.md5(str(s).encode("utf-8")).hexdigest()[:8]


def find_files(roots, patterns):
    files = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for pattern in patterns:
            files.extend(root.rglob(pattern))
    return sorted(set(files))


def col_or_na(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(pd.NA, index=df.index)


def coalesce_series(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    result = primary.copy()
    missing = ~primary.map(_nonempty)
    if not missing.any():
        return result
    result = result.astype(object)
    result.loc[missing] = fallback.loc[missing]
    return result


def first_existing_cols(cols, df):
    return [c for c in cols if c in df.columns]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root without assuming the script's directory."""
    current = (start or Path(__file__)).expanduser().resolve()
    if current.is_file() or current.suffix:
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() or (
            candidate / ".git"
        ).exists():
            return candidate
    raise RuntimeError(f"project root not found from {current}")


def resolve_google_drive_root(
    explicit_root: Path | None,
    *,
    required: bool,
) -> Path | None:
    """Resolve the Windows ``My Drive`` directory used for runtime paths."""
    candidates = (
        (explicit_root,)
        if explicit_root is not None
        else DEFAULT_GOOGLE_DRIVE_CANDIDATES
    )
    checked: list[str] = []
    for raw_candidate in candidates:
        candidate = Path(raw_candidate).expanduser()
        if candidate.name.lower() != "my drive":
            my_drive = candidate / "My Drive"
            checked.append(str(my_drive))
            if my_drive.is_dir():
                return my_drive.resolve()
        checked.append(str(candidate))
        if candidate.is_dir():
            return candidate.resolve()
    if required:
        raise FileNotFoundError(
            "Google Drive My Drive root not found; checked=" + str(checked)
        )
    return None


def resolve_default_search_roots(
    drive_root: Path,
) -> tuple[list[Path], list[dict[str, str]]]:
    """Resolve all five provenance roots, including Drive shortcut targets."""
    shortcut_root = drive_root.parent / ".shortcut-targets-by-id"
    roots: list[Path] = []
    records: list[dict[str, str]] = []
    for root_name in DEFAULT_SEARCH_ROOT_NAMES:
        direct = drive_root / root_name
        if direct.is_dir():
            resolved = direct.resolve()
            roots.append(resolved)
            records.append(
                {
                    "name": root_name,
                    "resolution": "direct",
                    "path": str(resolved),
                }
            )
            continue

        matches = (
            sorted(
                path.resolve()
                for path in shortcut_root.glob(f"*/{root_name}")
                if path.is_dir()
            )
            if shortcut_root.is_dir()
            else []
        )
        if not matches:
            raise FileNotFoundError(
                f"provenance root not found: {root_name}; "
                f"direct={direct}; shortcut_root={shortcut_root}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous Drive shortcut target for {root_name}: "
                f"{[str(path) for path in matches]}"
            )
        roots.append(matches[0])
        records.append(
            {
                "name": root_name,
                "resolution": "shortcut_target",
                "path": str(matches[0]),
            }
        )
    return roots, records


def resolve_search_roots(
    explicit_roots: list[str] | None,
    drive_root: Path | None,
) -> tuple[list[Path], list[dict[str, str]]]:
    """Resolve explicit roots or the complete project default root set."""
    if explicit_roots:
        roots = [Path(value).expanduser().resolve() for value in explicit_roots]
        missing = [str(path) for path in roots if not path.is_dir()]
        if missing:
            raise FileNotFoundError(
                f"provenance search roots missing={len(missing)}; sample={missing[:10]}"
            )
        if len(set(roots)) != len(roots):
            raise ValueError("provenance search roots must be distinct")
        return roots, [
            {
                "name": path.name,
                "resolution": "explicit",
                "path": str(path),
            }
            for path in roots
        ]
    if drive_root is None:
        raise FileNotFoundError(
            "Google Drive root is required when --search-roots is omitted"
        )
    return resolve_default_search_roots(drive_root)


def validate_output_paths(
    args,
    *,
    search_roots: list[Path] | None = None,
    script_path: Path | None = None,
) -> dict[str, Path]:
    output_values = {
        "center_csv": args.out_center_csv,
        "all_bbox_csv": args.out_all_bbox_csv,
        "audit_csv": args.out_audit_csv,
        "missing_csv": args.out_missing_csv,
        "lineage_json": args.out_lineage_json,
    }
    outputs = {
        name: Path(value).expanduser().resolve()
        for name, value in output_values.items()
    }
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("output paths must be distinct")

    project_root = find_project_root(script_path or Path(__file__))
    forbidden_roots = [
        (project_root / name).resolve()
        for name in ("data", "src", "scripts", "tests")
    ]
    for name, path in outputs.items():
        if any(_is_within(path, root) for root in forbidden_roots):
            raise ValueError(f"{name} cannot be written under a code directory: {path}")

    behavior_path = Path(args.behavior_csv).expanduser().resolve()
    if behavior_path in outputs.values():
        raise ValueError("an output path cannot overwrite the behavior input")

    resolved_search_roots = search_roots or [
        Path(root).expanduser().resolve()
        for root in (args.search_roots or [])
    ]
    for name, path in outputs.items():
        if any(_is_within(path, root) for root in resolved_search_roots):
            raise ValueError(
                f"{name} cannot be written inside a provenance search root: {path}"
            )

    if not args.dry_run and not args.overwrite:
        existing = [str(path) for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "output already exists; use a fresh path or --overwrite: "
                + ", ".join(existing)
            )
    return outputs


def _canonical_authority_value(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value).strip().replace("\\", "/")


def _canonical_video_value(value: Any) -> str:
    return _canonical_authority_value(value).lower()


def resolve_video_local_path(
    value: Any,
    *,
    drive_root: Path | None,
) -> str:
    """Map a canonical Colab video path to its Windows runtime path."""
    if not _nonempty(value):
        return ""
    if drive_root is None:
        return str(Path(str(value)).expanduser())
    mapped = map_drive_path(value, drive_root)
    return "" if mapped is None else str(mapped)


def _nonempty(value: Any) -> bool:
    if pd.isna(value):
        return False
    return bool(str(value).strip())


def _boolean_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _group_video_code(group_id: Any) -> str | None:
    match = GROUP_ID_PATTERN.fullmatch(str(group_id).strip())
    return match.group("group_video_code").lower() if match else None


def _video_hash_matches_group(video: Any, group_id: Any) -> bool:
    expected = _group_video_code(group_id)
    return bool(expected and _nonempty(video) and md5_code(video) == expected)


def collapse_equivalent_authority_rows(
    frame: pd.DataFrame,
    *,
    key: str,
    authority_columns: list[str],
    source_column: str,
    label: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    work = frame.copy()
    available = [column for column in authority_columns if column in work.columns]
    work["_authority_signature"] = work.apply(
        lambda row: tuple(
            _canonical_authority_value(row[column]) for column in available
        ),
        axis=1,
    )
    signature_counts = work.groupby(key, dropna=False)[
        "_authority_signature"
    ].nunique()
    conflict_keys = signature_counts[signature_counts > 1].index.astype(str).tolist()
    if conflict_keys:
        sample = conflict_keys[:10]
        raise ValueError(
            f"conflicting_{label}_authority_keys={len(conflict_keys)}; sample={sample}"
        )

    source_lists = work.groupby(key, dropna=False)[source_column].agg(
        lambda values: "|".join(sorted(set(str(value) for value in values)))
    )
    source_counts = work.groupby(key, dropna=False)[source_column].nunique()
    row_counts = work.groupby(key, dropna=False).size()
    collapsed = (
        work.sort_values([key, source_column])
        .drop_duplicates(subset=[key], keep="first")
        .drop(columns=["_authority_signature"])
    )
    collapsed[source_column] = collapsed[key].map(source_lists)
    collapsed[f"{label}_source_count"] = collapsed[key].map(source_counts).astype(int)
    collapsed[f"{label}_row_count"] = collapsed[key].map(row_counts).astype(int)
    return collapsed.reset_index(drop=True)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_img_name_fields(df: pd.DataFrame) -> pd.DataFrame:
    # Example: burst_color_000dacf1_400_f12_k3.jpg
    parsed = df["img_name"].astype(str).str.extract(IMAGE_NAME_PATTERN)
    df = df.copy()
    for column in parsed.columns:
        df[column] = parsed[column]

    for col in ["center_ms_img", "frame_idx", "k"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Important:
    # - `frame_idx` parsed from img_name is the source video frame index encoded in the file name.
    # - The original `frame` column may be a local/CVAT id, not a source
    #   video frame index.
    df["legacy_frame_index"] = df["frame_idx"].astype("Int64")
    df["frame_index"] = df["legacy_frame_index"]

    if "frame" in df.columns:
        df["cvat_frame_index"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")

    if "order" in df.columns:
        df["legacy_order"] = pd.to_numeric(df["order"], errors="coerce").astype("Int64")

    return df


def parse_behavior_group(df: pd.DataFrame) -> pd.DataFrame:
    # Example: burst_color_000dacf1_400
    parsed = df["group_id"].astype(str).str.extract(GROUP_ID_PATTERN)
    df = df.copy()
    for column in parsed.columns:
        df[column] = parsed[column]
    df["group_center_ms"] = pd.to_numeric(df["group_center_ms"], errors="coerce")
    return df


def candidate_keys(row):
    # bursts_candidates group_id example: pigs081119/color/400
    parts = str(row["group_id"]).replace("\\", "/").split("/")
    stream = parts[1] if len(parts) >= 2 else "color"

    keys = set()

    # Key from the tail of candidate group_id.
    if len(parts) >= 1:
        tail = parts[-1]
        if tail and tail != "nan":
            keys.add(tail)

    # Key from center_ts, with both rounded and truncated ms.
    if pd.notna(row.get("center_ts")):
        center_ms_round = int(round(float(row["center_ts"]) * 1000))
        center_ms_trunc = int(float(row["center_ts"]) * 1000)
        keys.add(str(center_ms_round))
        keys.add(str(center_ms_trunc))

    video_code = md5_code(row["video"])

    return [f"burst_{stream}_{video_code}_{ms}" for ms in sorted(keys)]


def _sample_key_values(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if frame.empty:
        return []
    return [
        "|".join(str(row[column]) for column in columns)
        for _, row in frame[columns].drop_duplicates().head(10).iterrows()
    ]


BEHAVIOR_AUTHORITY_COLUMNS = {
    "behavior_before_authority",
    "behavior_authority_task_frame",
    "behavior_authority_slot",
    "behavior_disagrees_with_authority",
    "behavior_authority_policy",
    "is_burst_first_task_frame",
    "hidden_attribute_present",
    "frame",
}


def validate_behavior_authority_contract(beh: pd.DataFrame) -> list[str]:
    """Reject stale majority-vote CSVs and inconsistent authority evidence."""
    missing = sorted(BEHAVIOR_AUTHORITY_COLUMNS.difference(beh.columns))
    if missing:
        return [f"missing_behavior_authority_columns={missing}"]

    issues: list[str] = []
    invalid_policy = beh["behavior_authority_policy"].astype("string").ne(
        EXPECTED_BEHAVIOR_AUTHORITY_POLICY
    )
    if invalid_policy.any():
        issues.append(
            "invalid_behavior_authority_policy="
            f"{int(invalid_policy.sum())}"
        )

    actor_keys = ["group_id", "pig_id"]
    behavior_counts = beh.groupby(actor_keys, dropna=False)["behavior"].nunique(
        dropna=False
    )
    inconsistent_behavior = behavior_counts[behavior_counts.ne(1)]
    if not inconsistent_behavior.empty:
        issues.append(
            "nonconstant_authority_behavior_actor_keys="
            f"{len(inconsistent_behavior)}"
        )

    authority_slot = pd.to_numeric(
        beh["behavior_authority_slot"],
        errors="coerce",
    )
    invalid_slot = authority_slot.isna() | ~authority_slot.isin(EXPECTED_ORDERS)
    if invalid_slot.any():
        issues.append(f"invalid_behavior_authority_slot={int(invalid_slot.sum())}")
    slot_counts = authority_slot.groupby(
        [beh["group_id"], beh["pig_id"]],
        dropna=False,
    ).nunique(dropna=False)
    inconsistent_slots = slot_counts[slot_counts.ne(1)]
    if not inconsistent_slots.empty:
        issues.append(
            f"inconsistent_authority_slot_actor_keys={len(inconsistent_slots)}"
        )

    first_flags = beh["is_burst_first_task_frame"].map(_boolean_flag)
    invalid_first_flag = first_flags.isna()
    if invalid_first_flag.any():
        issues.append(
            f"invalid_first_task_frame_flags={int(invalid_first_flag.sum())}"
        )
    first_counts = first_flags.eq(True).groupby(
        [beh["group_id"], beh["pig_id"]],
        dropna=False,
    ).sum()
    invalid_first_counts = first_counts[first_counts.ne(1)]
    if not invalid_first_counts.empty:
        issues.append(
            "invalid_first_task_frame_count_actor_keys="
            f"{len(invalid_first_counts)}"
        )

    authority_rows = beh.loc[first_flags.eq(True)].copy()
    if not authority_rows.empty:
        row_order = pd.to_numeric(authority_rows["order"], errors="coerce")
        row_slot = pd.to_numeric(
            authority_rows["behavior_authority_slot"],
            errors="coerce",
        )
        slot_mismatch = row_order.ne(row_slot)
        if slot_mismatch.any():
            issues.append(
                f"first_frame_authority_slot_mismatch={int(slot_mismatch.sum())}"
            )

        row_frame = pd.to_numeric(authority_rows["frame"], errors="coerce")
        authority_frame = pd.to_numeric(
            authority_rows["behavior_authority_task_frame"],
            errors="coerce",
        )
        frame_mismatch = row_frame.ne(authority_frame)
        if frame_mismatch.any():
            issues.append(
                f"first_frame_authority_task_frame_mismatch={int(frame_mismatch.sum())}"
            )

        label_mismatch = authority_rows[
            "behavior_before_authority"
        ].astype("string").ne(authority_rows["behavior"].astype("string"))
        if label_mismatch.any():
            issues.append(
                f"authority_source_label_mismatch={int(label_mismatch.sum())}"
            )

    hidden_present = beh["hidden_attribute_present"].map(_boolean_flag)
    missing_hidden = hidden_present.ne(True)
    if missing_hidden.any():
        issues.append(
            f"missing_explicit_hidden_attribute={int(missing_hidden.sum())}"
        )

    expected_disagreement = beh["behavior_before_authority"].astype("string").ne(
        beh["behavior"].astype("string")
    )
    actual_disagreement = beh["behavior_disagrees_with_authority"].map(
        _boolean_flag
    )
    disagreement_mismatch = actual_disagreement.isna() | actual_disagreement.ne(
        expected_disagreement
    )
    if disagreement_mismatch.any():
        issues.append(
            "behavior_disagreement_flag_mismatch="
            f"{int(disagreement_mismatch.sum())}"
        )
    return issues


def validate_behavior_contract(beh: pd.DataFrame) -> dict[str, Any]:
    issues: list[str] = []
    issues.extend(validate_behavior_authority_contract(beh))
    parsed_columns = [
        "stream",
        "video_code",
        "center_ms_img",
        "frame_idx",
        "k",
        "group_stream",
        "group_video_code",
        "group_center_ms",
    ]
    for column in parsed_columns:
        invalid = beh[column].isna() if column in beh.columns else pd.Series(True, index=beh.index)
        if invalid.any():
            issues.append(
                f"unparseable_{column}={int(invalid.sum())};"
                f" rows={_sample_key_values(beh.loc[invalid], ['img_name', 'group_id'])}"
            )

    numeric_order = pd.to_numeric(beh["order"], errors="coerce")
    invalid_order = numeric_order.isna() | ~numeric_order.isin(EXPECTED_ORDERS)
    if invalid_order.any():
        issues.append(
            f"invalid_order={int(invalid_order.sum())};"
            f" rows={_sample_key_values(beh.loc[invalid_order], ['img_name', 'order'])}"
        )
    beh["legacy_order"] = numeric_order.astype("Int64")

    comparisons = {
        "image_group_stream": ("stream", "group_stream"),
        "image_group_video_code": ("video_code", "group_video_code"),
        "image_group_center_ms": ("center_ms_img", "group_center_ms"),
        "image_order": ("k", "legacy_order"),
    }
    for label, (left_name, right_name) in comparisons.items():
        left = beh[left_name].astype("string").str.lower()
        right = beh[right_name].astype("string").str.lower()
        mismatch = left.ne(right).fillna(True)
        if mismatch.any():
            issues.append(
                f"{label}_mismatch={int(mismatch.sum())};"
                f" rows={_sample_key_values(beh.loc[mismatch], ['img_name', 'group_id'])}"
            )

    invalid_behavior = ~beh["behavior"].astype("string").isin(EXPECTED_BEHAVIORS)
    if invalid_behavior.any():
        issues.append(
            f"invalid_behavior={int(invalid_behavior.sum())};"
            f" values={sorted(beh.loc[invalid_behavior, 'behavior'].astype(str).unique())}"
        )
    invalid_hidden = ~beh["hidden"].astype("string").isin(EXPECTED_HIDDEN)
    if invalid_hidden.any():
        issues.append(
            f"invalid_hidden={int(invalid_hidden.sum())};"
            f" values={sorted(beh.loc[invalid_hidden, 'hidden'].astype(str).unique())}"
        )

    bbox_numeric = beh[["x1", "y1", "x2", "y2"]].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_bbox = (
        bbox_numeric.isna().any(axis=1)
        | (bbox_numeric["x2"] <= bbox_numeric["x1"])
        | (bbox_numeric["y2"] <= bbox_numeric["y1"])
        | (bbox_numeric[["x1", "y1"]] < 0).any(axis=1)
    )
    if "width" in beh.columns:
        width = pd.to_numeric(beh["width"], errors="coerce")
        invalid_bbox |= (
            width.isna()
            | (width <= 0)
            | (bbox_numeric["x1"] > width)
            | (bbox_numeric["x2"] > width)
        )
    if "height" in beh.columns:
        height = pd.to_numeric(beh["height"], errors="coerce")
        invalid_bbox |= (
            height.isna()
            | (height <= 0)
            | (bbox_numeric["y1"] > height)
            | (bbox_numeric["y2"] > height)
        )
    if invalid_bbox.any():
        issues.append(
            f"invalid_bbox={int(invalid_bbox.sum())};"
            f" rows={_sample_key_values(beh.loc[invalid_bbox], ['img_name', 'group_id', 'pig_id'])}"
        )

    source_frame = pd.to_numeric(beh["legacy_frame_index"], errors="coerce")
    center_ms = pd.to_numeric(beh["center_ms_img"], errors="coerce")
    invalid_frame_index = (
        source_frame.isna()
        | (source_frame < 0)
        | center_ms.isna()
        | (center_ms < 0)
    )
    if invalid_frame_index.any():
        issues.append(
            f"invalid_source_frame_index={int(invalid_frame_index.sum())};"
            f" rows={_sample_key_values(beh.loc[invalid_frame_index], ['img_name', 'group_id'])}"
        )

    image_identity = beh.groupby("img_name", dropna=False).agg(
        group_count=("group_id", "nunique"),
        order_count=("legacy_order", "nunique"),
        frame_count=("legacy_frame_index", "nunique"),
    )
    ambiguous_images = image_identity[
        (image_identity["group_count"] > 1)
        | (image_identity["order_count"] > 1)
        | (image_identity["frame_count"] > 1)
    ]
    if not ambiguous_images.empty:
        issues.append(
            f"ambiguous_image_identity={len(ambiguous_images)};"
            f" images={list(ambiguous_images.index[:10])}"
        )

    duplicate_actor_slot = beh.duplicated(
        subset=["group_id", "pig_id", "legacy_order"], keep=False
    )
    if duplicate_actor_slot.any():
        duplicate_keys = _sample_key_values(
            beh.loc[duplicate_actor_slot],
            ["group_id", "pig_id", "legacy_order"],
        )
        issues.append(
            f"duplicate_actor_slot={int(duplicate_actor_slot.sum())};"
            f" keys={duplicate_keys}"
        )

    incomplete_keys: list[tuple[Any, Any]] = []
    for (group_id, pig_id), group in beh.groupby(
        ["group_id", "pig_id"], dropna=False
    ):
        orders = set(int(value) for value in group["legacy_order"].dropna())
        if orders != set(EXPECTED_ORDERS) or len(group) != len(EXPECTED_ORDERS):
            incomplete_keys.append((group_id, pig_id))
    return {
        "issues": issues,
        "incomplete_actor_keys": incomplete_keys,
        "duplicate_actor_slot": bool(duplicate_actor_slot.any()),
    }


def validate_frame_mapping(beh: pd.DataFrame) -> None:
    mapping = (
        beh.dropna(subset=["group_id", "legacy_order", "legacy_frame_index"])
        .groupby(["group_id", "legacy_order"], dropna=False)["legacy_frame_index"]
        .nunique()
    )
    conflicting = mapping[mapping > 1]
    if not conflicting.empty:
        raise ValueError(
            "conflicting_group_frame_mapping="
            f"{len(conflicting)}; sample={list(conflicting.index[:10])}"
        )

    for group_id, group in beh.groupby("group_id", dropna=False):
        orders = set(int(value) for value in group["legacy_order"].dropna())
        if orders != set(EXPECTED_ORDERS):
            raise ValueError(
                f"incomplete_group_frame_mapping={group_id}; orders={sorted(orders)}"
            )
        frames = (
            group[["legacy_order", "legacy_frame_index"]]
            .drop_duplicates()
            .sort_values("legacy_order")["legacy_frame_index"]
            .tolist()
        )
        if len(frames) != len(EXPECTED_ORDERS) or len(set(frames)) != len(frames):
            raise ValueError(
                f"non_unique_group_frame_mapping={group_id}; frames={frames}"
            )


def build_frames_by_group(beh: pd.DataFrame) -> pd.Series:
    validate_frame_mapping(beh)
    cols = ["group_id", "legacy_order", "legacy_frame_index"]
    tmp = beh[cols].dropna().drop_duplicates().copy()
    tmp["legacy_order"] = tmp["legacy_order"].astype(int)
    tmp["legacy_frame_index"] = tmp["legacy_frame_index"].astype(int)
    tmp = tmp.sort_values(["group_id", "legacy_order", "legacy_frame_index"])

    def join_frames(g: pd.DataFrame) -> str:
        # Keep order 0..5 sequence, not numeric frame sorting only.
        frames = g.sort_values(
            ["legacy_order", "legacy_frame_index"]
        )["legacy_frame_index"].tolist()
        return "|".join(str(int(x)) for x in frames)

    return tmp.groupby("group_id", dropna=False).apply(join_frames)


def build_gt_support_audit(beh: pd.DataFrame) -> pd.DataFrame:
    expected_orders = set(range(6))
    rows = []

    key_cols = ["group_id", "pig_id"]
    for (group_id, pig_id), g in beh.groupby(key_cols, dropna=False):
        valid_bbox = (
            g["x1"].notna()
            & g["y1"].notna()
            & g["x2"].notna()
            & g["y2"].notna()
            & (g["x2"] > g["x1"])
            & (g["y2"] > g["y1"])
        )

        orders = sorted(int(x) for x in g["legacy_order"].dropna().unique())
        frames_by_order = (
            g.dropna(subset=["legacy_order", "legacy_frame_index"])
            .sort_values(["legacy_order", "legacy_frame_index"])
            .drop_duplicates(subset=["legacy_order"])
        )
        legacy_frames = [int(x) for x in frames_by_order["legacy_frame_index"].tolist()]

        dup_count = (
            int(g.duplicated(subset=["legacy_order"]).sum())
            if "legacy_order" in g.columns
            else 0
        )
        missing_orders = sorted(expected_orders - set(orders))

        rows.append(
            {
                "group_id": group_id,
                "pig_id": pig_id,
                "sample_id": f"{group_id}_{pig_id}",
                "expected_gt_frames": 6,
                "loaded_gt_frames": int(len(set(orders))),
                "legacy_gt_support_frames": "|".join(str(x) for x in legacy_frames),
                "missing_orders": "|".join(str(x) for x in missing_orders),
                "duplicate_order_rows": dup_count,
                "bbox_valid_rows": int(valid_bbox.sum()),
                "bbox_invalid_rows": int((~valid_bbox).sum()),
                "qa_status": (
                    "ok"
                    if len(set(orders)) == 6
                    and dup_count == 0
                    and int((~valid_bbox).sum()) == 0
                    else "review"
                ),
                "qa_notes": (
                    ""
                    if len(set(orders)) == 6
                    and dup_count == 0
                    and int((~valid_bbox).sum()) == 0
                    else "missing_or_duplicate_keyframe_bbox"
                ),
            }
        )

    return pd.DataFrame(rows)


MANIFEST_AUTHORITY_COLUMNS = [
    "manifest__img_path",
    "manifest__video",
    "manifest__day",
    "manifest__center_frame",
    "manifest__center_ts",
    "manifest__trigger_type",
    "manifest__roi_name",
    "manifest__near_roi",
    "manifest__group_id",
    "manifest__frames",
]
CANDIDATE_AUTHORITY_COLUMNS = [
    "candidate__group_id",
    "candidate__video",
    "candidate__day",
    "candidate__center_frame",
    "candidate__center_ts",
]


def _prefix_columns(frame: pd.DataFrame, prefix: str, *, keep: set[str]) -> pd.DataFrame:
    return frame.rename(
        columns={
            column: f"{prefix}{column}"
            for column in frame.columns
            if column not in keep
        }
    )


def _preserve_text_columns(
    frame: pd.DataFrame, columns: tuple[str, ...]
) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("string")
    return frame


def load_manifest_sources(
    search_roots: list[Path],
) -> tuple[pd.DataFrame, list[Path], list[dict[str, str]]]:
    files = find_files(
        search_roots,
        ["manifest_frame_attribute*.csv", "manifest*.csv"],
    )
    parts: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        frame = _preserve_text_columns(
            frame,
            ("img_path", "video", "group_id", "frames"),
        )
        if not {"img_path", "video"}.issubset(frame.columns):
            skipped.append(
                {"path": str(path), "reason": "missing_img_path_or_video"}
            )
            continue
        frame["source_file_manifest"] = str(path)
        frame["source_split_manifest"] = path.parent.name
        frame["img_name"] = frame["img_path"].map(
            lambda value: os.path.basename(str(value))
        )
        frame = _prefix_columns(
            frame,
            "manifest__",
            keep={"img_name", "source_file_manifest", "source_split_manifest"},
        )
        parts.append(frame)

    if not parts:
        return (
            pd.DataFrame(columns=["img_name", "manifest__video"]),
            files,
            skipped,
        )

    manifest = pd.concat(parts, ignore_index=True, sort=False)
    manifest = collapse_equivalent_authority_rows(
        manifest,
        key="img_name",
        authority_columns=MANIFEST_AUTHORITY_COLUMNS,
        source_column="source_file_manifest",
        label="manifest",
    )
    return manifest, files, skipped


def load_candidate_sources(
    search_roots: list[Path],
    label_groups: set[str],
) -> tuple[pd.DataFrame, list[Path]]:
    files = find_files(search_roots, ["bursts_candidates.csv", "*bursts_candidates*.csv"])
    parts: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        frame = _preserve_text_columns(frame, ("group_id", "video"))
        if not {"group_id", "video"}.issubset(frame.columns):
            continue
        frame["source_file_candidate"] = str(path)
        frame["source_split_candidate"] = path.parent.name
        rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            for label_group_id in candidate_keys(row):
                record = row.to_dict()
                record["label_group_id"] = label_group_id
                rows.append(record)
        if rows:
            expanded = pd.DataFrame(rows)
            expanded = expanded[expanded["label_group_id"].isin(label_groups)]
            if not expanded.empty:
                expanded = _prefix_columns(
                    expanded,
                    "candidate__",
                    keep={
                        "label_group_id",
                        "source_file_candidate",
                        "source_split_candidate",
                    },
                )
                parts.append(expanded)

    if not parts:
        return pd.DataFrame(columns=["label_group_id", "candidate__video"]), files

    candidates = pd.concat(parts, ignore_index=True, sort=False)
    candidates = collapse_equivalent_authority_rows(
        candidates,
        key="label_group_id",
        authority_columns=CANDIDATE_AUTHORITY_COLUMNS,
        source_column="source_file_candidate",
        label="candidate",
    )
    return candidates, files


def validate_manifest_metadata(
    combined: pd.DataFrame,
    *,
    frames_by_group: pd.Series,
) -> None:
    # ``manifest__group_id`` is a source metadata key such as
    # ``pigs281119/color/66``.  It is not the canonical burst ID encoded in
    # ``img_name``; timestamp rounding can also make their center suffixes
    # differ by one.  Preserve it for audit, but never use it as an identity
    # equality check or as a replacement for the canonical group key.

    if "manifest__frames" not in combined.columns:
        return
    present = combined["manifest__frames"].map(_nonempty)
    expected = combined["group_id"].map(frames_by_group)
    actual = combined["manifest__frames"].astype("string").str.replace(
        r"\s+", "", regex=True
    )
    mismatch = present & actual.ne(expected.astype("string"))
    if mismatch.any():
        raise ValueError(
            "manifest_frame_list_mismatch="
            f"{int(mismatch.sum())}; sample="
            f"{_sample_key_values(combined.loc[mismatch], ['img_name', 'group_id'])}"
        )


def validate_manifest_candidate_agreement(combined: pd.DataFrame) -> None:
    manifest_video = col_or_na(combined, "manifest__video")
    candidate_video = col_or_na(combined, "candidate__video")
    both = manifest_video.map(_nonempty) & candidate_video.map(_nonempty)
    mismatch = both & manifest_video.map(_canonical_video_value).ne(
        candidate_video.map(_canonical_video_value)
    )
    if mismatch.any():
        raise ValueError(
            "manifest_candidate_video_mismatch="
            f"{int(mismatch.sum())}; sample="
            f"{_sample_key_values(combined.loc[mismatch], ['group_id', 'img_name'])}"
        )

    manifest_day = col_or_na(combined, "manifest__day")
    candidate_day = col_or_na(combined, "candidate__day")
    both_day = manifest_day.map(_nonempty) & candidate_day.map(_nonempty)
    day_mismatch = both_day & manifest_day.map(
        _canonical_authority_value
    ).str.lower().ne(
        candidate_day.map(_canonical_authority_value).str.lower()
    )
    if day_mismatch.any():
        raise ValueError(
            "manifest_candidate_day_mismatch="
            f"{int(day_mismatch.sum())}; sample="
            f"{_sample_key_values(combined.loc[day_mismatch], ['group_id', 'img_name'])}"
        )


def validate_day_contract(combined: pd.DataFrame) -> dict[str, Any]:
    """Require an explicit day that agrees with the canonical video path."""
    resolved = combined["video_final"].map(_nonempty)
    missing_day = resolved & ~combined["day_final"].map(_nonempty)
    if missing_day.any():
        raise ValueError(
            "resolved_video_missing_day="
            f"{int(missing_day.sum())}; sample="
            f"{_sample_key_values(combined.loc[missing_day], ['group_id', 'video_final'])}"
        )

    def day_matches_video(row: pd.Series) -> bool:
        day = str(row["day_final"]).strip().lower()
        video_parts = {
            part.lower()
            for part in str(row["video_final"]).replace("\\", "/").split("/")
            if part
        }
        return day in video_parts

    mismatch = resolved & ~combined.loc[resolved].apply(
        day_matches_video,
        axis=1,
    ).reindex(combined.index, fill_value=False)
    if mismatch.any():
        raise ValueError(
            "day_video_path_mismatch="
            f"{int(mismatch.sum())}; sample="
            f"{_sample_key_values(combined.loc[mismatch], ['day_final', 'video_final'])}"
        )
    return {
        "resolved_rows": int(resolved.sum()),
        "missing_day_rows": int(missing_day.sum()),
        "day_video_mismatch_rows": int(mismatch.sum()),
    }


def validate_resolved_video_contract(
    combined: pd.DataFrame,
    *,
    allow_unresolved: bool,
    require_exists: bool,
) -> dict[str, Any]:
    resolved = combined["video_final"].map(_nonempty)
    unresolved_groups = sorted(
        combined.loc[~resolved, "group_id"].dropna().astype(str).unique()
    )
    if unresolved_groups and not allow_unresolved:
        raise ValueError(
            "unresolved_video_groups="
            f"{len(unresolved_groups)}; sample={unresolved_groups[:10]}"
        )

    wrong_hash = ~combined.loc[resolved].apply(
        lambda row: _video_hash_matches_group(row["video_final"], row["group_id"]),
        axis=1,
    )
    if wrong_hash.any():
        bad = combined.loc[resolved].loc[wrong_hash]
        raise ValueError(
            "video_group_hash_mismatch="
            f"{len(bad)}; sample="
            f"{_sample_key_values(bad, ['group_id', 'video_final'])}"
        )

    missing_files: list[str] = []
    if require_exists:
        missing_files = sorted(
            {
                str(Path(str(value)).expanduser())
                for value in combined.loc[resolved, "video_local_path"]
                if not Path(str(value)).expanduser().is_file()
            }
        )
        if missing_files:
            raise FileNotFoundError(
                "resolved_video_files_missing="
                f"{len(missing_files)}; sample={missing_files[:10]}"
            )
    return {
        "resolved_rows": int(resolved.sum()),
        "unresolved_rows": int((~resolved).sum()),
        "unresolved_groups": unresolved_groups,
        "missing_video_files": missing_files,
    }


# =========================
# MAIN
# =========================

def main():
    args = parse_args()

    drive_root = resolve_google_drive_root(
        args.drive_root,
        required=args.search_roots is None,
    )
    search_roots, search_root_records = resolve_search_roots(
        args.search_roots,
        drive_root,
    )
    output_paths = validate_output_paths(args, search_roots=search_roots)
    behavior_csv = Path(args.behavior_csv).expanduser().resolve()
    if not behavior_csv.exists():
        raise FileNotFoundError(f"behavior csv not found: {behavior_csv}")

    # =========================
    # 1. LOAD BEHAVIOR
    # =========================

    beh = pd.read_csv(behavior_csv, low_memory=False)
    beh = _preserve_text_columns(
        beh,
        ("img_name", "group_id", "pig_id", "behavior", "hidden"),
    )
    behavior_input_rows = len(beh)

    required = {
        "img_name",
        "group_id",
        "order",
        "pig_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "behavior",
        "hidden",
    }
    missing_required = sorted(required - set(beh.columns))
    if missing_required:
        raise ValueError(f"behavior csv thiếu cột bắt buộc: {missing_required}")

    beh = parse_img_name_fields(beh)
    beh = parse_behavior_group(beh)

    contract = validate_behavior_contract(beh)
    gt_support_audit = build_gt_support_audit(beh)
    if contract["issues"]:
        raise ValueError(
            "behavior_contract_failed: " + "; ".join(contract["issues"])
        )
    excluded_actor_keys = contract["incomplete_actor_keys"]
    excluded_set = set(excluded_actor_keys)
    gt_support_audit["excluded_by_contract"] = gt_support_audit.apply(
        lambda row: (row["group_id"], row["pig_id"]) in excluded_set,
        axis=1,
    )
    if excluded_actor_keys and not args.allow_incomplete_actor_keys:
        raise ValueError(
            "incomplete_actor_keys="
            f"{len(excluded_actor_keys)}; sample={excluded_actor_keys[:10]}"
        )
    if excluded_actor_keys:
        beh = beh.loc[
            ~beh[["group_id", "pig_id"]]
            .apply(tuple, axis=1)
            .isin(excluded_set)
        ].copy()
    if beh.empty:
        raise ValueError("no complete behavior rows remain after validation")

    # A behavior-side video column is never an authority. Keep it isolated so
    # a later merge cannot accidentally use it as the resolved source video.
    if "video" in beh.columns:
        beh = beh.rename(columns={"video": "behavior__video"})

    label_groups = set(beh["group_id"].dropna().unique())
    frames_by_group = build_frames_by_group(beh)
    if excluded_actor_keys:
        excluded_rows = gt_support_audit["excluded_by_contract"]
        gt_support_audit.loc[excluded_rows, "qa_status"] = "excluded_incomplete"
        gt_support_audit.loc[excluded_rows, "qa_notes"] = (
            "excluded_by_allow_incomplete_actor_keys"
        )

    print("Behavior rows:", len(beh))
    print("Behavior burst groups:", len(label_groups))
    print("Behavior group+pig records:", beh.groupby(["group_id", "pig_id"]).ngroups)
    print(
        "Group+pig with complete 6 keyframe bboxes:",
        int((gt_support_audit["qa_status"] == "ok").sum()),
    )
    print(
        "Group+pig needing bbox review:",
        int((gt_support_audit["qa_status"] != "ok").sum()),
    )
    if excluded_actor_keys:
        print("Excluded incomplete actor keys:", excluded_actor_keys)

    # =========================
    # 2. LOAD MANIFESTS
    # =========================

    manifest, manifest_files, skipped_manifest_files = load_manifest_sources(
        search_roots
    )
    print("\nManifest files:", len(manifest_files))
    for path in manifest_files:
        print(" -", path)
    if skipped_manifest_files:
        print("Skipped manifest files:", skipped_manifest_files)
    print("Manifest images:", manifest["img_name"].nunique())

    manifest_join = beh.merge(
        manifest,
        on="img_name",
        how="left",
        validate="many_to_one",
    )
    if len(manifest_join) != len(beh):
        raise ValueError("manifest_merge_changed_behavior_row_count")

    manifest_matched_mask = col_or_na(manifest_join, "manifest__video").map(_nonempty)
    print("Matched rows by manifest:", int(manifest_matched_mask.sum()))
    print(
        "Matched groups by manifest:",
        manifest_join.loc[manifest_matched_mask, "group_id"].nunique(),
    )

    # =========================
    # 3. LOAD BURSTS_CANDIDATES FALLBACK
    # =========================

    candidates, candidate_files = load_candidate_sources(search_roots, label_groups)
    print("\nCandidate files:", len(candidate_files))
    for path in candidate_files:
        print(" -", path)
    print("Candidate matched groups:", candidates["label_group_id"].nunique())

    # =========================
    # 4. COMBINE: MANIFEST FIRST, CANDIDATE FALLBACK
    # =========================

    combined = manifest_join.merge(
        candidates,
        left_on="group_id",
        right_on="label_group_id",
        how="left",
        validate="many_to_one",
    )
    if len(combined) != len(beh):
        raise ValueError("candidate_merge_changed_behavior_row_count")
    validate_manifest_metadata(combined, frames_by_group=frames_by_group)
    validate_manifest_candidate_agreement(combined)

    manifest_video = col_or_na(combined, "manifest__video")
    candidate_video = col_or_na(combined, "candidate__video")
    combined["video_final"] = coalesce_series(manifest_video, candidate_video)
    combined["day_final"] = coalesce_series(
        col_or_na(combined, "manifest__day"),
        col_or_na(combined, "candidate__day"),
    )
    combined["video_local_path"] = combined["video_final"].map(
        lambda value: resolve_video_local_path(
            value,
            drive_root=drive_root,
        )
    )
    combined["center_frame_final"] = coalesce_series(
        col_or_na(combined, "manifest__center_frame"),
        col_or_na(combined, "candidate__center_frame"),
    )
    combined["center_ts_final"] = coalesce_series(
        col_or_na(combined, "manifest__center_ts"),
        col_or_na(combined, "candidate__center_ts"),
    )
    combined["match_source"] = ""
    manifest_rows = manifest_video.map(_nonempty)
    candidate_rows = candidate_video.map(_nonempty)
    combined.loc[manifest_rows, "match_source"] = "manifest"
    combined.loc[~manifest_rows & candidate_rows, "match_source"] = "candidate"

    combined["legacy_frames_from_img_name"] = combined["group_id"].map(
        frames_by_group
    )
    combined["frames"] = coalesce_series(
        col_or_na(combined, "manifest__frames"),
        col_or_na(combined, "candidate__frames"),
    )
    combined["frames"] = coalesce_series(
        combined["frames"], combined["legacy_frames_from_img_name"]
    )

    # Preserve the historical output names, but source every field from the
    # explicitly prefixed manifest authority columns.
    output_aliases = {
        "img_path_manifest": "manifest__img_path",
        "trigger_type_manifest": "manifest__trigger_type",
        "roi_name_manifest": "manifest__roi_name",
        "near_roi_manifest": "manifest__near_roi",
        "candidate_group_id_manifest": "manifest__group_id",
        "frames_manifest": "manifest__frames",
    }
    for alias, source in output_aliases.items():
        combined[alias] = col_or_na(combined, source)
    if "image_path" not in combined.columns:
        combined["image_path"] = col_or_na(combined, "manifest__img_path")

    combined["sample_id"] = (
        combined["group_id"].astype(str) + "_" + combined["pig_id"].astype(str)
    )
    actor_slot_duplicate = combined.duplicated(
        subset=["group_id", "pig_id", "legacy_order"], keep=False
    )
    if actor_slot_duplicate.any():
        raise ValueError("combined_output_duplicate_actor_slot")
    if len(combined) != len(beh):
        raise ValueError("combined_row_count_changed")
    day_audit = validate_day_contract(combined)
    video_audit = validate_resolved_video_contract(
        combined,
        allow_unresolved=args.allow_unresolved_video,
        require_exists=args.require_video_exists,
    )

    # =========================
    # 5. SAVE ALL KEYFRAME BBOX OUTPUT
    # =========================

    all_keyframes = combined.copy()
    all_keyframes["trace_role"] = "historical_metadata_scaffold"
    all_keyframes["behavior_authority"] = "legacy_row_not_current_cvat_authority"

    all_out_cols = [
        "sample_id",
        "match_source",
        "source_split_manifest",
        "source_file_manifest",
        "source_split_candidate",
        "source_file_candidate",
        "day_final",
        "video_final",
        "video_local_path",
        "group_id",
        "pig_id",
        "behavior",
        "hidden",
        "img_name",
        "image_path",
        "legacy_order",
        "order",
        "k",
        "frame_index",
        "legacy_frame_index",
        "cvat_frame_index",
        "center_frame_final",
        "center_ts_final",
        "frames",
        "legacy_frames_from_img_name",
        "width",
        "height",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "bw",
        "bh",
        "cx_n",
        "cy_n",
        "bw_n",
        "bh_n",
        "category_name",
        "source",
        "in_feeder",
        "in_drinker",
        "in_toy",
        "speed_feat",
        "min_dist_other",
        "num_close_other",
        "behavior_coarse",
        "trigger_type_manifest",
        "roi_name_manifest",
        "near_roi_manifest",
        "trigger_type",
        "roi_name",
        "trace_role",
        "behavior_authority",
    ]

    all_out_cols = first_existing_cols(all_out_cols, all_keyframes)
    out_all = all_keyframes[all_out_cols].copy()

    # Sort to make group/pig/frame inspection easy.
    sort_cols = first_existing_cols(
        ["group_id", "pig_id", "legacy_order", "legacy_frame_index"],
        out_all,
    )
    out_all = out_all.sort_values(sort_cols).reset_index(drop=True)
    if len(out_all) != len(beh):
        raise ValueError("all_keyframe_output_row_count_changed")

    # =========================
    # 6. SAVE CENTER OUTPUT FOR BACKWARD COMPATIBILITY
    # =========================

    center = combined[combined["legacy_order"] == 3].copy()

    # center frame from img_name is most reliable for the bbox label on this row.
    center["center_frame_from_img"] = center["legacy_frame_index"].astype("Int64")
    center["center_frame_for_training"] = center["center_frame_from_img"]

    # These are only approximate helper fields and should not replace times.txt later.
    center["center_playback_sec"] = center["center_frame_for_training"].astype("float") / 30.0
    center["center_real_sec"] = center["center_frame_for_training"].astype("float") / 6.0

    center["frame_mismatch"] = (
        center["center_frame_final"].notna()
        & center["center_frame_from_img"].notna()
        & (
            center["center_frame_final"].astype("float")
            != center["center_frame_from_img"].astype("float")
        )
    )

    center_out_cols = [
        "sample_id",
        "match_source",
        "source_split_manifest",
        "source_file_manifest",
        "source_split_candidate",
        "source_file_candidate",
        "day_final",
        "video_final",
        "video_local_path",
        "group_id",
        "img_name",
        "center_frame_for_training",
        "center_frame_from_img",
        "center_frame_final",
        "center_ts_final",
        "center_playback_sec",
        "center_real_sec",
        "frames",
        "legacy_frames_from_img_name",
        "pig_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "behavior",
        "hidden",
        "trigger_type_manifest",
        "roi_name_manifest",
        "near_roi_manifest",
        "trigger_type",
        "roi_name",
        "frame_mismatch",
    ]

    center_out_cols = first_existing_cols(center_out_cols, center)
    out_center = center[center_out_cols].copy()
    if out_center.duplicated(["group_id", "pig_id"]).any():
        raise ValueError("center_output_duplicate_actor_key")

    # =========================
    # 7. SAVE AUDIT/MISSING
    # =========================

    missing = out_center[~out_center["video_final"].map(_nonempty)].copy()
    missing_groups = (
        missing["group_id"]
        .drop_duplicates()
        .to_frame()
        .sort_values("group_id")
    )

    if args.dry_run:
        print("\nDry run: validations passed; no output files written.")
        print("Resolved video audit:", video_audit)
        return

    write_csv_atomic(out_all, output_paths["all_bbox_csv"])
    write_csv_atomic(out_center, output_paths["center_csv"])
    write_csv_atomic(gt_support_audit, output_paths["audit_csv"])
    write_csv_atomic(missing_groups, output_paths["missing_csv"])

    def file_record(path: Path) -> dict[str, Any]:
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    lineage = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_role": "historical_metadata_scaffold",
        "behavior_authority": "legacy_row_not_current_cvat_authority",
        "behavior_csv": file_record(behavior_csv),
        "runtime_path_resolution": {
            "drive_root": "" if drive_root is None else str(drive_root),
            "search_roots": search_root_records,
            "canonical_video_column": "video_final",
            "runtime_video_column": "video_local_path",
        },
        "behavior_input_rows": behavior_input_rows,
        "behavior_validated_rows": len(beh),
        "excluded_incomplete_actor_keys": [
            [str(group_id), str(pig_id)]
            for group_id, pig_id in excluded_actor_keys
        ],
        "manifest_files": [file_record(path) for path in manifest_files],
        "skipped_manifest_files": skipped_manifest_files,
        "candidate_files": [file_record(path) for path in candidate_files],
        "provenance_rows": {
            "manifest_input": int(
                manifest["manifest_row_count"].sum()
                if "manifest_row_count" in manifest.columns
                else 0
            ),
            "manifest_unique_keys": len(manifest),
            "candidate_input": int(
                candidates["candidate_row_count"].sum()
                if "candidate_row_count" in candidates.columns
                else 0
            ),
            "candidate_unique_keys": len(candidates),
        },
        "manifest_images_after_collapse": int(manifest["img_name"].nunique()),
        "candidate_groups_after_collapse": int(
            candidates["label_group_id"].nunique()
        ),
        "output_files": {
            name: file_record(path)
            for name, path in output_paths.items()
            if name != "lineage_json"
        },
        "flags": {
            "allow_incomplete_actor_keys": args.allow_incomplete_actor_keys,
            "allow_unresolved_video": args.allow_unresolved_video,
            "require_video_exists": args.require_video_exists,
            "center_order": 3,
        },
        "resolved_video_audit": video_audit,
        "day_audit": day_audit,
        "row_counts": {
            "behavior_input": behavior_input_rows,
            "combined": len(combined),
            "all_keyframes": len(out_all),
            "center_keyframes": len(out_center),
        },
        "authority_rules": {
            "manifest": "video and provenance metadata",
            "candidate": "fallback only when manifest metadata is absent",
            "behavior": "labels and actor/frame keys only; never source video",
            "current_cvat": "not represented by this historical scaffold",
        },
    }
    write_json_atomic(lineage, output_paths["lineage_json"])

    # =========================
    # 8. REPORT
    # =========================

    print("\n===== FINAL REPORT =====")
    print("All keyframe bbox output:", args.out_all_bbox_csv)
    print("All keyframe bbox rows:", len(out_all))
    print(
        "All keyframe groups:",
        out_all["group_id"].nunique() if "group_id" in out_all.columns else "N/A",
    )
    print(
        "All keyframe group+pig:",
        out_all.groupby(["group_id", "pig_id"]).ngroups
        if {"group_id", "pig_id"}.issubset(out_all.columns)
        else "N/A",
    )

    print("\nCenter output:", args.out_center_csv)
    print("Center keyframe labels:", len(out_center))
    print(
        "Matched center rows:",
        out_center["video_final"].notna().sum()
        if "video_final" in out_center.columns
        else "N/A",
    )
    print(
        "Unmatched center rows:",
        out_center["video_final"].isna().sum()
        if "video_final" in out_center.columns
        else "N/A",
    )
    print(
        "Matched center groups:",
        out_center.loc[out_center["video_final"].notna(), "group_id"].nunique()
        if "video_final" in out_center.columns
        else "N/A",
    )
    print(
        "Unmatched center groups:",
        out_center.loc[out_center["video_final"].isna(), "group_id"].nunique()
        if "video_final" in out_center.columns
        else "N/A",
    )

    print("\nGT support audit:", args.out_audit_csv)
    print(gt_support_audit["qa_status"].value_counts(dropna=False))

    if "match_source" in out_center.columns:
        print("\nMatch source counts:")
        print(out_center["match_source"].value_counts(dropna=False))

    print(
        "Frame mismatch rows:",
        int(out_center["frame_mismatch"].sum())
        if "frame_mismatch" in out_center.columns
        else "N/A",
    )
    print("Missing groups saved:", args.out_missing_csv)

    all_groups = set(beh["group_id"].dropna().unique())
    center_groups = (
        set(out_center["group_id"].dropna().unique())
        if "group_id" in out_center.columns
        else set()
    )
    missing_center_group = sorted(all_groups - center_groups)

    print("Groups without center output:", len(missing_center_group))
    print(missing_center_group[:20])

    if "frame_mismatch" in out_center.columns and out_center["frame_mismatch"].any():
        tmp = out_center[out_center["frame_mismatch"]].copy()
        tmp["delta"] = tmp["center_frame_from_img"] - tmp["center_frame_final"]

        print("\nFrame mismatch delta describe:")
        print(tmp["delta"].describe())
        print(tmp["delta"].value_counts().sort_index().head(30))
        cols = first_existing_cols(
            [
                "group_id",
                "img_name",
                "match_source",
                "center_frame_from_img",
                "center_frame_final",
                "delta",
                "video_final",
            ],
            tmp,
        )
        print(tmp[cols].head(20))


if __name__ == "__main__":
    main()
