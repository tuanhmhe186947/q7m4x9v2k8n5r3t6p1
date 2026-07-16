from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from pig_behavior.classification_v2.datasets.image_context_index import (
    MANDATORY_CVAT_FRAME_INDICES,
    MANDATORY_CVAT_MEDIA_BASENAME,
    MANDATORY_CVAT_PIG_ID,
    MANDATORY_CVAT_VIDEO_KEY,
    audit_mandatory_cvat_video_case,
)

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


DEFAULT_INPUT = Path("outputs/classification_v2/review_policy/reviewed_frame_features.csv")
DEFAULT_AUDIT = Path("outputs/classification_v2/train_ready_windows/image_loader_smoke_audit.json")
DEFAULT_VIDEO_ROOT = Path("data/videos")
DEFAULT_LEGACY_CROP_ROOT = Path("data/raw/legacy_full_multigt_masked_nodup_16f/crops")

IMAGE_COLS = [
    "source_type",
    "dataset_id",
    "video_key",
    "source_video_key",
    "source_video_path",
    "frame_uid",
    "frame_index",
    "pig_id",
    "track_id",
    "object_track_key",
    "crop_path",
    "image_path",
    "frame_path",
    "x1",
    "y1",
    "x2",
    "y2",
    "image_width",
    "image_height",
    "behavior_after_review",
    "behavior",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _build_video_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not root.exists():
        return index

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v"}

    def add_alias(alias: object, path: Path) -> None:
        key = str(alias).replace("\\", "/").strip().lower()
        if not key:
            return
        index.setdefault(key, path)
        index.setdefault(Path(key).stem.lower(), path)

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in video_exts:
            continue
        stem = path.stem.lower()
        add_alias(path.name, path)
        add_alias(stem, path)
        for suffix in ["_30fps", "-30fps", " 30fps"]:
            if stem.endswith(suffix):
                base = stem[: -len(suffix)]
                add_alias(base, path)
                add_alias(base + path.suffix.lower(), path)
                add_alias(base + ".mp4", path)
    return index


def _resolve_video_path(
    row: pd.Series,
    video_root: Path,
    video_index: dict[str, Path],
) -> Path | None:
    raw_source_path = row.get("source_video_path")
    if pd.notna(raw_source_path):
        raw = str(raw_source_path).strip()
        if raw:
            path = Path(raw)
            candidates = [path]
            if not path.is_absolute():
                candidates.append(video_root / path)
            candidates.append(video_root / path.name)
            for candidate in candidates:
                if candidate.exists():
                    return candidate

    keys: list[str] = []
    for col in ["video_key", "source_video_key"]:
        value = row.get(col)
        if pd.notna(value):
            raw = str(value).strip().replace("\\", "/")
            stem = Path(raw).stem
            stems = [stem]
            lower_stem = stem.lower()
            for prefix in ["test video ", "tracking_annotation_", "tracking annotation "]:
                if lower_stem.startswith(prefix):
                    stems.append(stem[len(prefix) :])
            for candidate_stem in stems:
                keys.extend(
                    [
                        raw,
                        candidate_stem,
                        f"{candidate_stem}.mp4",
                        f"{candidate_stem}_30fps",
                        f"{candidate_stem}_30fps.mp4",
                        f"{raw}.mp4",
                        f"{raw}_30fps",
                        f"{raw}_30fps.mp4",
                    ]
                )
                if candidate_stem.lower().endswith("_30fps"):
                    base = candidate_stem[: -len("_30fps")]
                    keys.extend([base, f"{base}.mp4"])

    for key in keys:
        resolved = video_index.get(key.strip().lower())
        if resolved is not None:
            return resolved
    return None


def _legacy_relative_path(path_text: str) -> Path:
    normalized = path_text.replace("/", "\\")
    markers = [
        "\\legacy_full_multigt_masked_nodup_16f\\crops\\",
        "\\legacy_full_multigt_masked_nodup_16f\\",
        "\\crops\\",
    ]
    for marker in markers:
        if marker in normalized:
            return Path(normalized.split(marker, 1)[1])
    return Path(Path(normalized).name)


def _resolve_legacy_crop(row: pd.Series, crop_root: Path) -> Path | None:
    for col in ["crop_path", "image_path", "frame_path"]:
        value = row.get(col)
        if pd.isna(value):
            continue
        raw = str(value).strip()
        if not raw:
            continue
        path = Path(raw)
        candidates = [path]
        candidates.append(crop_root / _legacy_relative_path(raw))
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def _row_identity(row: pd.Series) -> dict[str, Any]:
    keys = [
        "source_type",
        "dataset_id",
        "video_key",
        "pig_id",
        "track_id",
        "object_track_key",
        "frame_index",
        "frame_uid",
        "behavior_after_review",
        "behavior",
    ]
    return {k: _json_default(row.get(k)) for k in keys if k in row.index}


def _bbox_from_row(row: pd.Series) -> tuple[float, float, float, float] | None:
    vals = []
    for col in ["x1", "y1", "x2", "y2"]:
        value = pd.to_numeric(row.get(col), errors="coerce")
        if pd.isna(value):
            return None
        vals.append(float(value))
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _check_legacy_row(row: pd.Series, crop_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"kind": "legacy_crop", "identity": _row_identity(row), "ok": False}
    path = _resolve_legacy_crop(row, crop_root)
    result["resolved_path"] = str(path) if path is not None else None
    if path is None:
        result["error"] = "missing_legacy_crop"
        return result
    try:
        with Image.open(path) as img:
            result["width"] = int(img.width)
            result["height"] = int(img.height)
            result["mode"] = img.mode
            result["ok"] = img.width > 0 and img.height > 0
            if not result["ok"]:
                result["error"] = "empty_image"
    except Exception as exc:
        result["error"] = f"image_open_failed: {exc}"
    return result


def _check_cvat_row(
    row: pd.Series,
    video_root: Path,
    video_index: dict[str, Path],
    capture_cache: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "cvat_video_bbox",
        "identity": _row_identity(row),
        "ok": False,
    }
    if cv2 is None:
        result["error"] = "cv2_unavailable"
        return result

    video_path = _resolve_video_path(row, video_root, video_index)
    result["resolved_video_path"] = str(video_path) if video_path is not None else None
    if video_path is None:
        result["error"] = "missing_video"
        return result

    frame_idx = pd.to_numeric(row.get("frame_index"), errors="coerce")
    if pd.isna(frame_idx):
        result["error"] = "missing_frame_index"
        return result
    frame_idx = int(frame_idx)

    bbox = _bbox_from_row(row)
    if bbox is None:
        result["error"] = "invalid_bbox"
        return result

    cap_key = str(video_path)
    cap = capture_cache.get(cap_key)
    if cap is None:
        cap = cv2.VideoCapture(str(video_path))
        capture_cache[cap_key] = cap
    if not cap.isOpened():
        result["error"] = "video_open_failed"
        return result

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result["video_frame_count"] = frame_count
    result["video_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    result["video_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result["video_fps"] = float(cap.get(cv2.CAP_PROP_FPS))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        result["error"] = "frame_read_failed"
        return result

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    clipped = (
        max(0, min(w, int(x1))),
        max(0, min(h, int(y1))),
        max(0, min(w, int(x2))),
        max(0, min(h, int(y2))),
    )
    cx1, cy1, cx2, cy2 = clipped
    result["frame_shape"] = [int(h), int(w), int(frame.shape[2]) if len(frame.shape) > 2 else 1]
    result["bbox"] = [x1, y1, x2, y2]
    result["bbox_clipped"] = [cx1, cy1, cx2, cy2]
    if cx2 <= cx1 or cy2 <= cy1:
        result["error"] = "empty_clipped_crop"
        return result
    crop = frame[cy1:cy2, cx1:cx2]
    result["crop_shape"] = [int(crop.shape[0]), int(crop.shape[1]), int(crop.shape[2])]
    result["ok"] = crop.size > 0
    if not result["ok"]:
        result["error"] = "empty_crop"
    return result


def _sample_rows(df: pd.DataFrame, source_type: str, sample_size: int) -> pd.DataFrame:
    source = df[df["source_type"].astype(str).eq(source_type)].copy()
    if source.empty or sample_size <= 0:
        return source.head(0)
    candidates = [
        "video_key",
        "pig_id",
        "track_id",
        "object_track_key",
        "frame_index",
    ]
    sort_cols = [column for column in candidates if column in source.columns]
    source = source.sort_values(sort_cols).reset_index(drop=True)
    if len(source) <= sample_size:
        return source
    source_span = len(source) - 1
    sample_span = max(1, sample_size - 1)
    positions = sorted({round(i * source_span / sample_span) for i in range(sample_size)})
    return source.iloc[positions].copy()


def _mandatory_cvat_case(df: pd.DataFrame) -> pd.DataFrame:
    frame_index = pd.to_numeric(df.get("frame_index"), errors="coerce")
    mask = (
        df["source_type"].astype(str).eq("cvat_tracking_xml")
        & df["video_key"].astype(str).eq(MANDATORY_CVAT_VIDEO_KEY)
        & df["pig_id"].astype(str).eq(MANDATORY_CVAT_PIG_ID)
        & frame_index.isin(MANDATORY_CVAT_FRAME_INDICES)
    )
    return df.loc[mask].copy()


def _audit_mandatory_gui_case(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert pixel-loader results into the shared exact-basename contract."""

    rows: list[dict[str, Any]] = []
    for result in results:
        if result.get("kind") != "cvat_video_bbox":
            continue
        identity = result.get("identity")
        if not isinstance(identity, dict):
            continue
        rows.append(
            {
                "video_key": identity.get("video_key"),
                "pig_id": identity.get("pig_id"),
                "frame_index": identity.get("frame_index"),
                "resolved_media_path": result.get("resolved_video_path"),
                "image_context_loadable": bool(result.get("ok")),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "video_key",
            "pig_id",
            "frame_index",
            "resolved_media_path",
            "image_context_loadable",
        ],
    )
    return audit_mandatory_cvat_video_case(frame)


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    available_cols = pd.read_csv(args.input_csv, nrows=0).columns.tolist()
    usecols = [c for c in IMAGE_COLS if c in available_cols]
    df = pd.read_csv(args.input_csv, usecols=usecols, low_memory=False)

    video_index = _build_video_index(args.video_root)
    capture_cache: dict[str, Any] = {}
    results: list[dict[str, Any]] = []

    try:
        legacy_rows = _sample_rows(df, "legacy_recovered", args.sample_per_source)
        cvat_rows = pd.concat(
            [
                _mandatory_cvat_case(df),
                _sample_rows(df, "cvat_tracking_xml", args.sample_per_source),
            ],
            ignore_index=True,
        )
        if not cvat_rows.empty:
            identity_columns = [
                "source_type",
                "video_key",
                "pig_id",
                "track_id",
                "frame_index",
            ]
            cvat_rows = cvat_rows.drop_duplicates(
                subset=[column for column in identity_columns if column in cvat_rows],
                keep="first",
            )

        for _, row in legacy_rows.iterrows():
            results.append(_check_legacy_row(row, args.legacy_crop_root))
        for _, row in cvat_rows.iterrows():
            results.append(_check_cvat_row(row, args.video_root, video_index, capture_cache))
    finally:
        for cap in capture_cache.values():
            try:
                cap.release()
            except Exception:
                pass

    errors = [r for r in results if not r.get("ok")]
    mandatory_case = _audit_mandatory_gui_case(results)
    result_kinds = [result.get("kind") for result in results]
    checked_by_kind = {}
    if result_kinds:
        checked_by_kind = pd.Series(result_kinds).value_counts(dropna=False).to_dict()

    audit = {
        "schema_version": "classification_v2.source_image_loader_audit.v2",
        "input_csv": str(args.input_csv),
        "video_root": str(args.video_root),
        "legacy_crop_root": str(args.legacy_crop_root),
        "output_audit": str(args.output_audit),
        "sample_per_source_requested": int(args.sample_per_source),
        "video_index_size": len(video_index),
        "source_rows": df["source_type"].value_counts(dropna=False).to_dict(),
        "checked_rows": len(results),
        "checked_by_kind": checked_by_kind,
        "ok_rows": sum(1 for r in results if r.get("ok")),
        "error_rows": len(errors),
        "errors": errors[:50],
        "mandatory_gui_video_case_checked_rows": mandatory_case["rows"],
        "mandatory_gui_video_case_expected_basename": (MANDATORY_CVAT_MEDIA_BASENAME),
        "mandatory_gui_video_case_resolved_basenames": mandatory_case["resolved_media_basenames"],
        "mandatory_gui_video_case_ok": mandatory_case["ok"],
        "mandatory_gui_video_case": mandatory_case,
        "results": results,
    }
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Smoke test classification_v2 image loading for train-ready windows.")
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--legacy-crop-root", type=Path, default=DEFAULT_LEGACY_CROP_ROOT)
    parser.add_argument("--output-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--sample-per-source", type=int, default=24)
    args = parser.parse_args()

    audit = run_check(args)
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.write_text(
        json.dumps(audit, indent=2, default=_json_default),
        encoding="utf-8",
    )

    summary = {key: audit[key] for key in audit if key != "results"}
    print(json.dumps(summary, indent=2, default=_json_default))
    if audit["error_rows"] or not audit["mandatory_gui_video_case_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
