from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from pig_behavior.classification_v2.datasets.image_sequence_dataset import letterbox_rgb_uint8

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


DEFAULT_WINDOWS = Path(
    "outputs/classification_v2/sequence_features_reviewed/"
    "sequence_window_manifest.csv"
)
DEFAULT_FRAMES = Path("outputs/classification_v2/review_policy/reviewed_frame_features.csv")
DEFAULT_AUDIT = Path(
    "outputs/classification_v2/train_ready_windows/"
    "image_sequence_loader_smoke_audit.json"
)
DEFAULT_VIDEO_ROOT = Path("data/videos")
DEFAULT_LEGACY_CROP_ROOT = Path(
    os.environ.get(
        "CLASSIFICATION_V2_LEGACY_CROP_ROOT",
        "outputs/legacy_16f_rebuild/"
        "legacy_16f_rebuild_20260718_v2/06_full_recovery/crops",
    )
)

FRAME_COLS = [
    "source_type",
    "dataset_id",
    "video_key",
    "source_video_key",
    "source_video_path",
    "object_track_key",
    "pig_id",
    "track_id",
    "frame_index",
    "crop_path",
    "image_path",
    "frame_path",
    "x1",
    "y1",
    "x2",
    "y2",
]


def _build_video_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not root.exists():
        return index
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v"}

    def add(alias: object, path: Path) -> None:
        key = str(alias).replace("\\", "/").strip().lower()
        if key:
            index.setdefault(key, path)
            index.setdefault(Path(key).stem.lower(), path)

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in video_exts:
            continue
        stem = path.stem.lower()
        add(path.name, path)
        add(stem, path)
        for suffix in ["_30fps", "-30fps", " 30fps"]:
            if stem.endswith(suffix):
                base = stem[: -len(suffix)]
                add(base, path)
                add(base + ".mp4", path)
                add(base + path.suffix.lower(), path)
    return index


def _candidate_video_keys(row: pd.Series) -> list[str]:
    keys: list[str] = []
    for col in ["video_key", "source_video_key"]:
        value = row.get(col)
        if pd.isna(value):
            continue
        raw = str(value).replace("\\", "/").strip()
        stem = Path(raw).stem
        stems = [stem]
        lower = stem.lower()
        for prefix in ["test video ", "tracking_annotation_", "tracking annotation "]:
            if lower.startswith(prefix):
                stems.append(stem[len(prefix) :])
        for s in stems:
            keys.extend([raw, s, f"{s}.mp4", f"{s}_30fps", f"{s}_30fps.mp4"])
            if s.lower().endswith("_30fps"):
                base = s[: -len("_30fps")]
                keys.extend([base, f"{base}.mp4"])
    return keys


def _resolve_video(row: pd.Series, video_root: Path, index: dict[str, Path]) -> Path | None:
    source_path = row.get("source_video_path")
    if pd.notna(source_path):
        path = Path(str(source_path).strip())
        for candidate in [path, video_root / path, video_root / path.name]:
            if candidate.exists():
                return candidate
    for key in _candidate_video_keys(row):
        hit = index.get(key.lower())
        if hit is not None:
            return hit
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


def _resolve_crop(row: pd.Series, crop_root: Path) -> Path | None:
    for col in ["crop_path", "image_path", "frame_path"]:
        value = row.get(col)
        if pd.isna(value):
            continue
        raw = str(value).strip()
        if not raw:
            continue
        path = Path(raw)
        for candidate in [path, crop_root / _legacy_relative_path(raw)]:
            if candidate.exists():
                return candidate
    return None


def _read_legacy_image(row: pd.Series, crop_root: Path, size: int) -> np.ndarray | None:
    path = _resolve_crop(row, crop_root)
    if path is None:
        return None
    try:
        img = Image.open(path).convert("RGB")
        return letterbox_rgb_uint8(np.asarray(img, dtype=np.uint8), size)
    except Exception:
        return None


def _read_cvat_crop(
    row: pd.Series, video_root: Path, video_index: dict[str, Path], cache: dict[str, Any], size: int
) -> np.ndarray | None:
    if cv2 is None:
        return None
    path = _resolve_video(row, video_root, video_index)
    if path is None:
        return None
    cap = cache.get(str(path))
    if cap is None:
        cap = cv2.VideoCapture(str(path))
        cache[str(path)] = cap
    if not cap.isOpened():
        return None
    frame_index = pd.to_numeric(row.get("frame_index"), errors="coerce")
    bbox = [pd.to_numeric(row.get(c), errors="coerce") for c in ["x1", "y1", "x2", "y2"]]
    if pd.isna(frame_index) or any(pd.isna(v) for v in bbox):
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1i, y1i = max(0, min(w, int(x1))), max(0, min(h, int(y1)))
    x2i, y2i = max(0, min(w, int(x2))), max(0, min(h, int(y2)))
    if x2i <= x1i or y2i <= y1i:
        return None
    crop = frame[y1i:y2i, x1i:x2i]
    if crop.size == 0:
        return None
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return letterbox_rgb_uint8(crop.astype(np.uint8), size)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _sample_windows(
    windows: pd.DataFrame,
    sample_per_source: int,
    *,
    include_invalid: bool,
) -> pd.DataFrame:
    if not include_invalid and "window_valid_for_main_train" in windows.columns:
        windows = windows[_as_bool(windows["window_valid_for_main_train"])].copy()
    samples = []
    for _source_type, source in windows.groupby("source_type", sort=True):
        source = source.sort_values(
            [
                "window_length_frames",
                "video_key",
                "object_track_key",
                "window_start_frame",
            ]
        )
        if len(source) <= sample_per_source:
            samples.append(source)
            continue
        positions = sorted(
            {
                round(
                    i * (len(source) - 1) / max(1, sample_per_source - 1)
                )
                for i in range(sample_per_source)
            }
        )
        samples.append(source.iloc[positions])
    return pd.concat(samples, ignore_index=True) if samples else windows.head(0)


def _check_window(
    window: pd.Series,
    frame_lookup: dict[str, pd.DataFrame],
    *,
    video_root: Path,
    video_index: dict[str, Path],
    crop_root: Path,
    cache: dict[str, Any],
    image_size: int,
) -> dict[str, Any]:
    source_type = str(window.get("source_type"))
    start = int(pd.to_numeric(window.get("window_start_frame"), errors="coerce"))
    end = int(pd.to_numeric(window.get("window_end_frame"), errors="coerce"))
    wanted = list(range(start, end + 1))
    object_key = str(window.get("object_track_key"))
    frames = frame_lookup.get(object_key)
    result = {
        "window_id": str(window.get("window_id")),
        "source_type": source_type,
        "video_key": str(window.get("video_key")),
        "object_track_key": object_key,
        "window_length_frames": int(window.get("window_length_frames")),
        "wanted_frames": len(wanted),
        "loaded_frames": 0,
        "missing_frames": 0,
        "tensor_shape": None,
        "ok": False,
    }
    if frames is None:
        result["missing_frames"] = len(wanted)
        result["error"] = "missing_object_track_frames"
        return result

    rows_by_frame = {int(r["frame_index"]): r for _, r in frames.iterrows()}
    seq = np.zeros((len(wanted), image_size, image_size, 3), dtype=np.uint8)
    for pos, frame_index in enumerate(wanted):
        row = rows_by_frame.get(frame_index)
        if row is None:
            result["missing_frames"] += 1
            continue
        if source_type == "legacy_recovered":
            image = _read_legacy_image(row, crop_root, image_size)
        else:
            image = _read_cvat_crop(row, video_root, video_index, cache, image_size)
        if image is None:
            result["missing_frames"] += 1
            continue
        seq[pos] = image
        result["loaded_frames"] += 1

    result["tensor_shape"] = list(seq.shape)
    result["ok"] = result["loaded_frames"] == len(wanted)
    if not result["ok"]:
        result["error"] = "incomplete_sequence_load"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test sequence image loading for classification_v2 windows."
    )
    parser.add_argument("--window-manifest-csv", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--frame-features-csv", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--legacy-crop-root", type=Path, default=DEFAULT_LEGACY_CROP_ROOT)
    parser.add_argument("--output-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--sample-per-source", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--include-invalid", action="store_true")
    args = parser.parse_args()

    windows = pd.read_csv(args.window_manifest_csv, low_memory=False)
    sample = _sample_windows(windows, args.sample_per_source, include_invalid=args.include_invalid)
    object_keys = set(sample["object_track_key"].astype(str))

    header = pd.read_csv(args.frame_features_csv, nrows=0).columns.tolist()
    usecols = [c for c in FRAME_COLS if c in header]
    frames = pd.read_csv(args.frame_features_csv, usecols=usecols, low_memory=False)
    frames = frames[frames["object_track_key"].astype(str).isin(object_keys)].copy()
    frames["frame_index"] = pd.to_numeric(frames["frame_index"], errors="coerce")
    frames = frames.dropna(subset=["frame_index"])
    frames["frame_index"] = frames["frame_index"].astype(int)
    frame_lookup = {
        str(k): g.sort_values("frame_index")
        for k, g in frames.groupby("object_track_key", sort=False)
    }

    video_index = _build_video_index(args.video_root)
    cache: dict[str, Any] = {}
    try:
        results = [
            _check_window(
                row,
                frame_lookup,
                video_root=args.video_root,
                video_index=video_index,
                crop_root=args.legacy_crop_root,
                cache=cache,
                image_size=args.image_size,
            )
            for _, row in sample.iterrows()
        ]
    finally:
        for cap in cache.values():
            try:
                cap.release()
            except Exception:
                pass

    errors = [r for r in results if not r["ok"]]
    audit = {
        "window_manifest_csv": str(args.window_manifest_csv),
        "frame_features_csv": str(args.frame_features_csv),
        "sample_per_source": args.sample_per_source,
        "include_invalid": bool(args.include_invalid),
        "image_size": args.image_size,
        "checked_windows": len(results),
        "ok_windows": len(results) - len(errors),
        "error_windows": len(errors),
        "video_index_size": len(video_index),
        "loaded_frames": int(sum(r["loaded_frames"] for r in results)),
        "missing_frames": int(sum(r["missing_frames"] for r in results)),
        "checked_by_source": (
            pd.Series([r["source_type"] for r in results])
            .value_counts(dropna=False)
            .to_dict()
        ),
        "errors": errors[:50],
        "results": results,
    }
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in audit if k != "results"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
