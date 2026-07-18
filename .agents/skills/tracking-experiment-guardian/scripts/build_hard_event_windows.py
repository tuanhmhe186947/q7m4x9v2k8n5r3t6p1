#!/usr/bin/env python3
"""Freeze difficult tracking windows from parent remapped identity events."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Return the SHA256 of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonnegative_int(raw: str) -> int:
    """Parse a non-negative integer CLI value."""

    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def positive_int(raw: str) -> int:
    """Parse a positive integer CLI value."""

    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-events-csv", type=Path, required=True)
    parser.add_argument("--assets-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--cluster-gap-frames", type=nonnegative_int, default=30)
    parser.add_argument("--score-padding-before", type=nonnegative_int, default=30)
    parser.add_argument("--score-padding-after", type=nonnegative_int, default=30)
    parser.add_argument("--warmup-frames", type=nonnegative_int, default=120)
    parser.add_argument("--min-switch-rows", type=positive_int, default=1)
    return parser.parse_args()


def read_frame_counts(path: Path | None) -> dict[str, int]:
    """Read optional per-video frame counts from tracking assets."""

    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video_stem", "video_frame_count"}
        if not required.issubset(reader.fieldnames or []):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"assets CSV missing columns: {missing}")
        counts: dict[str, int] = {}
        for row in reader:
            raw_count = str(row["video_frame_count"]).strip()
            if raw_count:
                counts[row["video_stem"]] = int(float(raw_count))
        return counts


def read_switch_frames(path: Path) -> dict[str, list[int]]:
    """Read remapped switch-event rows grouped by video."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video_stem", "frame", "event"}
        if not required.issubset(reader.fieldnames or []):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"identity-events CSV missing columns: {missing}")
        frames: dict[str, list[int]] = defaultdict(list)
        for row in reader:
            if "switch" not in str(row["event"]).lower():
                continue
            if "remapped" in row and str(row["remapped"]).lower() not in {
                "1",
                "true",
                "yes",
            }:
                raise ValueError("switch input contains a non-remapped event row")
            frames[row["video_stem"]].append(int(row["frame"]))
    if not frames:
        raise ValueError("no remapped identity-switch rows were found")
    return dict(frames)


def group_episode_frames(frames: list[int], max_gap: int) -> list[list[int]]:
    """Group unique switch frames into temporally local episodes."""

    unique_frames = sorted(set(frames))
    groups = [[unique_frames[0]]]
    for frame in unique_frames[1:]:
        if frame - groups[-1][-1] <= max_gap:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return groups


def build_rows(
    switch_frames: dict[str, list[int]],
    frame_counts: dict[str, int],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Build deterministic window rows from parent event episodes."""

    source_hash = file_sha256(args.identity_events_csv)
    assets_hash = file_sha256(args.assets_csv) if args.assets_csv else ""
    rows: list[dict[str, object]] = []
    for video_stem, all_frames in sorted(switch_frames.items()):
        episodes = group_episode_frames(all_frames, args.cluster_gap_frames)
        episode_rows: list[dict[str, object]] = []
        for episode_frames in episodes:
            first_frame = episode_frames[0]
            last_frame = episode_frames[-1]
            switch_rows = sum(
                first_frame <= frame <= last_frame for frame in all_frames
            )
            if switch_rows < args.min_switch_rows:
                continue
            score_start = max(0, first_frame - args.score_padding_before)
            score_end = last_frame + args.score_padding_after
            frame_count = frame_counts.get(video_stem)
            if frame_count is not None:
                score_end = min(score_end, frame_count - 1)
            tracking_start = max(0, score_start - args.warmup_frames)
            episode_rows.append(
                {
                    "video_stem": video_stem,
                    "episode_id": (
                        f"{video_stem}__switch_{first_frame}_{last_frame}"
                    ),
                    "first_switch_frame": first_frame,
                    "last_switch_frame": last_frame,
                    "switch_event_rows": switch_rows,
                    "unique_switch_frames": len(episode_frames),
                    "score_start_frame": score_start,
                    "score_end_frame": score_end,
                    "tracking_start_frame": tracking_start,
                    "tracking_max_frames": score_end - tracking_start + 1,
                    "actual_warmup_frames": score_start - tracking_start,
                    "video_frame_count": frame_count or "",
                    "source_identity_events_sha256": source_hash,
                    "source_assets_sha256": assets_hash,
                }
            )
        episode_rows.sort(
            key=lambda row: (
                -int(row["switch_event_rows"]),
                int(row["first_switch_frame"]),
            )
        )
        for rank, row in enumerate(episode_rows, start=1):
            row["video_priority_rank"] = rank
            rows.append(row)

    rows.sort(
        key=lambda row: (
            -int(row["switch_event_rows"]),
            str(row["video_stem"]),
            int(row["first_switch_frame"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["global_priority_rank"] = rank
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a new immutable window manifest."""

    if path.suffix.lower() != ".csv":
        raise ValueError("--output-csv must end with .csv")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    if not rows:
        raise ValueError("no episodes satisfy --min-switch-rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Build and write the frozen event-window manifest."""

    args = parse_args()
    switch_frames = read_switch_frames(args.identity_events_csv)
    frame_counts = read_frame_counts(args.assets_csv)
    rows = build_rows(switch_frames, frame_counts, args)
    write_rows(args.output_csv, rows)
    print(f"windows={len(rows)}")
    print(f"videos={len({row['video_stem'] for row in rows})}")
    print(f"output={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
