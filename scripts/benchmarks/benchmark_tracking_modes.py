#!/usr/bin/env python3
"""Run several tracking modes side by side without overwriting outputs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACKING_SCRIPT = PROJECT_ROOT / "scripts" / "track_videos.py"
DEFAULT_MODES = ("realtime", "bytetrack_raw", "hybrid_bytetrack")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", type=Path, default=[], help="Video path. May be repeated.")
    parser.add_argument("--videos", type=str, default=None, help="Comma-separated video paths.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config with a list or object containing `videos`.",
    )
    parser.add_argument("--weights", type=Path, required=True, help="YOLO weights path.")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(DEFAULT_MODES),
        default=list(DEFAULT_MODES),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/bench/modes"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--det-conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--tracker-type", choices=["bytetrack", "botsort"], default="bytetrack")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional argument forwarded to track_videos.py. May be repeated.",
    )
    return parser.parse_args(argv)


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def resolve_videos(args: argparse.Namespace) -> list[Path]:
    videos: list[Path] = list(args.video)
    if args.videos:
        videos.extend(Path(part.strip()) for part in args.videos.split(",") if part.strip())
    if args.config:
        payload = json.loads(args.config.read_text(encoding="utf-8"))
        entries = payload.get("videos", payload) if isinstance(payload, dict) else payload
        videos.extend(Path(str(item)) for item in entries)
    unique_videos: list[Path] = []
    seen: set[Path] = set()
    for video in videos:
        resolved = video.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_videos.append(resolved)
    return unique_videos


def build_command(
    args: argparse.Namespace,
    video: Path,
    mode: str,
    mode_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(TRACKING_SCRIPT),
        "--video",
        str(video),
        "--mode",
        mode,
        "--weights",
        str(args.weights),
        "--output-dir",
        str(mode_root),
    ]
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.det_conf is not None:
        command.extend(["--det-conf", str(args.det_conf)])
    if args.imgsz is not None:
        command.extend(["--imgsz", str(args.imgsz)])
    if args.tracker_type:
        command.extend(["--tracker-type", args.tracker_type])
    command.extend(args.extra_arg)
    return command


def ensure_output_dir(mode_root: Path, overwrite: bool) -> None:
    mode_root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return
    existing = [path for path in mode_root.iterdir() if path.name not in {"command.txt", "run_metadata.json"}]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory without --overwrite: {mode_root}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    videos = resolve_videos(args)
    failures: list[dict[str, Any]] = []

    for video in videos:
        for mode in args.modes:
            mode_root = args.output_root / mode / video.stem
            ensure_output_dir(mode_root, args.overwrite)
            command = build_command(args, video, mode, mode_root)
            (mode_root / "command.txt").write_text(shell_join(command) + "\n", encoding="utf-8")
            print(f"[RUN] {video.name} mode={mode}")
            start = time.perf_counter()
            result = subprocess.run(command, cwd=PROJECT_ROOT)
            runtime_sec = time.perf_counter() - start
            metadata = {
                "video": str(video),
                "mode": mode,
                "weights": str(args.weights),
                "det_conf": args.det_conf,
                "imgsz": args.imgsz,
                "tracker_type": args.tracker_type,
                "runtime_sec": runtime_sec,
                "returncode": result.returncode,
            }
            (mode_root / "run_metadata.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            if result.returncode != 0:
                failures.append(metadata)

    if failures:
        print(json.dumps(failures, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
