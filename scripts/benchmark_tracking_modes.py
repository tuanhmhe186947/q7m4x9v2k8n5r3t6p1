#!/usr/bin/env python3
"""Run tracking modes side by side without overwriting mode outputs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_SCRIPT = PROJECT_ROOT / "src" / "pig_behavior" / "data_preparation" / "tracking_annotation.py"
DEFAULT_MODES = ("realtime", "bytetrack_raw", "hybrid_bytetrack")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        action="append",
        type=Path,
        default=[],
        help="Video path. May be repeated.",
    )
    parser.add_argument(
        "--videos",
        type=str,
        default=None,
        help="Comma-separated video paths.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional JSON config containing either a list of video paths or "
            "an object with a 'videos' list."
        ),
    )
    parser.add_argument("--weights", type=Path, required=True, help="YOLO weights path.")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["realtime", "bytetrack_raw", "hybrid_bytetrack"],
        default=list(DEFAULT_MODES),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/tracking_benchmark"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--det-conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--tracker-type", choices=["bytetrack", "botsort"], default="bytetrack")
    parser.add_argument(
        "--cvat-video-xml-dir",
        type=Path,
        default=None,
        help="Optional directory for explicit CVAT XML files, grouped by video/mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty benchmark mode directory.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional argument forwarded to tracking_annotation.py. May be repeated.",
    )
    return parser.parse_args(argv)


def _config_videos(path: Path) -> list[Path]:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [Path(item) for item in data]
    if isinstance(data, dict) and isinstance(data.get("videos"), list):
        return [Path(item) for item in data["videos"]]
    raise ValueError("--config must be a JSON list or an object with a 'videos' list.")


def resolve_videos(args: argparse.Namespace) -> list[Path]:
    videos = list(args.video)
    if args.videos:
        videos.extend(Path(part.strip()) for part in args.videos.split(",") if part.strip())
    if args.config:
        videos.extend(_config_videos(args.config))
    unique: list[Path] = []
    seen: set[Path] = set()
    for video in videos:
        resolved = video.expanduser()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    if not unique:
        raise ValueError("Provide at least one --video, --videos, or --config entry.")
    return unique


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def build_command(
    args: argparse.Namespace,
    video: Path,
    mode: str,
    mode_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(TRACKING_SCRIPT),
        "--video",
        str(video),
        "--weights",
        str(args.weights),
        "--mode",
        mode,
        "--output-dir",
        str(args.output_root),
        "--tracker-type",
        args.tracker_type,
    ]
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.det_conf is not None:
        command.extend(["--det-conf", str(args.det_conf)])
    if args.imgsz is not None:
        command.extend(["--imgsz", str(args.imgsz)])
    if args.cvat_video_xml_dir is not None:
        xml_path = args.cvat_video_xml_dir / video.stem / mode / f"{video.stem}_{mode}.xml"
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        command.extend(["--cvat-video-xml", str(xml_path)])
    command.extend(args.extra_arg)
    return command


def ensure_output_dir(mode_dir: Path, overwrite: bool) -> None:
    mode_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return
    existing = [path for path in mode_dir.iterdir() if path.name not in {"command.txt", "run_metadata.json"}]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory without --overwrite: {mode_dir}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    videos = resolve_videos(args)
    failures: list[dict[str, Any]] = []

    for video in videos:
        for mode in args.modes:
            mode_dir = args.output_root / video.stem / mode
            ensure_output_dir(mode_dir, args.overwrite)
            command = build_command(args, video, mode, mode_dir)
            command_text = shell_join(command)
            (mode_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")

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
                "max_frames": args.max_frames,
                "runtime_sec": round(runtime_sec, 3),
                "returncode": result.returncode,
                "command": command,
            }
            (mode_dir / "run_metadata.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
                newline="\n",
            )
            if result.returncode != 0:
                failures.append(metadata)

    if failures:
        print(f"[FAIL] {len(failures)} tracking runs failed.", file=sys.stderr)
        return 1
    print("[OK] tracking benchmark complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
