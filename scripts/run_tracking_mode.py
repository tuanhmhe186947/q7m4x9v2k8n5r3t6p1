#!/usr/bin/env python3
"""Presentation-friendly runner for tracking mode comparisons."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.tracking.profiles import (  # noqa: E402
    PRESENTATION_PROFILES,
    get_presentation_profile,
)

DEFAULT_RULE_COMBO = "iou0_area0_condarea0_merge0"
DEFAULT_COMPARE_MODES = "bytetrack_raw,realtime,hybrid_bytetrack"
SUMMARY_COLUMNS = [
    "presentation_mode",
    "tracking_mode",
    "eval_config",
    "rule_output",
    "video_stem",
    "remapped_idsw",
    "remapped_hota_pct",
    "remapped_idf1_pct",
    "fp",
    "fn",
    "evaluated_frames",
]


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run tracking/evaluation using a named presentation mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python scripts/run_tracking_mode.py --mode hybrid_bytetrack --task eval -a
  python scripts/run_tracking_mode.py --mode realtime --task eval -v Pigs291119_000263_30fps
  python scripts/run_tracking_mode.py --mode bytetrack_raw --task eval -a
  python scripts/run_tracking_mode.py --task compare -v Pigs291119_000263_30fps

Default eval rule combo: {DEFAULT_RULE_COMBO}
Use --all-rule-combos to run the full rule benchmark matrix instead.
""",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(PRESENTATION_PROFILES),
        default="hybrid_bytetrack",
        help=(
            "Presentation mode to run. 'realtime' is the best current realtime "
            "default; use realtime_fast, realtime_balanced, or "
            "realtime_quality_delayed to choose a realtime variant explicitly."
        ),
    )
    parser.add_argument(
        "--compare-modes",
        nargs="?",
        const=DEFAULT_COMPARE_MODES,
        default=None,
        metavar="MODES",
        help=(
            "Mode list for --task compare. Omit MODES to compare "
            f"{DEFAULT_COMPARE_MODES}, or pass a comma-separated "
            "list such as realtime_fast,realtime_balanced,realtime_quality_delayed."
        ),
    )
    parser.add_argument(
        "--task",
        choices=["track", "eval", "compare"],
        default="eval",
        help=(
            "track=generate prediction/XML only; eval=run evaluation and let "
            "evaluate_tracking.py track missing predictions unless --eval-existing is set; "
            "compare=run eval for multiple modes and write CSV/Markdown summaries."
        ),
    )
    parser.add_argument(
        "--eval-existing",
        action="store_true",
        help="With --task eval, evaluate existing predictions only and do not run missing tracking.",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-v", "--video", type=str, help="Comma-separated names, paths, keys, or aliases.")
    group.add_argument("-a", "--all-videos", action="store_true", help="Run on all configured videos.")
    parser.add_argument(
        "--path-profile",
        type=str,
        default=None,
        help="Path profile forwarded to track_videos.py/evaluate_tracking.py as --profile.",
    )
    parser.add_argument("--path-config", type=str, default=None)
    parser.add_argument(
        "--rule-combo",
        action="append",
        default=None,
        help="Evaluation rule combo. Defaults to iou0_area0_condarea0_merge0.",
    )
    parser.add_argument(
        "--all-rule-combos",
        action="store_true",
        help="Do not force the default rule combo; let evaluation run all rule combos.",
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="List presentation modes and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    parser.add_argument(
        "--compare-output-root",
        type=str,
        default=None,
        help="Output root for --task compare summaries and per-mode eval outputs.",
    )
    parser.add_argument(
        "--compare-prediction-root",
        type=str,
        default=None,
        help="Prediction root for --task compare per-mode predictions.",
    )
    args, extra_args = parser.parse_known_args(argv)
    if args.compare_modes is not None and args.task != "compare":
        parser.error("--compare-modes is only valid with --task compare")
    if not args.list_modes and not args.video and not args.all_videos:
        parser.error("one of the arguments -v/--video -a/--all-videos is required")
    return args, extra_args


def _append_video_selection(cmd: list[str], args: argparse.Namespace) -> None:
    if args.all_videos:
        cmd.append("-a")
    else:
        cmd.extend(["-v", args.video])


def _append_path_selection(cmd: list[str], args: argparse.Namespace) -> None:
    if args.path_profile:
        cmd.extend(["--profile", args.path_profile])
    if args.path_config:
        cmd.extend(["--path-config", args.path_config])


def _rule_combos(args: argparse.Namespace) -> list[str]:
    if args.all_rule_combos:
        return []
    raw_combos = args.rule_combo or [DEFAULT_RULE_COMBO]
    combos: list[str] = []
    for raw_combo in raw_combos:
        combos.extend(part.strip() for part in raw_combo.split(",") if part.strip())
    return combos


def _tracking_command(args: argparse.Namespace, mode: str, eval_config: str) -> list[str]:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "track_videos.py")]
    _append_video_selection(cmd, args)
    _append_path_selection(cmd, args)
    cmd.extend(["--eval-config", eval_config, "--mode", mode])
    return cmd


def _evaluation_command(
    args: argparse.Namespace,
    mode: str,
    eval_config: str,
    *,
    output_root: Path | None = None,
    prediction_root: Path | None = None,
) -> list[str]:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_tracking.py")]
    _append_video_selection(cmd, args)
    _append_path_selection(cmd, args)
    cmd.extend(["--mode", mode, "--eval-config", eval_config])
    for combo in _rule_combos(args):
        cmd.extend(["--rule-combo", combo])
    if output_root is not None:
        cmd.extend(["--output-root", str(output_root)])
    if prediction_root is not None:
        cmd.extend(["--prediction-root", str(prediction_root)])
    if args.eval_existing:
        cmd.append("--no-run-missing-tracker")
    return cmd


def _selected_modes(args: argparse.Namespace) -> list[str]:
    if args.task != "compare":
        return [args.mode]
    raw_modes = args.compare_modes or DEFAULT_COMPARE_MODES
    modes = [part.strip() for part in raw_modes.split(",") if part.strip()]
    unknown = sorted(set(modes) - set(PRESENTATION_PROFILES))
    if unknown:
        choices = ", ".join(sorted(PRESENTATION_PROFILES))
        raise ValueError(f"Unknown mode(s): {', '.join(unknown)}. Choices: {choices}")
    return modes


def _compare_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.compare_output_root)
        if args.compare_output_root
        else PROJECT_ROOT / "outputs" / "eval" / "mode_compare" / run_name
    )
    prediction_root = (
        Path(args.compare_prediction_root)
        if args.compare_prediction_root
        else PROJECT_ROOT / "outputs" / "pred" / "mode_compare" / run_name
    )
    return output_root, prediction_root


def _read_metrics_rows(metrics_csv: Path) -> list[dict[str, str]]:
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_compare_summary(
    compare_output_root: Path,
    mode_metadata: dict[str, dict[str, str]],
) -> tuple[Path, Path]:
    summary_rows: list[dict[str, str]] = []
    for metrics_csv in sorted(compare_output_root.rglob("tracking_metrics.csv")):
        try:
            relative = metrics_csv.relative_to(compare_output_root)
        except ValueError:
            relative = metrics_csv
        presentation_mode = relative.parts[0] if relative.parts else ""
        metadata = mode_metadata.get(presentation_mode, {})
        rule_output = str(relative.parent)
        for row in _read_metrics_rows(metrics_csv):
            summary_rows.append(
                {
                    "presentation_mode": presentation_mode,
                    "tracking_mode": metadata.get("tracking_mode", ""),
                    "eval_config": metadata.get("eval_config", ""),
                    "rule_output": rule_output,
                    "video_stem": row.get("video_stem", ""),
                    "remapped_idsw": row.get("remapped_idsw", ""),
                    "remapped_hota_pct": row.get("remapped_hota_pct", ""),
                    "remapped_idf1_pct": row.get("remapped_idf1_pct", ""),
                    "fp": row.get("fp", ""),
                    "fn": row.get("fn", ""),
                    "evaluated_frames": row.get("evaluated_frames", ""),
                }
            )

    compare_output_root.mkdir(parents=True, exist_ok=True)
    csv_path = compare_output_root / "mode_comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    md_path = compare_output_root / "mode_comparison_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Mode Comparison Summary\n\n")
        handle.write(f"- Rows: `{len(summary_rows)}`\n")
        handle.write(f"- Output root: `{compare_output_root}`\n\n")
        handle.write("| " + " | ".join(SUMMARY_COLUMNS) + " |\n")
        handle.write("| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |\n")
        for row in summary_rows:
            handle.write(
                "| "
                + " | ".join(str(row.get(column, "")) for column in SUMMARY_COLUMNS)
                + " |\n"
            )
    return csv_path, md_path


def _print_modes() -> None:
    print("Available presentation modes:")
    for name, metadata in PRESENTATION_PROFILES.items():
        print(f" - {name}: tracking_mode={metadata['mode']}, eval_config={metadata['eval_config']}")
        print(f"   {metadata['description']}")


def main(argv: list[str] | None = None) -> int:
    args, extra_args = parse_args(argv)
    if args.list_modes:
        _print_modes()
        return 0

    commands: list[tuple[str, list[str]]] = []
    mode_metadata: dict[str, dict[str, str]] = {}
    try:
        selected_modes = _selected_modes(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    compare_output_root: Path | None = None
    compare_prediction_root: Path | None = None
    if args.task == "compare":
        compare_output_root, compare_prediction_root = _compare_roots(args)
    for mode_name in selected_modes:
        profile = get_presentation_profile(mode_name)
        tracking_mode = str(profile["mode"])
        eval_config = str(profile["eval_config"])
        mode_metadata[mode_name] = {
            "tracking_mode": tracking_mode,
            "eval_config": eval_config,
        }
        if args.task == "track":
            commands.append((mode_name, _tracking_command(args, tracking_mode, eval_config)))
        elif args.task == "compare":
            assert compare_output_root is not None
            assert compare_prediction_root is not None
            commands.append(
                (
                    mode_name,
                    _evaluation_command(
                        args,
                        tracking_mode,
                        eval_config,
                        output_root=compare_output_root / mode_name,
                        prediction_root=compare_prediction_root / mode_name,
                    ),
                )
            )
        else:
            commands.append((mode_name, _evaluation_command(args, tracking_mode, eval_config)))

    for _, cmd in commands:
        cmd.extend(extra_args)
        print(f"Command: {' '.join(cmd)}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            return result.returncode
    if args.task == "compare" and not args.dry_run:
        assert compare_output_root is not None
        csv_path, md_path = _write_compare_summary(compare_output_root, mode_metadata)
        print(f"[compare-summary-csv] {csv_path}")
        print(f"[compare-summary-md] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
