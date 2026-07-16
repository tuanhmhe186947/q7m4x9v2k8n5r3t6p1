#!/usr/bin/env python3
"""Audit one completed primary/repeat tracking pair and write a PASS lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.evaluation.tracking.repeatability import (  # noqa: E402
    TrackingRepeatabilityAuditConfig,
    audit_tracking_repeatability,
    write_tracking_repeatability_audit,
)


def _idsw_guard(value: str) -> tuple[str, int]:
    try:
        stem, maximum_text = value.rsplit("=", maxsplit=1)
        maximum = int(maximum_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "IDSW guard must use VIDEO_STEM=MAXIMUM."
        ) from exc
    if not stem or maximum < 0:
        raise argparse.ArgumentTypeError(
            "IDSW guard requires a stem and a non-negative maximum."
        )
    return stem, maximum


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-eval-dir", type=Path, required=True)
    parser.add_argument("--repeat-eval-dir", type=Path, required=True)
    parser.add_argument("--expected-video-count", type=int, default=13)
    parser.add_argument("--expected-commit")
    parser.add_argument(
        "--guard-remapped-idsw",
        action="append",
        default=[],
        type=_idsw_guard,
        metavar="VIDEO_STEM=MAXIMUM",
    )
    parser.add_argument("--expected-delay-frames", type=int)
    parser.add_argument("--expected-timing-contract")
    parser.add_argument(
        "--skip-input-rehash",
        action="store_true",
        help="Skip current input rehashing; intended only for focused tests.",
    )
    parser.add_argument(
        "--allow-dirty-auditor",
        action="store_true",
        help="Allow a dirty checker worktree; cannot support authority evidence.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Fresh JSON path for an immutable PASS audit record.",
    )
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> TrackingRepeatabilityAuditConfig:
    guards = dict(args.guard_remapped_idsw)
    if len(guards) != len(args.guard_remapped_idsw):
        raise ValueError("Duplicate IDSW guard video stems are not allowed.")
    return TrackingRepeatabilityAuditConfig(
        primary_eval_dir=args.primary_eval_dir,
        repeat_eval_dir=args.repeat_eval_dir,
        expected_video_count=args.expected_video_count,
        expected_commit=args.expected_commit,
        guard_video_max_remapped_idsw=guards,
        expected_delay_frames=args.expected_delay_frames,
        expected_timing_contract=args.expected_timing_contract,
        verify_input_hashes=not args.skip_input_rehash,
        require_clean_auditor=not args.allow_dirty_auditor,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = _config_from_args(args)
    if args.output is not None:
        output_path = write_tracking_repeatability_audit(config, args.output)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        print(f"[PASS] Tracking repeatability audit: {output_path}")
    else:
        payload = audit_tracking_repeatability(config)
        print("[PASS] Tracking repeatability audit")
    aggregate = payload["aggregate_metrics"]
    runtime = payload["runtime"]
    print(
        "  metrics: "
        f"IDSW={aggregate['remapped_idsw']} "
        f"HOTA={aggregate['remapped_hota_pct']}% "
        f"IDF1={aggregate['remapped_idf1_pct']}%"
    )
    print(
        "  evidence: "
        f"predictions={payload['verified_prediction_count']} "
        f"artifacts={payload['verified_artifact_count']} "
        f"mp4={payload['mp4_count']}"
    )
    print(
        "  runtime loop FPS primary/repeat: "
        f"{runtime['primary']['tracking_loop_effective_fps']:.3f}/"
        f"{runtime['repeat']['tracking_loop_effective_fps']:.3f}"
    )
    print(f"  authority_sha256: {payload['authority_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
