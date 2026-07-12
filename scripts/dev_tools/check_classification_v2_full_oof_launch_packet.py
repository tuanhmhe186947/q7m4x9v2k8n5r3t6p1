from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev_tools.write_classification_v2_full_oof_launch_packet import (  # noqa: E402
    SCHEMA_VERSION,
    build_launch_packet,
)


def main() -> None:
    """Check that the launch packet is reviewable and fail-closed."""

    parser = argparse.ArgumentParser(
        description="Check classification_v2 full OOF launch packet."
    )
    parser.add_argument(
        "--launch-packet-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/full_oof_launch_packet.json"
        ),
    )
    parser.add_argument(
        "--preflight-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_multimodal_oof_preflight.json"
        ),
    )
    parser.add_argument(
        "--run-plan-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_multimodal_oof_run_plan.json"
        ),
    )
    parser.add_argument(
        "--runtime-benchmark-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_benchmarks_visual_v3/"
            "summary_head/runtime_benchmark_audit.json"
        ),
    )
    parser.add_argument(
        "--authorization-template-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_authorization_template.json"
        ),
    )
    parser.add_argument(
        "--preflight-freshness-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_preflight_freshness_audit.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_launch_packet_audit.json"
        ),
    )
    args = parser.parse_args()

    errors: list[str] = []
    packet = _load_json(args.launch_packet_json, errors)
    expected = build_launch_packet(
        preflight=_load_json(args.preflight_json, errors),
        run_plan=_load_json(args.run_plan_json, errors),
        runtime_benchmark=_load_json(args.runtime_benchmark_json, errors),
        authorization_template=_load_json(args.authorization_template_json, errors),
        preflight_freshness=_load_json(args.preflight_freshness_json, errors),
    )
    errors.extend(_packet_errors(packet, expected))
    audit = {
        "schema_version": "classification_v2_full_oof_launch_packet_audit_v1",
        "valid": not errors,
        "errors": errors,
        "launch_packet_json": str(args.launch_packet_json),
        "packet_status": packet.get("status"),
        "packet_schema_version": packet.get("schema_version"),
        "packet_matches_current_inputs": packet == expected,
        "full_oof_execution_allowed": packet.get("full_oof_execution_allowed"),
        "authorization_required": packet.get("authorization_required"),
        "authorization_template_authorized": packet.get(
            "authorization_template_authorized"
        ),
        "launch_command": packet.get("launch_command"),
        "review_checklist_count": len(packet.get("review_checklist") or []),
        "preflight_config_sha256": packet.get("preflight_config_sha256"),
        "git_commit": packet.get("git_commit"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if errors:
        raise SystemExit(1)


def _packet_errors(packet: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Assert launch packet content matches current upstream audit evidence."""

    errors: list[str] = []
    if packet != expected:
        errors.append("launch_packet_does_not_match_current_inputs")
    if packet.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"launch_packet_schema_mismatch={packet.get('schema_version')}")
    if packet.get("valid") is not True:
        errors.append(f"launch_packet_not_valid={packet.get('errors')}")
    if packet.get("status") != "READY_FOR_HUMAN_AUTHORIZATION":
        errors.append(f"launch_packet_status_invalid={packet.get('status')}")
    if packet.get("full_oof_execution_allowed") is not False:
        errors.append("launch_packet_must_not_allow_execution")
    if packet.get("authorization_required") is not True:
        errors.append("launch_packet_must_require_authorization")
    if packet.get("authorization_template_authorized") is not False:
        errors.append("launch_packet_template_must_be_unauthorized")
    command = packet.get("launch_command") or []
    for token in ("--full", "--confirm-full-run", "--authorization-json"):
        if token not in command:
            errors.append(f"launch_packet_command_missing={token}")
    if len(packet.get("review_checklist") or []) < 6:
        errors.append("launch_packet_review_checklist_too_short")
    return errors


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing_json={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
