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
        "python_executable": packet.get("python_executable"),
        "launch_command": packet.get("launch_command"),
        "cmd_launch_command": packet.get("cmd_launch_command"),
        "launch_command_python_ready": _launch_command_python_ready(packet),
        "cmd_launch_command_ready": _cmd_launch_command_ready(packet),
        "cmd_launch_command_prefix_ready": _cmd_launch_command_prefix_ready(
            packet
        ),
        "cmd_launch_command_wraps_base_command": (
            _cmd_launch_command_wraps_base_command(packet)
        ),
        "cmd_launch_command_bat_present": bool(
            packet.get("cmd_launch_command_bat")
        ),
        "estimated_training_seconds_excluding_eval": packet.get(
            "estimated_training_seconds_excluding_eval"
        ),
        "estimated_training_minutes_excluding_eval": packet.get(
            "estimated_training_minutes_excluding_eval"
        ),
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
    cmd_command = packet.get("cmd_launch_command") or []
    python_executable = str(packet.get("python_executable") or "")
    if not python_executable:
        errors.append("launch_packet_missing_python_executable")
    if command and command[0] != python_executable:
        errors.append("launch_packet_python_executable_mismatch")
    if not cmd_command:
        errors.append("launch_packet_missing_cmd_launch_command")
    if "PYTHONPATH=%CD%\\src" not in cmd_command:
        errors.append("launch_packet_cmd_missing_pythonpath")
    if "&&" not in cmd_command:
        errors.append("launch_packet_cmd_missing_command_chaining")
    if not _cmd_launch_command_prefix_ready(packet):
        errors.append("launch_packet_cmd_prefix_invalid")
    if not _cmd_launch_command_wraps_base_command(packet):
        errors.append("launch_packet_cmd_does_not_wrap_base_command")
    if not packet.get("cmd_launch_command_bat"):
        errors.append("launch_packet_missing_cmd_launch_command_bat")
    if packet.get("estimated_training_seconds_excluding_eval") is None:
        errors.append("launch_packet_missing_training_runtime_estimate")
    if packet.get("estimated_training_minutes_excluding_eval") is None:
        errors.append("launch_packet_missing_training_runtime_minutes")
    required_tokens = (
        "--full",
        "--confirm-full-run",
        "--authorization-json",
        "--packed-image-cache",
        "--packed-image-cache-index",
        "--visual-context-cache-manifest",
        "--visual-context-packed-cache",
        "--visual-context-packed-cache-index",
        "--require-packed-visual-context",
    )
    for token in required_tokens:
        if token not in command:
            errors.append(f"launch_packet_command_missing={token}")
    missing_values = _missing_command_values(
        command,
        [
            "--packed-image-cache",
            "--packed-image-cache-index",
            "--visual-context-cache-manifest",
            "--visual-context-packed-cache",
            "--visual-context-packed-cache-index",
        ],
    )
    if missing_values:
        errors.append(f"launch_packet_command_missing_values={missing_values}")
    if len(packet.get("review_checklist") or []) < 6:
        errors.append("launch_packet_review_checklist_too_short")
    return errors


def _launch_command_python_ready(packet: dict[str, Any]) -> bool:
    """Return whether the raw Python command is bound to the audited executable."""

    command = packet.get("launch_command") or []
    python_executable = str(packet.get("python_executable") or "")
    return bool(python_executable and command and command[0] == python_executable)


def _cmd_launch_command_ready(packet: dict[str, Any]) -> bool:
    """Return whether the human CMD command keeps project-root PYTHONPATH setup."""

    cmd_command = packet.get("cmd_launch_command") or []
    return bool(
        cmd_command
        and "PYTHONPATH=%CD%\\src" in cmd_command
        and "&&" in cmd_command
    )


def _cmd_launch_command_prefix_ready(packet: dict[str, Any]) -> bool:
    """Require the human CMD command to enter project root before Python."""

    cmd_command = packet.get("cmd_launch_command") or []
    python_executable = str(packet.get("python_executable") or "")
    expected = [
        "cd",
        "/d",
        str(Path.cwd()),
        "&&",
        "set",
        "PYTHONPATH=%CD%\\src",
        "&&",
        python_executable,
    ]
    return cmd_command[: len(expected)] == expected


def _cmd_launch_command_wraps_base_command(packet: dict[str, Any]) -> bool:
    """Check that the CMD launch helper preserves the audited Python command."""

    cmd_command = packet.get("cmd_launch_command") or []
    launch_command = packet.get("launch_command") or []
    if len(cmd_command) < 8 or not launch_command:
        return False
    return cmd_command[7:] == launch_command


def _missing_command_values(command: list[str], options: list[str]) -> list[str]:
    """Return command options that are absent or followed by an empty value."""

    missing: list[str] = []
    for option in options:
        if option not in command:
            missing.append(option)
            continue
        index = command.index(option)
        if index + 1 >= len(command) or command[index + 1] == "":
            missing.append(option)
    return missing


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing_json={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
