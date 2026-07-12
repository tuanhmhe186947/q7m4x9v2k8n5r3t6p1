from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev_tools import (  # noqa: E402
    write_classification_v2_full_oof_postrun_registration_packet as packet_writer,
)

SCHEMA_VERSION = packet_writer.SCHEMA_VERSION
DEFAULT_FULL_OUTPUT_DIR = packet_writer.DEFAULT_FULL_OUTPUT_DIR
DEFAULT_REGISTRY_DIR = packet_writer.DEFAULT_REGISTRY_DIR
build_postrun_registration_packet = packet_writer.build_postrun_registration_packet


def main() -> None:
    """Validate the post-run registration packet remains deterministic."""

    parser = argparse.ArgumentParser(
        description="Check classification_v2 full OOF post-run registration packet."
    )
    parser.add_argument(
        "--packet-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_postrun_registration_packet.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FULL_OUTPUT_DIR)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument(
        "--runtime-benchmark-audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_benchmarks_visual_v3/"
            "summary_head/runtime_benchmark_audit.json"
        ),
    )
    parser.add_argument(
        "--completion-gate-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_completion_gate_audit.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_postrun_registration_packet_audit.json"
        ),
    )
    args = parser.parse_args()

    errors: list[str] = []
    packet = _load_json(args.packet_json, errors)
    expected = build_postrun_registration_packet(
        output_dir=args.output_dir,
        registry_dir=args.registry_dir,
        runtime_benchmark_audit_json=args.runtime_benchmark_audit_json,
        completion_gate_json=args.completion_gate_json,
    )
    errors.extend(_packet_errors(packet, expected))
    audit = {
        "schema_version": (
            "classification_v2_full_oof_postrun_registration_packet_audit_v1"
        ),
        "valid": not errors,
        "errors": errors,
        "packet_json": str(args.packet_json),
        "packet_status": packet.get("status"),
        "packet_schema_version": packet.get("schema_version"),
        "packet_matches_current_inputs": packet == expected,
        "runs_training": packet.get("runs_training"),
        "runs_registration": packet.get("runs_registration"),
        "q2_claim_allowed_by_packet": packet.get("q2_claim_allowed_by_packet"),
        "python_executable": packet.get("python_executable"),
        "register_command": packet.get("register_command"),
        "register_cmd_command_ready": _cmd_ready(
            packet.get("register_cmd_command") or [],
            packet.get("python_executable"),
        ),
        "completion_gate_command": packet.get("completion_gate_command"),
        "completion_gate_cmd_command_ready": _cmd_ready(
            packet.get("completion_gate_cmd_command") or [],
            packet.get("python_executable"),
        ),
        "required_artifact_count": len(packet.get("required_artifacts") or {}),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if errors:
        raise SystemExit(1)


def _packet_errors(packet: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Check that packet contents are complete, inert, and reproducible."""

    errors: list[str] = []
    if packet != expected:
        errors.append("postrun_packet_does_not_match_current_inputs")
    if packet.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"postrun_packet_schema_mismatch={packet.get('schema_version')}")
    if packet.get("valid") is not True:
        errors.append(f"postrun_packet_invalid={packet.get('errors')}")
    if packet.get("status") != "READY_FOR_POST_FULL_OOF_REGISTRATION":
        errors.append(f"postrun_packet_status_invalid={packet.get('status')}")
    if packet.get("runs_training") is not False:
        errors.append("postrun_packet_must_not_run_training")
    if packet.get("runs_registration") is not False:
        errors.append("postrun_packet_must_not_run_registration")
    if packet.get("q2_claim_allowed_by_packet") is not False:
        errors.append("postrun_packet_must_not_allow_q2_claim")
    if len(packet.get("required_artifacts") or {}) < 8:
        errors.append("postrun_packet_required_artifacts_incomplete")
    command = packet.get("register_command") or []
    python_executable = str(packet.get("python_executable") or "")
    if not python_executable:
        errors.append("postrun_packet_missing_python_executable")
    if command and command[0] != python_executable:
        errors.append("postrun_register_command_python_mismatch")
    for token in ("--paper-facing", "--artifact", "--run-audit-json"):
        if token not in command:
            errors.append(f"postrun_register_command_missing={token}")
    completion = packet.get("completion_gate_command") or []
    if completion and completion[0] != python_executable:
        errors.append("postrun_completion_command_python_mismatch")
    if "--registry-record-json" not in completion:
        errors.append("postrun_completion_command_missing_registry_record")
    if not _cmd_ready(
        packet.get("register_cmd_command") or [],
        python_executable,
    ):
        errors.append("postrun_register_cmd_missing_cmd_setup")
    if not _cmd_ready(
        packet.get("completion_gate_cmd_command") or [],
        python_executable,
    ):
        errors.append("postrun_completion_cmd_missing_cmd_setup")
    return errors


def _cmd_ready(command: list[str], python_executable: Any) -> bool:
    """Check that CMD commands enter the project before running Python."""

    if len(command) < 8:
        return False
    if command[:6] != [
        "cd",
        "/d",
        str(Path.cwd()),
        "&&",
        "set",
        "PYTHONPATH=%CD%\\src",
    ]:
        return False
    if command[6] != "&&":
        return False
    return command[7] == str(python_executable or "")


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing_packet_json={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
