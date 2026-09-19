from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import write_classification_v2_full_oof_postrun_registration_packet as packet_writer

from pig_behavior.classification_v2.experiments.record_contract import (  # noqa: E402
    check_parent_record_link,
)

SCHEMA_VERSION = packet_writer.SCHEMA_VERSION
DEFAULT_FULL_OUTPUT_DIR = packet_writer.DEFAULT_FULL_OUTPUT_DIR
DEFAULT_REGISTRY_DIR = packet_writer.DEFAULT_REGISTRY_DIR
build_postrun_registration_packet = packet_writer.build_postrun_registration_packet


def main() -> None:
    """Validate the post-run registration packet remains deterministic."""

    parser = argparse.ArgumentParser(description="Check classification_v2 full OOF post-run registration packet.")
    parser.add_argument(
        "--packet-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/full_oof_postrun_registration_packet.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FULL_OUTPUT_DIR)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument(
        "--runtime-benchmark-audit-json",
        type=Path,
        default=Path("outputs/classification_v2/model_benchmarks_visual_v3/summary_head/runtime_benchmark_audit.json"),
    )
    parser.add_argument(
        "--completion-gate-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/full_oof_completion_gate_audit.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/full_oof_postrun_registration_packet_audit.json"),
    )
    parser.add_argument(
        "--packet-md",
        type=Path,
        default=Path("outputs/classification_v2/model_design/full_oof_postrun_registration_packet.md"),
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
    parent_controls = _parent_control_audits(packet)
    for parent in parent_controls:
        errors.extend(f"postrun_parent_record_invalid={parent['record_json']}:{error}" for error in parent["errors"])
    md_audit = _markdown_runbook_audit(
        path=args.packet_md,
        expected_headings=(
            "## Calibration Command",
            "## Confusion Comparison Command",
            "## Ablation Report Command",
            "## Register Command",
            "## Completion Gate Command",
        ),
        min_bat_blocks=5,
    )
    errors.extend(md_audit["errors"])
    audit = {
        "schema_version": ("classification_v2_full_oof_postrun_registration_packet_audit_v1"),
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
        "calibration_cmd_command_ready": _cmd_ready(
            packet.get("calibration_cmd_command") or [],
            packet.get("python_executable"),
        ),
        "calibration_cmd_wraps_base_command": _cmd_wraps_base_command(
            packet.get("calibration_cmd_command") or [],
            packet.get("calibration_command") or [],
        ),
        "calibration_command_bat_present": bool(packet.get("calibration_command_bat")),
        "confusion_comparison_cmd_command_ready": _cmd_ready(
            packet.get("confusion_comparison_cmd_command") or [],
            packet.get("python_executable"),
        ),
        "confusion_cmd_wraps_base_command": _cmd_wraps_base_command(
            packet.get("confusion_comparison_cmd_command") or [],
            packet.get("confusion_comparison_command") or [],
        ),
        "confusion_comparison_command_bat_present": bool(packet.get("confusion_comparison_command_bat")),
        "ablation_report_cmd_command_ready": _cmd_ready(
            packet.get("ablation_report_cmd_command") or [],
            packet.get("python_executable"),
        ),
        "ablation_cmd_wraps_base_command": _cmd_wraps_base_command(
            packet.get("ablation_report_cmd_command") or [],
            packet.get("ablation_report_command") or [],
        ),
        "ablation_report_command_bat_present": bool(packet.get("ablation_report_command_bat")),
        "register_command": packet.get("register_command"),
        "register_cmd_command_ready": _cmd_ready(
            packet.get("register_cmd_command") or [],
            packet.get("python_executable"),
        ),
        "register_cmd_wraps_base_command": _cmd_wraps_base_command(
            packet.get("register_cmd_command") or [],
            packet.get("register_command") or [],
        ),
        "register_command_bat_present": bool(packet.get("register_command_bat")),
        "completion_gate_command": packet.get("completion_gate_command"),
        "completion_gate_cmd_command_ready": _cmd_ready(
            packet.get("completion_gate_cmd_command") or [],
            packet.get("python_executable"),
        ),
        "completion_cmd_wraps_base_command": _cmd_wraps_base_command(
            packet.get("completion_gate_cmd_command") or [],
            packet.get("completion_gate_command") or [],
        ),
        "completion_gate_command_bat_present": bool(packet.get("completion_gate_command_bat")),
        "completion_gate_precheck_valid": ((packet.get("completion_gate_precheck") or {}).get("valid")),
        "completion_gate_precheck_fail_closed": ((packet.get("completion_gate_precheck") or {}).get("fail_closed")),
        "completion_gate_precheck_claim_allowed": (
            (packet.get("completion_gate_precheck") or {}).get("q2_claim_allowed")
        ),
        "required_artifact_count": len(packet.get("required_artifacts") or {}),
        "required_postrun_provenance_count": len(packet.get("required_postrun_provenance") or {}),
        "register_postrun_provenance_paths_present": (_register_postrun_provenance_paths_present(packet)),
        "parent_control_count": len(parent_controls),
        "parent_controls_valid": all(parent["valid"] for parent in parent_controls),
        "parent_controls": parent_controls,
        "packet_md": str(args.packet_md),
        "markdown_runbook_valid": md_audit["valid"],
        "markdown_bat_block_count": md_audit["bat_block_count"],
        "markdown_wrapped_command_line_count": md_audit["wrapped_command_line_count"],
        "markdown_overlong_command_line_count": md_audit["overlong_command_line_count"],
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
    precheck = packet.get("completion_gate_precheck") or {}
    if precheck.get("valid") is not True:
        errors.append(f"postrun_completion_precheck_invalid={precheck.get('errors')}")
    if precheck.get("fail_closed") is not True:
        errors.append("postrun_completion_precheck_not_fail_closed")
    if len(packet.get("required_artifacts") or {}) < 8:
        errors.append("postrun_packet_required_artifacts_incomplete")
    postrun = packet.get("required_postrun_provenance") or {}
    if len(postrun) < 5:
        errors.append("postrun_packet_required_postrun_provenance_incomplete")
    command = packet.get("register_command") or []
    python_executable = str(packet.get("python_executable") or "")
    if not python_executable:
        errors.append("postrun_packet_missing_python_executable")
    if command and command[0] != python_executable:
        errors.append("postrun_register_command_python_mismatch")
    _require_option_values(
        errors,
        "postrun_register_command",
        command,
        (
            "--name",
            "--output-dir",
            "--metrics-json",
            "--experiment-stage",
            "--result-kind",
            "--primary-metric-unit",
            "--split-policy",
            "--dataset-snapshot-json",
            "--run-audit-json",
            "--calibration-audit-json",
            "--source-balanced-metrics-json",
            "--confusion-comparison-json",
            "--ablation-report-json",
            "--runtime-benchmark-audit-json",
            "--notes",
            "--parent-record-json",
            "--artifact",
        ),
    )
    for token in (
        "--paper-facing",
        "--artifact",
        "--run-audit-json",
        "--calibration-audit-json",
        "--source-balanced-metrics-json",
        "--confusion-comparison-json",
        "--ablation-report-json",
        "--runtime-benchmark-audit-json",
        "--parent-record-json",
    ):
        if token not in command:
            errors.append(f"postrun_register_command_missing={token}")
    parent_count = command.count("--parent-record-json")
    if parent_count < len(packet_writer._parent_record_artifacts()):
        errors.append(f"postrun_parent_records_incomplete={parent_count}")
    for key, path in postrun.items():
        if str(path) not in command:
            errors.append(f"postrun_register_missing_postrun_provenance={key}")
    completion = packet.get("completion_gate_command") or []
    if completion and completion[0] != python_executable:
        errors.append("postrun_completion_command_python_mismatch")
    _require_option_values(
        errors,
        "postrun_completion_command",
        completion,
        ("--output-dir", "--registry-record-json"),
    )
    if "--registry-record-json" not in completion:
        errors.append("postrun_completion_command_missing_registry_record")
    _require_option_values(
        errors,
        "postrun_calibration_command",
        packet.get("calibration_command") or [],
        ("--input-csv", "--output-dir", "--expected-fold-count"),
    )
    _require_option_values(
        errors,
        "postrun_confusion_comparison_command",
        packet.get("confusion_comparison_command") or [],
        (
            "--proposed-csv",
            "--baseline-csv",
            "--proposed-run-audit",
            "--baseline-run-audit",
            "--output-dir",
            "--expected-fold-count",
            "--bootstrap-iterations",
        ),
    )
    if not _cmd_ready(
        packet.get("calibration_cmd_command") or [],
        python_executable,
    ):
        errors.append("postrun_calibration_cmd_missing_cmd_setup")
    if not _cmd_wraps_base_command(
        packet.get("calibration_cmd_command") or [],
        packet.get("calibration_command") or [],
    ):
        errors.append("postrun_calibration_cmd_does_not_wrap_base_command")
    if not _cmd_ready(
        packet.get("confusion_comparison_cmd_command") or [],
        python_executable,
    ):
        errors.append("postrun_confusion_cmd_missing_cmd_setup")
    if not _cmd_wraps_base_command(
        packet.get("confusion_comparison_cmd_command") or [],
        packet.get("confusion_comparison_command") or [],
    ):
        errors.append("postrun_confusion_cmd_does_not_wrap_base_command")
    if not _cmd_ready(
        packet.get("ablation_report_cmd_command") or [],
        python_executable,
    ):
        errors.append("postrun_ablation_cmd_missing_cmd_setup")
    if not _cmd_wraps_base_command(
        packet.get("ablation_report_cmd_command") or [],
        packet.get("ablation_report_command") or [],
    ):
        errors.append("postrun_ablation_cmd_does_not_wrap_base_command")
    if not _cmd_ready(
        packet.get("register_cmd_command") or [],
        python_executable,
    ):
        errors.append("postrun_register_cmd_missing_cmd_setup")
    if not _cmd_wraps_base_command(
        packet.get("register_cmd_command") or [],
        packet.get("register_command") or [],
    ):
        errors.append("postrun_register_cmd_does_not_wrap_base_command")
    if not _cmd_ready(
        packet.get("completion_gate_cmd_command") or [],
        python_executable,
    ):
        errors.append("postrun_completion_cmd_missing_cmd_setup")
    if not _cmd_wraps_base_command(
        packet.get("completion_gate_cmd_command") or [],
        packet.get("completion_gate_command") or [],
    ):
        errors.append("postrun_completion_cmd_does_not_wrap_base_command")
    for key in (
        "calibration_command_bat",
        "confusion_comparison_command_bat",
        "ablation_report_command_bat",
        "register_command_bat",
        "completion_gate_command_bat",
    ):
        if not packet.get(key):
            errors.append(f"postrun_packet_missing_bat_command={key}")
    return errors


def _register_postrun_provenance_paths_present(packet: dict[str, Any]) -> bool:
    """Return whether every post-run provenance path is registered as artifact."""

    command = packet.get("register_command") or []
    return all(str(path) in command for path in (packet.get("required_postrun_provenance") or {}).values())


def _parent_control_audits(packet: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate linked parent controls before the full record is registered."""

    command = packet.get("register_command") or []
    paths: list[Path] = []
    for index, token in enumerate(command):
        if token != "--parent-record-json":
            continue
        next_index = index + 1
        if next_index >= len(command):
            paths.append(Path("<missing-parent-record-token>"))
            continue
        paths.append(Path(str(command[next_index])))
    return [check_parent_record_link(path) for path in paths]


def _require_option_values(
    errors: list[str],
    command_name: str,
    command: list[str],
    options: tuple[str, ...],
) -> None:
    """Ensure critical CLI options are followed by a non-option value."""

    for option in options:
        indices = [index for index, token in enumerate(command) if token == option]
        if not indices:
            errors.append(f"{command_name}_missing_option={option}")
            continue
        for index in indices:
            value_index = index + 1
            if value_index >= len(command):
                errors.append(f"{command_name}_missing_value={option}")
                continue
            value = str(command[value_index])
            if value == "" or value.startswith("--"):
                errors.append(f"{command_name}_invalid_value={option}:{value}")


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


def _cmd_wraps_base_command(command: list[str], base_command: list[str]) -> bool:
    """Check that the CMD helper preserves the exact underlying command."""

    if len(command) < 8 or not base_command:
        return False
    return command[7:] == base_command


def _markdown_runbook_audit(
    *,
    path: Path,
    expected_headings: tuple[str, ...],
    min_bat_blocks: int,
) -> dict[str, Any]:
    """Check generated Markdown has wrapped CMD blocks for human execution."""

    errors: list[str] = []
    if not path.exists():
        return {
            "valid": False,
            "errors": [f"missing_markdown_runbook={path}"],
            "bat_block_count": 0,
            "wrapped_command_line_count": 0,
            "overlong_command_line_count": 0,
        }
    text = path.read_text(encoding="utf-8")
    for heading in expected_headings:
        if heading not in text:
            errors.append(f"markdown_runbook_missing_heading={heading}")
    bat_block_count = text.count("```bat")
    if bat_block_count < min_bat_blocks:
        errors.append(f"markdown_runbook_bat_blocks_too_few={bat_block_count}")
    wrapped_lines = [line for line in text.splitlines() if line.rstrip().endswith("^")]
    if not wrapped_lines:
        errors.append("markdown_runbook_missing_cmd_continuation_lines")
    overlong_command_lines = [
        line for line in text.splitlines() if len(line) > 140 and (" && " in line or " --" in line)
    ]
    if overlong_command_lines:
        errors.append(f"markdown_runbook_has_unwrapped_long_command_lines={len(overlong_command_lines)}")
    return {
        "valid": not errors,
        "errors": errors,
        "bat_block_count": bat_block_count,
        "wrapped_command_line_count": len(wrapped_lines),
        "overlong_command_line_count": len(overlong_command_lines),
    }


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing_packet_json={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
