from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "classification_v2_full_oof_postrun_registration_packet_v1"
DEFAULT_FULL_OUTPUT_DIR = Path("outputs/classification_v2/model_full/full_multimodal_oof")
DEFAULT_REGISTRY_DIR = Path("outputs/classification_v2/experiment_registry")
DEFAULT_RECORD_JSON = DEFAULT_REGISTRY_DIR / "full_multimodal_oof_record.json"


def main() -> None:
    """Write post-run registry commands without registering an experiment."""

    parser = argparse.ArgumentParser(
        description="Write classification_v2 full OOF post-run registration packet."
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
            "full_oof_postrun_registration_packet.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_postrun_registration_packet.md"
        ),
    )
    args = parser.parse_args()

    packet = build_postrun_registration_packet(
        output_dir=args.output_dir,
        registry_dir=args.registry_dir,
        runtime_benchmark_audit_json=args.runtime_benchmark_audit_json,
        completion_gate_json=args.completion_gate_json,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_postrun_packet_markdown(packet), encoding="utf-8")
    print(json.dumps(packet, indent=2))
    if not packet["valid"]:
        raise SystemExit(1)


def build_postrun_registration_packet(
    *,
    output_dir: Path,
    registry_dir: Path,
    runtime_benchmark_audit_json: Path,
    completion_gate_json: Path,
) -> dict[str, Any]:
    """Describe the exact post-full-OOF registration and verification steps."""

    artifacts = _full_oof_artifacts(output_dir)
    postrun = _postrun_provenance_artifacts(output_dir)
    record_json = registry_dir / "full_multimodal_oof_record.json"
    calibration_command = _calibration_command(
        artifacts=artifacts,
        output_dir=output_dir,
    )
    confusion_command = _confusion_comparison_command(
        artifacts=artifacts,
        output_dir=output_dir,
    )
    ablation_command = _ablation_report_command()
    register_command = _register_command(
        output_dir=output_dir,
        registry_dir=registry_dir,
        artifacts=artifacts,
        postrun=postrun,
        runtime_benchmark_audit_json=runtime_benchmark_audit_json,
    )
    completion_command = _completion_gate_command(
        output_dir=output_dir,
        record_json=record_json,
    )
    packet = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "errors": [],
        "status": "READY_FOR_POST_FULL_OOF_REGISTRATION",
        "runs_training": False,
        "runs_registration": False,
        "q2_claim_allowed_by_packet": False,
        "output_dir": str(output_dir),
        "registry_record_json": str(record_json),
        "required_artifacts": {name: str(path) for name, path in artifacts.items()},
        "required_postrun_provenance": {
            name: str(path) for name, path in postrun.items()
        },
        "python_executable": sys.executable,
        "calibration_command": calibration_command,
        "confusion_comparison_command": confusion_command,
        "ablation_report_command": ablation_command,
        "register_command": register_command,
        "completion_gate_command": completion_command,
        "calibration_cmd_command": _cmd_command(calibration_command),
        "confusion_comparison_cmd_command": _cmd_command(confusion_command),
        "ablation_report_cmd_command": _cmd_command(ablation_command),
        "register_cmd_command": _cmd_command(register_command),
        "completion_gate_cmd_command": _cmd_command(completion_command),
        "calibration_command_bat": subprocess.list2cmdline(
            _cmd_command(calibration_command)
        ),
        "confusion_comparison_command_bat": subprocess.list2cmdline(
            _cmd_command(confusion_command)
        ),
        "ablation_report_command_bat": subprocess.list2cmdline(
            _cmd_command(ablation_command)
        ),
        "register_command_bat": subprocess.list2cmdline(
            _cmd_command(register_command)
        ),
        "completion_gate_command_bat": subprocess.list2cmdline(
            _cmd_command(completion_command)
        ),
        "postrun_order": [
            "Run full OOF only through the authorization-gated launch packet.",
            "Run calibration_command after full native-unit predictions exist.",
            "Run confusion_comparison_command after calibrated predictions exist.",
            "Run ablation_report_command to refresh shortcut/ablation evidence.",
            "Run the register_command after all provenance exists and passes.",
            "Run the completion_gate_command to unlock q2_claim_allowed if valid.",
            "Do not make a Q2 result claim until completion gate allows it.",
        ],
    }
    packet["errors"] = _packet_errors(packet)
    packet["valid"] = not packet["errors"]
    return packet


def render_postrun_packet_markdown(packet: dict[str, Any]) -> str:
    """Render a compact post-run runbook for human execution."""

    lines = [
        "# classification_v2 Full OOF Post-Run Registration Packet",
        "",
        f"Status: **{packet.get('status')}**",
        "",
        f"Output dir: `{packet.get('output_dir')}`",
        f"Registry record: `{packet.get('registry_record_json')}`",
        "",
        "## Calibration Command",
        "```bat",
        str(packet.get("calibration_command_bat") or ""),
        "```",
        "",
        "## Confusion Comparison Command",
        "```bat",
        str(packet.get("confusion_comparison_command_bat") or ""),
        "```",
        "",
        "## Ablation Report Command",
        "```bat",
        str(packet.get("ablation_report_command_bat") or ""),
        "```",
        "",
        "## Register Command",
        "```bat",
        str(packet.get("register_command_bat") or ""),
        "```",
        "",
        "## Completion Gate Command",
        "```bat",
        str(packet.get("completion_gate_command_bat") or ""),
        "```",
        "",
        "## Order",
    ]
    lines.extend(f"- {item}" for item in packet.get("postrun_order") or [])
    return "\n".join(lines) + "\n"


def _register_command(
    *,
    output_dir: Path,
    registry_dir: Path,
    artifacts: dict[str, Path],
    postrun: dict[str, Path],
    runtime_benchmark_audit_json: Path,
) -> list[str]:
    """Build the registry command with full OOF provenance paths."""

    parent_records = _parent_record_artifacts()
    command = [
        sys.executable,
        "scripts\\behavior_review_tools\\classification_v2_register_experiment.py",
        "--name",
        "full_multimodal_oof",
        "--output-dir",
        str(registry_dir),
        "--metrics-json",
        str(artifacts["metrics"]),
        "--experiment-stage",
        "paper_facing_candidate",
        "--paper-facing",
        "--result-kind",
        "model_evaluation",
        "--primary-metric-unit",
        "native_temporal_unit",
        "--split-policy",
        "recording_group_oof",
        "--dataset-snapshot-json",
        "outputs\\classification_v2\\training_snapshots\\c2v2_27ed5c9963904c52.json",
        "--run-audit-json",
        str(artifacts["run_audit"]),
        "--calibration-audit-json",
        str(postrun["calibration_audit"]),
        "--source-balanced-metrics-json",
        str(artifacts["source_balanced_report"]),
        "--confusion-comparison-json",
        str(postrun["confusion_comparison"]),
        "--ablation-report-json",
        str(postrun["ablation_report"]),
        "--runtime-benchmark-audit-json",
        str(runtime_benchmark_audit_json),
        "--notes",
        "Full multimodal native-OOF evaluation; Q2 internal validation only.",
    ]
    for path in parent_records:
        command.extend(["--parent-record-json", str(path)])
    for key in sorted(artifacts):
        command.extend(["--artifact", str(artifacts[key])])
    for key in sorted(postrun):
        command.extend(["--artifact", str(postrun[key])])
    return command


def _calibration_command(*, artifacts: dict[str, Path], output_dir: Path) -> list[str]:
    """Build cross-fit calibration command for full native-unit predictions."""

    return [
        sys.executable,
        "scripts\\behavior_review_tools\\classification_v2_cross_fit_calibration.py",
        "--input-csv",
        str(artifacts["unit_predictions"]),
        "--output-dir",
        str(output_dir / "calibration"),
        "--expected-fold-count",
        "13",
    ]


def _confusion_comparison_command(
    *,
    artifacts: dict[str, Path],
    output_dir: Path,
) -> list[str]:
    """Build command comparing full model errors against the native baseline."""

    return [
        sys.executable,
        "scripts\\behavior_review_tools\\classification_v2_compare_confusion_focus.py",
        "--proposed-csv",
        str(output_dir / "calibration" / "cross_fitted_calibrated_native_predictions.csv"),
        "--baseline-csv",
        "outputs\\classification_v2\\model_smoke\\native_majority_baseline"
        "\\native_majority_unit_predictions.csv",
        "--proposed-run-audit",
        str(artifacts["run_audit"]),
        "--baseline-run-audit",
        "outputs\\classification_v2\\model_smoke\\native_majority_baseline"
        "\\native_majority_audit.json",
        "--output-dir",
        str(output_dir / "confusion_focus"),
        "--expected-fold-count",
        "13",
        "--bootstrap-iterations",
        "2000",
    ]


def _ablation_report_command() -> list[str]:
    """Build command refreshing the predeclared Q2 ablation reporting audit."""

    return [
        sys.executable,
        "scripts\\dev_tools\\check_classification_v2_ablation_reporting.py",
    ]


def _completion_gate_command(*, output_dir: Path, record_json: Path) -> list[str]:
    """Build the post-registration completion gate command."""

    return [
        sys.executable,
        "scripts\\dev_tools\\check_classification_v2_full_oof_completion_gate.py",
        "--output-dir",
        str(output_dir),
        "--registry-record-json",
        str(record_json),
    ]


def _cmd_command(command: list[str]) -> list[str]:
    """Wrap post-run commands with project-root and PYTHONPATH setup for CMD."""

    return [
        "cd",
        "/d",
        str(Path.cwd()),
        "&&",
        "set",
        "PYTHONPATH=%CD%\\src",
        "&&",
        *command,
    ]


def _postrun_provenance_artifacts(output_dir: Path) -> dict[str, Path]:
    """Return Q2 model-evaluation provenance created after full OOF training."""

    return {
        "ablation_report": Path(
            "outputs/classification_v2/model_design/ablation_reporting_audit.json"
        ),
        "calibrated_predictions": (
            output_dir
            / "calibration"
            / "cross_fitted_calibrated_native_predictions.csv"
        ),
        "calibration_audit": (
            output_dir / "calibration" / "cross_fitted_calibration_audit.json"
        ),
        "confusion_comparison": (
            output_dir / "confusion_focus" / "confusion_focus_comparison.json"
        ),
        "high_confidence_hard_errors": (
            output_dir / "confusion_focus" / "high_confidence_hard_errors.csv"
        ),
    }


def _parent_record_artifacts() -> tuple[Path, ...]:
    """Return paper-facing control records linked by the full model record."""

    registry = Path("outputs/classification_v2/experiment_registry")
    return (
        registry / "native_majority_baseline_record.json",
        registry / "tabular_linear_baseline_record.json",
        registry / "tabular_nonlinear_baseline_record.json",
    )


def _full_oof_artifacts(output_dir: Path) -> dict[str, Path]:
    return {
        "metrics": output_dir / "full_multimodal_oof_metrics.json",
        "predictions": output_dir / "full_multimodal_oof_predictions.csv",
        "prediction_schema_audit": (
            output_dir / "full_multimodal_oof_prediction_schema_audit.json"
        ),
        "run_audit": output_dir / "full_multimodal_oof_audit.json",
        "source_balanced_native_units": (
            output_dir / "source_balanced_native_units.csv"
        ),
        "source_balanced_report": output_dir / "source_balanced_report.json",
        "source_balanced_selection": output_dir / "source_balanced_selection.csv",
        "unit_predictions": output_dir / "full_multimodal_oof_unit_predictions.csv",
    }


def _packet_errors(packet: dict[str, Any]) -> list[str]:
    """Ensure the packet is a runbook, not a hidden registration side effect."""

    errors: list[str] = []
    if packet.get("runs_training") is not False:
        errors.append("postrun_packet_must_not_run_training")
    if packet.get("runs_registration") is not False:
        errors.append("postrun_packet_must_not_run_registration")
    if packet.get("q2_claim_allowed_by_packet") is not False:
        errors.append("postrun_packet_must_not_allow_q2_claim")
    command = packet.get("register_command") or []
    required = {
        "--paper-facing",
        "--run-audit-json",
        "--calibration-audit-json",
        "--source-balanced-metrics-json",
        "--confusion-comparison-json",
        "--ablation-report-json",
        "--runtime-benchmark-audit-json",
        "--parent-record-json",
        "--artifact",
    }
    missing = sorted(required.difference(command))
    if missing:
        errors.append(f"register_command_missing_tokens={missing}")
    parent_record_count = command.count("--parent-record-json")
    if parent_record_count < len(_parent_record_artifacts()):
        errors.append(f"register_command_parent_records_incomplete={parent_record_count}")
    for key, path in (packet.get("required_postrun_provenance") or {}).items():
        if str(path) not in command:
            errors.append(f"register_command_missing_postrun_provenance={key}")
    _require_option_values(
        errors,
        "register_command",
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
    _require_option_values(
        errors,
        "calibration_command",
        packet.get("calibration_command") or [],
        ("--input-csv", "--output-dir", "--expected-fold-count"),
    )
    _require_option_values(
        errors,
        "confusion_comparison_command",
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
    completion = packet.get("completion_gate_command") or []
    _require_option_values(
        errors,
        "completion_command",
        completion,
        ("--output-dir", "--registry-record-json"),
    )
    if "--registry-record-json" not in completion:
        errors.append("completion_command_missing_registry_record_json")
    if not _cmd_ready(packet.get("calibration_cmd_command") or []):
        errors.append("calibration_cmd_command_missing_cmd_setup")
    if not _cmd_ready(packet.get("confusion_comparison_cmd_command") or []):
        errors.append("confusion_cmd_command_missing_cmd_setup")
    if not _cmd_ready(packet.get("ablation_report_cmd_command") or []):
        errors.append("ablation_cmd_command_missing_cmd_setup")
    if not _cmd_ready(packet.get("register_cmd_command") or []):
        errors.append("register_cmd_command_missing_cmd_setup")
    if not _cmd_ready(packet.get("completion_gate_cmd_command") or []):
        errors.append("completion_cmd_command_missing_cmd_setup")
    return errors


def _require_option_values(
    errors: list[str],
    command_name: str,
    command: list[str],
    options: tuple[str, ...],
) -> None:
    """Require critical CLI options to be followed by a non-option value."""

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


def _cmd_ready(command: list[str]) -> bool:
    """Return whether a stored CMD command enters the project import context."""

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
    return command[6] == "&&" and command[7] == sys.executable


if __name__ == "__main__":
    main()
