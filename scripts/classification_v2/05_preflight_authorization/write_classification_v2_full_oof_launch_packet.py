from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = (
    "Q2 internal recording-date/video-safe improvement only; no external "
    "farm/camera/cohort claim until external validation exists."
)
SCHEMA_VERSION = "classification_v2_full_oof_launch_packet_v1"


def main() -> None:
    """Write a human-review launch packet without authorizing full training."""

    parser = argparse.ArgumentParser(
        description="Write the classification_v2 full OOF launch packet."
    )
    parser.add_argument(
        "--preflight-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/full_multimodal_oof_preflight.json"),
    )
    parser.add_argument(
        "--run-plan-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/full_multimodal_oof_run_plan.json"),
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
        default=Path("outputs/classification_v2/model_design/full_oof_authorization_template.json"),
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
        default=Path("outputs/classification_v2/model_design/full_oof_launch_packet.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/classification_v2/model_design/full_oof_launch_packet.md"),
    )
    args = parser.parse_args()

    packet = build_launch_packet(
        preflight=_load_json(args.preflight_json),
        run_plan=_load_json(args.run_plan_json),
        runtime_benchmark=_load_json(args.runtime_benchmark_json),
        authorization_template=_load_json(args.authorization_template_json),
        preflight_freshness=_load_json(args.preflight_freshness_json),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_launch_packet_markdown(packet), encoding="utf-8")
    print(json.dumps(packet, indent=2))
    if not packet["valid"]:
        raise SystemExit(1)


def build_launch_packet(
    *,
    preflight: dict[str, Any],
    run_plan: dict[str, Any],
    runtime_benchmark: dict[str, Any],
    authorization_template: dict[str, Any],
    preflight_freshness: dict[str, Any],
) -> dict[str, Any]:
    """Combine pre-run evidence into one reviewable, fail-closed artifact."""

    config = run_plan.get("config") or {}
    runtime_config = runtime_benchmark.get("recommended_runtime_config") or {}
    command = _full_run_command(config)
    cmd_command = _cmd_launch_command(command)
    authorization_command = _authorization_command(preflight)
    cmd_authorization_command = _cmd_launch_command(authorization_command)
    errors = _launch_packet_errors(
        preflight=preflight,
        run_plan=run_plan,
        runtime_benchmark=runtime_benchmark,
        authorization_template=authorization_template,
        preflight_freshness=preflight_freshness,
        command=command,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not errors,
        "errors": errors,
        "status": "READY_FOR_HUMAN_AUTHORIZATION" if not errors else "INVALID",
        "claim_boundary": CLAIM_BOUNDARY,
        "full_oof_execution_allowed": False,
        "authorization_required": True,
        "authorization_template_authorized": authorization_template.get("authorized"),
        "preflight_config_sha256": preflight.get("config_sha256"),
        "run_plan_config_sha256": run_plan.get("config_sha256"),
        "git_commit": preflight.get("git_commit"),
        "snapshot_id": preflight.get("snapshot_id"),
        "snapshot_file_sha256": preflight.get("snapshot_file_sha256"),
        "lineage_audit_sha256": preflight.get("lineage_audit_sha256"),
        "ordered_window_id_sha256": (
            (preflight.get("lineage_binding_audit") or {}).get(
                "expected_ordered_window_id_sha256"
            )
        ),
        "output_dir": config.get("output_dir"),
        "selected_fold_count": run_plan.get("selected_fold_count"),
        "available_fold_count": run_plan.get("available_fold_count"),
        "total_eval_rows": run_plan.get("total_eval_rows"),
        "total_train_steps": run_plan.get("total_train_steps"),
        "estimated_training_seconds_excluding_eval": preflight.get(
            "estimated_training_seconds_excluding_eval"
        ),
        "estimated_training_minutes_excluding_eval": _runtime_minutes(
            preflight.get("estimated_training_seconds_excluding_eval")
        ),
        "runtime_estimate_scope": (
            "Training only; excludes evaluation, bootstrap metrics, startup, "
            "checkpoint IO, and manual review time."
        ),
        "runtime_benchmark": {
            "precision": runtime_config.get("precision"),
            "train_batch_size": runtime_config.get("train_batch_size"),
            "throughput_rows_per_sec": runtime_config.get("throughput_rows_per_sec"),
            "peak_reserved_memory_mb": runtime_config.get("peak_reserved_memory_mb"),
            "max_reserved_memory_mb": runtime_benchmark.get("max_reserved_memory_mb"),
            "warnings": runtime_benchmark.get("warnings") or [],
        },
        "cache_paths": {
            "actor_packed_npy": config.get("packed_image_cache_npy"),
            "actor_packed_index_csv": config.get("packed_image_cache_index_csv"),
            "visual_context_manifest_csv": config.get("visual_context_cache_manifest_csv"),
            "visual_context_packed_npy": config.get("visual_context_packed_cache_npy"),
            "visual_context_packed_index_csv": config.get("visual_context_packed_cache_index_csv"),
        },
        "python_executable": command[0],
        "launch_command": command,
        "cmd_launch_command": cmd_command,
        "cmd_launch_command_bat": subprocess.list2cmdline(cmd_command),
        "authorization_command": authorization_command,
        "cmd_authorization_command": cmd_authorization_command,
        "cmd_authorization_command_bat": subprocess.list2cmdline(cmd_authorization_command),
        "review_checklist": _review_checklist(),
    }


def render_launch_packet_markdown(packet: dict[str, Any]) -> str:
    """Render the launch packet as a compact runbook for human review."""

    checklist = packet.get("review_checklist") or []
    command = _wrap_bat_command_for_markdown(str(packet.get("cmd_launch_command_bat") or ""))
    authorization_command = _wrap_bat_command_for_markdown(
        str(packet.get("cmd_authorization_command_bat") or "")
    )
    lines = [
        "# classification_v2 Full OOF Launch Packet",
        "",
        f"Status: **{packet.get('status')}**",
        "",
        f"Claim boundary: {packet.get('claim_boundary')}",
        "",
        "## Binding",
        f"- Git commit: `{packet.get('git_commit')}`",
        f"- Snapshot: `{packet.get('snapshot_id')}`",
        f"- Snapshot file SHA256: `{packet.get('snapshot_file_sha256')}`",
        f"- Lineage audit SHA256: `{packet.get('lineage_audit_sha256')}`",
        "- Ordered window ID SHA256: "
        f"`{packet.get('ordered_window_id_sha256')}`",
        f"- Config SHA256: `{packet.get('preflight_config_sha256')}`",
        "",
        "## Execution",
        f"- Output dir: `{packet.get('output_dir')}`",
        f"- Folds: `{packet.get('selected_fold_count')}`",
        f"- Eval rows: `{packet.get('total_eval_rows')}`",
        f"- Train steps: `{packet.get('total_train_steps')}`",
        "- Estimated training minutes, excluding eval/bootstrap/IO: "
        f"`{packet.get('estimated_training_minutes_excluding_eval')}`",
        "",
        "## CMD Command",
        "```bat",
        command,
        "```",
        "",
        "## Authorization Command Template",
        "Review the packet, replace `<REVIEWER>` with the human reviewer name, "
        "then run this before the full OOF command.",
        "",
        "```bat",
        authorization_command,
        "```",
        "",
        "## Checklist",
    ]
    lines.extend(f"- {item}" for item in checklist)
    return "\n".join(lines) + "\n"


def _wrap_bat_command_for_markdown(command: str) -> str:
    """Wrap a CMD command for review without changing the audited JSON value."""

    wrapped = command.replace(" && ", " ^\n  && ")
    wrapped = wrapped.replace(
        " scripts\\classification_v2\\",
        " ^\n  scripts\\classification_v2\\",
    )
    return wrapped.replace(" --", " ^\n  --")


def _full_run_command(config: dict[str, Any]) -> list[str]:
    """Build the exact full OOF command while leaving authorization external."""

    return [
        sys.executable,
        "scripts\\classification_v2\\06_full_oof_training\\"
        "classification_v2_run_full_multimodal_oof.py",
        "--full",
        "--confirm-full-run",
        "--preflight-json",
        "outputs\\classification_v2\\model_design\\full_multimodal_oof_preflight.json",
        "--authorization-json",
        "outputs\\classification_v2\\model_design\\full_oof_authorization.json",
        "--output-dir",
        str(config.get("output_dir") or ""),
        "--packed-image-cache",
        str(config.get("packed_image_cache_npy") or ""),
        "--packed-image-cache-index",
        str(config.get("packed_image_cache_index_csv") or ""),
        "--visual-context-cache-manifest",
        str(config.get("visual_context_cache_manifest_csv") or ""),
        "--visual-context-packed-cache",
        str(config.get("visual_context_packed_cache_npy") or ""),
        "--visual-context-packed-cache-index",
        str(config.get("visual_context_packed_cache_index_csv") or ""),
        "--require-packed-visual-context",
        "--image-size",
        str(config.get("image_size") or ""),
        "--hidden-dim",
        str(config.get("hidden_dim") or ""),
        "--steps-per-fold",
        str(config.get("steps_per_fold") or ""),
        "--train-batch-size",
        str(config.get("train_batch_size") or ""),
        "--eval-batch-size",
        str(config.get("eval_batch_size") or ""),
        "--bootstrap-iterations",
        str(config.get("bootstrap_iterations") or ""),
        "--device",
        str(config.get("device") or ""),
        "--precision",
        str(config.get("precision") or ""),
    ]


def _authorization_command(preflight: dict[str, Any]) -> list[str]:
    """Build the explicit human authorization command template."""

    return [
        sys.executable,
        "scripts\\classification_v2\\05_preflight_authorization\\"
        "write_classification_v2_full_oof_authorization_file.py",
        "--authorize",
        "--reviewer",
        "<REVIEWER>",
        "--acknowledge-long-run",
        "--acknowledge-no-q2-claim",
        "--preflight-config-sha256",
        str(preflight.get("config_sha256") or ""),
        "--git-commit",
        str(preflight.get("git_commit") or ""),
    ]


def _cmd_launch_command(command: list[str]) -> list[str]:
    """Wrap the launch command with project-root and PYTHONPATH setup for CMD."""

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


def _runtime_minutes(seconds: Any) -> float | None:
    """Convert a preflight runtime estimate to minutes for human review."""

    if seconds is None:
        return None
    return round(float(seconds) / 60.0, 2)


def _launch_packet_errors(
    *,
    preflight: dict[str, Any],
    run_plan: dict[str, Any],
    runtime_benchmark: dict[str, Any],
    authorization_template: dict[str, Any],
    preflight_freshness: dict[str, Any],
    command: list[str],
) -> list[str]:
    """Validate that the packet is complete but still not an approval."""

    errors: list[str] = []
    if preflight.get("valid") is not True or preflight.get("errors"):
        errors.append(f"preflight_not_valid={preflight.get('errors')}")
    if run_plan.get("valid") is not True or run_plan.get("errors"):
        errors.append(f"run_plan_not_valid={run_plan.get('errors')}")
    if runtime_benchmark.get("valid") is not True:
        errors.append(f"runtime_benchmark_not_valid={runtime_benchmark.get('errors')}")
    if preflight.get("estimated_training_seconds_excluding_eval") is None:
        errors.append("missing_training_runtime_estimate")
    if preflight.get("config_sha256") != run_plan.get("config_sha256"):
        errors.append("preflight_run_plan_config_sha256_mismatch")
    if preflight_freshness.get("preflight_fresh") is not True:
        errors.append("preflight_not_fresh")
    if preflight_freshness.get("preflight_authorization_ready") is not True:
        errors.append("preflight_not_authorization_ready")
    if preflight_freshness.get("full_oof_execution_allowed") is not False:
        errors.append("launch_packet_must_not_authorize_execution")
    if authorization_template.get("authorized") is not False:
        errors.append("authorization_template_must_default_unauthorized")
    if authorization_template.get("preflight_config_sha256") != preflight.get("config_sha256"):
        errors.append("authorization_template_config_sha256_mismatch")
    if authorization_template.get("git_commit") != preflight.get("git_commit"):
        errors.append("authorization_template_git_commit_mismatch")
    expected_ordered_hash = (
        (preflight.get("lineage_binding_audit") or {}).get(
            "expected_ordered_window_id_sha256"
        )
    )
    expected_bindings = {
        "snapshot_id": preflight.get("snapshot_id"),
        "snapshot_file_sha256": preflight.get("snapshot_file_sha256"),
        "lineage_audit_sha256": preflight.get("lineage_audit_sha256"),
        "ordered_window_id_sha256": expected_ordered_hash,
    }
    for field, expected in expected_bindings.items():
        if not expected or authorization_template.get(field) != expected:
            errors.append(
                f"authorization_template_binding_mismatch={field}"
            )
    required_tokens = {
        "--full",
        "--confirm-full-run",
        "--authorization-json",
        "--packed-image-cache",
        "--packed-image-cache-index",
        "--visual-context-cache-manifest",
        "--visual-context-packed-cache",
        "--visual-context-packed-cache-index",
        "--require-packed-visual-context",
    }
    missing = sorted(required_tokens.difference(command))
    if missing:
        errors.append(f"launch_command_missing_required_tokens={missing}")
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
        errors.append(f"launch_command_missing_required_values={missing_values}")
    return errors


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


def _review_checklist() -> list[str]:
    return [
        "Confirm CUDA/GPU availability on the target machine.",
        "Confirm canonical packed actor and visual-context caches exist.",
        "Confirm output_dir is under outputs/classification_v2/model_full.",
        "Confirm preflight config hash and Git commit match authorization JSON.",
        "Confirm reviewer understands this is a long full OOF run.",
        "Confirm no Q2 claim is made until full OOF metrics are complete.",
    ]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
