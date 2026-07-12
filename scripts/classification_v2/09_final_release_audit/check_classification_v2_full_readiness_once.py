from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

FULL_COMMAND_TOKENS = (
    "--full",
    "--confirm-full-run",
    "--preflight-json",
    "--authorization-json",
    "--packed-image-cache",
    "--packed-image-cache-index",
    "--visual-context-cache-manifest",
    "--visual-context-packed-cache",
    "--visual-context-packed-cache-index",
    "--require-packed-visual-context",
)

AUTHORIZATION_COMMAND_TOKENS = (
    "--authorize",
    "--reviewer",
    "<REVIEWER>",
    "--acknowledge-long-run",
    "--acknowledge-no-q2-claim",
    "--preflight-config-sha256",
    "--git-commit",
)


def main() -> None:
    """Write one consolidated pre-full readiness verdict without running full OOF."""

    parser = argparse.ArgumentParser(
        description=(
            "Read existing classification_v2 pre-full artifacts and produce one non-training readiness verdict."
        )
    )
    parser.add_argument(
        "--model-design-dir",
        type=Path,
        default=Path("outputs/classification_v2/model_design"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/full_oof_readiness_once_audit.json"),
    )
    args = parser.parse_args()

    audit = build_readiness_audit(args.model_design_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def build_readiness_audit(model_design_dir: Path) -> dict[str, Any]:
    """Consolidate pre-full state so operators do not rerun every gate manually."""

    errors: list[str] = []
    warnings: list[str] = []
    q2_audit = _load_json(
        model_design_dir / "q2_progress_report_audit.json",
        errors,
        "q2_progress_report_audit",
    )
    preflight = _load_json(
        model_design_dir / "full_multimodal_oof_preflight.json",
        errors,
        "full_multimodal_oof_preflight",
    )
    authorization = _load_json(
        model_design_dir / "full_oof_authorization.json",
        errors,
        "full_oof_authorization",
    )
    launch = _load_json(
        model_design_dir / "full_oof_launch_packet.json",
        errors,
        "full_oof_launch_packet",
    )
    launch_audit = _load_json(
        model_design_dir / "full_oof_launch_packet_audit.json",
        errors,
        "full_oof_launch_packet_audit",
    )
    execution_gate = _load_json(
        model_design_dir / "full_oof_execution_gate_audit.json",
        errors,
        "full_oof_execution_gate_audit",
    )
    completion_gate = _load_json(
        model_design_dir / "full_oof_completion_gate_audit.json",
        errors,
        "full_oof_completion_gate_audit",
    )
    postrun = _load_json(
        model_design_dir / "full_oof_postrun_registration_packet.json",
        errors,
        "full_oof_postrun_registration_packet",
    )
    postrun_audit = _load_json(
        model_design_dir / "full_oof_postrun_registration_packet_audit.json",
        errors,
        "full_oof_postrun_registration_packet_audit",
    )
    git_state = _git_state()

    _check_pre_full_gates(q2_audit, preflight, git_state, errors)
    _check_launch_packet(launch, launch_audit, errors)
    _check_authorization_binding(authorization, preflight, errors)
    _check_execution_gate(execution_gate, errors)
    _check_postrun_packet(postrun, postrun_audit, completion_gate, errors)

    blockers = _authorization_blockers(authorization)
    status = _status(errors, blockers, launch_audit, completion_gate)
    if blockers:
        warnings.append("full_oof_waits_for_human_authorization")

    return {
        "schema_version": "classification_v2_full_oof_readiness_once_v1",
        "valid": not errors,
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "blockers": blockers,
        "git": git_state,
        "preflight": {
            "valid": preflight.get("valid"),
            "config_sha256": preflight.get("config_sha256"),
            "git_commit": preflight.get("git_commit"),
            "snapshot_id": preflight.get("snapshot_id"),
            "cuda_available": preflight.get("cuda_available"),
            "cuda_device_name": preflight.get("cuda_device_name"),
            "selected_fold_count": preflight.get("selected_fold_count"),
            "total_eval_rows": preflight.get("total_eval_rows"),
            "estimated_training_minutes_excluding_eval": round(
                float(preflight.get("estimated_training_seconds_excluding_eval") or 0) / 60.0,
                2,
            ),
        },
        "q2_progress": {
            "overall_status": q2_audit.get("overall_status"),
            "gate_count": q2_audit.get("gate_count"),
            "all_gates_passed": q2_audit.get("all_gates_passed"),
            "full_oof_execution_allowed": q2_audit.get("full_oof_execution_allowed"),
            "authorization_authorized": q2_audit.get("authorization_authorized"),
            "q2_claim_allowed": q2_audit.get("q2_claim_allowed"),
        },
        "authorization": {
            "authorized": authorization.get("authorized"),
            "acknowledges_long_run": authorization.get("acknowledges_long_run"),
            "acknowledges_no_q2_claim_until_verified": authorization.get("acknowledges_no_q2_claim_until_verified"),
            "reviewer_present": bool(str(authorization.get("reviewer") or "")),
            "reviewed_at_present": bool(str(authorization.get("reviewed_at") or "")),
        },
        "launch_packet": {
            "status": launch.get("status"),
            "cmd_launch_command_ready": launch_audit.get("cmd_launch_command_ready"),
            "authorization_command_ready": launch_audit.get("authorization_command_ready"),
            "markdown_runbook_valid": launch_audit.get("markdown_runbook_valid"),
            "markdown_overlong_command_line_count": launch_audit.get("markdown_overlong_command_line_count"),
        },
        "execution_gate": {
            "valid": execution_gate.get("valid"),
            "case_count": execution_gate.get("case_count"),
            "full_training_invoked": execution_gate.get("full_training_invoked"),
        },
        "postrun_packet": {
            "status": postrun.get("status"),
            "audit_valid": postrun_audit.get("valid"),
            "q2_claim_allowed_by_packet": postrun.get("q2_claim_allowed_by_packet"),
            "completion_gate_precheck_claim_allowed": postrun_audit.get("completion_gate_precheck_claim_allowed"),
            "markdown_runbook_valid": postrun_audit.get("markdown_runbook_valid"),
            "markdown_overlong_command_line_count": postrun_audit.get("markdown_overlong_command_line_count"),
        },
        "next_action": _next_action(blockers, launch),
    }


def _check_pre_full_gates(
    q2_audit: dict[str, Any],
    preflight: dict[str, Any],
    git_state: dict[str, Any],
    errors: list[str],
) -> None:
    """Verify the consolidated progress audit matches current git state."""

    _require_true(errors, "q2_progress_valid", q2_audit.get("valid"))
    _require_equal(
        errors,
        "q2_progress_status",
        q2_audit.get("overall_status"),
        "PASS_PARTIAL_ROADMAP",
    )
    _require_true(errors, "q2_all_gates_passed", q2_audit.get("all_gates_passed"))
    _require_equal(errors, "q2_gate_count", q2_audit.get("gate_count"), 44)
    _require_true(errors, "preflight_valid", preflight.get("valid"))
    _require_false(errors, "q2_claim_allowed", q2_audit.get("q2_claim_allowed"))
    _require_false(
        errors,
        "full_oof_execution_allowed",
        q2_audit.get("full_oof_execution_allowed"),
    )
    _require_equal(
        errors,
        "q2_current_git_commit",
        q2_audit.get("current_git_commit"),
        git_state.get("commit"),
    )
    _require_false(errors, "git_dirty", git_state.get("dirty"))


def _check_launch_packet(
    launch: dict[str, Any],
    launch_audit: dict[str, Any],
    errors: list[str],
) -> None:
    """Check the human launch packet is complete but still fail-closed."""

    _require_equal(
        errors,
        "launch_packet_status",
        launch.get("status"),
        "READY_FOR_HUMAN_AUTHORIZATION",
    )
    _require_false(
        errors,
        "launch_full_oof_execution_allowed",
        launch.get("full_oof_execution_allowed"),
    )
    _require_true(errors, "launch_audit_valid", launch_audit.get("valid"))
    _require_true(
        errors,
        "launch_cmd_ready",
        launch_audit.get("cmd_launch_command_ready"),
    )
    _require_true(
        errors,
        "authorization_cmd_ready",
        launch_audit.get("authorization_command_ready"),
    )
    _require_command_tokens(
        errors,
        "launch_cmd_bat",
        launch.get("cmd_launch_command_bat"),
        FULL_COMMAND_TOKENS,
    )
    _require_command_tokens(
        errors,
        "authorization_cmd_bat",
        launch.get("cmd_authorization_command_bat"),
        AUTHORIZATION_COMMAND_TOKENS,
    )
    _require_true(
        errors,
        "launch_markdown_runbook_valid",
        launch_audit.get("markdown_runbook_valid"),
    )
    _require_equal(
        errors,
        "launch_markdown_overlong_lines",
        launch_audit.get("markdown_overlong_command_line_count"),
        0,
    )


def _check_authorization_binding(
    authorization: dict[str, Any],
    preflight: dict[str, Any],
    errors: list[str],
) -> None:
    """Confirm authorization file is bound to this preflight even when locked."""

    _require_equal(
        errors,
        "authorization_schema",
        authorization.get("schema_version"),
        "classification_v2_full_oof_authorization_v1",
    )
    _require_equal(
        errors,
        "authorization_config_sha256",
        authorization.get("preflight_config_sha256"),
        preflight.get("config_sha256"),
    )
    _require_equal(
        errors,
        "authorization_git_commit",
        authorization.get("git_commit"),
        preflight.get("git_commit"),
    )


def _check_execution_gate(
    execution_gate: dict[str, Any],
    errors: list[str],
) -> None:
    """Ensure the pre-full execution gate remains a no-training audit."""

    _require_true(errors, "execution_gate_valid", execution_gate.get("valid"))
    _require_equal(errors, "execution_gate_case_count", execution_gate.get("case_count"), 4)
    _require_false(
        errors,
        "execution_gate_full_training_invoked",
        execution_gate.get("full_training_invoked"),
    )


def _check_postrun_packet(
    postrun: dict[str, Any],
    postrun_audit: dict[str, Any],
    completion_gate: dict[str, Any],
    errors: list[str],
) -> None:
    """Check postrun registration is prepared but cannot unlock Q2 pre-full."""

    _require_equal(
        errors,
        "postrun_packet_status",
        postrun.get("status"),
        "READY_FOR_POST_FULL_OOF_REGISTRATION",
    )
    _require_true(errors, "postrun_audit_valid", postrun_audit.get("valid"))
    _require_false(
        errors,
        "postrun_q2_claim_allowed_by_packet",
        postrun.get("q2_claim_allowed_by_packet"),
    )
    _require_true(errors, "completion_gate_valid", completion_gate.get("valid"))
    _require_false(
        errors,
        "completion_gate_q2_claim_allowed",
        completion_gate.get("q2_claim_allowed"),
    )
    _require_true(
        errors,
        "postrun_markdown_runbook_valid",
        postrun_audit.get("markdown_runbook_valid"),
    )
    _require_equal(
        errors,
        "postrun_markdown_overlong_lines",
        postrun_audit.get("markdown_overlong_command_line_count"),
        0,
    )


def _authorization_blockers(authorization: dict[str, Any]) -> list[str]:
    """Return human-action blockers that are expected before full OOF."""

    blockers: list[str] = []
    if authorization.get("authorized") is not True:
        blockers.append("authorization_authorized_false")
    if authorization.get("acknowledges_long_run") is not True:
        blockers.append("missing_long_run_acknowledgement")
    if authorization.get("acknowledges_no_q2_claim_until_verified") is not True:
        blockers.append("missing_no_q2_claim_acknowledgement")
    if not str(authorization.get("reviewer") or ""):
        blockers.append("missing_reviewer")
    if not str(authorization.get("reviewed_at") or ""):
        blockers.append("missing_reviewed_at")
    return blockers


def _status(
    errors: list[str],
    blockers: list[str],
    launch_audit: dict[str, Any],
    completion_gate: dict[str, Any],
) -> str:
    """Map detailed evidence into one operator-facing readiness state."""

    if errors:
        return "FAIL_PREFULL_READINESS"
    if blockers:
        return "PASS_PRE_FULL_READY_AUTHORIZATION_REQUIRED"
    if launch_audit.get("full_oof_execution_allowed") is True:
        return "READY_TO_RUN_FULL_OOF"
    if completion_gate.get("q2_claim_allowed") is True:
        return "Q2_CLAIM_ALLOWED_AFTER_POSTRUN"
    return "PASS_PREFULL_UNKNOWN_BLOCKER"


def _next_action(blockers: list[str], launch: dict[str, Any]) -> str:
    """Keep the next action explicit so the audit replaces repeated checking."""

    if blockers:
        return (
            "Human reviewer must authorize full_oof_authorization.json using "
            "the command template in full_oof_launch_packet.md."
        )
    if launch.get("full_oof_execution_allowed") is True:
        return "Run the audited full OOF command from full_oof_launch_packet.md."
    return "Inspect errors or rerun the execution gate after authorization."


def _require_command_tokens(
    errors: list[str],
    name: str,
    command: Any,
    tokens: tuple[str, ...],
) -> None:
    text = str(command or "")
    missing = [token for token in tokens if token not in text]
    if missing:
        errors.append(f"{name}_missing_tokens={missing}")


def _load_json(path: Path, errors: list[str], name: str) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing_{name}={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except Exception as exc:  # pragma: no cover - diagnostic fallback only.
        return {"commit": None, "dirty": None, "error": str(exc)}
    return {"commit": commit or None, "dirty": dirty}


def _require_true(errors: list[str], name: str, value: Any) -> None:
    if value is not True:
        errors.append(f"{name}_must_be_true={value}")


def _require_false(errors: list[str], name: str, value: Any) -> None:
    if value is not False:
        errors.append(f"{name}_must_be_false={value}")


def _require_equal(
    errors: list[str],
    name: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual != expected:
        errors.append(f"{name}_mismatch=actual:{actual},expected:{expected}")


if __name__ == "__main__":
    main()
