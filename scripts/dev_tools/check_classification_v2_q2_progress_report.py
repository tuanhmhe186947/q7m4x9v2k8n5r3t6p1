from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def main() -> None:
    """Validate q2_progress_report as the human-facing pre-full dashboard."""

    parser = argparse.ArgumentParser(
        description="Check classification_v2 Q2 progress report evidence."
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_progress_report.json"),
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_progress_report.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "q2_progress_report_audit.json"
        ),
    )
    args = parser.parse_args()

    audit = check_q2_progress_report(
        report_json=args.report_json,
        report_md=args.report_md,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_q2_progress_report(*, report_json: Path, report_md: Path) -> dict[str, Any]:
    """Return a fail-closed audit for pre-full Q2 progress reporting."""

    errors: list[str] = []
    report = _load_json(report_json, errors)
    markdown_text = _load_text(report_md, errors)
    evidence = report.get("evidence") or {}
    launch = evidence.get("full_oof_launch_packet") or {}
    postrun = evidence.get("full_oof_postrun_registration_packet") or {}
    freshness = evidence.get("full_oof_preflight_freshness") or {}
    authorization = evidence.get("full_oof_authorization_file") or {}
    completion = evidence.get("full_oof_completion_gate") or {}
    git_state = _git_state()

    _require_equal(
        errors,
        "schema_version",
        report.get("schema_version"),
        "classification_v2_q2_progress_report_v1",
    )
    _require_equal(errors, "overall_status", report.get("overall_status"), "PASS_PARTIAL_ROADMAP")
    _require_true(errors, "all_gates_passed", _all_gates_passed(report))
    _require_false(
        errors,
        "full_oof_execution_allowed",
        launch.get("full_oof_execution_allowed"),
    )
    _require_true(errors, "authorization_required", launch.get("authorization_required"))
    _require_false(errors, "authorization_authorized", authorization.get("authorized"))
    _require_false(errors, "completion_q2_claim_allowed", completion.get("q2_claim_allowed"))
    _require_true(errors, "completion_fail_closed", completion.get("fail_closed"))
    _require_true(errors, "preflight_fresh", freshness.get("preflight_fresh"))
    _require_false(errors, "git_dirty", freshness.get("git_dirty"))
    _require_equal(
        errors,
        "current_git_commit",
        freshness.get("current_git_commit"),
        git_state.get("commit"),
    )
    _require_false(errors, "actual_git_dirty", git_state.get("dirty"))
    _require_markdown_runbook_evidence(errors, "launch", launch)
    _require_markdown_runbook_evidence(errors, "postrun", postrun)
    _require_markdown_lines(markdown_text, errors)

    return {
        "schema_version": "classification_v2_q2_progress_report_audit_v1",
        "valid": not errors,
        "errors": errors,
        "report_json": str(report_json),
        "report_md": str(report_md),
        "overall_status": report.get("overall_status"),
        "gate_count": len(report.get("gates") or []),
        "all_gates_passed": _all_gates_passed(report),
        "current_git_commit": freshness.get("current_git_commit"),
        "git_dirty": freshness.get("git_dirty"),
        "actual_git_dirty": git_state.get("dirty"),
        "full_oof_execution_allowed": launch.get("full_oof_execution_allowed"),
        "authorization_required": launch.get("authorization_required"),
        "authorization_authorized": authorization.get("authorized"),
        "q2_claim_allowed": completion.get("q2_claim_allowed"),
        "launch_markdown_runbook_valid": launch.get("markdown_runbook_valid"),
        "launch_markdown_overlong_command_line_count": launch.get(
            "markdown_overlong_command_line_count"
        ),
        "postrun_markdown_runbook_valid": postrun.get("markdown_runbook_valid"),
        "postrun_markdown_overlong_command_line_count": postrun.get(
            "markdown_overlong_command_line_count"
        ),
    }


def _require_markdown_runbook_evidence(
    errors: list[str],
    name: str,
    evidence: dict[str, Any],
) -> None:
    """Require runbook evidence to be visible in q2_progress_report."""

    _require_true(
        errors,
        f"{name}_markdown_runbook_valid",
        evidence.get("markdown_runbook_valid"),
    )
    _require_equal(
        errors,
        f"{name}_markdown_overlong_command_line_count",
        evidence.get("markdown_overlong_command_line_count"),
        0,
    )
    if not evidence.get("markdown_bat_block_count"):
        errors.append(f"{name}_markdown_bat_block_count_missing")
    if not evidence.get("markdown_wrapped_command_line_count"):
        errors.append(f"{name}_markdown_wrapped_command_line_count_missing")


def _require_markdown_lines(text: str, errors: list[str]) -> None:
    """Check the rendered Markdown exposes fail-closed and runbook state."""

    required_lines = (
        "Status: **PASS_PARTIAL_ROADMAP**",
        "- Full OOF execution allowed: `False`",
        "- Authorization required: `True`",
        "- Launch runbook Markdown valid: `True`",
        "- Launch runbook overlong command lines: `0`",
        "- Postrun runbook Markdown valid: `True`",
        "- Postrun runbook overlong command lines: `0`",
        "- Authorized: `False`",
    )
    for line in required_lines:
        if line not in text:
            errors.append(f"q2_progress_markdown_missing_line={line}")


def _all_gates_passed(report: dict[str, Any]) -> bool:
    gates = report.get("gates") or []
    return bool(gates) and all(gate.get("passed") is True for gate in gates)


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


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing_report_json={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_text(path: Path, errors: list[str]) -> str:
    if not path.exists():
        errors.append(f"missing_report_md={path}")
        return ""
    return path.read_text(encoding="utf-8")


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
    except Exception:
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": dirty}


if __name__ == "__main__":
    main()
