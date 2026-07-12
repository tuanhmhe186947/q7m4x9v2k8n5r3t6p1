from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev_tools import (  # noqa: E402
    write_classification_v2_full_oof_authorization_file as writer,
)


def main() -> None:
    """Audit the full-OOF authorization writer without launching training."""

    parser = argparse.ArgumentParser(
        description="Check classification_v2 full OOF authorization writer."
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
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_authorization_writer_audit.json"
        ),
    )
    args = parser.parse_args()

    preflight = json.loads(args.preflight_json.read_text(encoding="utf-8"))
    audit = check_authorization_writer(preflight)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_authorization_writer(preflight: dict[str, Any]) -> dict[str, Any]:
    """Exercise draft, partial, and fully-authorized writer behavior."""

    cases = [
        _draft_case(preflight),
        _reject_case(
            preflight,
            name="partial_authorization_rejected",
            kwargs={
                "authorize": True,
                "reviewer": "",
                "reviewed_at": "",
                "acknowledge_long_run": False,
                "acknowledge_no_q2_claim": False,
                "preflight_config_sha256": "",
                "git_commit": "",
            },
            required_tokens=[
                "missing_reviewer",
                "missing_acknowledge_long_run",
                "missing_acknowledge_no_q2_claim",
                "preflight_config_sha256_mismatch",
                "git_commit_mismatch",
            ],
        ),
        _success_case(preflight),
    ]
    errors = [
        error
        for case in cases
        for error in case["errors"]
    ]
    return {
        "schema_version": (
            "classification_v2_full_oof_authorization_writer_audit_v1"
        ),
        "preflight_config_sha256": preflight.get("config_sha256"),
        "preflight_git_commit": preflight.get("git_commit"),
        "case_count": len(cases),
        "cases": cases,
        "full_training_invoked": False,
        "errors": errors,
        "valid": not errors,
    }


def _draft_case(preflight: dict[str, Any]) -> dict[str, Any]:
    """Confirm a no-flag writer invocation remains non-executable."""

    payload = _build(preflight, authorize=False)
    errors: list[str] = []
    if payload.get("authorized") is not False:
        errors.append("draft_authorized_not_false")
    if payload.get("acknowledges_long_run") is not False:
        errors.append("draft_long_run_ack_not_false")
    if payload.get("acknowledges_no_q2_claim_until_verified") is not False:
        errors.append("draft_no_q2_claim_ack_not_false")
    if payload.get("preflight_config_sha256") != preflight.get("config_sha256"):
        errors.append("draft_preflight_hash_mismatch")
    if payload.get("git_commit") != preflight.get("git_commit"):
        errors.append("draft_git_commit_mismatch")
    return {
        "name": "draft_authorization_is_fail_closed",
        "authorized": payload.get("authorized"),
        "acknowledges_long_run": payload.get("acknowledges_long_run"),
        "acknowledges_no_q2_claim": payload.get(
            "acknowledges_no_q2_claim_until_verified"
        ),
        "errors": errors,
    }


def _reject_case(
    preflight: dict[str, Any],
    *,
    name: str,
    kwargs: dict[str, Any],
    required_tokens: list[str],
) -> dict[str, Any]:
    """Confirm unsafe authorization requests are rejected deterministically."""

    return _case(
        name,
        lambda: writer.build_authorization_file(preflight=preflight, **kwargs),
        required_tokens,
        expect_reject=True,
    )


def _success_case(preflight: dict[str, Any]) -> dict[str, Any]:
    """Confirm complete human-review inputs produce an executable auth file."""

    payload = _build(
        preflight,
        authorize=True,
        reviewer="classification_v2_authorization_writer_audit",
        acknowledge_long_run=True,
        acknowledge_no_q2_claim=True,
        preflight_config_sha256=str(preflight.get("config_sha256") or ""),
        git_commit=str(preflight.get("git_commit") or ""),
    )
    errors: list[str] = []
    if payload.get("authorized") is not True:
        errors.append("complete_authorization_not_true")
    if payload.get("acknowledges_long_run") is not True:
        errors.append("complete_long_run_ack_not_true")
    if payload.get("acknowledges_no_q2_claim_until_verified") is not True:
        errors.append("complete_no_q2_claim_ack_not_true")
    if payload.get("reviewer") != "classification_v2_authorization_writer_audit":
        errors.append("complete_reviewer_mismatch")
    return {
        "name": "complete_authorization_is_executable",
        "authorized": payload.get("authorized"),
        "acknowledges_long_run": payload.get("acknowledges_long_run"),
        "acknowledges_no_q2_claim": payload.get(
            "acknowledges_no_q2_claim_until_verified"
        ),
        "errors": errors,
    }


def _case(
    name: str,
    action: Callable[[], dict[str, Any]],
    required_tokens: list[str],
    *,
    expect_reject: bool,
) -> dict[str, Any]:
    """Run one writer behavior case and check deterministic outcome."""

    rejected = False
    message = ""
    errors: list[str] = []
    try:
        action()
    except SystemExit as exc:
        rejected = True
        message = str(exc)
    if rejected is not expect_reject:
        errors.append(f"writer_case_reject_state_mismatch={name}")
    missing_tokens = [
        token for token in required_tokens if token not in message
    ]
    if missing_tokens:
        errors.append(f"writer_case_missing_tokens={name}:{missing_tokens}")
    return {
        "name": name,
        "rejected": rejected,
        "required_tokens": required_tokens,
        "missing_tokens": missing_tokens,
        "message": message,
        "errors": errors,
    }


def _build(
    preflight: dict[str, Any],
    *,
    authorize: bool,
    reviewer: str = "",
    reviewed_at: str = "",
    acknowledge_long_run: bool = False,
    acknowledge_no_q2_claim: bool = False,
    preflight_config_sha256: str = "",
    git_commit: str = "",
) -> dict[str, Any]:
    """Call the writer with explicit defaults for readable test cases."""

    return writer.build_authorization_file(
        preflight=preflight,
        authorize=authorize,
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        acknowledge_long_run=acknowledge_long_run,
        acknowledge_no_q2_claim=acknowledge_no_q2_claim,
        preflight_config_sha256=preflight_config_sha256,
        git_commit=git_commit,
    )


if __name__ == "__main__":
    main()
