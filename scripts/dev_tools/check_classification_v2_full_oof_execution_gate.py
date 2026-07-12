from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.behavior_review_tools.classification_v2_run_full_multimodal_oof import (  # noqa: E402
    _validate_full_execution_confirmation,
)
from scripts.dev_tools.check_classification_v2_full_runner_defaults import (  # noqa: E402
    _full_runner_default_config,
)


def main() -> None:
    """Audit that the real full-run gate refuses to start without approval."""

    parser = argparse.ArgumentParser(
        description="Check classification_v2 full OOF execution gate."
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
        "--authorization-template-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_authorization_template.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_execution_gate_audit.json"
        ),
    )
    args = parser.parse_args()

    audit = check_execution_gate(
        preflight_json=args.preflight_json,
        authorization_template_json=args.authorization_template_json,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_execution_gate(
    *,
    preflight_json: Path,
    authorization_template_json: Path,
) -> dict[str, object]:
    """Exercise runner gate failures without invoking model training."""

    config = _full_runner_default_config()
    cases = [
        _case(
            "missing_confirm_flag",
            lambda: _validate_full_execution_confirmation(
                config,
                preflight_json,
                authorization_template_json,
                confirmed=False,
            ),
            ["--confirm-full-run"],
        ),
        _case(
            "missing_authorization_json",
            lambda: _validate_full_execution_confirmation(
                config,
                preflight_json,
                None,
                confirmed=True,
            ),
            ["--authorization-json"],
        ),
        _case(
            "unauthorized_template_rejected",
            lambda: _validate_full_execution_confirmation(
                config,
                preflight_json,
                authorization_template_json,
                confirmed=True,
            ),
            [
                "full_run_authorization_requires_authorized_true",
                "full_run_authorization_must_acknowledge_long_run",
                "full_run_authorization_must_acknowledge_no_q2_claim",
            ],
        ),
    ]
    errors = [
        error
        for case in cases
        for error in case["errors"]
    ]
    return {
        "schema_version": "classification_v2_full_oof_execution_gate_v1",
        "preflight_json": str(preflight_json),
        "authorization_template_json": str(authorization_template_json),
        "case_count": len(cases),
        "cases": cases,
        "full_training_invoked": False,
        "errors": errors,
        "valid": not errors,
    }


def _case(
    name: str,
    action: Callable[[], None],
    required_tokens: list[str],
) -> dict[str, object]:
    """Run one gate case and require deterministic rejection tokens."""

    errors: list[str] = []
    rejected = False
    message = ""
    try:
        action()
    except ValueError as exc:
        rejected = True
        message = str(exc)
    if not rejected:
        errors.append(f"gate_case_not_rejected={name}")
    missing_tokens = [
        token for token in required_tokens if token not in message
    ]
    if missing_tokens:
        errors.append(f"gate_case_missing_tokens={name}:{missing_tokens}")
    return {
        "name": name,
        "rejected": rejected,
        "required_tokens": required_tokens,
        "missing_tokens": missing_tokens,
        "message": message,
        "errors": errors,
    }


if __name__ == "__main__":
    main()
