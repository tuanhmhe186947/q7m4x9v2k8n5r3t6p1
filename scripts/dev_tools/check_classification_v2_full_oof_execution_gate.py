from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.behavior_review_tools.classification_v2_run_full_multimodal_oof import (  # noqa: E402
    FULL_RUN_AUTHORIZATION_PURPOSE,
    FULL_RUN_AUTHORIZATION_SCHEMA_VERSION,
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
    preflight = _load_json_if_exists(preflight_json)
    with tempfile.TemporaryDirectory() as tmp_dir:
        missing_reviewer_json = _write_authorization_missing_reviewer(
            tmp_dir=Path(tmp_dir),
            preflight=preflight,
        )
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
            _case(
                "authorized_booleans_missing_reviewer_rejected",
                lambda: _validate_full_execution_confirmation(
                    config,
                    preflight_json,
                    missing_reviewer_json,
                    confirmed=True,
                ),
                [
                    "full_run_authorization_requires_reviewer",
                    "full_run_authorization_requires_reviewed_at",
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


def _load_json_if_exists(path: Path) -> dict[str, object]:
    """Load optional JSON evidence while keeping missing-file cases testable."""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_authorization_missing_reviewer(
    *,
    tmp_dir: Path,
    preflight: dict[str, object],
) -> Path:
    """Create a temporary near-valid authorization missing human identity."""

    authorization = {
        "schema_version": FULL_RUN_AUTHORIZATION_SCHEMA_VERSION,
        "authorized": True,
        "purpose": FULL_RUN_AUTHORIZATION_PURPOSE,
        "acknowledges_long_run": True,
        "acknowledges_no_q2_claim_until_verified": True,
        "reviewer": "",
        "reviewed_at": "",
        "preflight_config_sha256": preflight.get("config_sha256"),
        "git_commit": preflight.get("git_commit"),
    }
    path = tmp_dir / "authorized_missing_reviewer.json"
    path.write_text(json.dumps(authorization, indent=2), encoding="utf-8")
    return path


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
