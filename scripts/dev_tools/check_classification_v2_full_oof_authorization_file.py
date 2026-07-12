from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.behavior_review_tools.classification_v2_run_full_multimodal_oof import (  # noqa: E402
    FULL_RUN_AUTHORIZATION_PURPOSE,
)

SCHEMA_VERSION = "classification_v2_full_oof_authorization_file_audit_v1"


def main() -> None:
    """Audit the explicit full-OOF authorization file before launch."""

    parser = argparse.ArgumentParser(
        description="Check classification_v2 full OOF authorization JSON."
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
        "--authorization-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_authorization.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_authorization_file_audit.json"
        ),
    )
    args = parser.parse_args()

    audit = check_authorization_file(
        preflight_json=args.preflight_json,
        authorization_json=args.authorization_json,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_authorization_file(
    *,
    preflight_json: Path,
    authorization_json: Path,
) -> dict[str, Any]:
    """Return a fail-closed audit of the human authorization file."""

    errors: list[str] = []
    warnings: list[str] = []
    blocking_reasons: list[str] = []
    preflight = _load_json(preflight_json, errors, "preflight")
    authorization = _load_json(authorization_json, errors, "authorization")
    if errors:
        return _audit(
            preflight_json=preflight_json,
            authorization_json=authorization_json,
            preflight=preflight,
            authorization=authorization,
            errors=errors,
            warnings=warnings,
            blocking_reasons=blocking_reasons,
        )

    errors.extend(_binding_errors(preflight, authorization))
    blocking_reasons.extend(_authorization_blockers(authorization))
    return _audit(
        preflight_json=preflight_json,
        authorization_json=authorization_json,
        preflight=preflight,
        authorization=authorization,
        errors=errors,
        warnings=warnings,
        blocking_reasons=blocking_reasons,
    )


def _audit(
    *,
    preflight_json: Path,
    authorization_json: Path,
    preflight: dict[str, Any],
    authorization: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    blocking_reasons: list[str],
) -> dict[str, Any]:
    """Build a stable, reader-facing authorization audit payload."""

    execution_allowed = not errors and not blocking_reasons
    return {
        "schema_version": SCHEMA_VERSION,
        "preflight_json": str(preflight_json),
        "authorization_json": str(authorization_json),
        "preflight_valid": preflight.get("valid"),
        "preflight_config_sha256": preflight.get("config_sha256"),
        "preflight_git_commit": preflight.get("git_commit"),
        "authorization_exists": authorization_json.exists(),
        "authorization_schema_version": authorization.get("schema_version"),
        "authorization_purpose": authorization.get("purpose"),
        "authorized": authorization.get("authorized"),
        "acknowledges_long_run": authorization.get("acknowledges_long_run"),
        "acknowledges_no_q2_claim": authorization.get(
            "acknowledges_no_q2_claim_until_verified"
        ),
        "reviewer_present": bool(str(authorization.get("reviewer") or "")),
        "reviewed_at_present": bool(str(authorization.get("reviewed_at") or "")),
        "binds_preflight_config_sha256": (
            authorization.get("preflight_config_sha256")
            == preflight.get("config_sha256")
        ),
        "binds_git_commit": authorization.get("git_commit")
        == preflight.get("git_commit"),
        "full_oof_execution_allowed": execution_allowed,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }


def _binding_errors(
    preflight: dict[str, Any],
    authorization: dict[str, Any],
) -> list[str]:
    """Validate fields that must match even for a fail-closed auth draft."""

    errors: list[str] = []
    if preflight.get("valid") is not True:
        errors.append(f"preflight_invalid={preflight.get('errors')}")
    if authorization.get("schema_version") != (
        "classification_v2_full_oof_authorization_v1"
    ):
        errors.append("authorization_schema_version_invalid")
    if authorization.get("purpose") != FULL_RUN_AUTHORIZATION_PURPOSE:
        errors.append("authorization_purpose_invalid")
    if authorization.get("preflight_config_sha256") != preflight.get(
        "config_sha256"
    ):
        errors.append("authorization_preflight_config_sha256_mismatch")
    if authorization.get("git_commit") != preflight.get("git_commit"):
        errors.append("authorization_git_commit_mismatch")
    return errors


def _authorization_blockers(authorization: dict[str, Any]) -> list[str]:
    """List human-review fields missing before full OOF can run."""

    blockers: list[str] = []
    if authorization.get("authorized") is not True:
        blockers.append("authorization_requires_authorized_true")
    if authorization.get("acknowledges_long_run") is not True:
        blockers.append("authorization_requires_long_run_ack")
    if authorization.get("acknowledges_no_q2_claim_until_verified") is not True:
        blockers.append("authorization_requires_no_q2_claim_ack")
    if not str(authorization.get("reviewer") or ""):
        blockers.append("authorization_requires_reviewer")
    if not str(authorization.get("reviewed_at") or ""):
        blockers.append("authorization_requires_reviewed_at")
    return blockers


def _load_json(
    path: Path,
    errors: list[str],
    label: str,
) -> dict[str, Any]:
    """Load one JSON file and report a labeled error if it is missing."""

    if not path.exists():
        errors.append(f"missing_{label}_json={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
