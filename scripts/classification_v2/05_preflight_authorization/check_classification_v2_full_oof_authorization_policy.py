from __future__ import annotations

# ruff: noqa: I001

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pig_behavior.classification_v2.training.full_multimodal_oof import (  # noqa: E402
    FullMultimodalOofConfig,
    full_run_config_fingerprint,
)
from pig_behavior.classification_v2.training.full_run_contract import (  # noqa: E402
    FULL_RUN_AUTHORIZATION_PURPOSE,
    FULL_RUN_AUTHORIZATION_SCHEMA_VERSION,
    validate_full_run_authorization as _validate_full_run_authorization,
)


def main() -> None:
    """Check that full OOF cannot run without exact explicit authorization."""

    parser = argparse.ArgumentParser(
        description="Check classification_v2 full OOF authorization policy."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_authorization_policy_audit.json"
        ),
    )
    args = parser.parse_args()

    config = FullMultimodalOofConfig(run_mode="full")
    config_hash = full_run_config_fingerprint(config)
    preflight = {
        "config_sha256": config_hash,
        "git_commit": "commit-for-policy-check",
        "snapshot_id": "snapshot-for-policy-check",
        "snapshot_file_sha256": "snapshot-sha-for-policy-check",
        "lineage_audit_sha256": "lineage-sha-for-policy-check",
        "lineage_binding_audit": {
            "expected_ordered_window_id_sha256": (
                "ordered-window-sha-for-policy-check"
            )
        },
    }
    valid_authorization = _authorization(
        config_hash=config_hash,
        git_commit="commit-for-policy-check",
    )
    invalid_authorization = _authorization(
        config_hash="stale-config",
        git_commit="stale-commit",
    )
    invalid_authorization.update(
        {
            "authorized": False,
            "purpose": "wrong-purpose",
            "acknowledges_long_run": False,
            "acknowledges_no_q2_claim_until_verified": False,
        }
    )

    valid_errors = _validate_full_run_authorization(
        config,
        preflight,
        valid_authorization,
    )
    invalid_errors = _validate_full_run_authorization(
        config,
        preflight,
        invalid_authorization,
    )
    required_invalid_tokens = [
        "requires_authorized_true",
        "purpose_mismatch",
        "acknowledge_long_run",
        "acknowledge_no_q2_claim",
        "preflight_hash_mismatch",
        "git_commit_mismatch",
    ]
    missing_invalid_tokens = [
        token
        for token in required_invalid_tokens
        if not any(token in error for error in invalid_errors)
    ]
    errors: list[str] = []
    if valid_errors:
        errors.append(f"valid_authorization_rejected={valid_errors}")
    if missing_invalid_tokens:
        errors.append(f"invalid_authorization_not_rejected={missing_invalid_tokens}")

    audit = {
        "schema_version": "classification_v2_full_oof_authorization_policy_v1",
        "requires_authorization_json": True,
        "authorization_purpose": FULL_RUN_AUTHORIZATION_PURPOSE,
        "valid_authorization_errors": valid_errors,
        "invalid_authorization_errors": invalid_errors,
        "required_invalid_token_count": len(required_invalid_tokens),
        "missing_invalid_tokens": missing_invalid_tokens,
        "errors": errors,
        "valid": not errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if errors:
        raise SystemExit(1)


def _authorization(*, config_hash: str, git_commit: str) -> dict[str, object]:
    """Build the explicit approval payload expected by the full-run gate."""

    return {
        "schema_version": FULL_RUN_AUTHORIZATION_SCHEMA_VERSION,
        "authorized": True,
        "purpose": FULL_RUN_AUTHORIZATION_PURPOSE,
        "acknowledges_long_run": True,
        "acknowledges_no_q2_claim_until_verified": True,
        "preflight_config_sha256": config_hash,
        "git_commit": git_commit,
        "snapshot_id": "snapshot-for-policy-check",
        "snapshot_file_sha256": "snapshot-sha-for-policy-check",
        "lineage_audit_sha256": "lineage-sha-for-policy-check",
        "ordered_window_id_sha256": (
            "ordered-window-sha-for-policy-check"
        ),
        "reviewer": "authorization-policy-check",
        "reviewed_at": "2026-07-13T00:00:00+07:00",
    }


if __name__ == "__main__":
    main()
