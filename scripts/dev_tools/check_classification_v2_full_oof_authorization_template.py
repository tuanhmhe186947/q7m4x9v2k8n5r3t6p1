from __future__ import annotations

# ruff: noqa: I001

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.behavior_review_tools.classification_v2_run_full_multimodal_oof import (  # noqa: E402
    FULL_RUN_AUTHORIZATION_PURPOSE,
)
from scripts.dev_tools import (  # noqa: E402
    write_classification_v2_full_oof_authorization_template as template_writer,
)


def main() -> None:
    """Check that generated full OOF authorization templates fail closed."""

    parser = argparse.ArgumentParser(
        description="Check classification_v2 full OOF authorization template."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_authorization_template_audit.json"
        ),
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
        "--template-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_authorization_template.json"
        ),
    )
    args = parser.parse_args()

    errors: list[str] = []
    preflight = _load_json(args.preflight_json, errors)
    template = _load_json(args.template_json, errors)
    generated_template = template_writer.build_authorization_template(preflight)
    errors.extend(_template_errors(template, preflight))
    audit = {
        "schema_version": "classification_v2_full_oof_authorization_template_audit_v1",
        "preflight_json": str(args.preflight_json),
        "template_json": str(args.template_json),
        "preflight_valid": preflight.get("valid"),
        "preflight_config_sha256": preflight.get("config_sha256"),
        "preflight_git_commit": preflight.get("git_commit"),
        "template_authorized_default": template.get("authorized"),
        "template_acknowledges_long_run_default": template.get("acknowledges_long_run"),
        "template_acknowledges_no_claim_default": template.get(
            "acknowledges_no_q2_claim_until_verified"
        ),
        "template_purpose": template.get("purpose"),
        "template_binds_preflight_config_sha256": (
            template.get("preflight_config_sha256") == preflight.get("config_sha256")
        ),
        "template_binds_git_commit": (
            template.get("git_commit") == preflight.get("git_commit")
        ),
        "template_matches_writer_output": template == generated_template,
        "errors": errors,
        "valid": not errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if errors:
        raise SystemExit(1)


def _load_json(path: Path, errors: list[str]) -> dict[str, object]:
    if not path.exists():
        errors.append(f"missing_json={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _template_errors(
    template: dict[str, object],
    preflight: dict[str, object],
) -> list[str]:
    """Reject templates that look pre-approved or lose preflight identity."""

    errors: list[str] = []
    if preflight.get("valid") is not True or preflight.get("errors"):
        errors.append(f"template_preflight_not_valid={preflight.get('errors')}")
    if template.get("authorized") is not False:
        errors.append("template_must_default_authorized_false")
    if template.get("acknowledges_long_run") is not False:
        errors.append("template_must_default_long_run_ack_false")
    if template.get("acknowledges_no_q2_claim_until_verified") is not False:
        errors.append("template_must_default_no_claim_ack_false")
    if template.get("purpose") != FULL_RUN_AUTHORIZATION_PURPOSE:
        errors.append(f"template_purpose_mismatch={template.get('purpose')}")
    if not template.get("preflight_config_sha256"):
        errors.append("template_missing_preflight_config_sha256")
    elif template.get("preflight_config_sha256") != preflight.get("config_sha256"):
        errors.append(
            "template_preflight_config_hash_mismatch="
            f"{template.get('preflight_config_sha256')}!={preflight.get('config_sha256')}"
        )
    if not template.get("git_commit"):
        errors.append("template_missing_git_commit")
    elif template.get("git_commit") != preflight.get("git_commit"):
        errors.append(
            "template_git_commit_mismatch="
            f"{template.get('git_commit')}!={preflight.get('git_commit')}"
        )
    return errors


if __name__ == "__main__":
    main()
