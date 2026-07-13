from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import write_classification_v2_full_oof_authorization_template as template_writer


def main() -> None:
    """Write the explicit full-OOF authorization JSON used by the runner."""

    parser = argparse.ArgumentParser(
        description=(
            "Create classification_v2 full OOF authorization JSON. Without "
            "--authorize it writes a fail-closed file."
        )
    )
    parser.add_argument(
        "--preflight-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/full_multimodal_oof_preflight.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/full_oof_authorization.json"),
    )
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--reviewed-at", default="")
    parser.add_argument("--acknowledge-long-run", action="store_true")
    parser.add_argument("--acknowledge-no-q2-claim", action="store_true")
    parser.add_argument("--preflight-config-sha256", default="")
    parser.add_argument("--git-commit", default="")
    args = parser.parse_args()

    preflight = json.loads(args.preflight_json.read_text(encoding="utf-8"))
    authorization = build_authorization_file(
        preflight=preflight,
        authorize=args.authorize,
        reviewer=args.reviewer,
        reviewed_at=args.reviewed_at,
        acknowledge_long_run=args.acknowledge_long_run,
        acknowledge_no_q2_claim=args.acknowledge_no_q2_claim,
        preflight_config_sha256=args.preflight_config_sha256,
        git_commit=args.git_commit,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(authorization, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(authorization, indent=2))


def build_authorization_file(
    *,
    preflight: dict[str, Any],
    authorize: bool,
    reviewer: str,
    reviewed_at: str,
    acknowledge_long_run: bool,
    acknowledge_no_q2_claim: bool,
    preflight_config_sha256: str,
    git_commit: str,
) -> dict[str, Any]:
    """Build authorization JSON while keeping accidental writes fail-closed."""

    authorization = template_writer.build_authorization_template(preflight)
    authorization["reviewer"] = reviewer.strip()
    authorization["reviewed_at"] = reviewed_at.strip() or _utc_now_iso()
    authorization["note"] = (
        "Explicit full OOF authorization file. Q2 claim remains locked until "
        "full OOF, calibration, confusion, and registry gates pass."
    )
    if not authorize:
        return authorization

    errors = _authorization_errors(
        preflight=preflight,
        reviewer=authorization["reviewer"],
        acknowledge_long_run=acknowledge_long_run,
        acknowledge_no_q2_claim=acknowledge_no_q2_claim,
        preflight_config_sha256=preflight_config_sha256,
        git_commit=git_commit,
    )
    if errors:
        raise SystemExit("refusing_to_authorize_full_oof=" + json.dumps(errors, ensure_ascii=True))

    authorization["authorized"] = True
    authorization["acknowledges_long_run"] = True
    authorization["acknowledges_no_q2_claim_until_verified"] = True
    return authorization


def _authorization_errors(
    *,
    preflight: dict[str, Any],
    reviewer: str,
    acknowledge_long_run: bool,
    acknowledge_no_q2_claim: bool,
    preflight_config_sha256: str,
    git_commit: str,
) -> list[str]:
    """Return missing human-review fields before enabling full OOF."""

    errors: list[str] = []
    expected_hash = str(preflight.get("config_sha256") or "")
    expected_commit = str(preflight.get("git_commit") or "")
    if preflight.get("valid") is not True or preflight.get("errors"):
        errors.append("preflight_not_valid_for_authorization")
    if preflight.get("lineage_binding_valid") is not True:
        errors.append("preflight_lineage_binding_not_valid")
    if preflight.get("lineage_training_authorized") is not True:
        errors.append("preflight_lineage_training_not_authorized")
    if not reviewer:
        errors.append("missing_reviewer")
    if not acknowledge_long_run:
        errors.append("missing_acknowledge_long_run")
    if not acknowledge_no_q2_claim:
        errors.append("missing_acknowledge_no_q2_claim")
    if preflight_config_sha256 != expected_hash:
        errors.append("preflight_config_sha256_mismatch")
    if git_commit != expected_commit:
        errors.append("git_commit_mismatch")
    template = template_writer.build_authorization_template(preflight)
    for field in (
        "snapshot_id",
        "snapshot_file_sha256",
        "lineage_audit_sha256",
        "ordered_window_id_sha256",
    ):
        if not template.get(field):
            errors.append(f"missing_preflight_binding={field}")
    return errors


def _utc_now_iso() -> str:
    """Return an auditable UTC timestamp for generated authorization drafts."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    main()
