from __future__ import annotations

# ruff: noqa: I001

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


def main() -> None:
    """Write a non-authorized template tied to one full OOF preflight artifact."""

    parser = argparse.ArgumentParser(
        description="Write a classification_v2 full OOF authorization template."
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
            "full_oof_authorization_template.json"
        ),
    )
    args = parser.parse_args()

    preflight = json.loads(args.preflight_json.read_text(encoding="utf-8"))
    template = build_authorization_template(preflight)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(template, indent=2), encoding="utf-8")
    print(json.dumps(template, indent=2))


def build_authorization_template(preflight: dict[str, Any]) -> dict[str, Any]:
    """Create a reviewable approval draft without silently authorizing full OOF."""

    return {
        "schema_version": "classification_v2_full_oof_authorization_v1",
        "authorized": False,
        "purpose": FULL_RUN_AUTHORIZATION_PURPOSE,
        "acknowledges_long_run": False,
        "acknowledges_no_q2_claim_until_verified": False,
        "preflight_config_sha256": preflight.get("config_sha256"),
        "git_commit": preflight.get("git_commit"),
        "reviewer": "",
        "reviewed_at": "",
        "note": (
            "Set authorized and both acknowledgement fields to true only after "
            "reviewing the matching clean preflight, runtime estimate, cache "
            "paths, and no-claim boundary."
        ),
    }


if __name__ == "__main__":
    main()
