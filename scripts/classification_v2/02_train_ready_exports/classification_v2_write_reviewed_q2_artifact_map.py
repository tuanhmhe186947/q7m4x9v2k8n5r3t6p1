"""Write a reviewed-Q2 artifact map with isolated human and agent roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.contracts.reviewed_q2_artifact_map import (
    ReviewedQ2ArtifactMapError,
    build_reviewed_q2_artifact_map,
    write_reviewed_q2_artifact_map,
)


def parse_args() -> argparse.Namespace:
    """Require both IDs and an explicit agent-owned destination."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-review-run-id", required=True)
    parser.add_argument("--agent-audit-run-id", required=True)
    parser.add_argument(
        "--template-json",
        type=Path,
        default=Path(
            "configs/classification_v2/"
            "reviewed_q2_data_contract_template_v1.json"
        ),
    )
    parser.add_argument(
        "--layout-json",
        type=Path,
        default=Path(
            "configs/classification_v2/"
            "reviewed_q2_artifact_layout_v1.json"
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate in memory before writing only below the agent audit root."""

    args = parse_args()
    try:
        build = build_reviewed_q2_artifact_map(
            args.template_json,
            args.layout_json,
            human_review_run_id=args.human_review_run_id,
            agent_audit_run_id=args.agent_audit_run_id,
            output_path=args.output_json,
            project_root=args.project_root,
        )
        audit = write_reviewed_q2_artifact_map(
            build,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except (
        FileNotFoundError,
        ValueError,
        ReviewedQ2ArtifactMapError,
    ) as exc:
        errors = list(getattr(exc, "errors", (str(exc),)))
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "valid": False,
                    "errors": errors,
                    "artifact_written": False,
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        raise SystemExit(2) from exc
    print(json.dumps(audit, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
