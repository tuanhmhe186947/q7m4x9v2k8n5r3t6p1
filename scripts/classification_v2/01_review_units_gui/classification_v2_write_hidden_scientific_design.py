"""Write a pre-review Hidden scientific design from audited artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.hidden_review_science import (
    build_hidden_scientific_design,
    load_hidden_scientific_policy,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--template-audit-json", type=Path, required=True)
    parser.add_argument(
        "--scientific-policy-json",
        type=Path,
        default=Path(
            "configs/classification_v2/"
            "hidden_review_scientific_policy_v1.json"
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Allow insufficient support but mark the design non-authorizing.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.manifest_csv,
        args.template_audit_json,
        args.scientific_policy_json,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_json.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {args.output_json}. Use --overwrite explicitly."
        )

    manifest = pd.read_csv(args.manifest_csv, low_memory=False)
    template_audit = json.loads(
        args.template_audit_json.read_text(encoding="utf-8")
    )
    if template_audit.get("errors"):
        raise ValueError(
            "Template audit must pass before design declaration: "
            f"{template_audit['errors']}"
        )
    selection_contract = template_audit.get("selection_contract", {})
    _, policy_payload, policy_sha256 = load_hidden_scientific_policy(
        args.scientific_policy_json
    )
    design = build_hidden_scientific_design(
        manifest,
        manifest_sha256=sha256_file(args.manifest_csv),
        policy_payload=policy_payload,
        policy_sha256=policy_sha256,
        selection_contract=selection_contract,
        require_final_support=not args.smoke,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(design, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "[PASS] Hidden scientific design "
        f"scope={design['design_scope']} "
        f"support={design['planned_support_meets_final_gate']} "
        f"output={args.output_json}"
    )


if __name__ == "__main__":
    main()
