"""Build a view-only subset of the immutable Behavior candidate publication."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.safe_non_interaction_view import (
    audit_safe_non_interaction_view,
    build_safe_non_interaction_view,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--producer-sha", required=True)
    parser.add_argument("--output-view", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    return parser.parse_args()


def _forbidden_live_ledger_path(path: Path) -> bool:
    normalized = str(path).replace("/", "\\").casefold()
    return (
        "human_review_workspace\\classification_v2\\" in normalized
        and "\\human_decisions\\behavior\\" in normalized
    )


def main() -> None:
    args = parse_args()
    for path in (
        args.candidate_manifest,
        args.output_view,
        args.output_audit,
    ):
        if _forbidden_live_ledger_path(path):
            raise SystemExit("live Behavior decision-ledger paths are forbidden")

    actual_hash = sha256_file(args.candidate_manifest)
    if actual_hash != args.expected_candidate_sha256:
        raise SystemExit(
            "candidate manifest hash mismatch: "
            f"expected={args.expected_candidate_sha256} actual={actual_hash}"
        )
    candidates = pd.read_csv(args.candidate_manifest, low_memory=False)
    result = build_safe_non_interaction_view(
        candidates,
        producer_sha=args.producer_sha,
        input_sha256=actual_hash,
    )
    args.output_view.parent.mkdir(parents=True, exist_ok=True)
    result.view.to_csv(args.output_view, index=False, lineterminator="\n")
    independent = audit_safe_non_interaction_view(
        candidates,
        pd.read_csv(args.output_view, low_memory=False),
        expected_candidate_sha256=args.expected_candidate_sha256,
        actual_candidate_sha256=actual_hash,
    )
    payload = {
        **result.audit,
        "output_hashes": {
            "safe_non_interaction_review_view_sha256": sha256_file(
                args.output_view
            )
        },
        "independent_checker": independent,
        "valid": bool(independent["valid"]),
    }
    write_json(args.output_audit, payload)
    if not independent["valid"]:
        raise SystemExit(
            "safe non-interaction view checker failed: "
            + ";".join(independent["errors"])
        )
    print(
        "SAFE_NON_INTERACTION_VIEW_VALID "
        f"rows={len(result.view)} excluded="
        f"{result.audit['excluded_interaction_affected_count']}"
    )


if __name__ == "__main__":
    main()
