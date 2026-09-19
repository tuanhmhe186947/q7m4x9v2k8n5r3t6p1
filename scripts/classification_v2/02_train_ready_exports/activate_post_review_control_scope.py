"""Create a GUI-ready view of a frozen post-review control sample."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.post_review_residual_discovery import (
    activate_post_review_scope_for_gui,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-scope-csv", type=Path, required=True)
    parser.add_argument("--expected-control-scope-sha256", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_csv.exists():
        raise FileExistsError(f"output already exists: {args.output_csv}")
    actual_hash = sha256_file(args.control_scope_csv)
    if actual_hash.casefold() != args.expected_control_scope_sha256.casefold():
        raise ValueError(
            "control scope hash mismatch: "
            f"expected={args.expected_control_scope_sha256} actual={actual_hash}"
        )
    scope = pd.read_csv(
        args.control_scope_csv,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    view = activate_post_review_scope_for_gui(
        scope,
        cohort="POST_REVIEW_RESIDUAL_CONTROL",
        reason_code="INDEPENDENT_RESIDUAL_CONTROL",
    )
    view["review_item_id"] = [
        f"post_review_control_{index:07d}"
        for index in range(1, len(view) + 1)
    ]
    view["consistency_review_order"] = range(1, len(view) + 1)
    view["original_behavior"] = view["behavior_label"]
    view["final_scope_component"] = "POST_REVIEW_RESIDUAL_CONTROL"
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    view.to_csv(args.output_csv, index=False)
    print(f"PASS: activated {len(view)} post-review residual controls")
    print(args.output_csv.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
