"""Build one sequential authority from completed Behavior-review layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.review.composite_review_authority import (
    ReviewLayer,
    compose_behavior_review_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-universe-csv", type=Path, required=True)
    parser.add_argument("--layer-name", action="append", required=True)
    parser.add_argument(
        "--layer-decisions-csv",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--layer-quality-csv",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-layer-decisions-sha256",
        action="append",
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = list(args.layer_name)
    decision_paths = list(args.layer_decisions_csv)
    quality_paths = list(args.layer_quality_csv)
    expected_hashes = list(args.expected_layer_decisions_sha256)
    lengths = {len(names), len(decision_paths), len(quality_paths), len(expected_hashes)}
    if lengths != {len(names)} or not names:
        raise ValueError("review layer arguments must have identical nonzero lengths")
    if len(set(names)) != len(names):
        raise ValueError("review layer names must be unique")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")

    layers: list[ReviewLayer] = []
    input_layers: list[dict[str, Any]] = []
    for name, decisions_path, quality_path, expected_hash in zip(
        names,
        decision_paths,
        quality_paths,
        expected_hashes,
        strict=True,
    ):
        actual_hash = sha256_file(decisions_path)
        if actual_hash.casefold() != expected_hash.casefold():
            raise ValueError(
                f"decision hash mismatch for {name}: "
                f"expected={expected_hash} actual={actual_hash}"
            )
        layers.append(
            ReviewLayer(
                name=name,
                decisions=pd.read_csv(
                    decisions_path,
                    dtype=str,
                    keep_default_na=False,
                ),
                quality=pd.read_csv(
                    quality_path,
                    dtype=str,
                    keep_default_na=False,
                ),
            )
        )
        input_layers.append(
            {
                "name": name,
                "decisions": path_record(decisions_path),
                "quality": path_record(quality_path),
            }
        )

    source_units = pd.read_csv(
        args.source_universe_csv,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    result = compose_behavior_review_layers(source_units, layers)
    args.output_dir.mkdir(parents=True)
    output_paths = {
        "scope": args.output_dir / "composite_review_scope.csv",
        "decisions": args.output_dir / "composite_behavior_decisions.csv",
        "quality": args.output_dir / "composite_behavior_quality.csv",
        "lineage": args.output_dir / "composite_decision_lineage.csv",
    }
    for name, path in output_paths.items():
        result[name].to_csv(path, index=False)

    audit_path = args.output_dir / "composite_review_audit.json"
    write_json(audit_path, result["audit"])
    manifest = {
        "schema_version": "classification_v2.composite_behavior_review.v1",
        "status": "FROZEN_SEQUENTIAL_REVIEW_AUTHORITY",
        "code_sha": git_head(),
        "source_universe": path_record(args.source_universe_csv),
        "layers_in_application_order": input_layers,
        "outputs": {
            name: path_record(path) for name, path in output_paths.items()
        },
        "audit": path_record(audit_path),
        "automatic_label_change_outside_reviewed_keys": False,
        "review_metadata_entering_model_x": False,
        "selected_skills": [
            "agent-architecture-audit",
            "dataset-contract-leakage-guard",
        ],
    }
    manifest_path = args.output_dir / "composite_review_manifest.json"
    write_json(manifest_path, manifest)
    inventory = {
        "schema_version": "classification_v2.artifact_inventory.v1",
        "artifacts": [
            path_record(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file() and path.name != "artifact_inventory.json"
        ],
    }
    write_json(args.output_dir / "artifact_inventory.json", inventory)
    print("PASS: sequential composite Behavior authority written")
    print(args.output_dir.resolve())


def path_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


if __name__ == "__main__":
    main()
