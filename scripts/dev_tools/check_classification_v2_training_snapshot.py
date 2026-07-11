from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.contracts.training_snapshot import check_training_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check current classification_v2 artifacts against a frozen snapshot.")
    parser.add_argument("--snapshot-json", type=Path, default=None)
    parser.add_argument(
        "--contract-json",
        type=Path,
        default=None,
        help="Override the contract path recorded in the snapshot. Leave unset for lineage-safe checks.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _latest_snapshot(root: Path) -> Path:
    snapshots = sorted((root / "outputs/classification_v2/training_snapshots").glob("c2v2_*.json"))
    if not snapshots:
        raise FileNotFoundError("No training snapshot found under outputs/classification_v2/training_snapshots")
    # Snapshot IDs are content hashes, not chronological names.
    return max(snapshots, key=lambda path: path.stat().st_mtime)


def main() -> None:
    args = parse_args()
    snapshot_path = args.snapshot_json or _latest_snapshot(Path("."))
    result = check_training_snapshot(snapshot_path, contract_path=args.contract_json)
    summary = {
        k: result[k]
        for k in ["snapshot_path", "expected_snapshot_id", "current_snapshot_id", "valid", "errors", "warnings"]
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
