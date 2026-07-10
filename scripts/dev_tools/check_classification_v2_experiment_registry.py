from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 experiment registry record.")
    parser.add_argument(
        "--record-json",
        type=Path,
        default=Path("outputs/classification_v2/experiment_registry/spatial_tcn_smoke_train_record.json"),
    )
    args = parser.parse_args()
    record = json.loads(args.record_json.read_text(encoding="utf-8"))
    errors: list[str] = []
    required = ["schema_version", "name", "created_at_utc", "git_commit", "git_dirty", "artifacts", "record_path"]
    missing = [key for key in required if key not in record]
    if missing:
        errors.append(f"missing_record_keys={missing}")
    for artifact in record.get("artifacts", []):
        if not artifact.get("exists"):
            errors.append(f"missing_artifact={artifact.get('path')}")
        if artifact.get("hash_status") == "ok" and not artifact.get("sha256"):
            errors.append(f"missing_sha256={artifact.get('path')}")
    result = {
        "record_json": str(args.record_json),
        "name": record.get("name"),
        "artifact_count": len(record.get("artifacts", [])),
        "git_commit": record.get("git_commit"),
        "git_dirty": record.get("git_dirty"),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
