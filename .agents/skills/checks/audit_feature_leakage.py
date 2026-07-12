"""Fail closed when a declared model-X whitelist contains leakage fields."""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

from _common import finish, load_json

FORBIDDEN = (
    "*behavior*",
    "*label*",
    "manual_*",
    "review_*",
    "*corrected*",
    "target_*",
    "*policy*",
    "*_path",
    "*fold*",
    "source_type",
    "video_key",
    "dataset_id",
    "pig_id",
    "track_id",
    "review_unit_id",
    "window_id",
    "temporal_unit_key",
    "frame_uid",
)


def audit(path: Path) -> dict[str, object]:
    """Validate explicit features against project leakage patterns."""
    payload = load_json(path)
    features = payload if isinstance(payload, list) else payload.get("features", [])
    features = [str(value) for value in features]
    forbidden = sorted(
        feature
        for feature in features
        if any(fnmatch.fnmatch(feature.lower(), pattern) for pattern in FORBIDDEN)
    )
    errors: list[str] = []
    if not features:
        errors.append("empty_feature_whitelist")
    if len(features) != len(set(features)):
        errors.append("duplicate_feature_names")
    if forbidden:
        errors.append(f"forbidden_feature_count={len(forbidden)}")
    return {
        "check": "feature_leakage",
        "feature_count": len(features),
        "selection_policy": "explicit_only",
        "forbidden_features": forbidden,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whitelist-json", type=Path, required=True)
    return finish(audit(parser.parse_args().whitelist_json))


if __name__ == "__main__":
    raise SystemExit(main())
