from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(
    "/home/zeus/c2v2_overnight_20260812/"
    "bootstrap_87b32d04fe5d1f605fdd6b8dfe95f28a11666ff1/code"
)
INPUT_ROOT = Path("/teamspace/studios/this_studio/pig_e0_r3/inputs")
MANIFEST = Path(
    "/home/zeus/c2v2_overnight_20260812/source_media/legacy_crops/"
    "5946b371c0e377b18d419af22acc7aef921cbb1c099eb646c4cf4d3872ca4d7a/"
    "legacy_member_manifest.tsv"
)

sys.path.insert(0, str(CODE_ROOT / "src"))
from pig_behavior.classification_v2.training.legacy_media_resolution import (  # noqa: E402
    attach_canonical_legacy_media_paths,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main(output: Path) -> None:
    frame_columns = ["image_context_id", "source_type", "resolved_media_path"]
    window_columns = [
        "window_id", "source_type", "video_key", "object_track_key",
        "window_start_frame", "image_context_id_sequence",
    ]
    frames = pd.read_csv(
        INPUT_ROOT / "reviewed_rgb_v1/image_context_v2/image_frame_context_manifest.csv",
        usecols=frame_columns,
    )
    windows = pd.read_csv(
        INPUT_ROOT / "reviewed_rgb_v1/image_context_v2/image_window_context_manifest.csv",
        usecols=window_columns,
    )
    windows = windows.loc[windows["source_type"].astype(str).eq("legacy_recovered")]
    windows = windows.sort_values(
        ["video_key", "object_track_key", "window_start_frame", "window_id"],
        kind="mergesort",
    ).head(16).copy()
    required_ids = {
        value
        for sequence in windows["image_context_id_sequence"].astype(str)
        for value in sequence.split(";;")
        if value
    }
    frames = attach_canonical_legacy_media_paths(
        frames.loc[frames["image_context_id"].astype(str).isin(required_ids)].copy()
    )
    members = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        relative, size, sha256 = line.split("\t")
        members[relative] = {"size": int(size), "sha256": sha256}

    records = []
    for window in windows.itertuples(index=False):
        for context_id in str(window.image_context_id_sequence).split(";;"):
            frame = frames.loc[frames["image_context_id"].astype(str).eq(context_id)]
            if len(frame) != 1:
                raise RuntimeError(f"nonunique_context={context_id}:{len(frame)}")
            relative = str(frame.iloc[0]["resolved_media_path"])
            authority = members.get(relative)
            if authority is None:
                raise RuntimeError(f"unregistered_member={relative}")
            runtime = INPUT_ROOT / relative
            exists = runtime.is_file()
            actual_sha = digest(runtime) if exists else None
            records.append({
                "window_id": str(window.window_id),
                "image_context_id": context_id,
                "canonical_member_relative_path": relative,
                "expected_sha256": authority["sha256"],
                "runtime_path": str(runtime),
                "runtime_exists": exists,
                "runtime_size": runtime.stat().st_size if exists else None,
                "hash_match": actual_sha == authority["sha256"] if exists else False,
            })
    missing = [record for record in records if not record["runtime_exists"]]
    invalid = [record for record in records if record["runtime_exists"] and not record["hash_match"]]
    failed_windows = sorted({record["window_id"] for record in missing + invalid})
    payload = {
        "prior_runtime_verification_valid": False,
        "audit_mode": "READ_ONLY_RUNTIME_FILESYSTEM",
        "audit_scope": "EXACT_EXISTING_LEGACY_R128_16_WINDOW_COHORT",
        "cohort_window_count": len(windows),
        "expected_member_count": len(records),
        "runtime_present_count": sum(record["runtime_exists"] for record in records),
        "runtime_missing_count": len(missing),
        "runtime_hash_match_count": sum(record["hash_match"] for record in records),
        "runtime_hash_mismatch_count": len(invalid),
        "failed_window_ids": failed_windows,
        "missing_member_records": missing,
        "hash_mismatch_records": invalid,
        "records": records,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key != "records"}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
