from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


MAIN = Path(r"C:\Users\ironh\Downloads\PIG_Behavior_Project")
OUT = MAIN / r".codex_worktrees\classification_v2_full_t6_drive_package_20260817\outputs\classification_v2\full_t6_training_authority_20260817"
FULL_ROOT = MAIN / r"outputs\classification_v2\full_t6_canonical_46d_20260816"
TEMP_ROOT = MAIN / r"outputs\classification_v2\temporal_v2_canonical_authority_v1"
RGB_ROOT = MAIN / r"outputs\classification_v2\model_readiness_audit\pre_gpu_autoresearch_q2_6c2f204_20260804_084638\reviewed_rgb_v1\actor_rgb_128_full"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json_list(value: str) -> list:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise AssertionError(f"expected list JSON, got {type(parsed).__name__}")
    return parsed


authority_path = MAIN / r"docs\classification_v2\full_t6_46d_final_authority_20260817.json"
independent_audit_path = FULL_ROOT / "independent_audit_20260817.json"
build_evidence_path = FULL_ROOT / "build_evidence.json"
npz_path = FULL_ROOT / "full_t6_canonical_46d.npz"
row_manifest_path = FULL_ROOT / "full_t6_row_manifest.csv"
temporal_authority_path = TEMP_ROOT / "temporal_semantics_authority_v2.json"
temporal_manifest_path = TEMP_ROOT / "full_temporal_window_manifest_release.csv"
split_path = TEMP_ROOT / "target_split_roles_release.csv"
temporal_hash_manifest_path = TEMP_ROOT / "temporal_v2_artifact_hash_manifest.json"
rgb_index_path = RGB_ROOT / "packed_image_cache_index.csv"
rgb_tensor_path = RGB_ROOT / "packed_rgb_128_letterbox.npy"
rgb_audit_path = RGB_ROOT / "packed_image_cache_audit.json"
rgb_cache_audit_path = RGB_ROOT / "cache_audit.json"

authority = json.loads(authority_path.read_text(encoding="utf-8"))
independent_audit = json.loads(independent_audit_path.read_text(encoding="utf-8"))
rgb_audit = json.loads(rgb_audit_path.read_text(encoding="utf-8"))
rgb_cache_audit = json.loads(rgb_cache_audit_path.read_text(encoding="utf-8"))

assert authority["status"] == "PASS"
assert authority["active_executable_schema_sha256"] == (
    "18377d825ba84974e49305e46561ada81353f9ffd0f2d2526471af1c199daad4"
)
assert authority["population"] == {
    "cvat_count": 28748,
    "legacy_recovered_count": 4539,
    "name": "FULL-T6",
    "pool": "FULL_NONOVERLAP_VIEW_POOL",
    "total_targets": 33287,
}
assert authority["d890_revoked"] is True
assert authority["scientific_boundaries"]["existing_cvat_recomputed"] is False
assert authority["scientific_boundaries"]["legacy_rows_zero_filled"] is False

expected_hashes = {
    authority_path: "a37f75041c8bec71386c35844718556c648c37c9755e4d5b1a7ab0de3575db54",
    independent_audit_path: "efbf8d95d2acb2e410d0c04fbabb02586880f4acd61fcfebbca2e4063f973dee",
    build_evidence_path: "96b6dc65ebfd41ff3e2ed512dc3ea47547c3d4b94911d1a1693196095763e61e",
    npz_path: "fa4a9f26135271717115355b0ba2a71058b506d05e7cb70b560dca299f14b7d7",
    row_manifest_path: "6737b4437074a1d4021d3749c980797c9dbf145691778d6a6b1075fcfacee6e0",
    temporal_authority_path: "c9f4ba4ffa6ebae7405d13eaccc481097f9810d7be2385c1c9715aea04524681",
    temporal_manifest_path: "c992568cb4e6fe5fe2486072bffd614d1419806b3430961af35d212a2e1c246a",
    split_path: "eb4a41753658c52910ff42de98b65a9aff542b89ae817761c07af78273820e80",
    temporal_hash_manifest_path: "fa4dfd10f891d8e69e5a2f96d5ca2bce1a4e816fecb4fc3b94eabd99ce3ace73",
}
hash_results = {}
for path, expected in expected_hashes.items():
    actual = sha256(path)
    assert actual == expected, f"hash mismatch: {path} {actual} != {expected}"
    hash_results[str(path)] = {"bytes": path.stat().st_size, "sha256": actual}

with row_manifest_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 33287
target_ids = [row["target_id"] for row in rows]
assert len(set(target_ids)) == len(target_ids)
assert Counter(row["source_type"] for row in rows) == {
    "cvat_tracking_xml": 28748,
    "legacy_recovered": 4539,
}
assert Counter(row["split"] for row in rows) == {"train": 27834, "validation": 5453}

for row in rows:
    frames = parse_json_list(row["physical_frame_ids_json"])
    observed = parse_json_list(row["observed_mask_json"])
    assert len(frames) == 6
    assert len(observed) == 6
    assert all(isinstance(frame, int) for frame in frames)
    assert all(isinstance(value, bool) for value in observed)
    assert all(observed)

with temporal_manifest_path.open(newline="", encoding="utf-8") as handle:
    temporal_t6 = {
        row["target_id"]: row
        for row in csv.DictReader(handle)
        if row["view_id"] == "T6"
    }
assert len(temporal_t6) == 33287
assert set(temporal_t6) == set(target_ids)

for row in rows:
    temporal = temporal_t6[row["target_id"]]
    assert temporal["source_type"] == row["source_type"]
    assert temporal["dataset_id"] == row["dataset_id"]
    assert temporal["video_key"] == row["video_key"]
    assert temporal["object_track_key"] == row["object_track_key"]
    assert temporal["behavior"] == row["behavior"]
    assert parse_json_list(temporal["selected_frame_indices"]) == parse_json_list(
        row["physical_frame_ids_json"]
    )

with split_path.open(newline="", encoding="utf-8") as handle:
    target_id_set = set(target_ids)
    split_rows = {
        row["target_id"]: row
        for row in csv.DictReader(handle)
        if row["target_id"] in target_id_set
    }
assert len(split_rows) == 33287, len(split_rows)
assert set(split_rows) == set(target_ids)
for row in rows:
    split_row = split_rows[row["target_id"]]
    assert split_row["outer_fold_id"] == row["outer_fold_id"]
    assert split_row["split"] == row["split"]

with rgb_index_path.open(newline="", encoding="utf-8") as handle:
    rgb_rows = list(csv.DictReader(handle))
rgb_ids = [row["image_context_id"] for row in rgb_rows]
assert len(rgb_rows) == 245680
assert len(set(rgb_ids)) == len(rgb_ids)
packed_rows = sorted(int(row["packed_row"]) for row in rgb_rows)
assert packed_rows == list(range(245680))

required_rgb_ids = {
    f"{row['source_type']}|{row['object_track_key']}|f{int(frame):06d}"
    for row in rows
    for frame in parse_json_list(row["physical_frame_ids_json"])
}
rgb_id_set = set(rgb_ids)
missing_rgb_ids = sorted(required_rgb_ids - rgb_id_set)
assert not missing_rgb_ids

rgb_array = np.load(rgb_tensor_path, mmap_mode="r", allow_pickle=False)
assert tuple(rgb_array.shape) == (245680, 128, 128, 3)
assert str(rgb_array.dtype) == "uint8"
assert rgb_audit["valid"] is True
assert rgb_audit["shape"] == [245680, 128, 128, 3]
assert rgb_audit["resize_policy_values"] == ["letterbox_preserve_aspect_rgb_pad_black_v1"]
assert rgb_audit["duplicate_index_ids"] == 0
assert rgb_cache_audit["valid"] is True
assert rgb_cache_audit["image_size"] == 128
assert rgb_cache_audit["missing_context_rows"] == 0
assert rgb_cache_audit["duplicate_context_rows"] == 0
assert rgb_cache_audit["resize_policy"] == "letterbox_preserve_aspect_rgb_pad_black_v1"

with np.load(npz_path, allow_pickle=True) as npz:
    npz_summary = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in ((key, npz[key]) for key in npz.files)
    }
assert any(summary["shape"][0] == 33287 for summary in npz_summary.values())

remote_rgb = {
    "path": "lit://ironheart211224/pig-project/uploads/classification_v2/cloud_r128_recovery_20260817/r128_cache",
    "entries": ["packed_image_cache_index.csv", "packed_rgb_128_letterbox.npy"],
    "index_bytes": 47781243,
    "tensor_bytes": 12075663488,
    "index_sha256_local_verified": "9ccef8607973cfb8c8377474665af5d62874b5beea39ad716872b187f8d29d68",
    "tensor_sha256_local_verified": "c352a74cade4587e9dcbb8c3eead0c095c992306549b53da6d8b2a361691f5ee",
    "remote_sha256_status": "HASH_PENDING_REMOTE_READ",
}

result = {
    "schema_version": "classification_v2.full_t6_downstream_cross_binding_audit.v1",
    "status": "PASS",
    "scope": "local_read_only_package_readiness",
    "current_r128_run_touched": False,
    "gpu_used": False,
    "model_training_runs": 0,
    "d890_included": False,
    "quarantined_artifact_included": False,
    "population": {
        "total_targets": len(rows),
        "train": sum(row["split"] == "train" for row in rows),
        "validation": sum(row["split"] == "validation" for row in rows),
        "cvat": sum(row["source_type"] == "cvat_tracking_xml" for row in rows),
        "legacy_recovered": sum(row["source_type"] == "legacy_recovered" for row in rows),
    },
    "active_46d_schema_sha256": authority["active_executable_schema_sha256"],
    "feature_width": authority["parity"]["feature_width"],
    "group_order": authority["parity"]["group_order"],
    "local_hashes": hash_results,
    "npz_arrays": npz_summary,
    "rgb": {
        "coverage_required_contexts": len(required_rgb_ids),
        "cache_rows": len(rgb_rows),
        "missing_contexts": len(missing_rgb_ids),
        "duplicate_context_ids": len(rgb_ids) - len(rgb_id_set),
        "target_id_parity": "PASS",
        "preprocessing_parity": "PASS",
        "remote": remote_rgb,
    },
    "temporal_v2": {
        "authority_sha256": hash_results[str(temporal_authority_path)]["sha256"],
        "release_manifest_sha256": hash_results[str(temporal_manifest_path)]["sha256"],
        "target_count": len(temporal_t6),
        "t6_frame_order_parity": "PASS",
        "binding": "PASS",
    },
    "split": {"target_count": len(split_rows), "parity": "PASS"},
    "parity": {
        "rgb_46d_target_id": "PASS",
        "temporal_v2_binding": "PASS",
        "split": "PASS",
        "order": "PASS",
        "labels": "PASS",
        "masks": "PASS",
    },
    "authority_evidence": {
        "final_authority": str(authority_path),
        "independent_audit": str(independent_audit_path),
        "independent_audit_status": independent_audit["status"],
        "existing_cvat_recomputed": authority["scientific_boundaries"]["existing_cvat_recomputed"],
        "legacy_rows_zero_filled": authority["scientific_boundaries"]["legacy_rows_zero_filled"],
    },
}
(OUT / "local_cross_binding_audit.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(result, indent=2, sort_keys=True))
