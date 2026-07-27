from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
from PIL import Image

from pig_behavior.classification_v2.review.gui_readiness import (
    audit_behavior_gui_readiness,
)


def _load_gui_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "classification_v2"
        / "01_review_units_gui"
        / "review_temporal_unit_gui.py"
    )
    spec = importlib.util.spec_from_file_location("review_temporal_unit_gui", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fixture(root: Path, *, actor_available: bool = True) -> dict[str, Path]:
    pig = root / "pig"
    crops = root / "crops"
    pig.mkdir()
    crops.mkdir()
    crop = crops / "actor.png"
    Image.new("RGB", (8, 8), "white").save(crop)
    unit_key = "unit-1"
    pair_id = "pair-1"
    pd.DataFrame(
        [
            {
                "review_unit_id": unit_key,
                "temporal_unit_key": unit_key,
                "source_type": "legacy_recovered",
                "dataset_id": "legacy",
                "video_key": "video",
                "object_track_key": "actor-1",
                "unit_start_frame": 0,
                "unit_end_frame": 0,
                "display_frame_indices": "0",
                "review_relevant_evidence_available": True,
                "review_evidence_reason_auto": "",
            }
        ]
    ).to_csv(root / "units.csv", index=False)
    pd.DataFrame(
        [
            {
                "temporal_unit_key": unit_key,
                "source_type": "legacy_recovered",
                "dataset_id": "legacy",
                "video_key": "video",
                "object_track_key": "actor-1",
                "frame_index": 0,
                "crop_path": str(crop),
                "bbox_valid": True,
                "hidden": "Yes",
                "hidden_after_review": "Yes",
                "hidden_review_status": "reviewed",
                "hidden_is_trusted": True,
                "hidden_trust_status": "trusted_current_review",
                "hidden_source": "current_human_review",
            }
        ]
    ).to_csv(root / "native.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair_id": pair_id,
                "temporal_unit_key": unit_key,
                "source_type": "legacy_recovered",
                "dataset_id": "legacy",
                "video_key": "video",
                "object_track_key": "actor-1",
                "history_start_frame": -1,
                "history_end_frame": -1,
                "target_start_frame": 0,
                "target_end_frame": 0,
            }
        ]
    ).to_csv(pig / "pair_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair_id": pair_id,
                "object_track_key": "actor-1",
                "slot_role": "target",
                "global_slot_index": 0,
                "frame_index": 0,
                "frame_available": True,
                "frame_uid": "frame-0",
            }
        ]
    ).to_csv(pig / "slot_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair_id": pair_id,
                "global_slot_index": 0,
                "frame_uid": "frame-0",
                "frame_available": True,
                "pixel_available": actor_available,
                "pixel_status": "ok" if actor_available else "missing",
            }
        ]
    ).to_csv(pig / "difference_pixel_index.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair_id": pair_id,
                "slot_index": 0,
                "roi_class": "feeder",
                "pixel_geometry_expected": True,
                "pixel_available": True,
                "pixel_status": "ok",
            }
        ]
    ).to_csv(pig / "roi_visual_union_patch_index.csv", index=False)
    stat = crop.stat()
    (pig / "media_manifest.json").write_text(
        json.dumps(
            {
                "valid": True,
                "background_as_temporal_scene_used": False,
                "rejected_static_scene_candidates": [],
                "sources": [
                    {
                        "path": str(crop),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "authority_valid": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "hidden.json").write_text(
        json.dumps(
            {
                "candidate_transaction_state": "COMMITTED_VALIDATED",
                "authority_state": "CANDIDATE_VALIDATED",
            }
        ),
        encoding="utf-8",
    )
    return {
        "review_units_csv": root / "units.csv",
        "native_evidence_csv": root / "native.csv",
        "pig_strenet_artifact_dir": pig,
        "hidden_apply_manifest": root / "hidden.json",
        "legacy_crop_root": crops,
    }


def test_behavior_gui_readiness_accepts_published_exact_pixels(
    tmp_path: Path,
) -> None:
    inputs = _write_fixture(tmp_path)
    audit = audit_behavior_gui_readiness(
        **inputs,
        expected_hidden_reviewed_rows=1,
    )

    assert audit["valid"]
    assert audit["duplicate_review_keys"] == 0
    assert audit["missing_crop_media"] == 0
    assert audit["missing_required_actor_pixels"] == 0
    assert audit["missing_required_scene_media"] == 0
    assert audit["hidden_metadata_source"] == (
        "VALIDATED_CURRENT_CANONICAL_LEDGER"
    )


def test_behavior_gui_readiness_rejects_missing_actor_pixels(
    tmp_path: Path,
) -> None:
    inputs = _write_fixture(tmp_path, actor_available=False)
    audit = audit_behavior_gui_readiness(
        **inputs,
        expected_hidden_reviewed_rows=1,
    )

    assert not audit["valid"]
    assert audit["missing_required_actor_pixels"] == 1


def test_behavior_gui_loader_uses_bounded_column_projection(
    tmp_path: Path,
) -> None:
    module = _load_gui_module()
    path = tmp_path / "frames.csv"
    pd.DataFrame(
        [
            {
                "source_type": "legacy_recovered",
                "dataset_id": "legacy",
                "video_key": "video",
                "frame_index": 0,
                "pig_id": "Pig_1",
                "track_id": "track-1",
                "object_track_key": "actor-1",
                "x1": 0,
                "y1": 0,
                "x2": 4,
                "y2": 4,
                "crop_path": "actor.png",
                "unused_large_feature": "not-loaded",
            }
        ]
    ).to_csv(path, index=False)

    frames = module.load_gui_frame_features(path)

    assert "unused_large_feature" not in frames.columns
    assert set(module.GUI_REQUIRED_FRAME_COLUMNS).issubset(frames.columns)
