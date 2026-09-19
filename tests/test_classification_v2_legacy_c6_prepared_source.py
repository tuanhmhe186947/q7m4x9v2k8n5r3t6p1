from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.legacy_c6_prepared_source import (
    LINEAGE_SCOPE,
    PACKET_SCHEMA,
    load_legacy_c6_prepared_source,
    prepare_legacy_c6_tables,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256


def _source_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    units: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    roles = [*("train" for _ in range(8)), "outer", "validation"]
    for label in VALID_BEHAVIORS:
        for unit_index, role in enumerate(roles):
            key = f"unit::{label}::{unit_index}"
            video = f"video::{label}::{unit_index}"
            units.append(
                {
                    "temporal_unit_key": key,
                    "source_type": "legacy_recovered",
                    "dataset_id": "legacy_recovered_16f",
                    "video_key": video,
                    "behavior_label": label,
                    "native_unit_valid_for_development": True,
                    "native_unit_valid_for_main_eval": True,
                    "lineage_scope": LINEAGE_SCOPE,
                    "human_review_complete": False,
                }
            )
            fold_id = {
                "train": "native_oof_000",
                "outer": "native_oof_005",
                "validation": "native_oof_006",
            }[role]
            recording_group = {
                "train": f"train::{label}::{unit_index}",
                "outer": "outer",
                "validation": "validation",
            }[role]
            folds.append(
                {
                    "temporal_unit_key": key,
                    "recording_group_id": recording_group,
                    "oof_fold_id": fold_id,
                    "behavior_label": label,
                    "native_unit_valid_for_main_eval": True,
                    "lineage_scope": LINEAGE_SCOPE,
                    "human_review_complete": False,
                }
            )
            for slot in range(16):
                frames.append(
                    {
                        "temporal_unit_key": key,
                        "frame_uid": f"frame::{key}::{slot}",
                        "scene_frame_uid": f"scene::{key}::{slot}",
                        "relative_frame_index": slot,
                        "timestamp_sec": slot / 30.0,
                        "behavior_temporal_final": label,
                        "crop_path": f"crop::{key}::{slot}.jpg",
                        "bbox_valid": True,
                        "spatiotemporal_feature_valid": True,
                        "include_in_training": True,
                        "use_for_main_eval": True,
                        "lineage_scope": LINEAGE_SCOPE,
                        "human_review_complete": False,
                    }
                )
    return pd.DataFrame(frames), pd.DataFrame(units), pd.DataFrame(folds)


def test_prepare_excludes_outer_and_keeps_complete_train_validation() -> None:
    frames, units, folds = _source_tables()

    prepared = prepare_legacy_c6_tables(frames, units, folds)

    assert prepared.audit["train_native_units"] == 80
    assert prepared.audit["validation_native_units"] == 10
    assert prepared.audit["outer_holdout_media_reads"] == 0
    assert len(prepared.frames) == 90 * 16
    assert set(prepared.units["l5_role"]) == {"train", "validation"}
    assert not prepared.units["oof_fold_id"].eq("native_oof_005").any()


def test_prepare_all_eligible_train_uses_every_non_holdout_unit() -> None:
    frames, units, folds = _source_tables()

    prepared = prepare_legacy_c6_tables(
        frames,
        units,
        folds,
        train_units_per_class=None,
        train_selection_policy="all_eligible",
    )

    assert prepared.audit["train_selection_policy"] == "all_eligible"
    assert prepared.audit["train_units_per_class"] is None
    assert prepared.audit["train_native_units"] == 80
    assert prepared.audit["validation_native_units"] == 10
    assert len(prepared.frames) == 90 * 16


def test_loads_hash_bound_prepared_packet(tmp_path: Path) -> None:
    frames, units, folds = _source_tables()
    prepared = prepare_legacy_c6_tables(frames, units, folds)
    unit_path = tmp_path / "units.csv"
    frame_path = tmp_path / "frames.csv"
    tensor_path = tmp_path / "features.npy"
    index_path = tmp_path / "index.csv"
    packet_path = tmp_path / "packet.json"
    prepared.units.to_csv(unit_path, index=False)
    prepared.frames.to_csv(frame_path, index=False)
    np.save(
        tensor_path,
        np.ones((len(prepared.frames), 512), dtype=np.float32),
        allow_pickle=False,
    )
    index_columns = [
        "feature_row",
        "position",
        "temporal_unit_key",
        "slot_index",
        "timestamp_sec",
        "lineage_scope",
        "human_review_complete",
    ]
    prepared.frames[index_columns].to_csv(index_path, index=False)
    artifacts = {
        "selected_native_units": _spec(unit_path, tmp_path),
        "selected_frames": _spec(frame_path, tmp_path),
        "actor_feature_tensor": _spec(tensor_path, tmp_path),
        "actor_feature_index": _spec(index_path, tmp_path),
    }
    packet = {
        "schema_version": PACKET_SCHEMA,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "outer_holdout_media_reads": 0,
        "outer_holdout_features_created": 0,
        "outer_holdout_predictions_created": 0,
        "validation_native_units": 10,
        "artifacts": artifacts,
        "valid": True,
    }
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    source = load_legacy_c6_prepared_source(
        packet_path,
        repo_root=tmp_path,
    )

    assert source.base_view.feature_rows.shape == (90, 16)
    assert len(source.selection.train_positions) == 80
    assert len(source.selection.validation_positions) == 10
    assert source.selection.audit["outer_holdout_rows"] == 0


def _spec(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }
