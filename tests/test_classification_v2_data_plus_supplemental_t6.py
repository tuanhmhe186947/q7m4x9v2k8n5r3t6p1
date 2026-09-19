from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pig_behavior.tracking.profiles.hybrid_bytetrack import EVAL_CONFIGS

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "classification_v2"
    / "00_source_feature_temporal"
    / "classification_v2_build_data_plus_supplemental_t6.py"
)
SPEC = importlib.util.spec_from_file_location("data_plus_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _actor_rows(
    frames: list[int],
    behaviors: list[str] | None = None,
) -> pd.DataFrame:
    labels = behaviors or ["stand"] * len(frames)
    return pd.DataFrame(
        {
            "object_track_key": ["track-a"] * len(frames),
            "behavior": labels,
            "bbox_valid": [True] * len(frames),
            "frame_index": frames,
            "timestamp_sec": [value / 30.0 for value in frames],
            "video_key": ["pigs010119/000001"] * len(frames),
            "dataset_id": ["data_plus_pigs010119_000001"] * len(frames),
            "clip_id": ["PIGS010119-000001"] * len(frames),
            "clip_name": ["PIGS010119-000001.mp4"] * len(frames),
            "folder": ["data_stand"] * len(frames),
            "source_video_path": ["data/data_plus/example.mp4"] * len(frames),
            "annotation_xml_path": ["data/data_plus/example.xml"] * len(frames),
            "track_id": ["7"] * len(frames),
            "pig_id": ["ID_7"] * len(frames),
        }
    )


def _partner_v2_fixture(
    behavior: str,
    *,
    include_partner: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    frames = list(range(6))
    unit_id = "data-plus-v2-unit"
    clip_id = "clip-a"
    video_key = "pigs010119/000001"
    target_key = "target-track"
    units = pd.DataFrame(
        {
            "supplemental_unit_id": [unit_id],
            "target_object_track_key": [target_key],
            "source_clip_id": [clip_id],
            "video_key": [video_key],
            "source_frame_indices": [MODULE._json(frames)],
            "behavior": [behavior],
        }
    )
    selected_actors = pd.DataFrame(
        {
            "supplemental_unit_id": [unit_id] * 6,
            "clip_id": [clip_id] * 6,
            "dataset_id": ["data_plus_fixture"] * 6,
            "video_key": [video_key] * 6,
            "object_track_key": [target_key] * 6,
            "frame_index": frames,
            "x1": [10.0] * 6,
            "y1": [10.0] * 6,
            "x2": [30.0] * 6,
            "y2": [30.0] * 6,
            "image_width": [100] * 6,
            "image_height": [100] * 6,
            "behavior": [behavior] * 6,
        }
    )
    detection_rows: list[dict[str, object]] = []
    for frame_index in frames:
        detection_rows.append(
            {
                "clip_id": clip_id,
                "frame_index": frame_index,
                "tracker_id": 1,
                "x1": 10.0,
                "y1": 10.0,
                "x2": 30.0,
                "y2": 30.0,
                "confidence": 0.99,
                "image_width": 100,
                "image_height": 100,
                "track_source": "detected",
                "hidden": "No",
                "needs_review": "No",
            }
        )
        if include_partner:
            detection_rows.append(
                {
                    "clip_id": clip_id,
                    "frame_index": frame_index,
                    "tracker_id": 2,
                    "x1": 50.0,
                    "y1": 10.0,
                    "x2": 70.0,
                    "y2": 30.0,
                    "confidence": 0.95,
                    "image_width": 100,
                    "image_height": 100,
                    "track_source": "detected",
                    "hidden": "No",
                    "needs_review": "No",
                }
            )
    return (
        units,
        selected_actors,
        pd.DataFrame(detection_rows),
        {clip_id},
    )


def _extract_partner_v2(
    behavior: str,
    *,
    include_partner: bool,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    fixture = _partner_v2_fixture(
        behavior,
        include_partner=include_partner,
    )
    return MODULE.build_partner_context_v2(*fixture)


def _assert_partner_arrays_equal(
    left: dict[str, object],
    right: dict[str, object],
) -> None:
    for key in (
        "partner_track_key",
        "partner_mask",
        "partner_bbox_xyxy",
        "partner_confidence",
        "partner_geometry_6d",
        "partner_extraction_complete",
    ):
        np.testing.assert_array_equal(left[key], right[key])


@pytest.fixture(scope="module")
def current_data_plus_population(
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    pairs = MODULE.discover_clip_pairs(ROOT / "data" / "data_plus")
    valid = [
        pair
        for pair in pairs
        if str(pair["video_key"]) not in MODULE.FIXED_EXCLUSION_KEYS
    ]
    probes = {
        str(pair["clip_id"]): MODULE.probe_video(pair["media_path"])
        for pair in valid
    }
    actor_rows = pd.concat(
        [
            MODULE.load_actor_rows(pair, probes[str(pair["clip_id"])])
            for pair in valid
        ],
        ignore_index=True,
    )
    units, selected, _ = MODULE.unitize_actor_rows(actor_rows)
    return pairs, units, selected


def test_partner_v2_is_invariant_to_target_label_permutation() -> None:
    fixture = _partner_v2_fixture("stand", include_partner=True)
    baseline, _, _ = MODULE.build_partner_context_v2(*fixture)
    permuted = tuple(value.copy() if hasattr(value, "copy") else set(value) for value in fixture)
    permuted[0]["behavior"] = "fight"
    permuted[0]["class_label"] = "fight"
    permuted[0]["folder_hint"] = "data_fight"
    permuted[1]["behavior"] = "social-nose"
    permuted[1]["annotation_class"] = "social-nose"
    permuted[2]["source_class"] = "fight"
    permuted[2]["class_hint"] = "data_fight"

    changed, _, _ = MODULE.build_partner_context_v2(*permuted)

    _assert_partner_arrays_equal(baseline, changed)


def test_partner_v2_keeps_normalized_distance_then_track_key_ranking() -> None:
    fixture = list(_partner_v2_fixture("stand", include_partner=True))
    detections = fixture[2]
    closer = detections[detections["tracker_id"].eq(2)].copy()
    closer["tracker_id"] = 3
    closer[["x1", "x2"]] = [35.0, 45.0]
    fixture[2] = pd.concat([detections, closer], ignore_index=True)

    arrays, _, _ = MODULE.build_partner_context_v2(*fixture)

    first_keys = arrays["partner_track_key"][0, :, 0]
    second_keys = arrays["partner_track_key"][0, :, 1]
    assert (np.char.find(first_keys, "detector_3") >= 0).all()
    assert (np.char.find(second_keys, "detector_2") >= 0).all()
    distance_index = 4
    assert (
        arrays["partner_geometry_6d"][0, :, 0, distance_index]
        < arrays["partner_geometry_6d"][0, :, 1, distance_index]
    ).all()


@pytest.mark.parametrize("behavior", ["stand", "playwithtoy"])
def test_noninteraction_behavior_can_have_observed_partner(behavior: str) -> None:
    arrays, slots, frames = _extract_partner_v2(
        behavior,
        include_partner=True,
    )

    assert arrays["partner_mask"].shape == (1, 6, 2)
    assert arrays["partner_mask"][0, :, 0].all()
    assert not arrays["partner_mask"][0, :, 1].any()
    assert slots["partner_mask"].sum() == 6
    assert frames["partner_extraction_complete"].all()


@pytest.mark.parametrize("behavior", ["fight", "social-nose"])
def test_interaction_behavior_without_observed_partner_stays_missing(
    behavior: str,
) -> None:
    arrays, slots, frames = _extract_partner_v2(
        behavior,
        include_partner=False,
    )

    assert not arrays["partner_mask"].any()
    assert not slots["partner_mask"].any()
    assert frames["partner_extraction_complete"].all()
    np.testing.assert_array_equal(arrays["partner_bbox_xyxy"], 0.0)
    np.testing.assert_array_equal(arrays["partner_confidence"], 0.0)
    np.testing.assert_array_equal(arrays["partner_geometry_6d"], 0.0)


def test_partner_v2_ignores_partner_behavior_annotations() -> None:
    fixture = _partner_v2_fixture("fight", include_partner=True)
    with_annotations = list(fixture)
    with_annotations[2] = with_annotations[2].copy()
    with_annotations[2]["behavior"] = "social-nose"
    with_annotations[2]["partner_behavior"] = "fight"

    annotated, _, _ = MODULE.build_partner_context_v2(*with_annotations)
    stripped, _, _ = MODULE.build_partner_context_v2(*fixture)

    _assert_partner_arrays_equal(annotated, stripped)


def test_partner_v2_schema_and_masked_padding_contract() -> None:
    arrays, slots, _ = _extract_partner_v2("stand", include_partner=True)

    assert set(arrays) == {
        "supplemental_unit_id",
        "target_object_track_key",
        "source_frame_indices",
        "partner_track_key",
        "partner_mask",
        "partner_bbox_xyxy",
        "partner_confidence",
        "partner_geometry_6d",
        "partner_extraction_complete",
    }
    assert arrays["source_frame_indices"].shape == (1, 6)
    assert arrays["partner_track_key"].shape == (1, 6, 2)
    assert arrays["partner_bbox_xyxy"].shape == (1, 6, 2, 4)
    assert arrays["partner_confidence"].shape == (1, 6, 2)
    assert arrays["partner_geometry_6d"].shape == (1, 6, 2, 6)
    false_slots = ~arrays["partner_mask"]
    assert (arrays["partner_track_key"][false_slots] == "").all()
    assert (arrays["partner_bbox_xyxy"][false_slots] == 0.0).all()
    assert (arrays["partner_confidence"][false_slots] == 0.0).all()
    assert (arrays["partner_geometry_6d"][false_slots] == 0.0).all()
    assert not slots.loc[~slots["partner_mask"], "partner_track_key"].any()


def test_partner_v2_npz_roundtrip_requires_no_pickle(tmp_path: Path) -> None:
    arrays, _, _ = _extract_partner_v2("stand", include_partner=True)
    path = tmp_path / "partner_context_v2.npz"

    MODULE._atomic_npz(arrays, path)

    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        for key, expected in arrays.items():
            assert expected.dtype != object
            np.testing.assert_array_equal(loaded[key], expected)


def test_unitization_is_consecutive_stride_six_without_padding() -> None:
    source = _actor_rows([0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15])

    units, selected, dropped = MODULE.unitize_actor_rows(source)

    assert len(units) == 2
    assert units["source_frame_indices"].tolist() == [
        "[0,1,2,3,4,5]",
        "[10,11,12,13,14,15]",
    ]
    assert units["sampling_pattern"].eq("contiguous").all()
    assert units["unit_stride_frames"].eq(6).all()
    assert units["overlap_frames"].eq(0).all()
    assert units["padding_policy"].eq("none").all()
    assert selected.groupby("supplemental_unit_id").size().eq(6).all()
    assert dropped == {"stand": 2}


def test_unitization_never_crosses_behavior_boundary() -> None:
    source = _actor_rows(
        list(range(12)),
        ["stand"] * 6 + ["move"] * 6,
    )

    units, _, dropped = MODULE.unitize_actor_rows(source)

    assert len(units) == 2
    assert set(units["behavior"]) == {"stand", "move"}
    assert dropped == {"move": 0, "stand": 0}


def test_fold_eligibility_is_train_only_and_group_guarded(tmp_path: Path) -> None:
    fold_manifest = tmp_path / "folds.csv"
    pd.DataFrame(
        {
            "video_key": ["PIGS010119-000001_30fps"],
            "split_VG1": ["test"],
            "split_VG2": ["train"],
            "split_VG3": ["train"],
            "split_VG4": ["train"],
            "split_VG5": ["train"],
        }
    ).to_csv(fold_manifest, index=False)
    units = pd.DataFrame(
        {
            "supplemental_unit_id": ["existing", "new"],
            "video_key": ["pigs010119/000001", "pigs020119/000002"],
        }
    )

    eligibility, _ = MODULE.build_fold_eligibility(units, fold_manifest)

    existing_vg1 = eligibility[
        eligibility["supplemental_unit_id"].eq("existing")
        & eligibility["fold"].eq("VG1")
    ].iloc[0]
    existing_vg2 = eligibility[
        eligibility["supplemental_unit_id"].eq("existing")
        & eligibility["fold"].eq("VG2")
    ].iloc[0]
    new_rows = eligibility[eligibility["supplemental_unit_id"].eq("new")]
    assert existing_vg1["supplemental_role"] == "excluded"
    assert not bool(existing_vg1["eligible_for_training"])
    assert existing_vg2["supplemental_role"] == "train"
    assert bool(existing_vg2["eligible_for_training"])
    assert new_rows["eligible_for_training"].astype(bool).all()
    assert not eligibility["eligible_for_inner_validation"].astype(bool).any()
    assert not eligibility["eligible_for_outer_test"].astype(bool).any()


def test_partner_tracker_uses_exact_production_hybrid_profile(
    tmp_path: Path,
) -> None:
    clip_path = tmp_path / "clip.mp4"
    detector_path = tmp_path / "detector.pt"
    clip_path.write_bytes(b"test")
    detector_path.write_bytes(b"test")
    cfg, profile_hash = MODULE.build_partner_tracking_config(
        clip_path,
        detector_path,
        tmp_path / "tracking",
    )

    assert cfg.mode == "hybrid_bytetrack"
    assert cfg.device == "cpu"
    assert not cfg.half
    assert not cfg.show
    assert not cfg.write_output_video
    assert cfg.det_conf == 0.20
    assert cfg.hidden_owner_guard
    assert cfg.overlap_small_box_suppression
    assert cfg.hidden_suffix_id_swap_repair
    assert profile_hash == MODULE._canonical_hash(
        EVAL_CONFIGS["hybrid_bytetrack_best"]
    )


def test_fragmented_xml_exact_duplicates_relink_by_xml_id() -> None:
    pair = next(
        pair
        for pair in MODULE.discover_clip_pairs(ROOT / "data" / "data_plus")
        if pair["video_key"] == "pigs081119/000298"
    )
    probe = MODULE.probe_video(pair["media_path"])

    rows = MODULE.load_actor_rows(pair, probe)
    units, selected, dropped = MODULE.unitize_actor_rows(rows)

    assert len(rows) == 1202
    assert rows["identity_relink_applied"].astype(bool).all()
    assert rows["identity_authority"].eq("xml_ID_attribute_relinked").all()
    assert rows["xml_exact_duplicate_count"].astype(int).sum() == 120
    assert not rows.duplicated(["frame_index", "pig_id"]).any()
    assert set(rows["pig_id"]) == {"ID_1", "ID_2"}
    assert rows["image_width"].eq(1920).all()
    assert rows["image_height"].eq(1080).all()
    assert len(units) == 200
    assert len(selected) == 1200
    assert units["behavior"].eq("social-nose").all()
    assert dropped == {"social-nose": 2}


def test_fragmented_xml_nonidentical_same_id_frame_requires_review() -> None:
    rows = pd.DataFrame(
        {
            "frame_index": [0, 0],
            "pig_id": ["ID_1", "ID_1"],
            "behavior": ["social-nose", "social-nose"],
            "hidden": ["No", "No"],
            "track_label": ["Pig", "Pig"],
            "x1_raw": [10.0, 20.0],
            "y1_raw": [10.0, 20.0],
            "x2_raw": [30.0, 40.0],
            "y2_raw": [30.0, 40.0],
            "bbox_valid": [True, True],
            "track_id": ["1", "2"],
            "video_key": ["pigs081119/000298"] * 2,
        }
    )

    with pytest.raises(ValueError, match="requires Mini-CVAT identity review"):
        MODULE.normalize_fragmented_xml_identity_rows(rows)


def test_current_data_plus_pairing_and_locked_t6_count(
    current_data_plus_population: tuple[
        list[dict[str, object]],
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    pairs, units, selected = current_data_plus_population
    assert len(pairs) == 20
    assert len({str(pair["xml_path"]) for pair in pairs}) == 20
    pairing = {str(pair["clip_name"]): Path(pair["xml_path"]).name for pair in pairs}
    assert pairing["PIGS051119-0000.mp4"] == "PIGS051119-0001.xml"
    assert pairing["PIGS051119-0001.mp4"] == "PIGS051119-0001(1).xml"
    assert len(units) == 1252
    assert len(selected) == 1252 * 6
    assert units["behavior"].value_counts().to_dict() == {
        "stand": 327,
        "social-nose": 366,
        "playwithtoy": 337,
        "sitting": 102,
        "move": 65,
        "explore": 47,
        "fight": 6,
        "lying": 2,
    }


def test_current_fold_counts_and_canonical_hashes_are_guarded(
    current_data_plus_population: tuple[
        list[dict[str, object]],
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    _, units, _ = current_data_plus_population
    protected = (
        MODULE.DEFAULT_FOLD_MANIFEST,
        MODULE.DEFAULT_CANONICAL_ROW_MANIFEST,
        MODULE.DEFAULT_CANONICAL_NPZ,
    )
    before = {path: MODULE.sha256_file(path) for path in protected}

    eligibility, _ = MODULE.build_fold_eligibility(
        units,
        MODULE.DEFAULT_FOLD_MANIFEST,
    )

    eligible = eligibility[eligibility["eligible_for_training"].astype(bool)]
    assert eligible.groupby("fold").size().to_dict() == {
        "VG1": 657,
        "VG2": 1252,
        "VG3": 1252,
        "VG4": 1244,
        "VG5": 1197,
    }
    assert not eligibility["eligible_for_inner_validation"].astype(bool).any()
    assert not eligibility["eligible_for_outer_test"].astype(bool).any()
    after = {path: MODULE.sha256_file(path) for path in protected}
    assert before == after


def test_builder_source_has_no_behavior_conditioned_partner_authority() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for forbidden in (
        "INTERACTION_BEHAVIORS",
        "requires_partner",
        "xml_partner",
        "interaction_pairs",
    ):
        assert forbidden not in source
