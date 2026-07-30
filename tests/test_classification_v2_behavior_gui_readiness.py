from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
from PIL import Image

from pig_behavior.classification_v2.review.gui_readiness import (
    _audit_pair_alignment,
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


def test_behavior_gui_readiness_accepts_candidate_subset_of_pig_universe(
    tmp_path: Path,
) -> None:
    inputs = _write_fixture(tmp_path)
    units = pd.read_csv(inputs["review_units_csv"])
    pairs = pd.read_csv(inputs["pig_strenet_artifact_dir"] / "pair_manifest.csv")
    additional = pairs.iloc[[0]].copy()
    additional["pair_id"] = "pair-auto-carry"
    additional["temporal_unit_key"] = "unit-auto-carry"
    pairs = pd.concat([pairs, additional], ignore_index=True)

    assert _audit_pair_alignment(units, pairs) == []

    missing = units.assign(temporal_unit_key="unit-not-in-pig")
    assert _audit_pair_alignment(missing, pairs) == [
        "pig_review_unit_alignment_mismatch=1"
    ]


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


def test_temporal_gui_contact_sheet_cache_avoids_rerender() -> None:
    module = _load_gui_module()
    gui = module.ReviewUnitGui.__new__(module.ReviewUnitGui)
    gui.contact_sheet_cache = module.RenderedImageCache(max_items=2)
    gui.decisions = {"unit-1": {"manual_review_decision": "accept"}}
    frames = pd.DataFrame({"frame_index": [1, 2]})
    calls = {"render": 0}

    def render(
        unit: pd.Series,
        matched_frames: pd.DataFrame,
    ) -> tuple[Image.Image, list[str]]:
        calls["render"] += 1
        return Image.new("RGB", (32, 24), "white"), ["diagnostic"]

    gui._frame_rows_for_unit = lambda unit: frames
    gui._make_contact_sheet = render
    unit = pd.Series({"review_unit_id": "unit-1"})

    first, first_diagnostics, first_count = gui._contact_sheet_for_unit(unit)
    first.paste("red", (0, 0, 1, 1))
    second, second_diagnostics, second_count = gui._contact_sheet_for_unit(unit)

    assert calls["render"] == 1
    assert first_diagnostics == second_diagnostics == ["diagnostic"]
    assert first_count == second_count == 2
    assert second.getpixel((0, 0)) == (255, 255, 255)
    assert gui.decisions == {"unit-1": {"manual_review_decision": "accept"}}


def test_reviewer_actions_derive_training_fields_without_manual_weight() -> None:
    module = _load_gui_module()

    assert module.derive_training_fields("accept", "strong") == (
        "main_train",
        "1.0",
    )
    assert module.derive_training_fields("corrected", "medium") == (
        "correct_and_keep",
        "1.0",
    )
    assert module.derive_training_fields("corrected", "weak") == (
        "low_weight_train",
        "0.5",
    )
    assert module.derive_training_fields("accept", "boundary") == (
        "low_weight_train",
        "0.5",
    )
    assert module.derive_training_fields("exclude", "strong") == (
        "exclude",
        "0.0",
    )
    assert module.derive_training_fields("pending", "") == (
        "review_later",
        "0.0",
    )
    decisions = []
    for index, (decision, strength, correction) in enumerate(
        [
            ("accept", "strong", ""),
            ("corrected", "weak", "eat"),
            ("exclude", "boundary", ""),
            ("pending", "", ""),
        ]
    ):
        action, weight = module.derive_training_fields(decision, strength)
        decisions.append(
            {
                "review_unit_id": f"unit-{index}",
                "manual_review_decision": decision,
                "manual_corrected_behavior": correction,
                "manual_label_strength": strength,
                "manual_training_action": action,
                "manual_sample_weight": weight,
            }
        )
    canonical, _ = module.canonicalize_decisions(
        pd.DataFrame.from_records(decisions)
    )
    errors, _ = module.validate_decision_semantics(
        canonical,
        require_complete=False,
    )

    assert errors == []


def test_reviewer_summary_hides_audit_only_clutter() -> None:
    module = _load_gui_module()
    unit = pd.Series(
        {
            "behavior_label": "eat",
            "review_reason_codes": "roi_contradiction|motion_contradiction",
            "review_motion_evidence_available": True,
            "review_roi_evidence_available": True,
            "review_social_evidence_available": False,
            "review_posture_evidence_available": True,
            "source_type": "cvat",
            "video_key": "video-1",
            "pig_id": "Pig_1",
            "unit_start_frame": 12,
            "unit_end_frame": 17,
            "risk_components": "large technical payload",
            "behavior_sampling_probability": 0.123,
        }
    )

    summary = module.format_reviewer_summary(unit, [], 6)

    assert "Nhãn gốc: eat" in summary
    assert "roi contradiction" in summary
    assert "motion ✓" in summary
    assert "social —" in summary
    assert "risk_components" not in summary
    assert "large technical payload" not in summary
    assert "behavior_sampling_probability" not in summary


def test_label_shortcuts_are_disabled_while_typing() -> None:
    module = _load_gui_module()

    class FakeWidget:
        def __init__(self, widget_class: str) -> None:
            self.widget_class = widget_class

        def winfo_class(self) -> str:
            return self.widget_class

    for widget_class in ("Entry", "TEntry", "Text", "TCombobox"):
        assert not module.shortcut_allowed_for_widget(
            FakeWidget(widget_class)
        )
    assert module.shortcut_allowed_for_widget(FakeWidget("TButton"))
    assert module.shortcut_allowed_for_widget(None)


def test_behavior_shortcuts_cover_every_canonical_label_once() -> None:
    module = _load_gui_module()

    assert set(module.BEHAVIOR_SHORTCUTS) == set("1234567890")
    assert set(module.BEHAVIOR_SHORTCUTS.values()) == set(
        module.VALID_BEHAVIORS
    )


def test_legacy_nonzero_target_scope_ignores_overlapping_history() -> None:
    module = _load_gui_module()
    unit = pd.Series(
        {
            "unit_start_frame": 2,
            "unit_end_frame": 17,
            "display_frame_indices": ",".join(
                str(value) for value in range(2, 18)
            ),
            "review_pig_history_display_frame_indices": "2,3,4,5,6,7",
        }
    )
    observed = list(range(2, 18))

    assert module.decision_scope_frames(unit) == observed
    assert module.decision_scope_complete(unit, observed)


def test_decision_scope_still_blocks_an_actually_missing_actor_frame() -> None:
    module = _load_gui_module()
    unit = pd.Series(
        {
            "unit_start_frame": 2,
            "unit_end_frame": 17,
            "display_frame_indices": ",".join(
                str(value) for value in range(2, 18)
            ),
        }
    )
    observed = [value for value in range(2, 18) if value != 9]

    assert not module.decision_scope_complete(unit, observed)


def test_gui_rejects_universe_and_accepts_selective_candidate() -> None:
    module = _load_gui_module()
    candidate = pd.DataFrame(
        [
            {
                "candidate_tier": "TIER_2_HIGH_RISK",
                "include_in_review": True,
                "review_reason_codes": "behavior_evidence_conflict",
                "selection_predicate_version": "selection.v1",
                "selection_config_hash": "a" * 64,
                "review_predicate_global_mandatory": False,
            }
        ]
    )

    assert module.validate_candidate_gui_manifest(candidate) == []
    universe = pd.concat(
        [
            candidate,
            candidate.assign(
                candidate_tier="AUTO_CARRY_LOW_RISK",
                include_in_review=False,
                review_reason_codes="",
            ),
        ],
        ignore_index=True,
    )
    errors = module.validate_candidate_gui_manifest(universe)
    assert "noncandidate_gui_rows=1" in errors
    assert "auto_carry_rows_in_gui=1" in errors


def test_gui_rejects_global_mandatory_manifest() -> None:
    module = _load_gui_module()
    manifest = pd.DataFrame(
        [
            {
                "candidate_tier": "TIER_1_HARD_MANDATORY",
                "include_in_review": True,
                "review_reason_codes": "GLOBAL_MANDATORY_FORBIDDEN",
                "selection_predicate_version": "selection.v1",
                "selection_config_hash": "a" * 64,
                "review_predicate_global_mandatory": True,
            }
        ]
    )

    assert "global_mandatory_gui_rows=1" in (
        module.validate_candidate_gui_manifest(manifest)
    )
