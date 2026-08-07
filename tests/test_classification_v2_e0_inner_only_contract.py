"""Focused regression tests for the frozen inner-only E0 execution path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from pig_behavior.classification_v2.models.balanced.baselines import baseline_config
from pig_behavior.classification_v2.training.e0_inner_only import (
    E0_MODEL,
    E0ContractError,
    assert_e0_role_permitted,
    inspect_e0_execution_authority,
    load_e0_execution_authority,
)
from pig_behavior.classification_v2.training.full_multimodal_oof import (
    _ablation_settings,
)

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / (
    "docs/classification_v2/corrected_pooled_route_20260806/"
    "next_phase_20260806_r2/e0_execution_authority.json"
)
WRAPPER = ROOT / (
    "scripts/classification_v2/04_baselines_smokes/"
    "classification_v2_run_e0_inner_only.py"
)
HANDOFF = ROOT / (
    "docs/classification_v2/corrected_pooled_route_20260806/"
    "next_phase_20260806_r2/e0_l4_handoff.json"
)
TRANSFER_INVENTORY = ROOT / (
    "docs/classification_v2/corrected_pooled_route_20260806/"
    "lightning_phase2_20260807/remote_e0_transfer_inventory.json"
)


def test_e0_authority_resolves_exact_b3_modalities() -> None:
    authority = load_e0_execution_authority(AUTHORITY)
    report = inspect_e0_execution_authority(AUTHORITY)

    assert report["model"] == E0_MODEL
    assert report["temporal_view"] == "T6"
    assert report["outer_fold"] == "FOLD_3"
    assert report["seed"] == 20260804
    assert report["modalities"] == authority["modalities"]
    assert authority["modalities"]["geometry_dim"] == 6
    assert authority["modalities"]["motion_dim"] == 12
    assert authority["modalities"]["roi"] is False
    assert authority["modalities"]["social"] is False
    assert authority["modalities"]["interaction_context"] is False
    assert authority["modalities"]["visual_context"] is False
    assert authority["modalities"]["history"] == "none"
    assert authority["modalities"]["posture_auxiliary"] is False


def test_e0_model_has_no_quality_or_availability_controls() -> None:
    authority = load_e0_execution_authority(AUTHORITY)
    settings = authority["model_config"]
    config = baseline_config(
        E0_MODEL,
        target_length=6,
        hidden_dim=settings["hidden_dim"],
        temporal_encoder=settings["temporal_encoder"],
        backbone_name=settings["backbone_name"],
        pretrained_weight_enum=settings["pretrained_weight_enum"],
        image_size=settings["image_size"],
        dropout=settings["dropout"],
        include_controls=False,
    )

    assert config.numeric is not None
    assert config.numeric.groups == ("bbox_xywh_n", "bbox_shape_n", "motion_delta")
    assert config.control_names == ()
    assert config.availability_names == ()


def test_full_and_e0_are_semantically_distinct() -> None:
    full = _ablation_settings("full")
    e0 = load_e0_execution_authority(AUTHORITY)

    assert full["enable_interaction"] is True
    assert full["enable_visual_context"] is True
    assert "roi_class_relation" in full["spatial_groups"]
    assert "social_relation" in full["spatial_groups"]
    assert e0["modalities"]["interaction_context"] is False
    assert e0["modalities"]["visual_context"] is False
    assert e0["modalities"]["roi"] is False
    assert e0["modalities"]["social"] is False


def test_e0_blocks_outer_test_role() -> None:
    with pytest.raises(E0ContractError, match="outer test is blocked"):
        assert_e0_role_permitted("test")


def test_e0_handoff_uses_the_canonical_inner_only_command() -> None:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    canonical = handoff["canonical_execution_authority"]

    assert canonical["path"] == str(AUTHORITY.relative_to(ROOT)).replace("\\", "/")
    assert canonical["launch_uses_variant_full"] is False
    assert canonical["outer_test_access"] == "BLOCKED"
    assert canonical["sha256"] == sha256(AUTHORITY.read_bytes()).hexdigest()
    assert "--variants full" not in handoff["launch_command"]
    assert WRAPPER.name in handoff["launch_command"]
    assert "--resume-checkpoint $E0_RESUME_CHECKPOINT" in handoff["resume_command"]


def test_e0_transfer_inventory_binds_the_canonical_authority() -> None:
    inventory = json.loads(TRANSFER_INVENTORY.read_text(encoding="utf-8"))
    handoff_entry = next(
        entry
        for entry in inventory["entries"]
        if entry["local_path"] == str(HANDOFF.relative_to(ROOT)).replace("\\", "/")
    )

    assert inventory["canonical_execution_authority"]["sha256"] == sha256(
        AUTHORITY.read_bytes()
    ).hexdigest()
    assert handoff_entry["sha256"] == sha256(HANDOFF.read_bytes()).hexdigest()
    assert inventory["remote_transfer_total_size_gb"] <= 15
    assert "H5 feature bundle" in inventory["explicitly_excluded"]


def test_canonical_wrapper_inspects_and_blocks_outer_test(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    report_path = tmp_path / "inspection.json"
    inspect = subprocess.run(
        [
            sys.executable,
            str(WRAPPER),
            "--authority",
            str(AUTHORITY),
            "--mode",
            "inspect",
            "--report",
            str(report_path),
        ],
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert inspect.returncode == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["e0_launch_uses_variant_full"] is False
    assert report["outer_test_access"] == "BLOCKED"

    blocked = subprocess.run(
        [
            sys.executable,
            str(WRAPPER),
            "--authority",
            str(AUTHORITY),
            "--mode",
            "assert-outer-blocked",
        ],
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode == 0
