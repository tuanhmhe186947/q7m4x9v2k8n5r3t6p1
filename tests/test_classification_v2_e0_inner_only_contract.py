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
TRANSFER_PACKAGE = ROOT / (
    "docs/classification_v2/corrected_pooled_route_20260806/"
    "lightning_phase2_20260807/pre_gpu_e0_transfer_package.json"
)
TRANSFER_PACKAGE_SHA256 = TRANSFER_PACKAGE.with_suffix(".sha256")
TRANSFER_PACKAGE_VALIDATION = TRANSFER_PACKAGE.with_name(
    "pre_gpu_e0_transfer_package_validation.json"
)
E0_ENVIRONMENT_LOCK = ROOT / (
    "docs/classification_v2/corrected_pooled_route_20260806/"
    "next_phase_20260806_r2/e0_environment/uv.lock"
)
ROOT_DEVELOPMENT_LOCK = ROOT / "uv.lock"


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
    assert handoff["installation_command"] == (
        "uv sync --frozen --python 3.11 --extra pt"
    )


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


def test_e0_environment_lock_binds_canonical_staged_bytes() -> None:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    inventory = json.loads(TRANSFER_INVENTORY.read_text(encoding="utf-8"))
    environment_lock = authority["environment_lock"]
    staged_path = str(E0_ENVIRONMENT_LOCK.relative_to(ROOT)).replace("\\", "/")

    assert (
        environment_lock["lock_authority_type"]
        == "CASE_A_STAGED_LOCK_IS_INTENDED_AND_EXECUTABLE"
    )
    assert environment_lock["canonical_staged_lock_path"] == staged_path
    assert environment_lock["transfer_package_relative_path"] == "uv.lock"
    assert environment_lock["required_extra"] == "pt"
    assert environment_lock["lock_mutation_permitted"] is False
    assert environment_lock["sha256"] == sha256(E0_ENVIRONMENT_LOCK.read_bytes()).hexdigest()
    assert environment_lock["size_bytes"] == E0_ENVIRONMENT_LOCK.stat().st_size

    root_lock = environment_lock["root_development_lock"]
    assert root_lock["path"] == "uv.lock"
    assert root_lock["sha256"] == sha256(ROOT_DEVELOPMENT_LOCK.read_bytes()).hexdigest()
    assert root_lock["authority_role"] == "development_only_not_e0_remote_execution"
    assert root_lock["sha256"] != environment_lock["sha256"]

    inventory_lock = next(
        entry for entry in inventory["entries"] if entry["local_path"] == staged_path
    )
    assert inventory_lock["sha256"] == environment_lock["sha256"]
    assert inventory_lock["remote_destination"] == "$REMOTE_PROJECT_ROOT/uv.lock"


def test_e0_transfer_inventory_hashes_existing_file_entries() -> None:
    inventory = json.loads(TRANSFER_INVENTORY.read_text(encoding="utf-8"))
    checked = 0
    for entry in inventory["entries"]:
        expected_hash = entry.get("sha256")
        if entry.get("kind") != "file" or not expected_hash:
            continue
        local_path = Path(entry["local_path"])
        candidate = local_path if local_path.is_absolute() else ROOT / local_path
        if not candidate.is_file():
            continue
        assert candidate.stat().st_size == entry["size_bytes"]
        assert sha256(candidate.read_bytes()).hexdigest() == expected_hash
        checked += 1
    assert checked >= 5


def test_pre_gpu_transfer_package_binds_current_e0_artifacts() -> None:
    package = json.loads(TRANSFER_PACKAGE.read_text(encoding="utf-8"))
    inventory = json.loads(TRANSFER_INVENTORY.read_text(encoding="utf-8"))
    validation = json.loads(TRANSFER_PACKAGE_VALIDATION.read_text(encoding="utf-8"))
    recorded_hash = TRANSFER_PACKAGE_SHA256.read_text(encoding="utf-8").split()[0]

    assert sha256(TRANSFER_PACKAGE.read_bytes()).hexdigest() == recorded_hash
    assert package["pre_gpu_main_authority"]["git_ref"] == (
        "classification-v2-pre-gpu-authority-20260808"
    )
    assert inventory["pre_gpu_main_authority"]["git_ref"] == package[
        "pre_gpu_main_authority"
    ]["git_ref"]
    assert package["canonical_e0"]["authority_sha256"] == sha256(
        AUTHORITY.read_bytes()
    ).hexdigest()
    assert package["canonical_e0"]["handoff_sha256"] == sha256(
        HANDOFF.read_bytes()
    ).hexdigest()
    assert package["payload_inventory"]["sha256"] == sha256(
        TRANSFER_INVENTORY.read_bytes()
    ).hexdigest()
    assert package["effective_environment"]["required_extra"] == "pt"
    assert package["canonical_e0"]["outer_test_access"] == "BLOCKED"
    assert validation["status"] == "PASS"
    assert validation["package_descriptor"]["sha256"] == recorded_hash
    assert validation["config_only_preflight"]["outer_test_negative_access"] == "PASS; BLOCKED"


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
