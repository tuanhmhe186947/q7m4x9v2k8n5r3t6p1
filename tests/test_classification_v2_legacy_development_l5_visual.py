from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from pig_behavior.classification_v2.training.legacy_development_l5 import (
    LegacyL5Config,
    load_legacy_l5_config,
)
from pig_behavior.classification_v2.training.legacy_development_l5_visual import (
    GIB,
    MAX_WINDOWS_WEIGHT_FILE_PATH_CHARS,
    _cached_weight_report,
    _device_preflight_errors,
    _load_probe_sample,
    _spread_rows,
    _validate_windows_weight_path,
    _vram_budget_bytes,
    _weight_filename_and_prefix,
    legacy_l5_visual_probe_controls,
    prepare_legacy_l5_pretrained_weights,
)

CONFIG_PATH = Path("configs/classification_v2/legacy_development_l5_v1.json")


def test_legacy_l5_visual_probe_controls_are_exact() -> None:
    config = load_legacy_l5_config(CONFIG_PATH)

    controls = legacy_l5_visual_probe_controls(config)

    assert [control.control_id for control in controls] == ["V0", "V1", "V2"]
    assert [control.backbone_name for control in controls] == [
        "resnet18",
        "resnet18",
        "resnet34",
    ]
    assert [control.image_size for control in controls] == [160, 224, 224]
    assert [control.frame_batch_size for control in controls] == [16, 16, 8]


def test_legacy_l5_weight_filename_binds_torchvision_hash_prefix() -> None:
    filename, prefix = _weight_filename_and_prefix(
        "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    )

    assert filename == "resnet18-f37072fd.pth"
    assert prefix == "f37072fd"


def test_legacy_l5_weight_path_fails_before_windows_partial_overflow() -> None:
    if os.name != "nt":
        pytest.skip("Windows path guard")
    unsafe = Path("C:/") / ("x" * MAX_WINDOWS_WEIGHT_FILE_PATH_CHARS)

    with pytest.raises(ValueError, match="unsafe for Windows partial"):
        _validate_windows_weight_path(unsafe)


def test_legacy_l5_cached_weight_report_checks_full_digest(tmp_path: Path) -> None:
    payload = b"exact-weight-fixture"
    digest = hashlib.sha256(payload).hexdigest()
    path = tmp_path / f"resnet18-{digest[:8]}.pth"
    path.write_bytes(payload)

    passing = _cached_weight_report(
        path,
        expected_sha256_prefix=digest[:8],
    )
    failing = _cached_weight_report(
        path,
        expected_sha256_prefix="00000000",
    )

    assert passing["valid"] is True
    assert passing["sha256"] == digest
    assert failing["valid"] is False
    assert failing["errors"] == ["cached_weight_sha256_prefix_mismatch"]


def test_legacy_l5_probe_rows_are_deterministic_and_spread() -> None:
    rows = _spread_rows(100, 8)

    assert rows.tolist() == [0, 14, 28, 42, 56, 70, 84, 99]
    assert len(np.unique(rows)) == 8


def test_legacy_l5_probe_sample_binds_rgb_and_context_ids(
    tmp_path: Path,
) -> None:
    tensor = np.arange(10 * 8 * 8 * 3, dtype=np.uint16)
    tensor = (tensor % 256).astype(np.uint8).reshape(10, 8, 8, 3)
    tensor_path = tmp_path / "packed.npy"
    np.save(tensor_path, tensor)
    index_path = tmp_path / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_context_id", "packed_row"],
        )
        writer.writeheader()
        for index in range(10):
            writer.writerow(
                {
                    "image_context_id": f"context_{index}",
                    "packed_row": index,
                }
            )

    sample = _load_probe_sample(
        tensor_path=tensor_path,
        index_path=index_path,
        expected_rows=10,
        image_size=8,
        sample_rows=4,
    )

    assert sample["packed_rows"] == [0, 3, 6, 9]
    assert sample["images"].shape == (4, 8, 8, 3)
    assert sample["rgb_sha256"] == hashlib.sha256(
        tensor[[0, 3, 6, 9]].tobytes()
    ).hexdigest()
    assert len(sample["context_id_sha256"]) == 64


def test_legacy_l5_vram_budget_uses_smaller_total() -> None:
    declared = 4 * GIB
    smaller_actual = int(3.75 * GIB)

    declared_budget = _vram_budget_bytes(
        declared_bytes=declared,
        actual_total_bytes=6 * GIB,
        maximum_fraction=0.7,
    )
    actual_budget = _vram_budget_bytes(
        declared_bytes=declared,
        actual_total_bytes=smaller_actual,
        maximum_fraction=0.7,
    )

    assert declared_budget == int(declared * 0.7)
    assert actual_budget == int(smaller_actual * 0.7)


def test_legacy_l5_device_preflight_fails_closed_on_low_free_vram() -> None:
    total = 4 * GIB
    budget = int(total * 0.7)

    errors = _device_preflight_errors(
        declared_gib=4,
        actual_total_bytes=total,
        mem_info_total_bytes=total,
        free_bytes=budget - 1,
        budget_bytes=budget,
    )

    assert errors == [
        f"free_vram_below_allocator_budget={budget - 1}<{budget}"
    ]


def test_legacy_l5_weight_prepare_without_permission_never_downloads(
    tmp_path: Path,
) -> None:
    base = load_legacy_l5_config(CONFIG_PATH)
    config = LegacyL5Config(
        path=base.path,
        payload=base.payload,
        development_root=tmp_path / "development",
        primary_run_id=base.primary_run_id,
        l3_audit_relative_path=base.l3_audit_relative_path,
        l4_audit_relative_path=base.l4_audit_relative_path,
        l5_output_relative_path=base.l5_output_relative_path,
    )
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(
        json.dumps(
            {
                "status": "PASS_LEGACY_DEVELOPMENT_L5_READINESS",
                "lineage_scope": "legacy-only-unreviewed-development",
                "human_review_complete": False,
                "q2_claim_allowed": False,
                "canonical_full_oof_authorized": False,
                "outer_holdout_predictions_authorized": False,
                "pretrained_weight_prepare_authorized": True,
                "config_sha256": config.sha256,
                "errors": [],
                "valid": True,
            }
        ),
        encoding="utf-8",
    )

    result = prepare_legacy_l5_pretrained_weights(
        config,
        readiness_audit_path=readiness_path,
        weight_cache_root=config.development_root / ".torch_l5",
        allow_download=False,
    )

    assert result["valid"] is False
    assert result["pretrained_weight_downloads"] == 0
    assert len(result["artifacts"]) == 2
    assert all(
        "pretrained_weight_not_cached" in error
        for error in result["errors"]
        if not error.startswith("pretrained_weight_prepare_initialized_cuda")
    )
