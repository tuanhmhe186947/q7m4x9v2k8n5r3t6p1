"""Partner Behavior (PB) Teacher Feature Extraction V1 using Frozen F2 H5 Checkpoints.

Wraps the unified PB teacher extraction pipeline across FULL-T6 and DATA+ V2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from pig_behavior.classification_v2.features.unified_pb_teacher_f2 import (
    DEFAULT_DATA_PLUS_DIR,
    DEFAULT_F2_DIR,
    UnifiedPBTeacherOutputs,
    build_f2_config,
    extract_unified_pb_teacher_fold,
    load_frozen_f2_model,
)
from pig_behavior.classification_v2.features.unified_pb_teacher_f2 import (
    F2_EXPECTED_HASHES as _F2_EXPECTED_HASHES,
)
from pig_behavior.classification_v2.features.unified_pb_teacher_f2 import (
    sha256_file as _sha256_file,
)
from pig_behavior.classification_v2.models.f2_h5_rgb_model import (
    F2H5RgbClassifier,
)

DEFAULT_F2_CHECKPOINT_DIR = DEFAULT_F2_DIR
F2_EXPECTED_HASHES = _F2_EXPECTED_HASHES
build_f2_backbone_config = build_f2_config
load_frozen_f2_teacher = load_frozen_f2_model
PBTeacherF2Outputs = UnifiedPBTeacherOutputs
sha256_file = _sha256_file


def extract_pb_features_from_data_plus(
    model: F2H5RgbClassifier,
    fold_id: str,
    checkpoint_sha256: str,
    data_plus_dir: Path = DEFAULT_DATA_PLUS_DIR,
    *,
    device: torch.device | None = None,
    batch_size: int = 32,
    max_samples: int | None = None,
) -> tuple[UnifiedPBTeacherOutputs, pd.DataFrame, dict[str, Any]]:
    """Extract partner representations using unified extractor."""
    outputs, manifest, audit = extract_unified_pb_teacher_fold(
        model,
        fold_id,
        checkpoint_sha256,
        data_plus_dir=data_plus_dir,
        device=device,
        batch_size=batch_size,
        max_canonical_samples=1,
        max_data_plus_samples=max_samples,
    )
    keep = outputs.source_dataset == "data_plus_v2"
    data_plus_outputs = UnifiedPBTeacherOutputs(
        partner_probs=outputs.partner_probs[keep],
        partner_hidden=outputs.partner_hidden[keep],
        partner_mask=outputs.partner_mask[keep],
        sample_key=outputs.sample_key[keep],
        target_object_track_key=outputs.target_object_track_key[keep],
        partner_track_key=outputs.partner_track_key[keep],
        source_dataset=outputs.source_dataset[keep],
        split_role=outputs.split_role[keep],
        teacher_fold=outputs.teacher_fold,
        teacher_checkpoint_sha256=outputs.teacher_checkpoint_sha256,
    )
    data_plus_manifest = manifest[
        manifest["source_dataset"].eq("data_plus_v2")
    ].reset_index(drop=True)
    data_plus_count = int(keep.sum())
    data_plus_valid = int(data_plus_outputs.partner_mask.sum().item())
    data_plus_slots = data_plus_count * 2
    data_plus_audit = dict(audit)
    data_plus_audit.update(
        {
            "canonical_samples": 0,
            "data_plus_samples": data_plus_count,
            "total_samples": data_plus_count,
            "total_partner_slots": data_plus_slots,
            "valid_partner_slots": data_plus_valid,
            "missing_partner_slots": data_plus_slots - data_plus_valid,
        }
    )
    return data_plus_outputs, data_plus_manifest, data_plus_audit
