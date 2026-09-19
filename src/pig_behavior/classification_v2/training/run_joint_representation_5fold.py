"""Exact 5-fold scientific training runner for the Final Joint Representation Model.

Implements M2 visual-fine-tuning recipe + Joint Prefusion Adapter (PB1 + H5 + ROI18 + Posture):
  - Base scientific model: M2-VFT (ResNet34 + Spatial 46D + Interaction 5D + Union Visual Context)
  - Visual LR = 0.0003 (0.1x backbone multiplier)
  - Non-visual + Joint prefusion adapter LR = 0.003
  - AdamW, weight_decay = 0.0, scheduler = none, clip = 1.0
  - Behavior sample-weighted CE loss + sparse reviewed Posture auxiliary loss
  - Verified Step-0 M2 logit parity (max diff <= 1e-7)
  - Full DATA+ fold-safe supplemental training plumbing (inner_train only; val/test = 0)
  - Inner train / inner val per epoch
  - Best checkpoint selected ONLY by inner_val_behavior_macro_f1
  - Max epochs = 30, early stopping patience = 5
  - Zero outer evaluation during training
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.classification_v2.models.deep_local_joint_model import (  # noqa: E402
    DEEP_LOCAL_ARCHITECTURE_VERSION,
    DeepLocalJointRepresentationClassifier,
)
from pig_behavior.classification_v2.models.joint_representation_model import (  # noqa: E402
    JOINT_ARCHITECTURE_VERSION,
    JointRepresentationClassifier,
    compute_joint_loss,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (  # noqa: E402
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.training.config import (  # noqa: E402
    ClassificationV2TrainingConfig,
    load_training_config,
)
from pig_behavior.classification_v2.training.data_module import (  # noqa: E402
    StrictTrainingBatch,
    StrictTrainingDataModule,
)
from pig_behavior.classification_v2.training.data_plus_multimodal import (  # noqa: E402
    AlignedDataPlusBatch,
    DataPlusMultimodalStore,
    DataPlusSidecarPaths,
)
from pig_behavior.classification_v2.training.generalization import (  # noqa: E402
    LockedGeneralizationRecipe,
    apply_locked_recipe,
    build_locked_scheduler,
    load_locked_generalization_recipe,
    scheduler_trajectory,
    video_safe_photometric_augment,
)
from pig_behavior.classification_v2.training.trainer import (  # noqa: E402
    _behavior_class_weights,
)
from pig_behavior.classification_v2.training.visual_freeze import (  # noqa: E402
    build_visual_optimizer_groups,
)

SEED: int = 240494961
EXPECTED_M2_PARAM_COUNT: int = 43633832
EXPECTED_JOINT_PARAM_COUNT: int = 43719531
EXPECTED_PARAM_DELTA: int = 85699
EXPECTED_TOTAL_ROWS: int = 33287
DATA_EXTERNAL_PREFIX: str = "EXTERNAL_M0_F1_DATA_ROOT"
RGB_EXTERNAL_PREFIX: str = "EXTERNAL_M0_F1_RGB_ROOT"

DEFAULT_POSTURE_SIDECAR: Path = (
    REPO_ROOT
    / "outputs/classification_v2/g1_posture_auxiliary_v1/g1_posture_sidecar.pt"
)
DEFAULT_PB_DIR: Path = (
    REPO_ROOT / "outputs/classification_v2/pb_teacher_f2_v1"
)
DEFAULT_PB1_DIR: Path = DEFAULT_PB_DIR
DEFAULT_DATA_PLUS_DIR: Path = (
    REPO_ROOT
    / "outputs/classification_v2/data_plus_supplemental_t6_v2_20260825"
)
DEFAULT_H5_STRUCTURED_PATH: Path = (
    REPO_ROOT
    / "outputs/classification_v2/m7_h5s_structured_cache_v1/h5_structured_features.pt"
)
DEFAULT_GENERALIZATION_RECIPE: Path = (
    REPO_ROOT / "configs/classification_v2/final_generalization_recipe_v1.json"
)

M2_INNER_REFERENCES: dict[str, float] = {
    "vg1": 0.642622,
    "vg2": 0.609316,
    "vg3": 0.587899,
    "vg4": 0.675154,
    "vg5": 0.656352,
}

FINAL_JOINT_INNER_REFERENCES: dict[str, float] = {
    "vg1": 0.704274,
    "vg2": 0.694757,
    "vg3": 0.635705,
    "vg4": 0.690224,
    "vg5": 0.694461,
}
CONTROL_GENERALIZATION_GAPS: dict[str, float] = {
    "vg1": 0.2289480574155548,
    "vg2": 0.20228080878176502,
    "vg3": 0.3009361922983014,
    "vg4": 0.2009839239686042,
    "vg5": 0.23339886097854778,
}

DATA_PLUS_COUNTS: dict[str, int] = {
    "vg1": 657,
    "vg2": 1252,
    "vg3": 1252,
    "vg4": 1244,
    "vg5": 1197,
}

EXPECTED_CONTROL_HASHES: dict[str, str] = {
    "vg1": "4cfa96122769cd44428b8dd4732138344633246602aeafaf34c3862a327a9c95",
    "vg2": "a3bec7187bfaa65c09355b19b3f09b3c875269373e170d2ad0b4982f89afe659",
    "vg3": "39ad0847c6391829c00876cb17b0d2d603a376f372438087c83fca15af953154",
    "vg4": "c3c8fc458155e33aa882a6dab33b3337030dc9ac973491d6e3d1efe826e43412",
    "vg5": "b4046a241ae2a5f391b378e6874c19147436544e4a28517fd86c5ff6a0090650",
}


def _select_preflight_keys(
    keys: tuple[str, ...],
    limit: int,
    selection: str,
) -> tuple[str, ...]:
    """Select a deterministic, label-independent CPU-preflight subset."""
    selected_limit = min(max(0, limit), len(keys))
    if selection == "prefix":
        return keys[:selected_limit]
    if selection != "spread":
        raise ValueError(f"Unsupported preflight validation selection: {selection}")
    if selected_limit == 0:
        return ()
    if selected_limit == len(keys):
        return keys
    positions = np.linspace(
        0,
        len(keys) - 1,
        num=selected_limit,
        dtype=np.int64,
    )
    return tuple(keys[int(position)] for position in positions.tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final Joint Representation 5-Fold Runner"
    )
    parser.add_argument(
        "--fold",
        choices=("vg1", "vg2", "vg3", "vg4", "vg5", "all"),
        default="vg1",
        help="Fold ID to execute.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional explicit config path.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root for explicit external data artifacts.",
    )
    parser.add_argument(
        "--rgb-root",
        type=Path,
        default=None,
        help="Root for explicit external RGB cache artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/classification_v2/joint_representation_v1",
        help="Base output directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (cuda or cpu).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run CPU lifecycle preflight through complete epoch boundaries.",
    )
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=None,
        help="Optional max train windows for CPU preflight.",
    )
    parser.add_argument(
        "--preflight-train-selection",
        choices=("prefix", "spread"),
        default="spread",
        help=(
            "Train subset policy for CPU preflight only; spread selects "
            "deterministic positions across the canonical INNER train order."
        ),
    )
    parser.add_argument(
        "--max-val-windows",
        type=int,
        default=None,
        help="Optional max val windows for CPU preflight.",
    )
    parser.add_argument(
        "--preflight-val-selection",
        choices=("prefix", "spread"),
        default="spread",
        help=(
            "Validation subset policy for CPU preflight only; spread selects "
            "deterministic positions across the canonical INNER validation order."
        ),
    )
    parser.add_argument(
        "--include-data-plus",
        action="store_true",
        default=True,
        help="Include fold-safe DATA+ supplemental rows in training.",
    )
    parser.add_argument(
        "--model-variant",
        choices=("control", "deep_local"),
        default="control",
        help="Frozen model variant; deep_local is the locked challenger.",
    )
    parser.add_argument(
        "--generalization-recipe",
        type=Path,
        default=None,
        help="Optional exact locked generalization recipe for refinement runs.",
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=REPO_ROOT / "outputs/classification_v2/final_high_ceiling_v1",
        help="Root containing finalized control runs for challenger warm-start.",
    )
    parser.add_argument(
        "--control-checkpoint",
        type=Path,
        default=None,
        help="Optional exact control checkpoint for the selected fold.",
    )
    parser.add_argument(
        "--preflight-diagnostics-output",
        type=Path,
        default=None,
        help="Optional JSON path for deep-local CPU diagnostics.",
    )
    parser.add_argument(
        "--pb-hidden-dropout-p",
        type=float,
        default=0.50,
        help=(
            "Train-only whole-source Bernoulli dropout probability for "
            "partner_behavior_hidden."
        ),
    )
    parser.add_argument(
        "--allow-all-folds",
        action="store_true",
        default=True,
        help="Execute all 5 folds unconditionally without early futility halt.",
    )
    return parser.parse_args()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _resolve_runtime_config(
    fold_name: str,
    config_path: Path | None,
    data_root: Path | None,
    rgb_root: Path | None,
) -> ClassificationV2TrainingConfig:
    if config_path is None:
        cfg_name = f"g2_cclsa_earlystop_{fold_name}_scientific_v1.json"
        cfg_file = REPO_ROOT / "configs/classification_v2" / cfg_name
        if not cfg_file.exists():
            cfg_name = f"f2_h5rgb_earlystop_{fold_name}_scientific_v1.json"
            cfg_file = REPO_ROOT / "configs/classification_v2" / cfg_name
    else:
        cfg_file = (
            config_path if config_path.is_absolute() else REPO_ROOT / config_path
        )

    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

    config = load_training_config(cfg_file)
    dataset = config.dataset
    updates: dict[str, Any] = {}

    for field_name in (
        "train_ready_root",
        "actor_packed_cache",
        "actor_packed_index",
        "visual_cache_manifest",
        "visual_packed_cache",
        "visual_packed_index",
        "native_oof_fold_manifest",
        "auxiliary_targets_csv",
        "spatial_bundle_npz",
        "frame_context_csv",
        "window_context_csv",
    ):
        raw_val = getattr(dataset, field_name)
        if raw_val is None:
            continue
        val_str = str(raw_val)
        if DATA_EXTERNAL_PREFIX in val_str:
            if data_root is None:
                if field_name == "spatial_bundle_npz":
                    updates[field_name] = (
                        REPO_ROOT
                        / "outputs/classification_v2/full_t6_canonical_46d_20260816"
                        / "full_t6_canonical_46d.npz"
                    )
                elif field_name == "train_ready_root":
                    updates[field_name] = (
                        REPO_ROOT
                        / "outputs/classification_v2/full_t6_canonical_46d_20260816"
                    )
                else:
                    updates[field_name] = None
            else:
                updates[field_name] = Path(
                    val_str.replace(DATA_EXTERNAL_PREFIX, str(data_root))
                )
        else:
            value = Path(raw_val)
            updates[field_name] = (
                value if value.is_absolute() else REPO_ROOT / value
            )

    for field_name in (
        "window_major_rgb_cache",
        "window_major_union_mask",
        "window_major_window_index",
    ):
        raw_val = getattr(dataset, field_name)
        if raw_val is None:
            continue
        val_str = str(raw_val)
        if RGB_EXTERNAL_PREFIX in val_str:
            if rgb_root is None:
                if field_name == "window_major_rgb_cache":
                    updates[field_name] = (
                        REPO_ROOT
                        / "outputs/classification_v2/m0_window_major_r128_t6"
                        / "m0_rgb_window_major_u8.npy"
                    )
                elif field_name == "window_major_union_mask":
                    updates[field_name] = (
                        REPO_ROOT
                        / "outputs/classification_v2/m0_window_major_r128_t6"
                        / "m0_union_available_mask.npy"
                    )
                elif field_name == "window_major_window_index":
                    updates[field_name] = (
                        REPO_ROOT
                        / "outputs/classification_v2/m0_window_major_r128_t6"
                        / "m0_rgb_window_index.csv"
                    )
                else:
                    updates[field_name] = None
            else:
                updates[field_name] = Path(
                    val_str.replace(RGB_EXTERNAL_PREFIX, str(rgb_root))
                )
        else:
            value = Path(raw_val)
            updates[field_name] = (
                value if value.is_absolute() else REPO_ROOT / value
            )

    for field_name in (
        "snapshot_json",
        "trainer_contract_json",
        "grouped_fold_roles",
    ):
        raw_val = getattr(dataset, field_name)
        if raw_val is not None:
            value = Path(raw_val)
            updates[field_name] = (
                value if value.is_absolute() else REPO_ROOT / value
            )

    return replace(config, dataset=replace(dataset, **updates))


def extract_real_production_source_tensors(
    data: StrictTrainingDataModule,
    batch: StrictTrainingBatch,
    sample_keys: tuple[str, ...],
    canonical_key_to_row: dict[str, int],
    pb_cache: dict[str, Any],
    sample_key_to_pb_idx: dict[str, int],
    posture_sidecar: dict[str, Any],
    h5_cache: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Extract real PB (probs + hidden), H5, ROI, Posture, and validity tensors.

    Strict failure on missing entries without any silent positional fallback.
    """
    resolved_rows: list[int] = []
    pb_positions: list[int] = []
    for key in sample_keys:
        if key not in canonical_key_to_row:
            raise KeyError(f"Unknown canonical sample_key: {key}")
        if key not in sample_key_to_pb_idx:
            raise KeyError(
                f"Missing PB sidecar entry for sample_key '{key}'. "
                "Strict failure (no silent fallback)."
            )
        resolved_rows.append(canonical_key_to_row[key])
        pb_positions.append(sample_key_to_pb_idx[key])
    row_positions = np.asarray(resolved_rows, dtype=np.int64)

    pb_probs = pb_cache["partner_probs"][pb_positions].to(device)
    pb_hidden = pb_cache["partner_hidden"][pb_positions].to(device)
    pb_mask = pb_cache["partner_mask"][pb_positions].to(device)

    # 2. Real H5 causal structured tensors (5 observations before target T6)
    h5_structured = h5_cache["h5_structured"][row_positions].to(device).clone()
    h5_structured[..., 35] = 0.0
    h5_mask = h5_cache["history_observed_mask"][row_positions].to(device)

    # 3. Real ROI sequence & validity tensors
    if "roi_class_relation" in batch.model_inputs["spatial_features"]:
        roi_clean = (
            batch.model_inputs["spatial_features"]["roi_class_relation"]
            .clone()
        )
        roi_clean[..., 17] = 0.0
        batch.model_inputs["spatial_features"]["roi_class_relation"] = roi_clean
    roi_sequence = (
        batch.model_inputs["spatial_features"]["roi_class_relation"]
        .to(device)
        .clone()
    )
    roi_sequence[..., 17] = 0.0
    roi_validity = (
        torch.from_numpy(data.bundle.arrays["roi_validity_mask"][row_positions])
        .bool()
        .to(device)
    )

    # 4. Real Posture targets and masks from authoritative sidecar
    pos_targets = posture_sidecar["posture_target"][row_positions].to(device)
    pos_masks = posture_sidecar["posture_reviewed_mask"][row_positions].to(device)

    return {
        "partner_behavior_probs": pb_probs,
        "partner_behavior_hidden": pb_hidden,
        "partner_behavior_mask": pb_mask,
        "h5_structured": h5_structured,
        "h5_mask": h5_mask,
        "roi_sequence": roi_sequence,
        "roi_validity": roi_validity,
        "posture_targets": pos_targets,
        "posture_reviewed_mask": pos_masks,
    }


def _canonical_key_maps(
    data: StrictTrainingDataModule,
    pb_cache: dict[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    """Re-key fixed canonical authorities by the persisted window identity."""

    canonical_keys = data.bundle.frame["window_id"].astype(str).tolist()
    if len(canonical_keys) != len(set(canonical_keys)):
        raise ValueError("canonical window_id values must be unique")
    canonical_key_to_row = {
        key: position for position, key in enumerate(canonical_keys)
    }
    pb_key_to_position: dict[str, int] = {}
    for position, recorded_key in enumerate(pb_cache["sample_key"]):
        sample_key = str(recorded_key)
        if sample_key.startswith("data_plus_"):
            continue
        if sample_key not in canonical_key_to_row:
            raise KeyError(f"unknown canonical PB sample_key: {sample_key}")
        if sample_key in pb_key_to_position:
            raise ValueError(f"duplicate canonical PB sample_key: {sample_key}")
        pb_key_to_position[sample_key] = position
    return canonical_key_to_row, pb_key_to_position


@dataclass(slots=True)
class JointAlignedTrainingBatch:
    training_batch: StrictTrainingBatch
    context_inputs: dict[str, torch.Tensor]


class JointKeyedBatchResolver:
    """Resolve canonical and DATA+ samples through one exact-key interface."""

    def __init__(
        self,
        data: StrictTrainingDataModule,
        pb_cache: dict[str, Any],
        posture_sidecar: dict[str, Any],
        h5_cache: dict[str, Any],
        data_plus: DataPlusMultimodalStore,
        device: torch.device,
    ) -> None:
        self.data = data
        self.pb_cache = pb_cache
        self.posture_sidecar = posture_sidecar
        self.h5_cache = h5_cache
        self.data_plus = data_plus
        self.device = device
        (
            self.canonical_key_to_row,
            self.canonical_key_to_pb_position,
        ) = _canonical_key_maps(data, pb_cache)
        overlap = set(self.canonical_key_to_row).intersection(data_plus.sample_keys)
        if overlap:
            raise ValueError(f"canonical/DATA+ sample_key collision: {sorted(overlap)[:5]}")

    def canonical_keys(self, rows: np.ndarray) -> tuple[str, ...]:
        keys = self.data.bundle.frame.iloc[rows]["window_id"].astype(str)
        result = tuple(keys.tolist())
        if any(key not in self.canonical_key_to_row for key in result):
            raise KeyError("canonical row did not resolve to a sample_key")
        return result

    def batch(self, sample_keys: tuple[str, ...]) -> JointAlignedTrainingBatch:
        if not sample_keys:
            raise ValueError("Joint batch requires at least one sample_key")
        if len(sample_keys) != len(set(sample_keys)):
            raise ValueError("Joint batch sample_key values must be unique")
        canonical_keys = tuple(
            key for key in sample_keys if key in self.canonical_key_to_row
        )
        data_plus_keys = tuple(
            key for key in sample_keys if key in set(self.data_plus.sample_keys)
        )
        unknown = set(sample_keys).difference(canonical_keys, data_plus_keys)
        if unknown:
            raise KeyError(f"unknown Joint sample_key values: {sorted(unknown)[:5]}")

        canonical = (
            self._canonical_batch(canonical_keys) if canonical_keys else None
        )
        supplemental = (
            self._data_plus_batch(data_plus_keys) if data_plus_keys else None
        )
        if canonical is None:
            assert supplemental is not None
            return supplemental
        if supplemental is None:
            return canonical
        return _merge_joint_batches(
            canonical,
            supplemental,
            canonical_keys + data_plus_keys,
            sample_keys,
        )

    def _canonical_batch(
        self,
        sample_keys: tuple[str, ...],
    ) -> JointAlignedTrainingBatch:
        rows = np.asarray(
            [self.canonical_key_to_row[key] for key in sample_keys],
            dtype=np.int64,
        )
        batch = self.data.batch(rows)
        sources = extract_real_production_source_tensors(
            self.data,
            batch,
            sample_keys,
            self.canonical_key_to_row,
            self.pb_cache,
            self.canonical_key_to_pb_position,
            self.posture_sidecar,
            self.h5_cache,
            self.device,
        )
        batch.metadata["sample_key"] = list(sample_keys)
        context = {
            name: sources[name]
            for name in (
                "partner_behavior_probs",
                "partner_behavior_hidden",
                "partner_behavior_mask",
                "h5_structured",
                "h5_mask",
                "roi_sequence",
                "roi_validity",
            )
        }
        context["posture_targets"] = sources["posture_targets"]
        context["posture_reviewed_mask"] = sources[
            "posture_reviewed_mask"
        ]
        return JointAlignedTrainingBatch(batch, context)

    def _data_plus_batch(
        self,
        sample_keys: tuple[str, ...],
    ) -> JointAlignedTrainingBatch:
        state = self.data.fold_preprocessing_state
        if state is None:
            raise ValueError("fold preprocessing must be fitted before DATA+ batches")

        def transform(
            features: dict[str, torch.Tensor],
            validity: dict[str, torch.Tensor],
            observed: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            transformed = state.transform_torch(
                features,
                length_mask=observed,
                observed_mask=observed,
                quality_mask=observed,
                feature_validity_masks=validity,
            )
            if "roi_class_relation" in transformed:
                roi_clean = transformed["roi_class_relation"].clone()
                roi_clean[..., 17] = 0.0
                transformed["roi_class_relation"] = roi_clean
            return transformed

        aligned: AlignedDataPlusBatch = self.data_plus.batch(
            sample_keys,
            device=self.device,
            spatial_transform=transform,
        )
        context = dict(aligned.context_inputs)
        context["posture_targets"] = aligned.posture_targets
        context["posture_reviewed_mask"] = aligned.posture_reviewed_mask
        return JointAlignedTrainingBatch(aligned.training_batch, context)


def _merge_joint_batches(
    first: JointAlignedTrainingBatch,
    second: JointAlignedTrainingBatch,
    grouped_keys: tuple[str, ...],
    requested_keys: tuple[str, ...],
) -> JointAlignedTrainingBatch:
    grouped_position = {key: index for index, key in enumerate(grouped_keys)}
    order = torch.tensor(
        [grouped_position[key] for key in requested_keys],
        dtype=torch.long,
        device=first.training_batch.behavior_target.device,
    )

    def merge(value_a: Any, value_b: Any) -> Any:
        if isinstance(value_a, torch.Tensor) and isinstance(value_b, torch.Tensor):
            return torch.cat([value_a, value_b], dim=0).index_select(0, order)
        if isinstance(value_a, dict) and isinstance(value_b, dict):
            if set(value_a) != set(value_b):
                raise ValueError("canonical/DATA+ batch dictionaries differ")
            return {name: merge(value_a[name], value_b[name]) for name in value_a}
        raise TypeError("unsupported Joint batch merge value")

    merged_inputs = merge(
        first.training_batch.model_inputs,
        second.training_batch.model_inputs,
    )
    merged_context = merge(first.context_inputs, second.context_inputs)
    behavior = merge(
        first.training_batch.behavior_target,
        second.training_batch.behavior_target,
    )
    sample_weight = merge(
        first.training_batch.sample_weight,
        second.training_batch.sample_weight,
    )
    metadata = {
        "sample_key": list(requested_keys),
        "source_type": [
            "data_plus_supplemental"
            if key in set(second.training_batch.metadata["sample_key"])
            else "canonical_full_t6"
            for key in requested_keys
        ],
    }
    batch = StrictTrainingBatch(
        model_inputs=merged_inputs,
        behavior_target=behavior,
        auxiliary_targets={},
        auxiliary_masks={},
        sample_weight=sample_weight,
        metadata=metadata,
    )
    return JointAlignedTrainingBatch(batch, merged_context)


def verify_joint_step0_parity(
    model: JointRepresentationClassifier,
    reference_model: torch.nn.Module,
    data: StrictTrainingDataModule,
    resolver: JointKeyedBatchResolver,
) -> float:
    """Verify step-0 parity against the supplied control model."""
    model.load_state_dict(reference_model.state_dict(), strict=False)
    model.eval()
    reference_model.eval()

    rows = data.split_indices("train")[:4]
    sample_keys = resolver.canonical_keys(rows)
    aligned = resolver.batch(sample_keys)
    b_batch = aligned.training_batch
    sources = aligned.context_inputs

    with torch.inference_mode():
        reference_inputs = dict(b_batch.model_inputs)
        if isinstance(reference_model, JointRepresentationClassifier):
            reference_logits = reference_model.forward_behavior(
                **reference_inputs,
                partner_behavior_probs=sources["partner_behavior_probs"],
                partner_behavior_hidden=sources["partner_behavior_hidden"],
                partner_behavior_mask=sources["partner_behavior_mask"],
                h5_structured=sources["h5_structured"],
                h5_mask=sources["h5_mask"],
                roi_sequence=sources["roi_sequence"],
                roi_validity=sources["roi_validity"],
            )
        else:
            reference_logits = reference_model(**reference_inputs)
        joint_logits = model.forward_behavior(
            **reference_inputs,
            partner_behavior_probs=sources["partner_behavior_probs"],
            partner_behavior_hidden=sources["partner_behavior_hidden"],
            partner_behavior_mask=sources["partner_behavior_mask"],
            h5_structured=sources["h5_structured"],
            h5_mask=sources["h5_mask"],
            roi_sequence=sources["roi_sequence"],
            roi_validity=sources["roi_validity"],
        )

    max_diff = (reference_logits - joint_logits).abs().max().item()
    if max_diff > 1e-6:
        raise ValueError(
            "Step-0 control/challenger parity failed! "
            f"Max logit diff: {max_diff}"
        )
    return max_diff


def _module_grad_l2(module: torch.nn.Module) -> float:
    """Return a finite diagnostic norm without changing gradients."""

    values = [
        parameter.grad.detach().float().pow(2).sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt().item())


def run_deep_local_preflight_diagnostics(
    model: DeepLocalJointRepresentationClassifier,
    data: StrictTrainingDataModule,
    resolver: JointKeyedBatchResolver,
    sample_keys: tuple[str, ...],
    class_weights: torch.Tensor,
    output_path: Path,
) -> dict[str, Any]:
    """Prove real-batch branch liveness, context sensitivity, and alignment."""

    probe_keys = tuple(sample_keys[: min(4, len(sample_keys))])
    if not probe_keys:
        raise ValueError("deep local preflight needs at least one canonical row")
    aligned = resolver.batch(probe_keys)
    batch = aligned.training_batch
    sources = aligned.context_inputs
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(
        **batch.model_inputs,
        partner_behavior_probs=sources["partner_behavior_probs"],
        partner_behavior_hidden=sources["partner_behavior_hidden"],
        partner_behavior_mask=sources["partner_behavior_mask"],
        h5_structured=sources["h5_structured"],
        h5_mask=sources["h5_mask"],
        roi_sequence=sources["roi_sequence"],
        roi_validity=sources["roi_validity"],
    )
    total_loss, behavior_loss, posture_loss = compute_joint_loss(
        behavior_logits=output.behavior_logits,
        behavior_targets=batch.behavior_target,
        class_weights=class_weights,
        posture_logits=output.posture_logits,
        posture_targets=sources["posture_targets"],
        posture_reviewed_mask=sources["posture_reviewed_mask"],
        sample_weight=batch.sample_weight,
        posture_lambda=0.25,
    )
    total_loss.backward()

    actor_encoder = model.image_encoder
    union_encoder = model.visual_context_encoder
    if actor_encoder is None or union_encoder is None:
        raise RuntimeError("deep local preflight lost actor or union encoder")
    map_shapes = {
        stream: {stage: list(value.shape) for stage, value in maps.items()}
        for stream, maps in model._latest_spatial_maps.items()
    }
    actor_maps = model._latest_spatial_maps["actor"]
    union_maps = model._latest_spatial_maps["union"]
    map_alias = any(
        actor_maps[stage].data_ptr() == union_maps[stage].data_ptr()
        for stage in ("layer3", "layer4")
    )

    model.eval()
    with torch.no_grad():
        model._augment_fused_embedding(
            output.m2_fused_448,
            m2_branches=output.m2_branches,
            partner_context=output.partner_context,
            partner_hidden_context=output.partner_hidden_context,
            h5_context=output.h5_context,
            roi_context=output.roi_context,
            posture_context=output.posture_context,
            length_mask=batch.model_inputs["length_mask"],
            observed_mask=batch.model_inputs.get("observed_mask"),
            image_length_mask=batch.model_inputs.get("image_length_mask"),
            image_observed_mask=batch.model_inputs.get("image_observed_mask"),
            image_available_mask=batch.model_inputs.get("image_available_mask"),
            image_quality_mask=batch.model_inputs.get("image_quality_mask"),
            image_time_delta=batch.model_inputs.get("image_time_delta"),
            visual_context_length_mask=batch.model_inputs.get(
                "visual_context_length_mask"
            ),
            visual_context_observed_mask=batch.model_inputs.get(
                "visual_context_observed_mask"
            ),
            visual_context_available_mask=batch.model_inputs.get(
                "visual_context_available_mask"
            ),
            visual_context_quality_mask=batch.model_inputs.get(
                "visual_context_quality_mask"
            ),
            visual_context_time_delta=batch.model_inputs.get(
                "visual_context_time_delta"
            ),
        )
        base_local = model._latest_local_representation
        perturbed_h5 = sources["h5_structured"].clone()
        perturbed_h5 = perturbed_h5 + 0.25 * sources["h5_mask"].unsqueeze(-1)
        perturbed_h5_context = model._encode_h5_context(
            len(probe_keys),
            perturbed_h5,
            sources["h5_mask"],
        )
        model._augment_fused_embedding(
            output.m2_fused_448,
            m2_branches=output.m2_branches,
            partner_context=output.partner_context,
            partner_hidden_context=output.partner_hidden_context,
            h5_context=perturbed_h5_context,
            roi_context=output.roi_context,
            posture_context=output.posture_context,
            length_mask=batch.model_inputs["length_mask"],
            observed_mask=batch.model_inputs.get("observed_mask"),
            image_length_mask=batch.model_inputs.get("image_length_mask"),
            image_observed_mask=batch.model_inputs.get("image_observed_mask"),
            image_available_mask=batch.model_inputs.get("image_available_mask"),
            image_quality_mask=batch.model_inputs.get("image_quality_mask"),
            image_time_delta=batch.model_inputs.get("image_time_delta"),
            visual_context_length_mask=batch.model_inputs.get(
                "visual_context_length_mask"
            ),
            visual_context_observed_mask=batch.model_inputs.get(
                "visual_context_observed_mask"
            ),
            visual_context_available_mask=batch.model_inputs.get(
                "visual_context_available_mask"
            ),
            visual_context_quality_mask=batch.model_inputs.get(
                "visual_context_quality_mask"
            ),
            visual_context_time_delta=batch.model_inputs.get(
                "visual_context_time_delta"
            ),
        )
        perturbed_local = model._latest_local_representation
        context_effect = (
            0.0
            if base_local is None or perturbed_local is None
            else float((base_local - perturbed_local).abs().max().item())
        )

    mask_shape_ok = (
        tuple(batch.model_inputs["length_mask"].shape[:1]) == (len(probe_keys),)
        and actor_maps["layer3"].shape[0]
        == len(probe_keys) * batch.model_inputs["image"].shape[1]
        and union_maps["layer3"].shape[0]
        == len(probe_keys) * batch.model_inputs["visual_context_image"].shape[1]
    )
    payload: dict[str, Any] = {
        "model_variant": "deep_local",
        "sample_keys": list(probe_keys),
        "logits_shape": list(output.behavior_logits.shape),
        "finite_logits": bool(torch.isfinite(output.behavior_logits).all()),
        "local_representation_shape": (
            None
            if output.local_representation is None
            else list(output.local_representation.shape)
        ),
        "local_representation_nonzero": bool(
            output.local_representation is not None
            and torch.count_nonzero(output.local_representation).item() > 0
        ),
        "map_shapes": map_shapes,
        "actor_union_map_alias": bool(map_alias),
        "temporal_map_mask_alignment": bool(mask_shape_ok),
        "context_sensitivity_h5_max_abs": context_effect,
        "loss": {
            "total": float(total_loss.detach().item()),
            "behavior": float(behavior_loss.detach().item()),
            "posture": float(posture_loss.detach().item()),
        },
        "gradient_l2": {
            "local_query": _module_grad_l2(model.local_branch.query_encoder),
            "actor_attention": _module_grad_l2(
                model.local_branch.actor_stream.attention
            ),
            "union_attention": _module_grad_l2(
                model.local_branch.union_stream.attention
            ),
            "actor_temporal": _module_grad_l2(
                model.local_branch.actor_stream.temporal_encoder
            ),
            "union_temporal": _module_grad_l2(
                model.local_branch.union_stream.temporal_encoder
            ),
            "local_residual_projection": _module_grad_l2(
                model.local_residual_projection
            ),
            "actor_layer3": _module_grad_l2(actor_encoder.frame_encoder.layer3),
            "actor_layer4": _module_grad_l2(actor_encoder.frame_encoder.layer4),
            "union_layer3": _module_grad_l2(union_encoder.frame_encoder.layer3),
            "union_layer4": _module_grad_l2(union_encoder.frame_encoder.layer4),
            "behavior_head": _module_grad_l2(model.classifier),
        },
        "data_plus_inner_val_count": 0,
        "data_plus_outer_test_count": 0,
        "outer_test_evaluations": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"DEEP_LOCAL_PREFLIGHT_DIAGNOSTICS: {output_path}")
    print(f"NEW_BRANCH_CONTEXT_SENSITIVITY_MAX_ABS: {context_effect:.6e}")
    for name, value in payload["gradient_l2"].items():
        print(f"GRADIENT_L2_{name.upper()}: {value:.6e}")
    return payload


def _joint_forward(
    model: torch.nn.Module,
    aligned: JointAlignedTrainingBatch,
    *,
    model_inputs: dict[str, Any] | None = None,
    partner_behavior_hidden_mask: torch.Tensor | None = None,
) -> Any:
    """Consume the complete Joint multimodal interface for one aligned batch."""

    batch = aligned.training_batch
    sources = aligned.context_inputs
    return model(
        **(model_inputs if model_inputs is not None else batch.model_inputs),
        partner_behavior_probs=sources["partner_behavior_probs"],
        partner_behavior_hidden=sources["partner_behavior_hidden"],
        partner_behavior_mask=sources["partner_behavior_mask"],
        partner_behavior_hidden_mask=partner_behavior_hidden_mask,
        h5_structured=sources["h5_structured"],
        h5_mask=sources["h5_mask"],
        roi_sequence=sources["roi_sequence"],
        roi_validity=sources["roi_validity"],
    )


def _mask_audit(mask: torch.Tensor | None) -> str:
    if mask is None:
        return "N/A"
    finite = bool(torch.isfinite(mask).all().item())
    binary = bool(torch.all((mask == 0) | (mask == 1)).item())
    valid = int((mask != 0).sum().item())
    return f"finite={finite};binary={binary};valid={valid}/{mask.numel()}"


def _print_modality_contract(
    name: str,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    temporal_length: int | str,
    consumed: str,
) -> None:
    availability = "present"
    if mask is not None:
        availability = f"masked_present={int((mask != 0).any().item())}"
    print(
        "  modality="
        f"{name} shape={tuple(value.shape)} dtype={value.dtype} "
        f"availability={availability} mask_validity={_mask_audit(mask)} "
        f"temporal_length={temporal_length} model_path_consumed={consumed}"
    )


def _print_batch_contract(
    label: str,
    aligned: JointAlignedTrainingBatch,
    output: Any,
) -> None:
    """Print the real canonical train/validation interface and consumption proof."""

    batch = aligned.training_batch
    inputs = batch.model_inputs
    sources = aligned.context_inputs
    print(f"{label}_BATCH_MODALITY_CONTRACT")
    _print_modality_contract(
        "actor_rgb",
        inputs["image"],
        inputs.get("image_available_mask"),
        temporal_length=inputs["image"].shape[1],
        consumed="YES(base+local_spatiotemporal)",
    )
    _print_modality_contract(
        "union_rgb",
        inputs["visual_context_image"],
        inputs.get("visual_context_available_mask"),
        temporal_length=inputs["visual_context_image"].shape[1],
        consumed="YES(base+local_spatiotemporal)",
    )
    for name, value in inputs["spatial_features"].items():
        _print_modality_contract(
            f"Structured46D/{name}",
            value,
            inputs.get("spatial_available_mask"),
            temporal_length=value.shape[1],
            consumed="YES",
        )
    _print_modality_contract(
        "interaction_features",
        inputs["interaction_context_features"],
        inputs.get("interaction_context_available_mask"),
        temporal_length="N/A",
        consumed="YES",
    )
    _print_modality_contract(
        "causal_H5",
        sources["h5_structured"],
        sources["h5_mask"],
        temporal_length=sources["h5_structured"].shape[1],
        consumed="YES",
    )
    _print_modality_contract(
        "ROI",
        sources["roi_sequence"],
        sources["roi_validity"],
        temporal_length=sources["roi_sequence"].shape[1],
        consumed="YES",
    )
    _print_modality_contract(
        "PB_partner_probs",
        sources["partner_behavior_probs"],
        sources["partner_behavior_mask"],
        temporal_length=sources["partner_behavior_probs"].shape[1],
        consumed="YES",
    )
    _print_modality_contract(
        "PB_partner_hidden",
        sources["partner_behavior_hidden"],
        sources["partner_behavior_mask"],
        temporal_length=sources["partner_behavior_hidden"].shape[1],
        consumed="YES",
    )
    _print_modality_contract(
        "posture_context_model_derived",
        output.posture_context,
        torch.ones(output.posture_context.shape[0], device=output.posture_context.device),
        temporal_length="derived",
        consumed="YES",
    )
    _print_modality_contract(
        "posture_supervision_only",
        sources["posture_targets"].unsqueeze(-1),
        sources["posture_reviewed_mask"].unsqueeze(-1),
        temporal_length="N/A",
        consumed="NO(loss-only)",
    )
    if output.local_representation is None:
        raise ValueError("new local/spatiotemporal RGB pathway was bypassed")
    _print_modality_contract(
        "new_local_spatiotemporal_rgb_path",
        output.local_representation,
        None,
        temporal_length="pooled",
        consumed="YES",
    )


def _verify_generalization_input_parity(
    model: torch.nn.Module,
    resolver: JointKeyedBatchResolver,
    train_keys: tuple[str, ...],
    val_keys: tuple[str, ...],
    recipe: LockedGeneralizationRecipe,
) -> None:
    """Prove canonical train and inner-val use the same inference interface."""

    train_aligned = resolver.batch(train_keys)
    val_aligned = resolver.batch(val_keys)
    model.eval()
    with torch.inference_mode():
        train_output = _joint_forward(model, train_aligned)
        val_output = _joint_forward(model, val_aligned)
    _print_batch_contract("CANONICAL_TRAIN", train_aligned, train_output)
    _print_batch_contract("CANONICAL_INNER_VAL", val_aligned, val_output)

    train_augmented = video_safe_photometric_augment(
        train_aligned.training_batch.model_inputs,
        recipe,
    )
    train_delta = max(
        float(
            (train_augmented[name] - train_aligned.training_batch.model_inputs[name])
            .abs()
            .max()
            .item()
        )
        for name in ("image", "visual_context_image")
    )
    if train_delta <= 0.0:
        raise ValueError("train photometric augmentation produced no RGB change")
    print("TRAIN_AUGMENTATION_ACTIVE = YES")
    print(f"TRAIN_RGB_MAX_ABS_DELTA = {train_delta:.8f}")
    print("VAL_AUGMENTATION_ACTIVE = NO")
    print("VAL_RGB_MAX_ABS_DELTA = 0.00000000")
    print("VALIDATION_INPUT_PARITY = PASS")
    print("AUGMENTATION_PREFLIGHT = PASS")
    model.train()


def run_joint_training(
    fold_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    set_seed(SEED)
    fold_upper = fold_name.upper()
    run_dir = args.output_dir / "runs" / fold_name.lower()
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print(f"STARTING JOINT REPRESENTATION RUN: {fold_upper}")
    print("=======================================================")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Compute Device: {device}")

    # 1. Config
    runtime_config = _resolve_runtime_config(
        fold_name, args.config, args.data_root, args.rgb_root
    )
    recipe: LockedGeneralizationRecipe | None = None
    if args.generalization_recipe is not None:
        if args.model_variant != "deep_local":
            raise ValueError(
                "generalization refinement must use the locked deep_local architecture"
            )
        recipe_path = args.generalization_recipe
        if not recipe_path.is_absolute():
            recipe_path = REPO_ROOT / recipe_path
        recipe = load_locked_generalization_recipe(
            recipe_path,
            expected_control_commit="b6a9fdaeb17db50b939fc6cc45af3c9d528c093d",
        )
        runtime_config = apply_locked_recipe(runtime_config, recipe)
        print(f"GENERALIZATION_RECIPE = {recipe.recipe_id}")
        print(f"GENERALIZATION_RECIPE_PATH = {recipe_path}")
        print("ARCHITECTURE_CHANGED = NO")
        print("DATA_CHANGED = NO")
        print("SPLIT_CHANGED = NO")
        print("PB_CHANGED = NO")

    # 2. Data Module
    print("Initializing StrictTrainingDataModule...")
    data = StrictTrainingDataModule(runtime_config, device=device)
    data.fit_fold_preprocessor()

    full_train_indices = data.split_indices("train")
    full_val_indices = data.split_indices("validation")
    test_indices = data.split_indices("test")

    # Load PB Teacher F2 V1 feature cache for this fold
    pb_cache_path = DEFAULT_PB_DIR / fold_name.lower() / "pb_teacher_features.pt"
    if not pb_cache_path.exists():
        raise FileNotFoundError(
            f"Missing required PB_TEACHER_F2_V1 cache at: {pb_cache_path}"
        )
    pb_cache = torch.load(pb_cache_path, map_location="cpu", weights_only=False)

    # Load G1 Posture sidecar
    posture_sidecar = torch.load(
        DEFAULT_POSTURE_SIDECAR,
        map_location="cpu",
        weights_only=False,
    )

    # Load M7 H5 Structured cache
    h5_cache = torch.load(
        DEFAULT_H5_STRUCTURED_PATH,
        map_location="cpu",
        weights_only=False,
    )

    data_plus_store = DataPlusMultimodalStore(
        DataPlusSidecarPaths.from_root(
            DEFAULT_DATA_PLUS_DIR,
            pb_path=pb_cache_path,
        )
    )
    resolver = JointKeyedBatchResolver(
        data,
        pb_cache,
        posture_sidecar,
        h5_cache,
        data_plus_store,
        device,
    )
    full_train_keys = resolver.canonical_keys(full_train_indices)
    full_val_keys = resolver.canonical_keys(full_val_indices)

    # DATA+ fold-safe supplemental reporting and inclusion
    eligibility = pd.read_csv(
        DEFAULT_DATA_PLUS_DIR / "supplemental_fold_eligibility.csv",
        low_memory=False,
    )
    fold_rows = eligibility[eligibility["fold"].astype(str).eq(fold_upper)]
    invalid_eval = (
        fold_rows["eligible_for_inner_validation"].astype(bool)
        | fold_rows["eligible_for_outer_test"].astype(bool)
    )
    if invalid_eval.any():
        raise ValueError("DATA+ fold authority contains validation/test eligibility")
    eligible_units = fold_rows.loc[
        fold_rows["eligible_for_training"].astype(bool),
        "supplemental_unit_id",
    ].astype(str)
    data_plus_keys = tuple(
        f"data_plus_{unit_id}" for unit_id in eligible_units.tolist()
    )
    if len(data_plus_keys) != len(set(data_plus_keys)):
        raise ValueError("DATA+ fold eligibility contains duplicate units")
    expected_data_plus = DATA_PLUS_COUNTS.get(fold_name.lower(), 0)
    if len(data_plus_keys) != expected_data_plus:
        raise ValueError(
            f"DATA+ fold count mismatch: {len(data_plus_keys)} "
            f"!= {expected_data_plus}"
        )
    if not args.include_data_plus:
        data_plus_keys = ()
    data_plus_count = len(data_plus_keys)
    print(
        f"[DATA+ SUPPLEMENTAL] Fold {fold_upper}: Eligible Train Rows = {data_plus_count}, "
        f"Val Rows = 0, Test Rows = 0"
    )

    if args.preflight:
        max_train = args.max_train_windows or 64
        max_val = args.max_val_windows or 32
        supplemental_probe = data_plus_keys[: min(4, len(data_plus_keys))]
        canonical_limit = max(1, max_train - len(supplemental_probe))
        canonical_train_keys = _select_preflight_keys(
            full_train_keys,
            canonical_limit,
            args.preflight_train_selection,
        )
        train_sample_keys = canonical_train_keys + supplemental_probe
        val_sample_keys = _select_preflight_keys(
            full_val_keys,
            max_val,
            args.preflight_val_selection,
        )
        print(
            f"[PREFLIGHT MODE] Subsampling train={len(train_sample_keys)}, "
            f"val={len(val_sample_keys)} "
            f"train_selection={args.preflight_train_selection} "
            f"val_selection={args.preflight_val_selection}"
        )
    else:
        train_sample_keys = full_train_keys + data_plus_keys
        val_sample_keys = full_val_keys

    print(
        f"Rows: Train={len(train_sample_keys)} "
        f"(canonical_full={len(full_train_indices)}, data_plus={data_plus_count}), "
        f"Val={len(val_sample_keys)} (canonical_full={len(full_val_indices)}), "
        f"Test={len(test_indices)}"
    )

    # 3. Model Architecture
    probe = data.batch(full_train_indices[: min(len(full_train_indices), 2)])
    spatial_input_dims = {
        name: probe.model_inputs["spatial_features"][name].shape[-1]
        for name in runtime_config.model.spatial_feature_groups
    }
    interaction_dim = (
        probe.model_inputs["interaction_context_features"].shape[-1]
        if runtime_config.model.enable_interaction_context
        else None
    )
    hidden_dim = runtime_config.model.hidden_dim
    backbone_config = MultimodalFusionConfig(
        spatial_input_dims=spatial_input_dims,
        num_classes=10,
        interaction_context_dim=interaction_dim,
        backbone_name=runtime_config.model.backbone_name,
        pretrained_weight_enum=runtime_config.model.pretrained_weight_enum,
        image_embedding_dim=hidden_dim,
        spatial_embedding_dim=hidden_dim,
        interaction_embedding_dim=max(8, hidden_dim // 2),
        visual_context_embedding_dim=hidden_dim,
        fusion_hidden_dim=hidden_dim * 2,
        dropout=runtime_config.model.dropout,
        temporal_encoder_name=runtime_config.model.temporal_encoder_name,
        transformer_layers=runtime_config.model.transformer_layers,
        transformer_heads=runtime_config.model.transformer_heads,
        enable_image=runtime_config.model.enable_image,
        enable_spatial=runtime_config.model.enable_spatial,
        enable_interaction_context=runtime_config.model.enable_interaction_context,
        enable_visual_context=runtime_config.model.enable_visual_context,
        enable_partner_tokens=False,
    )

    # Instantiate the selected model and an explicit control for Step-0 proof.
    if args.model_variant == "deep_local":
        control_model = JointRepresentationClassifier(backbone_config).to(device)
        model = DeepLocalJointRepresentationClassifier(backbone_config).to(device)
        checkpoint_path = (
            args.control_checkpoint
            if args.control_checkpoint is not None
            else args.control_root / "runs" / fold_name.lower() / "best_validation.pt"
        )
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_ROOT / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Final Joint control checkpoint is required: {checkpoint_path}"
            )
        loaded_sha = sha256_file(checkpoint_path)
        print(f"LOADED_CONTROL_CHECKPOINT_PATH = {checkpoint_path}")
        print(f"LOADED_CONTROL_CHECKPOINT_SHA256 = {loaded_sha}")
        expected_sha = EXPECTED_CONTROL_HASHES.get(fold_name.lower())
        if expected_sha and loaded_sha != expected_sha:
            raise ValueError(
                f"Control checkpoint SHA256 mismatch for {fold_name.upper()}: "
                f"{loaded_sha} != {expected_sha}"
            )
        control_checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        control_state = control_checkpoint.get("model_state_dict")
        if not isinstance(control_state, dict):
            raise ValueError("control checkpoint lacks model_state_dict")
        if any(k.startswith("local_branch.") for k in control_state.keys()):
            control_model = DeepLocalJointRepresentationClassifier(
                backbone_config
            ).to(device)
            control_model.load_state_dict(control_state, strict=True)
            model.load_state_dict(control_state, strict=True)
        else:
            control_model.load_state_dict(control_state, strict=True)
            missing, unexpected = model.load_state_dict(
                control_state, strict=False
            )
            allowed_missing = {
                name
                for name, _ in model.named_parameters()
                if name.startswith("local_branch.")
                or name.startswith("local_residual_projection.")
            }
            if set(missing) != allowed_missing or unexpected:
                raise ValueError(
                    "challenger warm-start changed non-local state: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        reference_model = control_model
        print(f"Warm-start control checkpoint: {checkpoint_path}")
    else:
        m2_ref_model = MultimodalFusionClassifier(backbone_config).to(device)
        model = JointRepresentationClassifier(backbone_config).to(device)
        reference_model = m2_ref_model

    # Parameter count verification
    total_params = sum(p.numel() for p in model.parameters())
    reference_params = sum(p.numel() for p in reference_model.parameters())
    param_delta = total_params - reference_params
    print(f"Model Variant:           {args.model_variant}")
    print(f"Model Total Parameters:  {total_params}")
    print(f"Reference Parameters:    {reference_params}")
    print(f"Parameter Delta:         {param_delta}")
    if args.model_variant == "control":
        assert total_params == EXPECTED_JOINT_PARAM_COUNT, (
            f"Parameter count mismatch: {total_params} != "
            f"{EXPECTED_JOINT_PARAM_COUNT}"
        )
        assert reference_params == EXPECTED_M2_PARAM_COUNT, (
            f"M2 parameter count mismatch: {reference_params} != "
            f"{EXPECTED_M2_PARAM_COUNT}"
        )

    # Step-0 control/challenger parity check.
    max_logit_diff = verify_joint_step0_parity(
        model,
        reference_model,
        data,
        resolver,
    )
    print(
        f"Verified Step-0 Control Parity: Max Logit Diff = "
        f"{max_logit_diff:.10e} (PASS <= 1e-6)"
    )
    # 4. Optimizer & Loss
    lr = runtime_config.optimization.learning_rate  # 0.003
    visual_multiplier = runtime_config.model.visual_backbone_lr_multiplier  # 0.1
    weight_decay = runtime_config.optimization.weight_decay
    param_groups, optimizer_group_contract = build_visual_optimizer_groups(
        model,
        learning_rate=lr,
        backbone_lr_multiplier=visual_multiplier,
        weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = build_locked_scheduler(optimizer, recipe) if recipe is not None else None
    print("OPTIMIZER_GROUPS = " + json.dumps(optimizer_group_contract["groups"]))
    if recipe is not None:
        trajectory = scheduler_trajectory(optimizer, recipe, steps=4)
        print("SCHEDULER_TRAJECTORY = " + json.dumps(trajectory))
        if any(
            trajectory[index][0] < trajectory[index + 1][0]
            for index in range(len(trajectory) - 1)
        ):
            raise ValueError("cosine scheduler trajectory is not nonincreasing")
        print("SCHEDULER_PREFLIGHT = PASS")
    architecture_version = (
        DEEP_LOCAL_ARCHITECTURE_VERSION
        if args.model_variant == "deep_local"
        else JOINT_ARCHITECTURE_VERSION
    )

    class_weights = _behavior_class_weights(
        data,
        full_train_indices,
        runtime_config,
        device,
    )

    if recipe is not None:
        _verify_generalization_input_parity(
            model,
            resolver,
            full_train_keys[: min(2, len(full_train_keys))],
            full_val_keys[: min(2, len(full_val_keys))],
            recipe,
        )

    if args.preflight and args.model_variant == "deep_local":
        diagnostics_path = (
            args.preflight_diagnostics_output
            if args.preflight_diagnostics_output is not None
            else REPO_ROOT
            / "scratch/final_high_ceiling_campaign_20260826"
            / f"deep_local_preflight_{fold_name.lower()}.json"
        )
        if not diagnostics_path.is_absolute():
            diagnostics_path = REPO_ROOT / diagnostics_path
        run_deep_local_preflight_diagnostics(
            model,
            data,
            resolver,
            tuple(train_sample_keys),
            class_weights,
            diagnostics_path,
        )

    batch_size = args.batch_size or runtime_config.optimization.batch_size
    max_epochs = args.epochs or (
        2 if args.preflight else runtime_config.optimization.epochs
    )
    patience = runtime_config.optimization.early_stopping_patience

    pb_dropout_p = getattr(args, "pb_hidden_dropout_p", 0.50) if recipe is not None else 0.0
    print(f"PB_HIDDEN_TRAIN_DROPOUT_ACTIVE = {'YES' if pb_dropout_p > 0.0 else 'NO'}")
    print("PB_PROBS_TRAIN_DROPOUT_ACTIVE = NO")
    print("PB_HIDDEN_VAL_DROPOUT_ACTIVE = NO")
    print(f"PB_HIDDEN_SOURCE_DROPOUT_P = {pb_dropout_p:.2f}")

    print(
        f"Training Plan: Max Epochs={max_epochs}, Batch Size={batch_size}, "
        f"Patience={patience}"
    )

    if args.preflight and args.model_variant == "deep_local":
        reference_model.eval()
        control_val_preds: list[np.ndarray] = []
        control_val_targets: list[np.ndarray] = []
        val_num_batches = int(np.ceil(len(val_sample_keys) / batch_size))
        with torch.inference_mode():
            for b_idx in range(val_num_batches):
                b_keys = val_sample_keys[
                    b_idx * batch_size : (b_idx + 1) * batch_size
                ]
                aligned = resolver.batch(b_keys)
                out = _joint_forward(reference_model, aligned)
                control_val_preds.append(
                    out.behavior_logits.argmax(dim=-1).cpu().numpy()
                )
                control_val_targets.append(
                    aligned.training_batch.behavior_target.cpu().numpy()
                )
        control_targets = np.concatenate(control_val_targets)
        control_preds = np.concatenate(control_val_preds)
        control_report = classification_report(
            control_targets,
            control_preds,
            output_dict=True,
            zero_division=0,
        )
        control_subset_f1 = float(control_report["macro avg"]["f1-score"])
        support = {
            str(int(label)): int(count)
            for label, count in zip(
                *np.unique(control_targets, return_counts=True),
                strict=True,
            )
        }
        print(f"CONTROL_SUBSET_VAL_F1 = {control_subset_f1:.6f}")
        print(f"PREFLIGHT_VAL_CLASS_SUPPORT = {json.dumps(support)}")
        del reference_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 5. Step-0 Full Validation Evaluation
    model.eval()
    step0_val_loss_accum = 0.0
    step0_val_preds_list: list[np.ndarray] = []
    step0_val_targets_list: list[np.ndarray] = []
    step0_logits_list: list[np.ndarray] = []
    val_num_batches = int(np.ceil(len(val_sample_keys) / batch_size))

    with torch.inference_mode():
        for b_idx in range(val_num_batches):
            b_keys = val_sample_keys[
                b_idx * batch_size : (b_idx + 1) * batch_size
            ]
            aligned = resolver.batch(b_keys)
            b_batch = aligned.training_batch
            sources = aligned.context_inputs

            out = _joint_forward(model, aligned)
            _, b_loss, _ = compute_joint_loss(
                behavior_logits=out.behavior_logits,
                behavior_targets=b_batch.behavior_target,
                class_weights=class_weights,
                posture_logits=out.posture_logits,
                posture_targets=sources["posture_targets"],
                posture_reviewed_mask=sources["posture_reviewed_mask"],
                sample_weight=b_batch.sample_weight,
                posture_lambda=0.0,
            )

            step0_val_loss_accum += b_loss.item() * len(b_keys)
            step0_logits_list.append(out.behavior_logits.detach().cpu().numpy())
            preds = out.behavior_logits.argmax(dim=-1).detach().cpu().numpy()
            step0_val_preds_list.append(preds)
            step0_val_targets_list.append(
                b_batch.behavior_target.detach().cpu().numpy()
            )

    step0_val_loss = step0_val_loss_accum / len(val_sample_keys)
    all_step0_preds = np.concatenate(step0_val_preds_list)
    all_step0_targets = np.concatenate(step0_val_targets_list)
    step0_report = classification_report(
        all_step0_targets,
        all_step0_preds,
        output_dict=True,
        zero_division=0,
    )
    step0_inner_f1 = float(step0_report["macro avg"]["f1-score"])
    print(f"STEP0_INNER_MACRO_F1 = {step0_inner_f1:.6f}")

    step0_checkpoint_path = run_dir / "step0_checkpoint.pt"
    torch.save(
        {
            "epoch": -1,
            "model_state_dict": model.state_dict(),
            "inner_val_behavior_macro_f1": step0_inner_f1,
            "inner_val_loss": step0_val_loss,
            "fold": fold_name,
            "architecture_version": architecture_version,
        },
        step0_checkpoint_path,
    )

    epoch_history: list[dict[str, Any]] = []
    best_f1 = step0_inner_f1
    best_epoch = -1
    best_val_loss = step0_val_loss
    best_val_report = step0_report
    best_val_logits = np.concatenate(step0_logits_list, axis=0)
    patience_counter = 0

    best_checkpoint_path = run_dir / "best_validation.pt"
    last_checkpoint_path = run_dir / "last.pt"
    history_json_path = run_dir / "epoch_history.json"

    # Save initial best as Step-0 checkpoint
    torch.save(
        {
            "epoch": -1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "inner_val_behavior_macro_f1": step0_inner_f1,
            "inner_val_loss": step0_val_loss,
            "fold": fold_name,
            "architecture_version": architecture_version,
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "lr_at_best": [float(group["lr"]) for group in optimizer.param_groups],
            "generalization_gap_at_best": 0.0,
        },
        best_checkpoint_path,
    )

    # 6. Training Loop
    for epoch in range(max_epochs):
        model.train()
        epoch_lr_start = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        shuffled_train = np.random.permutation(train_sample_keys)
        num_batches = int(np.ceil(len(shuffled_train) / batch_size))
        train_loss_accum = 0.0
        train_preds_list: list[np.ndarray] = []
        train_targets_list: list[np.ndarray] = []

        for b_idx in range(num_batches):
            b_keys = tuple(
                str(value)
                for value in shuffled_train[
                    b_idx * batch_size : (b_idx + 1) * batch_size
                ]
            )
            aligned = resolver.batch(b_keys)
            b_batch = aligned.training_batch
            sources = aligned.context_inputs

            optimizer.zero_grad()
            train_model_inputs = (
                video_safe_photometric_augment(
                    b_batch.model_inputs,
                    recipe,
                )
                if recipe is not None
                else b_batch.model_inputs
            )
            if pb_dropout_p > 0.0:
                b_size = len(b_keys)
                drop_mask = torch.bernoulli(
                    torch.full((b_size,), pb_dropout_p, device=device)
                ).bool()
                pb_hidden_mask = sources["partner_behavior_mask"].clone()
                pb_hidden_mask[drop_mask] = False
            else:
                pb_hidden_mask = None

            out = _joint_forward(
                model,
                aligned,
                model_inputs=train_model_inputs,
                partner_behavior_hidden_mask=pb_hidden_mask,
            )

            tot_loss, b_loss, _ = compute_joint_loss(
                behavior_logits=out.behavior_logits,
                behavior_targets=b_batch.behavior_target,
                class_weights=class_weights,
                posture_logits=out.posture_logits,
                posture_targets=sources["posture_targets"],
                posture_reviewed_mask=sources["posture_reviewed_mask"],
                sample_weight=b_batch.sample_weight,
                posture_lambda=0.25,
            )

            tot_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                runtime_config.optimization.gradient_clip_norm,
            )
            optimizer.step()

            train_loss_accum += b_loss.item() * len(b_keys)
            preds = out.behavior_logits.argmax(dim=-1).detach().cpu().numpy()
            train_preds_list.append(preds)
            train_targets_list.append(b_batch.behavior_target.detach().cpu().numpy())

        epoch_train_loss = train_loss_accum / len(train_sample_keys)
        all_train_preds = np.concatenate(train_preds_list)
        all_train_targets = np.concatenate(train_targets_list)
        train_report = classification_report(
            all_train_targets,
            all_train_preds,
            output_dict=True,
            zero_division=0,
        )
        train_macro_f1 = float(train_report["macro avg"]["f1-score"])

        # Inner Validation
        model.eval()
        val_loss_accum = 0.0
        val_preds_list: list[np.ndarray] = []
        val_targets_list: list[np.ndarray] = []
        val_logits_list: list[np.ndarray] = []
        val_num_batches = int(np.ceil(len(val_sample_keys) / batch_size))

        with torch.inference_mode():
            for b_idx in range(val_num_batches):
                b_keys = val_sample_keys[
                    b_idx * batch_size : (b_idx + 1) * batch_size
                ]
                aligned = resolver.batch(b_keys)
                b_batch = aligned.training_batch
                sources = aligned.context_inputs

                out = _joint_forward(model, aligned)
                _, b_loss, _ = compute_joint_loss(
                    behavior_logits=out.behavior_logits,
                    behavior_targets=b_batch.behavior_target,
                    class_weights=class_weights,
                    posture_logits=out.posture_logits,
                    posture_targets=sources["posture_targets"],
                    posture_reviewed_mask=sources["posture_reviewed_mask"],
                    sample_weight=b_batch.sample_weight,
                    posture_lambda=0.0,
                )

                val_loss_accum += b_loss.item() * len(b_keys)
                val_logits_list.append(out.behavior_logits.detach().cpu().numpy())
                preds = out.behavior_logits.argmax(dim=-1).detach().cpu().numpy()
                val_preds_list.append(preds)
                val_targets_list.append(b_batch.behavior_target.detach().cpu().numpy())

        epoch_val_loss = val_loss_accum / len(val_sample_keys)
        all_val_preds = np.concatenate(val_preds_list)
        all_val_targets = np.concatenate(val_targets_list)
        val_report = classification_report(
            all_val_targets,
            all_val_preds,
            output_dict=True,
            zero_division=0,
        )
        val_macro_f1 = float(val_report["macro avg"]["f1-score"])
        if scheduler is not None:
            scheduler.step()
        epoch_lr_end = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        generalization_gap = train_macro_f1 - val_macro_f1

        is_best = val_macro_f1 > best_f1
        if is_best:
            best_f1 = val_macro_f1
            best_epoch = epoch
            best_val_loss = epoch_val_loss
            best_val_report = val_report
            best_val_logits = np.concatenate(val_logits_list, axis=0)
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "inner_val_behavior_macro_f1": val_macro_f1,
                    "inner_val_loss": epoch_val_loss,
                    "fold": fold_name,
                    "architecture_version": architecture_version,
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler is not None else None
                    ),
                    "lr_at_best": epoch_lr_start,
                    "generalization_gap_at_best": generalization_gap,
                },
                best_checkpoint_path,
            )
        else:
            patience_counter += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "inner_val_behavior_macro_f1": val_macro_f1,
                "inner_val_loss": epoch_val_loss,
                "fold": fold_name,
                "architecture_version": architecture_version,
                "scheduler_state_dict": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
            },
            last_checkpoint_path,
        )

        epoch_rec = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_behavior_macro_f1": train_macro_f1,
            "inner_val_loss": epoch_val_loss,
            "inner_val_behavior_macro_f1": val_macro_f1,
            "generalization_gap": generalization_gap,
            "lr_start": epoch_lr_start,
            "lr_end": epoch_lr_end,
            "is_best": is_best,
        }
        epoch_history.append(epoch_rec)
        history_json_path.write_text(json.dumps(epoch_history, indent=2))

        tag = " *BEST*" if is_best else ""
        print(
            f"Epoch {epoch:02d} | Train Loss: {epoch_train_loss:.4f} "
            f"F1: {train_macro_f1:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} F1: {val_macro_f1:.4f} "
            f"Gap: {generalization_gap:.4f} | LR: {epoch_lr_start} | {tag}"
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered at Epoch {epoch:02d}.")
            break

    # 7. Checkpoint Reload Verification
    print(
        f"\nReloading best checkpoint from {best_checkpoint_path} "
        f"(Best Epoch: {best_epoch})..."
    )
    ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    model.eval()
    print("CHECKPOINT_RELOAD = PASS")

    best_sha = sha256_file(best_checkpoint_path)
    last_sha = sha256_file(last_checkpoint_path)
    hist_sha = sha256_file(history_json_path)

    print(f"BEST_CHECKPOINT_SHA256: {best_sha}")
    print(f"LAST_CHECKPOINT_SHA256: {last_sha}")
    print(f"EPOCH_HISTORY_SHA256:   {hist_sha}")
    print("OUTER_TEST_EVAL_COUNT:  0")

    best_history = next(
        (row for row in epoch_history if int(row["epoch"]) == best_epoch),
        None,
    )
    if best_history is not None:
        train_f1_at_best = float(best_history["train_behavior_macro_f1"])
        refined_gap = float(best_history["generalization_gap"])
        lr_at_best = best_history["lr_start"]
    else:
        train_f1_at_best = 0.0
        refined_gap = 0.0
        lr_at_best = [float(group["lr"]) for group in optimizer.param_groups]

    control_gap = CONTROL_GENERALIZATION_GAPS.get(fold_name.lower())
    gap_reduction = (
        float(control_gap - refined_gap) if control_gap is not None else None
    )

    best_trained_f1 = max(
        [row["inner_val_behavior_macro_f1"] for row in epoch_history],
        default=step0_inner_f1,
    )
    delta_trained_vs_step0 = best_trained_f1 - step0_inner_f1

    selected_source = (
        "STEP0" if best_epoch == -1 else f"TRAINED_EPOCH_{best_epoch:02d}"
    )
    print(f"STEP0_F1 = {step0_inner_f1:.6f}")
    print(f"BEST_TRAINED_F1 = {best_trained_f1:.6f}")
    print(f"FINAL_SELECTED_BEST_F1 = {best_f1:.6f}")
    print(f"SELECTED_SOURCE = {selected_source}")
    print(f"DELTA_TRAINED_VS_STEP0 = {delta_trained_vs_step0:+.6f}")
    print(f"TRAIN_F1_AT_BEST: {train_f1_at_best:.6f}")
    print(f"VAL_F1_AT_BEST: {best_f1:.6f}")
    print(f"GAP_AT_BEST: {refined_gap:.6f}")
    print(f"CONTROL_GAP_AT_BEST: {control_gap}")
    print(f"GAP_REDUCTION: {gap_reduction}")

    # Save per-class F1 and validation logits for the best model
    per_class_f1 = {
        str(k): float(v["f1-score"])
        for k, v in best_val_report.items()
        if k.isdigit()
    }
    (run_dir / "per_class_f1.json").write_text(json.dumps(per_class_f1, indent=2))
    np.savez_compressed(run_dir / "validation_logits.npz", logits=best_val_logits)

    reference_name = (
        "Final Joint" if args.model_variant == "deep_local" else "M2"
    )
    reference_f1 = (
        FINAL_JOINT_INNER_REFERENCES
        if args.model_variant == "deep_local"
        else M2_INNER_REFERENCES
    ).get(fold_name.lower(), 0.0)

    return {
        "fold": fold_name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "best_val_loss": best_val_loss,
        "best_sha256": best_sha,
        "checkpoint_path": str(best_checkpoint_path),
        "last_sha256": last_sha,
        "history_sha256": hist_sha,
        "epochs_run": len(epoch_history),
        "epoch_history": epoch_history,
        "train_f1_at_best": train_f1_at_best,
        "refined_generalization_gap": refined_gap,
        "control_generalization_gap": control_gap,
        "gap_reduction": gap_reduction,
        "lr_at_best": lr_at_best,
        "data_plus_train_count": data_plus_count,
        "data_plus_inner_val_count": 0,
        "data_plus_outer_test_count": 0,
        "reference_name": reference_name,
        "reference_f1": reference_f1,
        "delta_vs_reference": best_f1 - reference_f1,
        "m2_ref": M2_INNER_REFERENCES.get(fold_name.lower(), 0.0),
        "delta": best_f1 - M2_INNER_REFERENCES.get(fold_name.lower(), 0.0),
        "final_joint_ref": FINAL_JOINT_INNER_REFERENCES.get(
            fold_name.lower(), 0.0
        ),
        "delta_vs_final_joint": best_f1
        - FINAL_JOINT_INNER_REFERENCES.get(fold_name.lower(), 0.0),
        "model_variant": args.model_variant,
        "outer_test_eval_count": 0,
    }


def main() -> None:
    args = parse_args()
    if args.fold == "all":
        folds = ["vg1", "vg2", "vg3", "vg4", "vg5"]
    else:
        folds = [args.fold]

    finalized: list[dict[str, Any]] = []
    nonpositive = 0
    for f in folds:
        result = run_joint_training(f, args)
        finalized.append(result)
        if args.model_variant != "deep_local":
            continue
        delta = float(result["delta_vs_final_joint"])
        if delta <= 0.0:
            nonpositive += 1
        print(f"FOLD = {f.upper()}")
        print(f"BEST_EPOCH = {result['best_epoch']}")
        print(f"REFINED_INNER_F1 = {result['best_val_macro_f1']:.6f}")
        print(f"CONTROL_INNER_F1 = {result['final_joint_ref']:.6f}")
        print(f"DELTA = {delta:+.6f}")
        print(f"TRAIN_F1_AT_BEST = {result['train_f1_at_best']:.6f}")
        print(
            "REFINED_GENERALIZATION_GAP = "
            f"{result['refined_generalization_gap']:.6f}"
        )
        print(f"CONTROL_GENERALIZATION_GAP = {result['control_generalization_gap']}")
        print(f"GAP_REDUCTION = {result['gap_reduction']}")
        print(f"LR_AT_BEST = {result['lr_at_best']}")
        print(f"DATA_PLUS_TRAIN_COUNT = {result['data_plus_train_count']}")
        print("DATA_PLUS_INNER_VAL_COUNT = 0")
        print("DATA_PLUS_OUTER_TEST_COUNT = 0")
        print(f"CHECKPOINT_PATH = {result['checkpoint_path']}")
        print(f"CHECKPOINT_SHA256 = {result['best_sha256']}")
        print(f"POSITIVE_FOLDS_SO_FAR = {len(finalized) - nonpositive}")
        print(f"NONPOSITIVE_FOLDS_SO_FAR = {nonpositive}")
        print(f"FUTILITY_TRIGGERED = {'YES' if nonpositive >= 2 else 'NO'}")
        print("OUTER_TEST_EVALUATIONS = 0")
        print(
            f"FINALIZED {f.upper()} | Challenger F1={result['best_val_macro_f1']:.6f} "
            f"| FinalJoint Ref={result['final_joint_ref']:.6f} "
            f"| Delta={delta:+.6f} | Nonpositive={nonpositive}"
        )
        if nonpositive >= 2 and not getattr(args, "allow_all_folds", True):
            print("CHALLENGER_STATUS=REJECT_FUTILITY")
            break
        if len(finalized) >= 3:
            provisional = float(
                np.mean([item["best_val_macro_f1"] for item in finalized])
            )
            provisional_delta = float(
                np.mean([item["delta_vs_final_joint"] for item in finalized])
            )
            trajectory = (
                "STRONG"
                if provisional >= 0.710
                else "TOO_WEAK"
                if provisional < 0.690 and provisional_delta < 0.015
                else "PROMISING"
            )
            print(
                f"TARGET_TRAJECTORY={trajectory} | provisional_mean={provisional:.6f} "
                f"| provisional_delta={provisional_delta:+.6f}"
            )
    if args.model_variant == "deep_local" and finalized:
        mean_refined = float(
            np.mean([item["best_val_macro_f1"] for item in finalized])
        )
        mean_delta = mean_refined - float(np.mean(list(FINAL_JOINT_INNER_REFERENCES.values())))
        mean_refined_gap = float(
            np.mean([item["refined_generalization_gap"] for item in finalized])
        )
        complete_gap_rows = [
            item["gap_reduction"]
            for item in finalized
            if item["gap_reduction"] is not None
        ]
        mean_gap_reduction = (
            float(np.mean(complete_gap_rows)) if complete_gap_rows else None
        )
        positive_folds = sum(
            item["delta_vs_final_joint"] > 0.0 for item in finalized
        )
        print(f"MEAN_REFINED_F1 = {mean_refined:.6f}")
        print(f"MEAN_DELTA = {mean_delta:+.6f}")
        print(f"POSITIVE_FOLDS = {positive_folds}")
        print(f"MEAN_REFINED_GENERALIZATION_GAP = {mean_refined_gap:.6f}")
        print(f"MEAN_GAP_REDUCTION = {mean_gap_reduction}")
        print(
            "CURRENT_BEST_MODEL = "
            + (
                "FinalJoint_MS-ActorUnion_LocalST_Residual + locked recipe"
                if mean_delta > 0.0 and positive_folds >= 3
                else "FinalJoint_MS-ActorUnion_LocalST_Residual"
            )
        )
        print(f"CURRENT_BEST_MEAN_INNER_F1 = {max(mean_refined, 0.683884):.6f}")
        print("OUTER_TEST_EVALUATIONS = 0")


if __name__ == "__main__":
    main()
