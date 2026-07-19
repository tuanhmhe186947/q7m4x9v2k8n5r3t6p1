"""Build a hash-bound legacy C6 source packet from the clean rebuild lineage."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    letterbox_rgb_uint8,
)
from pig_behavior.classification_v2.models.visual_backbones import (
    build_visual_frame_encoder,
)
from pig_behavior.classification_v2.training.legacy_c6_prepared_source import (
    PACKET_SCHEMA,
    prepare_legacy_c6_tables,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

BACKBONE_NAME = "resnet18"
WEIGHT_ENUM = "ResNet18_Weights.IMAGENET1K_V1"
IMAGE_SIZE = 224
FEATURE_DIM = 512


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-export", type=Path, required=True)
    parser.add_argument("--source-selection-audit", type=Path, required=True)
    parser.add_argument("--harmonized-frames", type=Path, required=True)
    parser.add_argument("--native-units", type=Path, required=True)
    parser.add_argument("--native-folds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-fold-id", default="native_oof_005")
    parser.add_argument("--validation-fold-id", default="native_oof_006")
    parser.add_argument(
        "--train-selection-policy",
        choices=("fixed_per_class", "all_eligible"),
        default="fixed_per_class",
    )
    parser.add_argument("--train-units-per-class", type=int, default=8)
    parser.add_argument(
        "--selection-salt",
        default="legacy_c6_rebuild_20260719_v1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    packet = build_legacy_c6_prepared_source(
        canonical_export=args.canonical_export,
        source_selection_audit=args.source_selection_audit,
        harmonized_frames=args.harmonized_frames,
        native_units=args.native_units,
        native_folds=args.native_folds,
        output_dir=args.output_dir,
        outer_fold_id=args.outer_fold_id,
        validation_fold_id=args.validation_fold_id,
        train_units_per_class=(
            args.train_units_per_class
            if args.train_selection_policy == "fixed_per_class"
            else None
        ),
        train_selection_policy=args.train_selection_policy,
        selection_salt=args.selection_salt,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(packet, indent=2, sort_keys=True))


def build_legacy_c6_prepared_source(
    *,
    canonical_export: Path,
    source_selection_audit: Path,
    harmonized_frames: Path,
    native_units: Path,
    native_folds: Path,
    output_dir: Path,
    outer_fold_id: str,
    validation_fold_id: str,
    train_units_per_class: int | None,
    train_selection_policy: str,
    selection_salt: str,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    """Select units, read only model-visible crops, and freeze V1 features."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    inputs = {
        "canonical_export": canonical_export.resolve(),
        "source_selection_audit": source_selection_audit.resolve(),
        "harmonized_frames": harmonized_frames.resolve(),
        "native_units": native_units.resolve(),
        "native_folds": native_folds.resolve(),
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"legacy C6 input missing {name}: {path}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    repo_root = Path.cwd().resolve()
    frames = pd.read_csv(inputs["harmonized_frames"], low_memory=False)
    units = pd.read_csv(inputs["native_units"], low_memory=False)
    folds = pd.read_csv(inputs["native_folds"], low_memory=False)
    selected = prepare_legacy_c6_tables(
        frames,
        units,
        folds,
        outer_fold_id=outer_fold_id,
        validation_fold_id=validation_fold_id,
        train_units_per_class=train_units_per_class,
        train_selection_policy=train_selection_policy,
        selection_salt=selection_salt,
    )
    units_path = output_dir / "selected_native_units.csv"
    frames_path = output_dir / "selected_harmonized_frames.csv"
    tensor_path = output_dir / "actor_resnet18_features_f32.npy"
    index_path = output_dir / "actor_resnet18_feature_index.csv"
    packet_path = output_dir / "prepared_source_manifest.json"
    selected.units.to_csv(units_path, index=False, lineterminator="\n")
    selected.frames.to_csv(frames_path, index=False, lineterminator="\n")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    _seed_runtime(device)
    encoder, contract = build_visual_frame_encoder(BACKBONE_NAME, WEIGHT_ENUM)
    if contract.output_dim != FEATURE_DIM:
        raise RuntimeError("legacy C6 actor feature dimension drift")
    encoder.eval().to(device)
    output = np.lib.format.open_memmap(
        tensor_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(selected.frames), FEATURE_DIM),
    )
    mean = torch.tensor(
        contract.input_mean,
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        contract.input_std,
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)
    media_reads = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.inference_mode():
            for start in range(0, len(selected.frames), batch_size):
                end = min(start + batch_size, len(selected.frames))
                images: list[np.ndarray] = []
                for crop_value in selected.frames.iloc[start:end]["crop_path"]:
                    crop_path = Path(str(crop_value))
                    if not crop_path.is_absolute():
                        crop_path = repo_root / crop_path
                    if not crop_path.is_file():
                        raise FileNotFoundError(crop_path)
                    with Image.open(crop_path) as image:
                        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                    images.append(letterbox_rgb_uint8(rgb, IMAGE_SIZE))
                    media_reads += 1
                batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
                batch = batch.to(device=device, dtype=torch.float32).div_(255.0)
                encoded = encoder((batch - mean) / std)
                values = encoded.detach().cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
                if values.shape != (end - start, FEATURE_DIM):
                    raise RuntimeError("legacy C6 actor feature batch shape drift")
                if not np.isfinite(values).all():
                    raise ValueError("legacy C6 actor features contain nonfinite values")
                output[start:end] = values
                del batch, encoded, values, images
        output.flush()
    finally:
        output.flush()
        _close_memmap(output)
        encoder.to("cpu")
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if media_reads != len(selected.frames):
        raise RuntimeError("legacy C6 actor media read count drift")
    feature_index_columns = [
        "feature_row",
        "position",
        "temporal_unit_key",
        "slot_index",
        "timestamp_sec",
        "frame_uid",
        "scene_frame_uid",
        "image_context_id",
        "crop_path",
        "l5_role",
        "recording_group_id",
        "behavior_label",
        "lineage_scope",
        "human_review_complete",
    ]
    index = selected.frames[feature_index_columns].copy()
    index["backbone_name"] = BACKBONE_NAME
    index["pretrained_weight_enum"] = WEIGHT_ENUM
    index["image_size"] = IMAGE_SIZE
    index["feature_dim"] = FEATURE_DIM
    index["feature_dtype"] = "float32"
    index.to_csv(index_path, index=False, lineterminator="\n")
    peak_vram = 0
    if device.type == "cuda":
        peak_vram = int(torch.cuda.max_memory_allocated(device))
    weights_path = _weights_path()
    artifacts = {
        "selected_native_units": _artifact_spec(units_path, repo_root),
        "selected_frames": _artifact_spec(frames_path, repo_root),
        "actor_feature_tensor": _artifact_spec(tensor_path, repo_root),
        "actor_feature_index": _artifact_spec(index_path, repo_root),
    }
    packet = {
        "schema_version": PACKET_SCHEMA,
        "status": "PASS_LEGACY_C6_PREPARED_SOURCE",
        "lineage_scope": "legacy-only-unreviewed-development",
        "human_review_complete": False,
        "review_status": (
            "operator_cvat_checked_pending_hidden_behavior_double_check"
        ),
        "canonical_source_name": "legacy_16f_rebuild_20260718_v2",
        "inputs": {
            name: _artifact_spec(path, repo_root)
            for name, path in inputs.items()
        },
        "artifacts": artifacts,
        "selection_audit": selected.audit,
        "outer_fold_id": outer_fold_id,
        "validation_fold_id": validation_fold_id,
        "train_native_units": selected.audit["train_native_units"],
        "validation_native_units": selected.audit["validation_native_units"],
        "train_selection_policy": selected.audit["train_selection_policy"],
        "train_units_per_class": selected.audit["train_units_per_class"],
        "model_visible_native_units": selected.audit[
            "model_visible_native_units"
        ],
        "model_visible_frame_rows": selected.audit["model_visible_frame_rows"],
        "actor_feature_control": "V1_resnet18_224_imagenet1k_v1",
        "backbone_name": BACKBONE_NAME,
        "pretrained_weight_enum": WEIGHT_ENUM,
        "weights_path": str(weights_path),
        "weights_sha256": file_sha256(weights_path),
        "image_size": IMAGE_SIZE,
        "resize_policy": "letterbox_preserve_aspect_rgb_pad_black_v1",
        "feature_dim": FEATURE_DIM,
        "feature_dtype": "float32",
        "source_media_reads": media_reads,
        "outer_holdout_metadata_units_read": selected.audit[
            "outer_metadata_units_read"
        ],
        "outer_holdout_media_reads": 0,
        "outer_holdout_features_created": 0,
        "outer_holdout_predictions_created": 0,
        "device": str(device),
        "batch_size": batch_size,
        "peak_vram_bytes": peak_vram,
        "builder_sha256": file_sha256(Path(__file__).resolve()),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "errors": [],
        "valid": True,
    }
    packet_path.write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {**packet, "packet_path": str(packet_path)}


def _seed_runtime(device: torch.device) -> None:
    torch.manual_seed(20260719)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260719)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _weights_path() -> Path:
    path = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
    if not path.is_file():
        raise FileNotFoundError(f"pretrained weight file missing: {path}")
    return path


def _artifact_spec(path: Path, repo_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(repo_root))
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "sha256": file_sha256(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _close_memmap(array: np.ndarray) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


if __name__ == "__main__":
    main()
