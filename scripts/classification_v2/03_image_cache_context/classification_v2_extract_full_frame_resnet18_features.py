"""Extract deterministic ResNet18 features from the full-frame cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.models.visual_backbones import (
    build_visual_frame_encoder,
)

LINEAGE_SCOPE = "legacy-only-unreviewed-development"
BACKBONE_NAME = "resnet18"
WEIGHT_ENUM = "ResNet18_Weights.IMAGENET1K_V1"
IMAGE_SIZE = 224
FEATURE_DIM = 512
SCHEMA = (
    "classification_v2.legacy_development_l6."
    "full_frame_resnet18_features.v1"
)


def extract_full_frame_features(
    *,
    packed_tensor: Path,
    packed_index: Path,
    packed_audit: Path,
    output_dir: Path,
    device_name: str,
    batch_size: int,
    checkpoint_every: int,
    overwrite: bool,
) -> dict[str, Any]:
    if batch_size <= 0 or checkpoint_every <= 0:
        raise ValueError("batch_size and checkpoint_every must be positive")
    packed_tensor = packed_tensor.resolve()
    packed_index = packed_index.resolve()
    packed_audit = packed_audit.resolve()
    output_dir = output_dir.resolve()
    source = _validate_source(packed_tensor, packed_index, packed_audit)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "full_frame_features_f32.npy"
    index_path = output_dir / "full_frame_feature_index.csv"
    audit_path = output_dir / "full_frame_feature_audit.json"
    if overwrite:
        for path in (feature_path, index_path, audit_path):
            path.unlink(missing_ok=True)
    if any(path.exists() for path in (feature_path, index_path, audit_path)):
        raise FileExistsError("full-frame feature output exists")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    torch.manual_seed(20260716)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260716)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    encoder, contract = build_visual_frame_encoder(BACKBONE_NAME, WEIGHT_ENUM)
    if contract.output_dim != FEATURE_DIM:
        raise ValueError("full-frame ResNet18 feature dimension drift")
    encoder.eval().to(device)
    weights_path = _weights_path()
    weights_hash = _sha256(weights_path)
    tensor = np.load(packed_tensor, mmap_mode="r")
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(source["rows"], FEATURE_DIM),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.inference_mode():
            for start in range(0, source["rows"], batch_size):
                end = min(start + batch_size, source["rows"])
                batch = torch.from_numpy(
                    np.array(tensor[start:end], copy=True)
                ).permute(0, 3, 1, 2)
                batch = batch.to(torch.float32).div_(255.0)
                mean = torch.tensor(
                    contract.input_mean,
                    dtype=torch.float32,
                ).view(1, 3, 1, 1)
                std = torch.tensor(
                    contract.input_std,
                    dtype=torch.float32,
                ).view(1, 3, 1, 1)
                encoded = encoder(((batch - mean) / std).to(device)).to("cpu")
                features[start:end] = encoded.numpy().astype(
                    np.float32,
                    copy=False,
                )
                if end % checkpoint_every == 0 or end == source["rows"]:
                    features.flush()
                del batch, encoded
    finally:
        features.flush()
        del features
        del tensor
        encoder.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    index = pd.read_csv(packed_index, low_memory=False)
    index.insert(1, "feature_row", np.arange(len(index), dtype=np.int64))
    index["control_id"] = "FULL_FRAME_CONTEXT"
    index["backbone_name"] = BACKBONE_NAME
    index["pretrained_weight_enum"] = WEIGHT_ENUM
    index["image_size"] = IMAGE_SIZE
    index["feature_dim"] = FEATURE_DIM
    index["feature_dtype"] = "float32"
    index.to_csv(index_path, index=False)
    audit = {
        "schema_version": SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_FULL_FRAME_RESNET18_FEATURES",
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": "legacy_16f",
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "source_tensor_sha256": source["tensor_sha256"],
        "source_index_sha256": source["index_sha256"],
        "source_audit_sha256": _sha256(packed_audit),
        "source_media_reads": 0,
        "outer_holdout_rows": 0,
        "backbone_name": BACKBONE_NAME,
        "pretrained_weight_enum": WEIGHT_ENUM,
        "weights_sha256": weights_hash,
        "weights_path": str(weights_path),
        "image_size": IMAGE_SIZE,
        "feature_dim": FEATURE_DIM,
        "feature_dtype": "float32",
        "rows": source["rows"],
        "device": str(device),
        "batch_size": batch_size,
        "checkpoint_every": checkpoint_every,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "feature_tensor_sha256": _sha256(feature_path),
        "feature_index_sha256": _sha256(index_path),
        "feature_tensor_path": str(feature_path),
        "feature_index_path": str(index_path),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "errors": [],
        "valid": True,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return audit


def _validate_source(
    tensor_path: Path,
    index_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("valid") is not True:
        raise ValueError("full-frame cache audit is not valid")
    tensor = np.load(tensor_path, mmap_mode="r")
    index = pd.read_csv(index_path, low_memory=False)
    rows = int(audit["packed_rows"])
    if tuple(tensor.shape) != (rows, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError("full-frame source tensor shape drift")
    if tensor.dtype != np.uint8 or len(index) != rows:
        raise ValueError("full-frame source dtype/index drift")
    if index["packed_row"].astype(int).tolist() != list(range(rows)):
        raise ValueError("full-frame packed row mapping drift")
    return {
        "rows": rows,
        "tensor_sha256": _sha256(tensor_path),
        "index_sha256": _sha256(index_path),
    }


def _weights_path() -> Path:
    path = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
    if not path.is_file():
        raise FileNotFoundError(f"pretrained weight missing={path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-tensor", type=Path, required=True)
    parser.add_argument("--packed-index", type=Path, required=True)
    parser.add_argument("--packed-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    audit = extract_full_frame_features(
        packed_tensor=args.packed_tensor,
        packed_index=args.packed_index,
        packed_audit=args.packed_audit,
        output_dir=args.output_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        overwrite=args.overwrite,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
