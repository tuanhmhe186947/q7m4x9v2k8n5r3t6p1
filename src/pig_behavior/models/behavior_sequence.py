"""PyTorch sequence classifier for pig behavior bursts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pig_behavior.config import (
    BEHAVIOR_SEQUENCE_FEATURES,
    BEHAVIOR_SEQUENCE_LABELS,
    BEHAVIOR_SEQUENCE_LENGTH,
    COARSE_LABELS,
)


@dataclass(slots=True)
class BehaviorFrameSample:
    """One cropped pig frame plus tabular context features."""

    crop_rgb: np.ndarray
    features: list[float]
    frame_index: int
    bbox_xyxy: list[float]


@dataclass(slots=True)
class BehaviorPrediction:
    """Sequence-level behavior prediction."""

    label: str
    confidence: float
    scores: dict[str, float]


class BehaviorSequenceClassifier:
    """Lazy loader for the notebook-trained behavior sequence model."""

    def __init__(self, model_path: Path, device: str | None = None) -> None:
        self.model_path = model_path
        self.device_name = device
        self._torch = None
        self._model = None
        self._device = None
        self._labels: list[str] = BEHAVIOR_SEQUENCE_LABELS

    @property
    def labels(self) -> list[str]:
        """Return labels inferred from the checkpoint."""
        return self._labels

    @property
    def loaded(self) -> bool:
        """Return whether the model has been loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load the checkpoint if needed."""
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Behavior classifier not found: {self.model_path}"
            )

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Behavior sequence classification requires torch and torchvision. "
                "Install with `pip install -e .[pt]`."
            ) from exc

        try:
            raw = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            raw = torch.load(self.model_path, map_location="cpu")
        state = _checkpoint_state_dict(raw)
        spec = _infer_model_spec(state)
        self._labels = _labels_for_class_count(spec["num_classes"])

        device_name = self.device_name or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._device = torch.device(device_name)
        self._torch = torch
        self._model = _build_model(
            num_behaviors=spec["num_classes"],
            extra_dim=spec["extra_dim"],
            d_model=spec["d_model"],
            num_layers=spec["num_layers"],
            backbone_name=spec["backbone_name"],
        )
        self._model.load_state_dict(state, strict=True)
        self._model.to(self._device)
        self._model.eval()

    def predict(self, samples: list[BehaviorFrameSample]) -> BehaviorPrediction:
        """Run one sequence prediction."""
        self.load()
        assert self._torch is not None
        assert self._model is not None
        assert self._device is not None

        samples = _pad_or_trim(samples, BEHAVIOR_SEQUENCE_LENGTH)
        image_tensors = [_crop_to_tensor(sample.crop_rgb) for sample in samples]
        feature_rows = [sample.features for sample in samples]

        seq_imgs = self._torch.stack(image_tensors, dim=0).unsqueeze(0)
        seq_feats = self._torch.tensor(feature_rows, dtype=self._torch.float32)
        seq_feats = seq_feats.unsqueeze(0)

        seq_imgs = seq_imgs.to(self._device)
        seq_feats = seq_feats.to(self._device)

        with self._torch.inference_mode():
            logits = self._model(seq_imgs, seq_feats)
            probs = self._torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        scores = {
            label: float(probability)
            for label, probability in zip(self._labels, probs, strict=False)
        }
        label, confidence = max(scores.items(), key=lambda item: item[1])
        return BehaviorPrediction(label=label, confidence=confidence, scores=scores)


def _checkpoint_state_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict) and "model_state" in raw:
        state = raw["model_state"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw

    if not isinstance(state, dict):
        raise ValueError("Unsupported behavior checkpoint format.")

    return {
        key.removeprefix("module."): value
        for key, value in state.items()
    }


def _infer_model_spec(state: dict[str, Any]) -> dict[str, Any]:
    head_weight = state.get("head.3.weight")
    if head_weight is None:
        raise ValueError("Checkpoint is missing head.3.weight.")

    cnn_proj_weight = state.get("cnn_proj.weight")
    if cnn_proj_weight is None:
        raise ValueError("Checkpoint is missing cnn_proj.weight.")

    d_model = int(state["head.0.weight"].shape[0])
    base_dim = int(cnn_proj_weight.shape[0])
    extra_dim = d_model - base_dim
    if extra_dim <= 0:
        extra_dim = len(BEHAVIOR_SEQUENCE_FEATURES)

    layer_indices = [
        int(key.split(".")[2])
        for key in state
        if key.startswith("transformer.layers.")
        and key.split(".")[2].isdigit()
    ]
    num_layers = max(layer_indices) + 1 if layer_indices else 2
    backbone_name = (
        "resnet34"
        if any(
            key.startswith("cnn.6.5.") or key.startswith("cnn.7.2.")
            for key in state
        )
        else "resnet18"
    )

    return {
        "num_classes": int(head_weight.shape[0]),
        "d_model": d_model,
        "extra_dim": extra_dim,
        "num_layers": num_layers,
        "backbone_name": backbone_name,
    }


def _labels_for_class_count(num_classes: int) -> list[str]:
    if num_classes == len(BEHAVIOR_SEQUENCE_LABELS):
        return BEHAVIOR_SEQUENCE_LABELS
    if num_classes == len(COARSE_LABELS):
        return COARSE_LABELS
    return [f"class_{index}" for index in range(num_classes)]


def _build_model(
    *,
    num_behaviors: int,
    extra_dim: int,
    d_model: int,
    num_layers: int,
    backbone_name: str,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models

    class CBAMBlock(nn.Module):
        def __init__(self, channels: int, reduction: int = 16) -> None:
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels, bias=False),
            )
            self.conv_spatial = nn.Conv2d(
                2,
                1,
                kernel_size=7,
                padding=3,
                bias=False,
            )

        def forward(self, x):
            batch, channels, _height, _width = x.shape
            avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch, channels)
            max_pool = F.adaptive_max_pool2d(x, 1).view(batch, channels)
            ca = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
            x = x * ca.view(batch, channels, 1, 1)

            avg_map = torch.mean(x, dim=1, keepdim=True)
            max_map, _ = torch.max(x, dim=1, keepdim=True)
            sa = torch.cat([avg_map, max_map], dim=1)
            sa = torch.sigmoid(self.conv_spatial(sa))
            return x * sa

    class BehaviorTransformerNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if backbone_name == "resnet18":
                backbone = models.resnet18(weights=None)
            elif backbone_name == "resnet34":
                backbone = models.resnet34(weights=None)
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}")

            self.cnn = nn.Sequential(*list(backbone.children())[:-2])
            cnn_out_channels = backbone.fc.in_features
            self.cbam = CBAMBlock(cnn_out_channels)

            base_dim = d_model - extra_dim
            if base_dim <= 0:
                raise ValueError("d_model must be greater than extra_dim.")
            self.cnn_proj = nn.Linear(cnn_out_channels, base_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 4,
                dropout=0.2,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
            self.attn_fc = nn.Linear(d_model, 1)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(d_model, num_behaviors),
            )

        def forward(self, seq_imgs, seq_feats):
            batch, steps, channels, height, width = seq_imgs.shape
            x = seq_imgs.reshape(batch * steps, channels, height, width)
            feat_map = self.cnn(x)
            feat_map = self.cbam(feat_map)
            feat_vec = F.adaptive_avg_pool2d(feat_map, 1).view(batch * steps, -1)
            feat_vec = self.cnn_proj(feat_vec).view(batch, steps, -1)

            h = torch.cat([feat_vec, seq_feats.to(feat_vec.dtype)], dim=-1)
            h_enc = self.transformer(h)
            attn_score = self.attn_fc(h_enc).squeeze(-1)
            attn_weight = torch.softmax(attn_score, dim=1).unsqueeze(-1)
            z = torch.sum(h_enc * attn_weight, dim=1)
            return self.head(z)

    return BehaviorTransformerNet()


def _crop_to_tensor(crop_rgb: np.ndarray):
    import torch
    from PIL import Image

    image = Image.fromarray(crop_rgb.astype("uint8"), mode="RGB")
    image = image.resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def _pad_or_trim(
    samples: list[BehaviorFrameSample],
    target_length: int,
) -> list[BehaviorFrameSample]:
    if not samples:
        raise ValueError("At least one behavior frame sample is required.")

    output = list(samples[:target_length])
    while len(output) < target_length:
        output.append(output[-1])
    return output
