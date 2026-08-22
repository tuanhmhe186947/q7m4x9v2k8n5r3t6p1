"""PyTorch sequence classifier for pig behavior bursts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pig_behavior.config import (
    BEHAVIOR_SEQUENCE_LABELS,
    BEHAVIOR_SEQUENCE_LENGTH,
)
from pig_behavior.models.sequence_architecture import (
    build_model as _build_model,
)
from pig_behavior.models.sequence_architecture import (
    checkpoint_state_dict as _checkpoint_state_dict,
)
from pig_behavior.models.sequence_architecture import (
    infer_model_spec as _infer_model_spec,
)
from pig_behavior.models.sequence_architecture import (
    labels_for_class_count as _labels_for_class_count,
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
