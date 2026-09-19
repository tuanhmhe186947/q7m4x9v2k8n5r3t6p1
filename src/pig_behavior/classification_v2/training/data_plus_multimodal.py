"""Strict sample-key loading for DATA+ multimodal training sidecars."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.data_module import (
    StrictTrainingBatch,
)

T6 = 6
H5 = 5
STRUCTURED_DIM = 46
SPATIAL_SLICES: dict[str, slice] = {
    "bbox_xywh_n": slice(0, 4),
    "bbox_shape_n": slice(4, 6),
    "motion_delta": slice(6, 18),
    "roi_class_relation": slice(18, 36),
    "social_relation": slice(36, 46),
}


def make_data_plus_sample_key(supplemental_unit_id: str) -> str:
    """Return the PB-compatible stable identity for one supplemental unit."""

    value = str(supplemental_unit_id).strip()
    if not value:
        raise ValueError("supplemental_unit_id must be non-empty")
    return f"data_plus_{value}"


@dataclass(frozen=True, slots=True)
class DataPlusSidecarPaths:
    """Versioned paths for the six keyed DATA+ modality authorities."""

    rgb: Path
    spatial: Path
    h5: Path
    roi: Path
    posture: Path
    pb: Path

    @classmethod
    def from_root(
        cls,
        root: Path,
        *,
        pb_path: Path | None = None,
    ) -> DataPlusSidecarPaths:
        base = Path(root)
        return cls(
            rgb=base / "data_plus_rgb_sidecar.npz",
            spatial=base / "data_plus_spatial_46d_sidecar.npz",
            h5=base / "data_plus_h5_sidecar.npz",
            roi=base / "data_plus_roi_sidecar.npz",
            posture=base / "data_plus_posture_sidecar.npz",
            pb=(pb_path or base / "data_plus_pb_sidecar.pt"),
        )


@dataclass(slots=True)
class AlignedDataPlusBatch:
    """One Joint-ready DATA+ batch plus non-model auxiliary targets."""

    sample_keys: tuple[str, ...]
    training_batch: StrictTrainingBatch
    context_inputs: dict[str, torch.Tensor]
    posture_targets: torch.Tensor
    posture_reviewed_mask: torch.Tensor
    source_keys: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class _Identity:
    sample_key: tuple[str, ...]
    supplemental_unit_id: tuple[str, ...]
    target_object_track_key: tuple[str, ...]


class DataPlusMultimodalStore:
    """Load every DATA+ branch through one strict sample-key authority."""

    def __init__(self, paths: DataPlusSidecarPaths) -> None:
        self.paths = paths
        self.rgb = _load_npz(paths.rgb, "RGB")
        self.spatial = _load_npz(paths.spatial, "46D")
        self.h5 = _load_npz(paths.h5, "H5")
        self.roi = _load_npz(paths.roi, "ROI")
        self.posture = _load_npz(paths.posture, "Posture")
        self.pb = _load_pb(paths.pb)

        self.identity = _identity_from(self.rgb, "RGB")
        _require_unique(self.identity.sample_key, "RGB")
        for name, sidecar in (
            ("46D", self.spatial),
            ("H5", self.h5),
            ("ROI", self.roi),
            ("Posture", self.posture),
        ):
            observed = _identity_from(sidecar, name)
            _require_unique(observed.sample_key, name)
            if observed != self.identity:
                raise ValueError(f"{name} sidecar identity mismatch")

        self._positions = {
            key: position for position, key in enumerate(self.identity.sample_key)
        }
        self._pb_positions = _pb_positions(self.pb)
        self._validate_tensor_contracts()

    def _validate_tensor_contracts(self) -> None:
        n_rows = len(self.identity.sample_key)
        _shape(self.rgb, "feature_tensor", (n_rows, 2, T6, None, None, 3))
        _shape(self.rgb, "validity_mask", (n_rows, 2, T6))
        _shape(self.rgb, "time_delta_seconds", (n_rows, T6))
        _shape(
            self.spatial,
            "feature_tensor",
            (n_rows, T6, STRUCTURED_DIM),
        )
        _shape(self.spatial, "validity_mask", (n_rows, T6))
        _shape(self.h5, "feature_tensor", (n_rows, H5, STRUCTURED_DIM))
        _shape(self.h5, "validity_mask", (n_rows, H5))
        _shape(self.roi, "feature_tensor", (n_rows, T6, 18))
        _shape(self.roi, "validity_mask", (n_rows, T6, 3))
        _shape(self.posture, "feature_tensor", (n_rows,))
        _shape(self.posture, "validity_mask", (n_rows,))
        _shape(self.spatial, "roi_validity_mask", (n_rows, T6, 3))
        _shape(
            self.spatial,
            "social_feature_validity_mask",
            (n_rows, T6, 10),
        )
        _shape(
            self.spatial,
            "motion_feature_validity_mask",
            (n_rows, T6, 12),
        )
        if not np.array_equal(
            self.roi["feature_tensor"],
            self.spatial["feature_tensor"][:, :, SPATIAL_SLICES["roi_class_relation"]],
        ):
            raise ValueError("ROI feature tensor does not match keyed structured46D")
        if not np.array_equal(
            self.roi["validity_mask"],
            self.spatial["roi_validity_mask"],
        ):
            raise ValueError("ROI validity does not match keyed structured46D")
        labels = _strings(self.spatial.get("class_label"), "46D.class_label")
        invalid = sorted(set(labels).difference(VALID_BEHAVIORS))
        if invalid:
            raise ValueError(f"invalid DATA+ behavior labels: {invalid}")

    @property
    def sample_keys(self) -> tuple[str, ...]:
        return self.identity.sample_key

    def batch(
        self,
        sample_keys: Sequence[str],
        *,
        device: torch.device,
        spatial_transform: Callable[
            [dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor],
            dict[str, torch.Tensor],
        ]
        | None = None,
    ) -> AlignedDataPlusBatch:
        """Resolve one batch by exact key; unknown or non-string keys fail."""

        keys = _requested_keys(sample_keys)
        positions: list[int] = []
        for key in keys:
            if key not in self._positions:
                raise KeyError(f"unknown DATA+ sample_key: {key}")
            positions.append(self._positions[key])
        missing_pb = [key for key in keys if key not in self._pb_positions]
        if missing_pb:
            raise KeyError(
                "PB sidecar has no fold-eligible DATA+ entry for "
                f"{missing_pb[:5]}"
            )
        pb_positions = [self._pb_positions[key] for key in keys]
        index = np.asarray(positions, dtype=np.int64)

        rgb = torch.from_numpy(
            np.ascontiguousarray(self.rgb["feature_tensor"][index])
        ).permute(0, 1, 2, 5, 3, 4).float().div_(255.0).to(device)
        rgb_valid = torch.from_numpy(
            np.ascontiguousarray(self.rgb["validity_mask"][index])
        ).bool().to(device)
        structured = torch.from_numpy(
            np.ascontiguousarray(self.spatial["feature_tensor"][index])
        ).float().to(device)
        spatial_valid = torch.from_numpy(
            np.ascontiguousarray(self.spatial["validity_mask"][index])
        ).bool().to(device)
        motion_valid = torch.from_numpy(
            np.ascontiguousarray(
                self.spatial["motion_feature_validity_mask"][index]
            )
        ).bool().to(device)
        social_valid = torch.from_numpy(
            np.ascontiguousarray(
                self.spatial["social_feature_validity_mask"][index]
            )
        ).bool().to(device)
        time_delta = torch.from_numpy(
            np.ascontiguousarray(self.rgb["time_delta_seconds"][index])
        ).float().to(device)

        spatial_features = {
            name: structured[:, :, feature_slice]
            for name, feature_slice in SPATIAL_SLICES.items()
        }
        if "roi_class_relation" in spatial_features:
            roi_clean = spatial_features["roi_class_relation"].clone()
            roi_clean[..., 17] = 0.0
            spatial_features["roi_class_relation"] = roi_clean
        spatial_feature_validity = {
            "motion_delta": motion_valid,
            "social_relation": social_valid,
        }
        if spatial_transform is not None:
            spatial_features = spatial_transform(
                spatial_features,
                spatial_feature_validity,
                spatial_valid,
            )

        interaction_available = social_valid[:, -1].any(dim=-1, keepdim=True)
        model_inputs: dict[str, Any] = {
            "image": rgb[:, 0],
            "length_mask": rgb_valid[:, 0],
            "image_length_mask": rgb_valid[:, 0],
            "image_observed_mask": rgb_valid[:, 0],
            "image_available_mask": rgb_valid[:, 0],
            "image_quality_mask": rgb_valid[:, 0],
            "image_time_delta": time_delta,
            "spatial_features": spatial_features,
            "spatial_length_mask": spatial_valid,
            "spatial_observed_mask": spatial_valid,
            "spatial_available_mask": spatial_valid,
            "spatial_quality_mask": spatial_valid,
            "spatial_feature_validity_masks": spatial_feature_validity,
            "spatial_time_delta": time_delta,
            "interaction_context_features": spatial_features[
                "social_relation"
            ][:, -1, :5],
            "interaction_context_available_mask": interaction_available,
            "interaction_context_quality_mask": interaction_available,
            "visual_context_image": rgb[:, 1],
            "visual_context_length_mask": rgb_valid[:, 0],
            "visual_context_observed_mask": rgb_valid[:, 1],
            "visual_context_available_mask": rgb_valid[:, 1],
            "visual_context_quality_mask": rgb_valid[:, 1],
            "visual_context_time_delta": time_delta,
        }
        labels = np.asarray(self.spatial["class_label"])[index].astype(str)
        label_to_index = {
            label: position for position, label in enumerate(VALID_BEHAVIORS)
        }
        target = torch.tensor(
            [label_to_index[label] for label in labels],
            dtype=torch.long,
            device=device,
        )
        training_batch = StrictTrainingBatch(
            model_inputs=model_inputs,
            behavior_target=target,
            auxiliary_targets={},
            auxiliary_masks={},
            sample_weight=torch.ones(len(keys), dtype=torch.float32, device=device),
            metadata={
                "sample_key": list(keys),
                "supplemental_unit_id": [
                    self.identity.supplemental_unit_id[pos] for pos in positions
                ],
                "target_object_track_key": [
                    self.identity.target_object_track_key[pos] for pos in positions
                ],
                "source_type": ["data_plus_supplemental"] * len(keys),
            },
        )
        context_inputs = {
            "partner_behavior_probs": _pb_tensor(
                self.pb,
                "partner_probs",
                pb_positions,
                device,
            ),
            "partner_behavior_hidden": _pb_tensor(
                self.pb,
                "partner_hidden",
                pb_positions,
                device,
            ),
            "partner_behavior_mask": _pb_tensor(
                self.pb,
                "partner_mask",
                pb_positions,
                device,
            ).bool(),
            "h5_structured": (
                _np_tensor(self.h5["feature_tensor"], index, device)
                .float()
                .clone()
            ),
            "h5_mask": _np_tensor(
                self.h5["validity_mask"], index, device
            ).bool(),
            "roi_sequence": (
                _np_tensor(self.roi["feature_tensor"], index, device)
                .float()
                .clone()
            ),
            "roi_validity": _np_tensor(
                self.roi["validity_mask"], index, device
            ).bool(),
        }
        context_inputs["h5_structured"][..., 35] = 0.0
        context_inputs["roi_sequence"][..., 17] = 0.0
        posture_targets = _np_tensor(
            self.posture["feature_tensor"], index, device
        ).long()
        posture_mask = _np_tensor(
            self.posture["validity_mask"], index, device
        ).bool()
        source_keys = {
            name: keys for name in ("rgb", "spatial", "h5", "roi", "posture", "pb")
        }
        return AlignedDataPlusBatch(
            sample_keys=keys,
            training_batch=training_batch,
            context_inputs=context_inputs,
            posture_targets=posture_targets,
            posture_reviewed_mask=posture_mask,
            source_keys=source_keys,
        )


def _load_npz(path: Path, label: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing DATA+ {label} sidecar: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _load_pb(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing DATA+ PB sidecar: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("PB sidecar must contain a mapping")
    return payload


def _identity_from(payload: dict[str, Any], label: str) -> _Identity:
    return _Identity(
        sample_key=_strings(payload.get("sample_key"), f"{label}.sample_key"),
        supplemental_unit_id=_strings(
            payload.get("supplemental_unit_id"),
            f"{label}.supplemental_unit_id",
        ),
        target_object_track_key=_strings(
            payload.get("target_object_track_key"),
            f"{label}.target_object_track_key",
        ),
    )


def _strings(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        raise ValueError(f"missing identity field {name}")
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    result = tuple(str(value) for value in array.tolist())
    if any(not value for value in result):
        raise ValueError(f"{name} contains an empty value")
    return result


def _require_unique(keys: tuple[str, ...], label: str) -> None:
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} sidecar contains duplicate sample_key")


def _pb_positions(payload: dict[str, Any]) -> dict[str, int]:
    keys = _strings(payload.get("sample_key"), "PB.sample_key")
    _require_unique(keys, "PB")
    positions: dict[str, int] = {}
    for position, key in enumerate(keys):
        positions[key] = position
        if key.startswith("data_plus_"):
            positions[f"data_plus_{key}"] = position
        elif not key.startswith("data_plus_"):
            positions[f"data_plus_{key}"] = position
    for name, tail in (
        ("partner_probs", (2, 10)),
        ("partner_hidden", (2, 256)),
        ("partner_mask", (2,)),
    ):
        value = payload.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"PB.{name} must be a tensor")
        if tuple(value.shape) != (len(keys), *tail):
            raise ValueError(f"PB.{name} shape mismatch")
    return positions


def _requested_keys(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError("sample_key must be a string")
        result.append(value)
    if not result:
        raise ValueError("at least one sample_key is required")
    if len(result) != len(set(result)):
        raise ValueError("requested sample_key values must be unique")
    return tuple(result)


def _shape(
    payload: dict[str, Any],
    name: str,
    expected: tuple[int | None, ...],
) -> None:
    if name not in payload:
        raise ValueError(f"sidecar missing {name}")
    actual = tuple(np.asarray(payload[name]).shape)
    if len(actual) != len(expected) or any(
        wanted is not None and observed != wanted
        for observed, wanted in zip(actual, expected, strict=True)
    ):
        raise ValueError(f"{name} shape mismatch: {actual} != {expected}")


def _np_tensor(
    value: np.ndarray,
    positions: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value[positions])).to(device)


def _pb_tensor(
    payload: dict[str, Any],
    name: str,
    positions: list[int],
    device: torch.device,
) -> torch.Tensor:
    index = torch.tensor(positions, dtype=torch.long)
    return payload[name].index_select(0, index).to(device)


__all__ = [
    "AlignedDataPlusBatch",
    "DataPlusMultimodalStore",
    "DataPlusSidecarPaths",
    "SPATIAL_SLICES",
    "make_data_plus_sample_key",
]
