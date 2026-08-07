"""Run one local loader-to-backward smoke on a current H5 feature bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as functional

from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_PREDICTIVE_FEATURES,
    SPATIAL_PREDICTIVE_GROUP_NAMES,
    load_current_spatial_tensor_bundle,
)
from pig_behavior.classification_v2.models.spatial_tcn import (
    SpatialTCNClassifier,
    SpatialTCNConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle_root
    arrays, audit = load_current_spatial_tensor_bundle(
        root / "X_spatial_sequences.npz",
        root / "spatial_sequence_audit.json",
    )
    cohort = pd.read_csv(root / "common_h5_matched_cohort.csv", low_memory=False)
    windows = pd.read_csv(root / "h5_window_manifest.csv", low_memory=False)
    if len(cohort) != len(windows) or len(windows) != arrays["length_mask"].shape[0]:
        raise ValueError("H5 loader row count does not match cohort/window authority")
    if not windows["window_id"].eq(cohort["h5_target_id"]).all():
        raise ValueError("H5 loader ordering does not match the matched cohort")
    selected: list[int] = []
    for source in ("cvat_tracking_xml", "legacy_recovered"):
        positions = cohort.index[cohort["source_type"].eq(source)].tolist()
        if positions:
            selected.append(int(positions[0]))
    if len(selected) != 2:
        raise ValueError("H5 loader smoke requires both CVAT and legacy rows")
    index = torch.tensor(selected, dtype=torch.long)
    model = SpatialTCNClassifier(
        SpatialTCNConfig(
            input_dims={
                group: len(SPATIAL_PREDICTIVE_FEATURES[group])
                for group in SPATIAL_PREDICTIVE_GROUP_NAMES
            },
            num_classes=10,
            hidden_dim=8,
            dropout=0.0,
        )
    )
    inputs = {
        group: torch.from_numpy(arrays[group]).index_select(0, index)
        for group in SPATIAL_PREDICTIVE_GROUP_NAMES
    }
    logits = model(
        inputs,
        length_mask=torch.from_numpy(arrays["length_mask"]).index_select(0, index),
        observed_mask=torch.from_numpy(arrays["observed_mask"]).index_select(0, index),
        feature_validity_masks={
            "motion_delta": torch.from_numpy(
                arrays["motion_feature_validity_mask"]
            ).index_select(0, index),
            "social_relation": torch.from_numpy(
                arrays["social_feature_validity_mask"]
            ).index_select(0, index),
        },
    )
    loss = functional.cross_entropy(logits, torch.zeros(len(selected), dtype=torch.long))
    loss.backward()
    if not torch.isfinite(loss):
        raise ValueError("H5 loader smoke produced non-finite loss")
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_sources": cohort.loc[selected, "source_type"].tolist(),
                "selected_rows": selected,
                "logits_shape": list(logits.shape),
                "loss_finite": True,
                "backward_pass": "PASS",
                "spatial_tensor_content_hash": audit["spatial_tensor_content_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
