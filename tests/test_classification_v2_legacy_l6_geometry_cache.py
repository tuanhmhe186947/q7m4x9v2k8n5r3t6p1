from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.training.legacy_development_l6_geometry_cache import (
    DATASET_ID,
    GEOMETRY_FEATURE_NAMES,
    LINEAGE_SCOPE,
    SOURCE_TYPE,
    _validate_cache_config_payload,
    compute_geometry_features,
    geometry_availability,
    single_source_probe_audit,
)

CONFIG_PATHS = (
    Path("configs/classification_v2/legacy_development_l6_geometry_cache_v1.json"),
    Path(
        "configs/classification_v2/"
        "legacy_development_l6_geometry_cache_repeat_v1.json"
    ),
)


def test_geometry_features_match_declared_math() -> None:
    frame = pd.DataFrame(
        {
            "image_width": [200.0],
            "image_height": [100.0],
            "x1": [20.0],
            "y1": [10.0],
            "x2": [60.0],
            "y2": [50.0],
            "bbox_context_valid": [True],
        }
    )

    geometry = compute_geometry_features(frame)

    bbox_diag = np.sqrt(40.0**2 + 40.0**2)
    image_diag = np.sqrt(200.0**2 + 100.0**2)
    box_diag_n = bbox_diag / image_diag
    expected = np.asarray(
        [
            0.2,
            0.3,
            0.2,
            0.4,
            0.08,
            1.0,
            box_diag_n,
            0.08 / box_diag_n**2,
        ]
    )
    assert list(GEOMETRY_FEATURE_NAMES) == [
        "cx_n",
        "cy_n",
        "bw_n",
        "bh_n",
        "area_n",
        "aspect_ratio",
        "box_diag_n",
        "box_compactness",
    ]
    np.testing.assert_allclose(geometry[0], expected, rtol=0.0, atol=1e-12)
    assert geometry_availability(frame, geometry).tolist() == [True]


def test_geometry_availability_fails_closed_on_invalid_or_nonfinite() -> None:
    frame = pd.DataFrame({"bbox_context_valid": [True, False]})
    geometry = np.ones((2, len(GEOMETRY_FEATURE_NAMES)), dtype=np.float64)
    geometry[0, 0] = np.nan

    availability = geometry_availability(frame, geometry)

    assert availability.tolist() == [False, False]


def test_source_probe_reports_single_source_as_not_estimable() -> None:
    frame = pd.DataFrame(
        {
            "source_type": [SOURCE_TYPE, SOURCE_TYPE],
            "dataset_id": [DATASET_ID, DATASET_ID],
        }
    )

    audit = single_source_probe_audit(frame)

    assert audit["status"] == "NOT_ESTIMABLE_SINGLE_LEGACY_SOURCE"
    assert audit["probe_fit_performed"] is False
    assert audit["two_source_result_reported"] is False
    assert audit["estimable"] is False
    drifted = frame.copy()
    drifted.loc[1, "source_type"] = "cvat"
    with pytest.raises(ValueError, match="source identity drift"):
        single_source_probe_audit(drifted)


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_cache_config_locks_legacy_16f_and_claim_boundary(
    config_path: Path,
) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    _validate_cache_config_payload(payload)

    changed_name = copy.deepcopy(payload)
    changed_name["source_identity"]["canonical_short_name"] = "legacy"
    with pytest.raises(ValueError, match="source identity drift"):
        _validate_cache_config_payload(changed_name)
    changed_merged = copy.deepcopy(payload)
    changed_merged["source_identity"]["merged_data"] = True
    with pytest.raises(ValueError, match="source identity drift"):
        _validate_cache_config_payload(changed_merged)
    changed_claim = copy.deepcopy(payload)
    changed_claim["q2_claim_allowed"] = True
    with pytest.raises(ValueError, match="q2_claim_allowed"):
        _validate_cache_config_payload(changed_claim)
    assert payload["lineage_scope"] == LINEAGE_SCOPE
