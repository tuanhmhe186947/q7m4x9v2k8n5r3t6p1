from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from pig_behavior.classification_v2.contracts.window_alignment import (
    ordered_window_id_sha256,
)
from pig_behavior.classification_v2.evaluation.domain_controls import (
    LABEL_INDEPENDENT_AVAILABILITY_COLUMNS,
    audit_domain_feature_shift,
    grouped_availability_behavior_probe,
    grouped_source_probe,
)

FEATURES = ["speed_mean_window", "roi_feeder_near_ratio_window"]


@dataclass(slots=True)
class ProbeFixture:
    features: pd.DataFrame
    metadata: pd.DataFrame
    mapping: pd.DataFrame
    roles: pd.DataFrame
    availability: pd.DataFrame
    ordered_hash: str


def test_grouped_source_probe_is_whitelist_bound_and_native_unit_oof() -> None:
    fixture = _probe_fixture()

    predictions, audit = grouped_source_probe(
        fixture.features,
        fixture.metadata,
        fixture.mapping,
        fixture.roles,
        feature_whitelist=FEATURES,
        expected_ordered_window_id_sha256=fixture.ordered_hash,
    )

    assert len(predictions) == 8
    assert predictions["temporal_unit_key"].nunique() == 8
    assert "window_id" not in predictions
    assert {"source_type", "split_group_key", "video_key"}.issubset(predictions)
    assert audit["statistical_unit"] == "native_temporal_unit"
    assert audit["eligible_window_rows"] == 16
    assert audit["eligible_native_unit_rows"] == 8
    assert audit["eligible_native_unit_to_oof_row_loss"] == 0
    assert audit["every_eligible_native_unit_tested_once"] is True
    assert audit["feature_whitelist"] == FEATURES
    assert audit["source_identifier_in_features"] is False
    assert all(
        fold["validation_and_test_excluded_from_fit"]
        for fold in audit["folds"]
    )


def test_availability_probe_uses_only_registered_label_independent_masks() -> None:
    fixture = _probe_fixture()

    predictions, audit = grouped_availability_behavior_probe(
        fixture.availability,
        fixture.metadata,
        fixture.mapping,
        fixture.roles,
        availability_feature_whitelist=list(
            LABEL_INDEPENDENT_AVAILABILITY_COLUMNS
        ),
        expected_ordered_window_id_sha256=fixture.ordered_hash,
    )

    assert len(predictions) == 8
    assert predictions["temporal_unit_key"].is_unique
    assert audit["diagnostic_only"] is True
    assert audit["availability_features_enter_classifier_x"] is False
    assert audit["target_derived_availability_columns"] == []
    assert audit["eligible_native_unit_to_oof_row_loss"] == 0


@pytest.mark.parametrize("mutation", ["extra", "reordered", "forbidden"])
def test_source_probe_rejects_nonexact_feature_contract(mutation: str) -> None:
    fixture = _probe_fixture()
    features = fixture.features.copy()
    whitelist = list(FEATURES)
    if mutation == "extra":
        features["numeric_extra"] = 1.0
    elif mutation == "reordered":
        features = features[list(reversed(FEATURES))]
    else:
        features = features.rename(columns={FEATURES[0]: "source_type"})
        whitelist[0] = "source_type"

    with pytest.raises(ValueError):
        grouped_source_probe(
            features,
            fixture.metadata,
            fixture.mapping,
            fixture.roles,
            feature_whitelist=whitelist,
            expected_ordered_window_id_sha256=fixture.ordered_hash,
        )


def test_source_probe_rejects_ordered_window_hash_drift() -> None:
    fixture = _probe_fixture()

    with pytest.raises(ValueError, match="lineage mismatch"):
        grouped_source_probe(
            fixture.features,
            fixture.metadata,
            fixture.mapping,
            fixture.roles,
            feature_whitelist=FEATURES,
            expected_ordered_window_id_sha256="0" * 64,
        )


def test_source_probe_applies_trainer_forbidden_patterns() -> None:
    fixture = _probe_fixture()
    features = fixture.features.rename(columns={FEATURES[0]: "derived_label"})
    whitelist = ["derived_label", FEATURES[1]]

    with pytest.raises(ValueError, match="forbidden X fields"):
        grouped_source_probe(
            features,
            fixture.metadata,
            fixture.mapping,
            fixture.roles,
            feature_whitelist=whitelist,
            forbidden_patterns=["*label*"],
            expected_ordered_window_id_sha256=fixture.ordered_hash,
        )


def test_source_probe_rejects_multi_native_window() -> None:
    fixture = _probe_fixture()
    fixture.mapping.loc[0, "num_temporal_units_window"] = 2

    with pytest.raises(ValueError, match="exactly one native unit"):
        grouped_source_probe(
            fixture.features,
            fixture.metadata,
            fixture.mapping,
            fixture.roles,
            feature_whitelist=FEATURES,
            expected_ordered_window_id_sha256=fixture.ordered_hash,
        )


def test_source_probe_rejects_native_metadata_conflict() -> None:
    fixture = _probe_fixture()
    fixture.metadata.loc[1, "source_type"] = "legacy_recovered"

    with pytest.raises(ValueError, match="metadata conflicts"):
        grouped_source_probe(
            fixture.features,
            fixture.metadata,
            fixture.mapping,
            fixture.roles,
            feature_whitelist=FEATURES,
            expected_ordered_window_id_sha256=fixture.ordered_hash,
        )


def test_source_probe_rejects_duplicate_grouped_role() -> None:
    fixture = _probe_fixture()
    fixture.roles = pd.concat(
        [fixture.roles, fixture.roles.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate grouped role"):
        grouped_source_probe(
            fixture.features,
            fixture.metadata,
            fixture.mapping,
            fixture.roles,
            feature_whitelist=FEATURES,
            expected_ordered_window_id_sha256=fixture.ordered_hash,
        )


def test_source_probe_rejects_role_metadata_conflict() -> None:
    fixture = _probe_fixture()
    unit = fixture.roles.loc[0, "temporal_unit_key"]
    fixture.roles.loc[
        fixture.roles["temporal_unit_key"].eq(unit),
        "source_type",
    ] = "wrong_source"

    with pytest.raises(ValueError, match="role/native metadata conflicts"):
        grouped_source_probe(
            fixture.features,
            fixture.metadata,
            fixture.mapping,
            fixture.roles,
            feature_whitelist=FEATURES,
            expected_ordered_window_id_sha256=fixture.ordered_hash,
        )


def test_availability_probe_rejects_label_gated_interaction_mask() -> None:
    fixture = _probe_fixture()
    availability = pd.DataFrame(
        {
            "window_id": fixture.metadata["window_id"],
            "interaction_context_ready": True,
        }
    )

    with pytest.raises(ValueError, match="target-derived"):
        grouped_availability_behavior_probe(
            availability,
            fixture.metadata,
            fixture.mapping,
            fixture.roles,
            availability_feature_whitelist=["interaction_context_ready"],
            expected_ordered_window_id_sha256=fixture.ordered_hash,
        )


def test_availability_probe_rejects_unknown_mask_payload() -> None:
    fixture = _probe_fixture()
    fixture.availability["scene_context_ready"] = fixture.availability[
        "scene_context_ready"
    ].astype(object)
    fixture.availability.loc[0, "scene_context_ready"] = "unknown"

    with pytest.raises(ValueError, match="unknown values"):
        grouped_availability_behavior_probe(
            fixture.availability,
            fixture.metadata,
            fixture.mapping,
            fixture.roles,
            availability_feature_whitelist=list(
                LABEL_INDEPENDENT_AVAILABILITY_COLUMNS
            ),
            expected_ordered_window_id_sha256=fixture.ordered_hash,
        )


def test_domain_feature_shift_uses_same_whitelist_and_window_hash() -> None:
    fixture = _probe_fixture()

    audit = audit_domain_feature_shift(
        fixture.features,
        fixture.metadata,
        feature_whitelist=FEATURES,
        expected_ordered_window_id_sha256=fixture.ordered_hash,
    )

    assert audit["schema_version"] == "classification_v2_domain_feature_shift_v2"
    assert audit["feature_whitelist"] == FEATURES
    assert audit["eligible_rows"] == 16
    assert audit["ordered_window_lineage_match"] is True


def _probe_fixture() -> ProbeFixture:
    metadata_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, float]] = []
    availability_rows: list[dict[str, object]] = []
    native_authority: list[dict[str, object]] = []
    for unit_index in range(8):
        group_index = unit_index // 2
        source = (
            "cvat_tracking_xml" if unit_index % 2 == 0 else "legacy_recovered"
        )
        behavior = "stand" if unit_index % 2 == 0 else "eat"
        temporal_unit_key = f"native_{unit_index:02d}"
        group = f"session_{group_index:02d}"
        video = f"video_{group_index:02d}"
        native_authority.append(
            {
                "temporal_unit_key": temporal_unit_key,
                "recording_group_id": group,
                "behavior_label": behavior,
                "native_unit_valid_for_main_eval": True,
                "source_type": source,
                "video_key": video,
                "group_index": group_index,
            }
        )
        for window_offset in range(2):
            window_id = f"window_{unit_index:02d}_{window_offset}"
            metadata_rows.append(
                {
                    "window_id": window_id,
                    "source_type": source,
                    "behavior_window_label": behavior,
                    "window_valid_for_main_train": True,
                    "split_group_key": group,
                    "video_key": video,
                }
            )
            mapping_rows.append(
                {
                    "window_id": window_id,
                    "temporal_unit_keys_window": temporal_unit_key,
                    "num_temporal_units_window": 1,
                }
            )
            source_signal = 1.0 if source == "legacy_recovered" else -1.0
            feature_rows.append(
                {
                    FEATURES[0]: source_signal + 0.01 * window_offset,
                    FEATURES[1]: float(group_index) + 0.02 * window_offset,
                }
            )
            availability_rows.append(
                {
                    "window_id": window_id,
                    "window_image_context_complete": True,
                    "scene_context_ready": behavior == "stand",
                    "scene_partner_context_ready": behavior == "eat",
                }
            )
    roles = _expanded_roles(native_authority)
    metadata = pd.DataFrame(metadata_rows)
    return ProbeFixture(
        features=pd.DataFrame(feature_rows, columns=FEATURES),
        metadata=metadata,
        mapping=pd.DataFrame(mapping_rows),
        roles=roles,
        availability=pd.DataFrame(
            availability_rows,
            columns=["window_id", *LABEL_INDEPENDENT_AVAILABILITY_COLUMNS],
        ),
        ordered_hash=ordered_window_id_sha256(metadata["window_id"]),
    )


def _expanded_roles(native_authority: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_index in range(4):
        for authority in native_authority:
            group_index = int(authority["group_index"])
            if group_index == fold_index:
                role = "test"
            elif group_index == (fold_index + 1) % 4:
                role = "validation"
            else:
                role = "train"
            rows.append(
                {
                    key: value
                    for key, value in authority.items()
                    if key != "group_index"
                }
                | {"outer_fold_id": f"fold_{fold_index:02d}", "role": role}
            )
    return pd.DataFrame(rows)
