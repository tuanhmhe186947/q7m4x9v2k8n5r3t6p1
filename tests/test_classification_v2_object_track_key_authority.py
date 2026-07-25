from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from pig_behavior.classification_v2.contracts.identifiers import (
    OBJECT_TRACK_KEY_VERSION,
    ensure_object_track_keys,
)
from pig_behavior.classification_v2.contracts.semantic_lineage import (
    build_semantic_domain_registry,
    compute_stage_semantics_hash,
)
from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_SCHEMA_HASH,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    REPO_ROOT
    / "docs"
    / "classification_v2"
    / "scientific_contract_v1"
    / "00_pipeline_contract.yaml"
)
CHECKER_PATH = (
    REPO_ROOT
    / "scripts"
    / "classification_v2"
    / "00_source_feature_temporal"
    / "check_classification_v2_frame_local_primitives.py"
)
EXPECTED_PHASE2_SCHEMA_HASH = (
    "ec0c511b5f5198240492be49c0492e543c9e38eb4a4ff446259b958c2a59963b"
)


def _checker_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "classification_v2_frame_local_independent_checker",
        CHECKER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker() -> ModuleType:
    return _checker_module()


@pytest.fixture(scope="module")
def primary_contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source_type": "cvat_tracking_xml",
        "dataset_id": "dataset-a",
        "video_key": "video-a",
        "frame_uid": "frame-a",
        "track_id": "7",
        "object_id_in_image": "object-7",
        "object_id": "",
        "pig_id": "Pig_7",
        "object_track_key": "",
    }
    row.update(overrides)
    return row


CONFORMANCE_CASES = [
    (
        "normal_legacy_track_id",
        [_row(source_type="legacy_recovered", track_id="legacy-1")],
        ["source=legacy_recovered|dataset=dataset-a|video=video-a|track_id=legacy-1"],
        False,
    ),
    (
        "normal_cvat_track_id",
        [_row(track_id="9")],
        ["source=cvat_tracking_xml|dataset=dataset-a|video=video-a|track_id=9"],
        False,
    ),
    (
        "object_id_fallback",
        [_row(track_id="", object_id_in_image="object-9")],
        ["source=cvat_tracking_xml|dataset=dataset-a|video=video-a|object_id=object-9"],
        False,
    ),
    (
        "blank_pig_id_valid_track",
        [_row(track_id="10", pig_id="")],
        ["source=cvat_tracking_xml|dataset=dataset-a|video=video-a|track_id=10"],
        False,
    ),
    (
        "duplicate_pig_id_two_tracks",
        [
            _row(frame_uid="frame-a", track_id="1", pig_id="same"),
            _row(frame_uid="frame-b", track_id="2", pig_id="same"),
        ],
        [
            "source=cvat_tracking_xml|dataset=dataset-a|video=video-a|track_id=1",
            "source=cvat_tracking_xml|dataset=dataset-a|video=video-a|track_id=2",
        ],
        False,
    ),
    (
        "same_track_across_videos",
        [
            _row(frame_uid="frame-a", video_key="video-a", track_id="1"),
            _row(frame_uid="frame-b", video_key="video-b", track_id="1"),
        ],
        [
            "source=cvat_tracking_xml|dataset=dataset-a|video=video-a|track_id=1",
            "source=cvat_tracking_xml|dataset=dataset-a|video=video-b|track_id=1",
        ],
        False,
    ),
    (
        "same_track_across_datasets",
        [
            _row(frame_uid="frame-a", dataset_id="dataset-a", track_id="1"),
            _row(frame_uid="frame-b", dataset_id="dataset-b", track_id="1"),
        ],
        [
            "source=cvat_tracking_xml|dataset=dataset-a|video=video-a|track_id=1",
            "source=cvat_tracking_xml|dataset=dataset-b|video=video-a|track_id=1",
        ],
        False,
    ),
    (
        "same_track_across_sources",
        [
            _row(frame_uid="frame-a", source_type="source-a", track_id="1"),
            _row(frame_uid="frame-b", source_type="source-b", track_id="1"),
        ],
        [
            "source=source-a|dataset=dataset-a|video=video-a|track_id=1",
            "source=source-b|dataset=dataset-a|video=video-a|track_id=1",
        ],
        False,
    ),
    (
        "integer_and_string_track_id",
        [
            _row(frame_uid="frame-a", track_id=1),
            _row(frame_uid="frame-b", track_id="1"),
        ],
        [
            "source=cvat_tracking_xml|dataset=dataset-a|video=video-a|track_id=1",
            "source=cvat_tracking_xml|dataset=dataset-a|video=video-a|track_id=1",
        ],
        False,
    ),
    (
        "input_row_order_permutation",
        [
            _row(frame_uid="frame-b", track_id="2"),
            _row(frame_uid="frame-a", track_id="1"),
        ],
        [
            "source=cvat_tracking_xml|dataset=dataset-a|video=video-a|track_id=2",
            "source=cvat_tracking_xml|dataset=dataset-a|video=video-a|track_id=1",
        ],
        False,
    ),
    (
        "missing_track_and_object",
        [_row(track_id="", object_id_in_image="", object_id="")],
        [],
        True,
    ),
    (
        "pig_id_only_forbidden",
        [
            _row(
                track_id="",
                object_id_in_image="",
                object_id="",
                pig_id="Pig_9",
            )
        ],
        [],
        True,
    ),
    (
        "special_characters_are_escaped",
        [
            _row(
                source_type="source|α",
                dataset_id="dataset=β",
                video_key="video/γ",
                track_id="track 1",
            )
        ],
        [
            "source=source%7C%CE%B1|dataset=dataset%3D%CE%B2|"
            "video=video%2F%CE%B3|track_id=track%201"
        ],
        False,
    ),
    (
        "leading_trailing_whitespace_is_trimmed",
        [
            _row(
                source_type=" source-a ",
                dataset_id=" dataset-a ",
                video_key=" video-a ",
                track_id=" 1 ",
            )
        ],
        ["source=source-a|dataset=dataset-a|video=video-a|track_id=1"],
        False,
    ),
]


@pytest.mark.parametrize(
    ("case_id", "records", "expected_keys", "expect_invalid"),
    CONFORMANCE_CASES,
    ids=[case[0] for case in CONFORMANCE_CASES],
)
def test_production_and_independent_reference_conform(
    case_id: str,
    records: list[dict[str, object]],
    expected_keys: list[str],
    expect_invalid: bool,
    checker: ModuleType,
    primary_contract: dict[str, object],
) -> None:
    del case_id
    rows = pd.DataFrame.from_records(records)
    identity_contract = primary_contract["object_track_key_contract"]
    assert isinstance(identity_contract, dict)
    reference = checker._reference_object_track_keys(
        rows,
        identity_contract,
    )
    if expect_invalid:
        with pytest.raises(
            ValueError,
            match="missing_object_track_authority",
        ):
            ensure_object_track_keys(rows, source_name="conformance")
        assert reference["expected_canonical_key"].eq("").all()
        assert reference["reason_code"].eq(
            "MISSING_IDENTITY_AUTHORITY"
        ).all()
        return
    production = ensure_object_track_keys(
        rows,
        source_name="conformance",
    )
    assert production["object_track_key"].tolist() == expected_keys
    assert reference["expected_canonical_key"].tolist() == expected_keys
    assert reference["reason_code"].eq("OK").all()


def test_contract_version_matches_production_constant(
    primary_contract: dict[str, object],
) -> None:
    identity_contract = primary_contract["object_track_key_contract"]
    assert isinstance(identity_contract, dict)
    assert identity_contract["schema_version"] == OBJECT_TRACK_KEY_VERSION
    assert identity_contract["pig_id_authoritative"] is False
    assert identity_contract["row_order_authoritative"] is False


def test_scope_components_prevent_cross_authority_collisions() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(frame_uid="a", source_type="source-a", track_id="1"),
            _row(frame_uid="b", source_type="source-b", track_id="1"),
            _row(frame_uid="c", dataset_id="dataset-b", track_id="1"),
            _row(frame_uid="d", video_key="video-b", track_id="1"),
        ]
    )
    production = ensure_object_track_keys(rows, source_name="scope")
    assert production["object_track_key"].nunique() == 4


def test_row_order_does_not_affect_key() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(frame_uid="a", track_id="1"),
            _row(frame_uid="b", track_id="2"),
        ]
    )
    ordered = ensure_object_track_keys(rows, source_name="ordered")
    shuffled = ensure_object_track_keys(
        rows.iloc[::-1],
        source_name="shuffled",
    )
    expected = ordered.set_index("frame_uid")["object_track_key"].sort_index()
    actual = shuffled.set_index("frame_uid")["object_track_key"].sort_index()
    pd.testing.assert_series_equal(actual, expected)


NEGATIVE_CONTROLS = [
    "pig_id_as_authority",
    "component_order_changed",
    "component_names_changed",
    "video_scope_omitted",
    "object_id_used_despite_track_id",
    "production_output_corrupted",
    "cross_video_duplicate_injected",
    "row_order_dependent_constructor",
]


@pytest.mark.parametrize("control", NEGATIVE_CONTROLS)
def test_independent_checker_rejects_key_negative_controls(
    control: str,
    checker: ModuleType,
    primary_contract: dict[str, object],
) -> None:
    source = pd.DataFrame.from_records(
        [
            _row(
                frame_uid="frame-a",
                video_key="video-a",
                track_id="1",
                object_id_in_image="object-a",
                pig_id="Pig_A",
            ),
            _row(
                frame_uid="frame-b",
                video_key="video-b",
                track_id="1",
                object_id_in_image="object-b",
                pig_id="Pig_B",
            ),
        ]
    )
    output = ensure_object_track_keys(
        source,
        source_name="negative-control",
    )
    if control == "pig_id_as_authority":
        output["object_track_key"] = "pig=" + source["pig_id"]
    elif control == "component_order_changed":
        output["object_track_key"] = (
            "video="
            + source["video_key"]
            + "|source="
            + source["source_type"]
            + "|dataset="
            + source["dataset_id"]
            + "|track_id="
            + source["track_id"]
        )
    elif control == "component_names_changed":
        output["object_track_key"] = (
            "src="
            + source["source_type"]
            + "|data="
            + source["dataset_id"]
            + "|vid="
            + source["video_key"]
            + "|track="
            + source["track_id"]
        )
    elif control == "video_scope_omitted":
        output["object_track_key"] = (
            "source="
            + source["source_type"]
            + "|dataset="
            + source["dataset_id"]
            + "|track_id="
            + source["track_id"]
        )
    elif control == "object_id_used_despite_track_id":
        output["object_track_key"] = (
            "source="
            + source["source_type"]
            + "|dataset="
            + source["dataset_id"]
            + "|video="
            + source["video_key"]
            + "|object_id="
            + source["object_id_in_image"]
        )
    elif control == "production_output_corrupted":
        output.loc[0, "object_track_key"] += "-corrupt"
    elif control == "cross_video_duplicate_injected":
        output.loc[1, "object_track_key"] = output.loc[0, "object_track_key"]
    elif control == "row_order_dependent_constructor":
        output["object_track_key"] = [
            f"row={position}" for position in range(len(output))
        ]
    else:
        raise AssertionError(control)
    identity_contract = primary_contract["object_track_key_contract"]
    assert isinstance(identity_contract, dict)
    reference = checker._reference_object_track_keys(
        source,
        identity_contract,
    )
    result = checker._audit_object_track_keys(output, reference)
    assert result["mismatches"] >= 1
    assert result["details"]


def test_checker_mismatch_details_are_structured(
    checker: ModuleType,
    primary_contract: dict[str, object],
) -> None:
    source = pd.DataFrame.from_records([_row()])
    output = ensure_object_track_keys(source, source_name="details")
    output.loc[0, "object_track_key"] = "wrong"
    identity_contract = primary_contract["object_track_key_contract"]
    assert isinstance(identity_contract, dict)
    reference = checker._reference_object_track_keys(
        source,
        identity_contract,
    )
    result = checker._audit_object_track_keys(output, reference)
    assert result["mismatches"] == 1
    assert set(result["details"][0]) == {
        "row_authority_key",
        "expected_canonical_key",
        "actual_object_track_key",
        "selected_identity_type",
        "selected_identity_value",
        "source",
        "dataset",
        "video",
        "reason_code",
    }


def test_checker_does_not_call_production_key_constructor() -> None:
    checker_source = CHECKER_PATH.read_text(encoding="utf-8")
    assert "ensure_object_track_keys" not in checker_source
    assert "contracts.identifiers" not in checker_source


def test_serialization_contract_changes_source_stage_semantics(
    primary_contract: dict[str, object],
) -> None:
    current_registry = build_semantic_domain_registry(primary_contract)
    changed_contract = copy.deepcopy(primary_contract)
    changed_identity = changed_contract["object_track_key_contract"]
    assert isinstance(changed_identity, dict)
    changed_identity["component_delimiter"] = "^"
    changed_registry = build_semantic_domain_registry(changed_contract)
    current_hash = compute_stage_semantics_hash(
        "stage.legacy_cvat_source_merge",
        current_registry,
    )
    changed_hash = compute_stage_semantics_hash(
        "stage.legacy_cvat_source_merge",
        changed_registry,
    )
    assert current_hash != changed_hash


def test_phase2_motion_schema_hash_is_unchanged() -> None:
    assert MOTION_SCHEMA_HASH == EXPECTED_PHASE2_SCHEMA_HASH
