from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.datasets.fold_event_weights import (
    audit_fold_event_weight_manifest,
    build_fold_event_weight_manifest,
)
from pig_behavior.classification_v2.training.data_module import (
    _validated_fold_class_weights,
)


def _windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2", "w3", "w4"],
            "temporal_unit_keys_json": [
                '["event-a"]',
                '["event-a","event-b"]',
                '["event-b"]',
                '["event-c"]',
                "[]",
            ],
            "window_valid_for_main_train": [True, True, True, True, False],
            "behavior_window_label": [
                "drink",
                "drink",
                "drink",
                "eat",
                "stand",
            ],
            "window_sample_weight": [1.0, 1.0, 1.0, 1.0, 0.0],
        }
    )


def _roles() -> pd.DataFrame:
    rows = []
    labels = {"event-a": "drink", "event-b": "drink", "event-c": "eat"}
    assignments = {
        "q2_outer_00": {
            "event-a": "train",
            "event-b": "train",
            "event-c": "train",
        },
        "q2_outer_01": {
            "event-a": "test",
            "event-b": "test",
            "event-c": "train",
        },
    }
    for fold_id, events in assignments.items():
        for event, role in events.items():
            rows.append(
                {
                    "outer_fold_id": fold_id,
                    "temporal_unit_key": event,
                    "role": role,
                    "behavior_label": labels[event],
                    "native_unit_valid_for_main_eval": True,
                }
            )
    return pd.DataFrame(rows)


def _selection() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": _windows()["window_id"],
            "fixed6_keep": [True, True, True, True, False],
        }
    )


def _build():
    return build_fold_event_weight_manifest(
        _windows(),
        _roles(),
        selection=_selection(),
        class_weight_power=1.0,
    )


def test_fold_event_weights_keep_every_window_in_every_fold() -> None:
    tables = _build()
    weights = tables.weights

    assert len(weights) == len(_windows()) * 2
    assert not weights.duplicated(["outer_fold_id", "window_id"]).any()
    assert tables.audit["errors"] == []
    assert tables.audit["valid"] is True


def test_fold_event_mass_uses_only_training_role_events() -> None:
    weights = _build().weights.set_index(["outer_fold_id", "window_id"])

    assert weights.loc[("q2_outer_00", "w0"), "fold_event_mass_weight"] == 0.5
    assert weights.loc[("q2_outer_00", "w1"), "fold_event_mass_weight"] == 1.0
    assert weights.loc[("q2_outer_00", "w2"), "fold_event_mass_weight"] == 0.5
    assert weights.loc[("q2_outer_01", "w0"), "fold_event_mass_weight"] == 0.0
    assert weights.loc[("q2_outer_01", "w3"), "fold_event_mass_weight"] == 1.0


def test_validation_test_and_invalid_rows_have_zero_training_weight() -> None:
    weights = _build().weights
    nontraining = ~weights["window_valid_for_fold_training_weight"]

    assert weights.loc[nontraining, "fold_event_sample_weight"].eq(0.0).all()
    assert weights.loc[
        nontraining,
        "fold_event_class_sample_weight",
    ].eq(0.0).all()


def test_temporal_view_selection_excludes_rows_without_dropping_them() -> None:
    selection = _selection()
    selection.loc[selection["window_id"].eq("w2"), "fixed6_keep"] = False

    tables = build_fold_event_weight_manifest(
        _windows(),
        _roles(),
        selection=selection,
        class_weight_power=1.0,
    )
    weights = tables.weights
    excluded = weights["window_id"].eq("w2")

    assert len(weights) == len(_windows()) * 2
    assert weights.loc[excluded, "fold_event_sample_weight"].eq(0.0).all()
    assert not weights.loc[
        excluded,
        "window_selected_for_training_view",
    ].any()


def test_temporal_view_selection_order_mismatch_is_rejected() -> None:
    selection = _selection().iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError, match="selection window order mismatch"):
        build_fold_event_weight_manifest(
            _windows(),
            _roles(),
            selection=selection,
        )


def test_fold_class_weights_use_native_event_mass_not_window_count() -> None:
    summary = _build().class_summary.set_index(
        ["outer_fold_id", "behavior_label"]
    )

    assert summary.loc[("q2_outer_00", "drink"), "native_event_mass"] == 2.0
    assert summary.loc[("q2_outer_00", "eat"), "native_event_mass"] == 1.0
    assert summary.loc[("q2_outer_00", "drink"), "fold_class_weight"] == 0.75
    assert summary.loc[("q2_outer_00", "eat"), "fold_class_weight"] == 1.5


def test_fold_weights_are_mean_one_bounded_and_report_effective_size() -> None:
    tables = _build()
    weights = tables.weights
    for fold_id, fold in weights.groupby("outer_fold_id"):
        train = fold.loc[fold["window_valid_for_fold_training_weight"]]
        assert train["fold_event_sample_weight"].mean() == pytest.approx(1.0)
        assert train["fold_event_class_sample_weight"].mean() == pytest.approx(
            1.0
        )
        assert train["fold_event_class_sample_weight"].max() <= 10.0
        assert tables.audit["folds"][fold_id][
            "event_class_effective_sample_size"
        ] > 0.0


def test_overlapping_window_cannot_cross_fold_roles() -> None:
    roles = _roles()
    mask = (
        roles["outer_fold_id"].eq("q2_outer_00")
        & roles["temporal_unit_key"].eq("event-b")
    )
    roles.loc[mask, "role"] = "validation"

    with pytest.raises(ValueError, match="overlapping window crosses roles"):
        build_fold_event_weight_manifest(_windows(), roles)


def test_eligible_train_native_event_without_window_is_rejected() -> None:
    roles = _roles()
    roles = pd.concat(
        [
            roles,
            pd.DataFrame(
                {
                    "outer_fold_id": ["q2_outer_00", "q2_outer_01"],
                    "temporal_unit_key": ["event-d", "event-d"],
                    "role": ["train", "test"],
                    "behavior_label": ["stand", "stand"],
                    "native_unit_valid_for_main_eval": [True, True],
                }
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="eligible_train_native_events_without_valid_window",
    ):
        build_fold_event_weight_manifest(_windows(), roles)


def test_invalid_native_unit_is_retained_with_zero_training_weight() -> None:
    roles = _roles()
    invalid = roles["temporal_unit_key"].eq("event-b")
    roles.loc[invalid, "native_unit_valid_for_main_eval"] = False

    tables = build_fold_event_weight_manifest(
        _windows(),
        roles,
        selection=_selection(),
        class_weight_power=1.0,
    )
    event_b = tables.weights["window_id"].isin(["w1", "w2"])

    assert not tables.weights.loc[
        event_b,
        "window_native_units_valid_for_main_eval",
    ].any()
    assert tables.weights.loc[
        event_b,
        "fold_event_sample_weight",
    ].eq(0.0).all()


def test_native_and_window_label_mismatch_is_rejected() -> None:
    roles = _roles()
    roles.loc[
        roles["temporal_unit_key"].eq("event-c"),
        "behavior_label",
    ] = "stand"

    with pytest.raises(ValueError, match="native/window label mismatch"):
        build_fold_event_weight_manifest(_windows(), roles)


def test_fold_weight_audit_rejects_reordered_rows() -> None:
    tables = _build()
    persisted = tables.weights.iloc[::-1].reset_index(drop=True)

    audit = audit_fold_event_weight_manifest(
        persisted,
        _windows(),
        _roles(),
        selection=_selection(),
        class_weight_power=1.0,
    )

    assert audit["valid"] is False
    assert audit["fold_window_order_mismatch_rows"] > 0


def test_fold_weight_audit_rejects_tampered_weight() -> None:
    tables = _build()
    persisted = tables.weights.copy()
    persisted.loc[0, "fold_event_sample_weight"] = 99.0

    audit = audit_fold_event_weight_manifest(
        persisted,
        _windows(),
        _roles(),
        selection=_selection(),
        class_weight_power=1.0,
    )

    assert audit["valid"] is False
    assert "numeric_mismatch_fold_event_sample_weight=1" in audit["errors"]


def test_training_loader_recomputes_and_rejects_class_weight_drift() -> None:
    fold = _build().weights
    fold = fold.loc[fold["outer_fold_id"].eq("q2_outer_00")].reset_index(
        drop=True
    )
    valid = fold["window_valid_for_fold_training_weight"]

    observed = _validated_fold_class_weights(
        fold,
        valid,
        power=1.0,
        max_weight=5.0,
    )
    assert observed["drink"] == 0.75
    fold.loc[fold["behavior_window_label"].eq("drink"), "fold_class_weight"] = 9.0
    with pytest.raises(ValueError, match="class-weight mismatch"):
        _validated_fold_class_weights(
            fold,
            valid,
            power=1.0,
            max_weight=5.0,
        )


def test_fold_event_weight_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    command = _builder_command(paths, dry_run=True)

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert not paths["weights"].exists()
    assert not paths["audit"].exists()


def test_fold_event_weight_cli_requires_overwrite_and_checker_can_dry_run(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    first = subprocess.run(
        _builder_command(paths),
        capture_output=True,
        text=True,
        check=False,
    )
    second = subprocess.run(
        _builder_command(paths),
        capture_output=True,
        text=True,
        check=False,
    )
    checker = subprocess.run(
        _checker_command(paths),
        capture_output=True,
        text=True,
        check=False,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode != 0
    assert "--overwrite" in second.stderr
    assert checker.returncode == 0, checker.stderr
    assert not paths["check"].exists()


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "windows": tmp_path / "windows.csv",
        "roles": tmp_path / "roles.csv",
        "selection": tmp_path / "selection.csv",
        "weights": tmp_path / "fold_weights.csv",
        "classes": tmp_path / "class_summary.csv",
        "events": tmp_path / "event_summary.csv",
        "audit": tmp_path / "audit.json",
        "check": tmp_path / "check.json",
    }
    _windows().to_csv(paths["windows"], index=False)
    _roles().to_csv(paths["roles"], index=False)
    _selection().to_csv(paths["selection"], index=False)
    return paths


def _builder_command(
    paths: dict[str, Path],
    *,
    dry_run: bool = False,
) -> list[str]:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "classification_v2"
        / "02_train_ready_exports"
        / "classification_v2_build_fold_event_weights.py"
    )
    command = [
        sys.executable,
        str(script),
        "--window-manifest-csv",
        str(paths["windows"]),
        "--fold-role-csv",
        str(paths["roles"]),
        "--selection-manifest-csv",
        str(paths["selection"]),
        "--output-csv",
        str(paths["weights"]),
        "--class-summary-csv",
        str(paths["classes"]),
        "--event-summary-csv",
        str(paths["events"]),
        "--audit-json",
        str(paths["audit"]),
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def _checker_command(paths: dict[str, Path]) -> list[str]:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "classification_v2"
        / "02_train_ready_exports"
        / "check_classification_v2_fold_event_weights.py"
    )
    return [
        sys.executable,
        str(script),
        "--fold-event-weight-csv",
        str(paths["weights"]),
        "--window-manifest-csv",
        str(paths["windows"]),
        "--fold-role-csv",
        str(paths["roles"]),
        "--selection-manifest-csv",
        str(paths["selection"]),
        "--output-json",
        str(paths["check"]),
        "--dry-run",
    ]
