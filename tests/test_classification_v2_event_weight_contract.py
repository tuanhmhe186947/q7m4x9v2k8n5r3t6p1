import pandas as pd
import pytest

from pig_behavior.classification_v2.datasets.event_weights import (
    audit_event_weight_manifest,
    build_event_weight_manifest,
)


def _windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2", "w3"],
            "temporal_unit_keys_json": [
                '["event-a"]',
                '["event-a","event-b"]',
                '["event-b"]',
                "[]",
            ],
            "window_valid_for_main_train": [True, True, True, False],
            "window_sample_weight": [1.0, 1.0, 1.0, 0.0],
        }
    )


def test_event_weights_conserve_one_mass_per_native_event() -> None:
    tables = build_event_weight_manifest(_windows())
    weights = tables.weights.set_index("window_id")

    assert weights.loc["w0", "event_balanced_sample_weight"] == 0.5
    assert weights.loc["w1", "event_balanced_sample_weight"] == 1.0
    assert weights.loc["w2", "event_balanced_sample_weight"] == 0.5
    assert weights.loc["w3", "event_balanced_sample_weight"] == 0.0
    assert tables.audit["unique_native_event_count"] == 2
    assert tables.audit["unweighted_event_mass_sum"] == 2.0
    assert tables.audit["event_mass_conservation_error"] == 0.0
    assert tables.audit["errors"] == []


def test_event_weights_reject_valid_window_without_native_event() -> None:
    windows = _windows()
    windows.loc[3, "window_valid_for_main_train"] = True

    with pytest.raises(ValueError, match="valid_windows_without_native_event=1"):
        build_event_weight_manifest(windows)


def test_event_weights_reject_duplicate_window_id() -> None:
    windows = _windows()
    windows.loc[1, "window_id"] = "w0"

    with pytest.raises(ValueError, match="duplicate_window_id_rows=2"):
        build_event_weight_manifest(windows)


def test_event_weights_reject_negative_base_weight() -> None:
    windows = _windows()
    windows.loc[0, "window_sample_weight"] = -1.0

    with pytest.raises(ValueError, match="invalid_base_weight_rows=1"):
        build_event_weight_manifest(windows)


def test_event_weights_audit_legacy_exact_cluster_fallback() -> None:
    windows = _windows().drop(columns="temporal_unit_keys_json")
    windows["temporal_unit_keys_window"] = ["cluster-a", "cluster-ab", "cluster-b", ""]

    tables = build_event_weight_manifest(windows)

    assert tables.audit["event_key_encoding"] == "legacy_exact_cluster_fallback"
    assert any("cannot decompose" in warning for warning in tables.audit["warnings"])


def test_event_weight_artifact_audit_rebuilds_expected_values() -> None:
    windows = _windows()
    weights = build_event_weight_manifest(windows).weights

    audit = audit_event_weight_manifest(weights, windows)

    assert audit["errors"] == []
    assert audit["event_mass_conservation_error"] == 0.0
    assert audit["numeric_mismatch_counts"] == {
        "event_count_window": 0,
        "windows_per_event": 0,
        "valid_windows_per_event": 0,
        "window_sample_weight": 0,
        "inverse_windows_per_event": 0,
        "event_balanced_sample_weight": 0,
    }


def test_event_weight_artifact_audit_rejects_tampered_window_weight() -> None:
    windows = _windows()
    weights = build_event_weight_manifest(windows).weights
    weights.loc[weights["window_id"].eq("w1"), "inverse_windows_per_event"] = 99.0

    audit = audit_event_weight_manifest(weights, windows)

    assert "numeric_mismatch_inverse_windows_per_event=1" in audit["errors"]
