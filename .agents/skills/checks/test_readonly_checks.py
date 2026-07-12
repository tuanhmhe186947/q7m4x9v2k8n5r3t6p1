"""Focused positive and fail-closed tests for project-local skill checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import audit_feature_leakage
import audit_model_forward_shapes
import audit_native_unit_uniqueness
import audit_prediction_count
import audit_split_overlap

SKILL_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = SKILL_ROOT / "examples" / "synthetic"


class ReadOnlyCheckTests(unittest.TestCase):
    """Prove valid fixtures pass and critical leakage defects fail closed."""

    def test_valid_synthetic_contracts_pass(self) -> None:
        split = audit_split_overlap.audit(FIXTURES / "fold_manifest.csv")
        features = audit_feature_leakage.audit(FIXTURES / "feature_whitelist.json")
        native = audit_native_unit_uniqueness.audit(
            FIXTURES / "native_units.csv",
            "temporal_unit_key",
        )
        predictions = audit_prediction_count.audit(
            FIXTURES / "evaluation_manifest.csv",
            FIXTURES / "predictions.csv",
            "temporal_unit_key",
        )
        shapes = audit_model_forward_shapes.audit(
            FIXTURES / "forward_shape_spec.json",
            FIXTURES / "forward_shapes_observed.json",
        )
        for report in (split, features, native, predictions, shapes):
            self.assertEqual(report["errors"], [])

    def test_target_feature_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.json"
            path.write_text('{"features": ["cx_n", "behavior_label"]}', encoding="utf-8")
            report = audit_feature_leakage.audit(path)
        self.assertTrue(report["errors"])
        self.assertEqual(report["forbidden_features"], ["behavior_label"])

    def test_group_role_overlap_fails_closed(self) -> None:
        text = (
            "outer_fold_id,role,recording_group_id,video_key,temporal_unit_key\n"
            "fold_00,train,date_a,video_a,unit_a\n"
            "fold_00,test,date_a,video_a,unit_a\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.csv"
            path.write_text(text, encoding="utf-8")
            report = audit_split_overlap.audit(path)
        self.assertTrue(report["errors"])
        self.assertTrue(report["overlaps"])

    def test_missing_prediction_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            path.write_text("temporal_unit_key,y_pred\nunit_a,stand\n", encoding="utf-8")
            report = audit_prediction_count.audit(
                FIXTURES / "evaluation_manifest.csv",
                path,
                "temporal_unit_key",
            )
        self.assertTrue(report["errors"])
        self.assertTrue(report["missing_keys"])


if __name__ == "__main__":
    unittest.main()
