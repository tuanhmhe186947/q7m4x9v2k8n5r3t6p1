import json
from pathlib import Path

import pytest

from pig_behavior.classification_v2.experiments.record_contract import check_experiment_record
from pig_behavior.classification_v2.experiments.registry import ExperimentRecordConfig, write_experiment_record


def test_experiment_records_are_immutable_by_default(tmp_path: Path) -> None:
    """Reusing a record name must fail instead of silently replacing lineage."""

    config = ExperimentRecordConfig(name="same_run", output_dir=tmp_path)
    write_experiment_record(config)

    with pytest.raises(FileExistsError, match="already exists"):
        write_experiment_record(config)


def test_paper_record_cannot_enable_overwrite(tmp_path: Path) -> None:
    """Even an explicit overwrite flag cannot mutate a paper-facing record."""

    with pytest.raises(ValueError, match="immutable"):
        write_experiment_record(
            ExperimentRecordConfig(
                name="paper_run",
                output_dir=tmp_path,
                paper_facing=True,
                overwrite=True,
            )
        )


def test_model_paper_record_requires_named_downstream_lineage(tmp_path: Path) -> None:
    """A generic artifact list cannot substitute for semantic Q2 model gates."""

    record_path = tmp_path / "record.json"
    record_path.write_text(
        json.dumps(
            {
                "schema_version": "classification_v2_experiment_record_v2",
                "name": "model",
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "git_commit": "abc",
                "git_dirty": False,
                "artifacts": [],
                "record_path": str(record_path),
                "paper_facing": True,
                "experiment_stage": "paper_facing_candidate",
                "provenance": {},
                "evaluation_contract": {
                    "result_kind": "model_evaluation",
                    "primary_metric_unit": "native_temporal_unit",
                    "split_policy": "recording_group_oof",
                    "external_generalization_claim": False,
                },
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )

    result = check_experiment_record(record_path)

    assert result["valid"] is False
    assert any("run_audit_json" in error for error in result["errors"])
    assert "missing_parent_experiment_records" in result["errors"]
