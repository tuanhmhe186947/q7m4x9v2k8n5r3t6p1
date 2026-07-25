from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pig_behavior.classification_v2.evaluation import (
    legacy_development_goal_completion as completion,
)

CONFIG_PATH = Path(
    "configs/classification_v2/legacy_development_goal_completion_v1.json"
)
OPTIONAL_GOAL_REASON = (
    "OPTIONAL_EXTERNAL_LEGACY_GOAL_BUNDLE_UNAVAILABLE:"
    "supply every hash-bound L0-L8 legacy-development artifact"
)


def _bound_paths(value: object) -> list[Path]:
    if isinstance(value, dict):
        paths = []
        if "path" in value and "sha256" in value:
            paths.append(Path(str(value["path"])))
        for child in value.values():
            paths.extend(_bound_paths(child))
        return paths
    if isinstance(value, list):
        paths = []
        for child in value:
            paths.extend(_bound_paths(child))
        return paths
    return []


def _goal_bundle_available() -> bool:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    required = [
        *_bound_paths(payload["frozen_inputs"]),
        *_bound_paths(payload["milestones"]),
    ]
    try:
        for path in required:
            with path.open("rb") as handle:
                handle.read(1)
    except OSError:
        return False
    return True


def test_checked_in_completion_config_is_fail_closed() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    completion._validate_config(payload)

    changed = copy.deepcopy(payload)
    changed["q2_claim_allowed"] = True
    with pytest.raises(ValueError, match="completion config boundary"):
        completion._validate_config(changed)


def test_bound_file_rejects_hash_drift(tmp_path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    spec = {
        "path": artifact.name,
        "sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        completion._bound_file(tmp_path, spec, "artifact")


def test_common_milestone_boundary_rejects_claim_drift() -> None:
    payload = {
        "status": "PASS_TEST",
        "lineage_scope": "legacy-only-unreviewed-development",
        "human_review_complete": False,
        "q2_claim_allowed": True,
        "canonical_full_oof_authorized": False,
        "reviewed_or_final_claim_allowed": False,
        "outer_holdout_predictions_authorized": False,
        "errors": [],
        "valid": True,
    }

    with pytest.raises(ValueError, match="q2_claim_allowed mismatch"):
        completion._validate_common(payload, "PASS_TEST", "test")


@pytest.mark.skipif(
    not _goal_bundle_available(),
    reason=OPTIONAL_GOAL_REASON,
)
def test_completion_dry_run_proves_current_l0_l8_evidence() -> None:
    result = completion.write_legacy_goal_completion_audit(
        CONFIG_PATH,
        enforce_git_guard=False,
        write_output=False,
    )

    assert result["status"] == "PASS_LEGACY_16F_GOAL_COMPLETION"
    assert result["goal_complete"] is True
    assert result["parent_reviewed_all_source_goal_complete"] is False
    assert [item["milestone"] for item in result["requirements"]] == [
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "L6",
        "L7",
        "L8",
    ]
    assert all(item["passed"] for item in result["requirements"])
    assert result["selected_run_lineage"]["fold_outputs_isolated"] is True
    assert result["selected_run_lineage"][
        "outer_holdout_predictions_created"
    ] == 0
    assert result["reviewed_or_final_claim_allowed"] is False
    assert result["q2_claim_allowed"] is False


def test_tracked_authority_has_no_forbidden_source_alias() -> None:
    audit = completion._forbidden_alias_audit(Path.cwd())

    assert audit["occurrences"] == []
    assert audit["valid"] is True
