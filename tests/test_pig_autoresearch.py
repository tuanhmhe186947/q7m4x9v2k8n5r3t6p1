from __future__ import annotations

import copy
import importlib.util
import json
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest

HARNESS_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "pig_autoresearch" / "harness.py"
)
HARNESS_SPEC = importlib.util.spec_from_file_location(
    "pig_autoresearch_harness",
    HARNESS_PATH,
)
assert HARNESS_SPEC is not None and HARNESS_SPEC.loader is not None
autoresearch = importlib.util.module_from_spec(HARNESS_SPEC)
HARNESS_SPEC.loader.exec_module(autoresearch)

TOOL_DIR = autoresearch.TOOL_DIR
POLICY_PATH = TOOL_DIR / "policy.json"
CANDIDATE_PATH = TOOL_DIR / "candidate.json"


def _policy() -> dict[str, object]:
    return autoresearch.load_policy(POLICY_PATH)


def _tracking_candidate(**updates: object) -> dict[str, object]:
    candidate = autoresearch._load_json(CANDIDATE_PATH)
    candidate.update(updates)
    return candidate


def _classification_candidate(**updates: object) -> dict[str, object]:
    candidate: dict[str, object] = {
        "schema_version": autoresearch.CANDIDATE_SCHEMA,
        "run_tag": f"classification-{uuid.uuid4().hex[:12]}",
        "mode": "classification",
        "stage": "synthetic_preflight",
        "method_id": "c2v2.reviewed_lineage",
        "hypothesis": "A bounded optimization change passes synthetic contracts.",
        "changed_family": "optimization",
        "parameters": {"learning_rate": 0.0002},
    }
    candidate.update(updates)
    return candidate


def _write_metrics(
    path: Path,
    rows: list[tuple[str, float | str, float | str]],
) -> None:
    lines = ["video_stem,remapped_idsw,remapped_hota_pct"]
    lines.extend(f"{stem},{idsw},{hota}" for stem, idsw, hota in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_tracking_plan_uses_supported_fail_closed_cli() -> None:
    plan = autoresearch.build_plan(_tracking_candidate(), _policy())
    command = plan["command"]

    assert command[:5] == [
        sys.executable,
        "-B",
        "-m",
        "pig_behavior.evaluation.tracking.cli",
        "--video",
    ]
    assert "scripts\\evaluate_tracking.py" not in " ".join(command)
    assert "--tracking-mode" in command
    assert "--path-config" in command
    assert "--profile" in command
    assert "--rule-combo" in command
    assert "iou0_area0_condarea0_merge0" in command
    assert "--evaluator-contract" in command
    assert "TRACKING_EVALUATOR_STANDARD_V2" in command
    assert "--expected-video-count" in command
    assert "--no-benchmark-rules" in command
    assert "--profile-override" in command
    assert "det_conf=0.2" in command
    assert command.count("--profile-override") == 6
    assert plan["baseline_parameters"]["det_conf"] == 0.25
    assert plan["effective_parameters"]["det_conf"] == 0.2
    assert plan["effective_parameters"]["track_match_iou"] == 0.8
    assert "--no-mp4" not in command


def test_classification_plan_mutates_only_optimization(tmp_path: Path) -> None:
    policy = _policy()
    candidate = _classification_candidate()
    plan = autoresearch.build_plan(candidate, policy)
    plan["effective_config"] = str(tmp_path / "effective_config.json")

    command = autoresearch._materialize_classification_config(
        candidate,
        policy,
        plan,
    )
    effective = json.loads((tmp_path / "effective_config.json").read_text())
    base_path = autoresearch.ROOT / policy["modes"]["classification"]["base_config"]
    base = json.loads(base_path.read_text(encoding="utf-8"))

    assert effective["optimization"]["learning_rate"] == 0.0002
    assert base["optimization"]["learning_rate"] != 0.0002
    assert command[-1] == "--synthetic-preflight"
    assert "--action" not in command
    assert "training" not in effective or effective["training"] == base.get("training")


def test_candidate_contract_rejects_unsafe_changes() -> None:
    policy = _policy()
    multiple = _tracking_candidate(
        parameters={"det_conf": 0.2, "track_high_conf": 0.4}
    )
    unknown = _tracking_candidate(parameters={"unknown": 1.0})

    with pytest.raises(autoresearch.ContractError, match="too_many_parameters"):
        autoresearch.validate_candidate(multiple, policy)
    with pytest.raises(autoresearch.ContractError, match="parameter_outside_family"):
        autoresearch.validate_candidate(unknown, policy)
    with pytest.raises(autoresearch.ContractError, match="path_escape"):
        autoresearch._resolve_under(autoresearch.ROOT, "../outside.json")


def test_tracking_plan_rejects_path_config_drift() -> None:
    policy = copy.deepcopy(_policy())
    policy["modes"]["tracking"]["fixed"]["path_config_sha256"] = "0" * 64

    with pytest.raises(
        autoresearch.ContractError,
        match="tracking_path_config_hash_mismatch",
    ):
        autoresearch.build_plan(_tracking_candidate(), policy)


def test_policy_rejects_control_plane_drift(tmp_path: Path) -> None:
    policy = copy.deepcopy(_policy())
    policy["control_plane"][0]["sha256"] = "0" * 64
    policy_path = tmp_path / "policy.json"
    autoresearch._atomic_write_json(policy_path, policy)

    with pytest.raises(autoresearch.ContractError, match="control_plane_hash_mismatch"):
        autoresearch.load_policy(policy_path)


def test_preflight_distinguishes_authorization_eligibility() -> None:
    tracking = autoresearch.preflight(CANDIDATE_PATH, POLICY_PATH)

    assert tracking["status"] == "warning"
    assert tracking["execution_ready"] is False
    assert tracking["authorization_eligible"] is False
    assert tracking["method_state"] is None
    assert tracking["authorization_request"]["candidate_sha256"] == (
        autoresearch._sha256(CANDIDATE_PATH)
    )


def test_frozen_tracking_method_cannot_enter_optimization() -> None:
    candidate = _tracking_candidate(method_id="tracking.current_baseline")
    plan = autoresearch.build_plan(candidate, _policy())
    entry = autoresearch._method_entry("tracking.current_baseline")

    assert entry is not None
    assert entry["state"] == "FROZEN"
    assert entry["state"] not in plan["allowed_method_states"]


def test_blocked_classification_is_synthetic_diagnostic_only() -> None:
    candidate = _classification_candidate()
    plan = autoresearch.build_plan(candidate, _policy())
    entry = autoresearch._method_entry("c2v2.reviewed_lineage")

    assert entry is not None
    assert entry["state"] == "BLOCKED"
    assert entry["state"] in plan["allowed_method_states"]
    assert plan["metric_eligible"] is False
    assert plan["command"][-1] == "--synthetic-preflight"


def test_authorization_binding_and_single_use_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    candidate = _classification_candidate()
    candidate_path = tmp_path / "candidate.json"
    policy_path = tmp_path / "policy.json"
    autoresearch._atomic_write_json(candidate_path, candidate)
    autoresearch._atomic_write_json(policy_path, policy)
    plan = autoresearch.build_plan(candidate, policy)
    state = autoresearch.git_state()
    now = datetime.now().astimezone()
    authorization = {
        "schema_version": autoresearch.AUTHORIZATION_SCHEMA,
        "authorization_id": uuid.uuid4().hex,
        "authorized": True,
        "consumed": False,
        "candidate_sha256": autoresearch._sha256(candidate_path),
        "policy_sha256": autoresearch._sha256(policy_path),
        "git_sha": state["git_sha"],
        "worktree_fingerprint": state["worktree_fingerprint"],
        "method_id": candidate["method_id"],
        "mode": candidate["mode"],
        "stage": candidate["stage"],
        "experiment_budget_seconds": policy["experiment_budget_seconds"],
        "expected_method_state": "BLOCKED",
        "reviewer": "pytest",
        "authority": "tests/test_pig_autoresearch.py",
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(minutes=5)).isoformat(),
    }
    monkeypatch.setattr(
        autoresearch,
        "_method_entry",
        lambda method_id: {"method_id": method_id, "state": "BLOCKED"},
    )

    autoresearch.validate_authorization(
        authorization,
        candidate_path,
        policy_path,
        candidate,
        policy,
        plan,
        state,
    )
    authorization_path = tmp_path / "permit.json"
    autoresearch._atomic_write_json(authorization_path, authorization)
    autoresearch._consume_authorization(authorization_path, authorization)

    consumed = autoresearch._load_json(authorization_path)
    assert consumed["consumed"] is True
    assert authorization_path.with_suffix(".json.claim").is_file()
    with pytest.raises(autoresearch.ContractError, match="already_claimed"):
        autoresearch._consume_authorization(authorization_path, authorization)


def test_tracking_metric_decision_enforces_all_guardrails(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    videos = ["video-a", "video-b"]
    acceptance = {
        "minimum_aggregate_idsw_gain": 1,
        "maximum_per_video_idsw_regression": 0,
        "maximum_aggregate_hota_drop": 0.0,
    }
    _write_metrics(
        baseline,
        [("video-a", 3, 90), ("video-b", 3, 90), ("ALL", 6, 90)],
    )
    _write_metrics(
        candidate,
        [("video-a", 2, 90), ("video-b", 3, 90), ("ALL", 5, 90)],
    )
    kept = autoresearch.compare_tracking_metrics(
        baseline,
        candidate,
        videos,
        acceptance,
    )
    assert kept["decision"] == "keep"

    _write_metrics(
        candidate,
        [("video-a", 1, 90), ("video-b", 4, 90), ("ALL", 5, 90)],
    )
    regressed = autoresearch.compare_tracking_metrics(
        baseline,
        candidate,
        videos,
        acceptance,
    )
    assert regressed["decision"] == "discard"
    assert regressed["gates"]["per_video_idsw_non_regression"] is False

    _write_metrics(
        candidate,
        [("video-a", 2, 89), ("video-b", 3, 89), ("ALL", 5, 89)],
    )
    hota_drop = autoresearch.compare_tracking_metrics(
        baseline,
        candidate,
        videos,
        acceptance,
    )
    assert hota_drop["decision"] == "discard"
    assert hota_drop["gates"]["aggregate_hota_guardrail"] is False

    _write_metrics(
        candidate,
        [
            ("video-a", 2, 90),
            ("video-b", 3, 90),
            ("video-extra", 0, 100),
            ("ALL", 5, 90),
        ],
    )
    with pytest.raises(autoresearch.ContractError, match="video_set_mismatch"):
        autoresearch.compare_tracking_metrics(
            baseline,
            candidate,
            videos,
            acceptance,
        )


def test_tracking_authorization_rejects_invalid_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(autoresearch, "ROOT", tmp_path)
    baseline = tmp_path / "baseline.csv"
    _write_metrics(baseline, [("video-a", 1, 90), ("ALL", 1, 90)])
    authorization = {
        "baseline_metrics_path": str(baseline),
        "baseline_metrics_sha256": autoresearch._sha256(baseline),
        "acceptance": {
            "minimum_aggregate_idsw_gain": False,
            "maximum_per_video_idsw_regression": 0,
            "maximum_aggregate_hota_drop": 0.0,
        },
    }

    with pytest.raises(autoresearch.ContractError, match="acceptance_value_invalid"):
        autoresearch._validate_tracking_authorization(
            authorization,
            tmp_path / "fresh-output",
        )


@pytest.mark.parametrize("bad_value", ["nan", "inf", "-1"])
def test_tracking_metric_reader_rejects_invalid_values(
    tmp_path: Path,
    bad_value: str,
) -> None:
    metrics = tmp_path / "tracking_metrics.csv"
    _write_metrics(metrics, [("video-a", bad_value, 90), ("ALL", 1, 90)])

    with pytest.raises(autoresearch.ContractError, match="tracking_metric"):
        autoresearch._read_tracking_metrics(metrics)


def test_tracking_metric_reader_rejects_duplicate_rows(tmp_path: Path) -> None:
    metrics = tmp_path / "tracking_metrics.csv"
    _write_metrics(
        metrics,
        [("video-a", 1, 90), ("video-a", 1, 90), ("ALL", 1, 90)],
    )

    with pytest.raises(autoresearch.ContractError, match="video_duplicate"):
        autoresearch._read_tracking_metrics(metrics)


def test_subprocess_timeout_is_enforced(tmp_path: Path) -> None:
    return_code, timed_out = autoresearch._run_command(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        tmp_path / "run.log",
        0.1,
    )

    assert timed_out is True
    assert return_code != 0


def test_error_observation_has_recovery_fields() -> None:
    observation = autoresearch._error_observation(
        "Autoresearch trial failed closed.",
        "contract_error",
        ["artifact.json"],
    )

    assert observation["status"] == "error"
    assert observation["summary"]
    assert observation["next_actions"]
    assert observation["artifacts"] == ["artifact.json"]
    assert observation["root_cause_hint"] == "contract_error"
    assert observation["safe_retry"]
    assert observation["stop_condition"]
