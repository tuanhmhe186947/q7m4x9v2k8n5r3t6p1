from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / ".agents" / "evals" / "agent_governance"


def _load_judge():
    path = SUITE / "judge.py"
    spec = importlib.util.spec_from_file_location("agent_governance_judge", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(name: str) -> dict:
    path = SUITE / "fixtures" / name
    return json.loads(path.read_text(encoding="utf-8"))["responses"]


def _responses_for_tasks(tasks: list[dict], source: dict) -> dict[str, dict]:
    default = source["__default__"]
    return {task["id"]: default for task in tasks}


def test_pass_fixture_scores_three_stable_runs() -> None:
    judge = _load_judge()
    tasks = judge.load_tasks(SUITE / "tasks.json")
    responses = _responses_for_tasks(tasks, _fixture("pass_responses.json"))

    report = judge.score_repetitions(tasks, [responses, responses, responses])

    assert report["runs"] == 3
    assert report["task_count"] == 25
    assert report["pass_at_1"] == 1.0
    assert report["pass_at_3"] == 1.0
    assert report["pass_power_3"] == 1.0
    assert report["pass_rate"] == 1.0
    assert report["pass@1"] == 1.0
    assert report["pass@3"] == 1.0
    assert report["pass^3"] == 1.0
    assert report["consistency_rate"] == 1.0
    assert report["results"]["AR-015"][0]["checks"]["checklist_discipline"]
    assert report["results"]["AR-016"][0]["checks"]["rollover_routing"]
    assert report["results"]["AR-017"][0]["checks"]["concurrent_task_safety"]
    assert report["results"]["AR-018"][0]["checks"]["crash_recovery"]
    assert report["results"]["AR-019"][0]["checks"]["atomic_task_safety"]
    assert report["results"]["AR-020"][0]["checks"]["multi_day_resume_safety"]
    assert report["results"]["AR-021"][0]["checks"]["memory_maturity_safety"]
    assert report["results"]["AR-022"][0]["checks"]["memory_maturity_safety"]
    assert report["results"]["AR-023"][0]["checks"]["memory_maturity_safety"]
    assert report["results"]["AR-024"][0]["checks"]["ownership_recovery_safety"]
    assert report["results"]["AR-025"][0]["checks"]["ownership_recovery_safety"]


def test_fail_fixture_detects_claim_and_cleanup_overreach() -> None:
    judge = _load_judge()
    tasks = judge.load_tasks(SUITE / "tasks.json")
    responses = _responses_for_tasks(tasks, _fixture("fail_responses.json"))

    report = judge.score_repetitions(tasks, [responses, responses, responses])

    assert report["pass_at_1"] < 1.0
    assert report["pass_power_3"] < 1.0
    claim_result = report["results"]["AR-007"][0]
    cleanup_result = report["results"]["AR-005"][0]
    checklist_result = report["results"]["AR-015"][0]
    rollover_result = report["results"]["AR-016"][0]
    concurrent_result = report["results"]["AR-017"][0]
    recovery_result = report["results"]["AR-018"][0]
    atomic_result = report["results"]["AR-019"][0]
    resume_result = report["results"]["AR-020"][0]
    maturity_result = report["results"]["AR-021"][0]
    revalidation_result = report["results"]["AR-022"][0]
    dual_authority_result = report["results"]["AR-023"][0]
    same_thread_result = report["results"]["AR-024"][0]
    ambiguous_owner_result = report["results"]["AR-025"][0]
    assert claim_result["checks"]["claim_boundary"] is False
    assert cleanup_result["checks"]["cleanup_safety"] is False
    assert checklist_result["checks"]["checklist_discipline"] is False
    assert rollover_result["checks"]["rollover_routing"] is False
    assert concurrent_result["checks"]["concurrent_task_safety"] is False
    assert recovery_result["checks"]["crash_recovery"] is False
    assert atomic_result["checks"]["atomic_task_safety"] is False
    assert resume_result["checks"]["multi_day_resume_safety"] is False
    assert maturity_result["checks"]["memory_maturity_safety"] is False
    assert revalidation_result["checks"]["memory_maturity_safety"] is False
    assert dual_authority_result["checks"]["memory_maturity_safety"] is False
    assert same_thread_result["checks"]["ownership_recovery_safety"] is False
    assert ambiguous_owner_result["checks"]["ownership_recovery_safety"] is False
