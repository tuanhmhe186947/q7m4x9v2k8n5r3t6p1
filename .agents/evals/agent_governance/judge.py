"""Deterministic judge for agent governance responses."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
VALIDATOR_PATH = (
    ROOT
    / ".agents"
    / "skills"
    / "project-state-steward"
    / "scripts"
    / "validate_governance_contracts.py"
)
SPEC = importlib.util.spec_from_file_location("governance_contracts", VALIDATOR_PATH)
assert SPEC is not None
assert SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def _items(response: dict[str, Any], field: str) -> set[str]:
    value = response.get(field, [])
    return set(value) if isinstance(value, list) else set()


def _fields_present(response: dict[str, Any], fields: list[str]) -> bool:
    return all(response.get(field) not in (None, "", [], {}) for field in fields)


def _criterion_result(
    criterion: dict[str, Any],
    response: dict[str, Any],
) -> bool:
    """Evaluate a declarative criterion without scenario-ID branching."""
    kind = criterion.get("type")
    if kind == "fields_present":
        return _fields_present(response, criterion.get("fields", []))
    if kind == "actions_all":
        return set(criterion.get("values", [])) <= _items(response, "actions")
    if kind == "actions_any":
        return bool(set(criterion.get("values", [])) & _items(response, "actions"))
    if kind == "observation_contract":
        observation = response.get(criterion.get("field", "observation"), {})
        return isinstance(observation, dict) and not VALIDATOR.validate_observation(
            observation
        )
    if kind == "checklist_contract":
        checklist = response.get(criterion.get("field", "task_checklist"), {})
        if not isinstance(checklist, dict):
            return False
        task_status = checklist.get("task_status")
        active_count = checklist.get("active_step_count")
        return (
            isinstance(task_status, str)
            and task_status in VALIDATOR.CHECKLIST_STATES
            and isinstance(active_count, int)
            and active_count <= criterion.get("max_active_step_count", 1)
            and checklist.get("done_steps_have_evidence") is True
            and checklist.get("open_steps_have_next_action") is True
        )
    raise ValueError(f"unknown_criterion_type:{kind}")


def judge_case(task: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    required_authorities = set(task.get("required_authorities", []))
    required_actions = set(task.get("required_actions", []))
    required_skills = set(task.get("required_skills", []))
    checks["authority_recall"] = required_authorities <= _items(
        response,
        "authority_refs",
    )
    checks["required_actions"] = required_actions <= _items(response, "actions")
    checks["reasoning_skill_selection"] = required_skills <= _items(
        response,
        "selected_skills",
    )
    forbidden = _items(response, "actions") & set(
        task.get("forbidden_actions", [])
    )
    checks["no_forbidden_action"] = not forbidden
    checks["required_fields"] = _fields_present(
        response,
        task.get("required_fields", []),
    )
    expected_claim = task.get("expected_claim_status")
    checks["claim_boundary"] = expected_claim is None or (
        response.get("claim_status") == expected_claim
    )
    expected_halt = task.get("expected_halt")
    checks["halt_behavior"] = expected_halt is None or (
        response.get("halt") is expected_halt
    )
    for criterion in task.get("criteria", []):
        name = criterion.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"criterion_missing_name:{task['id']}")
        if name in checks:
            raise ValueError(f"criterion_name_collision:{task['id']}:{name}")
        checks[name] = _criterion_result(criterion, response)
    passed = all(checks.values())
    return {
        "task_id": task["id"],
        "passed": passed,
        "checks": checks,
        "forbidden_observed": sorted(forbidden),
    }


def load_tasks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["tasks"]


def score_repetitions(
    tasks: list[dict[str, Any]],
    responses_by_run: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    results: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        task_results = []
        for responses in responses_by_run:
            task_results.append(judge_case(task, responses[task["id"]]))
        results[task["id"]] = task_results
    passed_values = [
        result["passed"]
        for task_results in results.values()
        for result in task_results
    ]
    first_passed = [
        task_results[0]["passed"] for task_results in results.values()
    ]
    any_passed = [
        any(result["passed"] for result in task_results)
        for task_results in results.values()
    ]
    all_passed = [
        all(result["passed"] for result in task_results)
        for task_results in results.values()
    ]
    consistent = [
        len({result["passed"] for result in task_results}) == 1
        for task_results in results.values()
    ]
    dimension_rates: dict[str, float] = {}
    dimension_names = set(
        check
        for task_results in results.values()
        for result in task_results
        for check in result["checks"]
    )
    for dimension in sorted(dimension_names):
        values = [
            result["checks"][dimension]
            for task_results in results.values()
            for result in task_results
            if dimension in result["checks"]
        ]
        dimension_rates[dimension] = sum(values) / len(values)
    pass_rate = sum(passed_values) / len(passed_values)
    pass_at_1 = sum(first_passed) / len(first_passed)
    pass_at_3 = sum(any_passed) / len(any_passed)
    pass_power_3 = sum(all_passed) / len(all_passed)
    return {
        "runs": len(responses_by_run),
        "task_count": len(tasks),
        "pass_rate": pass_rate,
        "pass@1": pass_at_1,
        "pass@3": pass_at_3,
        "pass^3": pass_power_3,
        "trial_pass_rate": pass_rate,
        "pass_at_1": pass_at_1,
        "pass_at_3": pass_at_3,
        "pass_power_3": pass_power_3,
        "consistency_rate": sum(consistent) / len(consistent),
        "dimension_rates": dimension_rates,
        "results": results,
    }
