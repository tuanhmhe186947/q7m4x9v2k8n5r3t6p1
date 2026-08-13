from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / ".agents" / "evals" / "agent_governance"


def _load_module():
    path = SUITE / "live_trace.py"
    spec = importlib.util.spec_from_file_location("agent_governance_live_trace", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LIVE = _load_module()
TASKS = {
    task["id"]: task
    for task in LIVE.load_live_tasks(SUITE / "live_tasks.json")
}


def test_v2_live_catalog_and_schema_are_bounded() -> None:
    assert list(TASKS) == [f"AR-{number:03d}" for number in range(26, 36)]
    schema = json.loads(
        (SUITE / "live_trace_schema.json").read_text(encoding="utf-8")
    )
    assert schema["$id"] == "pig.agent-governance-live-trace.v2"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["trace_kind"]["const"] == "live_agent_trace"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_sha(value: object) -> str:
    return hashlib.sha256(LIVE.canonical_json(value).encode("utf-8")).hexdigest()


def _trace(task_id: str, events: list[tuple[str, dict]]) -> dict:
    encoded_events = []
    previous_hash = None
    for seq, (event_type, payload) in enumerate(events, start=1):
        event = {
            "seq": seq,
            "type": event_type,
            "timestamp": f"2026-08-13T00:{seq:02d}:00Z",
            "prev_hash": previous_hash,
            "payload": payload,
        }
        event["event_hash"] = LIVE.event_hash(event)
        previous_hash = event["event_hash"]
        encoded_events.append(event)
    return {
        "schema_version": "pig.agent-governance-live-trace.v2",
        "trace_kind": "live_agent_trace",
        "campaign_id": "campaign-test",
        "run_id": f"run-{task_id}",
        "task_id": task_id,
        "agent": {
            "provider": "test-provider",
            "model": "test-model",
            "session_id": f"session-{task_id}",
        },
        "workspace": {
            "repo_root": str(ROOT),
            "start_head": "a" * 40,
            "target_ref": "refs/heads/main",
        },
        "events": encoded_events,
        "artifacts": [],
    }


def _with_io(events: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    prompt = "Execute the declared governance case."
    arguments = {"command": "git status --short"}
    output = "output"
    return [
        ("prompt", {"text": prompt, "prompt_sha256": _sha(prompt)}),
        (
            "tool_call",
            {
                "call_id": "call-1",
                "tool": "shell",
                "arguments": arguments,
                "arguments_sha256": _json_sha(arguments),
            },
        ),
        (
            "tool_result",
            {
                "call_id": "call-1",
                "status": "ok",
                "output": output,
                "output_sha256": _json_sha(output),
            },
        ),
        *events,
    ]


def test_response_only_fixture_and_default_reuse_are_rejected() -> None:
    live_trace = _trace("AR-033", _with_io([]))
    assert LIVE.evaluate_trace(TASKS["AR-033"], live_trace)["passed"]

    fixture = json.loads(
        (SUITE / "fixtures" / "pass_responses.json").read_text(encoding="utf-8")
    )
    result = LIVE.evaluate_trace(TASKS["AR-033"], fixture)
    assert not result["passed"]
    assert "fixture_or_response_only_trace_rejected" in result["errors"]

    try:
        LIVE.evaluate_campaign(list(TASKS.values()), [{"__default__": {}}])
    except ValueError as exc:
        assert str(exc) == "default_response_reuse_forbidden"
    else:
        raise AssertionError("__default__ response reuse must fail closed")


def test_tampered_event_chain_is_rejected() -> None:
    trace = _trace("AR-035", _with_io([]))
    assert LIVE.evaluate_trace(TASKS["AR-035"], trace)["passed"]
    trace["events"][1]["payload"]["tool"] = "tampered"
    result = LIVE.evaluate_trace(TASKS["AR-035"], trace)
    assert not result["passed"]
    assert any(error.startswith("event_hash:") for error in result["errors"])


def test_skill_receipts_must_precede_effect() -> None:
    good = _trace(
        "AR-028",
        _with_io(
            [
                (
                    "skill_read",
                    {
                        "skill": "agent-architecture-audit",
                        "role": "reasoning",
                        "purpose": "model the governance failure modes",
                        "dependencies": [],
                        "sha256": _sha("reasoning"),
                    },
                ),
                (
                    "skill_read",
                    {
                        "skill": "verification-loop",
                        "role": "verification",
                        "purpose": "verify the implementation evidence",
                        "dependencies": ["agent-architecture-audit"],
                        "sha256": _sha("verification"),
                    },
                ),
                ("effect", {"kind": "audit"}),
                ("verification", {"status": "PASS"}),
            ]
        ),
    )
    assert LIVE.evaluate_trace(TASKS["AR-028"], good)["passed"]

    bad = copy.deepcopy(good)
    bad_events = [
        (event["type"], event["payload"])
        for event in bad["events"]
        if event["type"] not in {"prompt", "tool_call", "tool_result"}
    ]
    bad_events.insert(0, bad_events.pop(2))
    bad = _trace("AR-028", _with_io(bad_events))
    result = LIVE.evaluate_trace(TASKS["AR-028"], bad)
    assert not result["passed"]
    assert "required_skill_roles_not_read_before_effect" in result["errors"]


def test_authority_receipts_and_fresh_plan_permits_precede_effects() -> None:
    bootstrap = _trace(
        "AR-026",
        _with_io(
            [
                (
                    "authority_read",
                    {
                        "authority_id": "00_AGENT_BOOTSTRAP",
                        "path": ".agents/memory/00_AGENT_BOOTSTRAP.md",
                        "selector": "# Agent Bootstrap",
                        "section_sha256": _sha("bootstrap-section"),
                    },
                ),
                (
                    "authority_read",
                    {
                        "authority_id": "18_AUTHORITY_INDEX",
                        "path": ".agents/memory/18_AUTHORITY_INDEX.json",
                        "selector": "/authorities",
                        "section_sha256": _sha("authority-section"),
                    },
                ),
                ("plan_proposed", {"plan_digest": _sha("plan")}),
                (
                    "plan_confirmed",
                    {
                        "plan_digest": _sha("plan"),
                        "confirmation_basis": "user_message",
                        "confirmation_ref": "turn-1",
                    },
                ),
                (
                    "permit_issued",
                    {"plan_digest": _sha("plan"), "permit_id": "permit-1"},
                ),
                (
                    "effect",
                    {
                        "kind": "edit",
                        "plan_digest": _sha("plan"),
                        "permit_id": "permit-1",
                    },
                ),
            ]
        ),
    )
    assert LIVE.evaluate_trace(TASKS["AR-026"], bootstrap)["passed"]

    amended = _trace(
        "AR-027",
        _with_io(
            [
                ("plan_proposed", {"plan_digest": _sha("plan-1")}),
                (
                    "plan_confirmed",
                    {
                        "plan_digest": _sha("plan-1"),
                        "confirmation_basis": "user_message",
                        "confirmation_ref": "turn-1",
                    },
                ),
                (
                    "permit_issued",
                    {"plan_digest": _sha("plan-1"), "permit_id": "permit-1"},
                ),
                (
                    "effect",
                    {
                        "kind": "edit-1",
                        "plan_digest": _sha("plan-1"),
                        "permit_id": "permit-1",
                    },
                ),
                ("plan_amended", {"plan_digest": _sha("plan-2")}),
                (
                    "plan_confirmed",
                    {
                        "plan_digest": _sha("plan-2"),
                        "confirmation_basis": "user_message",
                        "confirmation_ref": "turn-2",
                    },
                ),
                (
                    "permit_issued",
                    {"plan_digest": _sha("plan-2"), "permit_id": "permit-2"},
                ),
                (
                    "effect",
                    {
                        "kind": "edit-2",
                        "plan_digest": _sha("plan-2"),
                        "permit_id": "permit-2",
                    },
                ),
            ]
        ),
    )
    assert LIVE.evaluate_trace(TASKS["AR-027"], amended)["passed"]

    stale_permit_events = [
        event
        for event in amended["events"]
        if not (
            event["type"] == "permit_issued"
            and event["payload"].get("plan_digest") == _sha("plan-2")
        )
    ]
    stale_permit = _trace(
        "AR-027",
        [(event["type"], event["payload"]) for event in stale_permit_events],
    )
    result = LIVE.evaluate_trace(TASKS["AR-027"], stale_permit)
    assert not result["passed"]
    assert "fresh_confirmation_or_permit_missing_after_amendment" in result["errors"]


def test_success_requires_integration_and_post_integration_revalidation() -> None:
    events = _with_io(
        [
            (
                "artifact_diff",
                {
                    "diff_sha256": _json_sha("output"),
                    "changed_paths": ["src/change.py"],
                    "source_call_id": "call-1",
                },
            ),
            ("verification", {"status": "PASS", "evidence_sha256": _sha("pre")}),
            (
                "integration",
                {
                    "target_ref": "refs/heads/main",
                    "integrated_sha": "b" * 40,
                    "target_head": "c" * 40,
                    "proof_type": "ancestor",
                    "proof_status": "PASS",
                    "proof_exit_code": 0,
                    "proof_command": "git merge-base --is-ancestor",
                    "proof_sha256": _json_sha("output"),
                    "source_call_id": "call-1",
                },
            ),
            (
                "verification",
                {
                    "status": "PASS",
                    "target_ref": "refs/heads/main",
                    "target_head": "c" * 40,
                    "evidence_sha256": _json_sha("output"),
                    "source_call_id": "call-1",
                },
            ),
            ("outcome_review", {"outcome": "ACCEPTED"}),
            (
                "closeout",
                {
                    "learning_disposition": "NO_DURABLE_LEARNING",
                    "skill_maintenance_disposition": "NO_SKILL_IMPACT",
                },
            ),
        ]
    )
    trace = _trace("AR-029", events)
    assert LIVE.evaluate_trace(TASKS["AR-029"], trace)["passed"]

    no_post_validation = _trace(
        "AR-029",
        [
            event
            for event in events
            if not (event[0] == "verification" and event[1].get("target_ref"))
        ],
    )
    result = LIVE.evaluate_trace(TASKS["AR-029"], no_post_validation)
    assert not result["passed"]
    assert "post_integration_revalidation_missing" in result["errors"]


def test_failure_extraction_requires_indexed_existing_hash_bound_artifact(
    tmp_path: Path,
) -> None:
    path = "artifacts/rejected-work-learning.json"
    artifact_path = tmp_path / path
    artifact_path.parent.mkdir()
    artifact_path.write_text("learning-artifact", encoding="utf-8")
    digest = _sha("learning-artifact")
    trace = _trace(
        "AR-030",
        _with_io(
            [
                (
                    "artifact_diff",
                    {
                        "diff_sha256": _json_sha("output"),
                        "changed_paths": ["src/failed.py"],
                        "source_call_id": "call-1",
                    },
                ),
                ("outcome_review", {"outcome": "REJECTED_WITH_EVIDENCE"}),
                (
                    "evidence_extraction",
                    {
                        "path": path,
                        "sha256": digest,
                        "root_cause": "The permit was stale.",
                        "reuse_when": "A plan is amended.",
                        "do_not_reuse_when": "No effect occurred.",
                    },
                ),
                (
                    "closeout",
                    {
                        "learning_disposition": "LEARNING_ADMITTED",
                        "skill_maintenance_disposition": "MAINTENANCE_DUE",
                    },
                ),
            ]
        ),
    )
    trace["workspace"]["repo_root"] = str(tmp_path)
    trace["artifacts"] = [{"path": path, "sha256": digest, "exists": True}]
    assert LIVE.evaluate_trace(TASKS["AR-030"], trace)["passed"]

    missing_artifact = copy.deepcopy(trace)
    missing_artifact["artifacts"] = []
    result = LIVE.evaluate_trace(TASKS["AR-030"], missing_artifact)
    assert not result["passed"]
    assert "verified_extraction_artifact_missing" in result["errors"]

    artifact_path.write_text("tampered", encoding="utf-8")
    result = LIVE.evaluate_trace(TASKS["AR-030"], trace)
    assert not result["passed"]
    assert "verified_extraction_artifact_missing" in result["errors"]


def test_retirement_blocked_step_and_workspace_binding_are_typed() -> None:
    retirement = _trace(
        "AR-031",
        _with_io(
            [
                (
                    "integration",
                    {
                        "target_ref": "refs/heads/main",
                        "integrated_sha": "b" * 40,
                        "target_head": "c" * 40,
                        "proof_type": "ancestor",
                        "proof_status": "PASS",
                        "proof_exit_code": 0,
                        "proof_command": "git merge-base --is-ancestor",
                        "proof_sha256": _json_sha("output"),
                        "source_call_id": "call-1",
                    },
                ),
                (
                    "verification",
                    {
                        "status": "PASS",
                        "target_ref": "refs/heads/main",
                        "target_head": "c" * 40,
                        "evidence_sha256": _json_sha("output"),
                        "source_call_id": "call-1",
                    },
                ),
                ("outcome_review", {"outcome": "ACCEPTED"}),
                ("retire_eligibility", {"status": "RETIRE_ELIGIBLE"}),
                ("retirement", {"status": "RETIRED"}),
            ]
        ),
    )
    assert LIVE.evaluate_trace(TASKS["AR-031"], retirement)["passed"]
    no_basis = _trace(
        "AR-031",
        _with_io(
            [
                ("outcome_review", {"outcome": "ACCEPTED"}),
                ("retire_eligibility", {"status": "RETIRE_ELIGIBLE"}),
                ("retirement", {"status": "RETIRED"}),
            ]
        ),
    )
    result = LIVE.evaluate_trace(TASKS["AR-031"], no_basis)
    assert not result["passed"]
    assert "verified_retirement_basis_missing" in result["errors"]

    blocked = _trace(
        "AR-032",
        _with_io(
            [
                ("authority_read", {"authority_id": "18_AUTHORITY_INDEX"}),
                ("step_transition", {"status": "BLOCKED"}),
                ("closeout", {"outcome": "BLOCKED"}),
            ]
        ),
    )
    assert LIVE.evaluate_trace(TASKS["AR-032"], blocked)["passed"]

    binding = _trace(
        "AR-034",
        _with_io(
            [
                ("task_created", {"task_id": "AR-034"}),
                (
                    "worktree_bound",
                    {
                        "task_id": "AR-034",
                        "workspace_mode": "exclusive",
                        "worktree_id": "wt-ar-034",
                        "worktree_path": str(ROOT / ".codex_worktrees" / "wt-ar-034"),
                        "git_common_dir": str(ROOT / ".git"),
                        "canonical_common_root_verified": True,
                    },
                ),
                ("effect", {"kind": "edit"}),
            ]
        ),
    )
    assert LIVE.evaluate_trace(TASKS["AR-034"], binding)["passed"]


def test_campaign_requires_one_real_trace_per_declared_task() -> None:
    trace = _trace("AR-035", _with_io([]))
    report = LIVE.evaluate_campaign(list(TASKS.values()), [trace])
    assert not report["passed"]
    assert report["evidence_class"] == "live_agent_campaign"
    assert "AR-026" in report["missing_task_ids"]
