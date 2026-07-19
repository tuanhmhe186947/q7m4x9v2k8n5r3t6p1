import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_tracking_repeatability.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_tracking_repeatability_script",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
AUDITOR_CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR_CLI)


def test_tracker_runtime_nested_guardrails_are_applicable() -> None:
    runtime = {
        "guardrails": {
            "tracking_loop_effective_fps": {"status": "PASS"},
            "peak_memory": {"status": "PASS"},
        }
    }

    assert not AUDITOR_CLI._runtime_guardrails_not_applicable(runtime)


def test_post_video_runtime_guardrails_are_not_applicable() -> None:
    runtime = {"guardrails": {"status": "NOT_APPLICABLE"}}

    assert AUDITOR_CLI._runtime_guardrails_not_applicable(runtime)
