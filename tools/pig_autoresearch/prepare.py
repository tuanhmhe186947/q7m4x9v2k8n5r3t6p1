"""Run the project autoresearch preflight without executing an experiment."""

import json
import os
import subprocess

from harness import (
    ROOT,
    TOOL_DIR,
    _error_observation,
    _load_json,
    build_plan,
    load_policy,
    main,
)


def _probe_adapter() -> int:
    policy = load_policy(TOOL_DIR / "policy.json")
    candidate = _load_json(TOOL_DIR / "candidate.json")
    command = [*build_plan(candidate, policy)["command"], "--help"]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
            timeout=30,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        result = None
    if result is not None and result.returncode == 0:
        return 0
    reason = "adapter_probe_timeout" if result is None else "adapter_probe_failed"
    observation = _error_observation(
        "Autoresearch adapter environment is not ready.",
        reason,
        [str(TOOL_DIR / "policy.json"), str(TOOL_DIR / "candidate.json")],
    )
    if result is not None:
        observation["probe_output"] = result.stdout[-2000:]
    print(json.dumps(observation, ensure_ascii=True, sort_keys=True))
    return 2

if __name__ == "__main__":
    probe_status = _probe_adapter()
    raise SystemExit(probe_status or main(["--dry-run"]))
