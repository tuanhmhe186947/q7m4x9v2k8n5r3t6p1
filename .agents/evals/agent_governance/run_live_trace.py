"""Evaluate captured live-agent traces; never synthesize missing runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from live_trace import evaluate_campaign, load_live_tasks

SUITE = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=SUITE / "live_tasks.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.traces.read_text(encoding="utf-8"))
    if payload.get("evidence_class") != "live_agent_campaign_input":
        raise SystemExit("input is not a live agent campaign trace bundle")
    campaign_id = payload.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise SystemExit("live campaign input must declare campaign_id")
    traces = payload.get("traces")
    if not isinstance(traces, list):
        raise SystemExit("live campaign input must contain a traces array")
    if any(
        not isinstance(trace, dict) or trace.get("campaign_id") != campaign_id
        for trace in traces
    ):
        raise SystemExit("trace campaign_id does not match bundle")
    report = evaluate_campaign(load_live_tasks(args.tasks), traces)
    report["campaign_id"] = campaign_id
    report["suite_binding"] = {
        "tasks_sha256": _sha256(args.tasks),
        "trace_schema_sha256": _sha256(SUITE / "live_trace_schema.json"),
        "evaluator_sha256": _sha256(SUITE / "live_trace.py"),
        "runner_sha256": _sha256(Path(__file__).resolve()),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
