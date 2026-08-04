"""Run the deterministic agent governance regression suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from judge import load_tasks, score_repetitions

ROOT = Path(__file__).resolve().parents[3]
SUITE = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def worktree_fingerprint() -> str:
    status = git_value("status", "--porcelain=v1", "--untracked-files=all")
    return hashlib.sha256(status.encode("utf-8")).hexdigest()


def load_responses(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "responses" not in payload:
        raise ValueError("responses_file_missing_responses")
    return payload["responses"]


def response_for(
    responses: dict[str, Any],
    task_id: str,
    run_index: int,
) -> dict[str, Any]:
    value = responses.get(task_id, responses.get("__default__"))
    if value is None:
        raise ValueError(f"response_missing:{task_id}")
    if isinstance(value, list):
        if run_index >= len(value):
            raise ValueError(f"response_run_missing:{task_id}:{run_index + 1}")
        value = value[run_index]
    if not isinstance(value, dict):
        raise TypeError(f"response_not_object:{task_id}")
    return value


def verify_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    files = {
        "task_sha256": SUITE / "tasks.json",
        "judge_sha256": SUITE / "judge.py",
        "runner_sha256": SUITE / "run_regression.py",
    }
    for field, path in files.items():
        expected = manifest.get(field)
        actual = file_sha256(path)
        if not expected or expected == "PENDING":
            errors.append(f"manifest_unpinned:{field}")
        elif expected != actual:
            errors.append(f"manifest_hash_mismatch:{field}")
    return errors


def build_report(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    input_path: Path,
) -> dict[str, Any]:
    status = "PASS" if metrics["pass_power_3"] == 1.0 else "FAIL"
    return {
        "schema_version": "pig.agent-governance-report.v1",
        "suite_id": manifest["suite_id"],
        "status": status,
        "subject_type": manifest["subject_type"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "input_responses": str(input_path),
        "git_sha": git_value("rev-parse", "HEAD"),
        "dirty_worktree": bool(git_value("status", "--porcelain")),
        "dirty_worktree_fingerprint": worktree_fingerprint(),
        "manifest": manifest,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=SUITE / "manifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs must be at least 3")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_errors = verify_manifest(manifest)
    if manifest_errors:
        print(json.dumps({"status": "FAIL", "errors": manifest_errors}))
        return 1
    tasks = load_tasks(SUITE / "tasks.json")
    responses = load_responses(args.responses)
    responses_by_run = [
        {
            task["id"]: response_for(responses, task["id"], run_index)
            for task in tasks
        }
        for run_index in range(args.runs)
    ]
    metrics = score_repetitions(tasks, responses_by_run)
    report = build_report(manifest, metrics, args.responses)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
