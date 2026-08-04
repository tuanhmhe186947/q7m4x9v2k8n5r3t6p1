"""Fail-closed autoresearch harness for project-scoped experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = Path(__file__).resolve().parent
METHOD_STATE_PATH = ROOT / ".agents" / "memory" / "13_METHOD_STATE.json"
RUN_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
CANDIDATE_SCHEMA = "pig.autoresearch.candidate.v1"
POLICY_SCHEMA = "pig.autoresearch.policy.v1"
AUTHORIZATION_SCHEMA = "pig.autoresearch.authorization.v1"


class ContractError(ValueError):
    """Raised when an autoresearch contract fails closed."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid_json:{path}:{exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"json_object_required:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _resolve_under(base: Path, relative: str | Path) -> Path:
    resolved_base = base.resolve()
    candidate = Path(relative)
    resolved = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise ContractError(f"path_escape:{resolved}") from exc
    return resolved


def _git_bytes(*args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ContractError(f"git_failed:{' '.join(args)}:{message}")
    return result.stdout


def git_state() -> dict[str, Any]:
    sha = _git_bytes("rev-parse", "HEAD").decode().strip()
    branch = _git_bytes("branch", "--show-current").decode().strip()
    status = _git_bytes(
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    tracked_diff = _git_bytes("diff", "--binary", "HEAD")
    untracked = _git_bytes("ls-files", "--others", "--exclude-standard", "-z")
    digest = hashlib.sha256()
    digest.update(status)
    digest.update(b"\0tracked-diff\0")
    digest.update(tracked_diff)
    for raw_path in sorted(part for part in untracked.split(b"\0") if part):
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        path = ROOT / relative
        digest.update(b"\0untracked\0")
        digest.update(raw_path)
        if path.is_file():
            digest.update(_sha256(path).encode("ascii"))
    return {
        "git_sha": sha,
        "branch": branch,
        "dirty": bool(status),
        "worktree_fingerprint": digest.hexdigest(),
    }


def load_policy(path: Path) -> dict[str, Any]:
    policy = _load_json(path)
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise ContractError("policy_schema_mismatch")
    control_plane = policy.get("control_plane")
    if not isinstance(control_plane, list) or not control_plane:
        raise ContractError("policy_control_plane_required")
    for entry in control_plane:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ContractError("policy_control_plane_entry_invalid")
        relative = entry["path"]
        expected_hash = entry["sha256"]
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ContractError("policy_control_plane_entry_invalid")
        if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
            raise ContractError("policy_control_plane_hash_invalid")
        control_path = _resolve_under(TOOL_DIR, relative)
        if not control_path.is_file():
            raise ContractError(f"control_plane_file_missing:{control_path}")
        if _sha256(control_path) != expected_hash:
            raise ContractError(f"control_plane_hash_mismatch:{relative}")
    if int(policy.get("experiment_budget_seconds", 0)) <= 0:
        raise ContractError("invalid_experiment_budget")
    modes = policy.get("modes")
    if not isinstance(modes, dict) or set(modes) != {"tracking", "classification"}:
        raise ContractError("policy_modes_mismatch")
    return policy


def _validate_number(name: str, value: Any, rule: dict[str, Any]) -> None:
    expected = rule.get("type")
    if expected == "int":
        valid_type = isinstance(value, int) and not isinstance(value, bool)
    elif expected == "float":
        valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        raise ContractError(f"unknown_parameter_type:{name}:{expected}")
    if not valid_type:
        raise ContractError(f"parameter_type_mismatch:{name}")
    numeric = float(value)
    if numeric < float(rule["minimum"]) or numeric > float(rule["maximum"]):
        raise ContractError(f"parameter_out_of_range:{name}:{value}")


def validate_candidate(
    candidate: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "run_tag",
        "mode",
        "stage",
        "method_id",
        "hypothesis",
        "changed_family",
        "parameters",
    }
    if set(candidate) != required:
        raise ContractError("candidate_keys_mismatch")
    if candidate.get("schema_version") != CANDIDATE_SCHEMA:
        raise ContractError("candidate_schema_mismatch")
    run_tag = str(candidate.get("run_tag", ""))
    if not RUN_TAG_RE.fullmatch(run_tag):
        raise ContractError(f"invalid_run_tag:{run_tag}")
    if not str(candidate.get("hypothesis", "")).strip():
        raise ContractError("hypothesis_required")
    mode = str(candidate.get("mode", ""))
    mode_policy = policy["modes"].get(mode)
    if not isinstance(mode_policy, dict):
        raise ContractError(f"unsupported_mode:{mode}")
    stage = str(candidate.get("stage", ""))
    if stage not in mode_policy.get("stages", {}):
        raise ContractError(f"unsupported_stage:{mode}:{stage}")
    family = str(candidate.get("changed_family", ""))
    family_rules = mode_policy.get("families", {}).get(family)
    if not isinstance(family_rules, dict):
        raise ContractError(f"unsupported_family:{mode}:{family}")
    parameters = candidate.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ContractError("candidate_parameters_required")
    maximum = int(policy.get("max_parameters_per_trial", 1))
    if len(parameters) > maximum:
        raise ContractError(f"too_many_parameters:{len(parameters)}>{maximum}")
    for name, value in parameters.items():
        rule = family_rules.get(name)
        if not isinstance(rule, dict):
            raise ContractError(f"parameter_outside_family:{family}:{name}")
        _validate_number(name, value, rule)
    if not str(candidate.get("method_id", "")).strip():
        raise ContractError("method_id_required")
    return mode_policy


def _format_override(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_plan(
    candidate: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    mode_policy = validate_candidate(candidate, policy)
    mode = candidate["mode"]
    stage = candidate["stage"]
    stage_policy = mode_policy["stages"][stage]
    output_root = ROOT / policy["output_root"] / candidate["run_tag"]
    if mode == "tracking":
        fixed = mode_policy["fixed"]
        path_config = _resolve_under(ROOT, fixed["path_config"])
        if not path_config.is_file():
            raise ContractError(f"tracking_path_config_missing:{path_config}")
        if _sha256(path_config) != fixed["path_config_sha256"]:
            raise ContractError("tracking_path_config_hash_mismatch")
        effective_parameters = dict(fixed["baseline_parameters"])
        effective_parameters.update(candidate["parameters"])
        command = [
            sys.executable,
            "-B",
            "-m",
            "pig_behavior.evaluation.tracking.cli",
            "--video",
            ",".join(stage_policy["videos"]),
            "--tracking-mode",
            fixed["tracking_mode"],
            "--path-config",
            str(path_config),
            "--profile",
            fixed["path_profile"],
            "--rule-combo",
            fixed["eval_config"],
            "--evaluator-contract",
            fixed["evaluator_contract"],
            "--gap-tolerance-frames",
            "0",
            "--expected-video-count",
            str(len(stage_policy["videos"])),
            "--no-benchmark-rules",
            "--no-enable-offline-smoothing",
            "--identity-swap-guard",
            "--smooth-boxes",
            "--refine-boxes",
            "--output-root",
            str(output_root),
            "--prediction-root",
            str(output_root / "predictions"),
            "--force-track",
        ]
        for name, value in sorted(effective_parameters.items()):
            command.extend(["--profile-override", f"{name}={_format_override(value)}"])
        effective_config = None
    else:
        command = [
            sys.executable,
            "-B",
            str(
                ROOT
                / "scripts"
                / "classification_v2"
                / "04_baselines_smokes"
                / "check_c2v2_c6_temporal_controls.py"
            ),
            "--config",
            "__EFFECTIVE_CONFIG__",
            "--synthetic-preflight",
        ]
        effective_config = str(output_root / "effective_config.json")
    return {
        "mode": mode,
        "stage": stage,
        "method_id": candidate["method_id"],
        "metric_eligible": bool(stage_policy["metric_eligible"]),
        "allowed_method_states": list(stage_policy["allowed_method_states"]),
        "output_root": str(output_root),
        "command": command,
        "effective_config": effective_config,
        "baseline_parameters": mode_policy.get("fixed", {}).get(
            "baseline_parameters"
        ),
        "effective_parameters": effective_parameters if mode == "tracking" else None,
        "selected_skills": [
            "andrej-karpathy-skills",
            "agent-harness-construction",
            "tracking-experiment-guardian"
            if mode == "tracking"
            else "safe-refactor-test-guardian",
        ],
    }


def _method_entry(method_id: str) -> dict[str, Any] | None:
    payload = _load_json(METHOD_STATE_PATH)
    for entry in payload.get("entries", []):
        if entry.get("method_id") == method_id:
            return entry
    return None


def _parse_time(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ContractError(f"invalid_timestamp:{field}") from exc
    if parsed.tzinfo is None:
        raise ContractError(f"timezone_required:{field}")
    return parsed


def _validate_tracking_authorization(
    authorization: dict[str, Any],
    output_root: Path,
) -> None:
    baseline = _resolve_under(ROOT, authorization.get("baseline_metrics_path", ""))
    if not baseline.is_file():
        raise ContractError(f"baseline_metrics_missing:{baseline}")
    if _sha256(baseline) != authorization.get("baseline_metrics_sha256"):
        raise ContractError("baseline_metrics_hash_mismatch")
    acceptance = authorization.get("acceptance")
    required = {
        "minimum_aggregate_idsw_gain",
        "maximum_per_video_idsw_regression",
        "maximum_aggregate_hota_drop",
    }
    if not isinstance(acceptance, dict) or set(acceptance) != required:
        raise ContractError("tracking_acceptance_contract_mismatch")
    for name, value in acceptance.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ContractError(f"tracking_acceptance_value_invalid:{name}")
    if float(acceptance["minimum_aggregate_idsw_gain"]) <= 0.0:
        raise ContractError("tracking_acceptance_gain_must_be_positive")
    if output_root.exists():
        raise ContractError(f"output_root_not_fresh:{output_root}")


def validate_authorization(
    authorization: dict[str, Any],
    candidate_path: Path,
    policy_path: Path,
    candidate: dict[str, Any],
    policy: dict[str, Any],
    plan: dict[str, Any],
    state: dict[str, Any],
) -> None:
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ContractError("authorization_schema_mismatch")
    if authorization.get("authorized") is not True:
        raise ContractError("authorization_not_granted")
    if authorization.get("consumed") is not False:
        raise ContractError("authorization_already_consumed")
    bindings = {
        "candidate_sha256": _sha256(candidate_path),
        "policy_sha256": _sha256(policy_path),
        "mode": candidate["mode"],
        "stage": candidate["stage"],
        "method_id": candidate["method_id"],
        "git_sha": state["git_sha"],
        "worktree_fingerprint": state["worktree_fingerprint"],
        "experiment_budget_seconds": policy["experiment_budget_seconds"],
    }
    for field, expected in bindings.items():
        if authorization.get(field) != expected:
            raise ContractError(f"authorization_binding_mismatch:{field}")
    if not str(authorization.get("authorization_id", "")).strip():
        raise ContractError("authorization_id_required")
    if not str(authorization.get("reviewer", "")).strip():
        raise ContractError("authorization_reviewer_required")
    if not str(authorization.get("authority", "")).strip():
        raise ContractError("authorization_authority_required")
    issued = _parse_time(authorization.get("issued_at"), "issued_at")
    expires = _parse_time(authorization.get("expires_at"), "expires_at")
    now = datetime.now(expires.tzinfo)
    if issued >= expires or now >= expires:
        raise ContractError("authorization_expired_or_invalid")
    entry = _method_entry(candidate["method_id"])
    if entry is None:
        raise ContractError(f"method_not_registered:{candidate['method_id']}")
    method_state = entry.get("state")
    if authorization.get("expected_method_state") != method_state:
        raise ContractError("authorization_method_state_mismatch")
    if method_state not in plan["allowed_method_states"]:
        raise ContractError(f"method_state_not_allowed:{method_state}")
    output_root = Path(plan["output_root"])
    if candidate["mode"] == "tracking":
        _validate_tracking_authorization(authorization, output_root)
    elif output_root.exists():
        raise ContractError(f"output_root_not_fresh:{output_root}")


def _consume_authorization(path: Path, payload: dict[str, Any]) -> None:
    claim_path = path.with_suffix(f"{path.suffix}.claim")
    try:
        descriptor = os.open(
            claim_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as exc:
        raise ContractError("authorization_already_claimed") from exc
    os.close(descriptor)

    consumed = dict(payload)
    consumed["consumed"] = True
    consumed["consumed_at"] = datetime.now().astimezone().isoformat()
    _atomic_write_json(path, consumed)


def _materialize_classification_config(
    candidate: dict[str, Any],
    policy: dict[str, Any],
    plan: dict[str, Any],
) -> list[str]:
    mode_policy = policy["modes"]["classification"]
    base_path = ROOT / mode_policy["base_config"]
    if _sha256(base_path) != mode_policy["base_config_sha256"]:
        raise ContractError("classification_base_config_hash_mismatch")
    effective = _load_json(base_path)
    optimization = effective.get("optimization")
    if not isinstance(optimization, dict):
        raise ContractError("classification_optimization_missing")
    optimization.update(candidate["parameters"])
    effective_path = Path(plan["effective_config"])
    _atomic_write_json(effective_path, effective)
    return [
        str(effective_path) if part == "__EFFECTIVE_CONFIG__" else part
        for part in plan["command"]
    ]


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _run_command(
    command: list[str],
    log_path: Path,
    timeout_seconds: int,
) -> tuple[int, bool]:
    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH")
    src_path = str(ROOT / "src")
    environment["PYTHONPATH"] = (
        src_path if not current_pythonpath else os.pathsep.join([src_path, current_pythonpath])
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
            **popen_kwargs,
        )
        try:
            return process.wait(timeout=timeout_seconds), False
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            return process.returncode or 124, True


def _heartbeat(
    state_path: Path,
    stop: threading.Event,
    base_state: dict[str, Any],
) -> None:
    while not stop.wait(30):
        payload = dict(base_state)
        payload["last_update"] = datetime.now().astimezone().isoformat()
        _atomic_write_json(state_path, payload)


def _find_tracking_metrics(output_root: Path) -> Path:
    matches = sorted(output_root.rglob("tracking_metrics.csv"))
    if len(matches) != 1:
        raise ContractError(f"tracking_metrics_count:{len(matches)}")
    return matches[0]


def _read_tracking_metrics(path: Path) -> dict[str, dict[str, float]]:
    required = {"video_stem", "remapped_idsw", "remapped_hota_pct"}
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ContractError("tracking_metrics_columns_missing")
        for row in reader:
            key = str(row["video_stem"]).strip()
            if not key:
                raise ContractError("tracking_metric_video_stem_empty")
            if key in rows:
                raise ContractError(f"tracking_metric_video_duplicate:{key}")
            try:
                idsw = float(row["remapped_idsw"])
                hota = float(row["remapped_hota_pct"])
            except (TypeError, ValueError) as exc:
                raise ContractError(f"tracking_metric_not_numeric:{key}") from exc
            if not math.isfinite(idsw) or not math.isfinite(hota):
                raise ContractError(f"tracking_metric_not_finite:{key}")
            if idsw < 0 or not 0.0 <= hota <= 100.0:
                raise ContractError(f"tracking_metric_out_of_domain:{key}")
            rows[key] = {
                "remapped_idsw": idsw,
                "remapped_hota_pct": hota,
            }
    if "ALL" not in rows:
        raise ContractError("tracking_metrics_all_row_missing")
    return rows


def compare_tracking_metrics(
    baseline_path: Path,
    candidate_path: Path,
    videos: list[str],
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    baseline = _read_tracking_metrics(baseline_path)
    candidate = _read_tracking_metrics(candidate_path)
    expected_stems = set(videos) | {"ALL"}
    baseline_stems = set(baseline)
    candidate_stems = set(candidate)
    if baseline_stems != expected_stems or candidate_stems != expected_stems:
        missing = sorted(
            (expected_stems - baseline_stems) | (expected_stems - candidate_stems)
        )
        unexpected = sorted(
            (baseline_stems - expected_stems) | (candidate_stems - expected_stems)
        )
        raise ContractError(
            "tracking_metric_video_set_mismatch:"
            f"missing={','.join(missing)}:unexpected={','.join(unexpected)}"
        )
    aggregate_gain = baseline["ALL"]["remapped_idsw"] - candidate["ALL"]["remapped_idsw"]
    hota_drop = baseline["ALL"]["remapped_hota_pct"] - candidate["ALL"]["remapped_hota_pct"]
    maximum_regression = max(
        candidate[video]["remapped_idsw"] - baseline[video]["remapped_idsw"]
        for video in videos
    )
    gates = {
        "aggregate_idsw_gain": aggregate_gain
        >= float(acceptance["minimum_aggregate_idsw_gain"]),
        "per_video_idsw_non_regression": maximum_regression
        <= float(acceptance["maximum_per_video_idsw_regression"]),
        "aggregate_hota_guardrail": hota_drop
        <= float(acceptance["maximum_aggregate_hota_drop"]),
    }
    return {
        "decision": "keep" if all(gates.values()) else "discard",
        "gates": gates,
        "aggregate_idsw_gain": aggregate_gain,
        "maximum_per_video_idsw_regression": maximum_regression,
        "aggregate_hota_drop": hota_drop,
        "candidate_all": candidate["ALL"],
        "baseline_all": baseline["ALL"],
    }


def _append_result(policy: dict[str, Any], result: dict[str, Any]) -> Path:
    ledger = ROOT / policy["output_root"] / "results.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=True, sort_keys=True))
        handle.write("\n")
    return ledger


def _observation(
    status: str,
    summary: str,
    next_actions: list[str],
    artifacts: list[str],
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "status": status,
        "summary": summary,
        "next_actions": next_actions,
        "artifacts": artifacts,
    }
    payload.update(extra)
    return payload


def _error_observation(
    summary: str,
    root_cause_hint: str,
    artifacts: list[str],
) -> dict[str, Any]:
    return _observation(
        "error",
        summary,
        ["Fix the declared contract and create a new run tag before retrying."],
        artifacts,
        root_cause_hint=root_cause_hint,
        safe_retry="Use dry-run first; never weaken authority or lineage gates.",
        stop_condition="Stop if the same contract or authority gate fails again.",
    )


def preflight(
    candidate_path: Path,
    policy_path: Path,
) -> dict[str, Any]:
    policy = load_policy(policy_path)
    candidate = _load_json(candidate_path)
    plan = build_plan(candidate, policy)
    mode_policy = policy["modes"][candidate["mode"]]
    if candidate["mode"] == "tracking":
        fixed = mode_policy["fixed"]
        path_config = ROOT / fixed["path_config"]
        if _sha256(path_config) != fixed["path_config_sha256"]:
            raise ContractError("tracking_path_config_hash_mismatch")
    else:
        base = ROOT / mode_policy["base_config"]
        if _sha256(base) != mode_policy["base_config_sha256"]:
            raise ContractError("classification_base_config_hash_mismatch")
    entry = _method_entry(candidate["method_id"])
    method_state = None if entry is None else entry.get("state")
    authorization_eligible = method_state in plan["allowed_method_states"]
    state = git_state()
    authorization_request = {
        "schema_version": "pig.autoresearch.authorization-request.v1",
        "candidate_sha256": _sha256(candidate_path),
        "policy_sha256": _sha256(policy_path),
        "git_sha": state["git_sha"],
        "worktree_fingerprint": state["worktree_fingerprint"],
        "mode": candidate["mode"],
        "stage": candidate["stage"],
        "method_id": candidate["method_id"],
        "expected_method_state": method_state,
        "experiment_budget_seconds": policy["experiment_budget_seconds"],
    }
    status = "success" if authorization_eligible else "warning"
    summary = "Autoresearch contracts are valid."
    if not authorization_eligible:
        summary = "Contracts are valid, but execution authority is not ready."
    return _observation(
        status,
        summary,
        [
            "Register the campaign method and issue a single-use authorization."
            if not authorization_eligible
            else "Issue a candidate-bound single-use authorization before execution."
        ],
        [str(candidate_path), str(policy_path), str(METHOD_STATE_PATH)],
        execution_ready=False,
        authorization_eligible=authorization_eligible,
        authorization_request=authorization_request,
        method_state=method_state,
        plan=plan,
    )


def execute(
    candidate_path: Path,
    policy_path: Path,
    authorization_path: Path,
) -> dict[str, Any]:
    if candidate_path.resolve() != (TOOL_DIR / "candidate.json").resolve():
        raise ContractError("execute_requires_canonical_candidate")
    if policy_path.resolve() != (TOOL_DIR / "policy.json").resolve():
        raise ContractError("execute_requires_canonical_policy")
    policy = load_policy(policy_path)
    authorization_root = (ROOT / policy["authorization_root"]).resolve()
    resolved_authorization = authorization_path.resolve()
    try:
        resolved_authorization.relative_to(authorization_root)
    except ValueError as exc:
        raise ContractError("authorization_outside_authority_root") from exc
    candidate = _load_json(candidate_path)
    plan = build_plan(candidate, policy)
    authorization = _load_json(resolved_authorization)
    state_before = git_state()
    validate_authorization(
        authorization,
        candidate_path,
        policy_path,
        candidate,
        policy,
        plan,
        state_before,
    )
    _consume_authorization(resolved_authorization, authorization)
    protected_state = git_state()
    output_root = Path(plan["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    command = list(plan["command"])
    if candidate["mode"] == "classification":
        command = _materialize_classification_config(candidate, policy, plan)
    manifest = {
        "schema_version": "pig.autoresearch.run-manifest.v1",
        "run_tag": candidate["run_tag"],
        "candidate": candidate,
        "candidate_sha256": _sha256(candidate_path),
        "policy_sha256": _sha256(policy_path),
        "authorization_id": authorization["authorization_id"],
        "git_state": protected_state,
        "command": command,
        "plan": plan,
        "started_at": datetime.now().astimezone().isoformat(),
    }
    manifest_path = output_root / "run_manifest.json"
    state_path = output_root / "run_state.json"
    log_path = output_root / "run.log"
    _atomic_write_json(manifest_path, manifest)
    running_state = {
        "schema_version": "pig.autoresearch.run-state.v1",
        "status": "running",
        "run_tag": candidate["run_tag"],
        "last_update": datetime.now().astimezone().isoformat(),
    }
    _atomic_write_json(state_path, running_state)
    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat,
        args=(state_path, heartbeat_stop, running_state),
        daemon=True,
    )
    heartbeat.start()
    return_code, timed_out = _run_command(
        command,
        log_path,
        int(policy["experiment_budget_seconds"]),
    )
    heartbeat_stop.set()
    heartbeat.join(timeout=5)
    artifacts = [str(manifest_path), str(state_path), str(log_path)]
    failure: str | None = None
    if timed_out:
        failure = "experiment_timeout"
    elif return_code != 0:
        failure = f"child_exit_code:{return_code}"
    current_state = git_state()
    if current_state["worktree_fingerprint"] != protected_state["worktree_fingerprint"]:
        failure = "protected_worktree_changed_during_run"
    mp4_files = sorted(str(path) for path in output_root.rglob("*.mp4"))
    if mp4_files:
        failure = f"output_video_forbidden:{len(mp4_files)}"
    decision: dict[str, Any]
    if failure is None and candidate["mode"] == "tracking":
        try:
            metrics_path = _find_tracking_metrics(output_root)
            decision = compare_tracking_metrics(
                ROOT / authorization["baseline_metrics_path"],
                metrics_path,
                policy["modes"]["tracking"]["stages"][candidate["stage"]]["videos"],
                authorization["acceptance"],
            )
            artifacts.append(str(metrics_path))
        except ContractError as exc:
            failure = str(exc)
            decision = {"decision": "invalid"}
    elif failure is None:
        decision = {
            "decision": "diagnostic",
            "claim_boundary": "Synthetic preflight is not model-performance evidence.",
        }
    else:
        decision = {"decision": "crash"}
    result = {
        "schema_version": "pig.autoresearch.run-result.v1",
        "run_tag": candidate["run_tag"],
        "status": "error" if failure else "success",
        "return_code": return_code,
        "timed_out": timed_out,
        "failure": failure,
        "decision": decision,
        "finished_at": datetime.now().astimezone().isoformat(),
        "artifacts": artifacts,
    }
    result_path = output_root / "run_result.json"
    _atomic_write_json(result_path, result)
    ledger = _append_result(policy, result)
    artifacts.extend([str(result_path), str(ledger)])
    final_state = dict(running_state)
    final_state.update(
        {
            "status": "error" if failure else "success",
            "last_update": datetime.now().astimezone().isoformat(),
            "failure": failure,
        }
    )
    _atomic_write_json(state_path, final_state)
    if failure:
        return _error_observation(
            "Autoresearch trial failed closed.",
            failure,
            artifacts,
        )
    return _observation(
        "success",
        f"Trial completed with decision={decision['decision']}.",
        ["Review the run manifest and gate details before changing candidate.json."],
        artifacts,
        decision=decision,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=TOOL_DIR / "candidate.json")
    parser.add_argument("--policy", type=Path, default=TOOL_DIR / "policy.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.execute:
            if args.authorization is None:
                raise ContractError("execute_requires_authorization")
            report = execute(args.candidate, args.policy, args.authorization)
        else:
            report = preflight(args.candidate, args.policy)
    except ContractError as exc:
        report = _error_observation(
            "Autoresearch contract validation failed.",
            str(exc),
            [str(args.candidate), str(args.policy)],
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"success", "warning"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
