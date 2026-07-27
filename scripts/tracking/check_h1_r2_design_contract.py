"""Deterministically validate the design-only H1-r2 scientific contract."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

DESIGN_DIR = Path("docs/tracking/h1_r2")
FEATURES = DESIGN_DIR / "H1_R2_FEATURE_SEMANTICS_REGISTRY.csv"
GOLDEN = DESIGN_DIR / "H1_R2_GOLDEN_CASES.yaml"
DEVELOPMENT = DESIGN_DIR / "H1_R2_DEVELOPMENT_MANIFEST.csv"
VALIDATION = DESIGN_DIR / "H1_R2_VALIDATION_MANIFEST.csv"
GATE = DESIGN_DIR / "H1_R2_EVALUATION_GATE.json"
DECISION = DESIGN_DIR / "H1_R2_DESIGN_DECISION.json"
CONTRACT = DESIGN_DIR / "H1_R2_SCIENTIFIC_DESIGN_CONTRACT.md"

MANIFEST_COLUMNS = {
    "episode_id",
    "video_key",
    "recording_date_session",
    "start_frame",
    "end_frame",
    "episode_category",
    "positive_control_role",
    "selection_rationale",
    "source_authority",
    "gt_authority_status",
    "validation_eligible",
    "reason",
    "video_sha256",
    "gt_sha256",
    "frozen_hash",
}
FEATURE_COLUMNS = {
    "feature_id",
    "formula",
    "range",
    "normalization",
    "monotonic_direction",
    "missing_data_behavior",
    "clipping_policy",
    "hidden_available",
    "visible_available",
    "symmetric_definition",
    "causal",
    "skipped_frame_dependency",
    "primary_weight",
    "role",
}
H1_R1_EPISODES = {
    "E01_000233_contention_a",
    "E02_000233_contention_b",
    "E03_000233_crossing",
    "E04_000263_contention",
    "E05_000263_control_clean",
    "E06_000233_control_clean",
}
GOLDEN_CASES = {
    "hidden_owner_clearly_stronger",
    "visible_competitor_clearly_stronger",
    "identical_evidence",
    "missing_appearance_hidden_only",
    "missing_appearance_both",
    "long_hidden_weak_owner",
    "recent_hidden_strong_geometry",
    "bbox_scale_invariance_large_and_small",
    "skipped_detector_frame_valid_lk",
    "invalid_motion_hidden_only",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PROHIBITED_SCORE_NAMES = re.compile(r"\b(p_owner|owner_probability)\b")


class ContractError(RuntimeError):
    """Raised when the frozen design packet is incomplete or inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain one object")
    return value


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
    except OSError as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise ContractError(f"{path} must contain at least one row")
    return columns, rows


def _row_hash(row: dict[str, str]) -> str:
    payload = {
        key: value for key, value in row.items() if key != "frozen_hash"
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _check_source_authority(value: str) -> None:
    if value.startswith("git:"):
        spec = value.removeprefix("git:")
        result = subprocess.run(
            ["git", "cat-file", "-e", spec],
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise ContractError(f"unresolvable Git source authority: {spec}")
        return
    if value.startswith("PENDING:"):
        return
    if not Path(value).is_file():
        raise ContractError(f"missing source authority path: {value}")


def _check_manifest(
    path: Path,
) -> list[dict[str, str]]:
    columns, rows = _read_csv(path)
    if set(columns) != MANIFEST_COLUMNS:
        raise ContractError(f"{path} manifest columns are incomplete")
    hash_mismatches: list[str] = []
    for row in rows:
        episode = row["episode_id"]
        for field in MANIFEST_COLUMNS - {"frozen_hash"}:
            if not row[field].strip():
                raise ContractError(f"{episode}: empty {field}")
        for field in ("video_sha256", "gt_sha256"):
            if not HEX64.fullmatch(row[field]):
                raise ContractError(f"{episode}: invalid {field}")
        expected = _row_hash(row)
        if row["frozen_hash"] != expected:
            print(f"EXPECTED_HASH {episode} {expected}")
            hash_mismatches.append(episode)
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        if start < 0 or end <= start:
            raise ContractError(f"{episode}: invalid frame interval")
        _check_source_authority(row["source_authority"])
    if hash_mismatches:
        raise ContractError(
            f"frozen_hash mismatch: {', '.join(hash_mismatches)}"
        )
    return rows


def _check_features() -> dict[str, float]:
    columns, rows = _read_csv(FEATURES)
    if set(columns) != FEATURE_COLUMNS:
        raise ContractError("feature registry columns are incomplete")
    seen: set[str] = set()
    weights: dict[str, float] = {}
    for row in rows:
        feature_id = row["feature_id"]
        if feature_id in seen:
            raise ContractError(f"duplicate feature: {feature_id}")
        seen.add(feature_id)
        for field in (
            "formula",
            "range",
            "normalization",
            "monotonic_direction",
            "missing_data_behavior",
            "clipping_policy",
            "skipped_frame_dependency",
        ):
            if not row[field].strip():
                raise ContractError(f"{feature_id}: empty {field}")
        for field in (
            "hidden_available",
            "visible_available",
            "symmetric_definition",
            "causal",
        ):
            if row[field].lower() != "true":
                raise ContractError(f"{feature_id}: {field} must be true")
        weight = float(row["primary_weight"])
        if weight < 0.0:
            raise ContractError(f"{feature_id}: negative weight")
        if row["role"] == "score":
            weights[feature_id] = weight
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise ContractError("primary score weights must sum to one")
    return weights


def _check_golden(weights: dict[str, float]) -> None:
    golden = _read_json(GOLDEN)
    if golden.get("score_name") != "owner_preference_score":
        raise ContractError("golden score name is incorrect")
    if golden.get("calculated_independently_from_production") is not True:
        raise ContractError("golden cases lack independent calculation authority")
    golden_weights = golden.get("weights")
    if golden_weights != weights:
        raise ContractError("golden weights differ from feature registry")
    cases = golden.get("cases")
    if not isinstance(cases, list):
        raise ContractError("golden cases must be a list")
    if {case.get("id") for case in cases} != GOLDEN_CASES:
        raise ContractError("golden case coverage is incomplete")
    ordered_weights = list(golden_weights.values())
    tolerance = float(golden.get("tolerance", 0.0))
    for case in cases:
        hidden = [float(value) for value in case["hidden"]]
        visible = [float(value) for value in case["visible"]]
        if len(hidden) != len(ordered_weights) or len(visible) != len(
            ordered_weights
        ):
            raise ContractError(f"{case['id']}: feature vector length mismatch")
        if any(value < 0.0 or value > 1.0 for value in hidden + visible):
            raise ContractError(f"{case['id']}: unbounded feature value")
        hidden_q = sum(
            weight * value
            for weight, value in zip(ordered_weights, hidden, strict=True)
        )
        visible_q = sum(
            weight * value
            for weight, value in zip(ordered_weights, visible, strict=True)
        )
        score = min(1.0, max(0.0, 0.5 + 0.5 * (hidden_q - visible_q)))
        for actual, field in (
            (hidden_q, "expected_hidden_Q"),
            (visible_q, "expected_visible_Q"),
            (score, "expected_score"),
        ):
            if abs(actual - float(case[field])) > tolerance:
                raise ContractError(f"{case['id']}: incorrect {field}")


def _check_separation(
    development: list[dict[str, str]],
    validation: list[dict[str, str]],
) -> None:
    if {row["episode_id"] for row in development} != H1_R1_EPISODES:
        raise ContractError("development must contain the six H1-r1 episodes")
    for row in development:
        if row["validation_eligible"].lower() != "false":
            raise ContractError("H1-r1 episode marked validation eligible")
        if row["reason"] != "used_for_h1_r1_design_and_cost_audit":
            raise ContractError("H1-r1 validation exclusion reason changed")
    if any(row["validation_eligible"].lower() != "true" for row in validation):
        raise ContractError("validation row is not marked eligible")
    development_videos = {row["video_key"] for row in development}
    validation_videos = {row["video_key"] for row in validation}
    if development_videos & validation_videos:
        raise ContractError("development and validation video overlap")
    development_sessions = {
        row["recording_date_session"] for row in development
    }
    validation_sessions = {
        row["recording_date_session"] for row in validation
    }
    if development_sessions & validation_sessions:
        raise ContractError("development and validation session overlap")


def _check_gate_and_decision() -> None:
    gate = _read_json(GATE)
    decision = _read_json(DECISION)
    if gate.get("implementation_authorized") is not False:
        raise ContractError("evaluation gate authorizes implementation")
    if gate.get("evaluation_authorized") is not False:
        raise ContractError("evaluation gate authorizes evaluation")
    causality = gate.get("causality", {})
    if (
        causality.get("output_timing_contract") != "causal_framewise"
        or causality.get("output_delay_frames") != 0
        or causality.get("uses_future_frames") is not False
        or causality.get("prefix_invariance_required") is not True
    ):
        raise ContractError("causality gate is incomplete")
    quality = gate.get("quality_guardrails", {})
    for key in (
        "new_permanent_swap_allowed",
        "new_terminal_swap_allowed",
        "per_video_idsw_delta_maximum",
    ):
        if quality.get(key) != 0:
            raise ContractError(f"quality gate changed: {key}")
    if gate.get("artifact_guardrails", {}).get(
        "recursive_run_root_mp4_count"
    ) != 0:
        raise ContractError("no-MP4 gate is missing")
    primary = decision.get("primary_design", {})
    if primary.get("score_name") != "owner_preference_score":
        raise ContractError("primary score terminology changed")
    if primary.get("calibrated") is not False:
        raise ContractError("uncalibrated score marked calibrated")
    if primary.get("can_be_called_probability") is not False:
        raise ContractError("uncalibrated score called a probability")
    authorizations = decision.get("authorizations", {})
    for key in (
        "implementation_authorized",
        "evaluation_authorized",
        "promotion_authorized",
    ):
        if authorizations.get(key) is not False:
            raise ContractError(f"design decision authorizes {key}")


def _check_paths_and_terminology() -> None:
    required = (
        CONTRACT,
        FEATURES,
        GOLDEN,
        DEVELOPMENT,
        VALIDATION,
        GATE,
        DECISION,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ContractError(f"missing design deliverables: {missing}")
    text = "\n".join(path.read_text(encoding="utf-8") for path in required)
    match = PROHIBITED_SCORE_NAMES.search(text)
    if match:
        raise ContractError(
            f"prohibited uncalibrated score terminology: {match.group(0)}"
        )


def main() -> int:
    try:
        _check_paths_and_terminology()
        weights = _check_features()
        _check_golden(weights)
        development = _check_manifest(DEVELOPMENT)
        validation = _check_manifest(VALIDATION)
        _check_separation(development, validation)
        _check_gate_and_decision()
    except (ContractError, ValueError) as exc:
        print(f"H1_R2_DESIGN_CHECKER=FAIL: {exc}", file=sys.stderr)
        return 1
    pending_roles = sum(
        row["positive_control_role"].startswith("PENDING")
        for row in validation
    )
    print("H1_R2_DESIGN_CHECKER=PASS")
    print(f"FEATURE_COUNT={len(weights)}")
    print(f"GOLDEN_CASE_COUNT={len(GOLDEN_CASES)}")
    print(f"DEVELOPMENT_EPISODE_COUNT={len(development)}")
    print(f"VALIDATION_EPISODE_COUNT={len(validation)}")
    print(f"VALIDATION_PENDING_ROLE_COUNT={pending_roles}")
    print("IMPLEMENTATION_AUTHORIZED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
