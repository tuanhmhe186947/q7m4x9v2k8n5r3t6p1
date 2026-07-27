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
ROLE_ASSIGNMENTS = DESIGN_DIR / "H1_R2_VALIDATION_ROLE_ASSIGNMENTS.json"
ACTIVATION_GATE = DESIGN_DIR / "H1_R2_ACTIVATION_GATE_DECISION.json"
INDEPENDENT_REVIEW = DESIGN_DIR / "H1_R2_INDEPENDENT_DESIGN_REVIEW.json"
AUTHORIZATION = (
    DESIGN_DIR / "H1_R2_IMPLEMENTATION_AUTHORIZATION_DECISION.json"
)

BASE_MANIFEST_COLUMNS = {
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
VALIDATION_MANIFEST_COLUMNS = BASE_MANIFEST_COLUMNS | {
    "assignment_artifact_sha256"
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
VALIDATION_WINDOWS = {
    "V01_000085_blinded": ("Pigs281119_000085_30fps", 360, 540),
    "V02_000114_blinded": ("Pigs281119_000114_30fps", 960, 1140),
    "V03_000327_blinded": ("Pigs301119_000327_30fps", 660, 840),
    "V04_000328_blinded": ("Pigs301119_000328_30fps", 1260, 1440),
}
VALIDATION_ROLES = {
    "positive_hidden_owner_contention",
    "control_no_hidden_owner_contention",
    "ambiguous_exclude",
}
PRE_ASSIGNMENT_VALIDATION_SHA256 = (
    "979197b95eeeb407cd998ac256c5484c78de57ac876e09ffd6152230205c28b2"
)


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


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ContractError(f"cannot hash {path}: {exc}") from exc


def _authority_path(value: str) -> Path:
    path = Path(value)
    if path.is_file() or path.is_absolute():
        return path
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode == 0:
        common_checkout_path = Path(result.stdout.strip()).parent / path
        if common_checkout_path.is_file():
            return common_checkout_path
    return path


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
    if not _authority_path(value).is_file():
        raise ContractError(f"missing source authority path: {value}")


def _check_manifest(
    path: Path,
    expected_columns: set[str],
) -> list[dict[str, str]]:
    columns, rows = _read_csv(path)
    if set(columns) != expected_columns:
        raise ContractError(f"{path} manifest columns are incomplete")
    hash_mismatches: list[str] = []
    for row in rows:
        episode = row["episode_id"]
        for field in expected_columns - {"frozen_hash"}:
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
    for row in validation:
        eligible = row["validation_eligible"].lower()
        if row["positive_control_role"] == "ambiguous_exclude":
            if eligible != "false":
                raise ContractError("ambiguous validation row remains eligible")
        elif eligible != "true":
            raise ContractError("assigned validation row is not eligible")
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


def _check_role_assignments(
    validation: list[dict[str, str]],
) -> None:
    package = _read_json(ROLE_ASSIGNMENTS)
    if package.get("pre_assignment_validation_manifest_sha256") != (
        PRE_ASSIGNMENT_VALIDATION_SHA256
    ):
        raise ContractError("pre-assignment validation manifest hash changed")
    if package.get("boundaries_unchanged") is not True:
        raise ContractError("role assignment does not preserve boundaries")
    blindness = package.get("blindness_contract", {})
    for key in (
        "tracking_run_performed",
        "gpu_inference_performed",
        "h1_r2_outputs_viewed",
    ):
        if blindness.get(key) is not False:
            raise ContractError(f"role assignment blindness failed: {key}")
    package_sha256 = _file_sha256(ROLE_ASSIGNMENTS)
    if {
        row["assignment_artifact_sha256"] for row in validation
    } != {package_sha256}:
        raise ContractError("validation manifest role artifact hash mismatch")
    assignment_rows = package.get("assignments")
    if not isinstance(assignment_rows, list):
        raise ContractError("role assignments must be a list")
    assignments = {row.get("episode_id"): row for row in assignment_rows}
    if set(assignments) != set(VALIDATION_WINDOWS):
        raise ContractError("role assignment population changed")
    manifest_rows = {row["episode_id"]: row for row in validation}
    if set(manifest_rows) != set(VALIDATION_WINDOWS):
        raise ContractError("validation manifest population changed")
    for episode_id, (video_key, start, end) in VALIDATION_WINDOWS.items():
        assignment = assignments[episode_id]
        manifest = manifest_rows[episode_id]
        expected_boundary = (video_key, start, end)
        assignment_boundary = (
            assignment.get("video_key"),
            assignment.get("start_frame"),
            assignment.get("end_frame"),
        )
        manifest_boundary = (
            manifest["video_key"],
            int(manifest["start_frame"]),
            int(manifest["end_frame"]),
        )
        if assignment_boundary != expected_boundary:
            raise ContractError(f"{episode_id}: assigned boundary changed")
        if manifest_boundary != expected_boundary:
            raise ContractError(f"{episode_id}: manifest boundary changed")
        role = assignment.get("assigned_role")
        if role not in VALIDATION_ROLES:
            raise ContractError(f"{episode_id}: invalid assigned role")
        if manifest["positive_control_role"] != role:
            raise ContractError(f"{episode_id}: manifest role mismatch")
        for manifest_key, assignment_key in (
            ("video_sha256", "source_video_sha256"),
            ("gt_sha256", "gt_sha256"),
        ):
            if manifest[manifest_key] != assignment.get(assignment_key):
                raise ContractError(f"{episode_id}: evidence hash mismatch")
        if not HEX64.fullmatch(str(assignment.get("parent_evidence_sha256"))):
            raise ContractError(f"{episode_id}: invalid parent evidence hash")
        if not str(assignment.get("rationale", "")).strip():
            raise ContractError(f"{episode_id}: missing assignment rationale")


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


def _check_activation_gate(weights: dict[str, float]) -> None:
    activation = _read_json(ACTIVATION_GATE)
    for key, expected in (
        ("development_data_only", True),
        ("validation_data_used", False),
        ("threshold_frozen", True),
        ("score_calibrated", False),
        ("score_is_probability", False),
        ("validation_execution_may_change_gate", False),
        ("evaluation_authorized", False),
        ("promotion_authorized", False),
    ):
        if activation.get(key) is not expected:
            raise ContractError(f"activation gate changed: {key}")
    if activation.get("score_name") != "owner_preference_score":
        raise ContractError("activation gate score terminology changed")
    score = activation.get("score", {})
    if score.get("weights") != weights:
        raise ContractError("activation weights differ from feature registry")
    frozen = activation.get("frozen_activation_gate", {})
    expected_constants = {
        "activation_threshold": 0.6,
        "minimum_hidden_quality_margin_over_visible": 0.2,
        "minimum_detection_confidence": 0.25,
        "minimum_hidden_overlap_similarity": 0.5,
        "minimum_optional_evidence_count_per_candidate": 1,
        "track_freshness_max_detection_opportunities": 5,
        "maximum_hidden_stale_detection_opportunities": 5,
        "requires_selected_visible_competitor": True,
        "numeric_comparison_tolerance": 1e-12,
    }
    for key, expected in expected_constants.items():
        if frozen.get(key) != expected:
            raise ContractError(f"activation constant changed: {key}")
    if frozen.get("optional_evidence_set") != [
        "appearance_available",
        "motion_available",
    ]:
        raise ContractError("activation optional evidence set changed")
    if not frozen.get("abstention_behavior"):
        raise ContractError("activation abstention behavior is incomplete")
    development_rows = activation.get("development_episode_screening")
    if not isinstance(development_rows, list):
        raise ContractError("development screening table is missing")
    if {row.get("episode_id") for row in development_rows} != H1_R1_EPISODES:
        raise ContractError("development screening population changed")
    if any(
        row.get("h1_r2_features_available") is not False
        or row.get("activation") != "NOT_COMPUTABLE_PREIMPLEMENTATION"
        for row in development_rows
    ):
        raise ContractError("preimplementation screening claim changed")
    threshold_rows = activation.get("golden_case_threshold_screening")
    if not isinstance(threshold_rows, list):
        raise ContractError("golden threshold screening is missing")
    if [row.get("threshold") for row in threshold_rows] != [
        0.55,
        0.575,
        0.6,
        0.625,
        0.65,
    ]:
        raise ContractError("golden threshold screening grid changed")


def _check_independent_review() -> None:
    review = _read_json(INDEPENDENT_REVIEW)
    if review.get("review_result") != "PASS":
        raise ContractError("independent review did not pass")
    if review.get("independent_from_authoring_process") is not True:
        raise ContractError("review independence is not established")
    if review.get("implementation_authorized") is not True:
        raise ContractError("review does not authorize implementation")
    for key in ("evaluation_authorized", "promotion_authorized"):
        if review.get(key) is not False:
            raise ContractError(f"independent review authorizes {key}")
    checks = review.get("checks")
    if not isinstance(checks, list):
        raise ContractError("independent review checks are missing")
    if {check.get("id") for check in checks} != set(range(1, 13)):
        raise ContractError("independent review check coverage is incomplete")
    if any(check.get("result") != "PASS" for check in checks):
        raise ContractError("independent review contains a failed check")


def _check_authorization() -> bool:
    if not AUTHORIZATION.is_file():
        return False
    authorization = _read_json(AUTHORIZATION)
    if authorization.get("current_main_sha") != (
        "e807865d2c12cdf15ed82d57788b805063cd46f6"
    ):
        raise ContractError("authorization source main SHA changed")
    decisions = authorization.get("authorizations", {})
    if decisions.get("implementation_authorized") is not True:
        raise ContractError("implementation authorization is inconsistent")
    for key in (
        "evaluation_authorized",
        "runtime_evaluation_authorized",
        "promotion_authorized",
    ):
        if decisions.get(key) is not False:
            raise ContractError(f"authorization decision authorizes {key}")
    if authorization.get("h1_r2_tracking_results_viewed") is not False:
        raise ContractError("authorization viewed H1-r2 tracking results")
    bindings = authorization.get("sha256_bindings", {})
    expected_bindings = {
        "design_contract": CONTRACT,
        "feature_registry": FEATURES,
        "golden_cases": GOLDEN,
        "development_manifest": DEVELOPMENT,
        "validation_manifest": VALIDATION,
        "validation_role_assignment": ROLE_ASSIGNMENTS,
        "activation_gate": ACTIVATION_GATE,
        "evaluation_gate": GATE,
        "independent_review": INDEPENDENT_REVIEW,
        "checker": Path(__file__),
    }
    if set(bindings) != set(expected_bindings):
        raise ContractError("authorization hash-binding set is incomplete")
    for key, path in expected_bindings.items():
        if bindings[key] != _file_sha256(path):
            raise ContractError(f"authorization hash mismatch: {key}")
    return True


def _check_paths_and_terminology() -> None:
    required = (
        CONTRACT,
        FEATURES,
        GOLDEN,
        DEVELOPMENT,
        VALIDATION,
        GATE,
        DECISION,
        ROLE_ASSIGNMENTS,
        ACTIVATION_GATE,
        INDEPENDENT_REVIEW,
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
        development = _check_manifest(DEVELOPMENT, BASE_MANIFEST_COLUMNS)
        validation = _check_manifest(
            VALIDATION,
            VALIDATION_MANIFEST_COLUMNS,
        )
        _check_separation(development, validation)
        _check_role_assignments(validation)
        _check_gate_and_decision()
        _check_activation_gate(weights)
        _check_independent_review()
        implementation_authorized = _check_authorization()
    except (ContractError, ValueError) as exc:
        print(f"H1_R2_DESIGN_CHECKER=FAIL: {exc}", file=sys.stderr)
        return 1
    ambiguous_roles = sum(
        row["positive_control_role"] == "ambiguous_exclude"
        for row in validation
    )
    print("H1_R2_DESIGN_CHECKER=PASS")
    print(f"FEATURE_COUNT={len(weights)}")
    print(f"GOLDEN_CASE_COUNT={len(GOLDEN_CASES)}")
    print(f"DEVELOPMENT_EPISODE_COUNT={len(development)}")
    print(f"VALIDATION_EPISODE_COUNT={len(validation)}")
    print("VALIDATION_PENDING_ROLE_COUNT=0")
    print(f"VALIDATION_AMBIGUOUS_EXCLUDED_COUNT={ambiguous_roles}")
    authorized = "YES" if implementation_authorized else "NO"
    print(f"IMPLEMENTATION_AUTHORIZED={authorized}")
    print("EVALUATION_AUTHORIZED=NO")
    print("PROMOTION_AUTHORIZED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
