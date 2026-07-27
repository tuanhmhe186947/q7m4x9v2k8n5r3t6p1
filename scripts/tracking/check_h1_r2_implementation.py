"""Fail-closed checker for the frozen H1-r2 implementation contract."""

from __future__ import annotations

# ruff: noqa: E402
import ast
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.owner_preference import (
    OWNER_PREFERENCE_MAX_DETECTION_OPPORTUNITIES,
    OWNER_PREFERENCE_MIN_DETECTION_CONFIDENCE,
    OWNER_PREFERENCE_MIN_HIDDEN_OVERLAP,
    OWNER_PREFERENCE_MIN_QUALITY_MARGIN,
    OWNER_PREFERENCE_SCORE_NAME,
    OWNER_PREFERENCE_THRESHOLD,
    OWNER_PREFERENCE_WEIGHTS,
    OwnerPreferenceFeatures,
    appearance_similarity,
    motion_consistency,
    normalized_center_similarity,
    overlap_similarity,
    owner_preference_score,
    scale_similarity,
    track_freshness,
)
from pig_behavior.tracking.profiles.realtime import (
    EVAL_CONFIGS,
    PRESENTATION_PROFILES,
    REALTIME_FAST_CONFIG,
    REALTIME_FAST_H1_R2_CONFIG,
)
from pig_behavior.tracking.telemetry import resolve_output_timing_contract

DESIGN_DIR = ROOT / "docs" / "tracking" / "h1_r2"
FEATURE_REGISTRY = DESIGN_DIR / "H1_R2_FEATURE_SEMANTICS_REGISTRY.csv"
EVALUATION_GATE = DESIGN_DIR / "H1_R2_EVALUATION_GATE.json"
AUTHORIZATION = (
    DESIGN_DIR / "H1_R2_IMPLEMENTATION_AUTHORIZATION_DECISION.json"
)
PROFILE_AMENDMENT = (
    DESIGN_DIR / "H1_R2_PROFILE_ADDITION_AUTHORIZATION_DECISION.json"
)
OWNER_SOURCE = ROOT / "src" / "pig_behavior" / "tracking" / "owner_preference.py"
ASSOCIATION_SOURCE = (
    ROOT / "src" / "pig_behavior" / "tracking" / "association.py"
)
AUTHORIZED_MAIN_SHA = "8b55038c66a0fbb7c9b3388ed61ba6f60699be4b"
AUTHORIZED_PROFILE_NAME = "realtime_fast_h1_r2"
EXPECTED_REALTIME_FAST_HASH = (
    "9bf4ce6d07423ab517b4705c716e3eb012349b756b7c0591cc3458eac207808d"
)
EXPECTED_CONSTANTS = {
    "threshold": 0.60,
    "minimum_quality_margin": 0.20,
    "minimum_detection_confidence": 0.25,
    "minimum_hidden_overlap": 0.50,
    "maximum_detection_opportunities": 5,
}
FORBIDDEN_NAMES = re.compile(r"\b(p_owner|owner_probability)\b", re.IGNORECASE)
FORBIDDEN_RUNTIME_METADATA = (
    "video_key",
    "episode_id",
    "validation_role",
    "positive_hidden_owner_contention",
    "control_no_hidden_owner_contention",
    "ground_truth",
)


class ImplementationError(RuntimeError):
    """Raised when production code diverges from frozen H1-r2 authority."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ImplementationError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ImplementationError(f"{path} must contain one object")
    return value


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _historical_profiles() -> dict[str, object]:
    relative = "src/pig_behavior/tracking/profiles/realtime.py"
    result = subprocess.run(
        ["git", "show", f"{AUTHORIZED_MAIN_SHA}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        raise ImplementationError("authorized profile source is unavailable")
    namespace: dict[str, object] = {}
    exec(compile(result.stdout, relative, "exec"), namespace)  # noqa: S102
    return namespace


def _registry_weights() -> dict[str, float]:
    try:
        with FEATURE_REGISTRY.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ImplementationError(f"cannot read feature registry: {exc}") from exc
    score_rows = [row for row in rows if row["role"] == "score"]
    weights = {
        row["feature_id"]: float(row["primary_weight"]) for row in score_rows
    }
    if len(weights) != 8:
        raise ImplementationError("production contract must contain eight features")
    for row in score_rows:
        expected_range = (
            "{0,1}"
            if row["feature_id"].endswith("_available")
            else "[0,1]"
        )
        required = {
            "range": expected_range,
            "hidden_available": "true",
            "visible_available": "true",
            "symmetric_definition": "true",
            "causal": "true",
        }
        if any(row[key] != value for key, value in required.items()):
            raise ImplementationError(f"feature contract changed: {row['feature_id']}")
        if not row["missing_data_behavior"].strip():
            raise ImplementationError(
                f"missingness is undefined: {row['feature_id']}"
            )
    return weights


def _check_features_and_score() -> None:
    registry_weights = _registry_weights()
    if registry_weights != OWNER_PREFERENCE_WEIGHTS:
        raise ImplementationError("production weights differ from registry")
    production_fields = tuple(OwnerPreferenceFeatures.__dataclass_fields__)
    if production_fields != tuple(OWNER_PREFERENCE_WEIGHTS):
        raise ImplementationError("production feature order differs from weights")
    if set(production_fields) != set(registry_weights):
        raise ImplementationError("production features differ from registry")
    if OWNER_PREFERENCE_SCORE_NAME != "owner_preference_score":
        raise ImplementationError("score name changed")
    constants = {
        "threshold": OWNER_PREFERENCE_THRESHOLD,
        "minimum_quality_margin": OWNER_PREFERENCE_MIN_QUALITY_MARGIN,
        "minimum_detection_confidence": (
            OWNER_PREFERENCE_MIN_DETECTION_CONFIDENCE
        ),
        "minimum_hidden_overlap": OWNER_PREFERENCE_MIN_HIDDEN_OVERLAP,
        "maximum_detection_opportunities": (
            OWNER_PREFERENCE_MAX_DETECTION_OPPORTUNITIES
        ),
    }
    if constants != EXPECTED_CONSTANTS:
        raise ImplementationError("frozen activation constants changed")

    box = np.asarray([0.0, 0.0, 10.0, 10.0])
    shifted = np.asarray([5.0, 0.0, 15.0, 10.0])
    doubled = np.asarray([0.0, 0.0, 20.0, 20.0])
    expected_iou = 50.0 / 150.0
    expected_center = 1.0 - (2.0 * 5.0) / (
        math.sqrt(200.0) + math.sqrt(200.0)
    )
    if not math.isclose(overlap_similarity(box, shifted), expected_iou):
        raise ImplementationError("overlap formula mismatch")
    if not math.isclose(
        normalized_center_similarity(box, shifted),
        expected_center,
    ):
        raise ImplementationError("center formula mismatch")
    if scale_similarity(box, doubled) != 0.0:
        raise ImplementationError("scale formula mismatch")
    same_appearance = appearance_similarity(
        np.asarray([1.0, 0.0]),
        np.asarray([1.0, 0.0]),
    )
    missing_appearance = appearance_similarity(None, np.asarray([1.0, 0.0]))
    if same_appearance != (1.0, 1.0) or missing_appearance != (0.5, 0.0):
        raise ImplementationError("appearance formula or mask mismatch")
    if motion_consistency(box, box, available=True) != (1.0, 1.0):
        raise ImplementationError("motion formula or mask mismatch")
    if motion_consistency(None, box, available=False) != (0.5, 0.0):
        raise ImplementationError("motion missingness mismatch")
    if track_freshness(0) != 1.0 or track_freshness(5) != 0.0:
        raise ImplementationError("freshness formula mismatch")

    low = OwnerPreferenceFeatures(*(0.0 for _ in range(8)))
    high = OwnerPreferenceFeatures(*(1.0 for _ in range(8)))
    if owner_preference_score(low, high) != 0.0:
        raise ImplementationError("lower-bound score formula mismatch")
    if owner_preference_score(high, low) != 1.0:
        raise ImplementationError("upper-bound score formula mismatch")
    if owner_preference_score(high, high) != 0.5:
        raise ImplementationError("neutral score formula mismatch")


def _check_symmetric_integration_and_terminology() -> None:
    owner_text = OWNER_SOURCE.read_text(encoding="utf-8")
    association_text = ASSOCIATION_SOURCE.read_text(encoding="utf-8")
    if FORBIDDEN_NAMES.search(owner_text + "\n" + association_text):
        raise ImplementationError("runtime uses probability-like score terminology")
    lowered_owner = owner_text.lower()
    for token in FORBIDDEN_RUNTIME_METADATA:
        if token in lowered_owner:
            raise ImplementationError(f"runtime depends on forbidden metadata: {token}")

    tree = ast.parse(association_text)
    calls: dict[str, ast.Call] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in {
            "hidden_features",
            "visible_features",
        }:
            continue
        if not isinstance(node.value, ast.Call):
            continue
        function = node.value.func
        if isinstance(function, ast.Name) and (
            function.id == "_h1_r2_features_for_track"
        ):
            calls[target.id] = node.value
    if set(calls) != {"hidden_features", "visible_features"}:
        raise ImplementationError("hidden/visible common feature calls are missing")
    hidden_call = calls["hidden_features"]
    visible_call = calls["visible_features"]
    if len(hidden_call.args) != len(visible_call.args):
        raise ImplementationError("hidden/visible feature signatures differ")
    hidden_tail = [ast.dump(arg) for arg in hidden_call.args[1:]]
    visible_tail = [ast.dump(arg) for arg in visible_call.args[1:]]
    if hidden_tail != visible_tail:
        raise ImplementationError("hidden/visible feature definitions are asymmetric")
    if [ast.dump(item.value) for item in hidden_call.keywords] != [
        ast.dump(item.value) for item in visible_call.keywords
    ]:
        raise ImplementationError("hidden/visible feature keywords are asymmetric")


def _check_profiles_and_causality() -> None:
    historical = _historical_profiles()
    old_fast = historical["REALTIME_FAST_CONFIG"]
    old_realtime = historical["PRESENTATION_PROFILES"]["realtime"]
    if REALTIME_FAST_CONFIG != old_fast:
        raise ImplementationError("realtime_fast changed from authorized main")
    if PRESENTATION_PROFILES["realtime"] != old_realtime:
        raise ImplementationError("realtime presentation profile changed")
    if _payload_sha256(REALTIME_FAST_CONFIG) != EXPECTED_REALTIME_FAST_HASH:
        raise ImplementationError("realtime_fast semantic hash changed")
    if set(REALTIME_FAST_H1_R2_CONFIG) != {
        *REALTIME_FAST_CONFIG,
        "h1_r2_owner_preference",
    }:
        raise ImplementationError("candidate does not inherit realtime_fast exactly")
    for key, value in REALTIME_FAST_CONFIG.items():
        if REALTIME_FAST_H1_R2_CONFIG[key] != value:
            raise ImplementationError(f"candidate changed RF_ACC23 key: {key}")
    if REALTIME_FAST_H1_R2_CONFIG["h1_r2_owner_preference"] is not True:
        raise ImplementationError("candidate does not enable H1-r2")
    h1_profiles = {name for name in EVAL_CONFIGS if "h1_r2" in name}
    h1_presentations = {
        name for name in PRESENTATION_PROFILES if "h1_r2" in name
    }
    if h1_profiles != {AUTHORIZED_PROFILE_NAME}:
        raise ImplementationError("unexpected H1-r2 evaluation profile")
    if h1_presentations != {AUTHORIZED_PROFILE_NAME}:
        raise ImplementationError("unexpected H1-r2 presentation profile")
    if PRESENTATION_PROFILES[AUTHORIZED_PROFILE_NAME]["eval_config"] != (
        AUTHORIZED_PROFILE_NAME
    ):
        raise ImplementationError("candidate is not explicit opt-in")
    cfg = TrackingConfig(mode="realtime", **REALTIME_FAST_H1_R2_CONFIG)
    if cfg.causal_hidden_detection_reservation:
        raise ImplementationError("H1-r1 reservation is enabled")
    if cfg.enable_offline_smoothing or cfg.realtime_motion_pair_stabilizer:
        raise ImplementationError("offline or delayed repair is enabled")
    if cfg.detect_every_n_frames != 2:
        raise ImplementationError("candidate detector cadence changed")
    if resolve_output_timing_contract(cfg) != ("causal_framewise", 0):
        raise ImplementationError("candidate is not causal delay zero")


def _check_authority() -> None:
    authorization = _read_json(AUTHORIZATION)
    amendment = _read_json(PROFILE_AMENDMENT)
    decisions = authorization.get("authorizations", {})
    if decisions.get("implementation_authorized") is not True:
        raise ImplementationError("implementation is not authorized")
    for key in (
        "evaluation_authorized",
        "runtime_evaluation_authorized",
        "promotion_authorized",
    ):
        if decisions.get(key) is not False:
            raise ImplementationError(f"original authority enables {key}")
        if amendment.get(key) is not False:
            raise ImplementationError(f"profile amendment enables {key}")
    if amendment.get("profile_addition_authorized") is not True:
        raise ImplementationError("profile addition is not authorized")
    if amendment.get("authorized_profile_name") != AUTHORIZED_PROFILE_NAME:
        raise ImplementationError("authorized profile name changed")
    gate = _read_json(EVALUATION_GATE)
    if gate.get("evaluation_authorized") is not False:
        raise ImplementationError("evaluation gate is authorized")
    if gate.get("promotion_authorized") is True:
        raise ImplementationError("promotion gate is authorized")
    causality = gate.get("causality", {})
    if causality.get("uses_future_frames") is not False:
        raise ImplementationError("future-frame use is not forbidden")


def main() -> int:
    try:
        _check_features_and_score()
        _check_symmetric_integration_and_terminology()
        _check_profiles_and_causality()
        _check_authority()
    except (ImplementationError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"H1_R2_IMPLEMENTATION_CHECKER=FAIL: {exc}", file=sys.stderr)
        return 1
    print("H1_R2_IMPLEMENTATION_CHECKER=PASS")
    print("FEATURE_COUNT=8")
    print("SCORE_NAME=owner_preference_score")
    print("SCORE_CALIBRATED=NO")
    print("SCORE_IS_PROBABILITY=NO")
    print("THRESHOLD=0.60")
    print(f"PROFILE_NAME={AUTHORIZED_PROFILE_NAME}")
    print("REALTIME_FAST_CONFIG_UNCHANGED=YES")
    print("REALTIME_PRESENTATION_PROFILE_UNCHANGED=YES")
    print("EVALUATION_AUTHORIZED=NO")
    print("RUNTIME_EVALUATION_AUTHORIZED=NO")
    print("PROMOTION_AUTHORIZED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
