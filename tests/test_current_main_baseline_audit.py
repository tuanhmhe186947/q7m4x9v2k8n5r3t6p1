from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "tracking" / "run_current_main_baseline_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("baseline_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_taxonomy_is_complete_and_contract_bound() -> None:
    module = _load_module()

    assert len(module.MECHANISMS) == 13
    assert "TRACK_TERMINATION_OR_REVIVAL_POLICY" in module.MECHANISMS
    assert "SHORT_TRANSIENT_IDENTITY_ERROR" in module.MECHANISMS


def test_unresolved_gt_fails_closed_for_mechanistic_authority() -> None:
    module = _load_module()
    event = {
        "gt_authority": "UNRESOLVED_EXCLUDE_FROM_MECHANISM_RANKING",
    }

    primary, secondary = module.classify_event(event)

    assert primary == "GT_OR_EVALUATION_AMBIGUITY"
    assert secondary == ""


def test_short_authoritative_event_is_transient() -> None:
    module = _load_module()
    event = {
        "gt_authority": "AUTHORITATIVE_FOR_MECHANISTIC_CONCLUSIONS",
        "start_frame": 100,
        "duration_frames": 2,
    }

    primary, _secondary = module.classify_event(event)

    assert primary == "SHORT_TRANSIENT_IDENTITY_ERROR"


def test_detection_dominance_selects_detector_first() -> None:
    module = _load_module()
    summary = [
        {
            "primary_mechanism": mechanism,
            "authoritative_wrong_id_frames": (
                60 if mechanism == "DETECTION_MISS_OR_DROPOUT" else 0
            ),
            "authoritative_event_count": (
                2 if mechanism == "DETECTION_MISS_OR_DROPOUT" else 0
            ),
            "permanent_swaps": 0,
            "terminal_swaps": 0,
            "causal_evidence_available_events": (
                2 if mechanism == "DETECTION_MISS_OR_DROPOUT" else 0
            ),
        }
        for mechanism in module.MECHANISMS
    ]
    ranking = module.rank_mechanisms(summary)

    decision = module.tracking_decision(ranking, summary)

    assert decision["decision"] == "IMPROVE_DETECTOR_FIRST"
    assert decision["new_implementation_authorized"] is False
    assert decision["validation_authorized"] is False
