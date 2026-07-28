from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_ROOT = (
    REPO_ROOT / "docs" / "tracking" / "three_mode_historical_reconstruction"
)


def _load_json(name: str) -> dict[str, object]:
    return json.loads((AUTHORITY_ROOT / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_consolidated_authority_binds_independent_records() -> None:
    authority = _load_json(
        "THREE_MODE_HISTORICAL_RECONSTRUCTION_AUTHORITY_20260729.json"
    )

    assert authority["starting_main_sha"] == (
        "94fac247e8fce120f08334890f57e73eda71eb04"
    )
    assert authority["decision"] == (
        "PASS_WITH_EXPLICIT_NOT_RECOVERABLE_LINEAGES"
    )
    assert authority["ready_for_superseding_method_freeze_decision"] is True
    assert authority["ready_for_unseen_data_authority_freeze"] is False
    assert authority["ready_for_unseen_evaluation"] is False
    assert authority["ready_to_promote"] is False

    for reference in authority["independent_authorities"].values():
        path = REPO_ROOT / reference["path"]
        assert path.is_file()
        assert _sha256(path) == reference["sha256"]

    for key in ("artifact_search", "causal_matrix", "claim_matrix"):
        reference = (
            authority[key]["inventory"]
            if key == "artifact_search"
            else authority[key]
        )
        path = REPO_ROOT / reference["path"]
        assert path.is_file()
        assert _sha256(path) == reference["sha256"]


def test_prediction_parity_fingerprints_are_frozen() -> None:
    expected = {
        "B0": {
            "canonical_content_equal": False,
            "identity_differences": 0,
            "hidden_state_differences": 1978,
            "bbox_exact_differences": 187096,
        },
        "R0": {
            "canonical_content_equal": True,
            "identity_differences": 0,
            "hidden_state_differences": 0,
            "bbox_exact_differences": 0,
        },
        "B1": {
            "canonical_content_equal": False,
            "identity_differences": 5920,
            "hidden_state_differences": 1959,
            "bbox_exact_differences": 187162,
        },
    }

    for mode, fingerprint in expected.items():
        report = _load_json(
            f"{mode}_HISTORICAL_VS_CURRENT_PREDICTION_PARITY_20260729.json"
        )
        aggregate = report["aggregate"]
        assert aggregate["videos_compared"] == 13
        assert aggregate["first_row_count"] == 187200
        assert aggregate["second_row_count"] == 187200
        assert aggregate["row_additions"] == 0
        assert aggregate["row_removals"] == 0
        for key, value in fingerprint.items():
            assert aggregate[key] == value


def test_claim_matrix_and_paths_are_closeout_safe() -> None:
    csv_paths = sorted(AUTHORITY_ROOT.glob("*.csv"))
    json_paths = sorted(AUTHORITY_ROOT.glob("*.json"))
    for path in [*csv_paths, *json_paths]:
        assert ".codex_tmp" not in path.read_text(encoding="utf-8")

    claim_path = AUTHORITY_ROOT / "THREE_MODE_CLAIM_CORRECTION_20260729.csv"
    with claim_path.open(encoding="utf-8", newline="") as handle:
        claims = {row["claim_id"]: row for row in csv.DictReader(handle)}

    assert claims["B0_CURRENT_BETTER_THAN_HISTORICAL_RAW"]["status"] == (
        "AUTHORIZED_WITH_LIMITATION"
    )
    assert claims["R0_CURRENT_BETTER_THAN_HISTORICAL_FAST"]["status"] == (
        "CONTRADICTED"
    )
    assert claims["B1_CURRENT_REPRESENTS_HISTORICAL_H5B_H4"]["status"] == (
        "CONTRADICTED"
    )
    assert claims["R0_EVALUATOR_ALONE_EXPLAINS_CHANGE"]["status"] == (
        "AUTHORIZED"
    )
