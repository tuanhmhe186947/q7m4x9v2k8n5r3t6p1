from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.posture_proposal import (
    PostureAutoValidationPolicy,
    PostureReviewScopePolicy,
    build_posture_review_scope,
    evaluate_posture_auto_validation,
    wilson_lower_bound,
)


def test_only_audited_high_precision_upright_stratum_is_auto_validated() -> None:
    audited_count = 250
    proposals = pd.DataFrame(
        {
            "native_temporal_unit_key": [
                *(f"upright-{index}" for index in range(audited_count)),
                "sitting-rare",
                "upright-low-confidence",
                "upright-transition",
            ],
            "proposal_stratum": ["social-high"] * (audited_count + 3),
            "posture_proposed": ["upright"] * audited_count
            + ["sitting", "upright", "upright"],
            "posture_confidence": [0.999] * audited_count
            + [0.999, 0.5, 0.999],
            "posture_temporal_consistent": [True] * (audited_count + 3),
            "posture_transition_flag": [False] * (audited_count + 2) + [True],
        }
    )
    audit = pd.DataFrame(
        {
            "native_temporal_unit_key": [
                f"upright-{index}" for index in range(audited_count)
            ],
            "posture_target": ["upright"] * audited_count,
            "posture_valid_mask": [True] * audited_count,
        }
    )
    policy = PostureAutoValidationPolicy(
        confidence_threshold=0.95,
        minimum_audit_rows_per_stratum=200,
    )

    evaluated, report = evaluate_posture_auto_validation(
        proposals,
        audit,
        policy=policy,
    )
    indexed = evaluated.set_index("native_temporal_unit_key")

    assert report["auto_validated_rows"] == audited_count
    assert report["strata"][0]["precision_lower_bound"] >= 0.98
    assert indexed.at["upright-0", "posture_authority"] == "AUTO_VALIDATED"
    assert indexed.at["sitting-rare", "posture_review_reason"] == (
        "NON_UPRIGHT_REQUIRES_REVIEW"
    )
    assert indexed.at["upright-low-confidence", "posture_review_reason"] == (
        "LOW_CONFIDENCE"
    )
    assert indexed.at["upright-transition", "posture_review_reason"] == (
        "POSTURE_TRANSITION"
    )


def test_imperfect_stratum_fails_strict_precision_lower_bound() -> None:
    count = 250
    proposals = _upright_proposals(count)
    targets = ["lying"] * 5 + ["upright"] * (count - 5)
    audit = pd.DataFrame(
        {
            "native_temporal_unit_key": proposals["native_temporal_unit_key"],
            "posture_target": targets,
            "posture_valid_mask": [True] * count,
        }
    )

    evaluated, report = evaluate_posture_auto_validation(
        proposals,
        audit,
        policy=PostureAutoValidationPolicy(
            confidence_threshold=0.95,
            minimum_audit_rows_per_stratum=200,
        ),
    )

    assert not report["strata"][0]["passed"]
    assert report["auto_validated_rows"] == 0
    assert evaluated["posture_review_required"].all()


def test_zero_audit_support_never_auto_validates() -> None:
    proposals = _upright_proposals(10)
    empty_audit = pd.DataFrame(
        columns=[
            "native_temporal_unit_key",
            "posture_target",
            "posture_valid_mask",
        ]
    )

    evaluated, report = evaluate_posture_auto_validation(
        proposals,
        empty_audit,
        policy=PostureAutoValidationPolicy(
            confidence_threshold=0.9,
            minimum_audit_rows_per_stratum=5,
        ),
    )

    assert report["strata"][0]["precision_lower_bound"] is None
    assert report["auto_validated_rows"] == 0
    assert evaluated["posture_authority"].eq("UNRESOLVED").all()


def test_wilson_lower_bound_is_strict_for_small_perfect_samples() -> None:
    assert wilson_lower_bound(10, 10, z=1.6448536269514722) < 0.98
    assert wilson_lower_bound(250, 250, z=1.6448536269514722) >= 0.98


def test_posture_review_scope_is_deterministic_and_keeps_mandatory_rows() -> None:
    proposals = _review_scope_proposals()
    policy = PostureReviewScopePolicy(
        confidence_threshold=0.9,
        upright_control_rows_per_stratum=2,
        seed=20260801,
    )

    first, first_audit = build_posture_review_scope(proposals, policy=policy)
    second, second_audit = build_posture_review_scope(proposals, policy=policy)

    assert first.equals(second)
    assert first_audit == second_audit
    selected_keys = set(first["native_temporal_unit_key"])
    assert {
        "a-sitting",
        "a-low-confidence",
        "a-inconsistent",
        "a-transition",
    }.issubset(selected_keys)
    assert first_audit["mandatory_rows"] == 4
    assert first_audit["upright_control_rows"] == 4


def test_posture_review_scope_samples_declared_upright_controls_per_stratum() -> None:
    scope, audit = build_posture_review_scope(
        _review_scope_proposals(),
        policy=PostureReviewScopePolicy(
            confidence_threshold=0.9,
            upright_control_rows_per_stratum=2,
            seed=17,
        ),
    )

    controls = scope.loc[
        scope["posture_review_scope_reason"].eq("AUTO_UPRIGHT_RANDOM_CONTROL")
    ]
    assert controls.groupby("proposal_stratum").size().to_dict() == {
        "stratum-a": 2,
        "stratum-b": 2,
    }
    probabilities = {
        row["proposal_stratum"]: row["sampling_probability"]
        for row in audit["control_strata"]
    }
    assert probabilities == {"stratum-a": 0.4, "stratum-b": 2 / 3}
    assert set(
        controls.groupby("proposal_stratum")[
            "posture_control_sampling_probability"
        ].first()
    ) == {0.4, 2 / 3}


def test_posture_review_scope_is_independent_of_input_row_order() -> None:
    proposals = _review_scope_proposals()
    shuffled = proposals.sample(frac=1.0, random_state=99)
    policy = PostureReviewScopePolicy(
        confidence_threshold=0.9,
        upright_control_rows_per_stratum=2,
        seed=11,
    )

    ordered, _ = build_posture_review_scope(proposals, policy=policy)
    reordered, _ = build_posture_review_scope(shuffled, policy=policy)

    assert set(ordered["native_temporal_unit_key"]) == set(
        reordered["native_temporal_unit_key"]
    )


def test_posture_review_scope_cli_exports_hashed_synthetic_scope(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    proposals_path = tmp_path / "proposals.csv"
    policy_path = tmp_path / "policy.json"
    scope_path = tmp_path / "posture_review_scope.csv"
    audit_path = tmp_path / "posture_review_scope_audit.json"
    _review_scope_proposals().to_csv(proposals_path, index=False)
    policy_path.write_text(
        json.dumps(
            {
                "confidence_threshold": 0.9,
                "upright_control_rows_per_stratum": 2,
                "review_scope_seed": 20260801,
                "real_execution_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    script = root / (
        "scripts/classification_v2/02_train_ready_exports/"
        "build_posture_review_scope.py"
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--proposals-csv",
            str(proposals_path),
            "--policy-json",
            str(policy_path),
            "--behavior-label-authority",
            "SYNTHETIC_TEST",
            "--output-review-scope-csv",
            str(scope_path),
            "--output-audit-json",
            str(audit_path),
        ],
        cwd=root,
        check=True,
    )

    scope = pd.read_csv(scope_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert len(scope) == 8
    assert audit["behavior_label_authority"] == "SYNTHETIC_TEST"
    assert audit["output_review_scope"]["sha256"]


def test_real_posture_review_scope_requires_explicit_policy_authorization(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    proposals_path = tmp_path / "proposals.csv"
    policy_path = tmp_path / "policy.json"
    _review_scope_proposals().to_csv(proposals_path, index=False)
    policy_path.write_text(
        json.dumps(
            {
                "confidence_threshold": 0.9,
                "upright_control_rows_per_stratum": 2,
                "review_scope_seed": 20260801,
                "real_execution_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    script = root / (
        "scripts/classification_v2/02_train_ready_exports/"
        "build_posture_review_scope.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--proposals-csv",
            str(proposals_path),
            "--policy-json",
            str(policy_path),
            "--behavior-label-authority",
            "FROZEN_HUMAN_REVIEWED",
            "--output-review-scope-csv",
            str(tmp_path / "scope.csv"),
            "--output-audit-json",
            str(tmp_path / "audit.json"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "real_execution_authorized=true" in result.stderr


def _upright_proposals(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "native_temporal_unit_key": [f"unit-{index}" for index in range(count)],
            "proposal_stratum": ["social-high"] * count,
            "posture_proposed": ["upright"] * count,
            "posture_confidence": [0.99] * count,
            "posture_temporal_consistent": [True] * count,
            "posture_transition_flag": [False] * count,
        }
    )


def _review_scope_proposals() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stratum, count in (("stratum-a", 5), ("stratum-b", 3)):
        for index in range(count):
            rows.append(
                {
                    "native_temporal_unit_key": f"{stratum}-upright-{index}",
                    "proposal_stratum": stratum,
                    "posture_proposed": "upright",
                    "posture_confidence": 0.99,
                    "posture_temporal_consistent": True,
                    "posture_transition_flag": False,
                }
            )
    rows.extend(
        [
            {
                "native_temporal_unit_key": "a-sitting",
                "proposal_stratum": "stratum-a",
                "posture_proposed": "sitting",
                "posture_confidence": 0.99,
                "posture_temporal_consistent": True,
                "posture_transition_flag": False,
            },
            {
                "native_temporal_unit_key": "a-low-confidence",
                "proposal_stratum": "stratum-a",
                "posture_proposed": "upright",
                "posture_confidence": 0.4,
                "posture_temporal_consistent": True,
                "posture_transition_flag": False,
            },
            {
                "native_temporal_unit_key": "a-inconsistent",
                "proposal_stratum": "stratum-a",
                "posture_proposed": "upright",
                "posture_confidence": 0.99,
                "posture_temporal_consistent": False,
                "posture_transition_flag": False,
            },
            {
                "native_temporal_unit_key": "a-transition",
                "proposal_stratum": "stratum-a",
                "posture_proposed": "upright",
                "posture_confidence": 0.99,
                "posture_temporal_consistent": True,
                "posture_transition_flag": True,
            },
        ]
    )
    return pd.DataFrame(rows)
