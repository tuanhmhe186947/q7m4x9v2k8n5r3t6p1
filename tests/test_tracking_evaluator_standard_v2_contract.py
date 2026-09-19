from __future__ import annotations

import pytest

from pig_behavior.evaluation.tracking.contracts import (
    EVALUATOR_CONTRACT_ID,
    HOTA_ALPHAS,
    LEGACY_EVALUATOR_CONTRACT_ID,
    MetricContractError,
    build_metric_metadata,
    validate_report_contract,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        **build_metric_metadata(
            include_hidden=True,
            evaluator_code_sha="a" * 40,
        ),
        "video_stem": "synthetic",
        "hota": 1.0,
    }
    row.update(overrides)
    return row


def test_standard_v2_contract_accepts_homogeneous_rows() -> None:
    first = _row(video_stem="one")
    second = _row(video_stem="two")
    assert validate_report_contract([first, second]) == EVALUATOR_CONTRACT_ID


def test_unversioned_report_is_historical_only() -> None:
    row = {"video_stem": "old", "hota": 0.9}
    with pytest.raises(MetricContractError, match="historical-only"):
        validate_report_contract([row])
    assert (
        validate_report_contract([row], allow_historical_legacy=True)
        == LEGACY_EVALUATOR_CONTRACT_ID
    )


def test_historical_legacy_fields_remain_readable_without_relabeling() -> None:
    row = {
        "video_stem": "old",
        "remapped_hota": 0.9,
        "permanent_swap": 1,
        "terminal_swap": 1,
    }
    frozen = dict(row)

    assert (
        validate_report_contract([row], allow_historical_legacy=True)
        == LEGACY_EVALUATOR_CONTRACT_ID
    )
    assert row == frozen


def test_legacy_contract_cannot_generate_a_new_report() -> None:
    with pytest.raises(MetricContractError, match="read-only"):
        validate_report_contract(
            [{"evaluator_contract_id": LEGACY_EVALUATOR_CONTRACT_ID}]
        )


def test_mixed_version_report_fails_closed() -> None:
    with pytest.raises(MetricContractError, match="Mixed"):
        validate_report_contract([_row(), {"video_stem": "old"}])


@pytest.mark.parametrize(
    "field",
    ["permanent_swap", "terminal_swap", "remapped_hota", "remapped_assa"],
)
def test_legacy_metric_fields_are_forbidden_in_v2(field: str) -> None:
    with pytest.raises(MetricContractError, match="forbidden"):
        validate_report_contract([_row(**{field: 0})])


def test_v2_metadata_must_be_identical_across_rows() -> None:
    with pytest.raises(MetricContractError, match="include_hidden"):
        validate_report_contract([_row(), _row(include_hidden=False)])


def test_v2_alpha_set_is_exact() -> None:
    assert len(HOTA_ALPHAS) == 19
    assert HOTA_ALPHAS[0] == 0.05
    assert HOTA_ALPHAS[-1] == 0.95
    with pytest.raises(MetricContractError, match="alpha"):
        validate_report_contract([_row(hota_threshold_set=[0.5])])


def test_json_round_tripped_alpha_list_is_accepted() -> None:
    row = _row(hota_threshold_set=list(HOTA_ALPHAS))
    assert validate_report_contract([row]) == EVALUATOR_CONTRACT_ID


def test_metric_config_hash_binds_hidden_policy() -> None:
    hidden = build_metric_metadata(
        include_hidden=True,
        evaluator_code_sha="a" * 40,
    )
    visible = build_metric_metadata(
        include_hidden=False,
        evaluator_code_sha="a" * 40,
    )
    assert hidden["metric_config_sha256"] != visible["metric_config_sha256"]
