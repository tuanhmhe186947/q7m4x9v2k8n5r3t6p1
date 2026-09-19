from pig_behavior.classification_v2.evaluation.metrics_payload_contract import check_paper_metrics_payload


def test_percentile_interval_need_not_contain_point_estimate() -> None:
    """A percentile bootstrap interval may exclude the original-sample estimate."""

    payload = {
        "statistical_unit": "native_temporal_unit",
        "native_temporal_metrics": {
            "rows": 10,
            "accuracy": 0.5,
            "macro_f1_supported": 0.4,
            "macro_recall_supported": 0.6,
            "per_class": {},
            "focus_pair_confusions": {},
        },
        "confidence_intervals": {
            name: {
                "estimate": 0.6,
                "ci_low": 0.1,
                "ci_high": 0.5,
                "method": "unit_bootstrap_percentile",
            }
            for name in ("accuracy", "macro_f1_supported", "macro_recall_supported")
        },
        "sesoi": {
            "primary_metric": "macro_f1_supported",
            "minimum_effect_size": 0.02,
            "comparison_required": True,
        },
    }

    result = check_paper_metrics_payload(payload)

    assert result["valid"] is True
    assert result["errors"] == []
