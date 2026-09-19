from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.metrics_payload_contract import check_paper_metrics_payload
from pig_behavior.classification_v2.evaluation.native_temporal_metrics_gate import (
    check_native_temporal_metrics_gate,
    default_evaluation_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a classification_v2 native temporal metrics payload.")
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--result-kind", default="model_evaluation")
    parser.add_argument("--split-policy", default="recording_group_oof")
    args = parser.parse_args()

    payload = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    contract = default_evaluation_contract()
    contract.update({"result_kind": args.result_kind, "split_policy": args.split_policy})

    result = check_native_temporal_metrics_gate(
        evaluation_contract=contract,
        metrics_payload=payload,
        paper_facing=True,
        experiment_stage="paper_facing_candidate",
    )
    audit = payload.get("native_temporal_prediction_audit", {})
    if audit.get("errors"):
        result["errors"].append(f"native_temporal_prediction_audit_errors={audit.get('errors')}")
    if audit.get("valid") is False:
        result["errors"].append("native_temporal_prediction_audit_invalid")
    metrics_contract = check_paper_metrics_payload(payload)
    result["paper_metrics_payload_contract"] = metrics_contract
    result["errors"].extend(f"paper_metrics_payload_contract:{error}" for error in metrics_contract["errors"])
    result["warnings"].extend(f"paper_metrics_payload_contract:{warning}" for warning in metrics_contract["warnings"])
    result["valid"] = not result["errors"]
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
