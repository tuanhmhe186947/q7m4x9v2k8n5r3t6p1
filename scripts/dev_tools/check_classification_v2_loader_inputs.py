from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.loader_input_audit import audit_loader_input_contract


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check classification_v2 loader/sampler input leakage contract."
    )
    parser.add_argument(
        "--trainer-contract-json",
        type=Path,
        default=Path("configs/classification_v2/trainer_contract_v1.json"),
    )
    parser.add_argument(
        "--model-input-contract-json",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/model_input_contract.json"),
    )
    parser.add_argument(
        "--source-domain-audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/source_domain_controls/"
            "source_domain_control_audit.json"
        ),
    )
    parser.add_argument("--source-domain-manifest-csv", type=Path, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/loader_input_audit.json"),
    )
    args = parser.parse_args()

    result = audit_loader_input_contract(
        trainer_contract_json=args.trainer_contract_json,
        model_input_contract_json=args.model_input_contract_json,
        source_domain_audit_json=args.source_domain_audit_json,
        source_domain_manifest_csv=args.source_domain_manifest_csv,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
