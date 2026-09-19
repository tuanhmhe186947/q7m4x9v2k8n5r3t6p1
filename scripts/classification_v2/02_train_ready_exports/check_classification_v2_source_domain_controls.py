from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.evaluation.source_domain_controls import (
    audit_source_domain_control_view,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 source/domain control artifacts.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/classification_v2/source_domain_controls/source_domain_selection_manifest.csv"),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("outputs/classification_v2/source_domain_controls/source_domain_control_audit.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/classification_v2/source_domain_control_v1.json"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = pd.read_csv(args.manifest, low_memory=False)
    x_path = Path(config["train_ready_root"]) / config["tabular_x"]
    x_columns = list(pd.read_csv(x_path, nrows=0).columns)
    source_labels = sorted(manifest["source_type"].dropna().astype(str).unique())
    audit = audit_source_domain_control_view(
        manifest,
        x_columns=x_columns,
        forbidden_patterns=config.get("forbidden_x_patterns"),
        source_labels=source_labels,
    )
    frozen = json.loads(args.audit_json.read_text(encoding="utf-8"))
    drift_fields = ["rows", "eligible_rows", "kept_rows", "reason_counts", "source_counts_kept"]
    drift = [field for field in drift_fields if frozen.get(field) != audit.get(field)]
    if drift:
        audit["errors"].append(f"audit_drift={drift}")
        audit["valid"] = False
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if audit["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
