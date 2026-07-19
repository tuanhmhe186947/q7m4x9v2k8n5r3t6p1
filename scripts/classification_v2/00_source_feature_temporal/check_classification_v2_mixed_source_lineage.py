"""Check that a mixed merge used the locked legacy and XML source set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.contracts.merged_source_lineage import (
    audit_mixed_source_lineage,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage-json", type=Path, required=True)
    parser.add_argument("--legacy-export", type=Path, required=True)
    parser.add_argument("--classification-dir", type=Path, required=True)
    parser.add_argument("--expected-xml-count", type=int, default=12)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    audit = audit_mixed_source_lineage(
        args.lineage_json,
        legacy_export=args.legacy_export,
        classification_dir=args.classification_dir,
        expected_xml_count=args.expected_xml_count,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit("FAIL: mixed source lineage")
    print("PASS: mixed source lineage")


if __name__ == "__main__":
    main()
