from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.paper_grade_protocol import write_paper_grade_protocol_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 paper-grade Q2 protocol.")
    parser.add_argument(
        "--protocol-json",
        type=Path,
        default=Path("configs/classification_v2/paper_grade_protocol_v1.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/paper_grade_protocol/paper_grade_protocol_audit.json"),
    )
    args = parser.parse_args()
    audit = write_paper_grade_protocol_audit(args.protocol_json, args.output_json)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if audit["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
