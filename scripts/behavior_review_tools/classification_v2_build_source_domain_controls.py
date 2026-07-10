from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.source_domain_controls import (
    build_source_domain_control_from_paths,
    write_source_domain_control_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build classification_v2 source/domain control view.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/classification_v2/source_domain_control_v1.json"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = build_source_domain_control_from_paths(config)
    paths = write_source_domain_control_outputs(result, output_dir=Path(config["output_dir"]))
    audit = {**result.audit, "config": str(args.config), **paths}
    audit_path = Path(paths["audit_json"])
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if audit["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
