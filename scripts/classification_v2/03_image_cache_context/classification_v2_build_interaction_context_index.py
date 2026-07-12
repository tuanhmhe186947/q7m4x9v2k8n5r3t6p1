from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.datasets.interaction_context_index import (
    InteractionContextIndexConfig,
    build_interaction_context_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build classification_v2 interaction context audit index.")
    parser.add_argument("--root", type=Path, default=Path("outputs/classification_v2/train_ready_windows"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classification_v2/train_ready_windows"))
    args = parser.parse_args()

    result = build_interaction_context_index(InteractionContextIndexConfig(root=args.root, output_dir=args.output_dir))
    print(
        json.dumps(
            {
                "manifest_path": str(result.manifest_path),
                "audit_path": str(result.audit_path),
                "audit": result.audit,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
