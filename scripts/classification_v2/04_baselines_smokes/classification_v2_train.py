"""Run one strict classification_v2 training configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.config import load_training_config
from pig_behavior.classification_v2.training.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an audited classification_v2 fold.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/classification_v2/multimodal_context_multitask.json"),
    )
    args = parser.parse_args()
    audit = run_training(load_training_config(args.config))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
