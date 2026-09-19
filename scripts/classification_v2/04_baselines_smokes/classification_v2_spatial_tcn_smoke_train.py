from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.spatial_tcn_smoke import (
    SpatialTCNSmokeTrainConfig,
    run_spatial_tcn_smoke_train,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run split-safe SpatialTCN smoke training for classification_v2.")
    parser.add_argument("--root", type=Path, default=Path("outputs/classification_v2/train_ready_windows"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/spatial_tcn_smoke_train"),
    )
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--per-class-train", type=int, default=8)
    parser.add_argument("--per-class-eval", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_spatial_tcn_smoke_train(
        SpatialTCNSmokeTrainConfig(
            root=args.root,
            output_dir=args.output_dir,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            steps=args.steps,
            per_class_train=args.per_class_train,
            per_class_eval=args.per_class_eval,
            seed=args.seed,
            device=args.device,
        )
    )
    print(json.dumps(result.audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
