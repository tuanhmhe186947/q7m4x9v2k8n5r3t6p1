from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.multimodal_smoke import (
    MultimodalSmokeTrainConfig,
    run_multimodal_smoke_train,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny classification_v2 multimodal smoke training.")
    parser.add_argument("--root", type=Path, default=Path("outputs/classification_v2/train_ready_windows"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/multimodal_smoke_train"),
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--per-class-train", type=int, default=1)
    parser.add_argument("--per-class-eval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    result = run_multimodal_smoke_train(
        MultimodalSmokeTrainConfig(
            root=args.root,
            output_dir=args.output_dir,
            image_size=args.image_size,
            hidden_dim=args.hidden_dim,
            steps=args.steps,
            per_class_train=args.per_class_train,
            per_class_eval=args.per_class_eval,
            lr=args.lr,
            device=args.device,
        )
    )
    print(json.dumps(result.audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
