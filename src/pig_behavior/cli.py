"""Command line entry point for the pig behavior classification pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pig behavior classification pipeline",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "export", "infer", "all"],
        default="all",
        help="Pipeline mode to run.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to an image for inference mode.",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional bounding box for inference in pixel coordinates.",
    )
    parser.add_argument(
        "--tabular",
        type=float,
        nargs=6,
        default=None,
        metavar=("FEEDER", "DRINKER", "TOY", "SPEED", "MIN_DIST", "N_CLOSE"),
        help="Tabular features for hybrid inference.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one epoch on a small subset for quick validation.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Override the processed CSV path.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Override the directory containing training images.",
    )
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Use an image-only model instead of the hybrid model.",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="Use four coarse labels instead of eight fine-grained labels.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs1",
        type=int,
        default=None,
        help="Override phase 1 epoch count.",
    )
    parser.add_argument(
        "--epochs2",
        type=int,
        default=None,
        help="Override phase 2 epoch count.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    """Build a TrainConfig from parsed CLI arguments."""
    from pig_behavior.config import TrainConfig

    cfg = TrainConfig(
        dry_run=args.dry_run,
        use_hybrid=not args.image_only,
        use_coarse_labels=args.coarse,
        batch_size=args.batch_size,
    )
    if args.csv_path is not None:
        cfg.csv_path = args.csv_path
    if args.images_dir is not None:
        cfg.images_dir = args.images_dir
    if args.epochs1 is not None:
        cfg.phase1_epochs = args.epochs1
    if args.epochs2 is not None:
        cfg.phase2_epochs = args.epochs2
    return cfg


def print_run_summary(cfg) -> None:
    """Print a concise summary of the requested run."""
    model_kind = "hybrid" if cfg.use_hybrid else "image-only"
    label_kind = "coarse" if cfg.use_coarse_labels else "fine-grained"

    print()
    print("=" * 72)
    print("Pig behavior classification")
    print(f"Model:   {cfg.backbone} ({model_kind})")
    print(f"Classes: {cfg.num_classes} ({label_kind})")
    print(f"Labels:  {', '.join(cfg.labels)}")
    print(f"CSV:     {cfg.csv_path}")
    print(f"Images:  {cfg.images_dir}")
    print(f"Dry run: {cfg.dry_run}")
    print("=" * 72)
    print()


def main() -> int:
    """Run the selected pipeline mode."""
    args = parse_args()
    cfg = build_config(args)
    print_run_summary(cfg)

    from pig_behavior.data_loader import build_datasets
    from pig_behavior.export import benchmark_tflite, export_onnx, export_tflite
    from pig_behavior.inference import run_inference
    from pig_behavior.train import train as run_training

    datasets = None

    if args.mode in {"train", "all"}:
        datasets = build_datasets(cfg)
        results = run_training(cfg, datasets)
        print(f"Training complete. Test accuracy: {results['test_accuracy']:.4f}")

    if args.mode in {"export", "all"}:
        representative_ds = datasets["train"] if datasets and cfg.quantize else None
        tflite_path = export_tflite(cfg, representative_ds=representative_ds)
        export_onnx(cfg)
        avg_latency = benchmark_tflite(tflite_path, cfg)
        print(f"Export complete. Average latency: {avg_latency:.2f} ms")

    if args.mode == "infer":
        if args.image is None:
            print("ERROR: --image is required in infer mode.", file=sys.stderr)
            return 2
        if cfg.use_hybrid and args.tabular is None:
            print(
                "ERROR: --tabular is required for hybrid inference. "
                "Use --image-only to run without tabular features.",
                file=sys.stderr,
            )
            return 2

        run_inference(
            cfg,
            image_path=args.image,
            bbox=tuple(args.bbox) if args.bbox else None,
            tabular_features=args.tabular,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
