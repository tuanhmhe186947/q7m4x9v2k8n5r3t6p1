from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.datasets.tf_sequence_dataset import load_train_ready_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 train-ready dataset loader.")
    parser.add_argument("--root", type=Path, default=Path("outputs/classification_v2/train_ready_windows"))
    parser.add_argument("--no-spatial", action="store_true")
    args = parser.parse_args()

    ds = load_train_ready_dataset(args.root, load_spatial=not args.no_spatial)
    result = {
        **ds.audit,
        "split_indices": {
            split: int(len(ds.split_indices(split))) for split in ["train", "val", "test"]
        },
        "class_counts_train": ds.class_counts("train"),
        "class_counts_val": ds.class_counts("val"),
        "class_counts_test": ds.class_counts("test"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
