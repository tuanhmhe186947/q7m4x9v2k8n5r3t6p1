from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.train_ready_features import build_train_ready_window_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export leakage-safe X/y/mask/sample_weight tables from reviewed "
            "sequence windows."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/sequence_features_reviewed/"
            "sequence_window_features.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows"),
    )
    parser.add_argument("--label-col", default="behavior_window_label")
    parser.add_argument("--mask-col", default="window_valid_for_main_train")
    parser.add_argument("--sample-weight-col", default="window_sample_weight")
    parser.add_argument(
        "--trainer-contract-json",
        type=Path,
        default=Path("configs/classification_v2/trainer_contract_v1.json"),
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing derived export files in the selected output dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)
    if not args.trainer_contract_json.exists():
        raise FileNotFoundError(args.trainer_contract_json)
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be > 0")

    contract_bytes = args.trainer_contract_json.read_bytes()
    trainer_contract = json.loads(contract_bytes.decode("utf-8"))
    feature_whitelist = trainer_contract.get("tabular_feature_whitelist")
    if not isinstance(feature_whitelist, list) or not feature_whitelist:
        raise ValueError("Trainer contract has no tabular_feature_whitelist")

    x_path = args.output_dir / "X_window_features.csv"
    y_path = args.output_dir / "y_behavior.csv"
    mask_path = args.output_dir / "train_mask.csv"
    weight_path = args.output_dir / "sample_weight.csv"
    audit_path = args.output_dir / "train_ready_audit.json"
    output_paths = [x_path, y_path, mask_path, weight_path, audit_path]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Derived export files already exist; pass --overwrite explicitly: "
            f"{existing}"
        )

    df = pd.read_csv(args.input_csv, low_memory=False)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    tables = build_train_ready_window_tables(
        df,
        label_col=args.label_col,
        mask_col=args.mask_col,
        sample_weight_col=args.sample_weight_col,
        feature_whitelist=feature_whitelist,
    )
    if tables.audit["errors"]:
        raise ValueError(f"Train-ready feature audit failed: {tables.audit['errors']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables.x.to_csv(x_path, index=False)
    tables.y.rename(args.label_col).to_frame().to_csv(y_path, index=False)
    tables.mask.rename(args.mask_col).to_frame().to_csv(mask_path, index=False)
    tables.sample_weight.rename(args.sample_weight_col).to_frame().to_csv(weight_path, index=False)

    audit = {
        "input_csv": str(args.input_csv),
        "trainer_contract": {
            "path": str(args.trainer_contract_json),
            "version": trainer_contract.get("version"),
            "sha256": hashlib.sha256(contract_bytes).hexdigest(),
            "feature_count": len(feature_whitelist),
        },
        "outputs": {
            "X_window_features": str(x_path),
            "y_behavior": str(y_path),
            "train_mask": str(mask_path),
            "sample_weight": str(weight_path),
            "audit_json": str(audit_path),
        },
        "rows": {
            "input": int(len(df)),
            "X": int(len(tables.x)),
            "y": int(len(tables.y)),
            "mask_true": int(tables.mask.sum()),
            "mask_false": int((~tables.mask).sum()),
        },
        "feature_selection": tables.audit,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[OK] wrote {x_path} rows={len(tables.x)} "
        f"cols={len(tables.x.columns)}"
    )
    print(f"[OK] wrote {y_path} rows={len(tables.y)}")
    print(
        f"[OK] wrote {mask_path} true={int(tables.mask.sum())} "
        f"false={int((~tables.mask).sum())}"
    )
    print(f"[OK] wrote {weight_path}")
    print(f"[OK] wrote {audit_path}")


if __name__ == "__main__":
    main()
