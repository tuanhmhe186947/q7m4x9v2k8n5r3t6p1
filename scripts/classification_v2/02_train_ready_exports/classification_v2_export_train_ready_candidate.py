"""Export one explicit, leakage-safe train-ready candidate from reviewed windows."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.contracts.semantic_lineage import file_sha256
from pig_behavior.classification_v2.train_ready_features import (
    build_train_ready_window_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trainer-contract-json", required=True, type=Path)
    return parser.parse_args()


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    contract = json.loads(args.trainer_contract_json.read_text(encoding="utf-8"))
    whitelist = [str(value) for value in contract["tabular_feature_whitelist"]]
    outputs = {
        "x": args.output_dir / "X_window_features.csv",
        "y": args.output_dir / "y_behavior.csv",
        "mask": args.output_dir / "train_mask.csv",
        "weight": args.output_dir / "sample_weight.csv",
        "audit": args.output_dir / "train_ready_audit.json",
    }
    require_output_paths_available(outputs.values(), overwrite=False)
    windows = pd.read_csv(args.input_csv, low_memory=False)
    tables = build_train_ready_window_tables(
        windows,
        feature_whitelist=whitelist,
    )
    audit = dict(tables.audit)
    audit.update(
        {
            "input_csv": str(args.input_csv),
            "input_sha256": file_sha256(args.input_csv),
            "trainer_contract_json": str(args.trainer_contract_json),
            "trainer_contract_sha256": file_sha256(args.trainer_contract_json),
            "row_count": len(windows),
            "output_written": True,
        }
    )
    if audit.get("errors"):
        print(json.dumps(audit, indent=2, ensure_ascii=True))
        return 2
    _write_csv_atomic(tables.x, outputs["x"])
    _write_csv_atomic(
        tables.y.rename("behavior_window_label").to_frame(),
        outputs["y"],
    )
    _write_csv_atomic(
        tables.mask.rename("window_valid_for_main_train").to_frame(),
        outputs["mask"],
    )
    _write_csv_atomic(
        tables.sample_weight.rename("window_sample_weight").to_frame(),
        outputs["weight"],
    )
    _write_json_atomic(audit, outputs["audit"])
    print(json.dumps({"status": "PASS", **audit}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
