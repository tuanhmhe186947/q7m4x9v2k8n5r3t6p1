"""Run lineage-bound native-unit source and availability diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.contracts.window_alignment import (
    require_ordered_window_ids,
)
from pig_behavior.classification_v2.evaluation.domain_controls import (
    LABEL_INDEPENDENT_AVAILABILITY_COLUMNS,
    audit_domain_feature_shift,
    grouped_availability_behavior_probe,
    grouped_source_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exact-whitelist classification_v2 domain controls at "
            "native temporal-unit grain."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows"),
    )
    parser.add_argument(
        "--native-mapping",
        type=Path,
        default=Path(
            "outputs/classification_v2/sequence_features_reviewed/"
            "sequence_window_manifest.csv"
        ),
    )
    parser.add_argument(
        "--grouped-roles",
        type=Path,
        default=Path(
            "outputs/classification_v2/q2_grouped_folds/"
            "q2_outer_inner_roles.csv"
        ),
    )
    parser.add_argument(
        "--trainer-contract-json",
        type=Path,
        default=Path("configs/classification_v2/trainer_contract_v1.json"),
    )
    parser.add_argument("--train-ready-audit-json", type=Path, default=None)
    parser.add_argument("--image-window-manifest", type=Path, default=None)
    parser.add_argument("--interaction-window-manifest", type=Path, default=None)
    parser.add_argument(
        "--availability-columns",
        nargs="+",
        default=list(LABEL_INDEPENDENT_AVAILABILITY_COLUMNS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/domain_controls"),
    )
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _resolved_paths(args)
    output_paths = _output_paths(args.output_dir)
    require_output_paths_available(output_paths.values(), overwrite=args.overwrite)

    trainer_contract_bytes = args.trainer_contract_json.read_bytes()
    trainer_contract = json.loads(trainer_contract_bytes.decode("utf-8"))
    whitelist = trainer_contract.get("tabular_feature_whitelist")
    if not isinstance(whitelist, list) or not whitelist:
        raise ValueError("trainer contract has no tabular_feature_whitelist")
    forbidden_patterns = trainer_contract.get("forbidden_x_patterns")
    if not isinstance(forbidden_patterns, list) or not forbidden_patterns:
        raise ValueError("trainer contract has no forbidden_x_patterns")
    train_ready_audit = _read_json(paths["train_ready_audit"])
    expected_window_hash = _validate_lineage_contract(
        trainer_contract_bytes,
        whitelist,
        train_ready_audit,
    )

    feature_path = args.root / "X_window_features.csv"
    observed_feature_columns = list(pd.read_csv(feature_path, nrows=0).columns)
    if observed_feature_columns != whitelist:
        raise ValueError(
            "X_window_features.csv does not match the ordered trainer whitelist: "
            f"observed={len(observed_feature_columns)}, expected={len(whitelist)}"
        )
    features = pd.read_csv(feature_path, low_memory=False)
    metadata = pd.read_csv(args.root / "split_manifest.csv", low_memory=False)
    native_mapping = pd.read_csv(paths["native_mapping"], low_memory=False)
    roles = pd.read_csv(args.grouped_roles, low_memory=False)
    availability = _load_availability_table(
        metadata,
        paths["image_window_manifest"],
        paths["interaction_window_manifest"],
        args.availability_columns,
    )

    source_predictions, source_audit = grouped_source_probe(
        features,
        metadata,
        native_mapping,
        roles,
        feature_whitelist=whitelist,
        expected_ordered_window_id_sha256=expected_window_hash,
        forbidden_patterns=forbidden_patterns,
        max_iter=args.max_iter,
    )
    availability_predictions, availability_audit = (
        grouped_availability_behavior_probe(
            availability,
            metadata,
            native_mapping,
            roles,
            availability_feature_whitelist=args.availability_columns,
            expected_ordered_window_id_sha256=expected_window_hash,
            max_iter=args.max_iter,
        )
    )
    shift_audit = audit_domain_feature_shift(
        features,
        metadata,
        feature_whitelist=whitelist,
        expected_ordered_window_id_sha256=expected_window_hash,
        forbidden_patterns=forbidden_patterns,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_predictions.to_csv(output_paths["source_predictions"], index=False)
    availability_predictions.to_csv(
        output_paths["availability_predictions"],
        index=False,
    )
    source_payload = {
        "prediction_csv": str(output_paths["source_predictions"]),
        **source_audit,
    }
    availability_payload = {
        "prediction_csv": str(output_paths["availability_predictions"]),
        **availability_audit,
    }
    _write_json(output_paths["source_audit"], source_payload)
    _write_json(output_paths["availability_audit"], availability_payload)
    _write_json(output_paths["feature_shift_audit"], shift_audit)
    print(
        json.dumps(
            {
                "source_probe": source_payload,
                "availability_probe": availability_payload,
                "feature_shift_audit": str(output_paths["feature_shift_audit"]),
            },
            indent=2,
        )
    )


def _resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "train_ready_audit": args.train_ready_audit_json
        or args.root / "train_ready_audit.json",
        "native_mapping": args.native_mapping,
        "image_window_manifest": args.image_window_manifest
        or args.root / "image_window_context_manifest.csv",
        "interaction_window_manifest": args.interaction_window_manifest
        or args.root / "interaction_window_context_manifest.csv",
    }


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "source_predictions": output_dir / "grouped_source_probe_predictions.csv",
        "source_audit": output_dir / "grouped_source_probe_audit.json",
        "availability_predictions": (
            output_dir / "grouped_availability_behavior_probe_predictions.csv"
        ),
        "availability_audit": (
            output_dir / "grouped_availability_behavior_probe_audit.json"
        ),
        "feature_shift_audit": output_dir / "domain_feature_shift_audit.json",
    }


def _validate_lineage_contract(
    trainer_contract_bytes: bytes,
    whitelist: list[str],
    train_ready_audit: dict[str, Any],
) -> str:
    trainer = train_ready_audit.get("trainer_contract", {})
    expected_contract_hash = hashlib.sha256(trainer_contract_bytes).hexdigest()
    if trainer.get("sha256") != expected_contract_hash:
        raise ValueError("train-ready audit trainer-contract SHA256 mismatch")
    if trainer.get("feature_count") != len(whitelist):
        raise ValueError("train-ready audit trainer feature count mismatch")
    selected = train_ready_audit.get("feature_selection", {}).get(
        "feature_columns",
        [],
    )
    if selected != whitelist:
        raise ValueError("train-ready audit feature order differs from trainer whitelist")
    window_hash = train_ready_audit.get("window_alignment", {}).get(
        "reference_ordered_window_id_sha256"
    )
    if not isinstance(window_hash, str) or len(window_hash) != 64:
        raise ValueError("train-ready audit lacks ordered-window SHA256")
    return window_hash


def _load_availability_table(
    metadata: pd.DataFrame,
    image_manifest_path: Path,
    interaction_manifest_path: Path,
    columns: list[str],
) -> pd.DataFrame:
    image = pd.read_csv(image_manifest_path, low_memory=False)
    interaction = pd.read_csv(interaction_manifest_path, low_memory=False)
    require_ordered_window_ids(
        "split_manifest",
        metadata["window_id"],
        {
            "image_window_manifest": image["window_id"],
            "interaction_window_manifest": interaction["window_id"],
        },
    )
    sources = {
        "window_image_context_complete": image,
        "scene_context_ready": interaction,
        "scene_partner_context_ready": interaction,
    }
    availability = metadata[["window_id"]].reset_index(drop=True).copy()
    for column in columns:
        if column not in sources or column not in sources[column]:
            raise ValueError(f"availability column has no declared source: {column}")
        availability[column] = sources[column][column].reset_index(drop=True)
    return availability


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
