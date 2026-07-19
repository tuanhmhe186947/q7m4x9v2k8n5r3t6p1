"""Build fresh C6 union/full-frame inputs from one prepared rebuild packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.datasets.visual_interaction_context import (
    VisualInteractionCacheConfig,
    build_visual_interaction_cache,
)
from pig_behavior.classification_v2.training.legacy_c6_modality_inputs import (
    IMAGE_SIZE,
    build_legacy_c6_full_frame_pixel_cache,
    prepare_legacy_c6_modality_context,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmonized-frames", type=Path, required=True)
    parser.add_argument("--selected-native-units", type=Path, required=True)
    parser.add_argument("--prepared-source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--union-padding-ratio", type=float, default=0.15)
    parser.add_argument("--max-open-videos", type=int, default=4)
    parser.add_argument(
        "--selected-modality",
        action="append",
        choices=("union_context", "full_frame_context"),
        help="Context cache to build; repeat for more than one modality.",
    )
    args = parser.parse_args()
    audit = build_legacy_c6_modality_inputs(
        harmonized_frames_path=args.harmonized_frames,
        selected_units_path=args.selected_native_units,
        prepared_source_manifest=args.prepared_source_manifest,
        output_dir=args.output_dir,
        union_padding_ratio=args.union_padding_ratio,
        max_open_videos=args.max_open_videos,
        selected_modalities=(
            tuple(args.selected_modality)
            if args.selected_modality
            else None
        ),
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


def build_legacy_c6_modality_inputs(
    *,
    harmonized_frames_path: Path,
    selected_units_path: Path,
    prepared_source_manifest: Path,
    output_dir: Path,
    union_padding_ratio: float,
    max_open_videos: int,
    selected_modalities: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Write immutable context manifests and selected-slot pixel caches."""

    if not 0.0 <= union_padding_ratio <= 1.0:
        raise ValueError("union padding ratio must be in [0, 1]")
    if max_open_videos <= 0:
        raise ValueError("max_open_videos must be positive")
    if selected_modalities is None:
        selected_modalities = ("union_context", "full_frame_context")
    if not selected_modalities or len(set(selected_modalities)) != len(
        selected_modalities
    ):
        raise ValueError("C6 selected modalities must be unique and nonempty")
    unsupported = set(selected_modalities) - {
        "union_context",
        "full_frame_context",
    }
    if unsupported:
        raise ValueError(f"unsupported C6 context modalities={sorted(unsupported)}")
    inputs = {
        "harmonized_frames": harmonized_frames_path.resolve(),
        "selected_native_units": selected_units_path.resolve(),
        "prepared_source_manifest": prepared_source_manifest.resolve(),
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"legacy C6 modality input missing {name}: {path}")
    prepared = json.loads(
        inputs["prepared_source_manifest"].read_text(encoding="utf-8")
    )
    if prepared.get("valid") is not True:
        raise ValueError("legacy C6 prepared source is invalid")
    if prepared.get("lineage_scope") != "legacy-only-unreviewed-development":
        raise ValueError("legacy C6 prepared source lineage drift")
    if prepared.get("human_review_complete") is not False:
        raise ValueError("legacy C6 prepared source review claim drift")
    selected_spec = prepared.get("artifacts", {}).get("selected_native_units")
    if not isinstance(selected_spec, dict):
        raise ValueError("legacy C6 prepared selected-unit artifact is missing")
    if file_sha256(inputs["selected_native_units"]) != selected_spec.get(
        "sha256"
    ):
        raise ValueError("legacy C6 selected-unit hash differs from packet")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    frames = pd.read_csv(inputs["harmonized_frames"], low_memory=False)
    units = pd.read_csv(inputs["selected_native_units"], low_memory=False)
    tables = prepare_legacy_c6_modality_context(frames, units)
    frame_path = output_dir / "image_frame_context_manifest.csv"
    window_path = output_dir / "image_window_context_manifest.csv"
    union_selection_path = output_dir / "union_context_selection.csv"
    full_selection_path = output_dir / "full_frame_scene_selection.csv"
    context_audit_path = output_dir / "modality_context_audit.json"
    tables.frame_context.to_csv(
        frame_path,
        index=False,
        lineterminator="\n",
    )
    tables.window_context.to_csv(
        window_path,
        index=False,
        lineterminator="\n",
    )
    tables.union_selection.to_csv(
        union_selection_path,
        index=False,
        lineterminator="\n",
    )
    tables.full_frame_selection.to_csv(
        full_selection_path,
        index=False,
        lineterminator="\n",
    )
    context_audit_path.write_text(
        json.dumps(tables.audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    union_audit: dict[str, Any] | None = None
    if "union_context" in selected_modalities:
        union_root = output_dir / "union_pixel_cache"
        union_audit = build_visual_interaction_cache(
            VisualInteractionCacheConfig(
                frame_context_csv=frame_path,
                output_dir=union_root,
                selection_csv=union_selection_path,
                image_size=IMAGE_SIZE,
                padding_ratio=union_padding_ratio,
                source_type="legacy_recovered",
                preview_limit=0,
                checkpoint_every=256,
                max_open_videos=max_open_videos,
                resume=False,
            )
        )
        if union_audit.get("valid") is not True:
            raise RuntimeError(f"legacy C6 union cache failed={union_audit}")
    full_audit: dict[str, Any] | None = None
    if "full_frame_context" in selected_modalities:
        full_root = output_dir / "full_frame_pixel_cache"
        full_audit = build_legacy_c6_full_frame_pixel_cache(
            tables.full_frame_selection,
            full_root,
        )
    artifacts = {
        "image_frame_context_manifest": _artifact_spec(frame_path),
        "image_window_context_manifest": _artifact_spec(window_path),
        "union_context_selection": _artifact_spec(union_selection_path),
        "full_frame_scene_selection": _artifact_spec(full_selection_path),
        "context_audit": _artifact_spec(context_audit_path),
    }
    if union_audit is not None:
        artifacts.update(
            {
                "union_pixel_manifest": _artifact_spec(
                    union_root / "visual_context_manifest.csv"
                ),
                "union_pixel_audit": _artifact_spec(
                    union_root / "visual_context_cache_audit.json"
                ),
            }
        )
    if full_audit is not None:
        artifacts.update(
            {
                "full_frame_pixel_tensor": _artifact_spec(
                    full_root / f"packed_rgb_{IMAGE_SIZE}_letterbox.npy"
                ),
                "full_frame_pixel_index": _artifact_spec(
                    full_root / "packed_image_cache_index.csv"
                ),
                "full_frame_pixel_audit": _artifact_spec(
                    full_root / "packed_image_cache_audit.json"
                ),
            }
        )
    audit = {
        "schema_version": "classification_v2.legacy_c6_modality_inputs.v1",
        "status": "PASS_LEGACY_C6_MODALITY_INPUTS",
        "lineage_scope": "legacy-only-unreviewed-development",
        "human_review_complete": False,
        "review_status": (
            "operator_cvat_checked_pending_hidden_behavior_double_check"
        ),
        "inputs": {
            name: _artifact_spec(path) for name, path in inputs.items()
        },
        "artifacts": artifacts,
        "context_audit": tables.audit,
        "union_cache_audit": union_audit,
        "full_frame_cache_audit": full_audit,
        "selected_modalities": list(selected_modalities),
        "outer_holdout_media_reads": 0,
        "outer_holdout_features_created": 0,
        "outer_holdout_predictions_created": 0,
        "errors": [],
        "valid": True,
    }
    audit_path = output_dir / "modality_inputs_manifest.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {**audit, "manifest_path": str(audit_path)}


def _artifact_spec(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


if __name__ == "__main__":
    main()
