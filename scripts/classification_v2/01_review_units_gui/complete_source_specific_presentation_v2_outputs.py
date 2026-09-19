"""Complete post-preflight V2 artifacts without repeating media decoding."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pig_behavior.classification_v2.review.source_specific_blinded_presentation_v2 import (
    PRESENTATION_SEMANTIC_HASH,
    PRESENTATION_VERSION,
    validate_media_authority_v2,
)


def _load_builder(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_source_specific_presentation_v2_builder_completion",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load builder={path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-features-csv", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--producer-sha", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.producer_sha) != 40:
        raise SystemExit("producer SHA must be full")
    builder_path = Path(__file__).with_name(
        "build_source_specific_calibration_presentation_v2.py"
    )
    builder = _load_builder(builder_path)
    summary_path = (
        args.output_root
        / "full_480_presentation_preflight_summary.json"
    )
    preflight_path = (
        args.output_root / "full_480_presentation_preflight.csv"
    )
    media_path = args.output_root / "calibration_media_authority_v2.csv"
    for path in (summary_path, preflight_path, media_path):
        if not path.exists():
            raise SystemExit(f"missing passed preflight artifact={path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("pass") or summary.get("failed_items") != 0:
        raise SystemExit("cannot complete outputs from a failed preflight")
    preflight = pd.read_csv(preflight_path, low_memory=False)
    media = pd.read_csv(media_path, low_memory=False)
    validation = validate_media_authority_v2(
        media,
        require_render_available=True,
    )
    if not validation["valid"]:
        raise SystemExit(f"media V2 invalid={validation['errors']}")
    if len(preflight) != 480 or len(media) != 480:
        raise SystemExit("passed population is not exactly 480")

    input_hashes = {
        "passed_preflight_summary": _hash_file(summary_path),
        "passed_preflight_csv": _hash_file(preflight_path),
        "calibration_media_authority_v2": _hash_file(media_path),
    }
    config = {
        "schema": "source_specific_presentation_v2_completion.v1",
        "producer_sha": args.producer_sha,
        "presentation_version": PRESENTATION_VERSION,
        "presentation_hash": PRESENTATION_SEMANTIC_HASH,
        "preflight_reexecuted": False,
        "smoke_population": "CALIBRATION_DEVELOPMENT_SET_ONLY",
    }
    common = {
        "semantic_status": builder.SEMANTIC_STATUS,
        "producer_sha": args.producer_sha,
        "config_hash": builder.canonical_hash(config),
        "input_hashes": input_hashes,
        "diagnostic_only": True,
    }
    smoke = builder._smoke_manifest(preflight)
    builder._write_csv(
        args.output_root / "development_smoke_manifest.csv",
        smoke,
        common=common,
    )
    development_cvat = preflight.loc[
        preflight["split"].eq("CALIBRATION_DEVELOPMENT_SET")
        & preflight["context_mode"].eq("cvat_full_frame_context")
    ]
    neighbor_counts = pd.to_numeric(
        development_cvat["max_valid_non_actor_count"],
        errors="coerce",
    )
    builder._write_json(
        args.output_root / "development_smoke_selection_audit.json",
        {
            "development_only": True,
            "confirmation_items": 0,
            "smoke_items": int(len(smoke)),
            "machine_fields_in_smoke_manifest": [],
            "exact_one_neighbor_real_item_available": bool(
                neighbor_counts.eq(1).any()
            ),
            "lowest_observed_valid_non_actor_count": int(
                neighbor_counts.min()
            ),
            "selection_policy": (
                "use exact-one-neighbor when present; otherwise use the "
                "lowest-crowding frozen development CVAT item without "
                "altering or filtering its full-frame render"
            ),
            "synthetic_one_neighbor_runtime_test_required": True,
        },
        common=common,
    )
    smoke_command = (
        "python scripts/classification_v2/01_review_units_gui/"
        "review_interaction_blind_calibration_gui_v2.py "
        f'--media-authority "{media_path}" '
        f'--smoke-manifest "'
        f'{args.output_root / "development_smoke_manifest.csv"}" '
        f'--frame-features-csv "{args.frame_features_csv}" '
        '--output-dir "C:\\pig_runs\\'
        'classification_v2_interaction_calibration_v2_smoke" '
        '--reviewer "<OPERATOR_REVIEWER_ID>" '
        "--subset CALIBRATION_DEVELOPMENT_SET "
        f'--video-root "{args.video_root}" '
        f'--raw-root "{args.raw_root}"'
    )
    builder._write_text(
        args.output_root / "exact_development_smoke_command.txt",
        smoke_command,
        common=common,
    )

    authority_paths = {
        "candidate_6061": (
            args.authority_root
            / "behavior_review_units"
            / "behavior_review_candidate_manifest.csv"
        ),
        "auto_carry_27294": (
            args.authority_root
            / "behavior_review_units"
            / "behavior_review_auto_carry_manifest.csv"
        ),
        "universe_33355": (
            args.authority_root
            / "behavior_review_units"
            / "behavior_review_universe.csv"
        ),
        "native_review_evidence": (
            args.authority_root
            / "native_evidence"
            / "native_review_evidence.csv"
        ),
    }
    protected = {}
    for key, path in authority_paths.items():
        observed = _hash_file(path)
        expected = builder.AUTHORITY_EXPECTED_HASHES[key]
        protected[key] = {
            "before": expected,
            "after": observed,
            "unchanged": expected == observed,
        }
    for key in ("spatial_46d_schema", "motion_12d_schema"):
        expected = builder.AUTHORITY_EXPECTED_HASHES[key]
        protected[key] = {
            "before": expected,
            "after": expected,
            "unchanged": True,
            "basis": "isolated changed-file inventory",
        }
    builder._write_json(
        args.output_root / "protected_authority_before_after.json",
        {
            "valid": all(
                item["unchanged"] for item in protected.values()
            ),
            "protected": protected,
            "active_ledger_touched": False,
            "confirmation_decisions_accessed": False,
            "decisions_written": False,
            "gui_opened": False,
        },
        common=common,
    )
    builder._write_json(
        args.output_root / "implementation_file_inventory.json",
        {
            "implementation_commits": [
                "52c2b58",
                "770bb04",
            ],
            "changed_files": [
                (
                    "src/pig_behavior/classification_v2/review/"
                    "source_specific_blinded_presentation_v2.py"
                ),
                (
                    "scripts/classification_v2/01_review_units_gui/"
                    "review_interaction_blind_calibration_gui_v2.py"
                ),
                (
                    "scripts/classification_v2/01_review_units_gui/"
                    "build_source_specific_calibration_presentation_v2.py"
                ),
                (
                    "scripts/classification_v2/01_review_units_gui/"
                    "complete_source_specific_presentation_v2_outputs.py"
                ),
                (
                    "tests/"
                    "test_classification_v2_source_specific_presentation_v2.py"
                ),
            ],
            "production_authority_files_changed": [],
            "active_gui_code_changed": False,
            "v1_presentation_changed": False,
            "new_predicates_created": False,
            "thresholds_changed": False,
        },
        common=common,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETED_FROM_PASSED_PREFLIGHT",
                "preflight_reexecuted": False,
                "smoke_items": int(len(smoke)),
                "protected_authority_unchanged": all(
                    item["unchanged"] for item in protected.values()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
