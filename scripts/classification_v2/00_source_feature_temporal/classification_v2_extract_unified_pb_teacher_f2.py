"""CLI pipeline to generate unified PB teacher features for FULL-T6 and DATA+ V2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.classification_v2.features.unified_pb_teacher_f2 import (  # noqa: E402
    DEFAULT_CANONICAL_MANIFEST_DIR,
    DEFAULT_CANONICAL_SPATIAL_NPZ,
    DEFAULT_DATA_PLUS_DIR,
    DEFAULT_F2_DIR,
    F2_EXPECTED_HASHES,
    extract_unified_pb_teacher_fold,
    load_frozen_f2_model,
    sha256_file,
)

UNIFIED_EXTRACTOR_SOURCE = (
    SRC_ROOT
    / "pig_behavior"
    / "classification_v2"
    / "features"
    / "unified_pb_teacher_f2.py"
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "classification_v2" / "pb_teacher_f2_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified PB Teacher F2 V1 Extraction CLI"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_F2_DIR,
        help="Directory containing F2 checkpoints vg1..vg5/best_validation.pt",
    )
    parser.add_argument(
        "--canonical-manifest-dir",
        type=Path,
        default=DEFAULT_CANONICAL_MANIFEST_DIR,
        help="Directory containing FULL-T6 canonical manifests",
    )
    parser.add_argument(
        "--canonical-spatial-npz",
        type=Path,
        default=DEFAULT_CANONICAL_SPATIAL_NPZ,
        help="Path to full_t6_canonical_46d.npz",
    )
    parser.add_argument(
        "--data-plus-dir",
        type=Path,
        default=DEFAULT_DATA_PLUS_DIR,
        help="Directory containing DATA+ V2 authority artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Target directory for output unified PB teacher artifacts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference compute device (default: cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size",
    )
    parser.add_argument(
        "--fold",
        choices=["all", "vg1", "vg2", "vg3", "vg4", "vg5"],
        default="all",
        help="Extract all folds or one independently reproducible fold",
    )
    return parser.parse_args()


def run_unified_extraction(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=======================================================", flush=True)
    print("STARTING UNIFIED PB TEACHER F2 V1 EXTRACTION PIPELINE", flush=True)
    print(f"Checkpoints Dir:        {args.checkpoint_dir}", flush=True)
    print(f"Canonical Manifest Dir: {args.canonical_manifest_dir}", flush=True)
    print(f"DATA+ V2 Dir:           {args.data_plus_dir}", flush=True)
    print(f"Output Dir:             {output_dir}", flush=True)
    print(f"Compute Device:         {device}", flush=True)
    print("=======================================================\n", flush=True)

    overall_audit: dict[str, Any] = {
        "pipeline_name": "PB_TEACHER_F2_V1_UNIFIED",
        "status": "IN_PROGRESS",
        "checkpoint_dir": str(args.checkpoint_dir),
        "canonical_manifest_dir": str(args.canonical_manifest_dir),
        "data_plus_dir": str(args.data_plus_dir),
        "output_dir": str(output_dir),
        "folds": {},
    }

    total_canon_samples = 0
    total_dp_samples = 0
    total_valid_slots = 0
    total_missing_slots = 0

    folds = (
        ["vg1", "vg2", "vg3", "vg4", "vg5"]
        if args.fold == "all"
        else [args.fold]
    )
    for fold in folds:
        ckpt_path = args.checkpoint_dir / fold / "best_validation.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for fold {fold}: {ckpt_path}"
            )

        ckpt_sha = sha256_file(ckpt_path)
        expected_sha = F2_EXPECTED_HASHES.get(fold)
        print(f"Loading {fold.upper()} teacher: {ckpt_path}", flush=True)
        print(f"  SHA256:   {ckpt_sha}", flush=True)
        print(f"  Expected: {expected_sha}", flush=True)

        model, loaded_sha = load_frozen_f2_model(ckpt_path, device=device)

        # Run extraction
        outputs, manifest_df, fold_audit = extract_unified_pb_teacher_fold(
            model=model,
            fold_id=fold,
            checkpoint_sha256=loaded_sha,
            canonical_manifest_dir=args.canonical_manifest_dir,
            canonical_spatial_npz=args.canonical_spatial_npz,
            data_plus_dir=args.data_plus_dir,
            device=device,
            batch_size=args.batch_size,
        )

        fold_out_dir = output_dir / fold
        fold_out_dir.mkdir(parents=True, exist_ok=True)

        features_pt_path = fold_out_dir / "pb_teacher_features.pt"
        manifest_csv_path = fold_out_dir / "pb_teacher_manifest.csv"
        audit_json_path = fold_out_dir / "pb_teacher_audit.json"

        torch.save(outputs.to_dict(), features_pt_path)
        manifest_df.to_csv(manifest_csv_path, index=False)

        fold_audit["features_pt_path"] = str(features_pt_path)
        fold_audit["features_pt_size_bytes"] = features_pt_path.stat().st_size
        fold_audit["features_pt_sha256"] = sha256_file(features_pt_path)
        fold_audit["manifest_csv_path"] = str(manifest_csv_path)
        fold_audit["manifest_csv_sha256"] = sha256_file(manifest_csv_path)
        fold_audit["persistent_key_schema"] = {
            "canonical": "persisted FULL-T6 window_id/target_id",
            "data_plus": "supplemental_unit_id with persisted sample_key",
            "unknown_or_duplicate_key_policy": "fail_closed",
        }
        fold_audit["source_file_sha256"] = {
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
            str(UNIFIED_EXTRACTOR_SOURCE): sha256_file(
                UNIFIED_EXTRACTOR_SOURCE
            ),
        }
        fold_manifest_path = (
            args.canonical_manifest_dir
            / f"m2_vft_earlystop_{fold}_manifest.csv"
        )
        fold_audit["input_authority_sha256"] = {
            str(fold_manifest_path): sha256_file(fold_manifest_path),
            str(args.canonical_spatial_npz): sha256_file(
                args.canonical_spatial_npz
            ),
            str(args.data_plus_dir / "supplemental_t6_manifest.csv"): (
                sha256_file(
                    args.data_plus_dir / "supplemental_t6_manifest.csv"
                )
            ),
            str(args.data_plus_dir / "supplemental_fold_eligibility.csv"): (
                sha256_file(
                    args.data_plus_dir / "supplemental_fold_eligibility.csv"
                )
            ),
            str(args.data_plus_dir / "data_plus_partner_f2_inputs.npz"): (
                sha256_file(
                    args.data_plus_dir / "data_plus_partner_f2_inputs.npz"
                )
            ),
        }

        with open(audit_json_path, "w", encoding="utf-8") as fp:
            json.dump(fold_audit, fp, indent=2)

        overall_audit["folds"][fold] = fold_audit
        total_canon_samples += fold_audit["canonical_samples"]
        total_dp_samples += fold_audit["data_plus_samples"]
        total_valid_slots += fold_audit["valid_partner_slots"]
        total_missing_slots += fold_audit["missing_partner_slots"]

        avail_pct = fold_audit["partner_slot_availability_rate"] * 100
        miss_pct = fold_audit["missing_slot_rate"] * 100
        print(
            f"-> {fold.upper()} DONE: "
            f"Canonical={fold_audit['canonical_samples']}, "
            f"DATA+={fold_audit['data_plus_samples']} | "
            f"Total={fold_audit['total_samples']} | "
            f"Valid slots={fold_audit['valid_partner_slots']} ({avail_pct:.2f}%) | "
            f"Missing slots={fold_audit['missing_partner_slots']} ({miss_pct:.2f}%)\n",
            flush=True,
        )

    overall_audit["status"] = "PASS"
    tot_slots = total_valid_slots + total_missing_slots
    overall_audit["summary"] = {
        "total_canonical_samples": total_canon_samples,
        "total_data_plus_samples": total_dp_samples,
        "total_unified_samples": total_canon_samples + total_dp_samples,
        "total_valid_partner_slots": total_valid_slots,
        "total_missing_partner_slots": total_missing_slots,
        "overall_availability_rate": total_valid_slots / tot_slots if tot_slots > 0 else 0.0,
        "overall_missing_rate": total_missing_slots / tot_slots if tot_slots > 0 else 0.0,
    }

    summary_json_path = output_dir / "pb_teacher_f2_v1_migration_audit.json"
    with open(summary_json_path, "w", encoding="utf-8") as fp:
        json.dump(overall_audit, fp, indent=2)

    print(
        f"Unified PB extraction complete! Audit written to: {summary_json_path}",
        flush=True,
    )
    return overall_audit


if __name__ == "__main__":
    cli_args = parse_args()
    run_unified_extraction(cli_args)
