"""CLI entrypoint for extracting PB Teacher features using frozen F2 H5 checkpoints."""

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

from pig_behavior.classification_v2.features.pb_teacher_f2_extraction import (  # noqa: E402
    DEFAULT_DATA_PLUS_DIR,
    DEFAULT_F2_CHECKPOINT_DIR,
    F2_EXPECTED_HASHES,
    extract_pb_features_from_data_plus,
    load_frozen_f2_teacher,
    sha256_file,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "classification_v2"
    / "pb_teacher_f2_v1_features"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PB Teacher F2 V1 Extraction CLI"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_F2_CHECKPOINT_DIR,
        help="Directory containing F2 checkpoints vg1..vg5/best_validation.pt",
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
        help="Target directory for output PB teacher artifacts",
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
        default=32,
        help="Inference batch size",
    )
    return parser.parse_args()


def run_extraction(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=======================================================", flush=True)
    print("STARTING PB TEACHER F2 V1 EXTRACTION PIPELINE", flush=True)
    print(f"Checkpoints Dir: {args.checkpoint_dir}", flush=True)
    print(f"DATA+ V2 Dir:    {args.data_plus_dir}", flush=True)
    print(f"Output Dir:      {output_dir}", flush=True)
    print(f"Compute Device:  {device}", flush=True)
    print("=======================================================\n", flush=True)

    overall_audit: dict[str, Any] = {
        "pipeline_name": "PB_TEACHER_F2_V1",
        "status": "IN_PROGRESS",
        "checkpoint_dir": str(args.checkpoint_dir),
        "data_plus_dir": str(args.data_plus_dir),
        "output_dir": str(output_dir),
        "folds": {},
    }

    total_extracted_samples = 0
    total_valid_slots = 0
    total_missing_slots = 0

    for fold in ["vg1", "vg2", "vg3", "vg4", "vg5"]:
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

        model, loaded_sha = load_frozen_f2_teacher(ckpt_path, device=device)

        # Run extraction
        outputs, manifest_df, fold_audit = extract_pb_features_from_data_plus(
            model=model,
            fold_id=fold,
            checkpoint_sha256=loaded_sha,
            data_plus_dir=args.data_plus_dir,
            device=device,
            batch_size=args.batch_size,
        )

        fold_out_dir = output_dir / fold
        fold_out_dir.mkdir(parents=True, exist_ok=True)

        features_pt_path = fold_out_dir / "partner_teacher_features.pt"
        manifest_csv_path = fold_out_dir / "partner_teacher_manifest.csv"

        torch.save(outputs.to_dict(), features_pt_path)
        manifest_df.to_csv(manifest_csv_path, index=False)

        fold_audit["features_pt_path"] = str(features_pt_path)
        fold_audit["features_pt_size_bytes"] = features_pt_path.stat().st_size
        fold_audit["manifest_csv_path"] = str(manifest_csv_path)

        overall_audit["folds"][fold] = fold_audit
        total_extracted_samples += fold_audit["total_samples"]
        total_valid_slots += fold_audit["valid_partner_slots"]
        total_missing_slots += fold_audit["missing_partner_slots"]

        avail_pct = fold_audit["partner_slot_availability_rate"] * 100
        miss_pct = fold_audit["missing_slot_rate"] * 100
        print(
            f"-> {fold.upper()} DONE: {fold_audit['total_samples']} samples | "
            f"Valid slots: {fold_audit['valid_partner_slots']} ({avail_pct:.2f}%) | "
            f"Missing slots: {fold_audit['missing_partner_slots']} ({miss_pct:.2f}%)\n",
            flush=True,
        )

    overall_audit["status"] = "PASS"
    overall_audit["summary"] = {
        "total_extracted_samples_across_folds": total_extracted_samples,
        "total_valid_partner_slots": total_valid_slots,
        "total_missing_partner_slots": total_missing_slots,
        "overall_availability_rate": (
            total_valid_slots / (total_valid_slots + total_missing_slots)
        ),
        "overall_missing_rate": (
            total_missing_slots / (total_valid_slots + total_missing_slots)
        ),
    }

    audit_json_path = output_dir / "pb_teacher_f2_v1_audit.json"
    with open(audit_json_path, "w", encoding="utf-8") as fp:
        json.dump(overall_audit, fp, indent=2)

    print(f"Extraction complete! Audit written to: {audit_json_path}", flush=True)
    return overall_audit


if __name__ == "__main__":
    cli_args = parse_args()
    run_extraction(cli_args)
