"""Run bounded inner-only parity, CPU I/O, and visual checks for RGB resolution."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
    image_sequence_collate,
    letterbox_rgb_uint8,
)
from pig_behavior.classification_v2.datasets.resolution_pipeline import (
    build_inner_resolution_binding,
    resolve_audited_media_path,
    split_image_context_sequence,
)
from pig_behavior.classification_v2.models.balanced.balanced_model import (
    BalancedCausalModel,
)
from pig_behavior.classification_v2.models.balanced.baselines import baseline_config
from pig_behavior.classification_v2.models.balanced.synthetic import (
    SyntheticBatchSpec,
    synthetic_batch,
)
from pig_behavior.classification_v2.models.balanced.visual import (
    SharedFrameVisualEncoder,
    VisualEncoderConfig,
)
from pig_behavior.classification_v2.models.visual_backbones import (
    NO_PRETRAINED_WEIGHTS,
    SUPPORTED_VISUAL_BACKBONES,
    build_visual_frame_encoder,
)

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded inner-only CPU validation for source-backed 64/128/160/224 RGB."
        )
    )
    parser.add_argument("--frame-context-csv", required=True, type=Path)
    parser.add_argument("--window-context-csv", required=True, type=Path)
    parser.add_argument("--inner-selection-csv", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--packed-64-npy", required=True, type=Path)
    parser.add_argument("--packed-64-index-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--parity-windows-per-source", type=int, default=3)
    parser.add_argument("--benchmark-windows-per-source", type=int, default=16)
    parser.add_argument("--dataloader-batch-size", type=int, default=4)
    parser.add_argument("--expected-inner-windows", type=int, default=39454)
    parser.add_argument("--expected-inner-observations", type=int, default=201792)
    args = parser.parse_args()
    if args.parity_windows_per_source <= 0 or args.benchmark_windows_per_source <= 0:
        raise ValueError("bounded sample sizes must be positive")
    if args.dataloader_batch_size <= 0:
        raise ValueError("dataloader_batch_size must be positive")

    binding = build_inner_resolution_binding(
        frame_context_csv=args.frame_context_csv,
        window_context_csv=args.window_context_csv,
        inner_selection_csv=args.inner_selection_csv,
        media_root=args.media_root,
        expected_window_count=args.expected_inner_windows,
        expected_observation_count=args.expected_inner_observations,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    indices_by_source = _indices_by_source(binding.windows)
    parity = _run_64_parity(
        binding,
        packed_npy=args.packed_64_npy,
        packed_index_csv=args.packed_64_index_csv,
        indices_by_source=indices_by_source,
        windows_per_source=args.parity_windows_per_source,
    )
    correctness = _run_runtime_correctness(binding, indices_by_source)
    benchmark = _run_io_benchmark(
        binding,
        packed_npy=args.packed_64_npy,
        packed_index_csv=args.packed_64_index_csv,
        indices_by_source=indices_by_source,
        windows_per_source=args.benchmark_windows_per_source,
        batch_size=args.dataloader_batch_size,
    )
    contact_sheet = _write_contact_sheet(binding, args.output_dir)
    model_smoke = _run_model_smoke(binding, indices_by_source)
    audit = {
        "schema_version": "classification_v2_resolution_runtime_check_v1",
        "scientific_identity_sha256": binding.identity_sha256,
        "inner_window_count": binding.window_count,
        "inner_observation_count": binding.observation_count,
        "outer_examples_accessed": False,
        "worker_configuration": {"num_workers": 0},
        "old64_new64_parity": parity,
        "cpu_correctness": correctness,
        "io_benchmark": benchmark,
        "cvat_video_random_access": _classify_cvat_random_access(benchmark),
        "on_the_fly_224_io_feasible": _classify_224_feasibility(benchmark),
        "contact_sheet": contact_sheet,
        "model_forward": model_smoke,
        "pretrained_backbone_interface_available": {
            "resnet18": "resnet18" in SUPPORTED_VISUAL_BACKBONES,
            "resnet34": "resnet34" in SUPPORTED_VISUAL_BACKBONES,
        },
    }
    audit_path = args.output_dir / "resolution_runtime_check.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


def _indices_by_source(windows: pd.DataFrame) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for source_type, source_windows in windows.groupby("source_type", sort=True):
        ordered = source_windows.sort_values(
            ["video_key", "object_track_key", "window_start_frame", "window_id"],
            kind="mergesort",
        )
        result[str(source_type)] = ordered.index.astype(int).tolist()
    required = {"cvat_tracking_xml", "legacy_recovered"}
    if set(result) != required:
        raise ValueError(f"inner source set mismatch: {sorted(result)}")
    return result


def _run_64_parity(
    binding,
    *,
    packed_npy: Path,
    packed_index_csv: Path,
    indices_by_source: dict[str, list[int]],
    windows_per_source: int,
) -> dict[str, Any]:
    old = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_dataframe=binding.frames,
            window_context_dataframe=binding.windows,
            packed_image_cache_npy=packed_npy,
            packed_image_cache_index_csv=packed_index_csv,
            image_size=64,
            image_cache_size=0,
            media_root=binding.media_root,
        )
    )
    source = binding.build_dataset(64, image_cache_size=0)
    reports: dict[str, Any] = {}
    try:
        for source_type, indices in indices_by_source.items():
            comparisons = []
            for index in indices[:windows_per_source]:
                cached = old[index]
                runtime = source[index]
                same_identity = (
                    cached["image_context_ids"] == runtime["image_context_ids"]
                    and cached["expected_frame_indices"] == runtime["expected_frame_indices"]
                    and cached["window_id"] == runtime["window_id"]
                )
                cached_array = cached["image"].numpy()
                runtime_array = runtime["image"].numpy()
                comparisons.append(
                    {
                        "window_id": runtime["window_id"],
                        "same_scientific_identity": same_identity,
                        "bitwise_identical": bool(
                            np.array_equal(cached_array, runtime_array)
                        ),
                        "numeric_max_abs_difference": float(
                            np.abs(cached_array - runtime_array).max()
                        ),
                        "cached_errors": cached["errors"],
                        "runtime_errors": runtime["errors"],
                    }
                )
            reports[source_type] = _classify_parity(comparisons)
    finally:
        old.close()
        source.close()
    classifications = [report["classification"] for report in reports.values()]
    overall = _worst_parity_classification(classifications)
    return {
        "classification": overall,
        "source_reports": reports,
        "mismatch_reason": _parity_reason(overall, reports),
    }


def _classify_parity(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    identity_ok = all(item["same_scientific_identity"] for item in comparisons)
    errors = any(item["cached_errors"] or item["runtime_errors"] for item in comparisons)
    bitwise = all(item["bitwise_identical"] for item in comparisons)
    numeric = all(item["numeric_max_abs_difference"] <= 1e-7 for item in comparisons)
    if identity_ok and not errors and bitwise:
        classification = "BITWISE_IDENTICAL"
    elif identity_ok and not errors and numeric:
        classification = "NUMERICALLY_EQUIVALENT"
    elif identity_ok and not errors:
        classification = "SEMANTICALLY_EQUIVALENT_BUT_NOT_NUMERICALLY_IDENTICAL"
    else:
        classification = "FAIL"
    return {
        "classification": classification,
        "compared_windows": len(comparisons),
        "comparisons": comparisons,
    }


def _worst_parity_classification(values: list[str]) -> str:
    order = [
        "BITWISE_IDENTICAL",
        "NUMERICALLY_EQUIVALENT",
        "SEMANTICALLY_EQUIVALENT_BUT_NOT_NUMERICALLY_IDENTICAL",
        "FAIL",
    ]
    return max(values, key=order.index)


def _parity_reason(overall: str, reports: dict[str, Any]) -> str:
    if overall == "BITWISE_IDENTICAL":
        return "none"
    if overall == "FAIL":
        return "identity_or_media-load mismatch; inspect source reports"
    mismatched = [
        source
        for source, report in reports.items()
        if report["classification"] != "BITWISE_IDENTICAL"
    ]
    return (
        "source cache reproduction differs for "
        f"{mismatched}; inspect interpolation, source compression, crop rounding, "
        "color conversion, and padding in the recorded comparison."
    )


def _run_runtime_correctness(binding, indices_by_source: dict[str, list[int]]) -> dict[str, Any]:
    selected = _deterministic_correctness_indices(binding, indices_by_source)
    checks: dict[str, Any] = {}
    reference: dict[int, dict[str, Any]] = {}
    for resolution in (64, 128, 160, 224):
        dataset = binding.build_dataset(resolution, image_cache_size=0)
        try:
            rows = []
            for selection_key, index in selected.items():
                item = dataset[index]
                rows.append(
                    {
                        "selection_key": selection_key,
                        "window_id": item["window_id"],
                        "shape": [int(value) for value in item["image"].shape],
                        "image_context_ids": item["image_context_ids"],
                        "expected_frame_indices": item["expected_frame_indices"],
                        "errors": item["errors"],
                        "label": str(
                            binding.windows.iloc[index]["_resolution_binding_label"]
                        ),
                        "role": str(
                            binding.windows.iloc[index]["_resolution_binding_role"]
                        ),
                    }
                )
            if resolution == 64:
                reference = {index: row for index, row in zip(selected.values(), rows, strict=True)}
            identity_ok = all(
                row["image_context_ids"] == reference[index]["image_context_ids"]
                and row["expected_frame_indices"] == reference[index]["expected_frame_indices"]
                and row["label"] == reference[index]["label"]
                and row["role"] == reference[index]["role"]
                for index, row in zip(selected.values(), rows, strict=True)
            )
            shape_ok = all(
                row["shape"] == [6, 3, resolution, resolution]
                for row in rows
            )
            errors = [row for row in rows if row["errors"]]
            checks[str(resolution)] = {
                "status": "PASS" if identity_ok and shape_ok and not errors else "FAIL",
                "selected_windows": rows,
                "identity_order_label_role_unchanged": identity_ok,
                "shapes_exact": shape_ok,
                "media_errors": errors,
            }
        finally:
            dataset.close()
    checks["missing_media_fails_closed"] = _check_missing_media_fails_closed(binding)
    checks["condition_availability"] = {
        name: {
            "real_inner_observation_available": bool(mask.any()),
            "selected_window_id": str(
                binding.windows.iloc[selected[f"condition:{name}"]]["window_id"]
            ),
        }
        for name, mask in _inner_frame_condition_masks(binding).items()
    }
    return checks


def _deterministic_correctness_indices(
    binding,
    indices_by_source: dict[str, list[int]],
) -> dict[str, int]:
    labels = ["fight", "social-nose", "move", "playwithtoy", "sitting"]
    result: dict[str, int] = {}
    windows = binding.windows
    for label in labels:
        candidates = windows.loc[
            windows["_resolution_binding_label"].astype(str).eq(label)
        ].sort_values("window_id", kind="mergesort")
        if candidates.empty:
            raise ValueError(f"missing required deterministic class sample: {label}")
        result[f"class:{label}"] = int(candidates.index[0])
    for name, mask in _inner_frame_condition_masks(binding).items():
        candidates = binding.frames.loc[mask].sort_values(
            "image_context_id",
            kind="mergesort",
        )
        if candidates.empty:
            result[f"condition:{name}"] = indices_by_source["legacy_recovered"][0]
            continue
        context_id = str(candidates.iloc[0]["image_context_id"])
        matching = windows.loc[
            windows["image_context_id_sequence"].astype(str).map(
                lambda value, target_context_id=context_id: target_context_id
                in split_image_context_sequence(value)
            )
        ].sort_values("window_id", kind="mergesort")
        if matching.empty:
            raise ValueError(f"condition context is absent from bound windows: {name}")
        result[f"condition:{name}"] = int(matching.index[0])
    return result


def _inner_frame_condition_masks(binding) -> dict[str, pd.Series]:
    return {
        "portrait": (binding.frames["x2"] - binding.frames["x1"])
        <= (binding.frames["y2"] - binding.frames["y1"]) * 0.67,
        "landscape": (binding.frames["x2"] - binding.frames["x1"])
        >= (binding.frames["y2"] - binding.frames["y1"]) * 1.5,
        "edge_of_frame": (
            (binding.frames["x1"] <= 0)
            | (binding.frames["y1"] <= 0)
            | (binding.frames["x2"] >= binding.frames["image_width"])
            | (binding.frames["y2"] >= binding.frames["image_height"])
        ),
        "partially_clipped": (
            (binding.frames["x1"] < 0)
            | (binding.frames["y1"] < 0)
            | (binding.frames["x2"] > binding.frames["image_width"])
            | (binding.frames["y2"] > binding.frames["image_height"])
        ),
    }


def _check_missing_media_fails_closed(binding) -> dict[str, Any]:
    legacy = binding.frames.loc[
        binding.frames["source_type"].astype(str).eq("legacy_recovered")
    ].sort_values("image_context_id", kind="mergesort")
    context_id = str(legacy.iloc[0]["image_context_id"])
    frames = binding.frames.copy()
    frames.loc[
        frames["image_context_id"].astype(str).eq(context_id),
        "resolved_media_path",
    ] = "missing_inner_media_for_fail_closed_check.jpg"
    dataset = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_dataframe=frames,
            window_context_dataframe=binding.windows,
            image_size=64,
            image_cache_size=0,
            media_root=binding.media_root,
        )
    )
    try:
        index = next(
            int(row.index)
            for row in binding.windows.reset_index().itertuples(index=False)
            if context_id in split_image_context_sequence(
                str(row.image_context_id_sequence)
            )
        )
        item = dataset[index]
        missing_slots = int((item["observed_mask"] == 0).sum().item())
        return {
            "status": "PASS" if missing_slots >= 1 and item["errors"] else "FAIL",
            "window_id": item["window_id"],
            "missing_slots": missing_slots,
            "errors": item["errors"],
        }
    finally:
        dataset.close()


def _run_io_benchmark(
    binding,
    *,
    packed_npy: Path,
    packed_index_csv: Path,
    indices_by_source: dict[str, list[int]],
    windows_per_source: int,
    batch_size: int,
) -> dict[str, Any]:
    source_indices = {
        source: indices[:windows_per_source]
        for source, indices in indices_by_source.items()
    }
    packed_indices = [
        index
        for source in sorted(source_indices)
        for index in source_indices[source]
    ]
    packed = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_dataframe=binding.frames,
            window_context_dataframe=binding.windows,
            packed_image_cache_npy=packed_npy,
            packed_image_cache_index_csv=packed_index_csv,
            image_size=64,
            image_cache_size=0,
            media_root=binding.media_root,
        )
    )
    try:
        packed_report = _benchmark_dataset(packed, packed_indices, batch_size)
    finally:
        packed.close()
    runtime: dict[str, dict[str, Any]] = {}
    for resolution in (64, 160, 224):
        runtime[str(resolution)] = {}
        for source_type, indices in source_indices.items():
            dataset = binding.build_dataset(resolution, image_cache_size=0)
            try:
                runtime[str(resolution)][source_type] = _benchmark_dataset(
                    dataset,
                    indices,
                    batch_size,
                )
            finally:
                dataset.close()
    return {"packed64": packed_report, "runtime": runtime}


def _benchmark_dataset(
    dataset: ClassificationV2ImageSequenceDataset,
    indices: list[int],
    batch_size: int,
) -> dict[str, Any]:
    process_before = _process_memory_bytes()
    started = time.perf_counter()
    frames = 0
    errors = 0
    for index in indices:
        item = dataset[index]
        frames += len(item["image_context_ids"])
        errors += len(item["errors"])
    direct_elapsed = time.perf_counter() - started
    direct_audit = dataset.image_load_audit()
    dataloader_dataset = dataset
    loader = DataLoader(
        Subset(dataloader_dataset, indices),
        batch_size=batch_size,
        num_workers=0,
        collate_fn=image_sequence_collate,
    )
    started = time.perf_counter()
    loaded_windows = 0
    for batch in loader:
        loaded_windows += len(batch["window_id"])
    dataloader_elapsed = time.perf_counter() - started
    return {
        "windows": len(indices),
        "frames": frames,
        "media_errors": errors,
        "direct": _rate_report(len(indices), frames, direct_elapsed),
        "dataloader_num_workers": 0,
        "dataloader": _rate_report(loaded_windows, frames, dataloader_elapsed),
        "image_load_audit": direct_audit,
        "peak_process_memory_bytes_if_available": _process_memory_bytes(),
        "process_memory_delta_bytes_if_available": _optional_delta(
            process_before,
            _process_memory_bytes(),
        ),
    }


def _rate_report(windows: int, frames: int, elapsed: float) -> dict[str, float]:
    safe_elapsed = max(elapsed, 1e-12)
    return {
        "elapsed_seconds": round(elapsed, 6),
        "samples_per_second": round(windows / safe_elapsed, 6),
        "frames_per_second": round(frames / safe_elapsed, 6),
    }


def _process_memory_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _optional_delta(before: int | None, after: int | None) -> int | None:
    return None if before is None or after is None else after - before


def _classify_cvat_random_access(benchmark: dict[str, Any]) -> str:
    reports = [
        benchmark["runtime"][resolution]["cvat_tracking_xml"]
        for resolution in ("64", "160", "224")
    ]
    seeks = [
        report["image_load_audit"]["video_seek_count"]
        for report in reports
    ]
    frames = [report["frames"] for report in reports]
    if any(report["media_errors"] for report in reports):
        return "BLOCKING"
    if all(
        seek <= max(1, frame_count // 6)
        for seek, frame_count in zip(seeks, frames, strict=True)
    ):
        return "EFFICIENT"
    if all(seek <= frame_count for seek, frame_count in zip(seeks, frames, strict=True)):
        return "ACCEPTABLE_WITH_DATALOADER"
    return "CACHE_RECOMMENDED"


def _classify_224_feasibility(benchmark: dict[str, Any]) -> str:
    reports = benchmark["runtime"]["224"]
    if any(report["media_errors"] for report in reports.values()):
        return "NO"
    rates = [report["direct"]["frames_per_second"] for report in reports.values()]
    return "YES" if min(rates) >= 10.0 else "LIKELY"


def _write_contact_sheet(binding, output_dir: Path) -> dict[str, Any]:
    selected = _deterministic_correctness_indices(
        binding,
        _indices_by_source(binding.windows),
    )
    keys = [
        "class:fight",
        "class:social-nose",
        "class:move",
        "class:playwithtoy",
        "class:sitting",
    ]
    samples = [(key, selected[key]) for key in keys]
    datasets = {
        resolution: binding.build_dataset(resolution, image_cache_size=0)
        for resolution in (64, 160, 224)
    }
    try:
        panel_size = 224
        label_height = 42
        canvas = Image.new(
            "RGB",
            (panel_size * 4, (panel_size + label_height) * len(samples)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        sample_records = []
        for row_index, (key, window_index) in enumerate(samples):
            window = binding.windows.iloc[window_index]
            context_id = split_image_context_sequence(
                str(window["image_context_id_sequence"])
            )[0]
            frame = binding.frames.set_index("image_context_id").loc[context_id]
            native = _load_native_crop(frame, binding.media_root)
            images = [native]
            for resolution in (64, 160, 224):
                item = datasets[resolution][window_index]
                if item["errors"]:
                    raise RuntimeError(
                        f"contact-sheet runtime media error: {item['errors']}"
                    )
                rgb = np.transpose(item["image"][0].numpy(), (1, 2, 0))
                images.append(Image.fromarray((rgb * 255.0).round().astype(np.uint8)))
            y = row_index * (panel_size + label_height)
            labels = [
                f"{key.replace('class:', '')} native {native.width}x{native.height}",
                "runtime 64",
                "runtime 160",
                "runtime 224",
            ]
            for col, (image, text) in enumerate(zip(images, labels, strict=True)):
                if col == 1:
                    image = image.resize((panel_size, panel_size), Image.Resampling.NEAREST)
                else:
                    image = letterbox_rgb_uint8(np.asarray(image.convert("RGB")), panel_size)
                    image = Image.fromarray(image)
                canvas.paste(image, (col * panel_size, y))
                draw.text((col * panel_size + 4, y + panel_size + 4), text, fill="black")
            sample_records.append(
                {
                    "selection_key": key,
                    "window_id": str(window["window_id"]),
                    "image_context_id": context_id,
                    "source_type": str(frame["source_type"]),
                    "label": str(window["_resolution_binding_label"]),
                }
            )
        sheet_path = output_dir / "resolution_contact_sheet_inner.png"
        samples_path = output_dir / "resolution_contact_sheet_samples.json"
        canvas.save(sheet_path)
        samples_path.write_text(json.dumps(sample_records, indent=2), encoding="utf-8")
        return {
            "path": str(sheet_path),
            "sample_ids_path": str(samples_path),
            "sample_count": len(sample_records),
            "samples": sample_records,
        }
    finally:
        for dataset in datasets.values():
            dataset.close()


def _load_native_crop(frame: pd.Series, media_root: Path) -> Image.Image:
    source_type = str(frame["source_type"])
    path = resolve_audited_media_path(frame["resolved_media_path"], media_root)
    if source_type == "legacy_recovered":
        with Image.open(path) as image:
            return image.convert("RGB").copy()
    if source_type != "cvat_tracking_xml" or cv2 is None:
        raise RuntimeError(f"native crop source is unavailable: {source_type}")
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame["frame_index"]))
        ok, image_bgr = capture.read()
    finally:
        capture.release()
    if not ok or image_bgr is None:
        raise RuntimeError(f"contact-sheet video decode failed: {path}")
    height, width = image_bgr.shape[:2]
    x1 = max(0, min(width, int(float(frame["x1"]))))
    y1 = max(0, min(height, int(float(frame["y1"]))))
    x2 = max(0, min(width, int(float(frame["x2"]))))
    y2 = max(0, min(height, int(float(frame["y2"]))))
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError("contact-sheet crop is empty")
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def _run_model_smoke(binding, indices_by_source: dict[str, list[int]]) -> dict[str, Any]:
    index = indices_by_source["legacy_recovered"][0]
    frame_encoder_shapes: dict[str, list[int]] = {}
    sequence_encoder_shapes: dict[str, list[int]] = {}
    full_model_shapes: dict[str, list[int]] = {}
    for resolution in (64, 128, 160, 224):
        dataset = binding.build_dataset(resolution, image_cache_size=0)
        try:
            item = dataset[index]
            if item["errors"]:
                raise RuntimeError(f"model smoke media error: {item['errors']}")
            model, contract = build_visual_frame_encoder(
                "smoke_cnn",
                NO_PRETRAINED_WEIGHTS,
            )
            model.eval()
            sequence_encoder = SharedFrameVisualEncoder(VisualEncoderConfig())
            sequence_encoder.eval()
            with torch.no_grad():
                frame_output = model(item["image"])
                sequence_output = sequence_encoder(
                    item["image"].unsqueeze(0),
                    item["observed_mask"].unsqueeze(0),
                )
                config = baseline_config(
                    "B1_ACTOR_T6_SEQUENCE",
                    image_size=resolution,
                )
                full_model = BalancedCausalModel(config)
                full_model.eval()
                full_output = full_model(
                    synthetic_batch(
                        SyntheticBatchSpec(
                            contract=config.batch_contract,
                            batch_size=1,
                            image_size=resolution,
                        )
                    )
                )["logits"]
            frame_encoder_shapes[str(resolution)] = [
                int(value) for value in frame_output.shape
            ]
            sequence_encoder_shapes[str(resolution)] = [
                int(value) for value in sequence_output.shape
            ]
            full_model_shapes[str(resolution)] = [
                int(value) for value in full_output.shape
            ]
            if tuple(frame_output.shape) != (6, contract.output_dim):
                raise RuntimeError(
                    "smoke CNN forward shape mismatch at "
                    f"{resolution}: {frame_output.shape}"
                )
            if tuple(sequence_output.shape) != (1, 6, contract.output_dim):
                raise RuntimeError(
                    "shared sequence encoder shape mismatch at "
                    f"{resolution}: {sequence_output.shape}"
                )
            if tuple(full_output.shape) != (1, config.batch_contract.num_classes):
                raise RuntimeError(
                    "full B1 model shape mismatch at "
                    f"{resolution}: {full_output.shape}"
                )
        finally:
            dataset.close()
    return {
        "current_model_resolution_agnostic": True,
        "backbone": "smoke_cnn",
        "frame_encoder_output_shapes": frame_encoder_shapes,
        "shared_sequence_encoder_output_shapes": sequence_encoder_shapes,
        "full_b1_synthetic_batch_logits_shapes": full_model_shapes,
    }


if __name__ == "__main__":
    main()
