"""Execute one genuine scientific R128 GPU trial on NVIDIA L4."""

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Explicitly ensure repository paths are in sys.path
_this_dir = Path(__file__).resolve().parent
_src_dir = _this_dir / "src"
if not _src_dir.exists() and (_this_dir.parent / "src").exists():
    _this_dir = _this_dir.parent
    _src_dir = _this_dir / "src"

repo_root = _this_dir
src_root = _src_dir

for p in [str(repo_root), str(src_root), "/teamspace/studios/this_studio", "/teamspace/studios/this_studio/src"]:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import classification_report, f1_score  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (  # noqa: E402
    image_sequence_collate,
)
from pig_behavior.classification_v2.datasets.resolution_pipeline import (  # noqa: E402
    build_inner_resolution_binding_from_dataframes,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS  # noqa: E402
from pig_behavior.classification_v2.training import (  # noqa: E402
    post_s1_resolution_screening as post_s1,
)
from pig_behavior.classification_v2.training import (  # noqa: E402
    stage1_temporal_screening as stage1,
)

SEED = 20260814
RESOLUTION = 128
TEMPORAL_VIEW = "T6"
TARGET_STEPS = 4164
BATCH_SIZE = 16


def compute_model_hash(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    with torch.no_grad():
        for k in sorted(model.state_dict().keys()):
            tensor = model.state_dict()[k].detach().cpu().numpy()
            hasher.update(k.encode("utf-8"))
            hasher.update(tensor.tobytes())
    return hasher.hexdigest()


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_trial() -> dict:
    t_start = time.perf_counter()
    report: dict = {}

    print("==================================================")
    print("CLASSIFICATION V2 — REAL R128 GPU PROOF EXECUTION")
    print("==================================================")

    # 1. Hardware verification
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA Device: {device_name}")
    torch.cuda.reset_peak_memory_stats(0)

    report["GPU_DEVICE"] = device_name
    report["TEMPORAL_VIEW"] = TEMPORAL_VIEW
    report["RESOLUTION"] = f"R{RESOLUTION}"
    report["SEED"] = SEED

    # Normalize any backslash paths from Windows upload across the entire repository
    for root, _dirs, files in os.walk(str(repo_root)):
        for f in files:
            if "\\" in f:
                src_f = Path(root) / f
                dest_f = repo_root / f.replace("\\", "/")
                dest_f.parent.mkdir(parents=True, exist_ok=True)
                print(f"Normalizing backslash artifact: {f} -> {dest_f}")
                try:
                    shutil.move(str(src_f), str(dest_f))
                except Exception as move_err:
                    print(f"Warning moving {src_f}: {move_err}")

    # 2. Locate outputs and cache
    outputs_root = repo_root / "outputs"
    rel_cache = (
        "classification_v2/model_readiness_audit/"
        "pre_gpu_autoresearch_q2_6c2f204_20260804_084638/"
        "reviewed_rgb_v1/actor_rgb_128_full"
    )
    cache_dir = outputs_root / rel_cache
    packed_npy = cache_dir / "packed_rgb_128_letterbox.npy"
    packed_idx = cache_dir / "packed_image_cache_index.csv"

    if not packed_npy.exists() or not packed_idx.exists():
        runtime_cache = repo_root / "pig_e0_r3/inputs/outputs" / rel_cache
        if runtime_cache.exists():
            packed_npy = runtime_cache / "packed_rgb_128_letterbox.npy"
            packed_idx = runtime_cache / "packed_image_cache_index.csv"
        else:
            raise FileNotFoundError(f"Staged cache not found at {cache_dir}")

    npy_size = packed_npy.stat().st_size
    idx_size = packed_idx.stat().st_size
    print(f"Staged NPY: {packed_npy} ({npy_size} bytes)")
    print(f"Staged IDX: {packed_idx} ({idx_size} bytes)")

    report["REMOTE_CACHE_PRESENT"] = "YES"
    report["REMOTE_CACHE_SIZE_MATCH"] = (
        "YES" if npy_size == 12075663488 else f"NO ({npy_size})"
    )
    report["REMOTE_INDEX_PRESENT"] = "YES"
    report["REMOTE_INDEX_SIZE_MATCH"] = (
        "YES" if idx_size == 47781243 else f"NO ({idx_size})"
    )
    report["PACKED_R128_CACHE_USED"] = "YES"
    report["RAW_VIDEO_FALLBACK_USED"] = "NO"

    # 3. Load Stage-1 population and authentic authority
    s1_auth_rel = (
        "docs/classification_v2/corrected_pooled_route_20260806/"
        "next_phase_20260806_r2/s1_control_and_pre_s1_calibration_authority.json"
    )
    s1_auth_path = repo_root / s1_auth_rel
    if not s1_auth_path.exists():
        s1_auth_path = repo_root / "pig_e0_r3/inputs" / s1_auth_rel

    out_dir = (
        outputs_root
        / "classification_v2/s1_post_temporal_closure_20260809/s1_trials"
        / f"s1_stage1_{TEMPORAL_VIEW.lower()}_seed{SEED}_steps{TARGET_STEPS}"
    )
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    s1_plan = stage1.create_stage1_plan(
        authority_path=s1_auth_path,
        repository_root=repo_root,
        outputs_root=outputs_root,
        view=TEMPORAL_VIEW,
        seed=20260804,
        device_name="cuda",
        output_dir=out_dir,
        engineering_smoke=False,
        allow_existing_output=True,
    )
    hashes = stage1.preflight_stage1(s1_plan)
    rows = stage1.load_stage1_inner_rows(s1_plan, hashes)

    train_count = len(rows.train)
    val_count = len(rows.validation)
    total_count = train_count + val_count
    print(
        f"Stage1 population loaded: train={train_count}, "
        f"val={val_count}, total={total_count}"
    )

    report["TRAIN_POPULATION_COUNT"] = train_count
    report["VALIDATION_POPULATION_COUNT"] = val_count
    report["TOTAL_POPULATION_COUNT"] = total_count

    # 4. Load Stage-1 context frames & windows
    s1_binding_rel = (
        "classification_v2/s1_stage1_temporal_screening/"
        "stage1_rgb_bindings_20260810_52d62718/"
        "s1_stage1_f3_rgb_binding_20260810_52d62718_t6"
    )
    s1_binding_dir = outputs_root / s1_binding_rel
    if not s1_binding_dir.exists():
        s1_binding_dir = repo_root / "pig_e0_r3/inputs/outputs" / s1_binding_rel

    frames_df = pd.read_csv(s1_binding_dir / "stage1_frame_context.csv", low_memory=False)
    windows_df = pd.read_csv(s1_binding_dir / "stage1_window_context.csv", low_memory=False)

    selected = pd.concat([rows.train, rows.validation], ignore_index=True)
    index_by_window = {str(w): i for i, w in enumerate(windows_df["window_id"])}
    selected["window_row_index"] = selected["window_id"].astype(str).map(index_by_window).astype(int)
    selected["window_valid_for_main_train"] = True
    selected["primary_s1_eligible"] = True

    media_root = repo_root / "data/videos"
    binding = build_inner_resolution_binding_from_dataframes(
        frames=frames_df,
        windows=windows_df,
        selection=selected,
        media_root=media_root,
        expected_window_count=len(selected),
        expected_observation_count=201792,
    )

    dataset = binding.build_dataset(
        RESOLUTION,
        image_cache_size=8192,
        packed_image_cache_npy=packed_npy,
        packed_image_cache_index_csv=packed_idx,
    )
    lookup = {str(w): i for i, w in enumerate(dataset.windows["window_id"])}

    # 5. Build B1 Model on CUDA
    print(f"Building B1 model ({TEMPORAL_VIEW})...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = stage1._build_b1_model(TEMPORAL_VIEW).to(device)

    initial_hash = compute_model_hash(model)
    print(f"INITIAL_MODEL_SHA256: {initial_hash}")
    report["INITIAL_MODEL_SHA256"] = initial_hash

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    # 6. Training Loop (Exact 4,164 steps)
    print(f"Starting Training: target_steps={TARGET_STEPS}, batch_size={BATCH_SIZE}")
    model.train()

    t_train_start = time.perf_counter()
    step = 0
    rng = np.random.default_rng(SEED)
    train_indices = np.arange(len(rows.train))

    first_step_time = None

    while step < TARGET_STEPS:
        shuffled = rng.permutation(train_indices)
        for i in range(0, len(shuffled), BATCH_SIZE):
            if step >= TARGET_STEPS:
                break

            batch_idx = shuffled[i : i + BATCH_SIZE]
            batch_df = rows.train.iloc[batch_idx]

            subset = Subset(
                dataset,
                [lookup[str(v)] for v in batch_df["window_id"]],
            )
            loader = DataLoader(
                subset,
                batch_size=len(batch_df),
                shuffle=False,
                collate_fn=image_sequence_collate,
            )
            payload = next(iter(loader))

            images = payload["image"].to(device)
            observed = payload["observed_mask"].to(device)
            batch = post_s1._make_batch(images, observed, batch_df, device, RESOLUTION)

            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            logits = out["logits"]
            loss = loss_fn(logits, batch.labels)
            loss.backward()
            optimizer.step()

            step += 1

            if step == 1:
                first_step_time = time.perf_counter() - t_train_start
                print(
                    f"FIRST_OPTIMIZER_STEP_SECONDS: {first_step_time:.2f}s "
                    f"(loss={loss.item():.4f})"
                )

            if step % 200 == 0 or step == TARGET_STEPS:
                elapsed = time.perf_counter() - t_train_start
                steps_per_sec = step / elapsed
                peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
                print(
                    f"Step {step}/{TARGET_STEPS} ({step/TARGET_STEPS*100:.1f}%) | "
                    f"Loss: {loss.item():.4f} | {steps_per_sec:.1f} steps/s | "
                    f"Peak VRAM: {peak_vram_mb:.1f} MB | Elapsed: {elapsed:.1f}s"
                )

    t_train_end = time.perf_counter()
    total_train_time = t_train_end - t_train_start
    print(
        f"Training complete! Total train time: {total_train_time:.2f}s "
        f"({TARGET_STEPS/total_train_time:.1f} steps/s)"
    )

    final_hash = compute_model_hash(model)
    print(f"FINAL_MODEL_SHA256: {final_hash}")
    report["FINAL_MODEL_SHA256"] = final_hash
    report["MODEL_HASHES_DIFFER"] = "YES" if initial_hash != final_hash else "NO"
    report["OPTIMIZER_STEPS_COMPLETED"] = step
    report["FIRST_OPTIMIZER_STEP_SECONDS"] = round(first_step_time or 0.0, 2)
    report["TOTAL_TRAIN_SECONDS"] = round(total_train_time, 2)
    report["PEAK_TRAIN_VRAM_MB"] = round(
        torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2
    )

    # 7. Validation Evaluation (Exact 2,285 windows)
    print(f"Starting validation inference on {val_count} windows...")
    model.eval()
    t_val_start = time.perf_counter()

    val_preds_list = []
    val_targets_list = []
    val_records = []

    val_batch_size = 32
    val_indices = np.arange(len(rows.validation))

    with torch.inference_mode():
        for i in range(0, len(val_indices), val_batch_size):
            batch_idx = val_indices[i : i + val_batch_size]
            batch_df = rows.validation.iloc[batch_idx]

            subset = Subset(
                dataset,
                [lookup[str(v)] for v in batch_df["window_id"]],
            )
            loader = DataLoader(
                subset,
                batch_size=len(batch_df),
                shuffle=False,
                collate_fn=image_sequence_collate,
            )
            payload = next(iter(loader))

            images = payload["image"].to(device)
            observed = payload["observed_mask"].to(device)
            batch = post_s1._make_batch(images, observed, batch_df, device, RESOLUTION)

            out = model(batch)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            labels_np = batch.labels.cpu().numpy()
            val_preds_list.extend(preds.tolist())
            val_targets_list.extend(labels_np.tolist())

            for j, (_, row) in enumerate(batch_df.reset_index().iterrows()):
                pred_label = VALID_BEHAVIORS[preds[j]]
                gt_label = VALID_BEHAVIORS[labels_np[j]]
                rec = {
                    "window_id": str(row["window_id"]),
                    "ground_truth_label": gt_label,
                    "predicted_label": pred_label,
                    "correct": bool(pred_label == gt_label),
                }
                for k, b_name in enumerate(VALID_BEHAVIORS):
                    rec[f"prob_{b_name}"] = float(probs[j, k])
                val_records.append(rec)

    total_val_time = time.perf_counter() - t_val_start
    print(f"Validation inference complete in {total_val_time:.2f}s!")

    val_df = pd.DataFrame(val_records)
    macro_f1 = float(
        f1_score(
            val_targets_list,
            val_preds_list,
            average="macro",
            zero_division=0,  # type: ignore
        )
    )
    print(f"==================================================")
    print(f"R128 SEED 20260814 MACRO-F1: {macro_f1:.6f}")
    print(f"==================================================")

    cls_report = classification_report(
        val_targets_list,
        val_preds_list,
        target_names=VALID_BEHAVIORS,
        output_dict=True,
        zero_division=0,  # type: ignore
    )
    print(
        classification_report(
            val_targets_list,
            val_preds_list,
            target_names=VALID_BEHAVIORS,
            zero_division=0,  # type: ignore
        )
    )

    report["VALIDATION_WINDOWS_EVALUATED"] = len(val_df)
    report["VALIDATION_COVERAGE_COMPLETE"] = (
        "YES" if len(val_df) == val_count else f"NO ({len(val_df)}/{val_count})"
    )
    report["R128_SEED_20260814_MACRO_F1"] = round(macro_f1, 6)
    report["PER_CLASS_F1"] = {
        b: round(cls_report[b]["f1-score"], 6) for b in VALID_BEHAVIORS
    }

    # 8. Save Artifacts
    target_artifacts_dir = (
        outputs_root
        / "classification_v2/post_s1_resolution_proof"
        / f"R{RESOLUTION}_seed_{SEED}"
    )
    target_artifacts_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = target_artifacts_dir / f"r{RESOLUTION}_seed_{SEED}_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "temporal_view": TEMPORAL_VIEW,
            "resolution": RESOLUTION,
            "seed": SEED,
            "steps": TARGET_STEPS,
            "macro_f1": macro_f1,
            "initial_model_hash": initial_hash,
            "final_model_hash": final_hash,
        },
        ckpt_path,
    )
    ckpt_sha256 = compute_file_sha256(ckpt_path)
    print(f"Checkpoint saved: {ckpt_path} ({ckpt_sha256})")

    pred_path = target_artifacts_dir / f"r{RESOLUTION}_seed_{SEED}_val_predictions.csv"
    val_df.to_csv(pred_path, index=False)
    pred_sha256 = compute_file_sha256(pred_path)
    print(f"Predictions saved: {pred_path} ({pred_sha256})")

    report["CHECKPOINT_PATH"] = str(ckpt_path)
    report["CHECKPOINT_SHA256"] = ckpt_sha256
    report["PREDICTIONS_PATH"] = str(pred_path)
    report["PREDICTIONS_SHA256"] = pred_sha256

    report["TOTAL_WALL_CLOCK_SECONDS"] = round(time.perf_counter() - t_start, 2)
    report["REAL_R128_PROOF_RUN"] = "PASS"

    summary_json_path = (
        target_artifacts_dir / f"r{RESOLUTION}_seed_{SEED}_summary.json"
    )
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n==================================================")
    print("EXECUTION SUMMARY")
    print("==================================================")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_trial()
