# CUDA benchmark readiness — 2026-07-27

**No package was installed, upgraded, or replaced.** A CUDA-capable environment
already exists on this host; the previously reported blocker was an
environment-*selection* problem, not a missing installation.

## 1. Hardware and driver

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4096 MiB |
| Compute capability | 8.6 |
| Driver | 610.62 |

## 2. Environment inventory

| Environment | torch | CUDA build | `cuda.is_available()` |
|---|---|---|---|
| `PIG_Behavior_Project/.venv` (project default) | `2.12.1+cpu` | none | **False** |
| `conda: pig_project` | `2.5.1+cu121` | `12.1` | **True** |
| `conda: headroom_env` | `2.12.1+cpu` | none | False |
| `conda: roboflow` | not installed | — | — |
| `conda: VinAI` | not installed | — | — |

`pig_project` is the CUDA-capable environment. Creating a new
`pig_tracking_cuda` environment is **not required** and was not done.

Python differs between the two (`3.11.9` in `.venv` vs `3.11.15` in
`pig_project`), both inside the project's declared `>=3.10,<3.12` constraint.

## 3. Validation checklist — all PASS

Executed with `conda: pig_project`, bounded to 12 decoded frames of
`Pigs291119_000302_30fps.mp4`, warm-up excluded, `torch.cuda.synchronize()`
around each timed call, `save=False`.

| Check | Result |
|---|---|
| `torch.cuda.is_available() == True` | **PASS** |
| Expected GPU name visible | **PASS** — `NVIDIA GeForce RTX 3050 Laptop GPU` |
| CUDA tensor operation succeeds | **PASS** — 512×512 matmul, finite result |
| Ultralytics detector loads | **PASS** — ultralytics `8.4.76`, opencv `5.0.0` |
| Bounded inference smoke | **PASS** — n=10 timed frames |
| Output video generation disabled | **PASS** — `save=False`; no `runs/` directory created |
| MP4 created by this task | **PASS** — `0` |

## 4. Measured throughput

| Metric | Value |
|---|---|
| GPU detector median | `0.0700 s/frame` |
| CPU detector median (previously measured) | `0.9650 s/frame` |
| Speedup | **13.8×** |
| Peak CUDA memory | `160.5 MB` of 4096 MiB |
| Estimated detector-only, full-13 at `detect_every_n=2` | **≈13.7 minutes** |

The CPU estimate for the same work was 3.14 hours. Full-13 reproduction moves
from infeasible to routine.

## 5. MP4 accounting — honest statement

A recursive scan of `outputs/` finds **3** MP4 files:

```
outputs/pred/best3-roboflow/hybrid_bytetrack/smooth/Pigs281119_000085_30fps/tracked_pigs_with_ids.mp4
outputs/pred/best3-roboflow/hybrid_bytetrack/smooth/Pigs291119_000263_30fps/tracked_pigs_with_ids.mp4
outputs/pred/best3-roboflow/hybrid_bytetrack/smooth/Pigs291119_000302_30fps/tracked_pigs_with_ids.mp4
```

All three are dated **2026-06-27**, a month old, and live under a legacy
`best3-roboflow` hybrid prediction root — not a tracking run root created by
this work. Files created in the last 30 minutes: **0**.

So `MP4_CREATED_BY_THIS_TASK=0`, but a repository-wide
`recursive_mp4_count=0` claim would be **false**. Any future run must assert the
zero-MP4 gate against its own run root, not against `outputs/` as a whole.

## 6. Remaining gap before reproduction is authorized

CUDA readiness is proven, but two items block a *comparable* RF_ACC23
reproduction:

1. **Host runtime is non-repeatable.** The recovered artifacts
   (`TRACKING_RF_ACC23_FULL13_RUNTIME_DECISION_20260723.json`) attribute the
   full-13 runtime failure to host power/clock drift: GPU clocks sagging to
   765 MHz against a healthy 1275–1500 MHz, power draw 15–17 W against 21–23 W,
   and the Windows AC power overlay set to *Best power efficiency*
   (`961cc777-2547-4f9d-8174-7d86181b8a7a`) where *Best performance*
   (`ded574b5-45a0-4f42-8737-46345c09c238`) is required. That overlay must be
   corrected and verified **before** any runtime measurement, or the same
   non-attributable result will recur.

2. **Semantic equivalence between lineages is unproven.** The recovered metrics
   were produced at `b0d9009` in the `PIG_task_tracking` lineage, whose tracking
   tree is `752d55d3…`. `main` carries `8ba3d50d…`. A quality reproduction on
   `main` is only comparable once that difference is characterized.

Note also that `pig_project` carries torch `2.5.1` and ultralytics `8.4.76`,
while the project default carries torch `2.12.1` and ultralytics `8.4.72`. A
reproduction run must record which environment produced it; the two are not
interchangeable for numeric comparison.

## 7. If a dedicated environment is later wanted

Not required, and **not executed**. For review only, a combination consistent
with driver 610.62 and compute capability 8.6 would be:

```
conda create -n pig_tracking_cuda python=3.11
conda activate pig_tracking_cuda
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics==8.4.76 opencv-python-headless scipy tqdm
```

This mirrors the already-validated `pig_project` combination rather than
guessing a newer one. Do not run it without explicit authorization; the
existing environment already satisfies every checklist item.
