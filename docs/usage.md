# Pipeline Usage Guide

This guide covers command-line workflows for video tracking, tracking evaluation, configuration profiles, and integrations.

---

## 1. Tracking CLI Entrypoints

### A. Presentation Wrapper (`run_tracking_mode.py`)
Run or compare the three primary tracking profiles using a single command:

```bash
# Run the retrospective offline pipeline (Hybrid-ByteTrack)
python scripts/run_tracking_mode.py --mode hybrid_bytetrack --video data/videos/sample.mp4

# Run the causal online pipeline (RealTime-Fast)
python scripts/run_tracking_mode.py --mode realtime_fast --video data/videos/sample.mp4

# Run the frozen technical baseline (Raw ByteTrack)
python scripts/run_tracking_mode.py --mode bytetrack_raw --video data/videos/sample.mp4

# Compare all three modes sequentially
python scripts/run_tracking_mode.py --task compare --video data/videos/sample.mp4
```

### B. Direct Video Tracking (`track_videos.py`)
Run tracking on one or more videos and save identity annotations in CVAT XML or CSV format:

```bash
python scripts/track_videos.py \
    --mode hybrid_bytetrack \
    --eval-config hybrid_bytetrack_best \
    --video-dir data/videos \
    --output-dir outputs/tracking
```

### C. Confirmatory Tracking Evaluation (`evaluate_tracking.py`)
Evaluate tracker outputs against ground-truth CVAT XML annotations using standard HOTA, IDF1, and IDSW metrics:

```bash
python scripts/evaluate_tracking.py \
    --mode hybrid_bytetrack \
    --eval-config hybrid_bytetrack_best \
    --gt-dir data/ground_truth \
    --output-dir outputs/eval
```

---

## 2. Tracking Profiles Reference

| Profile Name | Engine (`--mode`) | Config (`--eval-config`) | Processing Mode | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **Hybrid-ByteTrack** | `hybrid_bytetrack` | `hybrid_bytetrack_best` | Retrospective Offline | Two-pass association, causal + backward smoothing, identity swap recovery (8 IDSW) |
| **RealTime-Fast** | `realtime` | `realtime_fast` | Causal Online | Frame skipping, adaptive velocity propagation, bounded latency (39 IDSW) |
| **Raw ByteTrack** | `bytetrack_raw` | `bytetrack_raw` | Baseline | Pure ByteTrack without domain-specific association guards (64 IDSW) |

---

## 3. Tracking Optimization & Parameter Search

Use `scripts/optimize_tracking_metrics.py` to evaluate parameter sets over validation splits:

```bash
python scripts/optimize_tracking_metrics.py \
    --mode hybrid_bytetrack \
    --candidates configs/tracking/optimization_presets.yaml \
    --gt-dir data/ground_truth \
    --output-dir outputs/optimization
```

---

## 4. Integrations & External Workflows

### Roboflow Integration
For remote cloud inference or model exports via the Roboflow Inference API:

1. Export your API key:
   ```bash
   export ROBOFLOW_API_KEY="your_api_key_here"
   ```
2. Execute the workflow client:
   ```bash
   python scripts/integrations/run_roboflow_workflow.py \
       --workspace "pig-behavior" \
       --workflow-id "pig-detection-tracking" \
       --input "data/videos/sample.mp4"
   ```

### Environment Variables

| Variable | Required | Description |
| :--- | :---: | :--- |
| `ROBOFLOW_API_KEY` | Optional | API token for Roboflow hosted inference workflows |
| `PIG_DETECTOR_PATH` | Optional | Path to local YOLOv8 weights (defaults to `models/pig_detector_yolov8.pt`) |
| `CUDA_VISIBLE_DEVICES` | Optional | GPU device indexing for PyTorch runtime |
