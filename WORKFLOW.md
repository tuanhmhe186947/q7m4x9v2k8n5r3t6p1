# Project Workflow: Data, Training, and Evaluation

This document maps out the processing pipelines, training protocols, and validation workflows for the Pig Behavior Project.

---

## 1. Data Preparation & Annotation Workflow

### 1.1 Region-of-Interest (ROI) Definition
The physical scene includes boundaries for feeders, drinkers, and toys. These zones are defined in:
* COCO annotations file: `data/annotations/roi/ROI_annotations.coco.json`
* Background image & occupancy mask: `data/annotations/scene/background.png` & `mask.png`

### 1.2 Annotation Cleaning & Dataset Construction
When raw labels are exported from CVAT (video XML format), they are processed to build tabular classification datasets:
* Run the dataset builder script:
  ```bash
  pig-build-classification-data
  ```
* This parses CVAT XML files, filters frames, and merges them with spatial ROI zones to produce:
  * Merged annotations: `data/processed/classification/<timestamp>/behavior_clean_merged.csv`
  * Features dataset: `data/processed/classification/<timestamp>/behavior_with_feats_rectROI.csv`

---

## 2. Detection, Tracking & Occlusion Workflow

### 2.1 Generating Tracking Predictions
To output bounding boxes and track IDs from raw videos:
```bash
pig-track-for-annotation --video data/videos/Pigs281119_000085_30fps.mp4
```
* **Hardware fallbacks:** By default, tracking inference runs on Nvidia CUDA GPU. If CUDA is unavailable, the pipeline falls back to CPU execution at an acceptable FPS target for on-farm deployment.

### 2.2 Depth-Based Occlusion Inference (RGB-D)
For multi-pig occlusion resolution using 3D depth-sensing inputs:
* Utilizes depth maps/videos (`depth_video_path`) and calibration numpy files:
  * Scale: `depth_scale.npy` (depth scaling factor)
  * Camera Intrinsics: `inverse_intrinsic.npy` (intrinsic matrix)
  * Extrinsic Rotation: `rot.npy` (rotation matrix)
* These calibration files are extracted directly from the internal and external parameters of cameras (e.g., Intel RealSense) during dataset capture.
* Overlay logic checks IoU overlap. If overlapping, the detection with the larger depth (further away) is flagged as `occluded = True`.

### 2.3 Tracking Evaluation
Evaluate the tracker outputs against ground-truth CVAT XML files:
```bash
pig-tracking-eval --run-missing-tracker
```
This evaluates trackers based on Multi-Object Tracking (MOT) metrics.

---

## 3. Behavior Classifier Training Workflow

### 3.1 Model Architecture & Configuration
The behavior sequence classifier is built to receive:
* **Video Inputs:** Bounding box crops of length 6 frames, sampled at a stride of 3 frames (timestamps: `t-9, t-6, t-3, t, t+3, t+6`).
* **Tabular Features:** A 10D temporal feature vector per step in the sequence.

### 3.2 Baseline Training Configuration
* **Optimizer:** AdamW or SGD
* **Learning Rate (LR):** Initial LR $1\times 10^{-3}$ to $1\times 10^{-4}$ with a Cosine Annealing scheduler.
* **Batch Size:** 16 or 32 (adjusted based on available VRAM).
* **Epochs:** 100 - 200 epochs with Early Stopping.
* **Command:**
  ```bash
  pig-behavior --mode train
  ```
* **Training Smoke Test:**
  ```bash
  pig-behavior --mode train --dry-run
  ```

---

## 4. Model Export & Inference

### 4.1 ONNX & TFLite Export
To compile the trained PyTorch/TensorFlow behavior model into lightweight inference formats:
```bash
pig-behavior --mode export
```

### 4.2 Local Inference (Command Line)
```bash
pig-behavior --mode infer \
  --backend pt \
  --pt-model models/behavior/pig_behavior_sequence.pt \
  --image data/raw/images_clean/example.jpg
```

### 4.3 Running the FastAPI & Dashboard App
Start the dashboard server:
```bash
# Set environment variables
set PIG_BEHAVIOR_MODEL_BACKEND=pt
set PIG_BEHAVIOR_PT_MODEL_PATH=models\behavior\pig_behavior_sequence.pt
set PIG_BEHAVIOR_DETECT_MODEL_PATH=models\detector\pig_detector_yolo.pt
set PIG_BEHAVIOR_VIDEO_PATH=data\videos\pigs101219_full.mp4
set PIG_BEHAVIOR_BEHAVIOR_STRIDE_FRAMES=3

# Start the dashboard API
pig-behavior-api
```
Access the dashboard at `http://127.0.0.1:8000/dashboard` to view real-time tracking, occlusion mitigation, and classified behavior stream aggregation.

---

## 5. Quality Standards & CI/CD Pipeline

### 5.1 Static Analysis & Quality Checks
To ensure code robustness, the project enforces:
* **Linting & Formatting:** Done via `ruff`:
  ```bash
  ruff check src main.py tools tests
  ```
* **Type-Checking:** Checked via `mypy` to validate type annotations and verify NumPy array shapes/dimensions in 2D-to-3D projection functions:
  ```bash
  mypy src/
  ```
* **Testing:** Executed via `pytest`:
  ```bash
  pytest -q
  ```

### 5.2 CI/CD Integration
GitHub Actions workflow `.github/workflows/ci.yml` is used to:
1. Trigger on every `push` and `pull_request` to target branches.
2. Setup Python environment and install dependencies.
3. Run `ruff` lint check.
4. Run `mypy` type checks.
5. Execute unit tests using `pytest`.
