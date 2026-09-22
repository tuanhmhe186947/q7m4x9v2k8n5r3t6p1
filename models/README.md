# Models and Checkpoint Catalog

Model weights (`*.pt`, `*.pth`, `*.ckpt`, `*.onnx`) are not distributed in this Git repository.

## Expected Model Categories

1. **Pig Detector & Tracker**:
   - Expected path: `models/detector/pig_detector_yolov8x_30fps.pt` (or default `models/pig_detector_yolov8.pt`): YOLOv8 detector fine-tuned on commercial pig pen video frames.

2. **Behavior Recognition Classifiers (5-Fold Cross Validation)**:
   - **Multimodal Spatio-Temporal Baseline**: Spatial-temporal visual backbone with 10-class behavior head.
   - **Joint-Representation Baseline**: Multimodal partner-behavior conditioned model.
   - **Class-Aware Spatial-Gated Model**: Class-aware gating with localized spatial attention.
   - **Spatial-Gated Ensemble**: Evaluated ensemble combining spatial-gated and joint representations.
   - **Architectural Factor Ablations**: Scaffold Control (Config N), Local-only (Config L), and Gate-only (Config G).

## Checkpoint Placement and External Distribution

For external distribution and paper reproduction:
- Pretrained weights are preserved in research storage and can be attached to tagged releases (e.g., GitHub Releases) or persistent archives.
- When downloaded, place checkpoints into their respective paths:
  - Detector weights: `models/detector/`
  - Behavior sequence classifiers: `models/behavior/`

