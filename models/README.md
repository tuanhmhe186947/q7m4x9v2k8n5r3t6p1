# Models and Checkpoint Catalog

Model weights (`*.pt`, `*.pth`, `*.ckpt`, `*.onnx`) are binary research artifacts and
are excluded from the Git tree to keep repository clone size lightweight (< 50 MB).

## Expected Model Categories

1. **Pig Detector & Tracker**:
   - `models/detector/pig_detector_yolov8x_30fps.pt`: YOLOv8x fine-tuned on commercial pig pen video frames.
2. **Behavior Recognition Classifiers (5-Fold Cross Validation)**:
   - **Model A Baseline** (`final_high_ceiling_v1`): Spatial-temporal visual backbone with 10-class behavior head.
   - **Model B Joint Representation** (`joint_representation_v1`): Multimodal partner-behavior conditioned model.
   - **Model J Architecture** (`final_model_j_replay_v1`): Class-aware gating with localized spatial attention.
   - **Model J_PBNEW** (`final_model_j_pbnew_ejteacher_v1`): State-of-the-art single model distilled with E_J teacher.
   - **2x2 Factor Ablations**: Scaffold Config N, Local-only Config L, and Gate-only Config G.

## Checkpoint Traceability and Manifests

All 30 physical training checkpoints are indexed with file size and provenance in:
`docs/paper/freeze_corrected_20260917/checkpoint_manifest.csv`

## External Distribution

For external distribution and public paper reproduction:
- Pretrained weights are recommended to be hosted via **GitHub Releases** (attached to tagged releases) or a persistent archive (e.g. Zenodo, Hugging Face Hub).
- Checkpoints should be placed into their respective directory structures under `outputs/classification_v2/` or `models/` as referenced in the configuration files.
