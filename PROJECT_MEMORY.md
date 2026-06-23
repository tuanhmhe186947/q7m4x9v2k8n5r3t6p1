# Project Memory: Pig Behavior Project

This document serves as the long-term, medium-term, and short-term memory registry for the Pig Behavior Project, outlining the system architecture, code contracts, and active targets.

## Long-Term Memory (Core Architecture & Tech Stack)

### Project Overview
The Pig Behavior Project is an AI-powered system designed to detect, track, and classify the behaviors of group-housed pigs. It uses a multi-stage computer vision pipeline:
1. **Detection & Tracking:** A YOLO-based detector/tracker (implemented via `ultralytics`) produces 2D bounding boxes and unique track IDs for pigs from video.
2. **RGB-D/3D Occlusion Inference (Auxiliary):** When depth data is available, 2D bounding boxes are projected using depth maps to infer occlusions and filter overlapping bounding boxes in 3D space. This acts as an auxiliary module to resolve heavy occlusions, while the core tracking pipeline remains fully functional on standalone color (RGB) video feeds.
3. **Behavior Classification:** For each tracked pig, a temporal sequence of cropped frames (sequence length = 6 frames, using stride/offsets) is combined with a set of 10 tabular normalized sequence features (positions, sizes, speed, proximity to other pigs, and region ROI intersections). The classifier predicts behaviors such as `lying` (stay/hold), `move`, and interactions.
4. **Dashboard:** A FastAPI service streams video feed, processes detection/tracking, classifies behaviors in real time, and runs a visualization dashboard.

### Core Stack & Hardware Support
* **Language:** Python `>=3.10, <3.12`
* **Deep Learning & CV:** PyTorch (`torch`, `torchvision`), TensorFlow (for model export/TFLite), `ultralytics` (YOLO)
* **Image/Video Processing:** OpenCV (`opencv-python-headless`), Pillow
* **Tabular Data & Analytics:** numpy, pandas, scikit-learn, scipy
* **API & Dashboard:** FastAPI, Uvicorn, python-multipart
* **Testing & Quality:** pytest, ruff, MyPy (strict type & shape check for 2D/3D projection matrices)
* **Hardware Targets:**
  * **Primary:** Nvidia CUDA GPU for both training and inference (especially for YOLO detection and 3D spatial matrix calculations).
  * **Fallback:** CPU execution fallback at acceptable FPS for deployment in resource-constrained farm environments.

---

## Medium-Term Memory (Data Contracts & Milestones)

### Key Files & Locations
* **Config:** [src/pig_behavior/config.py](file:///c:/Users/ironh/Downloads/PIG_Behavior_Project/src/pig_behavior/config.py)
* **Models:**
  * YOLO Detector: `models/detector/pig_detector_yolo.pt`
  * Behavior Classifier: `models/behavior/pig_behavior_sequence.pt`
* **Processed Datasets:** `data/processed/classification/<timestamp>/behavior_with_feats_rectROI.csv`
* **Annotations:**
  * Feeder/Drinker/Toy ROIs: `data/annotations/roi/ROI_annotations.coco.json`
  * Background & Mask: `data/annotations/scene/background.png` & `mask.png`
  * CVAT XML Ground Truth: `data/annotations/tracking/`

### Tabular Sequence Features
The behavior model accepts a 10-dimensional feature vector per step in the sequence:
* `cx_n`, `cy_n` (Normalized center coordinates of the bounding box)
* `bw_n`, `bh_n` (Normalized bounding box width and height)
* `speed_feat` (Estimated speed of the pig)
* `min_dist_other` (Distance to the closest neighboring pig)
* `num_close_other` (Count of nearby pigs)
* `in_feeder`, `in_drinker`, `in_toy` (Binary indicators of presence in specific ROIs)

### High Priority Behavior Classes
Focus is on small pens (approx. 8 pigs) and narrow-space interactions:
1. `lying` (stay/hold)
2. `move`
3. Overlapping interactions / lying on top of each other (nằm đè lên nhau) - key source of tracker noise.

---

## Short-Term Memory (Active Checklist)

- [ ] Align behavior classifier training pipeline parameters.
- [ ] Address benchmark video evaluation fixes (evaluating all ground-truth/tracked videos in tracking evaluation).
- [ ] Implement MyPy configuration and type-check critical 2D/3D transform components.
- [ ] Set up GitHub Actions CI for Ruff linting and Pytest unit tests.
