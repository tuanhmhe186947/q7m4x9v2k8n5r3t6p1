# Identity-Preserving Tracking and Multimodal Pig Behavior Recognition

> An end-to-end computer-vision pipeline for preserving individual identity and recognizing behavior over time in group-housed pigs.

[![CI](https://github.com/tuanhmhe186947/q7m4x9v2k8n5r3t6p1/actions/workflows/ci.yml/badge.svg)](https://github.com/tuanhmhe186947/q7m4x9v2k8n5r3t6p1/actions/workflows/ci.yml)
[![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

```text
+-----------------------+     +-------------------------------+     +--------------------------------+     +-------------------------------+
|      Input Video      |     |     Detection & Tracking      |     |      Behavior Recognition      |     |     Longitudinal Analysis     |
| Continuous Farm Feed  | --> | • YOLOv8 Detector             | --> | • Spatio-Temporal Model        | --> | • 10 Behavior Time Budgets    |
| 30 FPS Top-Down View  |     | • Online RealTime-Fast        |     | • Pre-GAP Spatial Gating       |     | • Near-Zero Profile Distortion|
|                       |     | • Offline Hybrid-ByteTrack    |     | • Social Relational Context    |     |   (TV = 0.0001)               |
+-----------------------+     +-------------------------------+     +--------------------------------+     +-------------------------------+
```

---

## Key Findings

| Pipeline Stage | Baseline System | Proposed Primary System | Key Impact |
| :--- | :--- | :--- | :--- |
| **Multi-Object Tracking** | **Raw ByteTrack Baseline**<br>HOTA: 89.41% \| IDF1: 94.23% \| IDSW: 64 | **Offline Hybrid-ByteTrack**<br>HOTA: **93.55%** \| IDF1: **98.41%** \| IDSW: **8** | **-87.5% identity switches** on 12 held-out videos |
| **Behavior Recognition** | **Multimodal Baseline**<br>Macro-F1: 0.6778 ± 0.0268 | **Final Spatial-Gated Ensemble**<br>Macro-F1: **0.6939 ± 0.0377** | Pre-GAP spatial attention + relational ensemble |
| **Profile Preservation** | **Raw ByteTrack Baseline**<br>Total Variation: 0.0953 | **Offline Hybrid-ByteTrack**<br>Total Variation: **0.0001** | **-99.9% distortion reduction**; produced substantially lower profile distortion on the evaluated subset |

---

## System Overview

Continuous precision livestock monitoring requires observing animals over extended periods without confusing individual subjects. This repository implements an integrated two-stage framework:
1. **Identity-Preserving Multi-Object Tracking**: Detects individuals with YOLOv8 and preserves identities across dense interactions using causal online association (`Online RealTime-Fast`) or retrospective two-pass smoothing (`Offline Hybrid-ByteTrack`).
2. **Multimodal Spatio-Temporal Behavior Recognition**: Classifies individual time units into 10 behavior classes integrating motion dynamics, local spatial attention, and social context.
3. **Downstream Longitudinal Profiling**: Quantifies how identity errors propagate into individual behavior profiles and demonstrates that minimizing identity switches produced substantially lower profile distortion on the evaluated subset.

---

## Tracking Confirmatory Results

Evaluated on 12 held-out independent videos (21,600 frames, 96 pig trajectories) under `TRACKING_EVALUATOR_STANDARD_V2`:

| System Variant | Pipeline Mode | HOTA (%) | IDF1 (%) | Identity Switches (Lower is better) | Key Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw ByteTrack Baseline** | Online Reference | `89.41%` [█████████░] | `94.23%` [█████████▌] | `64`  [■■■■■■■■■■■■■■■■] | Unmodified baseline |
| **Online RealTime-Fast** | Causal Streaming | `90.33%` [█████████░] | `95.03%` [█████████▌] | `39`  [■■■■■■■■■■] (-39.1%) | Real-time edge streaming |
| **Offline Hybrid-ByteTrack** | Retrospective | **`93.55%`** [█████████▎] | **`98.41%`** [█████████▊] | **`8`**   [■■] **(-87.5%)** | **highest HOTA/IDF1 among the evaluated systems** |

---

## Behavior Recognition (5-Fold Cross-Validation)

Evaluated across 10 behavioral categories using video-group-disjoint 5-fold cross-validation (VG1–VG5):

| Model Architecture | Description | Macro-F1 (Mean ± SD) | Relative Visual Score | Gain vs. Baseline |
| :--- | :--- | :---: | :--- | :---: |
| **Joint-Representation Baseline** | Relational joint-feature representation | `0.6708 ± 0.0349` | `███████████████▌░░░` | -0.0070 |
| **Multimodal Spatio-Temporal Baseline** | High-ceiling baseline (motion + visual) | `0.6778 ± 0.0268` | `████████████████░░░` | — |
| **Class-Aware Spatial-Gated Model** | Pre-GAP spatial attention with class gating | `0.6846 ± 0.0292` | `████████████████▌░░` | +0.0068 |
| **Baseline Fixed Ensemble** | Equal-weighted logit ensemble of baselines | `0.6926 ± 0.0402` | `█████████████████░░` | +0.0148 |
| **Final Spatial-Gated Ensemble** | Authoritative evaluated ensemble | **`0.6939 ± 0.0377`** | **`█████████████████▍` [Best]** | **+0.0161** |

---

## Downstream Profile Preservation

Tracking identity swaps corrupt longitudinal individual activity budgets. We quantify profile distortion on the 12-video development overlap subset (DEV12) using Total Variation ($L_1$) distance against human ground-truth annotations:

| Tracking Method | Profile Total Variation ($L_1$) | Visual Distortion Level (Lower is better) | Distortion Reduction | Practical Impact |
| :--- | :---: | :--- | :---: | :--- |
| **Raw ByteTrack Baseline** | `0.0953` | `████████████████████████` (0.0953) | Baseline | Severe time-budget distortion from ID swaps |
| **Online RealTime-Fast** | `0.0309` | `████████` (0.0309) | **-67.6%** | Causal online tracking alternative |
| **Offline Hybrid-ByteTrack** | **`0.0001`** | `▏` (**0.0001**) | **-99.9%** | **Near-zero distortion; produced substantially lower profile distortion on the evaluated subset** |

---

## Repository Structure

```text
.
├── configs/            # Tracking and behavior model configurations
├── docs/               # Technical guides, API specifications, and usage docs
│   ├── api.md          # REST API reference documentation
│   ├── reproduction.md # System nomenclature and execution reproduction guide
│   └── usage.md        # Command-line interface and profile usage
├── models/             # Detector and behavior classification model weights
├── scripts/            # CLI runners and evaluation tools
├── src/pig_behavior/   # Core library (tracking, behavior models, API)
└── tests/              # Bounded public unit and contract test suite
```

---

## Quick Start

```bash
git clone https://github.com/tuanhmhe186947/q7m4x9v2k8n5r3t6p1.git
cd q7m4x9v2k8n5r3t6p1
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .

# Run Offline Hybrid-ByteTrack on a sample video
python scripts/run_tracking_mode.py --mode hybrid_bytetrack --video sample.mp4

# Run causal real-time online tracking
python scripts/run_tracking_mode.py --mode realtime_fast --video sample.mp4
```

For advanced CLI options and batch tracking, see the [Usage Guide](docs/usage.md).

---

## Scientific Reproduction

Complete reproduction instructions for tracking, cross-validation, and profile distortion experiments are provided in the [Reproduction Guide](docs/reproduction.md).

---

## Data and Model Availability

- **Model Weights**: Detection and spatiotemporal behavior model weights are structured in `models/`.
- **FastAPI Service**: Interactive inference server is documented in [docs/api.md](docs/api.md).

---

## Citation

```bibtex
@article{pig_behavior_2026,
  title={Identity-Preserving Tracking and Multimodal Pig Behavior Recognition},
  author={Tuan, H. M.},
  year={2026},
  url={https://github.com/tuanhmhe186947/q7m4x9v2k8n5r3t6p1}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
