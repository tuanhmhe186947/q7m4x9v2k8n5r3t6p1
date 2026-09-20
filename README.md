# Identity-Preserving Tracking and Multimodal Pig Behavior Recognition

> An end-to-end computer-vision pipeline for preserving individual identity and recognizing behavior over time in group-housed pigs.

[![CI](https://github.com/tuanhmhe186947/q7m4x9v2k8n5r3t6p1/actions/workflows/ci.yml/badge.svg)](https://github.com/tuanhmhe186947/q7m4x9v2k8n5r3t6p1/actions/workflows/ci.yml)
[![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![System Pipeline](docs/assets/system_pipeline.svg)

---

## Highlights

| Component | Baseline | Causal Online | Retrospective Offline | Primary Target |
| :--- | :--- | :--- | :--- | :--- |
| **Tracking** (12 held-out videos) | **Raw ByteTrack Baseline**<br>HOTA: 89.41% \| IDF1: 94.23% \| IDSW: 64 | **Online RealTime-Fast**<br>HOTA: 90.33% \| IDF1: 95.03% \| IDSW: 39 | **Offline Hybrid-ByteTrack**<br>HOTA: **93.55%** \| IDF1: **98.41%** \| IDSW: **8** | -87.5% identity switches |
| **Behavior** (5-fold CV) | **Multimodal Baseline**<br>Macro-F1: 0.6778 ± 0.0268 | **Class-Aware Spatial-Gated**<br>Macro-F1: 0.6846 ± 0.0292 | **Final Spatial-Gated Ensemble**<br>Macro-F1: **0.6939 ± 0.0377** | Sample SD across VG1–VG5 |
| **Profile Distortion** (DEV12) | **Raw ByteTrack Baseline**<br>TV = 0.0953 | **Online RealTime-Fast**<br>TV = 0.0309 | **Offline Hybrid-ByteTrack**<br>TV = **0.0001** | Near-zero profile distortion |

---

## System Overview

Continuous precision livestock monitoring requires observing animals over extended periods without confusing individual subjects. This repository implements an integrated two-stage framework:
1. **Identity-Preserving Multi-Object Tracking**: Detects individuals with YOLOv8 and preserves identities across dense interactions using causal online association (`Online RealTime-Fast`) or retrospective two-pass smoothing (`Offline Hybrid-ByteTrack`).
2. **Multimodal Spatio-Temporal Behavior Recognition**: Classifies individual time units into 10 behavior classes integrating motion dynamics, local spatial attention, and social context.
3. **Downstream Longitudinal Profiling**: Quantifies how identity errors propagate into individual behavior profiles and demonstrates that minimizing identity switches preserves longitudinal activity budgets.

---

## Tracking Results

Tracking performance was evaluated on a held-out confirmatory cohort of 12 videos (21,600 frames, 96 individual pig trajectories) under `TRACKING_EVALUATOR_STANDARD_V2`. System-level comparisons demonstrate a substantial reduction in identity switches without compromising detection accuracy.

![Tracking Summary](docs/assets/tracking_summary.svg)

- **Raw ByteTrack Baseline**: Unmodified reference baseline (HOTA: `89.41%`, IDF1: `94.23%`, 64 ID switches).
- **Online RealTime-Fast**: Causal online tracker with frame skipping for real-time edge deployment (HOTA: `90.33%`, IDF1: `95.03%`, 39 ID switches, -39.1%).
- **Offline Hybrid-ByteTrack**: Retrospective two-pass smoothing association for archival analysis (HOTA: `93.55%`, IDF1: `98.41%`, 8 ID switches, -87.5%).
- Full evaluation metrics and 95% bootstrap confidence intervals are available in [docs/paper/tables/tracking_confirmatory.md](docs/paper/tables/tracking_confirmatory.md).

---

## Behavior Recognition

Behavior recognition evaluates 10 mutually exclusive behavioral categories using video-isolated 5-fold cross-validation (VG1–VG5) to prevent video leakage across train and validation splits. No separate distinct outer cohort was materialized.

![Behavior CV Summary](docs/assets/behavior_cv_summary.svg)

- **Multimodal Spatio-Temporal Baseline**: High-ceiling baseline with motion dynamics and visual features (Macro-F1: `0.6778 ± 0.0268`).
- **Joint-Representation Baseline**: Relational joint-feature representation across individuals (`0.6708 ± 0.0349`).
- **Class-Aware Spatial-Gated Model**: Pre-GAP spatial attention with class-aware gating (`0.6846 ± 0.0292`).
- **Baseline Fixed Ensemble**: Equal-weighted logit ensemble of baselines (`0.6926 ± 0.0402`).
- **Final Spatial-Gated Ensemble**: Authoritative production ensemble (`0.6939 ± 0.0377`).
- Detailed fold-by-fold results are documented in [docs/paper/tables/behavior_cv.md](docs/paper/tables/behavior_cv.md).

---

## Identity Preservation and Profile Distortion

Tracking errors directly degrade longitudinal behavioral monitoring: when two animals swap identities, their accumulated activity budgets cross-contaminate. Using the 12-video development behavior-overlap subset (DEV12), we quantify profile distortion using the Total Variation ($L_1$) distance across fixed human annotations.

![Profile Distortion Summary](docs/assets/profile_distortion_summary.svg)

- **Raw ByteTrack Baseline**: Substantial profile distortion ($\text{TV} = 0.0953$).
- **Online RealTime-Fast**: Reduces distortion by 67.6% ($\text{TV} = 0.0309$) in causal online streaming.
- **Offline Hybrid-ByteTrack**: Reduces distortion by 99.9% ($\text{TV} = 0.0001$), producing near-zero profile distortion on the evaluated 12-video development subset.
- Dataset cohort definitions are detailed in [docs/paper/tables/dataset_roles.md](docs/paper/tables/dataset_roles.md).

---

## Repository Structure

```text
.
├── configs/            # Tracking and behavior model configurations
├── data/               # Manifests, splits, and sample video sequences
├── docs/               # Technical guides, paper tables, and reproduction docs
│   ├── assets/         # Vector diagrams, provenance, and benchmark plots
│   ├── paper/tables/   # Authoritative paper result tables
│   ├── api.md          # REST API reference documentation
│   ├── reproduction.md # Scientific claim and experiment reproduction guide
│   └── usage.md        # Command-line interface and profile usage
├── models/             # Detector and behavior classification model weights
├── scripts/            # CLI runners, evaluation tools, and paper checks
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
python scripts/run_tracking_mode.py --mode hybrid_bytetrack --video data/videos/sample.mp4

# Run causal real-time online tracking
python scripts/run_tracking_mode.py --mode realtime_fast --video data/videos/sample.mp4
```

For advanced CLI options and batch tracking, see the [Usage Guide](docs/usage.md).

---

## Reproducing the Paper

To verify quantitative metric tokens against the frozen evidence ledger:

```bash
python scripts/paper/check_claim_numbers.py
```

Expected output: `PASS: 100% of metric tokens in manuscript narratives match the master evidence ledger.`
Complete reproduction instructions are provided in the [Reproduction Guide](docs/reproduction.md).

---

## Data and Model Availability

- **Model Weights**: Detection and spatiotemporal behavior model weights are structured in `models/`.
- **Dataset Manifests**: Cohort splits, unit manifests, and annotation specifications are under `data/` and documented in [docs/paper/tables/dataset_roles.md](docs/paper/tables/dataset_roles.md).
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
