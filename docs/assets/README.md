# Visual Assets & Figure Provenance

This directory contains publication-quality vector figures used in the repository documentation and research manuscript summaries. All figures are deterministically generated from frozen evaluation tables and master evidence ledgers.

---

## 1. Generation Script

All vector figures can be regenerated deterministically via:

```bash
python scripts/visualization/generate_readme_figures.py
```

Dependencies: `matplotlib >= 3.7.0`

---

## 2. Asset Catalog & Data Provenance

| Asset File | Display Title | Source Authority | Dataset Role | Key Metrics |
| :--- | :--- | :--- | :--- | :--- |
| `system_pipeline.svg` | End-to-End System Architecture | Architectural specification | Metric-free conceptual overview | Detection, Causal/Retrospective Tracking, Spatio-Temporal Representation, Longitudinal Profiling |
| `tracking_summary.svg` | Tracking Confirmatory Summary | `docs/paper/final_tracking_confirmatory_table.csv` | Confirmatory 12-video held-out cohort (21,600 frames, 96 pig trajectories) | HOTA (%), IDF1 (%), ID Switches (IDSW) |
| `behavior_cv_summary.svg` | 5-Fold Behavior Recognition CV | `docs/paper/final_behavior_cv_table.csv` | Group-aware 5-fold cross-validation (VG1–VG5, zero video leakage) | Macro-F1 (mean ± sample SD) across 10 behavior classes |
| `profile_distortion_summary.svg` | Longitudinal Profile Distortion | `docs/paper/master_evidence_ledger.csv` | DEV12 development behavior-overlap subset (12 videos) | A1 Total Variation ($L_1$) distance vs. ground truth profiles |

---

## 3. Detailed Specifications

### `system_pipeline.svg`
- **Purpose**: Hero diagram illustrating the two-stage computer vision pipeline: YOLOv8 individual detection, identity-preserving tracking (online causal vs. offline retrospective), multimodal behavior modeling (motion dynamics, spatial gating, relational context), and longitudinal time-budget profile synthesis.
- **Design**: Clean vector card layout with subtle directional flow and color coding. Metric-free to ensure focus on architectural hierarchy.

### `tracking_summary.svg`
- **Purpose**: System-level comparison of tracking performance.
- **Layout**: Two-panel horizontal layout.
  - Left panel: Dot plot showing HOTA and IDF1 accuracy metrics (higher is better).
  - Right panel: Horizontal bar plot showing identity switches (lower is better, showing reduction from 64 to 39 to 8).
- **Evaluated Systems**:
  1. *Raw ByteTrack Baseline*: Unmodified standard ByteTrack.
  2. *Online RealTime-Fast*: Causal online tracker with frame skipping and identity preservation.
  3. *Offline Hybrid-ByteTrack*: Retrospective two-pass smoothing association.

### `behavior_cv_summary.svg`
- **Purpose**: Forest / dot-and-whisker plot of 5-fold cross-validation behavior recognition results.
- **Layout**: Horizontal dot-and-whisker plot on range $[0.60, 0.76]$ with sample standard deviation error bars.
- **Evaluated Systems**:
  1. *Multimodal Spatio-Temporal Baseline*: Primary high-ceiling baseline (`0.6778 ± 0.0268`).
  2. *Joint-Representation Baseline*: Relational joint-feature representation (`0.6708 ± 0.0349`).
  3. *Class-Aware Spatial-Gated Model*: Pre-GAP spatial attention with class gating (`0.6846 ± 0.0292`).
  4. *Baseline Fixed Ensemble*: Equal-weighted logit ensemble of baselines (`0.6926 ± 0.0402`).
  5. *Final Spatial-Gated Ensemble*: Locked authoritative production system (`0.6939 ± 0.0377`).

### `profile_distortion_summary.svg`
- **Purpose**: Quantifies downstream distortion in longitudinal individual activity budgets caused by tracking identity swaps.
- **Layout**: Horizontal bar / point display showing Total Variation distance ($0.0953 \to 0.0309 \to 0.0001$).
- **Key Finding**: Demonstrates that minimizing identity switches (from 64 to 8) directly preserves longitudinal individual time budgets, reducing profile distortion to near-zero ($0.0001$).
