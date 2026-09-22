# Scientific Reproduction Guide

This document outlines the protocol for inspecting system nomenclature, running confirmatory tracking evaluations, and evaluating cross-validation models reported in the project.

---

## System & Model Nomenclature Mapping

To bridge public-facing semantic names with internal code identifiers and experimental run logs:

| Public Name | Internal Experiment ID | Role |
| :--- | :--- | :--- |
| Multimodal Spatio-Temporal Baseline | Model A (`final_high_ceiling_v1`) | Primary high-ceiling baseline |
| Joint-Representation Baseline | Model B (`joint_representation_v1`) | Relational joint-feature representation |
| Class-Aware Spatial-Gated Model | Model J (`NearFinalModelJ` / `J_F2`) | Pre-GAP spatial attention with class gating |
| Baseline Fixed Ensemble | E0 (`0.50*A + 0.50*B`) | Equal-weighted baseline logit ensemble |
| Final Spatial-Gated Ensemble | E_J_FIXED50 (`0.50*J + 0.50*B`) | Locked authoritative production system |
| Raw ByteTrack Baseline | `bytetrack_raw` | Unmodified ByteTrack reference baseline |
| Online RealTime-Fast | `realtime_fast` | Causal online tracker with frame skipping |
| Offline Hybrid-ByteTrack | `hybrid_bytetrack_best` | Retrospective two-pass offline association |

---

## 1. Tracking Confirmatory Evaluation (12 Held-Out Videos)

The tracking results evaluate 12 held-out independent videos (21,600 frames, 96 pig trajectories) under `TRACKING_EVALUATOR_STANDARD_V2`:

```bash
# Evaluate the retrospective offline tracker (Offline Hybrid-ByteTrack)
python scripts/evaluate_tracking.py \
    --mode hybrid_bytetrack \
    --eval-config hybrid_bytetrack_best \
    -a

# Evaluate the causal online tracker (Online RealTime-Fast)
python scripts/evaluate_tracking.py \
    --mode realtime \
    --eval-config realtime_fast \
    -a

# Evaluate the frozen technical baseline (Raw ByteTrack Baseline)
python scripts/evaluate_tracking.py \
    --mode bytetrack_raw \
    --eval-config bytetrack_raw \
    -a
```

- **Reference Metrics**:
  - Raw ByteTrack Baseline: HOTA 89.41%, IDF1 94.23%, 64 ID switches
  - Online RealTime-Fast: HOTA 90.33%, IDF1 95.03%, 39 ID switches (-39.1%)
  - Offline Hybrid-ByteTrack: HOTA 93.55%, IDF1 98.41%, 8 ID switches (-87.5%)

> **Note on Data Availability**: Running full trajectory evaluation against ground truth requires the 12 held-out video files and ground-truth CVAT XML annotations (`data/ground_truth/`), which are private research artifacts and not distributed in this public repository.

---

## 2. Behavior Recognition 5-Fold Cross-Validation

The behavior recognition experiments utilize 5-fold group-aware cross-validation across video cohorts VG1–VG5 to ensure video-group-disjoint evaluation:

- **Reference Metrics**:
  - Multimodal Spatio-Temporal Baseline: Macro-F1 = 0.6778 ± 0.0268
  - Joint-Representation Baseline: Macro-F1 = 0.6708 ± 0.0349
  - Class-Aware Spatial-Gated Model: Macro-F1 = 0.6846 ± 0.0292
  - Baseline Fixed Ensemble: Macro-F1 = 0.6926 ± 0.0402
  - Final Spatial-Gated Ensemble: Macro-F1 = 0.6939 ± 0.0377

> **Note on Reproduction**: These cross-validation results are reported from frozen experimental evidence. Full re-evaluation requires the original video crops, precomputed multimodal feature caches (HDF5), and trained model checkpoints that are private research artifacts and not distributed in this public repository.

---

## 3. Downstream Profile Distortion (DEV12 Cohort)

To measure how identity switches distort longitudinal behavioral budgets, individual time budgets are projected across the 12-video development overlap cohort:

- **Total Variation ($L_1$ Metric)**:
  - Raw ByteTrack Baseline: `0.0953`
  - Online RealTime-Fast: `0.0309` (-67.6%)
  - Offline Hybrid-ByteTrack: `0.0001` (-99.9%)

> **Note on Reproduction**: Profile distortion metrics are computed from frozen trajectory-to-behavior projection evidence on the DEV12 cohort. Running this analysis requires the paired tracking outputs, aligned human behavioral annotations, and private dataset splits not distributed in this public repository.

