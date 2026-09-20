# Scientific Reproduction Guide

This document outlines the protocol for verifying scientific claims, running confirmatory tracking evaluations, and inspecting cross-validation models reported in the paper.

---

## System & Model Nomenclature Mapping

To bridge public-facing semantic names used in the manuscript and README with internal code identifiers and experimental run logs:

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

## 1. Automated Claim & Metric Token Verification

All quantitative numbers reported in the paper narrative are mechanically checked against the frozen evidence ledger:

```bash
python scripts/paper/check_claim_numbers.py
```

- **Expected Outcome**: 100% of numeric tokens match the master evidence ledger across 455 target metrics.

---

## 2. Tracking Confirmatory Evaluation (12 Held-Out Videos)

The tracking results evaluate 12 held-out independent videos (21,600 frames, 96 pig trajectories) under `TRACKING_EVALUATOR_STANDARD_V2`:

```bash
# Evaluate the retrospective offline tracker (Offline Hybrid-ByteTrack)
python scripts/evaluate_tracking.py \
    --mode hybrid_bytetrack \
    --eval-config hybrid_bytetrack_best \
    --gt-dir data/ground_truth/confirmatory_12 \
    --output-dir outputs/reproduction/tracking_hybrid

# Evaluate the causal online tracker (Online RealTime-Fast)
python scripts/evaluate_tracking.py \
    --mode realtime \
    --eval-config realtime_fast \
    --gt-dir data/ground_truth/confirmatory_12 \
    --output-dir outputs/reproduction/tracking_realtime

# Evaluate the frozen technical baseline (Raw ByteTrack Baseline)
python scripts/evaluate_tracking.py \
    --mode bytetrack_raw \
    --eval-config bytetrack_raw \
    --gt-dir data/ground_truth/confirmatory_12 \
    --output-dir outputs/reproduction/tracking_raw
```

- **Reference Numbers**: See [tracking_confirmatory.md](paper/tables/tracking_confirmatory.md).

---

## 3. Behavior Recognition 5-Fold Cross-Validation

The behavior recognition experiments utilize 5-fold group-aware cross-validation across video cohorts VG1–VG5 to ensure zero video leakage:

```bash
# Evaluate 5-fold ensemble predictions (Final Spatial-Gated Ensemble)
python scripts/classification_v2/08_multimodal_eval/evaluate_jpbnew_b_ensemble.py \
    --alpha 0.50 \
    --output-dir outputs/reproduction/behavior_ej
```

- **Reference Numbers**: See [behavior_cv.md](paper/tables/behavior_cv.md).

---

## 4. Downstream Profile Distortion (DEV12 Cohort)

To measure how identity switches distort longitudinal behavioral budgets, individual time budgets are projected across the 12-video development overlap cohort:

```bash
python scripts/paper/generate_profile_distortion_metrics.py \
    --output-dir outputs/reproduction/profile_tv
```

- **Total Variation (A1 Metric)**:
  - Raw ByteTrack Baseline: `0.0953`
  - Online RealTime-Fast: `0.0309`
  - Offline Hybrid-ByteTrack: `0.0001`
