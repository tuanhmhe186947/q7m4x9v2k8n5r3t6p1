# Scientific Reproduction Guide

This document outlines the protocol for verifying scientific claims, running confirmatory tracking evaluations, and inspecting cross-validation models reported in the paper.

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
# Evaluate the retrospective offline tracker (Hybrid-ByteTrack)
python scripts/evaluate_tracking.py \
    --mode hybrid_bytetrack \
    --eval-config hybrid_bytetrack_best \
    --gt-dir data/ground_truth/confirmatory_12 \
    --output-dir outputs/reproduction/tracking_hybrid

# Evaluate the causal online tracker (RealTime-Fast)
python scripts/evaluate_tracking.py \
    --mode realtime \
    --eval-config realtime_fast \
    --gt-dir data/ground_truth/confirmatory_12 \
    --output-dir outputs/reproduction/tracking_realtime

# Evaluate the frozen technical baseline (Raw ByteTrack)
python scripts/evaluate_tracking.py \
    --mode bytetrack_raw \
    --eval-config bytetrack_raw \
    --gt-dir data/ground_truth/confirmatory_12 \
    --output-dir outputs/reproduction/tracking_raw
```

- **Reference Numbers**: See [tracking_confirmatory.md](tables/tracking_confirmatory.md).

---

## 3. Behavior Recognition 5-Fold Cross-Validation

The behavior recognition experiments utilize 5-fold group-aware cross-validation across video cohorts VG1–VG5 to ensure zero video leakage:

```bash
# Evaluate 5-fold ensemble predictions (E_J_FIXED50)
python scripts/classification_v2/08_multimodal_eval/evaluate_jpbnew_b_ensemble.py \
    --alpha 0.50 \
    --output-dir outputs/reproduction/behavior_ej
```

- **Reference Numbers**: See [behavior_cv.md](tables/behavior_cv.md).

---

## 4. Downstream Profile Distortion (DEV12 Cohort)

To measure how identity switches distort longitudinal behavioral budgets, individual time budgets are projected across the 12-video development overlap cohort:

```bash
python scripts/paper/generate_profile_distortion_metrics.py \
    --output-dir outputs/reproduction/profile_tv
```

- **Total Variation (A1 Metric)**:
  - Raw ByteTrack: `0.0953`
  - RealTime-Fast: `0.0309`
  - Hybrid-ByteTrack: `0.0001`
