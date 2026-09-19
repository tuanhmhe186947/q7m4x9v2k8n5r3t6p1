# Table Plan for Manuscript

This document specifies the structure, headings, target placement, data sources, and notes for
all
tables in the main manuscript and supplementary materials.

---

## Main Paper Tables

### Table 1: Multi-Object Tracking Performance on Independent Evaluation Cohort
- **Target Placement**: Section 4.1
- **Title**: Multi-Object Tracking Performance on Independent Evaluation Cohort (12 Held-Out
  Videos,
  21,600 Frames).
- **Columns**:
  - `Method`: Raw ByteTrack, RealTime-Fast, Hybrid-ByteTrack.
  - `Role`: Frozen Baseline, Causal Online (15 Hz), Retrospective Offline (30 Hz).
  - `Agg HOTA (%)`: Aggregate Higher Order Tracking Accuracy.
  - `Video Mean HOTA (%) [95% CI]`: Mean per-video HOTA with 95% bootstrap confidence interval.
  - `DetA (%)`: Detection Accuracy.
  - `AssA (%)`: Association Accuracy.
  - `Agg IDF1 (%) [95% CI]`: Identification F1 score with 95% bootstrap confidence interval.
  - `Total IDSW (Median)`: Total discrete identity switches and median per video.
  - `Wrong-ID Exposure (Frames)`: Cumulative frames where tracks held mismatched identities.
  - `Wrong-ID Time (s)`: Physical duration of wrong-identity exposure in seconds.
  - `Wrong-ID % Matched`: Fraction relative to authoritative matched ground-truth frames
    (canonical).
  - `Wrong-ID % All-GT`: Fraction relative to all 172,800 ground-truth frames.
  - `Episodes`: Total identity error episodes.
  - `Persistent Swaps`: Swaps lasting longer than 5 seconds.
- **Source**: `outputs/paper_confirmatory/tracking/FINAL_TRACKING_CONFIRMATORY_AUTHORITY.json`

---

### Table 2: Paired Bootstrap Contrasts Between Tracking Methods
- **Target Placement**: Section 4.1
- **Title**: Paired Statistical Contrasts Between Tracking Methods Across Independent Evaluation
  Videos
  (B = 10,000 Bootstrap Iterations).
- **Columns**:
  - `Tracker Comparison`: Hybrid vs. Raw, Hybrid vs. RealTime, RealTime vs. Raw.
  - `Delta HOTA (%) [95% CI]`: Paired difference in HOTA with 95% percentile bootstrap CI.
  - `Bootstrap Positive Fraction`: Fraction of bootstrap iterations where Delta > 0.
  - `Delta IDF1 (%) [95% CI]`: Paired difference in IDF1 with 95% percentile bootstrap CI.
  - `Bootstrap Positive Fraction`: Fraction of bootstrap iterations where Delta > 0.
  - `Relative IDSW Reduction (%)`: Percentage reduction in total identity switches.
- **Source**: `outputs/paper_confirmatory/tracking/FINAL_TRACKING_CONFIRMATORY_AUTHORITY.json`

---

### Table 3: 5-Fold Cross-Validation Performance of Spatiotemporal Behavior Models
- **Target Placement**: Section 4.2
- **Title**: 5-Fold Cross-Validation Performance of Spatiotemporal Behavior Recognition Models
  Across
  33,287 Canonical Units.
- **Columns**:
  - `Model / System`: Model A, Model B, Model J Replay, Model K, $J_{PBNEW}$, $E_0$,
    $E_{J\_FIXED50}$,
    $E_{J\_PBNEW}$.
  - `VG1` to `VG5`: Macro-F1 scores for each of the five balanced video-group folds.
  - `Mean Macro-F1`: Fold-mean Macro-F1 score across the 5 folds.
  - `Sample SD ($s$, ddof=1)`: Sample standard deviation (primary reporting convention).
  - `Pop. SD ($\sigma$, ddof=0)`: Population standard deviation.
  - `Role / Status`: Architectural role, promotion status, and governance notes.
- **Source**: `docs/paper/final_behavior_cv_table.csv`

---

### Table 4: Model J 2x2 Factorial Ablation Matrix and Effect Decomposition
- **Target Placement**: Section 4.3
- **Title**: Model J 2x2 Factorial Ablation Matrix Across 5 Cross-Validation Folds.
- **Columns**:
  - `Configuration`: Config N (Scaffold), Config L (Local Only), Config G (Gate Only), Model J
    (Full).
  - `Local Spatial Attention`: Binary (Yes/No).
  - `Class-Aware Gating`: Binary (Yes/No).
  - `VG1` to `VG5`: Macro-F1 scores for each fold.
  - `Mean Macro-F1`: Fold-mean Macro-F1 score.
  - `Delta vs. Config N`: Absolute change relative to the neutral scaffold baseline.
- **Footer Decompositions**: Scaffold Effect (+0.000000), Local Effect Neutral (+0.000000), Gate
  Effect No Local (+0.005109), Full Effect vs. Scaffold (+0.006857), Descriptive Interaction
  (+0.001748).
- **Source**: `outputs/classification_v2/final_j_ablation_v1/summary.json`

---

### Table 5: Downstream Behavioral Profile Distortion Across 96 Pig Sessions
- **Target Placement**: Section 4.4
- **Title**: Downstream Behavioral Profile Distortion Under Fixed Human Ground Truth Across 96
  Pig-Video Sessions (DEV12 Cohort).
- **Columns**:
  - `Tracking Mode`: Raw ByteTrack, RealTime-Fast, Hybrid-ByteTrack.
  - `Mean A1 TV Distortion`: Mean Total Variation distance between predicted and true behavioral
    budgets.
  - `95% Cluster Bootstrap CI`: Percentile confidence interval clustered at the video session
    level.
  - `Median A1 TV`: Median distortion across 96 sessions.
  - `IQR A1 TV`: Interquartile range.
  - `Mean A2 TV Distortion (Coverage Adjusted)`: Total Variation distance incorporating coverage
    drop.
  - `Mean Tracking Coverage (%)`: Mean percentage of ground-truth frames matched.
- **Source**:
  `outputs/paper_analysis/identity_profile_propagation_dev12_rawbaseline_v1/summary.json`

---

### Table 6: Pairwise Profile Distortion Comparisons and Rank Correlation Analysis
- **Target Placement**: Section 4.4 / 4.5
- **Title**: Pairwise Statistical Comparisons and Rank Correlation Analysis Between Tracking
  Error
  Metrics and Downstream Profile Distortion.
- **Panel A: Pairwise Profile Comparisons**:
  - `Tracker Comparison`, `Mean Delta A1 TV`, `95% Bootstrap CI`, `LOOV Sign Consistency`, `LOOV
    Range`.
- **Panel B: Rank Association Analysis**:
  - `Tracking Method`, `Metric (Exposure vs. IDSW)`, `Spearman $\rho$ vs. A1 TV`, `95% Bootstrap
    CI`,
    `Delta Abs Rho ($\Delta |\rho|$)`, `95% Bootstrap CI for Delta`.
- **Source**: `outputs/paper_analysis/identity_profile_propagation_dev12_rawbaseline_v1/`

---

## Supplementary Tables

- **Table S1: Master Dataset Role and Partition Inventory**: Full specification of all cohorts,
  video
  counts, frame counts, annotation modalities, and access restrictions
  (`docs/paper/dataset_role_table.csv`).
- **Table S2: Per-Class F1 Scores Across Behavior Models**: Breakdown across all 10 ethogram
  categories
  for Model A, Model B, Model J, $J_{PBNEW}$, and $E_{J\_FIXED50}$
  (`docs/paper/behavior_per_class_uncertainty.csv`).
- **Table S3: Per-Video Tracking Breakdown on Independent Evaluation Cohort**: Detailed metrics
  for each
  of the 12 held-out videos for all three tracking systems.
- **Table S4: System Complexity and Runtime Profile**: Model parameter counts, floating-point
  operations
  (FLOPs), inference latency (ms/frame), and VRAM footprint
  (`docs/paper/system_complexity_table.csv`).
