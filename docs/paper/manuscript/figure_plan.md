# Figure Plan for Manuscript

This document specifies the figure layout, captions, data sources, and visual specifications for
the
main manuscript and supplementary materials.

---

## Main Paper Figures

### Figure 1: End-to-End System Overview and Dual-Mode Architecture
- **Title**: Overview of the Dual-Mode Multi-Object Tracking and Multimodal Behavior Recognition
  Framework.
- **Type**: Multi-panel conceptual and architectural diagram.
- **Sub-panels**:
  - **(a) Data Acquisition and Sensor Setup**: Overhead top-down camera view of a commercial pig
    pen
    (8 pigs, feeders, drinkers, enrichment toys).
  - **(b) Dual-Mode Tracking Framework**:
    - *Causal Online Mode (RealTime-Fast)*: 15 Hz detector cadence, Kalman filtering, velocity
      gating,
      zero future lookahead for streaming alert systems.
    - *Retrospective Offline Mode (Hybrid-ByteTrack)*: 30 Hz detector cadence, bidirectional
      Kalman
      smoothing, spatial-temporal graph conflict resolution for longitudinal profiling.
  - **(c) Multimodal Behavior Recognition**: Actor RGB crops ($T=6$), 6D spatial geometry, 12D
    relative motion, and Top-$K$ partner social context routed through Model J with local
    spatial
    attention and class-aware gating.
  - **(d) Downstream Behavioral Profile Compilation**: Accumulation of individual activity time
    budgets across 10 ethogram categories and computation of Total Variation (TV) profile
    distortion.
- **Placement**: Section 1 or Section 3 (Full-width, 2-column).

---

### Figure 2: Multi-Object Tracking Performance on Independent Evaluation Cohort
- **Title**: Multi-Object Tracking Quality and Error Exposure Across 12 Independent Evaluation
  Videos.
- **Type**: Multi-panel comparative bar and boxplot visualization.
- **Sub-panels**:
  - **(a) Higher Order Tracking Accuracy (HOTA) and Identification F1 (IDF1)**: Aggregate scores
    and
    95% bootstrap confidence intervals comparing Raw ByteTrack (89.41% HOTA, 94.23% IDF1),
    RealTime-Fast
    (90.33% HOTA, 95.03% IDF1), and Hybrid-ByteTrack (93.55% HOTA, 98.41% IDF1).
  - **(b) Identity Switch Distribution (IDSW)**: Boxplot of per-video identity switch counts
    across the
    12 independent videos, highlighting the 87.5% reduction achieved by Hybrid-ByteTrack (8
    IDSW,
    median 0.0) and 39.1% reduction by RealTime-Fast (39 IDSW, median 0.0) vs. Raw ByteTrack (64
    IDSW,
    median 3.0).
  - **(c) Cumulative Wrong-Identity Exposure Duration**: Physical time (seconds) and frame count
    during which tracks carried an incorrect ground-truth identity (Raw: 953.43 s; RealTime:
    767.63 s;
    Hybrid: 286.57 s).
- **Source Data**:
  `outputs/paper_confirmatory/tracking/FINAL_TRACKING_CONFIRMATORY_AUTHORITY.json`
- **Placement**: Section 4.1 (1-column or 1.5-column).

---

### Figure 3: Model J Factorial Architectural Ablation and Effect Decomposition
- **Title**: Architectural Factor Decomposition of Spatio-Temporal Behavior Recognition Models.
- **Type**: Dual-panel graphic combining 5-fold Macro-F1 bars and waterfall effect
  decomposition.
- **Sub-panels**:
  - **(a) 5-Fold Validation Macro-F1 Across Factorial Configurations**: Bar chart with error
    bars
    showing Model A baseline (0.677789 +/- 0.0240), neutral scaffold Config N (0.677789),
    local-spatial
    only Config L (0.677789), class-aware gate only Config G (0.682898 +/- 0.0203), and full
    Model J
    (0.684646 +/- 0.0267).
  - **(b) Waterfall Effect Decomposition**: Stepwise decomposition showing Scaffold Effect
    (+0.000000),
    Neutral Local Effect (+0.000000), Class Gate Effect (+0.005109), and Descriptive Interaction
    (+0.001748), illustrating Case D synergistic interaction.
- **Source Files**: `docs/figures/model_j_factor_ablation/figure1_mean_macro_f1.png` and
  `docs/figures/model_j_factor_ablation/figure4_factor_decomposition.png`
- **Placement**: Section 4.3 (1.5-column).

---

### Figure 4: Downstream Behavioral Profile Distortion Under Fixed Ground Truth
- **Title**: Paired Video-Level Behavioral Profile Distortion Across Three Tracking Modes.
- **Type**: Paired dot-and-line plot across 12 development videos (96 pig-video sessions).
- **Content**: Displays video-level mean A1 Total Variation profile distortion comparing Raw
  ByteTrack
  (mean 0.0953), RealTime-Fast (mean 0.0309), and Hybrid-ByteTrack (mean 0.0001). Trajectory
  lines
  connecting the three trackers for each video demonstrate 100% leave-one-video-out sign
  consistency
  (all 12 video-folds show Raw > RealTime > Hybrid).
- **Source File**:
  `docs/figures/identity_profile_propagation_dev12_rawbaseline/figure1_paired_video_a1_tv.png`
- **Placement**: Section 4.4 (1-column).

---

### Figure 5: Predictive Capacity of Error Duration vs. Discrete Switches
- **Title**: Association Between Upstream Tracking Error Metrics and Downstream Profile
  Distortion.
- **Type**: Dual-panel scatter plot and forest plot.
- **Sub-panels**:
  - **(a) Scatter Correlation**: Scatter plot and linear rank fit of cumulative wrong-ID
    exposure
    frames versus A1 TV profile distortion for Raw ByteTrack ($\rho = 0.9381$) and RealTime-Fast
    ($\rho = 0.9698$).
  - **(b) Forest Plot of Correlation Differences**: Point estimates and 95% cluster bootstrap
    confidence intervals for $\Delta |\rho| = |\rho_{exposure}| - |\rho_{idsw}|$. Shows strictly
    positive interval for Raw ByteTrack (+0.0668 [0.0116, 0.1501]), interval spanning zero for
    RealTime-Fast (+0.0512 [-0.0006, 0.1577]), and degenerate status for Hybrid-ByteTrack (0
    IDSW).
- **Source Files**:
  `docs/figures/identity_profile_propagation_dev12_rawbaseline/figure2_wrong_id_vs_a1_tv.png`
  and `figure3_exposure_vs_idsw_association.png`
- **Placement**: Section 4.5 (1.5-column).

---

## Supplementary Figures

- **Figure S1: Paired Cross-Validation Fold Trajectories Across Model J Configurations**:
  Fold-by-fold
  trajectory plot across VG1 to VG5 for Configs N, L, G, and J (`FIG-ABL-2`).
- **Figure S2: Per-Class F1 Deltas Across Ablation Configurations vs. Neutral Scaffold**:
  Grouped bar
  chart decomposing F1 changes across the 10 ethogram categories (`FIG-ABL-3`).
- **Figure S3: Per-Class Behavioral Profile Distortion Decomposed Across Ethogram Categories**:
  Breakdown of absolute percentage error across each of the 10 behaviors for Raw ByteTrack,
  RealTime-Fast,
  and Hybrid-ByteTrack (`FIG-PROF-4`).
- **Figure S4: Dual-Axis Synthesis Connecting Tracking Metrics to Profile Errors**:
  Comprehensive
  two-panel plot linking HOTA, IDF1, and IDSW to A1/A2 TV distortion (`FIG-PROF-5`).
