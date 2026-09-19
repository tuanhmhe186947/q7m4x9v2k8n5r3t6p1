# Supplementary Materials Plan

This document outlines the detailed contents, tables, figures, and protocols reserved for the
Supplementary Materials document supporting the main manuscript.

---

## S1. Extended Dataset Governance and Ethogram Specifications

### S1.1 Swine Behavioral Ethogram: Operational Definitions and Boundary Cases
- Detailed operational inclusion and exclusion criteria for all 10 behavioral categories
  (`drink`,
  `eat`, `fight`, `social-nose`, `explore`, `lying`, `stand`, `move`, `sitting`, `playwithtoy`).
- Specific guidelines for resolving ambiguous transition boundaries (e.g., distinguishing brief
  locomotion adjustments during standing from active movement bouts; differentiating
  non-aggressive
  facial sniffing from low-intensity aggressive head knocks).
- Quality control and consensus review procedures across multiple expert annotators.

### S1.2 Master Dataset Role and Partition Inventory
- Complete tabular inventory of all five dataset partitions, reproducing Table S1 based on
  `docs/paper/dataset_role_table.csv`:
  - Tracking Development Cohort (Full, 13 videos, 187,200 frames).
  - Tracking Development / Behavior Overlap Cohort (DEV12, 12 videos, 172,800 frames, 28,800
    canonical
    units).
  - Tracking Independent Confirmatory Cohort (12 videos, 21,600 evaluated frames,
    $CONFIRMATORY_B$).
  - Behavior Recognition 5-Fold Balanced Cross-Validation Cohort (33,287 units, 678 video clips
    across
    VG1 to VG5).
  - Behavior Nested Outer Protocol (defined in `outer_oof_contract.json` as
    `FROZEN_PROTOCOL_NOT_AUTHORIZED`, not materialized on disk; $OUTER\_TEST\_EVALUATIONS = 0$).

---

## S2. Multi-Object Tracking Specifications and Per-Video Diagnostics

### S2.1 Upstream Detector and Association Hyperparameters
- Full hyperparameter specification table:
  - YOLOv8 detector input resolution, confidence threshold (0.25), NMS IoU threshold (0.45), max
    proposals (32).
  - RealTime-Fast: 15 Hz detector cadence (step = 2), Kalman filter state vector dimensions,
    velocity
    gating threshold, maximum track coasting frames.
  - Hybrid-ByteTrack: 30 Hz detector cadence (step = 1), high-confidence matching threshold
    (0.50),
    low-confidence matching threshold (0.10), bidirectional Kalman smoothing window, trajectory
    graph
    conflict resolution parameters.

### S2.2 Standard V2 Evaluation Contract Details
- Mathematical definition of bipartite Hungarian matching over the 19-point alpha threshold grid
  $\alpha \in [0.05, 0.95]$.
- Explicit formulas for both wrong-ID exposure denominators:
  $$\text{wrong\_id\_fraction\_matched} = \frac{\text{wrong\_id\_matched\_frames}}
  {\text{authoritative\_matched\_gt\_frames}}$$
  $$\text{wrong\_id\_fraction\_all\_gt} = \frac{\text{wrong\_id\_matched\_frames}}{172,800}$$

### S2.3 Per-Video Tracking Breakdown on Independent Evaluation Cohort
- Full per-video performance table reporting HOTA, DetA, AssA, IDF1, IDSW, wrong-ID exposure
  frames,
  and error episodes for each of the 12 held-out confirmatory videos across all three tracking
  systems.

---

## S3. Spatio-Temporal Model Architectures and Cross-Validation Diagnostics

### S3.1 Deep Neural Network Architectural Specifications
- Layer-by-layer architectural diagrams and parameter counts for Model A, Model B, Model J,
  Model K,
  and $J_{PBNEW}$.
- Specification of the local spatial attention mechanism: attention head count, key/query/value
  projection dimensions, spatial pooling strategies.
- Specification of the class-aware gating module: feed-forward layer topology, sigmoid
  activation, and
  residual modulation pathways.

### S3.2 Training Hyperparameters and Optimization Regimes
- Optimizer (AdamW), initial learning rate ($3 \times 10^{-4}$), weight decay ($1 \times
  10^{-4}$),
  batch size (16), gradient accumulation steps, mixed-precision FP16 settings.
- Hardware specifications: NVIDIA L4 GPU (24 GB VRAM) on cloud compute infrastructure with
  deterministic
  seeding.

### S3.3 Per-Class Behavior Metrics and Confusion Matrices
- Tabular breakdown of per-class Precision, Recall, and F1 scores across the 10 ethogram
  categories
  for Model A, Model B, Model J, $J_{PBNEW}$, and $E_{J\_FIXED50}$ (Table S2).
- Normalized 10-class confusion matrices for Model A, Model J, and the final ensemble
  $E_{J\_FIXED50}$,
  illustrating error reduction across specific behavior pairs.

---

## S4. Downstream Behavioral Profile Distortion: Derivations and Diagnostics

### S4.1 Mathematical Derivations of Profile Distortion Metrics
- Formal proofs of metric properties for Metric A1 (Total Variation distance) and Metric A2
  (Coverage-Adjusted TV distance), including bounds, symmetry, and triangle inequality.

### S4.2 Leave-One-Video-Out Cross-Validation (LOOV) Diagnostics
- Tabular reporting of LOOV mean delta A1 TV values across all 12 video-folds, demonstrating
  100% sign
  consistency across all three tracker pairs:
  - Raw vs. RealTime: range [+0.0540, +0.0747] (12/12 positive).
  - Raw vs. Hybrid: range [+0.0705, +0.1038] (12/12 positive).
  - RealTime vs. Hybrid: range [+0.0164, +0.0336] (12/12 positive).

### S4.3 Session-Level Rank Correlation Scatter Diagnostics
- Full scatter plots and residual diagnostic plots for all 96 individual pig sessions,
  displaying
  wrong-ID exposure frames versus A1 TV distortion and discrete IDSW versus A1 TV distortion.

---

## S5. Reproducibility Package, Artifact Checksums, and Audit Logs

### S5.1 Software and Computational Environment
- Operating system, Python version (3.11.9), PyTorch version (2.5.1+cu121), CUDA driver version,
  and
  exact package requirements from `uv.lock`.

### S5.2 Cryptographic Checksums of Frozen Artifacts
- Complete SHA256 checksum registry reproducing
  `docs/paper/freeze_corrected_20260917/artifact_hashes.csv`
  and `checkpoint_manifest.csv`, guaranteeing 100% byte-for-byte reproducibility of all trained
  model
  weights, evaluation arrays, and summary ledgers.
