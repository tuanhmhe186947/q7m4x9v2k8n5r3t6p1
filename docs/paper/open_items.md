# Open Items and Unresolved Status Ledger

This document tracks all remaining technical, experimental, and manuscript items for the
pig behavior project paper.

## 1. Tracking Confirmatory Cohort Status
- **Current State**: `PREFLIGHT_CONTRACT_COMPLETE_UNSEEN_STILL_CLOSED`
- **Authority**:
  `docs/tracking/method_standardization/TRACKING_UNSEEN_PREFLIGHT_AUTHORITY_20260730.json`
- **Files Accessed**: `0 / 12` videos.
- **Scientific Rationale**: The 12 independent confirmatory tracking videos are
  deliberately held closed to prevent developmental leakage, hyperparameter tuning, or
  split contamination.
- **Next Action**: Execute inference and Standard V2 evaluation on the confirmatory cohort
  only when authorized for final publication verification.

## 2. Behavior Outer Test Set Status
- **Current State**: `CLOSED / UNTOUCHED`
- **Evaluations Executed**: `OUTER_TEST_EVALUATIONS = 0`
- **Scientific Rationale**: Preserving the outer behavior test set guarantees that all
  single-model comparisons (Model A, B, J, K, J_PBNEW), factorial ablations (N, L, G, J),
  and ensemble selections (E_0, E_J) reflect genuine cross-validation without test
  leakage.
- **Next Action**: Perform a single, un-tuned evaluation on the outer test set as part of
  final manuscript submission checks.

## 3. Raw ByteTrack Upstream Detector Comparability
- **Current State**: `PARTIAL`
- **Authority**:
  `outputs/paper_analysis/identity_profile_propagation_dev12_rawbaseline_v1/summary.json`
- **Technical Note**: The frozen Raw ByteTrack baseline uses historical B0 predictions
  evaluated under the identical Standard V2 bipartite matching contract. However, upstream
  detector parameters differ slightly (det_conf=0.25, max_raw_detections=32 vs historical
  0.20/20 live topology).
- **Scientific Reporting**: The paper reports Raw ByteTrack as a whole-pipeline historical
  baseline rather than claiming purely isolated association dominance.

## 4. Provenance and Artifact Completeness
- **Unresolved Provenance Items**: `0`
- **Physical Checkpoints**: All 15 Model J ablation checkpoints, 5 J_PBNEW checkpoints,
  Model A, and Model B weights are verified on disk.
- **Evaluation Outputs**: 100% of tracking CSVs, behavior prediction arrays, and identity-
  to-profile tables are materialized in committed workspace paths.
- **Blockers for Ledger**: `NONE`.
- **Manuscript Readiness**: `READY_FOR_FULL_MANUSCRIPT_DRAFT = YES`.
