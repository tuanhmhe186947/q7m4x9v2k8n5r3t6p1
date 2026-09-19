# Scientific Correction Log: Confirmatory & Behavior Audits

**Task ID**: PAPER-CONFIRMATORY-CORRECTNESS-REPAIR-20260917
**Date**: 2026-09-17

### 1. Tracking Confirmatory Provenance (CONFIRMATORY_B)
- **Finding**: Prediction and evaluation artifacts physically existed in August 2026 (
ideo_test_tracking_20260820_020217) under commit db1cccb709b9e42122e169e8d00507ab1c9ee8f1.
- **Correction**: Reclassified from new one-shot run to CONFIRMATORY_B (re-evaluated, verified,
  and certified physical August execution artifacts under Standard V2 without parameter
  changes).

### 2. Wrong-ID Exposure Denominator
- **Finding**: Inconsistent reporting between matched frames and total GT frames.
- **Correction**: Established wrong_id_fraction_matched (dividing by authoritative matched GT
  frames) as canonical standard per TRACKING_EVALUATOR_STANDARD_V2. Documented both
  denominators across all tables:
  - RealTime-Fast: 13.36% matched / 13.33% all-GT (767.63 s).
  - Hybrid-ByteTrack: 4.99% matched / 4.98% all-GT (286.57 s).
  - Raw ByteTrack: 16.65% matched / 16.55% all-GT (953.43 s).

### 3. Bootstrap Terminology
- **Finding**: Improper labeling of bootstrap positive proportion as a $-value (>0)$, and
  unsupported claims of 'significance'.
- **Correction**: Renamed to BOOTSTRAP_POSITIVE_FRACTION. Removed 'significantly outperformed'
  terminology. Primary inference established as paired point effect + 95% nonparametric
  bootstrap CI.

### 4. Behavior Outer Semantics
- **Finding**: Erroneous conflation of
ull_t6_video_group_balanced_5fold_manifest.csv (33,287 samples, 678 videos, 5 VG groups) with
an independent outer test set.
- **Correction**: Established that these 33,287 units constitute 100% of the 5-fold cross-
  validation development dataset. Outer evaluation was defined in outer_oof_contract.json as a
  4-fold outer OOF protocol (FROZEN_PROTOCOL_NOT_AUTHORIZED), but never materialized on disk.
  Outer status is formally classified as NESTED_OUTER_DEFINED_BUT_NOT_MATERIALIZED.

### 5. Behavior Primary Authorities
- **Finding**: Synthetic ~80-83% numbers were drafted into summary tables without protocol
  authority.
- **Correction**: Formally classified the 80-83% numbers as INVALID_SUPERSEDED. Re-established
  the physical 5-fold fold-mean Macro-F1 values as sole primary authorities:
  - Model A: 0.677789 (sample SD 0.026822)
  - Model B: 0.670838 (sample SD 0.034919)
  - Model J: 0.684646 (sample SD 0.029199)
  - J_PBNEW: 0.684770 (sample SD 0.025253)
  - E_J_FIXED50: 0.693905 (sample SD 0.037650)
