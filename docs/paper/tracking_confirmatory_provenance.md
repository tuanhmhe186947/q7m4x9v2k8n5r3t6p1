# Tracking Confirmatory Cohort Provenance and Status

## 1. Confirmatory Status Authority
- **Status**: `PREFLIGHT_CONTRACT_COMPLETE_UNSEEN_STILL_CLOSED`
- **Authority File**:
  `docs/tracking/method_standardization/TRACKING_UNSEEN_PREFLIGHT_AUTHORITY_20260730.json`
- **Unseen Files Accessed**: `0 / 12` videos (`UNSEEN_FILES_ACCESSED = 0`).
- **Predictions Missing**: `CONFIRMATORY_PREDICTIONS_MISSING = YES`.
- **Scientific Decision**: In accordance with the strict non-simulation and zero-leakage
  policy, trackers were not silently executed on confirmatory videos during development.
  Performance is reported as `NOT_MEASURED` until formal authorized evaluation.

## 2. Frozen Execution Protocol for Future One-Shot Evaluation
If the user authorizes execution on the 12 confirmatory videos, the exact frozen
configuration is:
- **Evaluator**: `TRACKING_EVALUATOR_STANDARD_V2` (IoU threshold $\ge 0.50$, bipartite
  Hungarian matching, `include_hidden=True`, 30 fps).
- **Methods to Evaluate**:
  1. `realtime_fast`: Frozen config hash
`9bf4ce6d07423ab517b4705c716e3eb012349b756b7c0591cc3458eac207808d`.
  2. `bytetrack_raw`: Frozen config hash
`547ae86e3be26671a9a148cb0e613ea1c602a0ff842a977ce9b7f1d217c10e41`.
  3. `hybrid_bytetrack`: Retrospective offline smoothing pipeline.
- **Execution Rule**: No parameter adjustment, threshold tuning, or method reselection is
  permitted after unseen video decoding.
