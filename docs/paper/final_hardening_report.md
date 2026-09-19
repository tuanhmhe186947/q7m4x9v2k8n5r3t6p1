# Final Paper Scientific Hardening Report

**Task ID**: `PAPER-FINAL-HARDENING-20260917`
**Date**: September 17, 2026
**Status**: `COMPLETE / MANUSCRIPT_READY`

---

## 1. Domain Readiness Scorecard

| Domain | Readiness Status | Evidence / Artifact |
| :--- | :--- | :--- |
| **Tracking Development Results** | `MANUSCRIPT_READY` | 12-video DEV cohort fully validated; Standard V2 parity PASS. |
| **Tracking Confirmatory Results** | `PARTIAL / HELD_CLOSED` | Preflight contract PASS; predictions missing; held closed to preserve test integrity. |
| **Behavior Cross-Validation Results** | `MANUSCRIPT_READY` | Model A (0.6778), Model B (0.6708), Model J (0.6846), J_PBNEW (0.6848). |
| **Behavior Model Uncertainty** | `MANUSCRIPT_READY` | 10,000 video-cluster bootstrap complete; A vs EJ excludes zero; A vs J includes zero. |
| **Model J Factorial Ablation** | `MANUSCRIPT_READY` | 2x2 matrix complete; sanitized language applied; descriptive interaction documented. |
| **Ensemble Systems** | `MANUSCRIPT_READY` | E_J Fixed 50/50 (0.6939) promoted; E_J_PBNEW rejected (2/5 folds); weights locked. |
| **Identity-to-Profile Propagation** | `MANUSCRIPT_READY` | A1 TV (0.0953 -> 0.0309 -> 0.0001); LOOV 12/12 positive; Claim B sanitized. |
| **Runtime & Complexity Evidence** | `MANUSCRIPT_READY` | Tracking telemetry PASS; model parameters/checkpoints verified; GPU latency benchmark noted. |
| **Figure Quality Control** | `MANUSCRIPT_READY` | 9 candidate figures verified; QC table generated with caption caveats. |
| **Table Quality Control** | `MANUSCRIPT_READY` | 7 candidate tables verified; 100% numerical provenance established. |
| **Reproducibility Freeze Package** | `MANUSCRIPT_READY` | `docs/paper/freeze_20260917/` populated with 8 manifests and protocol docs. |
| **Behavior Outer Readiness** | `MANUSCRIPT_READY` | `OUTER_TEST_EVALUATIONS = 0`; zero leakage verified; protocol documented. |

---

## 2. Final Open-Item Classification

### A. Scientific Blockers
- **NONE**: Zero scientific blockers for drafting the complete thesis/paper manuscript
  from validated cross-validation and development evidence.

### B. Optional Strengthening
- **GPU Inference Latency Profiling**: Running isolated millisecond-per-window profiling
  on Nvidia L4 GPU for Model J and Ensemble E_J across batch sizes 1, 8, 32 (documented in
  `docs/paper/runtime_benchmark_needed.md`).

### C. Manuscript / Presentation Items
- Assemble full LaTeX manuscript integrating verified figures (Figs 1–5 profile, Figs 1, 4
  ablation) and verified tables (Tables 1–5).
- Include standard caption caveats:
  - Raw ByteTrack reflects whole-pipeline detector cadence differences.
  - Model J descriptive interaction varied across folds.
  - Claim B wrong-ID exposure superiority is strictly positive for Raw ByteTrack and
directionally consistent with confidence interval spanning zero for RealTime-Fast.

### D. Requires Explicit User Authorization
- **One-Shot Tracking Confirmatory Execution**: Decoding and running Standard V2 tracking
  evaluation on the 12 held-out confirmatory videos.
- **One-Shot Behavior Outer Evaluation**: Evaluating the locked best system (E_J Fixed
  50/50) on the 33,287 outer test units.
