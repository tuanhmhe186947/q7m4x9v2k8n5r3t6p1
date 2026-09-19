# Behavior Outer Test Set Readiness Audit

## ABSOLUTE SCIENTIFIC CONSTRAINT
- `OUTER_TEST_EVALUATIONS = 0`
- The outer behavior test set was **NOT** evaluated in this task and has **NEVER** been
  evaluated during development.
- All single-model comparisons (Model A, B, J, K, J_PBNEW), factorial matrix runs (N, L,
  G, J), and ensemble selections (E_0, E_J) were executed strictly on cross-validation
  folds.

## 1. Outer Test Data Audit
- **Outer Manifest Authority**: `outputs/classification_v2/video_group_balanced_5fold_auth
  ority_20260821/video_group_balanced_5fold_audit.json`
- **Total Outer Test Units**: 33,287 canonical units partitioned across 5 disjoint outer
  test folds:
  - Outer Fold VG1: 5,963 units (167 video keys)
  - Outer Fold VG2: 7,522 units (56 video keys)
  - Outer Fold VG3: 6,560 units (258 video keys)
  - Outer Fold VG4: 5,820 units (156 video keys)
  - Outer Fold VG5: 7,422 units (41 video keys)
- **Disjoint Video Integrity**: `video_overlap_by_fold = 0` across all 5 outer folds. No
  video appears in both training and outer test.
- **Zero Outer Leakage**: Zero test rows were accessed for hyperparameter search, early
  stopping, distillation, or ensemble weighting.

## 2. Locked System for One-Shot Outer Evaluation
- **System Identity**: `E_J_FIXED50` ($0.50 \cdot J + 0.50 \cdot B$).
- **No Reselection**: Under project charter rules, if outer evaluation is authorized, no
  model reselection, weight adjustment, or post-hoc threshold tuning is permitted.

## 3. Physical Readiness Checklist
- [x] Outer dataset manifest exists and is verified.
- [x] Outer groups are completely disjoint from training groups.
- [x] Outer labels were never used for model development or early stopping.
- [x] Model J physical checkpoints exist on disk for all 5 folds.
- [x] Model B physical checkpoints/logits exist on disk for all 5 folds.
- [x] Strict inference runner exists and is deterministic.
- [x] Evaluator contract is frozen (`10-class Macro-F1`).

## 4. One-Shot Execution Protocol (Requires Explicit User Authorization)
If the user explicitly commands outer evaluation:
```cmd
python scripts/classification_v2/evaluate_outer_test.py ^
  --system E_J_FIXED50 ^
  --manifest outputs/classification_v2/video_group_balanced_5fold_authority_20260821/full_
t6_video_group_balanced_5fold_manifest.csv ^
  --seed 240494961
```
**Current Execution State**: `BEHAVIOR_OUTER_READY = YES`, `OUTER_TEST_EVALUATIONS = 0`.
