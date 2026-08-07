# Paid-GPU gate resolution options

> Status: Historical decision record, superseded for the current Phase-2 E0
> route. It is retained only as provenance for the 2026-08-06 permit decision
> and does not grant paid-GPU authorization.

Ngày: 2026-08-06
Plan authority: `PRE_GPU_AUTORESEARCH_EXECUTION_PLAN_20260804.md`
Plan SHA-256: `cd96fb3d37bc0dc0f366e2a7ef76511f39e0284ed9a0a132675774697620dbce`

## Current evidence

The current Classification V2 code authority is `884016a`. The older A12
artifact at `4d025df` reported a B3-minus-B2 Macro-F1 gain of `+0.038073`, but
that commit did not bind the fold-local event-weight manifest in the paired
runner. The current `884016a` correction filters zero-weight training rows
through `_fold_training_mask`; its independently checked paired gain is
`+0.005264`, with mean target-recall gain `-0.2917`.

The snapshot, effective-window index, and base split hashes are identical in
both artifacts. Therefore this is a code-authority reconciliation, not a data
change. The old result is `STALE_FOR_CURRENT_CODE_AUTHORITY` and must not be
used for a GPU permit.

The remaining independent limitation is structural: the frozen date-grouped
outer split has a CVAT-only held-out fold and a legacy-only held-out fold. A
full source-balanced held-out estimate cannot be obtained for those folds
without changing the scientific split/metric authority.

## Route 1 — strict claim-grade permit

Keep the current permit rule unchanged. First authorize a versioned scientific
amendment that defines either:

1. a new outer split in which every source-balanced held-out estimate is
   estimable while preserving date and native-unit purity; or
2. a formally registered A12 estimand that combines source-balanced folds with
   the predeclared cross-source transfer controls for the single-source folds.

Then rebuild the split-dependent artifacts, rerun A12 under the new authority,
lock a finalist, expand four-fold native OOF/calibration, and regenerate the
result package. The amendment must invalidate the current permit inputs and
start again at P0.

## Route 2 — engineering-only paid pilot amendment

Add a versioned permit class such as `G19_ENGINEERING_PILOT` to the plan. It
would allow a remote one-fold/one-seed pilot only with:

- no paper or model-performance claim;
- no auto-promotion or finalist selection;
- no claim-grade OOF or calibration;
- current A12 and G14 limitations shown in every run manifest;
- an explicit budget, timeout, checkpoint, and stop rule.

This route changes the plan's scientific authorization and requires explicit
user acceptance before any paid resource is started. It does not convert the
current A12 result into `PASS`.

## Current decision

No permit is issued. No split or gate threshold is changed. The next action is
to obtain explicit authority for one of the two routes above; until then,
`G10=INCONCLUSIVE`, `G14=PARTIAL`, `G17/G18/G19=BLOCKED` remains the correct
fail-closed state.

Evidence:
`outputs/classification_v2/model_readiness_audit/`
`a12_authority_reconciliation_884016a_20260806_r1/`
`a12_authority_reconciliation.json`
