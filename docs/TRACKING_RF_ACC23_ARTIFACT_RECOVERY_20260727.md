# RF_ACC23 artifact recovery — 2026-07-27

Supersedes the `MISSING` metric statuses in
`TRACKING_RF_ACC23_AUTHORITY_CORRECTION_20260727.md`. The original RF_ACC23 run
artifacts were **found**, not reproduced. Nothing was rerun.

## 1. Where they were

The external audit document supplied by the user
(`docs/external_audits/BAO_CAO_AUDIT_RF_ACC23_20260726.md`, §1.2) states that
tracking work moved to the `PIG_task_tracking` worktree on branch
`task/update-tracking`. That lead was correct.

```
Worktree : C:\Users\ironh\Downloads\PIG_task_tracking
Branch   : task/update-tracking
HEAD     : 411e596b21261d80be679dedce6b2ce150aa3b6b  (2026-07-23 09:24:11 +0700)
           "test(tracking): gate RF_ACC23 on Windows power mode"
Tracking tree object : 752d55d3086d5bea6254a6e72eb2632589ba03e5
```

That tracking tree differs from `main`'s `8ba3d50d…`, so this lineage is **not**
byte-identical to the promoted RF_ACC23 on `main`. Read-only; nothing in that
worktree was modified.

## 2. Recovered artifacts and their hashes

| Artifact (in `PIG_task_tracking/docs/`) | SHA-256 |
|---|---|
| `TRACKING_RF_ACC23_HARD6_DECISION_20260721.json` | `7e9ccb3668ed58b99605d2f74d14b97c2079ba0b3ab0f8f60414b65c48adce79` |
| `TRACKING_RF_ACC23_FULL13_RUNTIME_DECISION_20260723.json` | `ae370dae1e2bf08402f465a7e08b6d932898a917c4031f72dfd3a1179b310c49` |
| `TRACKING_RF_ACC23_TAIL_RUNTIME_CANARY_DECISION_20260723.json` | `88ccc45325a7f5bae9fa9a22e3b65ff68ad4f1f9d53ac2a6881da078eea31a6e` |

Also present: `TRACKING_RF_ACC23_CORE_RUNTIME_PLAN_20260721.json`,
`TRACKING_RF_ACC23_TAIL_RUNTIME_CANARY_PLAN_20260723.json`, and the full
`RF_ACC10`–`RF_ACC22` experiment series.

## 3. Full-13 quality — now ARTIFACT_VERIFIED

From `TRACKING_RF_ACC23_FULL13_RUNTIME_DECISION_20260723.json`
(`run_id=20260723_rf_acc23_full13_instrumented_v1`, `commit=b0d9009`):

| Metric | Claimed in brief | Artifact value | Status |
|---|---|---|---|
| IDSW | `53` | `53` | **ARTIFACT_VERIFIED** |
| HOTA | `≈97.044%` | `0.9704398315450558` | **ARTIFACT_VERIFIED** |
| IDF1 | `≈97.077%` | `0.9707702337312571` | **ARTIFACT_VERIFIED** |
| FP/FN | `486/610` | `486` / `610` | **ARTIFACT_VERIFIED** |
| fragments | `107` | `107` | **ARTIFACT_VERIFIED** |

Quality status in the artifact is `PASS`. The identical FP/FN and fragments
versus the 2026-07-20 parent are now explained by artifact rather than
inference: RF_ACC23 changes identity assignment only.

## 4. Contract fields — now ARTIFACT_VERIFIED

From the Hard6 decision `contract` block:

```
include_hidden            = true
rule_combo                = iou0_area0_condarea0_merge0
delay_frames              = 0
output_timing_contract    = causal_framewise
detect_every_n_frames     = 2
detector_precision        = FP32
generated_mp4_allowed     = false
uses_future_frames        = false
```

This closes `INCLUDE_HIDDEN_CONFIRMED` and the evaluation-contract question.

## 5. Corrections the artifacts force

### 5.1 Hard6 membership — the brief is WRONG

| Source | Hard6 set |
|---|---|
| Task brief / external audit footnote | `000114, 000231, 000233, 000263, 000327, 000302` |
| **Run artifact** (`run.videos`) | **`000216, 000226, 000231, 000233, 000263, 000302`** |

The artifact set contains `000216` and `000226`; it does **not** contain
`000114` or `000327`. Four of six match. Any Hard6 comparison built on the
brief's membership would compare different populations.

### 5.2 The metrics did not come from `d925c90`

| Run | Commit |
|---|---|
| Hard6 candidate v1 | `0b40423c1c1c03c280cdbe243dc3b9286fed12ea` |
| Hard6 candidate v4 | `2bcdbfca8823c5193fd2dcd9ed14c96a1ad2697e` |
| Full-13 instrumented v1 | `b0d9009` |
| Tail canary v2 | `b1d07d4d74807e58a3a63383d51a53794223cd70` |
| **Promotion on `main`** | **`d925c90`** |

`d925c90` is the promotion commit only. No metric was produced at it. The
claim "RF_ACC23 metrics came from `d925c90`" is **CONTRADICTED**.

### 5.3 Hard6 wrong-ID is 7722 → 4531, not 8579 → 5219

The Hard6 artifact records
`wrong_id_matched_animal_frames: 7722 → 4531` on the six-video subset.
The figure `8579 → 5219` appears nowhere in the RF_ACC23 series; the only
repository match for those digits is
`TRACKING_HYBRID_LANE_COMPLETION_DECISION_20260719.json`, a different lineage.
`8579 → 5219` therefore remains **UNRESOLVED** for RF_ACC23.

### 5.4 No artifact ever authorized promotion

Every RF_ACC23 decision document sets promotion to false:

| Artifact | Field | Value |
|---|---|---|
| Hard6 decision | `promote_profile` | `false` |
| Hard6 decision | `full13_authorized` | `false` |
| Full-13 runtime decision | `promotion_authorized` | `false` |
| Full-13 runtime decision | `full13_rerun_authorized` | `false` |
| Tail canary decision | `promotion_authorized` | `false` |

A search across the entire `docs/` tree of that lineage for
`"promotion_authorized": true` or `"promote_profile": true` returns **no
match**. `d925c90` (2026-07-23 10:17) promoted RF_ACC23 as the realtime default
roughly one hour after the full-13 runtime decision (09:23) and the canary
(09:02), both of which withheld promotion authority.

This is recorded as a factual governance observation. It does not impugn the
quality evidence, which passed its gate; the withheld authority was **runtime**,
not quality.

## 6. Runtime — the real reason promotion was withheld

Full-13 instrumented run against its frozen gates:

| Gate | Required | Observed | Result |
|---|---|---|---|
| mean tracking loop FPS | ≥ `36.84` | `32.38` | FAIL |
| mean frame p95 | ≤ `44.13` ms | `58.25` ms | FAIL |
| end-to-end FPS | ≥ `26.80` | `23.37` | FAIL |

The artifact attributes this to host state, not to the algorithm: GPU
temperature only 62 °C, clocks sagging to 765 MHz against a healthy
1275–1500 MHz, power draw 15–17 W against a healthy 21–23 W, and the Windows
AC power overlay set to *Best power efficiency*
(`961cc777-…`) where *Best performance* (`ded574b5-…`) was required.

The Hard6 lineage shows the same instability: identical P-core-High runs
varied between `38.01` and `29.83` FPS, and an earlier session recorded the GPU
power limit dropping `75 W → 55 W` mid-experiment.

The tail canary (two videos, clean isolated session) **passed all runtime
gates** — loop `39.27` FPS, p95 `41.43` ms, end-to-end `27.54` FPS, no thermal
throttle — which supports the host-drift interpretation rather than an
intrinsic slowdown.

**Native 30 FPS was not achieved on full-13.**

## 7. Status after recovery

| Item | Before | After |
|---|---|---|
| Full-13 IDSW / HOTA / IDF1 / FP / FN / fragments | MISSING | **ARTIFACT_VERIFIED** |
| `include_hidden`, rule combo, delay, `detect_every_n` | MISSING / partial | **ARTIFACT_VERIFIED** |
| Hard6 IDSW `55→49`, HOTA `91.317→94.487`, IDF1 `90.593→94.289` | LINEAGE_REPORTED | **ARTIFACT_VERIFIED** (on the corrected membership) |
| Hard6 membership | LINEAGE_REPORTED | **CONTRADICTED** — corrected in §5.1 |
| Metrics from `d925c90` | MISSING | **CONTRADICTED** — §5.2 |
| wrong-ID `8579→5219` | MISSING | **UNRESOLVED** — §5.3 |
| `000302` RF_ACC23 guardrail | MISSING | **PARTIAL** — listed in Hard6 `unchanged_videos`; no standalone IDSW recorded |
| Runtime winner | MISSING | **ARTIFACT_VERIFIED FAIL** on full-13; host-attributed |

## 8. Residual limits

1. The recovered lineage's tracking tree (`752d55d3…`) is **not** identical to
   the promoted tree on `main` (`8ba3d50d…`). These metrics describe the
   RF_ACC23 algorithm as run in that lineage. Byte-level equivalence between
   `b0d9009` and `d925c90` tracking semantics has **not** been proven here and
   should be established before these numbers are treated as `main`'s numbers.
2. `semantic_config_sha256` in the artifacts is
   `5af8176f6bc96529256e051a8919637abb7aec7dc8e429a41ee3521b986b3dcf`. The
   lineage-lock export on `main` computed
   `9bf4ce6d07423ab517b4705c716e3eb012349b756b7c0591cc3458eac207808d`. These
   hash **different payload structures** and are not directly comparable; no
   equivalence or divergence should be inferred from the difference.
3. The 13-video set remains a development/evaluation set.
