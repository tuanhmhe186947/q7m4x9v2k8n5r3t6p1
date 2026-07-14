# Classification V2 Legacy 16F Development Execution Ledger

Version: 1.0

Opened: 2026-07-14

Scope: `legacy-only-unreviewed-development`

Goal authority:
`classification_v2-legacy-16f-development-goal-prompt.md`

Parent goal: canonical reviewed all-source P0-P8, currently blocked and not
replaced by this ledger.

## Status Vocabulary

- `PASS`: current-lineage evidence satisfies every declared gate.
- `PASS IN CODE`: fixtures prove implementation, but project-data evidence is
  still missing.
- `IN_PROGRESS`: safe implementation or verification remains.
- `BLOCKED`: an external dependency prevents further meaningful work.
- `NOT_STARTED`: prerequisite work has not passed.

## Milestone Ledger

| Milestone | Status | Current evidence or next gate |
|---|---|---|
| L0 state reconciliation | PASS | Authority, lane, counts, and claim boundary locked |
| L1 short packet | PASS | Exact cache, slot, fold, and repeat gate at `00dc2e0` |
| L2 full legacy lineage | PASS | Full repeat-bound lineage at `59647e2` |
| L3 immutable inputs | PASS | Committed-SHA gate at `0414adc` |
| L4 model correctness | PASS | Real-cache correctness gate at `3ef4235` |
| L5 core baselines | IN_PROGRESS | 4 GiB VRAM probe PASS at `93449ae`; feature caches next |
| L6 modality loop | NOT_STARTED | Requires retained L5 baseline |
| L7 imbalance policy | NOT_STARTED | Requires retained L6 candidate |
| L8 candidate/handback | NOT_STARTED | Requires controlled L0-L7 evidence |

## Settled Contracts

- Native and evaluation unit: one complete 16-frame burst.
- Input tiers: T6, T8, T12, and T16.
- Sampling: stride-3 event-balanced sliding plus one centered matched window.
- Primary metrics aggregate predictions to the native 16-frame burst.
- Grouping is recording/video safe; `pig_id` is annotation-local.
- Every artifact and metric carries the legacy-only unreviewed claim flag.
- Exact short evidence is mandatory before any full expansion.

## Current Short Evidence

Root:

```text
outputs/classification_v2/legacy_only_unreviewed_development/
short_temporal_tiers_v3_20260714
```

Observed evidence:

- 496 frame rows;
- 31 complete native bursts;
- 310 sliding windows;
- 124 centered matched windows;
- eight temporal model views;
- all 10 canonical labels;
- zero duplicate native keys;
- zero outside-native windows;
- zero missing timing slots;
- zero dropped or relabeled rows;
- event-mass maximum error `0.0`;
- all 13 repeated CSV outputs byte-identical;
- every stage 00-05 CSV and audit carries the exact unreviewed claim pair;
- strict real-tier loader audit valid for all eight views.

Latest completed achievement:

```text
aae63c3 feat: preserve legacy lineage claims end to end
```

Verification at that boundary:

- focused claim, tier, cache, and L1 tests: 47 passed;
- classification regression: 463 passed, 181 deselected;
- Ruff, `py_compile`, `compileall`, diff check, and long-line scan PASS;
- optimizer steps: zero;
- full dataset reads: zero;
- bounded video decode count: 64 for each independent cache build.

## L0 Reconciliation Refresh

- Reconciled HEAD: `fc6c594`; the implementation boundary remains `c41f1ed`.
- The three pre-existing worktree changes match the prompt's preserve list.
- Bounded packet counts, claim flags, loader status, and recorded hashes match.
- `CLASSIFICATION_V2_CURRENT_STATE.md` and the core ledger still report the
  earlier 429-test boundary; this lane's later verified boundary is 442 passed
  with 181 deselected.
- This documentation drift does not authorize training or change L1 gates.

## L1 PASS Evidence

- 496 exact `legacy_video_bbox` contexts and 310 window rows;
- 2,728 image slots with zero native-unit or frame-order mismatch;
- packed tensor shape `[496, 160, 160, 3]`, dtype `uint8`;
- zero cache misses, source-media reads, and packed pixel mismatches;
- 31 native bursts, four videos, three recording-date folds;
- zero recording-group, video, or window-fold leakage;
- exact class/source support-table reconciliation;
- all ten primary/repeat artifacts byte-identical;
- packed tensor SHA256
  `982724aa0c99852dd53805c2e1a7557daf0db9677f007e6fa32dd765eda4b105`;
- cache manifest SHA256
  `7cc08b8f2d7796a17d5294ad706cd82f6c7f6e27a3e55269d593664632732637`.
- refreshed L1 audit is bound to `aae63c37f5f16f032503ec4ef5bae966397e7396`;
- its dirty-worktree record contains only the three preserved user changes.

This PASS authorizes only the equivalent full legacy L2 data build. Model
training, canonical full OOF, reviewed/final naming, and Q2 claims remain false.
The cache/fold boundary remains `00dc2e0`; claim hardening rollback is
`git revert aae63c3`.

## Achievement Log

| Date | Achievement | Result | Commit |
|---|---|---|---|
| 2026-07-14 | Legacy development lane authorized | PASS | `a2323a7` |
| 2026-07-14 | Temporal tier controls documented | PASS | `1e9b393` |
| 2026-07-14 | Tier manifests and audits | PASS IN CODE | `ef0b3bd` |
| 2026-07-14 | Exact temporal model input binding | PASS IN CODE | `21b34fd` |
| 2026-07-14 | Absolute burst frame indices | PASS | `2049a2d` |
| 2026-07-14 | Strict short-packet loader audit | PASS | `c41f1ed` |
| 2026-07-14 | Legacy L1 cache and fold gate | PASS | `00dc2e0` |
| 2026-07-14 | End-to-end legacy lineage claims | PASS | `aae63c3` |
| 2026-07-14 | Deterministic full legacy L2 lineage | PASS | `59647e2` |
| 2026-07-14 | Immutable legacy L3 input gate | PASS | `0414adc` |
| 2026-07-14 | Legacy L4 model-correctness ladder | PASS | `3ef4235` |
| 2026-07-15 | Memory-safe L5 224px cache gate | PASS | `15a5368` |
| 2026-07-15 | Pretrained 4 GiB VRAM gate | PASS | `93449ae` |

## L2 PASS Evidence

Roots:

```text
outputs/classification_v2/legacy_only_unreviewed_development/
full_legacy_lineage_v2_20260714
full_legacy_lineage_v2_repeat_20260714
```

- 72,864 exact frame/object rows and 4,554 native 16-frame bursts;
- T6/T8/T12/T16 sliding rows are 18,216/13,662/9,108/4,554;
- 45,540 all-sliding rows and 18,216 centered-matched rows;
- all ten labels, zero row loss, label drift, cross-burst windows, or bad timing;
- all eight strict loader views pass shape, timing, observation, and mask checks;
- all 24 primary/repeat lineage CSV pairs are byte-identical and claim-safe;
- 4,545 development-valid and 9 policy-invalid bursts are retained explicitly;
- consolidated audit has zero errors and is bound to `59647e2`;
- focused gate: 51 passed; classification regression: 467 passed, 181 deselected.

The audit is `08_l2_audit/legacy_development_l2_audit.json` under the primary
root. L2 authorizes only the L3 immutable-input freeze; model training remains
unauthorized.

## L3 PASS Evidence

- 72,864 RGB actor crops use exact 160x160 aspect-preserving letterbox;
- packed tensor shape is `[72864, 160, 160, 3]`, dtype `uint8`;
- all 72,864 packed rows passed fresh pixel and loader equivalence;
- zero cache misses, loader failures, source-media reads, or pixel mismatches;
- 45,540 windows inherit 12 recording-date folds without leakage;
- all 40 frozen artifacts verified by size, hash, rows, shape, and dtype;
- primary/repeat image, fold, class, and source manifests are byte-identical;
- `quality_mask` is control-only and forbidden metadata is excluded from X;
- length-to-label uplift and maximum T6/T8/T12/T16 class drift are `0.0`;
- the committed-SHA audit is bound to `0414adc` with zero errors;
- focused L3 tests: 8 passed; classification regression: 475 passed,
  181 deselected; Ruff, compile, diff, and long-line checks pass.

The audit is `13_l3_audit/legacy_development_l3_audit.json` under the primary
root. L3 authorizes only bounded L4 model-correctness work. Accuracy/F1
comparison, canonical full OOF, reviewed/final naming, and Q2 claims remain
unauthorized.

## L4 PASS Evidence

- exact T16 centered input and `native_oof_005` are config-hash frozen;
- all 4,554 native units remain in lineage and nine policy-invalid units are
  excluded from optimizer input;
- masked-value invariance, temporal order sensitivity, and invalid-mask
  rejection pass on real packed crops;
- visual, temporal, and behavior-head gradients are finite and nonzero;
- two deterministic steps match exactly, including model and optimizer state;
- checkpoint load and the next optimizer step are exactly equivalent;
- 20 unique real native bursts reach 1.0 memorization with loss ratio
  `0.0002209130`;
- one epoch covers all 3,897 eligible train bursts and 62,352 packed frames;
- cache misses, source reads, video decodes, and train/test group overlap are 0;
- peak VRAM is 464,460,288 bytes; held-out predictions and metrics are absent;
- focused L4 tests: 4 passed; classification regression: 479 passed,
  181 deselected; Ruff, compile, diff, and long-line checks pass.

The audits are under `14_l4_model_correctness` in the primary root. L4
authorizes only controlled L5 development baselines. Canonical full OOF,
reviewed/final naming, and Q2 claims remain unauthorized.

## L5 224px Cache Gate Evidence

- frozen config SHA256 is
  `24bfca98440b2cc5983451eb639868610d724eadf89ea36db907540cb8f86e4c`;
- local GPU is declared as 4 GiB with a 70% peak-VRAM ceiling;
- ResNet18/ResNet34 frame batches are capped at 16/8 and OOM retry is forbidden;
- all 72,864 direct-source 224px letterboxed crops exist with zero failures;
- packed tensor shape is `[72864, 224, 224, 3]`, dtype `uint8`, with zero
  missing files, shape mismatches, dtype mismatches, or pixel mismatches;
- the packer uses `flush_close_reopen_each_checkpoint_v1` and records 72
  mapping reopens; the all-row audit reopens its read-only mapping 36 times;
- strict packed loading records 60/60 packed hits, zero cache misses, zero
  source-media loads, and zero video decodes or seeks;
- focused tests: 24 passed; classification regression: 490 passed,
  181 deselected; Ruff, compile, diff, and long-line checks pass;
- the full audit is
  `15_l5_core_baselines/legacy_development_l5_224_cache_full_audit.json` and is
  bound to commit `15a5368` with status `PASS_LEGACY_DEVELOPMENT_L5_CACHE_FULL`.

This gate authorized only the exact pretrained-weight and short VRAM work
recorded below. It did not authorize a model-quality claim.

## L5 Pretrained Weight And VRAM Gate Evidence

- commit `93449ae` enforces a 210-character Windows weight-path limit after the
  first long-path attempt stopped before download or CUDA initialization;
- ResNet18 V1 weight SHA256 is
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`;
- ResNet34 V1 weight SHA256 is
  `b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f`;
- CPU-only weight preparation records CUDA uninitialized before and after;
- the RTX 3050 Laptop GPU reports 4,294,443,008 total bytes;
- the allocator is hard-capped at 3,006,110,105 bytes with no OOM retry;
- V0 ResNet18-160 batch 16 peaks at 125,829,120 reserved bytes;
- V1 ResNet18-224 batch 16 peaks at 171,966,464 reserved bytes;
- V2 ResNet34-224 batch 8 peaks at 169,869,312 reserved bytes;
- all controls run two exact passes over 64 spread rows with identical feature
  hashes, 512 float32 features, zero nonfinite values, and zero OOM;
- allocated and reserved CUDA bytes return to zero after every control;
- focused visual tests: 9 passed; classification regression: 499 passed,
  181 deselected; Ruff, compile, diff, and long-line checks pass.

The weight and probe audits are under `15_l5_core_baselines`. This gate permits
the exact cached-feature short gate and, only if that passes unchanged, its full
expansion. Baseline metrics, outer-holdout predictions, reviewed/final naming,
canonical full OOF, and Q2 claims remain unauthorized.

## Full-Run Boundary

The full legacy L2 data build is complete. Any semantic change invalidates its
dependent evidence. Canonical full OOF remains outside this goal.

## Handback Boundary

Do not mark this goal complete until L0-L8 PASS and
`legacy_16f_goal_completion_audit.json` exists with immutable hashes. After
completion, update project memory and return to the parent chat. The parent
goal must re-audit reviewed all-source P0-P8 rather than inheriting a false PASS.
