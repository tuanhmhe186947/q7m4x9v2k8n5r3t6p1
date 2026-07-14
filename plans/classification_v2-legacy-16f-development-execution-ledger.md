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
| L2 full legacy lineage | IN_PROGRESS | L1 authorizes the data build only |
| L3 immutable inputs | NOT_STARTED | Requires deterministic full lineage |
| L4 model correctness | NOT_STARTED | Requires frozen L3 snapshot |
| L5 core baselines | NOT_STARTED | Requires L4 PASS |
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

## Full-Run Boundary

Necessary full legacy runs have standing user permission only after the exact
short semantic configuration passes. A semantic change invalidates dependent
short evidence. Canonical full OOF remains outside this goal.

## Handback Boundary

Do not mark this goal complete until L0-L8 PASS and
`legacy_16f_goal_completion_audit.json` exists with immutable hashes. After
completion, update project memory and return to the parent chat. The parent
goal must re-audit reviewed all-source P0-P8 rather than inheriting a false PASS.
