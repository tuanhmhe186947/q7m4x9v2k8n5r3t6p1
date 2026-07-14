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
| L1 short packet | IN_PROGRESS | Temporal/loader PASS; cache and folds pending |
| L2 full legacy lineage | NOT_STARTED | Requires complete L1 |
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
short_temporal_tiers_v2_20260714
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
- strict real-tier loader audit valid for all eight views.

Latest completed achievement:

```text
c41f1ed test: audit real legacy temporal tier loading
```

Verification at that boundary:

- focused tests: 27 passed;
- classification regression: 442 passed, 181 deselected;
- Ruff, `py_compile`, `compileall`, diff check, and long-line scan PASS;
- optimizer steps: zero;
- full dataset reads: zero.

## L1 Remaining Gates

1. Exact selected-window to image-context slot mapping.
2. Letterbox metadata and aspect-ratio audit.
3. Cache and packed-index one-to-one coverage.
4. Strict loader proof of zero source-media fallback.
5. Recording-date/video-safe native-burst folds.
6. Window-to-native fold inheritance and class-by-fold support.
7. Independent repeated output hashes for all new L1 artifacts.

L1 completion authorizes only the full legacy data rebuild.

## Achievement Log

| Date | Achievement | Result | Commit |
|---|---|---|---|
| 2026-07-14 | Legacy development lane authorized | PASS | `a2323a7` |
| 2026-07-14 | Temporal tier controls documented | PASS | `1e9b393` |
| 2026-07-14 | Tier manifests and audits | PASS IN CODE | `ef0b3bd` |
| 2026-07-14 | Exact temporal model input binding | PASS IN CODE | `21b34fd` |
| 2026-07-14 | Absolute burst frame indices | PASS | `2049a2d` |
| 2026-07-14 | Strict short-packet loader audit | PASS | `c41f1ed` |

## Full-Run Boundary

Necessary full legacy runs have standing user permission only after the exact
short semantic configuration passes. A semantic change invalidates dependent
short evidence. Canonical full OOF remains outside this goal.

## Handback Boundary

Do not mark this goal complete until L0-L8 PASS and
`legacy_16f_goal_completion_audit.json` exists with immutable hashes. After
completion, update project memory and return to the parent chat. The parent
goal must re-audit reviewed all-source P0-P8 rather than inheriting a false PASS.

