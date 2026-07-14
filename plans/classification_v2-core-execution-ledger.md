# Classification V2 Core Execution Ledger

Version: 1.0

Date opened: 2026-07-13

Starting code SHA: `ea88946`

Authority:

1. `classification_v2-core-classifier-roadmap.md` defines the P0-P8 critical path.
2. `classification_v2-scientific-performance-upgrade-roadmap.md` supplies compatible detail.
3. `classification_v2-core-scientific-execution-goal-prompt.md` is the execution contract.
4. `docs/CLASSIFICATION_V2_CURRENT_STATE.md` is the active evidence snapshot.

## Status Vocabulary

- `PASS`: every required gate has direct, current-lineage evidence.
- `IN_PROGRESS`: independent implementation or verification remains possible.
- `BLOCKED`: a declared dependency prevents completion of the phase.
- `NOT_STARTED`: no phase implementation may be promoted yet.
- `HISTORICAL_ONLY`: evidence is useful for engineering but is not active-lineage proof.

Human-review blockers never become `PASS` through synthetic data, stale artifacts, or inferred
decisions. A phase may be both `IN_PROGRESS` and partially blocked; the table records the
completion status, while its evidence records independent work that can continue.

## Canonical Phase Ledger

### P0 Data, Folds, Baseline, And Shortcuts

- Mapping: M0, M1, and M8.
- Status: `IN_PROGRESS`.
- Evidence: the 688/63/438 technical chain and exact 110-column X contract pass.
- Blocker: Hidden and behavior review are incomplete; no reviewed snapshot is immutable.

### P1-P6 Development Phases

- P1 visual, mapped to M3/M8/M9: `BLOCKED` by the unfrozen P0 snapshot and hashes.
- P2 temporal, mapped to M4/M8/M9: `BLOCKED` by P0/P1; the active reviewed
  fixed-six packet cannot exist until human review completes.
- P3 spatial/ROI, mapped to M6/M8/M9: `BLOCKED` by P2 and the active snapshot.
- P4 social, mapped to M6/M8/M9: `BLOCKED` by P3 and missing shortcut controls.
- P5 imbalance, mapped to M2/M8/M9: `BLOCKED` by the missing strong P4 model.
- P6 hierarchy, mapped to M5/M8/M9: `BLOCKED` by P1-P5 error analysis and review.

### P7-P8 Confirmatory Phases

- P7 finalist lock, mapped to M10: `BLOCKED` until P0-P6 and bounded pilots pass.
- P8 OOF/package, mapped to core M11/M12: `BLOCKED` until P7 and authorization.
- The old 13-fold run remains `HISTORICAL_ONLY`, not active-lineage evidence.

M7 five-class work and publication-only M12 work are optional P9. They cannot block or alter
the locked 10-class P0-P8 result.

## Active P0 Gate Detail

Passing gates:

- Raw `data/` immutability: `PASS`; rebuilds write only derived outputs.
- Enhanced features: `PASS`; the current artifact has 245,664 rows.
- Representative chain: `PASS`; it has 688 frames, 63 units, and 438 windows.
- Exact X whitelist: `PASS`; 110 fields pass the technical leakage audit.
- Hidden design/template: `PASS`; v6 has 5,131 target-independent items.
- Hidden media: `PASS`; 5,131/5,131 resolve under hash-bound audit v2.
- Exact CVAT resolver: `PASS`; six frames resolve to the exact `_30fps.mp4` file.

Blocked human-lineage gates:

- Hidden decisions/apply: `BLOCKED`; 30/5,131 decisions are resolved.
- Behavior decisions/apply: `BLOCKED`; 3/4,670 rows exist and one is pending.
- Reviewed frames/windows: `BLOCKED` by both upstream decision gates.
- Immutable snapshot/hashes: `BLOCKED` until the reviewed lineage passes.

Independent in-progress gates:

- Identifier code contract: `PASS`; scene and object identity are versioned as
  `scene_frame_uid` and object-level `frame_uid`.
- Hidden review key migration: `PASS`; all 5,171 v5 rows map onto identifier v2.
  All 30 decisions then carry into v6 with zero payload/context changes and
  byte-level input/output hashes.
- Identifier active-lineage rebuild: `PASS`; bounded 688/63/438 evidence has
  ordered frame/window lineage and 8/8 deterministic stage pairs.
- Snapshot/launch binding: `PASS IN CODE`; snapshot v2 and preflight v2 bind
  split/image/interaction order plus artifact, config, and code hashes.
- Temporal views: `PASS IN CODE` at `bb225ff`; fixed-six observed-time, phase,
  native 6/16, selection-ledger, order/hash, and missing-slot contracts pass.
- Structural shortcut probes: `PASS IN CODE`; source/length/padding/timing,
  quality, availability, and metadata-to-label signatures fail closed on
  fixtures. Learned embedding probes remain a later model-stage gate.
- Fold-local preprocessing: `PASS IN CODE` at `97f83c5`; fit keys, train-only
  statistics, missingness indicators, and resume hashes fail closed on fixtures.
- Native-event weighting: `PASS IN CODE` at `73b901d`; overlapping windows
  share event mass and all class statistics remain training-fold-only.
- Run lineage and registry: `PASS IN CODE` at `16cdb93`; every run is isolated
  by `fold_id/run_id`, checkpoint schema v2 binds exact lineage, and terminal
  registry rows are append-only and independently mergeable.
- Configurable model factory: `PASS IN CODE` at `318bf58`; ten model modes and
  four temporal encoders enforce exact branch, mask, shape, and lineage rules.
  Checkpoint schema v3 and registry v2 bind the explicit `model_mode`.
- Production visual backbones: `PASS IN CODE` at `07ed768`; exact ResNet18/34
  weight-enum and normalization contracts support V0/V1/V2 controlled forwards
  without pretrained download or optimizer steps during tests.
- Visual freeze schedule: `PASS IN CODE` at `2bd2fda`; actor and union-context
  backbones share frozen, `layer4_only`, and optional full stages with stable
  optimizer groups and differential learning rates.
- Synthetic visual correctness: `PASS IN CODE` at `3be22f8`; deterministic
  ResNet18-160 gradients, ten-class tiny overfit, eval recalibration, and
  in-memory resume pass without project-data access or training authorization.
- Fixed-six timing loader: `PASS IN CODE` at `111f152`; ordered real timing is
  aligned without row loss. Checkpoint schema v4 and registry v3 bind the
  separate slot-manifest hash.
- Native paired evaluation: `PASS IN CODE` at `1b6ba3d`; complete authority
  coverage, fixed ten-class metrics, fold/target reconciliation, and paired
  recording-cluster lineage pass synthetic fail-closed checks.
- Native checkpoint selection: `PASS IN CODE` at `abae856`; grouped inner
  native-unit supported macro-F1 selects checkpoints, NLL breaks ties, and
  outer-test predictions cannot tune the model. New runs use checkpoint v6,
  run identity v3, run manifest v3, prediction manifest v2, and registry v5.
- Historical baseline control: `PASS IN CODE` at `e5d6417`; the old OOF and
  legacy checkpoint are hash-registered with explicit non-promotion status.

## Independent Work Queue

Work proceeds in dependency order without fabricating human evidence:

1. Hidden target-independent clustered gate: `COMPLETED IN CODE`.
2. Complete v6 human Hidden review: `BLOCKED ON 5,101 MISSING DECISIONS`.
3. Fixed-six manifests and structural shortcut probes: `COMPLETED IN CODE`.
4. Fold-local preprocessing, event balancing, and lineage: `COMPLETED IN CODE`.
5. Configurable model/mask/shape/backbone contracts: `COMPLETED IN CODE`.
6. Real fixed-six timing loader and hash lineage: `COMPLETED IN CODE`.
7. Native-collapse and paired-evaluation contracts: `COMPLETED IN CODE`.
8. Historical baseline engineering control: `COMPLETED IN CODE`.
9. Synthetic visual one-batch/tiny-overfit gate: `COMPLETED IN CODE`.
10. Visual freeze and native checkpoint selection: `COMPLETED IN CODE`.
11. Whitelist-bound native source/missingness probes: `IN_PROGRESS`.

Item 11 must remove positional/all-numeric probe behavior, bind exact feature
order and window lineage, aggregate at native-unit grain, fit only training
roles, and keep source/availability metadata outside classifier X.

## Promotion And Full-Run Lock

The user permits a full run only after the exact short-run configuration passes. This is a
conditional permission, not a current full-OOF authorization. A semantic change to data,
cache, split, temporal view, model, loss, or resize invalidates dependent smoke evidence.

No full OOF may start until P0-P7 pass and the launch packet binds the exact code, snapshot,
cache, fold, whitelist, and config hashes to a measured one-fold runtime and explicit user
authorization.

## Achievement Log

| Date | Achievement | Result | Commit |
|---|---|---|---|
| 2026-07-13 | Core/scientific execution contract | PASS | `ea88946` |
| 2026-07-13 | Phase ledger opened | IN_PROGRESS | `475eefe` |
| 2026-07-13 | Exact CVAT video resolver gate | PASS | `97a1bc3` |
| 2026-07-13 | Identifier schema and duplicate-key guards | PASS | `a4dafed` |
| 2026-07-13 | Canonical reader/merge identifier migration | PASS | `b5d4f9d` |
| 2026-07-13 | Scene-based context grouping | PASS | `0125203` |
| 2026-07-13 | Source parser object identifiers | PASS | `c4ecfde` |
| 2026-07-13 | Scene-based social features | PASS | `8b41854` |
| 2026-07-13 | Identifier propagation through derived contracts | PASS | `9c14368` |
| 2026-07-13 | X/prediction guards and manifest-v2 audit | PASS | `1d3ab4d` |
| 2026-07-13 | Hidden review identifier and decision migration | PASS | `b5f3213` |
| 2026-07-13 | Ordered snapshot-v2 contract | PASS | `7cb4637` |
| 2026-07-13 | Preflight/execution lineage binding | PASS | `dd0e6ff` |
| 2026-07-13 | Fixed-six temporal views and shortcut contract | PASS IN CODE | `bb225ff` |
| 2026-07-14 | Fold-local preprocessing contract | PASS IN CODE | `97f83c5` |
| 2026-07-14 | Native-event fold weighting | PASS IN CODE | `73b901d` |
| 2026-07-14 | Run lineage and append-only registry | PASS IN CODE | `16cdb93` |
| 2026-07-14 | Mask-safe configurable model factory | PASS IN CODE | `318bf58` |
| 2026-07-14 | Strict fixed-six timing loader | PASS IN CODE | `111f152` |
| 2026-07-14 | Native-unit paired evaluation contract | PASS IN CODE | `1b6ba3d` |
| 2026-07-14 | Historical baseline controls | PASS IN CODE | `e5d6417` |
| 2026-07-14 | Hidden target-independent selection | PASS IN CODE | `2c0cf21` |
| 2026-07-14 | Hidden clustered scientific gate | PASS IN CODE | `6949ad0` |
| 2026-07-14 | Hidden final-support preflight | PASS IN CODE | `e9a585d` |
| 2026-07-14 | Hidden v5-to-v6 decision carry | PASS | `32eaa2b` |
| 2026-07-14 | Hidden migration artifact hashes | PASS | `aaf8460` |
| 2026-07-14 | Hidden media input-hash audit | PASS | `f2179e3` |
| 2026-07-14 | Audited ResNet18/34 backbone interface | PASS IN CODE | `07ed768` |
| 2026-07-14 | Synthetic visual tiny-overfit gate | PASS IN CODE | `3be22f8` |
| 2026-07-14 | Shared visual freeze schedule | PASS IN CODE | `2bd2fda` |
| 2026-07-14 | Native-unit checkpoint selection | PASS IN CODE | `abae856` |
| 2026-07-14 | Source/missingness probe hardening | IN_PROGRESS | pending |

## Next Checkpoint

The next checkpoint is human completion of the v6 Hidden workload. Coverage is
30/5,131 with 5,101 missing; random and high-risk reviewed support are both zero.
After review, rerun complete coverage and the scientific gate without
`--report-only`. Hidden apply remains forbidden until both pass. Behavior review
and the immutable P0 snapshot remain downstream blockers.
