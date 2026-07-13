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
- P2 temporal, mapped to M4/M8/M9: `BLOCKED` by P0/P1 and unlocked fixed-6 views.
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
- Hidden design/template: `PASS`; v5 has 5,171 unique review items.
- Exact CVAT resolver: `PASS`; six frames resolve to the exact `_30fps.mp4` file.

Blocked human-lineage gates:

- Hidden decisions/apply: `BLOCKED`; 30/5,171 decisions are resolved.
- Behavior decisions/apply: `BLOCKED`; 3/4,670 rows exist and one is pending.
- Reviewed frames/windows: `BLOCKED` by both upstream decision gates.
- Immutable snapshot/hashes: `BLOCKED` until the reviewed lineage passes.

Independent in-progress gates:

- Identifier code contract: `PASS`; scene and object identity are versioned as
  `scene_frame_uid` and object-level `frame_uid`.
- Hidden review key migration: `PASS`; all 5,171 workload rows and 30 existing
  decisions map one-to-one with zero human-payload changes.
- Identifier active-lineage rebuild: `PASS`; bounded 688/63/438 evidence has
  ordered frame/window lineage and 8/8 deterministic stage pairs.
- Snapshot/launch binding: `PASS IN CODE`; snapshot v2 and preflight v2 bind
  split/image/interaction order plus artifact, config, and code hashes.
- Temporal views: build audited fixed6 observed-time, phase, and native views.
- Shortcut probes: implement fold/source/length/missingness checks on fixtures.

## Independent Work Queue

Work proceeds in dependency order without fabricating human evidence:

1. Complete Hidden clustered uncertainty and target-independent prevalence gates.
2. Implement fixed-6 view manifests and source/length shortcut probes on fixtures.
3. Complete fold-local preprocessing, event balancing, and lineage registry contracts.
4. Complete configurable model/mask/shape contracts without downloading weights.
5. Complete native-collapse and paired-evaluation tests with synthetic predictions.
6. Reconcile the historical baseline only as a registered engineering control.

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

## Next Checkpoint

The next isolated achievement is a fixed-6 observed-time temporal-view contract
plus source, length, padding, and missingness shortcut probes on fixtures. It
must not train or infer human decisions. Hidden/behavior review remains the
separate blocker for freezing the reviewed P0 snapshot.
