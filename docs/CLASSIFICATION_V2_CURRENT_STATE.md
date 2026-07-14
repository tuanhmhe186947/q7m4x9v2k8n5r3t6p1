# Classification V2 Current State

## Authority

This file is the authoritative status snapshot for the active
`classification_v2` lineage as of 2026-07-14. It records current gates, not
historical intent. When an older memory, plan, report, or runbook conflicts with
this file, use this file together with `02_CURRENT_DECISION.md` and the data
rebuild runbook.

The active objective is a leakage-safe, reproducible 10-class multimodal
spatio-temporal classifier. The accepted claim boundary is internal
recording-date/video-safe validation, not external-farm generalization.

## Active Data Lineage

The current rebuild starts from 245,664 enhanced frame/object rows:

- 172,800 `cvat_tracking_xml` rows;
- 72,864 `legacy_recovered` rows;
- 4,181 rows currently marked `Hidden=Yes`;
- 241,483 rows currently marked `Hidden=No`.

The active versioned Hidden template is:

```text
outputs/classification_v2/rebuilds/hidden_review_v6_full_20260714
```

It contains 5,131 unique, target-independent review items:

- 4,122 Hidden Yes confirmations;
- 384 high-risk Hidden No items;
- 601 stratified-random Hidden No items;
- 24 clean negative controls.

Media audit v2 is bound to the manifest and frame-context hashes. Its 24-item
dual-source smoke resolves 12 video and 12 crop items; the full gate resolves
4,613 video and 518 crop items. Both report `media_missing=0`.

Independent coverage reports zero target-derived fields, risk/stratum drift,
missing untrusted Hidden Yes items, and high-risk cap violations. The manifest
SHA256 is
`3e4fec14c466a89370a1e20d913cb024bd1dda1fa8db9c1fabdf8a51fa31072e`.

All 30 existing v5 decisions were migrated through identifier v2 and carried
into v6 with zero payload/context drift. Current coverage is 30/5,131, leaving
5,101 missing. The decision CSV SHA256 is
`7bc19943f00ca4168c9fee8af0528b4d1d69899f45f33d35022acfc28609b310`;
the root-local carry audit is
`gui/hidden_review_decision_carry_v5_identifier_v2_to_v6_audit.json`.
The report-only scientific gate is blocked; no v6 hidden-reviewed frame
artifact is final.

The existing behavior manifest contains 4,670 mandatory review units. The
current decision files contain three rows: one accept, one exclude, and one
pending. There are 4,667 missing decisions. Behavior apply is fail-closed and
must not emit a reviewed dataset from this incomplete payload.

## Technical Short-Chain Evidence

The bounded legacy+CVAT identifier-v2 chain at
`outputs/classification_v2/rebuilds/scientific_smoke_identifier_v2_20260713`
passes the machine-readable lineage and technical gates at commit `a83d5a5`:

- 688 selected, enhanced, and harmonized frame/object rows;
- 63 native intervals and 63 review units;
- 438 sequence windows, tabular-X rows, target rows, and spatial-X rows;
- all 10 behaviors and both sources represented;
- exact 110-column tabular whitelist and 73 temporal evidence fields;
- 342 trainable windows with zero missing spatial slots;
- zero duplicate temporal-unit, review-unit, or window IDs;
- eight of eight context-to-window CSVs byte-identical in an independent root;
- exact ordered sequence/image/train-ready/spatial `window_id` hash agreement.

The statuses are `PASS_IDENTIFIER_V2_TECHNICAL_HUMAN_REVIEW_BLOCKED` and
`PASS_TECHNICAL_SMOKE_HUMAN_REVIEW_BLOCKED`. They prove data generation,
identifier propagation, positional alignment, leakage separation, and bounded
determinism. They do not authorize reviewed data, training, or full OOF. The
critical alignment commits are `bfdf913` and `a83d5a5`.

Snapshot and launch lineage are hardened by `7cb4637` and `dd0e6ff`:

- snapshot v2 uses the exporter-compatible ordered-key digest;
- split, image-window, and interaction-window rows must match in count/order;
- invalid contracts cannot be frozen as immutable snapshots;
- preflight and execution recompute artifact, snapshot, lineage, and code binds;
- human authorization binds snapshot and lineage hashes, not only model config.

Temporal-view code is hardened by `bb225ff`:

- fixed-six observed-time and normalized-phase views share exact keyed slots;
- both sources reuse post-harmonization six-frame windows;
- legacy is not quantile-sampled across its native 16-frame burst;
- every source window stays in a selection ledger and every native unit stays
  in the 6/16 ablation;
- persisted rows/order/hashes and structural shortcuts fail closed.

Independent training-contract engineering is also complete in code:

- `97f83c5` fits preprocessing only from the declared training fold;
- `73b901d` balances overlapping windows by unique native-event mass;
- `16cdb93` isolates artifacts under `fold_id/run_id`, binds checkpoint schema
  v2 to code/data/cache/fold/config identity, and appends immutable run rows;
- caller checks consume the returned lineage directory instead of guessing an
  artifact path from the requested output root.
- `318bf58` adds ten exact model modes, four mask-safe temporal encoders,
  branch availability/quality contracts, checkpoint schema v3, and registry
  v2. The smoke factory downloads no weights and performs no optimizer steps.
- `07ed768` adds versioned ResNet18/ResNet34 frame encoders with exact ImageNet
  enum and normalization contracts. Random-init forwards separate ResNet18
  160/224 resolution from ResNet18/34 capacity without weight download.
- `3be22f8` adds a deterministic, synthetic-only ResNet18-160 gate. It verifies
  ten-class tiny overfit, backbone/head gradients, eval-safe BatchNorm
  recalibration, and in-memory model/optimizer resume without project data.
- `111f152` loads real ordered fixed-six timing without dropping unselected
  windows. Checkpoint schema v4 and registry v3 bind the separate temporal slot
  manifest hash; corrupt order, slot identity, masks, or timing fail closed.
- `1b6ba3d` collapses strict ten-class window probabilities to the complete
  native-unit authority, rejects fold/target/count drift, and binds paired
  recording-cluster comparisons to identical unit mappings.
- `e5d6417` registers the old full OOF and legacy sequence checkpoint with
  explicit non-promotion claim flags and independently checked hashes.

The current classification regression is 385 passed and 181 deselected. This
evidence remains synthetic/fixture-only; it has not frozen or trained the
incomplete human-review lineage.

## Active Gate Status

| Gate | Status | Evidence or blocker |
|---|---|---|
| Raw `data/` immutable | PASS | No rebuild writes under `data/` |
| Enhanced frame features | PASS | 245,664 rows audited |
| Technical legacy+CVAT chain | PASS | 688/63/438 counts; 8/8 repeatability |
| Exact model-X contract | PASS | 110 tabular fields; no review/target leakage |
| Hidden v6 template | PASS | 5,131 target-independent items; audit clean |
| Hidden v6 media | PASS | 5,131/5,131 resolved; hashes bound; zero missing |
| Hidden decision carry | PASS | 30/30 preserved; hashes and context match |
| Hidden human decisions | FAIL | 30/5,131 resolved; 5,101 missing |
| Hidden scientific gate | BLOCKED | Random/high-risk reviewed support is zero |
| Hidden decision apply | BLOCKED | Requires resolved coverage |
| Temporal rebuild from Hidden-reviewed data | BLOCKED | Upstream apply missing |
| Behavior human decisions | FAIL | 4,667 missing, one pending |
| Behavior decision apply | BLOCKED | Complete gate fails |
| Reviewed train-ready snapshot | FAIL | No complete reviewed lineage exists |
| Snapshot/preflight code contract | PASS | Ordered lineage and hash binding tested |
| Temporal-view code contract | PASS IN CODE | 22 fixture tests; active packet blocked |
| Fixed-six timing loader/lineage | PASS IN CODE | Ordered timing and hash tests pass |
| Fold-local preprocessing/weights | PASS IN CODE | Train-only fit and native-event tests |
| Run lineage/registry | PASS IN CODE | 33 focused tests; checkpoint smoke PASS |
| Model factory/mask contract | PASS IN CODE | 10 modes; 4 temporal encoders |
| ResNet18/34 backbone interface | PASS IN CODE | V0/V1/V2 forward audit PASS |
| Synthetic visual correctness gate | PASS IN CODE | 20 events; accuracy 1.0 |
| Native paired evaluation | PASS IN CODE | 31 focused tests; strict synthetic OOF PASS |
| Historical baseline control | PASS IN CODE | Registered as non-performance evidence |
| Model smoke on active snapshot | NOT RUN | Snapshot hashes are not frozen |
| Full OOF on active snapshot | NOT AUTHORIZED | Smoke and launch gates not reached |
| Q2 result claim | NOT ALLOWED | Active reviewed evaluation is absent |

## Historical Model Evidence

A full engineering OOF run exists under:

```text
outputs/classification_v2/model_full/full_multimodal_oof
```

It was produced at commit `18d6692` from the previous artifact lineage. It has
73,668 window predictions and 32,727 native temporal predictions, with accuracy
`0.5216793473` and supported macro-F1 `0.4156053847`.

This run remains useful for debugging compute, checkpoint, and execution
wiring. It is not classifier-performance evidence: commit `bfdf913` found
151,440 positional mismatches across 160,740 split-to-image and
split-to-interaction window rows. Its input lineage also predates the current
two-sided Hidden review and complete behavior-review gates. Old reports that
label it paper-facing or `PASS_PARTIAL_ROADMAP` do not authorize a Q2 claim.

Commit `e5d6417` records this boundary in:

```text
outputs/classification_v2/experiment_registry/historical_controls/
historical_baseline_reconciliation_18d6692.json
```

The audit hashes 34 artifacts totaling 527,948,648 bytes and reproduces the
known mismatch. Those hashes prove registration-time integrity only because
the origin run did not bind its input bytes. The legacy checkpoint at
`models/behavior/pig_behavior_sequence.pt` is safely inspected as a ten-output
ResNet34 architecture reference. It has no verified reviewed-data hash,
grouped split, paired predictions, training config, or seed lineage, so it is
not a model-quality baseline.

## Required Execution Order

1. Media gate is complete; rerun it only if manifest/context hashes change.
2. Complete and scientifically audit all v6 Hidden decisions.
3. Apply Hidden decisions with row/schema preservation checks.
4. Rebuild temporal harmonization and windows from the Hidden-reviewed artifact.
5. Rebuild behavior review units for the same versioned lineage.
6. Complete all behavior decisions and pass the fail-closed coverage gate.
7. Apply behavior decisions without dropping rows or overwriting enhanced data.
8. Rebuild reviewed windows, native units, X/y/masks/weights, and image indexes.
9. Build fixed-six/phase/native temporal views and pass shortcut audits.
10. Freeze data, cache, feature-whitelist, fold, and temporal-view hashes.
11. Run one-batch, tiny-overfit, resume, runtime, and one-fold smoke gates.
12. Obtain a new full-run authorization bound to frozen hashes and code SHA.
13. Run finalists only, then grouped native-unit evaluation and completion gates.

Detailed commands are in
`CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`. Script ownership
and stage order are documented in `scripts/classification_v2/README.md` and
`classification_v2_q2_multimodal_workflow.md`.

## Non-Negotiable Contracts

- Legacy native/review units are 16-frame bursts.
- CVAT anchors represent six-frame intervals `k..k+5`.
- `pig_id` is annotation-local, not cross-video biological identity.
- Training windows are created only after temporal harmonization.
- The primary view uses existing harmonized six-frame windows for both sources;
  native 6/16 length is an ablation, not an implicit primary input.
- No label, review, path, stable ID, fold ID, or target-derived field enters X.
- No random frame/window split and no overlapping windows across split roles.
- No silent row drop, silent relabel, silent trust promotion, or raw-data edit.
- Every changed semantic lineage must pass a short representative run before
  broader processing or training.
