# Classification V2 Current State

## Authority

This file is the authoritative status snapshot for the active
`classification_v2` lineage as of 2026-07-19. It records current gates, not
historical intent. When an older memory, plan, report, or runbook conflicts with
this file, use this file together with `02_CURRENT_DECISION.md` and the data
rebuild runbook.

## Active source decision updated 2026-07-20

The next reviewed training target is a mixed lineage: the locked legacy 16f
P0-P10 export plus the exact 12 behavior XML files in
`data/annotations/classification`. The older `data/annotations/tracking`
directory is not behavior authority. The mixed source must bind input hashes,
row counts, source types and merged-output hash before review starts.

The legacy-outside-main statements below describe the prior 2026-07-19
development boundary and remain historical for those earlier experiments. They
do not authorize training: both Hidden and behavior review remain required for
the new mixed lineage.

The active objective is a leakage-safe, reproducible 10-class multimodal
spatio-temporal classifier. The accepted claim boundary is internal
recording-date/video-safe validation, not external-farm generalization.

## Human-Review Execution Contract

Hidden smoke is isolated to `%HSM%` and `%HSMDEC%`. Full Hidden decisions begin
only from `%HREV%` and are written to `%HDEC%`; no smoke row is carried into
the clean full authority.

A complete legacy 16f behavior review must build review units with
`--include-all-retained-legacy-units` and pass template coverage with
`--require-complete-legacy`. The full review manifest must contain every
retained `legacy_burst_16` native unit. A selective review queue is not complete
legacy coverage.

## Active Model-Search Sequence

The binding research order is:

1. create a strong, stable base with moderate tuning so it is a trustworthy
   measurement instrument;
2. freeze it and screen seven singles, all 21 pairs, beam-search larger subsets,
   and leave-one-out with three matched controls;
3. freeze the selected subset, then choose fusion structures from paired
   evidence for all ten behaviors;
4. jointly tune backbone, temporal model, and fusion on rented GPUs;
5. repeat matched modality ablations on the tuned strong finalist.

The first legacy screening pass is complete: A128 is best only among seven
measured bases, and seed-matched all-seven fusion shows modality signal without
beating actor-only or improving NLL. Rented-GPU joint tuning and tuned-finalist
confirmatory ablation have not run. Therefore neither naive concatenation nor
A128 is the final architecture. Keep all valid prior artifacts reusable and
rerun only after a semantic change or failed artifact audit.

The all-seven run is a diagnostic endpoint only. Pairwise, larger-subset beam,
leave-one-out, fixed-subset fusion-family search, and failure attribution remain
incomplete. No modality can be dropped until input quality, modality-only,
actor-residual, permutation, optimization and stronger-fusion probes distinguish
absence of signal from redundancy, low power, or model-capacity failure.

No exhaustive combination search is authorized on legacy 16f. It remains a
method/correctness pilot; complete subset and fusion selection waits for the
review-complete frozen main snapshot.

## Current Execution Boundary

For the active main classification branch, human review has not started and no
reviewed lineage has been handed off. Main-branch agent work therefore stops
before apply, post-review rebuild, snapshot, project-data model smoke and
training. Agent writes use only a fresh
`outputs/classification_v2/agent_audits/<AUDIT_RUN_ID>` root; the operator-owned
`human_review_workspace/classification_v2/<RUN_ID>` root is read-only to agents.

The isolated `legacy-only-unreviewed-development` lane has separate explicit
development authority and is now closed by its C6 full-development handback.
That authority never transfers training, review, OOF or Q2 permission to the
main branch.

Commits `5675235` and `ee70389` complete the generated-contract train-ready
export and independent reviewed-Q2 P0 preflight in code. They require explicit
agent-root paths, reject canonical fallback and cannot write the human root.
This is fixture-level engineering PASS, not a real positive P0 integration.
That integration waits for a clean `REVIEW_STAGE=behavior_complete` handoff.

After handoff, map, contract, model-input manifest, snapshot and P0 output must
remain under the same agent root. P0 may authorize model smoke when every
review/data/leakage gate passes, but `full_oof_authorized` remains false.

## Historical Main Classification And Legacy Lineage Boundary (2026-07-19)

Legacy 16f is not currently part of the active source manifest for the main
classification branch. The main branch must bind its own versioned source
manifest; it must not implicitly consume a legacy export or inherit a legacy
goal's authorization, PASS state, snapshot, folds, metrics, or review coverage.

The previous `legacy-only-unreviewed-development` lane was an isolated prompt/
goal-orchestration and configuration-screening lane. Its handback can nominate
configs or hypotheses for retesting, but no result transfers to the main branch
without fresh main-lineage review, data, fold, shortcut and short-run gates.

The canonical standalone legacy 16f reference is:

```text
outputs/legacy_16f_rebuild/legacy_16f_rebuild_20260718_v2
```

P0-P10 PASS. After explicit source-policy filtering, it contains 27,330 anchors,
4,555 actors, 666 groups and 72,880 export rows. Each retained actor has 16
frames and all six anchors. The canonical export SHA256 is
`fbd6300fca8fdab0b2c644626397ec6c6aa79f80b48a383f54e745cbcbcbcad3`.

This is a structural/source/lineage PASS only. It is not human-reviewed data.
Verified coverage for the new lineage remains `Hidden=0 decisions` and
`behavior=0 decisions`. Legacy 16f must still complete two-sided frame/object
Hidden review and complete native-unit 16-frame behavior review before this
legacy lineage can be called reviewed or used as train-ready evidence. These
legacy review gates are separate from the main branch and do not block or
satisfy main-branch coverage while legacy remains outside its source manifest.

The three excluded actor keys are source-quality exclusions, not review
decisions:

- `burst_color_11c02639_300 / ID_3`
- `burst_color_5532ba8c_200 / ID_5`
- `burst_color_77fe4f70_33 / ID_1`

The older mixed technical reference contains 245,664 enhanced frame/object
rows and is historical, not the expected count for a canonical rebuild:

- 172,800 `cvat_tracking_xml` rows;
- 72,864 pre-rebuild `legacy_recovered` rows;
- 4,181 rows currently marked `Hidden=Yes`;
- 241,483 rows currently marked `Hidden=No`.

The existing versioned Hidden template is a technical design reference:

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

All 30 existing v5 payload rows were migrated through identifier v2 and carried
into v6 with zero payload/context drift. The user has now confirmed that no
human review has started. These rows therefore have unverified provenance and
are not review authority, despite their embedded reviewer metadata. Verified
human coverage is 0/5,131. The forensic decision CSV SHA256 is
`7bc19943f00ca4168c9fee8af0528b4d1d69899f45f33d35022acfc28609b310`;
the root-local carry audit is
`gui/hidden_review_decision_carry_v5_identifier_v2_to_v6_audit.json`.
The report-only scientific gate is blocked; no v6 hidden-reviewed frame
artifact is final. The clean authority must use a new root under
`human_review_workspace/classification_v2/<RUN_ID>`, rebuild from immutable
source inputs, and must not carry these rows.

The existing behavior manifest contains 4,670 mandatory review units. Three
pilot payload rows exist, but the user has not verified them as human review.
Verified behavior coverage is therefore 0/4,670. Behavior apply is fail-closed
and must not emit a reviewed dataset from this payload.

## Legacy 16f C6 full-development handback (2026-07-19)

The isolated rebuild-bound lane completed temporal controls, A128 freeze,
22-mode short modality screening, paired promotion freeze, fresh full-source
actor/union features, a seven-mode full-development confirmation, paired
evaluation and final freeze. Short screening produced 44 valid run packets and
promoted only ROI and union context to the full-development confirmation.

The 22-mode screen included actor-only plus geometry, motion, ROI, numeric
social, pen, union, and full-frame branches under zero, availability-only, and
real controls. Its per-class artifacts retain all ten behaviors. Promotion of
only ROI and union was a global compute/promotion decision; it did not delete
the other five branches or establish that they cannot help a subset of
behaviors.

The full packet contains 3,650 train and 241 validation native units. Process
`8652` completed actor-only plus ROI and union zero/availability/real modes at
345 optimizer steps each. ROI real minus zero is `+0.0173996682`, with interval
`[-0.0453411623,0.0837155156]`, so it misses gain and positive-CI gates. Union
real minus zero is `+0.0317916340`, with interval
`[-0.0382937684,0.1025301139]`; its NLL worsens by `+0.1611725788` and rare
group macro-F1 changes by `-0.0444444444`.

Final artifact:
`outputs/c6fd_20260719_v6/full_matrix/`
`c6_full_development_freeze.json`. Its status is
`PASS_C6_FULL_DEVELOPMENT_FREEZE`, decision is
`RETAIN_A128_ACTOR_ONLY_FOR_C6_LEGACY_HANDBACK`, and SHA256 is
`bf7cc849e49c56458af4ea91c1824ab46b839f527e799945b64dc96fc4d86e61`.
The quality status remains
`TECHNICALLY_CLEAN_UNREVIEWED_DOUBLE_CHECK_PENDING`.

This closes only legacy C6 development. It does not alter the main source
manifest, complete Hidden or behavior review, authorize full OOF, or permit a
Q2 claim. ROI, union and A128 remain retest hypotheses for the future frozen
reviewed main lineage; the legacy result cannot reject them on full data.

The retest universe is broader than ROI and union. Geometry, motion, numeric
social, pen, and full-frame context also remain deferred hypotheses because
short point estimates show class-specific gains that a global macro-F1 gate can
hide. Main-lineage selection must report all ten classes, paired per-class
video-cluster uncertainty, source/availability strata, calibration, and harm
bounds before any behavior-conditional fusion or modality rejection.

## Technical Short-Chain Evidence

The bounded legacy+CVAT identifier-v2 chain at
`outputs/classification_v2/rebuilds/scientific_smoke_identifier_v2_20260713`
passes the machine-readable lineage and technical gates at commit `a83d5a5`:

Commit `23e2f71` fixes the identifier checker CSV loader and a fresh bounded
current-code rerun again reports
`PASS_IDENTIFIER_V2_TECHNICAL_HUMAN_REVIEW_BLOCKED`.

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
- `2bd2fda` adds the shared frozen, `layer4_only`, and optional full visual
  schedule for actor and union-context ResNets. Differential-LR optimizer
  groups remain stable across checkpoint v5, run identity v2, run manifest v2,
  and registry v4 resume boundaries. Its V0/V1/V2 audit uses zero optimizer
  steps, zero project-data rows, and no pretrained download.
- `abae856` selects checkpoints only from grouped inner-validation native-unit
  supported macro-F1, with native NLL as tie-breaker. Checkpoint v6, run
  identity v3, run manifest v3, prediction manifest v2, registry v5, and run
  audit v3 bind the policy; outer-test predictions remain evaluation-only.
- `9b04209` binds source probes to the exact trainer whitelist and ordered-window
  SHA256, collapses repeated windows to native units, and fits grouped training
  roles only. Its availability-only behavior diagnostic rejects label-gated
  masks and keeps source/readiness metadata outside classifier X.
- `111f152` loads real ordered fixed-six timing without dropping unselected
  windows. Its checkpoint v4 and registry v3 first bound the temporal slot
  manifest hash; new runs use native-selection checkpoint v6 and registry v5.
  Corrupt order, slot identity, masks, or timing still fail closed.
- `1b6ba3d` collapses strict ten-class window probabilities to the complete
  native-unit authority, rejects fold/target/count drift, and binds paired
  recording-cluster comparisons to identical unit mappings.
- `e5d6417` registers the old full OOF and legacy sequence checkpoint with
  explicit non-promotion claim flags and independently checked hashes.

The current classification regression is 429 passed and 181 deselected. This
evidence remains synthetic/fixture-only; it has not frozen or trained the
incomplete human-review lineage.

## Active Gate Status

| Gate | Status | Evidence or blocker |
|---|---|---|
| Raw `data/` immutable | PASS | No rebuild writes under `data/` |
| Enhanced technical reference | PASS | 245,664 rows audited; not clean authority |
| Technical legacy+CVAT chain | PASS | 688/63/438 counts; 8/8 repeatability |
| Exact model-X contract | PASS | 110 tabular fields; no review/target leakage |
| Hidden v6 template | PASS | 5,131 target-independent items; audit clean |
| Hidden v6 media | PASS | 5,131/5,131 resolved; hashes bound; zero missing |
| Hidden decision carry | UNVERIFIED | 30 rows preserved technically; not human authority |
| Hidden human decisions | FAIL | 0/5,131 user-verified decisions |
| Hidden scientific gate | BLOCKED | Random/high-risk reviewed support is zero |
| Hidden decision apply | BLOCKED | Requires resolved coverage |
| Temporal rebuild from Hidden-reviewed data | BLOCKED | Upstream apply missing |
| Behavior human decisions | FAIL | 0/4,670 user-verified decisions |
| Behavior decision apply | BLOCKED | Complete gate fails |
| Reviewed train-ready snapshot | FAIL | No complete reviewed lineage exists |
| Snapshot/preflight code contract | PASS IN CODE | Active-root writer/runner integration pending |
| Temporal-view code contract | PASS IN CODE | 22 fixture tests; active packet blocked |
| Fixed-six timing loader/lineage | PASS IN CODE | Ordered timing and hash tests pass |
| Fold-local preprocessing/weights | PASS IN CODE | Train-only fit and native-event tests |
| Run lineage/registry | PASS IN CODE | 33 focused tests; checkpoint smoke PASS |
| Model factory/mask contract | PASS IN CODE | 10 modes; 4 temporal encoders |
| ResNet18/34 backbone interface | PASS IN CODE | V0/V1/V2 forward audit PASS |
| Visual freeze/resume schedule | PASS IN CODE | V0/V1/V2 zero-step audit PASS |
| Synthetic visual correctness gate | PASS IN CODE | 20 events; accuracy 1.0 |
| Native checkpoint selection | PASS IN CODE | Inner native F1/NLL; outer test excluded |
| Native source/missingness probes | PASS IN CODE | Exact X/hash; 14 focused tests |
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

## Pig-STRENet causal-history artifact status

The exporter and bounded audit are implemented in:

```text
src/pig_behavior/classification_v2/features/pig_strenet_artifacts.py
scripts/classification_v2/03_image_cache_context/
classification_v2_build_pig_strenet_artifacts.py
```

The implementation exports causal history/target pairs, T0-PM controls,
stabilized actor-crop differences, all-class ROI dynamics and geometry, fixed
top-K social graphs, numeric history/transition features, model-X whitelists,
packed tensors, masks and immutable artifact hashes. Legacy relative positions
and actual source frames are separate fields. Derived-view weights conserve one
unit of native-event mass.

The current verified bounded run is
`outputs/classification_v2/agent_audits/pig_strenet_media_bridge_legacy_20260719_canary11`.
It passed 8 events, 96 slots, 288 ROI rows, fixed `K=3` social edges and an
`[8,11,32,32]` difference tensor. Actor crops came from legacy crop files;
scene ROI pixels came from the source video. Training, OOF and data
modification were all false. The input remains unreviewed-development lineage.

The media bridge also resolves XML/CVAT scene frames from the real video and
actor crops from video plus bbox when no crop file exists. It rejects static
`background.png`/`Image #1` candidates as temporal scene media and writes
`media_manifest.json` plus per-pixel provenance indexes.

A full one-video XML technical canary is
`outputs/classification_v2/agent_audits/pig_strenet_media_bridge_xml_20260719_canary02`.
It passed 2,400 pairs, difference shape `[2400,11,32,32]`, and 86,256/86,256
expected ROI pixel patches. The 144 additional XML rows are naturally missing
frame-zero history slots and are excluded from the expected pixel denominator.
Both corrected canaries have `media_manifest.valid=true` and no future-frame
use. This validates media/export lineage only; it does not authorize review,
training, accuracy claims, OOF or promotion.

The earlier XML technical canary remains preserved at
`outputs/classification_v2/agent_audits/pig_strenet_xml_real_20260719_canary01/`
`07_pig_strenet_attempt2`; its prior scene-pixel block was an exporter media
resolution limitation. The corrected canary supersedes that limitation without
deleting the old evidence.

## Required Execution Order

1. Media gate is complete; rerun it only if manifest/context hashes change.
2. Build a clean target-independent Hidden manifest under a new human-review
   root, then complete and scientifically audit its decisions from zero.
3. Apply Hidden decisions with row/schema preservation checks.
4. Rebuild temporal harmonization and windows from the Hidden-reviewed artifact.
5. Rebuild behavior review units for the same versioned lineage.
6. Complete all behavior decisions and pass the fail-closed coverage gate.
7. Apply behavior decisions without dropping rows or overwriting enhanced data.
8. Rebuild reviewed windows, native units, X/y/masks/weights, and image indexes.
9. Build fixed-six/phase/native temporal views and pass shortcut audits.
10. Freeze data, cache, feature-whitelist, fold, and temporal-view hashes.
11. Pair `SF128` against `A128` on the frozen main development folds.
12. Retest all seven modality branches with three controls and all ten classes;
    require behavior-specific uncertainty, availability strata, and harm bounds.
13. Test only predeclared behavior-conditional fusion candidates, one family at
    a time, using development predictions rather than outer-fold predictions.
14. Run one-batch, tiny-overfit, resume, runtime, and representative short gates
    after every semantic change; bounded full-development confirms survivors.
15. Lock finalist configs, seeds, folds, hashes, metrics, and compute estimate.
16. Obtain full-OOF authorization bound to the finalist and current code SHA.
17. Run finalists only, then calibration, grouped native-unit evaluation,
    confusion/ablation reports, registry registration, and completion gates.

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
