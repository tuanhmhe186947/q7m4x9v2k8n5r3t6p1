# Classification V2 Current State

## Current pre-GPU authority index (2026-08-08)

Resolve the immutable release with
`classification-v2-pre-gpu-authority-20260808^{commit}`. Do not treat a
temporary worktree SHA as execution authority. Unless stated otherwise, current
Classification V2 authorities are under
`docs/classification_v2/corrected_pooled_route_20260806/`.

- DATA / REVIEW: `post_review_20260807/updated_campaign_readiness_decision.json`
  and amended reviewed snapshot `ab86e2…9610c` are current.
- SPLIT / A12: `authority_binding.json` and E0 data references bind grouped
  split `557156…7e63b`; no fold redesign is permitted.
- TRACKING: `docs/tracking/reconciliation/CURRENT_TRACKING_STATE_INVENTORY_20260729.json`.
  RF_ACC23 is separate active tracking work, not E0 input.
- FEATURES / TEMPORAL: E0 freezes actor RGB, geometry 6D, motion 12D, their
  masks, central legacy T6 offsets 5..10, and T6/T8/T12/T16 support semantics.
- H5: `temporal_h5_20260807/temporal_h5_handoff.md` binds `COMMON_H5_T6_R`.
  H5 is five pre-target frames; H6/H12/H24 are deferred.
- POSTURE: `post_review_20260807/posture_500_completed_authority.json` binds
  500-human-gold and P0/P1; posture is OFF in E0.
- MODEL DEVELOPMENT: B0-B3, inner-only selection, finite S1, imbalance, and
  evaluation rules remain controlled by the corrected route.
- E0: `next_phase_20260806_r2/e0_execution_authority.json` and
  `e0_l4_handoff.json` bind B3/T6/FOLD_3/seed 20260804 and the 16-step pilot.
- EVALUATION: FOLD_3 outer-test data, metrics, checkpoint selection, and
  predictions are fail-closed for E0.
- EXTERNAL ARTIFACTS: `remote_e0_transfer_inventory.json` binds required
  external inputs while large artifacts remain outside Git.
- REJECTED / DEFERRED: `pre_gpu_worktree_inventory_20260808.json` preserves
  provenance; social, motion, H5, and outer-OOF candidates are not current E0.

The E0 staged lock is `next_phase_20260806_r2/e0_environment/uv.lock` with
SHA-256 `6b783d…103ca`; it is copied verbatim as package-root `uv.lock` and
used with extra `pt`. Root `uv.lock` is development-only. E0 remains an
engineering validation, not scientific model selection; no paid execution is
authorized by this index.

## S2 Top-K K=3 social branch at DEV_PASS (2026-08-04)

The isolated S2 branch is implemented at commit `a8f727a5`, its real-data
compatibility checker is committed at `5bfc7180`, and the immutable evidence
binding is committed at `6c2f2049`. The bound bundle manifest SHA256 is
`694948f570dfde2c6771efc2ad46f8a79bd0ba20ccda5768d18af631b9139119`;
the CPU compatibility report SHA256 is
`cd2eb003eb40fc2b0dae3eff0c7ce0ba2953b0fab6f9be868f8796d118f2cc37`.

The bundle contains `245,680` frame rows and `165,305` unified windows with
social tensors `[T,3,10]`. Eight T6/T8/T12/T16 by CVAT/legacy strata, finite
forward/backward, social-branch gradients, partner permutation invariance,
invalid-partner and invalid-motion isolation, tiny overfit, checkpoint reload,
and optimizer resume all pass on CPU. The final focused suite passed `62`
tests; Ruff, compile, JSON parsing, and diff checks pass.

This is implementation and bounded compatibility evidence only. No paired
S0/S1/S2 behavior comparison, GPU training, full model training, or paper
metric exists. Training-mass, reviewed RGB, and A12 gates remain unresolved.
S3/GAT remains `BLOCKED_PENDING_S2_PAIRED_SCREENING_GATE`.

## Post-review frame-label amendment V1 frozen (2026-08-03)

The operator confirmed the exact boundary for
`Pigs281119_000114_30fps`, track `2`, ID `ID_3`:

- f180-f186: `playwithtoy`;
- f187-f191: `stand`;
- last observed nose contact with the toy: f186.

The six-frame native unit f186-f191 is therefore a resolved label transition.
It must be excluded from training with sample weight zero; it must not be forced
to either class. Eight current windows contain this unit. Three were previously
trainable and are now invalid: T6 f186-f191, T8 f180-f187, and T12 f180-f191.

The non-overwriting amended engineering authority is
`outputs/classification_v2/agent_audits/`
`reviewed_engineering_snapshot_amendment_v4_1decfe4_20260803_231000`.
Its snapshot JSON SHA256 is
`ab86e2e04267cfdc8248f9bdb8774615479d67a3589f7a25844bb1a4c93a639e`.
The bound candidate manifest SHA256 is
`c9a277e2ab1088d2a43833a86a0dcc031f32870367ec9e378f4dcb8032632f03`,
and the effective reviewed-frame authority SHA256 is
`4400f36c473954784ae3d8d520eb5e1b5e79a23792d21dd0475bfb419d061a4f`.

The amended authority contains `245,680` reviewed frames and `165,305`
unified windows: `159,410` trainable and `5,895` excluded. All `12/12` target
frames match the accepted boundary; five labels changed, the six mixed-unit
frames have weight zero, and all eight affected windows are excluded with
zero effective weight. The T6/T8/T12/T16 by CVAT/legacy loader-forward checks,
finite-gradient backward check, tiny overfit, checkpoint reload, and optimizer
resume checks passed on CPU. The numeric 46D order and the 24 hashed memmap
shards remain unchanged.

Method state now marks snapshot V3 `SUPERSEDED` for future training and
`c2v2.reviewed_lineage.amendment_v1` as `FROZEN`. Amendment V1 permits local
smoke and bounded pilot/debug work only. Screening, claim-grade training,
paper metrics, and final checkpoint promotion remain blocked.

The accepted social ladder is S0 no-social, S1 current social 10D, S2 masked
permutation-invariant Top-K with K=3, then S3 small GAT only if S2 passes its
predeclared gate. S2 is authorized to implement and screen; S3 is not.

## Reviewed engineering snapshot V3 frozen (2026-08-03)

The post-review engineering pipeline is complete at code SHA
`e666d85342f794752605efdb7ce767564290c321`. The selected immutable authority
is
`C:\pig_runs\classification_v2_reviewed_engineering_snapshot_20260803_v1\snapshot_v3`,
snapshot ID `reviewed_engineering_4c430dfae2d193dc`. Earlier snapshot V1 lacks
the final bounded-smoke binding, and snapshot V2 used a short audit SHA; neither
is the selected final authority.

The final package audit is
`audits\reviewed_post_review_package_e666d85`. It validates `165,305` windows,
`159,413` trainable rows, all four window lengths and both source provenances,
zero declared grouped leakage, `24/24` spatial shard hashes, the tensor-content
hash, and eight individually hashed loader-forward smoke reports. The bounded
smoke contract also binds finite nonzero gradients, substantial tiny-overfit
loss reduction, zero checkpoint-reload delta, and zero optimizer-resume delta.
The final focused suite passed `127` tests; Ruff, explicit compile, both output
inventory audits, and the V1-to-V3 protected-authority comparison passed.

This state is engineering authority only. It is not final training or paper
metric authority. The selector remains `DEVELOPMENT_DIAGNOSTIC_ONLY` and needs
a fresh holdout. Original-versus-reviewed replay awaits an original-label-only
sidecar on identical corrected-source X and split. The posture ablation awaits
a frozen posture authority. No GPU, full training, cloud run, or final-test
access occurred. A later execution must be separately selected and registered.

## Superseded: Reviewed rebuild and low-memory spatial smokes pass (2026-08-03)

This section is retained as the pre-snapshot engineering checkpoint.

Review close now contains `3,243` reviewed units. The two final fixed-point
`explore` decisions remain unchanged by explicit user decision. The reviewed
rebuild under
`C:\pig_runs\classification_v2_reviewed_rebuild_20260802_v1` contains
`165,305` T6/T8/T12/T16 windows: `159,413` trainable and `5,892` excluded.
The frozen development split contains `134,412` train and `25,001` validation
windows, with all declared native-unit, actor, source-video, and group overlap
counts equal to zero.

Spatial tensors are materialized as 24 immutable NPY memmap shards. Their
schema hash is
`18377d825ba84974e49305e46561ada81353f9ffd0f2d2526471af1c199daad4`
and tensor-content hash is
`10c978c20fe0b3d344bd6359e574e6201225ed3ca4a4c68d9d6301d85115aa35`.
All eight T6/T8/T12/T16 by CVAT/legacy forward strata passed. Tiny overfit,
finite-gradient, checkpoint reload, and exact optimizer-resume checks passed.

Commit `a034440e93726973a3062282ede4d6b8ad0a41cc` replaces full spatial-NPZ
loading with batch-indexed memmap shards, chunked content hashing, and a
256 MiB fail-fast fallback gate. The current tensor payload is 920,418,240
bytes, so it cannot silently fall back to full-memory loading. This is
engineering validation only: no GPU or full model training ran, model metrics
are not paper authority, and the final test remains sealed. Next work is the
reviewed replay/behavior-posture ablation contract and remaining per-T audit.

## Composite review frozen; residual verification pending (2026-08-02)

This section is superseded by the reviewed-close state above and is retained
only for lineage provenance.

The completed primary, consistency-v3, and four-unit micro-review layers were
composed sequentially into one frozen authority. The composition contains
`3,082` unique reviewed units, `447` final corrections relative to the source
labels, `471` units changed at least once, and `24` units later returned to
their source label. Input-label and quality-semantic conflict counts are zero.

The current composite authority directory is:

```text
composite_behavior_review_3082_faee589_20260802_055557_v2
```

Review-informed reverse inspection of the remaining `30,273` unreviewed units
selected `39` additional suspicious units without changing their labels. The
scope has `12` HIGH units in six `fight -> move/explore -> fight` runs and `27`
MEDIUM units. It has zero overlap with the `3,082` reviewed units. Use only:

```text
post_review_residual_suspicion_3082_faee589_20260802_055557_v5
```

An independent `120`-unit control was frozen with seed `20260801`. It was
sampled without review outcomes from `30,234` residual units after excluding
all composite-reviewed and targeted units. Do not resample it. Use only:

```text
post_review_residual_control_120_3082_faee589_20260802_055557_v2
```

The required closure order is: review the targeted `39`, review the frozen
control `120`, compose both decision layers after the current composite, then
run one fixed-point audit only for newly created HIGH fight-bounded gaps.
Review selection metadata remains audit-only and must never enter model-X.
No residual finding authorizes automatic relabeling.

Train-ready export and training remain blocked until review closure and the
corrected-source evidence, bbox-derived features, adjusted-ROI features, and
T6/T8/T12/T16 windows are rebuilt under immutable hashes.

### Residual-review presentation correction

The raw targeted-39 and control-120 scopes contained only target frame indices,
so the GUI could not display longer context even in `Full context` mode. Their
selection authorities remain unchanged, but their old direct-GUI commands are
superseded by presentation-only context views:

```text
post_review_residual_suspicion_context_39_faee589_20260802_063045_v1
post_review_residual_control_context_120_faee589_20260802_063045_v1
```

For every CVAT item, the new views retain the exact six-frame decision target,
add six evenly sampled context frames on each side to the contact sheet, and
provide continuous playback up to 90 frames before and after the target. The
39-unit scope has extended context for all items. The 120-unit control has it
for 102 CVAT items; 18 legacy items retain their original 16-frame actor crops
because no trusted adjacent full-scene actor authority exists.

Both views preserve review-unit order, temporal-unit keys, target frames, input
labels, and output-session paths. Existing decisions therefore remain mapped
to the same targets and resume by `review_unit_id`. The presentation builders
did not read or write existing decisions.

## Superseded: V3 complete; four-unit micro-review remained (2026-08-02)

The targeted 697-unit v3 consistency rereview is structurally complete:
`578 accept`, `119 corrected`, no missing, duplicate, pending, or excluded
units. Decision and strength ledgers are byte-identical at SHA256
`e9294ed939dc6cb60dbc95468d9617bb13a290d219e00a21852654a72f310d77`.

The post-review continuity audit resolved 44 earlier temporal findings but
created six new HIGH non-fight-between-fight islands. Two were explicitly
accepted again in v3. Four were not reviewed in v3 and remain a bounded
micro-review scope: `Pigs291119_000216_30fps`, tracks 6 and 7, anchors 378 and
390. All four currently carry `social-nose` between neighboring `fight` units.

V3 adds 352 partner units not present in the primary 2,729 ledger. The future
composite authority therefore contains 3,081 unique reviewed units before the
four-unit resolution. Integration must translate final decisions relative to
original source labels and preserve earlier notes as provenance. It must not
overwrite the primary ledger row-for-row with v3.

Final review closure still requires the four-unit micro-review and an
independent residual-control review of at least 120 filtered-out units. The
corrected mini-CVAT source burst also requires evidence, crop, and bbox-derived
feature rebuilding before train-ready export.

## Primary Behavior review complete; consistency re-review required (2026-08-01)

The primary combined Behavior ledger has complete terminal coverage:
`2,729/2,729`, with `2,354 accept`, `375 corrected`, `0 exclude`, and no pending,
blank, or duplicate unit keys. The canonical decision and strength ledgers are
byte-identical at SHA256
`3982c7e606f54a5c8d87b795e40c2c775d20fd668213b291ea932c4fecbcc9e3`.

Primary completion is not final review closure. The first 704-unit scope was
superseded after a real f1794 case proved that nearest-at-target pairing can
replace a persistent fight partner after separation. The corrected scope uses
bidirectional partner history over the context window, ranks candidates by
temporal support, and places synchronized actor/partner units adjacently.

`outputs/classification_v2/review_authority/`
`behavior_consistency_rereview_3982c7e_20260801_203434_v2/`

The corrected scope contains 1,184 unique units. Candidate partner status is
review context only and never an automatic behavior correction. The original
ledger remained byte-identical before and after audit. Final authority still
requires this consistency review, the independent residual-control review of at
least 120 units, frozen source-correction authority, and normal post-review gates.

## Hidden complete-unit smoke contract patch (2026-07-21)

Code authority `150b2b9929b412d3882ebc118bc2432185e0987b` created operator
lineage `c2v2_human_review_20260721_reviewer01_v2`. Its source/frame rebuild
passed with 245,680 rows and exact v1 key/hash equivalence. The complete-unit
scope also passed at 704 rows and 64 native units: 32 legacy 16f plus 32 CVAT
6f units.

V2 then stopped at the Hidden complete-unit builder because the CLI inferred
full scientific support validation from the absence of row caps. It is frozen
as `STOPPED_AT_HIDDEN_COMPLETE_UNIT_SMOKE`, `FAILED_UPSTREAM_GATE`,
`NOT_RESUMABLE_AFTER_SEMANTIC_CHANGE`, `NOT_REVIEW_AUTHORITY`, and
`NOT_TRAIN_READY`. The partial manifest/context CSVs under its smoke root are
failure evidence only and must not be used or overwritten. Hidden carry-forward,
temporal full build, behavior review, and downstream sequence work did not run.
Behavior decisions remain zero; no GUI or training ran.

The replacement builder contract requires explicit
`--design-scope {smoke,full}`. Row caps only bound debug input and never select
scientific scope. Smoke preserves structural checks without final-support
quotas; full requires final support and rejects row caps. Canonical outputs are
published transactionally only after every gate passes. The next operator
lineage must use a new code SHA and a new versioned RUN_ID, proposed as
`c2v2_human_review_20260721_reviewer01_v3`; it must not resume v2.

## Active blocker-patch status (2026-07-21)

Current v1 run `c2v2_human_review_20260720_reviewer01_v1` has Hidden review
PASS at 5,240/5,240 decisions over 245,680 rows. Behavior decisions remain
0. The run stopped before the behavior GUI because annotation-consistent
windows were not sufficient scientific evidence for main training.

The existing `%SEQ0%` (`data\\04_sequence_unreviewed`) is preserved as
`PROVISIONAL_UNREVIEWED`; it is not train-ready and is not reused as `%SEQ1%`.
The patched contract requires full native-unit behavior review (legacy 16f,
CVAT 6f), frame-level apply, and full window recomputation afterward. A new
code SHA and new run ID are required. Hidden decisions may be carried forward
only through an exact key/hash audit and dry-run, never by copying a decision
CSV blindly.

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
