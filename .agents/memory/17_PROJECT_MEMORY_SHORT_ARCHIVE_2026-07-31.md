# Project Memory Short Archive

## 2026-07-31 automatic project-state stewardship

- `project-state-steward` is the project-local session-close authority after
  material code, config, test, data-contract, or artifact changes.
- Do not default to code skills alone; use the portfolio ledger to select a
  reasoning or verification skill for nontrivial work.
- `continuous-learning-v2` is present locally, but the current configuration
  does not yet have a live observer hook or project-scoped instinct store
  initialized.
- Do not rely on chat context alone for recurring corrections.
- When a mistake or user correction matters beyond the current chat, record
  the root cause, the validated fix, and the condition under which the fix
  should be reused.
- Prefer a concise project-memory rule that captures both the error and the
  corrective method, not the error alone.

## 2026-07-29 four-method tracking reconciliation freeze

- Clean active registry is exactly `bytetrack_raw`, `hybrid_bytetrack`,
  `realtime_fast`, and `rf_hybrid`; standardized B1 and the obsolete 2x2 are
  not active scientific methods.
- `hybrid_bytetrack` remains bound to the accepted historical lineage and
  surviving final XML authority. Exact numerical runtime remains
  `RUNTIME_PROVENANCE_INCOMPLETE` and is not replaced by standardized output.
- Frozen State 8 development result is `TRANSFER_SIGNAL_MIXED`: `rf_hybrid`
  lowers IDSW `29 -> 18`, but HOTA `0.888187232 -> 0.878280776`, IDF1
  `0.971892400 -> 0.957881350`, and wrong-ID frames
  `11893 -> 14515`.
- No retuning, per-video rules, B1, unseen data, detector execution, tracker
  execution, or MP4 was used. The four-method freeze passed 370 production
  tests with 10 skips.
- State 9 authorizes only a separately scoped frozen unseen RQ2 comparison of
  `realtime_fast` versus `rf_hybrid`; unseen access in this task remains zero.

## 2026-07-29 three-mode historical prediction reconstruction

- Surviving historical prediction bytes were found and frozen for all three
  lineages. No historical tracker or detector execution was needed.
- B0/raw changed at the detector-producer boundary: historical persistent
  `model.track` at effective confidence `0.20` differs from current cached
  `model.predict` evidence at `0.25/32`. Current B0 is broadly better under the
  common Standard-V2 contract, with the explicit limitation that IDSW worsened
  from `76` to `84`.
- Historical RF_ACC23 and current R0 have identical scientific rows and
  identical Standard-V2 metrics. The apparent historical/current change is
  evaluator-only; a current-R0 improvement claim is contradicted.
- Historical H5b/H4 remains stronger than current B1 and current R0. The failed
  cache replay collapsed all `262` extra low-confidence rows before the tracker
  boundary and used `model.predict` rows instead of historical persistent
  `model.track`/ByteTrack state.
- Decision: `PASS_WITH_EXPLICIT_NOT_RECOVERABLE_LINEAGES`. A docs-only
  superseding method-freeze decision is the next bounded tracking task. Unseen
  data access, unseen evaluation, and promotion remain unauthorized.

## 2026-07-29 tracking scientific-grounding authority

- One ledger now separates historical and current method identities, effective
  configurations, metric provenance, parameter support, claims, and
  contradictions.
- Current B0/B1/R0/R1 remain a valid current-artifact development 2x2 with
  limitations. It does not contain the true historical-best hybrid artifact.
- Historical H5b/H4 is a stronger frozen Standard-V2 prediction authority than
  current R0, but executable parity is not established and irrecoverability is
  not yet proven.
- Historical improvement claims for B0/raw and R0/fast are unauthorized
  without version-specific prediction and evaluator parity.
- The unseen-method freeze remains suspended. The only next tracking task is a
  read-only B1 cache-path/post-load/tracker-boundary audit.

## 2026-07-29 historical H5b/H4 executable reproduction

- The frozen `0.20/64` cache and promoted H5b/H4 configuration reproduce
  deterministically across two complete 13-video runs: raw shapes, final
  prediction semantics, repair ledgers, rows, identities, Hidden state, and
  bboxes are identical between runs.
- Reproduction against the surviving historical XML authority fails:
  `5920` identity differences, `1959` Hidden-state differences, and `187129`
  bbox tolerance violations, with no row or frame additions/removals.
- Reproduced Standard-V2 metrics equal current B1, not historical H5b/H4:
  HOTA `0.849873389`, IDF1 `0.914081197`, IDSW `64`, wrong-ID frames `32125`,
  terminal episodes `26`, persistent swaps `11`.
- Decision: `FAIL_METRIC_PARITY`. The frozen reconstructed detector cache does
  not recover the missing historical detector-row evidence. Do not tune,
  promote, supersede the unseen freeze, or access unseen data.

## 2026-07-28 historical H5b/H4 detector-cache authority

- Frozen a non-disposable full-frame detector cache for the locked 13-video,
  23,400-frame development population at confidence `0.20`, max detections
  `64`, NMS IoU `0.50`, image size `640`, and every-frame cadence.
- The current raw detector producer is semantically equivalent for this
  detector-only scope. Historical package parity and exact missing historical
  detector-row parity remain unproven.
- All 13 cache partitions pass schema, coverage, confidence, bbox, hash, and
  replay checks. A 273-call all-video stratified repeat is exactly equal.
- Decision:
  `PASS_HISTORICAL_DETECTOR_CACHE_REPRODUCTION_FROZEN`.
- Next gate is H5b/H4 executable prediction reproduction and byte/semantic
  parity against the 13 surviving historical XMLs. Unseen access, tracker
  execution in this task, evaluation, and promotion remain unauthorized.

## 2026-07-28 historical H5b/H4 corrective evaluation

- All 23 manifest-bound artifacts from
  `20260719_h5b_h4_full13_combined_v2`, including 13 final XMLs, survive and
  were frozen byte-for-byte outside the historical worktree.
- Standard-V2 historical results: HOTA `0.900290656`, IDF1 `0.991501068`,
  IDSW `0`, wrong-ID frames `24`, terminal episodes `0`, persistent swaps `0`.
- Historical H5b/H4 is broadly better than R0 by the frozen rule.
- Repair logic matches current B1, but detector evidence differs:
  historical `confidence=0.20/max_raw_detections=64` versus current B1 cache
  `0.25/32`.
- Suspend the current unseen-method freeze pending a separate deterministic
  executable reproduction. Do not access unseen data or promote from
  prediction bytes alone.

## 2026-07-28 tracking profile registry retirement

- The active presentation-profile registry is exactly `bytetrack_raw`,
  `realtime_fast`, and `hybrid_bytetrack`.
- `realtime` is retired, not redirected; old commands must migrate explicitly
  to `realtime_fast`. `realtime_balanced`, `realtime_quality_delayed`, and
  rejected `realtime_fast_h1_r2` are historical-only.
- Closed H1-r2 owner-preference, H1-r3 shadow, and H2-CDSP shadow runtime paths
  were removed. Their locked documents and evidence remain unchanged.
- Shared causal association, offline repair, cache/replay, generic telemetry,
  evaluator, and artifact-hashing code remain available.
- Retained semantic hashes are locked in
  `docs/tracking/profile_retirement/`; no real tracking, detector inference,
  GPU inference, or evidence regeneration was performed.

## 2026-07-27 H2-CDSP scientific design frozen

- `H2_CAUSAL_DROPOUT_STATE_PRESERVATION` passed its deterministic design
  checker and independent scientific review.
- Historical RF_ACC23 evidence at `b0d9009` remains
  `MECHANISM_DISCOVERY_ONLY`; current-main failure prevalence is
  `NOT_MEASURED`, and unresolved event `000216` is excluded.
- H2 preserves a bounded same-track snapshot across short dropout. It never
  reserves, directly assigns, creates, or emits a detection or track.
- Frozen state confidence has a six-frame half-life; uncertainty grows
  monotonically; preservation is finite at ten frames; 18/18 golden cases
  pass.
- The only permitted future association consumer is one-for-one substitution
  for absent/degraded track-local evidence through unchanged ordinary
  association.
- A separate current-main side-effect-free shadow authorization is the next
  possible step. It must pass on at least two video keys and two recording
  sessions. Implementation and all tracking execution remain unauthorized.

## 2026-07-27 H1 family closed; RF_ACC23 error taxonomy

- H1-r1/r2/r3 are closed for the current study. H1-r3 ended
  `FAIL_NO_SHADOW_ACTIVATION`; its near-threshold maximum authorizes no tuning.
- The recovered RF_ACC23 full-13 evidence contains ten connected identity-error
  events, conserving 4,922 wrong-ID rows and all 53 ID-switch rows.
- `OCCLUSION_OWNER_LOSS` is dominant: six events and 2,092 wrong-ID frames
  (42.503%). Track-birth errors account for 1,619, long-hidden re-entry for
  868, and unresolved `000216` GT authority for 343.
- H1-r3 had score coverage for two of six hidden-contention events but zero
  frozen-gate activations, so assignment effect, quality, and safety remain
  unmeasured.
- Design-stage decision: propose only
  `CAUSAL_DROPOUT_STATE_PRESERVATION`, a bounded state-lifecycle hypothesis,
  not another hidden-owner score. No implementation or run is authorized.
- Validation was not executed. Runtime and promotion remain unauthorized.

## 2026-07-27 H1-r3 shadow prerequisite failed: no activation

- The reservation-disabled observer was side-effect-free and baseline outputs
  remained semantically identical on all six frozen development episodes.
- `1,518` shadow pairs were observed; `958` were core eligible and scored, but
  `0` crossed the frozen `0.625` score and `0.25` support-margin gate.
- The maximum uncalibrated `owner_preference_lower_bound` was
  `0.59954303046`; no positive or control episode had a would-activation.
- Shared-cache replay used zero detector/GPU inference, produced zero MP4s,
  and preserved causal delay zero, prefix invariance, and no-future access.
- Decision: `FAIL_NO_SHADOW_ACTIVATION`. Association implementation is not
  ready; evaluation, validation, runtime, and promotion remain unauthorized.
- The disabled diagnostic observer may remain as reusable telemetry, but it
  provides no authority to change assignments or tune the frozen gate.

## 2026-07-27 H1-r3 scientific design frozen

- Independent review: `PASS_DESIGN`; deterministic checker: `PASS`.
- Primary eligibility is symmetric IoU/recency core evidence with conservative
  lower/upper bounds for optional appearance and motion.
- There is no hidden-only overlap floor. Center and scale remain diagnostics
  with zero score weight to avoid correlated geometry double counting.
- Primary score is uncalibrated `owner_preference_lower_bound`; threshold is
  `0.625`, equivalent to relative-support margin `0.25`.
- Sixteen raw-state golden cases pass: seven realistic non-perfect hidden
  activations, two visible-support cases, and seven ambiguous/ineligible cases.
- The existing six development and four validation windows remain unchanged
  and disjoint. Validation was not executed and no validation output exists.
- Before intervention, a separately authorized reservation-disabled telemetry
  phase must test the frozen operating region using score-blind owner labels.
- H1-r3 implementation, evaluation, runtime evaluation, and promotion remain
  unauthorized.

## 2026-07-27 H1-r2 missingness audit

- The frozen development decision remains
  `H1_R2_DEVELOPMENT_DECISION=FAIL_NO_ACTIVATION`; it is not reopened.
- Row integrity passed for all `770` candidate pairs: `737` were labeled
  `missing_evidence`, `33` were scored, and `0` owner preferences applied.
- All eight feature values and the appearance/motion masks were valid on both
  sides in every row. The `737` abstentions instead exactly match a present
  hidden overlap value below the hidden-only `0.50` eligibility floor.
- No feature-plumbing defect was found. The evidence contract is too
  restrictive, and the remaining score-plus-margin operating range is
  infeasible on the frozen development evidence.
- Diagnostic lower-cutoff screening does not separate beneficial evidence from
  harmful or ambiguous evidence and authorizes no threshold change.
- `ROOT_DIAGNOSIS=MULTIPLE_DESIGN_FAILURES`.
- A principled H1-r3 design may be prepared around symmetric relative geometry
  and explicit missingness, but implementation and evaluation are unauthorized.
- Validation artifacts remained hash-identical and validation was not executed.

## 2026-07-27 H1-r2 development evaluation failed: no activation

- `H1_R2_IMPLEMENTATION_COMPLETE=YES`
- `H1_R2_DEVELOPMENT_DECISION=FAIL_NO_ACTIVATION`
- `H1_R2_VALIDATION_EVALUATION_AUTHORIZED=NO`
- `H1_R2_RUNTIME_EVALUATION_AUTHORIZED=NO`
- `H1_R2_PROMOTION_AUTHORIZED=NO`
- Frozen development population: six episodes, four positives and two
  controls, all completed through paired shared-cache replay.
- Complete H1-r2 export: `770` candidate pairs; `737` missing-evidence
  abstentions; `33` valid below-threshold pairs; `0` applications.
- No episode had real activation or association-output divergence. Equal
  metrics do not demonstrate benefit: remapped IDSW is `21` in both arms and
  wrong-ID matched frames are `135` in both arms.
- Shared cache, validation hashes, causal delay zero, prefix invariance,
  no-future-frame, and recursive zero-MP4 checks pass.
- The exact repeat was correctly not run after the frozen no-activation hard
  stop.
- Threshold `0.60`, coefficients, missingness semantics, source code,
  `realtime_fast`, and `realtime` were unchanged. The uncalibrated score is not
  a probability.
- Validation, runtime evaluation, threshold tuning, and promotion remain
  unauthorized.

## 2026-07-26 Classification V2 correction accepted

- `SCIENTIFIC_ACCEPTED_SHA=a35e0b9aae8b55167b4562cfc7e26a45e2b4e312`
- `OPERATIONAL_FINAL_EXECUTION_SHA=PENDING_EXACT_SHA_AUDIT`
- The operational stage interface is centralized in
  `configs/classification_v2/lineage_rebuild_v1.yaml` and the three lineage
  scripts under `scripts/classification_v2/`.

- `ACCEPTED_IMPLEMENTATION_SHA=a35e0b9aae8b55167b4562cfc7e26a45e2b4e312`
- `OBJECT_TRACK_KEY_EXACT_SHA_REAUDIT=PASS`
- `MAIN_SYNC_STATUS=COMPLETE`
- `PHASE1_4_INTEGRATED_ACCEPTANCE=PASS`
- `PHASE4_HUMAN_SIGNOFF=APPROVED`
- `REVIEWER=TuanHM`
- `REVIEW_DATE=2026-07-26`
- `REVIEWED_SHA=a35e0b9aae8b55167b4562cfc7e26a45e2b4e312`
- `READY_FOR_LINEAGE_REBUILD_PLANNING=YES`
- `READY_TO_REBUILD_FRAME_LOCAL=NO`
- Object key contract: `schema.classification_v2.object_track_key`,
  `classification_v2.object_track_key.v1`, RFC3986 UTF-8 escaping,
  track-first/object-fallback, `PIG_ID_AUTHORITATIVE=NO`.
- Production object-track-key values did not change.
- The V6 ignored-input and eight legacy path failures remain pre-existing
  limitations; no official rebuild or model execution is authorized.
- Current source-path authority is the completed P0-P10 run
  `outputs/legacy_16f_rebuild/legacy_16f_rebuild_20260718_v2`, the 12 behavior
  XMLs, ROI JSON, and video root recorded in README. The former
  `legacy_full_multigt_masked_nodup_16f` paths are superseded.

## 2026-07-25 Classification V2 acceptance reopened

- `MAIN_SYNC_STATUS=CODE_INTEGRATED_BUT_ACCEPTANCE_REOPENED`
- `PHASE1_4_INTEGRATED_ACCEPTANCE=REOPENED`
- `READY_FOR_LINEAGE_REBUILD_PLANNING=NO`
- `READY_TO_REBUILD_FRAME_LOCAL=NO`
- One Group A independent-checker defect reproduces at accepted SHA
  `76a0458e39769d3e7fac865dd16439a0ed3c3a04`.
- One V6 node requires Git-ignored inputs absent from fresh worktrees.
- Eight legacy-source nodes are environment/path limitations; both SHAs pass
  the full 17-node legacy file when Python may read `G:\My Drive`.
- No documentation-introduced failure was found. All execution authority
  remains false.

## 2026-07-25 historical exact-SHA Phase 1–4 acceptance

- Final accepted implementation:
  `76a0458e39769d3e7fac865dd16439a0ed3c3a04`.
- Exact-SHA Phase 4 and integrated Phase 1–4 acceptance: `PASS`.
- Human sign-off: `TuanHM`, reviewed 2026-07-24, `APPROVED`.
- Production chain: 28 production-built and 28 valid manifests, zero
  audit-assembled manifests, and 26 current-authoritative upstream links.
- Cross-root determinism and runtime dependency closure 17/17 passed.
- Negative controls passed 14/14 integrated, 15/15 builder, and 7/7 extended.
- Transaction injections passed 9/9 without a prior candidate and 9/9 with a
  prior candidate; no valid new or partial authoritative output survived.
- Phase 1–4 invariants passed. Phase 2 schema hash remains
  `ec0c511b5f5198240492be49c0492e543c9e38eb4a4ff446259b958c2a59963b`.
- That sign-off authorized planning at the time; the reopened gate above
  supersedes it. All execution authorization remains false.
- Intermediate SHAs `c1df0cc4`, `f42f0d33`, and `99d63723` remain preserved
  as superseded implementation/correction candidates.

## 2026-07-20 active mixed reviewed lineage decision

- The active training target is a mixed lineage: the locked legacy 16f
  P0-P10 export plus the exact 12 behavior XML files under
  `data/annotations/classification`, with videos resolved from `data/videos`.
- `data/annotations/tracking` is forensic/stale behavior input and must not be
  used as behavior authority. Source paths, hashes, row counts and source type
  must be recorded in the merged-source lineage manifest.
- The mixed export is not reviewed or train-ready until two-sided Hidden review,
  complete behavior review, Pig-STRENet artifact gate, snapshot and leakage
  gates pass. The older legacy-outside-main note below is historical.

## 2026-07-20 review-evidence and sampling authority

- Build Pig-STRENet review artifacts after temporal harmonization and before
  behavior review-unit selection. Join exactly by temporal_unit_key; missing or
  extra keys fail closed. Pair labels are ignored by the review bridge.
- History-to-target evidence is valid only when both history and target are
  complete. Missing history zeroes transition features; GUI shows H frames
  before T frames only when history is complete.
- Behavior review uses mandatory census, high-risk, random residual audit,
  clean control and not-selected cohorts. Random stores exact inclusion
  probability and inverse weight. Clean control or not-selected never means
  human-verified clean.
- The predeclared behavior scientific gate estimates intervention only in the
  post-mandatory/high-risk residual pool, with source/video/native-unit
  uncertainty. High-risk yield is enrichment evidence, not prevalence.
- Hidden adjacent evidence requires absolute frame-index delta 1. Persistent
  pair contact/overlap and bbox instability use only real adjacent frames and
  remain independent of behavior labels.
- Every review_pig and review sampling field is audit-only and forbidden from
  model-X.

## 2026-07-19 human-review execution correction

- Hidden smoke stays entirely on `%HSM%` plus `%HSMDEC%`; it never writes the
  full decision authority `%HDEC%`. Full Hidden review starts only from
  `%HREV%` and writes decisions to `%HDEC%`.
- A lineage claiming complete legacy 16f behavior review must build with
  `--include-all-retained-legacy-units` and check with
  `--require-complete-legacy`. The full review manifest must contain every
  retained `legacy_burst_16` native unit, including stable units with no
  evidence conflict.
- The default builder remains a selective review queue when the full-legacy
  flag is absent. Selective coverage cannot be renamed complete legacy review.

## 2026-07-19 seed-matched all-seven fusion and model-search direction

- `combined_all7__real` is one diagnostic endpoint and a stress test for naive
  concatenation. It is not a modality-subset optimization and must never be
  reported as the best fusion search.
- Subset search is staged: seven singles, all 21 pairs, predeclared beam search
  for triples and larger sets, leave-one-out around the best set and all-seven,
  then fusion-family comparisons on a frozen modality set. Every subset keeps
  parameter-matched-zero, availability-only, and real controls.
- A negative result must receive one failure code before `DROP`: bad/absent
  input, no extractable signal, actor-redundant signal, underpowered evidence,
  optimization failure, or encoder/fusion-capacity failure. Use modality-only,
  actor-residual, permutation, learning/gradient, and stronger-fusion probes to
  separate these causes and route `RETEST_FULL_DATA` or
  `RETEST_STRONGER_MODEL` when appropriate.
- Do not execute the exhaustive subset ladder to select a final model on legacy
  16f. Use this lineage only for implementation, synthetic checks, short
  representative canaries and failure-taxonomy validation. Run the full
  7-single/21-pair/beam/leave-one-out search on frozen reviewed main data.
- The measured temporal-base matrix selected `A128` only as the best of seven
  tested frozen-ResNet18 temporal bases. It is a strong controlled reference,
  not a global-best model claim and not an architecture limit imposed by the
  local RTX 3050.
- The authoritative all-seven fusion lineage uses seed `20260719`, matching
  the selected A128 run. Its actor-only arm exactly reproduces macro-F1
  `0.4016091326`, accuracy `0.6680497925`, and NLL `0.9581952707` on 241
  validation units in 32 video clusters.
- Naive all-seven real fusion reaches macro-F1 `0.3990463886`. It improves
  over same-width zero by `+0.0597244883` and availability-only by
  `+0.0475594846`, but both cluster intervals include zero. Against same-run
  actor-only it is `-0.0025627441`, interval
  `[-0.0870213782,0.0927085883]`, with lower accuracy and worse NLL.
- Preserve every earlier seed/config run as diagnostic evidence. Never delete
  it or rerun it merely because a later lineage supersedes interpretation;
  rerun only after an actual semantic change or failed artifact audit.
- Use a two-pass research flow: establish a strong controlled base; screen
  modalities/fusion with matched controls; jointly tune visual backbone,
  temporal model and fusion on rented GPUs; then repeat confirmatory modality
  ablations on the tuned finalist. RTX 3050 is a correctness host, not a
  research-capacity boundary.

## 2026-07-19 C6 modality interpretation correction

- The C6 short matrix tested all seven optional branches—`geometry`, `motion`,
  `roi`, `numeric_social`, `pen_context`, `union_context`, and
  `full_frame_context`—with parameter-matched-zero, availability-only, and
  real-value controls. All ten behavior classes remain in the per-class
  outputs, including low-support `fight`, `move`, and `playwithtoy`.
- Only ROI and union context were authorized for the legacy full-development
  confirmation. The other five branches are deferred by the global promotion
  gate; they are not deleted, disproven for every behavior, or removed from
  future reviewed-lineage hypotheses.
- Short point estimates show conditional utility: ROI is positive for
  `drink/eat/lying/move/sitting`, motion for `drink/eat/fight/social-nose/move/`
  `sitting`, numeric social for `drink/eat/fight/social-nose/lying/move`, and
  pen context for `eat/explore/lying/stand/sitting`. Union context is positive
  for `eat/lying/move/sitting`; geometry is positive mainly for `drink/move`.
  These are descriptive per-behavior signals, not promotion claims.
- The correct future question is behavior-conditional fusion on the frozen
  reviewed main lineage: measure per-class F1/precision/recall, paired
  video-cluster uncertainty, NLL/calibration, availability strata, and harm
  bounds before selecting a modality or fusion rule. Do not reduce this to a
  single global macro-F1 rejection.

## 2026-07-19 C6 legacy full-development handback

- The promotion freeze authorized only `roi` and `union_context` for one
  full-development confirmation. The fresh full packet has 3,650 train and
  241 validation native units; actor and union inputs are fixed ResNet18
  ImageNet1K V1 `.npy` features.
- Static and synthetic gates passed. The modality cache has 3,891 units and
  23,346 C6 slots, with ROI shape `[3891,6,18]`, union shape
  `[3891,6,512]`, 216 explicitly unavailable union slots, and zero media or
  outer-holdout reads during cache/training.
- Process `8652` completed all seven `full01` modes at 345 optimizer steps
  each. ROI real minus zero is `+0.0173996682`, interval
  `[-0.0453411623,0.0837155156]`; ROI misses the `0.02` margin and positive-CI
  gates. Union real minus zero is `+0.0317916340`, interval
  `[-0.0382937684,0.1025301139]`, but NLL worsens by `+0.1611725788` and the
  rare-group macro-F1 delta is `-0.0444444444`.
- Final freeze status is `PASS_C6_FULL_DEVELOPMENT_FREEZE`; both modalities
  are not retained and the handback decision is
  `RETAIN_A128_ACTOR_ONLY_FOR_C6_LEGACY_HANDBACK`. Freeze SHA256 is
  `bf7cc849e49c56458af4ea91c1824ab46b839f527e799945b64dc96fc4d86e61`.
- This remains unreviewed legacy development evidence. Hidden/behavior review,
  full OOF, Q2, reviewed/final naming, and main-branch promotion remain false;
  every conclusion requires retest on the main frozen reviewed lineage.

## 2026-07-19 main classification excludes legacy 16f

- Legacy 16f is a relatively clean standalone reference lineage, but it is not
  currently selected by the main classification source manifest.
- The completed `legacy-only-unreviewed-development` lane exercised prompt/goal
  orchestration and screened configurations or hypotheses for main-branch
  retesting. Its handback does not transfer PASS, authorization, review
  coverage, snapshot, folds or metrics to the main goal.
- Legacy 16f still requires two-sided frame/object Hidden review and complete
  native-unit 16-frame behavior review before reviewed/train-ready use.
- While legacy remains outside the main manifest, legacy review neither
  satisfies nor blocks the main branch's own review coverage.
- A main experiment must bind its own source manifest, reviewed snapshot,
  folds, hashes, shortcut audits and short-run authorization.

## 2026-07-19 C6 legacy screening result

- C6 temporal controls completed 18/18 runs (9 modes x 2 fresh repeats) on
  the rebuild lineage. Freeze artifact status is `PASS_C6_TEMPORAL_BASE_FREEZE`
  with decision `FREEZE_PRIOR_A128_FOR_C6_MODALITY_SCREENING`; freeze SHA256 is
  `150b61fddc42464d4d4767d55d5615d63507a02df6b74805c0ba85faffcf9a69`.
- C6 modality matrix then completed 22 modes x 2 fresh repeats and 14 paired
  comparisons. Decision is `PASS`; process IDs are `29324` and `5064`, with
  zero packet errors and two-sided claim flags still disabled.
- Config SHA256 is
  `3c59107ae87b1126073ede05d3eabef7437d8bd04cec988e3e6e7928709c4137`;
  cache manifest SHA256 is
  `d1af472d681224e7e37d3dbea549967f64fd7237ada77b2154a1b78c7efd7984`;
  decision SHA256 is
  `d20610127901eb8cae8f2861e7718fdd9c640a9cdefbd74b2dba428e596949bb`.
- This is technically clean but unreviewed double-check evidence: it does
  not authorize full development, reviewed/final claims, Q2, or main-branch
  classification. Actor and context inputs are fixed `.npy` feature caches;
  the run records zero source-media reads.

## 2026-07-19 legacy 16f P0-P10 rebuild PASS

- Run `outputs/legacy_16f_rebuild/legacy_16f_rebuild_20260718_v2` completed
  P0-P10 without training, OOF, raw-data edits or tracking changes.
- Filter accounting is exact: 27,665 raw CVAT rows minus five rows from three
  declared task_3 actor keys equals 27,660; minus 330 reviewed video-policy
  rows equals 27,330 retained anchors.
- The canonical universe has 4,555 actors, 666 groups and 72,880 rows. Every
  actor has six anchors at relative offsets `0,3,6,9,12,15` and 16 frames.
- The three task_3 exclusions are actor-level, not a whole-task exclusion:
  `11c02639_300/ID_3`, `5532ba8c_200/ID_5`, and `77fe4f70_33/ID_1` occur zero
  times in behavior source, P5 inputs, dense recovery and canonical export.
- P5 applies the same reviewed actor policy before authority and anchor
  coverage. Its canonical audit is `PASS` with zero errors, warnings,
  incomplete retained actors or new authority keys.
- P9 reloads raw CVAT for source hashes, then audits only retained actor keys.
  Export has zero authority discrepancy, invalid bbox or Hidden mismatch.
- Native anchor coordinates are exact in `x1_raw..y2_raw`; operational bbox
  columns clip image-bound values and report 94 clipped anchor coordinates.
- Final audit is `08_audits/legacy_16f_rebuild_completion_audit.json` with
  `status=PASS` and `errors=[]`.

## 2026-07-18 mixed CVAT XML/JSON authority

- Legacy 16f ingestion selects one annotation authority per task: XML first
  when `annotations.xml` exists, otherwise `annotations.json`. It never merges
  both formats inside one task and never falls back to stale JSON after an XML
  parse or manifest-binding failure.
- The current source binding is `task_0..task_2 = XML` and `task_3 = JSON`.
  XML `image@id` and JSON `shape.frame` must resolve to the exact image name in
  each task's `data/manifest.jsonl`.
- Behavior authority is the actor label on the lowest CVAT task frame within
  each burst, not suffix `k0`. Current data has 149/674 bursts whose first task
  frame is `k1..k4`. The selected slot/frame/name must remain explicit audit
  metadata. Slots `k0..k5` remain independent bbox/Hidden anchors.
- The read-only native-source audit loads 27,626 boxes with no missing ID,
  missing behavior or invalid bbox. It still finds two conflicting rows for
  `(burst_color_1ec47b87_14233, k4, ID_1)` in `task_2`: one `fight` bbox and
  one `eat` bbox. Full recovery remains blocked.
- First-frame audit selects 4,606 actor authorities: 4,598 have complete
  `k0..k5`, eight have incomplete anchor sets, and seven additional sampled
  actors are absent from the first task frame and must be excluded. Final
  counts must be recomputed against the fresh nodup metadata scaffold.

## 2026-07-18 legacy 16f clean-root workflow

- Seven legacy root CSV artifacts were moved, with SHA-256 manifest, to
  `outputs/_archive/legacy_16f_root_artifacts_20260718`; no raw data moved.
- No root-level legacy CSV is current input authority. Recreate provenance,
  duplicate filtering and the nodup metadata scaffold inside a versioned run.
- `exclude_source_videos.csv` is operator policy and cannot be inferred or
  replaced by an empty file. `manual_review_stable.csv` is historical review
  evidence and is not carried into the new human-review lineage.
- Use `docs/LEGACY_16F_REBUILD_FROM_SCRATCH_RUNBOOK.md`. Duplicate utilities
  now require explicit input/output paths and fail closed on unresolved keys.
- A full recovery is allowed only after unchanged input hashes, CVAT audit PASS
  and a complete one-group recovery/output check PASS.

## 2026-07-17 legacy CVAT six-anchor rebuild contract

- Rebuild legacy 16f from all native `data/data/task_0..task_3` exports; the
  old dense map is comparison evidence, not current bbox/behavior authority.
- The former fixed-`k0` behavior contract is superseded. The actor label on the
  first CVAT task frame in each burst maps to all anchors and recovered frames.
  Other labels are disagreement evidence only and never vote.
- Every CVAT bbox at `k0..k5` is an independent GT anchor. The ten intervening
  frames are recovered between anchors; a `k0` bbox is never copied forward.
- CVAT task frame indices must resolve through each task's `manifest.jsonl`;
  choose the minimum task frame separately inside each burst.
- Input audit is fail-closed on duplicate `(group_id, slot, pig_id)`, invalid
  authority behavior, invalid bbox, or slot/frame-map errors. Validation must
  prove first-frame behavior propagation, exact anchor bboxes, 16-frame
  completeness, key preservation, and no duplicates before export.
- The 2026-07-17 real-data audit is currently blocked by two duplicate anchor
  identities (four rows). Do not run recovery or model training until a new
  CVAT hash passes the short audit and one-burst recovery gate.

## 2026-07-17 C6 temporal perturbation matrix code-ready hold

- The isolated C6 temporal matrix now covers masked mean, masked TCN, small
  Transformer with real/constant/shuffled deltas, and TCN/Transformer sequence
  shuffle. `MW317` and `MW381` are capacity-matched mean controls.
- Sequence shuffle is deterministic per native-unit key and must be applied to
  every time-aligned modality and content mask. Delta shuffle preserves frame
  order and the within-unit positive-delta multiset.
- Timing claims require real-versus-constant and real-versus-shuffled controls
  to differ by the declared identifiability threshold. Uniform legacy timing
  cannot support a negative conclusion about timing utility.
- Timing-only source identification, paired native-unit evidence, per-source
  effects, recording-cluster uncertainty, and seed robustness are required
  before transfer to the frozen mixed-reviewed full-data base.
- Config `legacy_development_c6_temporal_controls_code_ready_v1.json` remains
  fail-closed. Only static/synthetic checks are authorized until a clean
  lineage handoff supplies hashes, ID, fresh output root, and authorization.

## 2026-07-17 C6 modality matrix code-ready hold

- The legacy 16f source is being cleaned by the user. Do not build the C6
  modality cache, train, or interpret existing legacy metrics until a new
  clean-lineage handoff is provided.
- The code-ready matrix is actor-only plus separate geometry, motion, ROI,
  numeric-social, pen, union, and full-frame branches. Every optional branch
  has parameter-matched-zero, availability-only, and real-value controls.
- Config
  `legacy_development_c6_modality_matrix_code_ready_v1.json` is fail-closed
  with `data_run_authorized=false`. A clean handoff requires new input hashes,
  a nonblank handoff ID, and a new versioned output root before enabling it.
- Static and synthetic gates cover 22 modes, forward/backward, missingness,
  train-only normalization, exact C6 offsets, and checkpoint resume. They read
  zero project-data rows and create zero project-data optimizer steps.
- After handoff, build the cache and run two short repeats first. Full
  development remains forbidden until paired real-versus-zero and
  real-versus-availability evidence passes the declared gate. Full OOF is not
  authorized by this matrix.

## 2026-07-17 temporal base transfer boundary

- Stage A v3 is a legacy screening packet, not a full-data base decision.
- `SF128` is the simplest carried control. `A128` is a conditional
  mixed-reviewed retest. `M128`, `TCN128`, and `TR128` are not carried from
  the legacy expansion.
- The packet SHA256 is
  `b3250ed5391d46e37469a22f16353bbc5f038fa250897c37056fe64a132a6910`.
- A final base must be chosen on frozen mixed-reviewed native units with
  paired source, missingness, group-metric, support, and uncertainty reports.
- A legacy gain cannot promote a model when it is source-specific or driven by
  availability, unsupported classes, or an unmatched protocol.

## 2026-07-17 legacy one-sequence temporal sampling decision

- Keep each complete legacy 16-frame burst as the native evaluation unit, but
  do not use contiguous T16 as a model input in this comparison.
- The controlled views are C6 offsets `5-10`, C8 offsets `4-11`, and S6
  offsets `0,3,6,9,12,15`, with exactly one sequence per native unit.
- S6 versus C6 is the primary span-at-fixed-length comparison. C8 versus C6
  is the secondary sequence-length comparison. Cache, native units, model,
  loss, seed, epochs, and 345 optimizer steps are identical.
- Full development macro-F1 is C6 `0.3708555386`, C8 `0.3588478457`, and S6
  `0.3334808033` on 245 native units and 33 video clusters.
- S6 minus C6 is `-0.0373747353`, interval
  `[-0.0871566209, 0.0263536824]`. C8 minus C6 is `-0.0120076929`, interval
  `[-0.0414907389, 0.0171363991]`.
- Retain C6 as the one-sequence legacy working view. S6 has descriptive gains
  for `drink` and `move`, but their class intervals include zero and `move`
  has only eight units. C8 improves NLL but not macro-F1.
- Historical sliding T6 remains context only because it uses four windows per
  native unit and different optimizer exposure; it is not a paired control.
- Decision SHA256: `cdd24a27162ec46bc68214e6820e3aa41aebe86da53acd6903da175bcced2cfa`.
  This is unreviewed legacy development evidence, not Q2 or full-OOF evidence.

## 2026-07-17 pen-boundary context candidate

- The fixed-camera pen mask is suitable as label-independent spatial context,
  but it is camera calibration rather than a biological invariant.
- Enhanced frame features bind the mask hash and derive signed boundary
  distance, bbox-inside ratio, near-boundary state, approach/retreat speed and
  parallel speed without dropping rows.
- Spatial group `pen_boundary_context` is not in the current trainer whitelist
  or full model. Its first legal promotion comparison is paired
  `actor_geometry_motion` versus `actor_geometry_motion_pen`; both receive the
  same generic motion group and all other variables stay fixed.
- Mask path/hash, availability, quality, inward normals, `pen_center_inside`
  and binary `pen_near_boundary` are audit or derivation only. They must not
  enter model-X.
## 2026-07-20 Fast promotion and Pareto refresh

- The far-right visible close-competitor guard is promoted in
  `realtime_fast` at profile commit `74cad2b`; algorithm commit is `62f140b`.
- Full-13 primary/repeat improve IDSW `69 -> 59`, HOTA `94.35% -> 95.63%`,
  IDF1 `93.91% -> 95.37%`, FP/FN `506/630 -> 486/610`, with no per-video
  IDSW regression. `000302` improves `6 -> 0`.
- Raw remains an immutable authority (`145/88.91%/88.47%`); it is not rerun
  because its code, inputs and evaluation contract are unchanged.
- Fast is promoted as a causal quality reference, not a speed winner. Native
  throughput/common-harness gates remain pending. Balanced remains a
  non-dominated quality trade-off; Quality remains delayed/negative evidence.
- A Balanced far-right guard screen was rejected at two frozen windows: both
  parent/candidate pairs tied at IDSW `0`, HOTA/IDF1 `100%`, FP/FN `0/0`.
  No Balanced default changed and no later stage was opened.
- Authority: `docs/TRACKING_REALTIME_PARETO_SELECTION_DECISION_20260720.json`.

## 2026-07-19 realtime selection is multi-objective

- A realtime winner is not the profile with the highest HOTA/IDF1 or lowest
  IDSW alone. It must first pass causal/fixed-delay, prefix, repeatability,
  lineage, memory and zero-MP4 gates.
- The Pareto comparison then includes identity, HOTA/IDF1, FP/FN, fragments,
  effective FPS, loop-FPS, p50/p95 latency, stage timing, delay and deployment
  cost. A speed claim requires comparable primary/repeat runtime measurements.
- Raw authority is now PASS: IDSW `145`, HOTA `88.91%`, IDF1 `88.47%`,
  loop-FPS `22.65/27.03`, repeatability PASS, and `mp4_count=0`.
- Fast remains the current causal reference (IDSW `69`), but its profile guard
  is open because `000302` has IDSW `6` versus the frozen ceiling `2`. The final
  operational winner therefore remains pending both a Fast guard follow-up and
  a common-harness runtime audit. Balanced and Quality retain their measured
  trade-offs and cannot be judged by accuracy alone.
- Authority:
  `docs/TRACKING_REALTIME_PARETO_SELECTION_DECISION_20260719.json`.

## 2026-07-19 RQ4 closure and realtime Quality role

- Commit `1a1d008` replaces motion-pair `deepcopy` with schema-aware cloning.
  Five global/fixed-lag reference variants remain output-equivalent; the real
  artifact benchmark improves from median `1.4471 s` to `0.6514 s` (`2.221x`).
- QW01 primary/repeat preserve IDSW `32`, HOTA `94.75%`, IDF1 `95.94%`,
  FP/FN `5/6`, and semantic hash `8472ed87...86857`.
- End-to-end runtime still fails: effective FPS is `19.42/16.91`, p95 is
  `60.84/91.02 ms`, and repeat loop-FPS ratio is `0.856`. The paired parent
  also misses the frozen floor at `23.64 FPS` and p95 `47.41 ms`; this does
  not waive the candidate gate.
- Retain the output-equivalent implementation improvement, but do not promote
  finite-delay Quality or run later windows, video, hard set, or full-13.
- Quality was a mandatory realtime challenger, not an optional final step.
  Its finite-delay screen is now closed as invalid; delay `-1` remains only a
  post-video upper bound. Proceed to Pareto selection among valid authorities.
- Authority is
  `docs/TRACKING_RQ4_QUALITY_COPY_PERFORMANCE_DECISION_20260719.json`.

## 2026-07-19 realtime Pareto selection

- RB1 and RB2 Balanced candidates were rejected at frozen window 1 because
  lower IDSW came with IDF1/HOTA regression. RB3 improved IDSW `121 -> 111`
  but failed repeat p95 and the Balanced identity target.
- Fast remains the lowest-IDSW causal reference: IDSW `69`, HOTA `94.35%`,
  IDF1 `93.91%`, FP/FN `506/630`, with repeatability PASS. Its `000302` guard
  is open (`6` versus ceiling `2`), and its measured timing has no speed claim.
- Balanced remains a valid causal quality reference with HOTA `95.68%` and
  IDF1 `95.76%`, but IDSW `121` is `52` above Fast and misses the locked
  Balanced target `<=86`; it is not the operational winner.
- Quality was evaluated as the mandatory challenger. Delay `-1` is upper-bound
  evidence only, and no finite-delay candidate passed all runtime gates.
- Retain `realtime_fast` as the current causal reference after raw authority
  PASS, but do not call it the final winner until the `000302` guard and common
  runtime audit pass. Balanced and Quality stay in comparison evidence.
- Authority is
  `docs/TRACKING_REALTIME_PARETO_SELECTION_DECISION_20260719.json`.

## 2026-07-19 RQ3 rejection and RQ4 performance gate

- RQ3 memory horizons `10/15/20` all preserve the RQ2 S1 prediction and QW01
  quality: IDSW `32`, HOTA `94.75%`, IDF1 `95.94%`, and FP/FN `5/6`.
- Memory 10 has the lowest postprocess cost, but its clean run reaches only
  `17.20 FPS` with p95 `72.21 ms`, below the frozen `24.08 FPS` and
  `45.29 ms` gates. Even its optimistic paired-loop ceiling is `21.24 FPS`.
- RQ3 is rejected at QW01. Do not run its later windows, full video, hard set,
  or full-13. All RQ3 roots have zero MP4.
- RQ4 is frozen as output-equivalent implementation work: replace expensive
  motion-pair shape `deepcopy` operations with schema-aware cloning, require
  exact synthetic and real-artifact equality, then rerun only QW01 runtime.
- Authorities are
  `docs/TRACKING_RQ3_QUALITY_RUNTIME_HORIZON_DECISION_20260719.json` and
  `docs/TRACKING_RQ4_QUALITY_COPY_PERFORMANCE_PLAN_20260719.json`.

## 2026-07-19 RQ2 Quality decision and RQ3 runtime gate

- RQ2 S1 (`min_gain=0.02`, two-ID component gate) improves QW01 from IDSW
  `36 -> 32`, HOTA `91.51% -> 94.75%`, and IDF1 `91.17% -> 95.94%`, with
  FP/FN unchanged at `5/6`.
- S2 regresses to IDSW `42`, HOTA `90.70%`, and IDF1 `87.56%`; S3 ties the
  parent. On QW02-QW04 S1 has no material gain, so only one independent episode
  improves and the two-episode funnel gate fails.
- S1 is also below the frozen realtime promotion floor at `18.99 FPS` versus
  `24.08 FPS`; every RQ2 root has recursive `mp4_count=0`. Do not run its full
  video, hard set, or full-13. Decision authority:
  `docs/TRACKING_RQ2_QUALITY_ID_SAFE_DECISION_20260719.json`.
- RQ3 is frozen as a single-family memory-horizon screen (`10/15/20` frames)
  on QW01 only, keeping the S1 identity gate and fixed lag 15. It is the next
  Quality action; no profile default changes before promotion evidence.

## 2026-07-19 RQ1 rejection and RQ2 Quality gate

- RQ1 fixed-lag implementation is commit `f21775f`; global lag `0`, Fast, and
  Balanced remain unchanged, and 199 tracking tests pass.
- On frozen Quality QW01, lags `12/15/30` all reduce IDSW `36 -> 32` but also
  reduce HOTA `91.51% -> 91.12%` and IDF1 `91.17% -> 89.88%`. Reject RQ1 at
  QW01; do not run its later windows, video, hard set, or full-13.
- All four QW01 roots have MP4 count zero. RQ1 changed 61 payload rows over
  frames `809-838` and `863-878`, showing an unsafe persistent ID5/ID6 relabel.
- RQ2 was the completed Quality experiment. Its S1 signal passed QW01 quality
  but failed the runtime/two-episode promotion gate; RQ3 is the active
  continuation under a separately frozen authority.
- Authorities are `docs/TRACKING_RQ1_FIXED_LAG_DECISION_20260719.json` and
  `docs/TRACKING_RQ2_QUALITY_ID_SAFE_PLAN_20260719.json`.

## 2026-07-19 paper priority and profile-redesign lock

- The dependency order `hybrid_bytetrack` -> `realtime_fast` ->
  `realtime_balanced` -> `realtime_quality` controls experiment sequencing,
  not the final realtime ranking. Hybrid is complete; fast is the current
  causal reference. The bounded RB3 decision is closed without profile
  promotion; Quality is now the active mandatory challenger.
- Final realtime selection must compare every valid Fast, Balanced, and Quality
  authority. If finite-delay Quality is Pareto-best, it becomes the paper's
  selected realtime method and the main table is raw -> Quality -> hybrid.
- The paper-critical table needs only same-contract `bytetrack_raw`, one
  selected causal realtime method, and `hybrid_bytetrack_best`. Three realtime
  modes are optional and must earn inclusion through distinct scientific value.
- A weak or semantically misleading profile implementation may be superseded
  or rebuilt. Preserve its old artifacts as labeled evidence; the replacement
  must pass frozen causal, quality, latency, lineage, and zero-MP4 gates against
  the relevant raw and operational references.
- Quality remains eligible to become the paper's selected realtime method if a
  causal or finite-delay version passes prefix/latency gates and wins the
  declared Pareto comparison. The current delay-`-1` global-post-video profile
  can be reported only as delayed/post-video evidence, not as a realtime winner.
- Authority is
  `docs/TRACKING_PAPER_PRIORITY_AND_REDESIGN_POLICY_20260719.json`.

## 2026-07-19 realtime quality selection gate

- `realtime_quality` must be evaluated before locking any realtime winner. The
  paper may still show fewer than three realtime profiles, but profile-table
  omission is a presentation decision, not permission to skip Quality.
- The current Quality result (`IDSW 166`, HOTA `97.66%`, IDF1 `97.58%`) is a
  useful post-video upper bound only: it uses a global graph, delay `-1`, and
  about `12.09 FPS`. It cannot be called the realtime winner.
- RB3 quality is repeatable (`IDSW 121 -> 111`), but its repeat p95
  `60.06 ms` fails the frozen `45.29 ms` gate, so it remains opt-in. Rebuild
  Quality from the last promoted causal base and test rolling fixed-lag
  candidates at `12/15/30` frames with identity, runtime, and zero-MP4 gates.
- If a valid finite-delay Quality profile is Pareto-best, select it for the
  paper; otherwise retain the best valid causal profile and label the old
  global-graph result as delayed evidence.
- The detailed authority is
  `docs/TRACKING_REALTIME_QUALITY_SELECTION_GATE_20260719.json`.
- The RB3 decision authority is
  `docs/TRACKING_RB3_RESERVED_REID_HOLD_DECISION_20260719.json`.

## 2026-07-18 H4 far-camera component decision

- H4 fixes the `000328` far-right Hidden bbox family: full-video IDSW
  `4 -> 0`, HOTA `98.317% -> 98.476%`, IDF1 `99.125% -> 99.215%`,
  and remapped FP/FN `126/126 -> 113/113`.
- The four-video hard set improves aggregate IDSW `8 -> 4` with no
  regression, but only `000328` changes. The frozen two-video gain gate
  blocks H4-only full-13 and profile promotion.
- H4 remains a proven component. H5 for the independent `000233`
  identity-payload family is active next; realtime remains closed.
- Decision authority is
  `docs/TRACKING_H4_FAR_CAMERA_GEOMETRY_DECISION_20260718.json`.

## 2026-07-18 hybrid residual-first correction

- Promoting one hybrid candidate closes only that experiment. It does not mean
  that the `hybrid_bytetrack` lane is complete or authorize realtime work.
- The promoted include-Hidden authority still has eight IDSW: four on `000233`
  at frames `1111-1114` and four on `000328` at frames `1347-1355`.
- `000328` is currently classified as a geometry/occlusion-hold failure: IDs
  remain stable, but the far-right Hidden `ID_7` bbox expands over visible
  `ID_5`. `000233` is a separate identity-payload conflict.
- Continue hybrid through isolated residual families and the staged funnel.
  Open realtime only after a separate hybrid lane-completion decision passes.
- The active frozen plan is
  `docs/TRACKING_HYBRID_RESIDUAL_PLAN_20260718.json`.

## 2026-07-18 hybrid near-wall geometry promotion and priority lock

- The critical path is `hybrid_bytetrack` first. Realtime transfer is a
  separate later experiment; do not tune realtime in the hybrid experiment.
- Near-wall Hidden bbox geometry is promoted in `hybrid_bytetrack_best`.
  Full-13 primary/repeat preserve IDSW `8`, improve FP/FN `1630 -> 1622`,
  HOTA `98.31% -> 98.32%`, and raw IDF1 despite both reports rounding to
  `99.13%`.
- Exactly 111 bbox rows change on three videos. IDs, shape keys, Behavior,
  `Hidden`, `occluded`, and every other non-geometry payload remain equal.
- Repeatability authority is `PASS`: 26 semantic predictions, 46 artifacts,
  input rehash enabled, tracker runtime `NOT_APPLICABLE`, and zero MP4.
- Algorithm, auditor, and profile commits are `3391dbd`, `b66428e`, and
  `4876217`. Full lineage and negative evidence are in
  `docs/TRACKING_PROMOTION_DECISION_20260718_HYBRID_NEAR_WALL_GEOMETRY.json`.
- For the later realtime lane, use `realtime_fast` as the operational
  reference. Balanced must pass a predeclared identity-stability/latency gate
  and add material value versus fast; improving only over old balanced is not
  sufficient.

## 2026-07-18 realtime_balanced hidden-reservation promotion

- `realtime_balanced` promotes causal hidden-detection reservation with
  `min_iom=0.96`, `min_gain=0.17`, `max_alternative_cost=0.25`, visible hold,
  and `hold_min_gain=0.17`.
- Full-13 improves IDSW `133 -> 121`, IDF1 `93.71% -> 95.76%`, HOTA
  `93.93% -> 95.68%`, FP/FN `449/587 -> 448/586`, and fragments `130 -> 127`.
  Five videos improve and eight tie; none regress in IDSW, IDF1, or HOTA.
- `000231` has the declared FP/FN trade-off `61/79 -> 62/80`, while IDSW
  improves `28 -> 26`, IDF1 `76.06% -> 93.92%`, and HOTA
  `77.29% -> 92.97%`.
- Reject `max_alternative_cost=0.30`: despite aggregate IDSW `119`, it drops
  `000216` IDF1/HOTA to `90.10%/91.45%` through a persistent ID 5/8 error.
  Threshold `0.25` blocks cost `0.283780` and retains the useful `000233`
  event at cost `0.238421`.
- Primary/repeat metrics and semantic predictions match; authority is `PASS`
  and all roots contain zero MP4. Promotion commit is `e8d39d7`; lineage and
  rollback are in
  `docs/TRACKING_PROMOTION_DECISION_20260718_REALTIME_BALANCED_HIDDEN_RESERVATION.json`.

## 2026-07-18 realtime_fast visible-competitor preference promotion

- Full-13 primary and repeat use `include_hidden=true`,
  `iou0_area0_condarea0_merge0`, causal framewise output, and no MP4. Their
  prediction semantic hashes are identical for all 13 videos.
- Promoting `realtime_visible_better_competitor_prefer=true` in the named
  `realtime_fast` profile reduces aggregate IDSW `87 -> 69`, HOTA
  `93.89% -> 94.35%`, IDF1 `93.21% -> 93.91%`, FP/FN `564/688 -> 506/630`,
  and fragments `114 -> 110`.
- The gain is concentrated in `000231` (`IDSW 30 -> 12`); the other 12 videos
  are unchanged and no video regresses in IDSW. This is accepted as an overall
  tracking improvement; the per-video concentration remains a risk to test on
  newly sampled sessions.
- Profile promotion is commit `456fc97`. The authoritative decision,
  lineage, rollback, and runtime claim boundary are in
  `docs/TRACKING_PROMOTION_DECISION_20260718_REALTIME_FAST_VISIBLE_PREFER.json`.

## 2026-07-18 hybrid far-camera identity guard

- Commits `7254670` and `e74a8fa` add and separately promote a far-camera veto
  for mixed detected/occlusion-hold identity swaps in `hybrid_bytetrack`.
- The camera contract is left-near/clear/large and right-far/blurred/small;
  the promoted threshold is normalized center-x `0.67`. Wall behavior remains
  outside the far-only veto and the scene wall source is `mask.png`.
- Full-13 primary and repeat both improve IDSW `10 -> 8` with no per-video
  IDSW regression. HOTA/IDF1 stay `98.31%/99.13%` at report precision.
- The raw trade-off is FP/FN `1628 -> 1630`, accepted as an overall tracking
  improvement. Keep the `000302` local geometry delta as a separate next
  ablation target; do not hide or reinterpret it.
- All target, guardrail, primary, repeat, and prediction roots have zero MP4.
  The authority and rollback are in
  `docs/TRACKING_PROMOTION_DECISION_20260718_HYBRID_FAR_IDENTITY_GUARD.json`.
## 2026-07-16 agent-output isolation for reviewed Q2 rebuild

- While human review has not been handed off, agent work is limited to static,
  synthetic and read-only audits.
- Agent-owned writes use a unique
  `outputs/classification_v2/agent_audits/<AUDIT_RUN_ID>` root.
- Human review remains operator-owned under
  `human_review_workspace/classification_v2/<RUN_ID>`; the agent must not open
  its GUI or write apply/rebuild artifacts there.
- After handoff, generated map, `data_contract.json`,
  `model_input_contract.json`, `snapshot.json` and P0 audit remain under the
  same agent root. Never fall back to canonical output folders.
- A short/synthetic gate is required before each semantic full run; no full
  training or full OOF is authorized by this isolation rule.

## 2026-07-16 clean human-review authority reset

- The user confirms that no Hidden or behavior human review has started.
- Existing 30-row Hidden and 3-row behavior payloads are unverified pilot/
  legacy artifacts, even where embedded metadata names `ironh`. Do not carry,
  apply, or count them as human evidence.
- New reviewed lineage starts from zero decisions under
  `human_review_workspace/classification_v2/<RUN_ID>`.
- Agent audits must use
  `outputs/classification_v2/agent_audits/<AUDIT_RUN_ID>`; while review is
  active and after handoff, agents may read but must not write the selected
  human root. Downstream agent output uses another versioned root.
- The runbook and current-state authority were corrected on 2026-07-16. Human
  handoff requires exact `RUN_ID`, review stage, reviewer ID, and file hashes.

## 2026-07-16 legacy_16f L0-L8 goal completion

- Commits `6798d8b` and `fca9c80` lock the bounded T6 event-balanced
  development candidate, its checkpoint/prediction/metric links, model card,
  experiment matrix, ablation registry, and rejected-experiment registry.
- The candidate has native macro-F1 `0.5343181014`, accuracy `0.6857142857`,
  NLL `1.1206917661`, and a 33-video macro-F1 interval
  `[0.3893135266, 0.5734933334]` on 245 development-validation native units.
- The L8 candidate lock SHA256 is
  `b91949711e15c493a07375c4f7fa5f44535220dfdbac68f095d2effee4be6ba6`.
- Commit `91a6c2a` adds the fail-closed L0-L8 completion checker. The handback
  `legacy_16f_goal_completion_audit.json` has SHA256
  `4b6bad32834fbede2001dee5627e5fbfa0005afb758f2c6a3cbfb125be3166f6`.
- L0-L8 are PASS only for `legacy-only-unreviewed-development`. Human review,
  reviewed/final naming, canonical full OOF, outer-holdout prediction, and Q2
  claim authorization all remain false. Resume and re-audit the parent
  reviewed all-source P0-P8 goal separately.

## 2026-07-16 legacy L6 full-frame-context short decision

- Commit `1d22abd` freezes the paired evaluator for the six already-complete
  full-frame short GPU processes. Every mode uses 30 optimizer steps and
  134,924 parameters, peaks at 96,468,992 reserved bytes, has no OOM/retry,
  and cleans CUDA allocation/reservation to zero.
- Zero, availability-only, and full-frame macro-F1 are `0.2697662759`,
  `0.2721987509`, and `0.2942624204`; all per-mode repeats are deterministic.
- Full-frame minus zero is `+0.0244961445`, but its 33-video cluster interval
  is `[-0.0668714797, 0.0725200014]` and NLL worsens by `+0.2414525889`.
- Full-frame minus availability-only is `+0.0220636696`, with interval
  `[-0.0809709233, 0.0671747502]`; NLL worsens by `+0.3144303865`.
- The valid decision is
  `DO_NOT_EXPAND_FULL_FRAME_CONTEXT_FROM_CURRENT_SHORT_EVIDENCE`. Do not run a
  full confirmation or carry full-frame values into the legacy candidate.
- L6 is PASS with the parameter-matched T6 zero as the retained simple base.
  Continue L7 with separate event-balanced CE, effective-number CE, and
  Balanced Softmax policies. Reassess every rejected modality on frozen
  merged-reviewed data.
- The decision artifact SHA256 is
  `e006dc6636ede5a35e71414448be1dc96f0f71e29f5f2a1b6d0230fa0c49c6bf`.

## 2026-07-16 legacy L6 union-context short decision

- Commits `27cc764` and `4a3a998` freeze the crash-bounded actor-partner union
  trainer and paired evaluator. All six fresh GPU processes use 30 optimizer
  steps and 134,924 parameters, peak at 96,468,992 reserved bytes, have no
  OOM/retry, and clean CUDA allocation/reservation to zero.
- Zero, availability-only, and union macro-F1 are `0.2697662759`,
  `0.2721987509`, and `0.2297779478`; every per-mode repeat is deterministic.
- Union minus zero is `-0.0399883282`, with 33-video cluster interval
  `[-0.0909594431, 0.0147461009]`. Accuracy falls by `0.0204081633`, NLL
  worsens by `0.2082319675`, and rare-group macro-F1 falls by `0.1033780579`.
- The valid decision is
  `DO_NOT_EXPAND_UNION_CONTEXT_FROM_CURRENT_SHORT_EVIDENCE`. Do not run a full
  union confirmation or carry union values into the next candidate.
- S3 remains insufficient, so core-roadmap S4 full-frame context is the next
  authorized one-family test from the parameter-matched T6 zero. Reassess
  union context on merged-reviewed data.
- The decision artifact SHA256 is
  `bc37887890c0a7ea5e8eb993595547c30e69693f0f06ac8e12d0d2e453829778`.

## 2026-07-15 legacy L6 numeric-social short decision

- Six fresh GPU processes close the parameter-matched zero,
  availability-only, and numeric-social short matrix. Every process used 30
  optimizer steps and 69,664 parameters, peaked at 73,400,320 reserved bytes,
  had no OOM or retry, and cleaned CUDA allocation/reservation to zero.
- Zero, availability-only, and numeric-social macro-F1 are `0.2620738697`,
  `0.2621547321`, and `0.2624282011`; every per-mode repeat is deterministic.
- Numeric-social minus zero is `+0.0003543314`, with 33-video cluster interval
  `[-0.0342531654, 0.0398565230]`; accuracy changes by `-0.0326530612` and NLL
  worsens by `+0.2248711988`.
- The valid decision is
  `DO_NOT_EXPAND_SOCIAL_RELATION_FROM_CURRENT_SHORT_EVIDENCE`. Do not run a
  full numeric-social confirmation or carry its values into the next model.
- Core roadmap S2 permits top-K only after numeric-social S1 passes. Therefore
  top-K is `DEFERRED_NOT_AUTHORIZED`; continue L6 actor-partner union-crop work
  from the parameter-matched T6 zero because the interaction-context gap
  remains. Reassess numeric social and top-K on merged-reviewed data.
- The short matrix and decision SHA256 values are
  `1294bd8bd72e26701bbedea5c97e4fa9820b4233313b84d563153f42a5bd42a5` and
  `60c685094bddf83bf180174535afdf184133c2643d258cc09215be43b6d1ff61`.

## 2026-07-15 legacy L6 numeric-social cache

- Commits `6372dc6` through `7a42ce6` freeze the ten-feature numeric-social
  cache and independent repeat evaluator for the T6 sliding universe.
- Primary and repeat tensors have shape `[15588, 6, 10]`; 92,664 of 93,528
  slots are available, 864 are unavailable, and all four artifacts are
  byte-identical across separate roots.
- All 15,588 windows are locally rebased. There are 74,669 valid consecutive
  same-partner pairs, zero media reads, and zero outer-holdout slots.
- Partner IDs are audit metadata only. Top-K partner, geometry, motion, and ROI
  values do not enter model X.
- The primary manifest SHA256 is
  `5a0f66842e4fd0d8af363d3da1ebb762edd118586d62dd8a3bea4f4e6399a192`;
  the repeat-gate SHA256 is
  `3d4206c6679bc8f0cebe77c6da764ce8edb29deb2417f0cfacf82b6311d28d9f`.
- Cache evidence is PASS. The subsequent short decision rejects full
  numeric-social expansion; the core-roadmap strategy gate advances L6 to the
  actor-partner union crop and defers top-K.

## 2026-07-15 legacy L6 ROI full confirmation decision

- Commits `8445ae8` through `3e59197` freeze the audited 18-feature ROI cache:
  15,588 windows, 93,528 T6 slots, complete availability, and byte-identical
  independent builds. Geometry supplies order only and no geometry values enter
  the ROI tensor.
- Commits `c32d3fa` through `821e931` close the crash-bounded short matrix and
  paired evaluator. Six fresh GPU processes each used 30 steps, peaked at
  73,400,320 reserved bytes, had no OOM/retry, and cleaned CUDA usage to zero.
- ROI macro-F1 is `0.2886109023`, versus `0.2420943922` for the
  parameter-matched zero and `0.2405788407` for availability-only.
- ROI minus zero is `+0.0465165101`, with 33-video cluster interval
  `[0.0134524177, 0.0768469401]`; ROI minus availability-only is
  `+0.0480320616`, with interval `[0.0178011460, 0.0771756533]`.
- The hash-bound full confirmation is complete at
  `l6r_full_decision_v1.json` under
  `outputs/classification_v2/legacy_only_unreviewed_development/l6r_full_v1/`;
  its artifact SHA256 is
  `5a9a2b4b61b7ddeef0b5155ec69b678d73f0acd53917db98d1d6271cab5f1af3`.
- Full zero, availability-only, and ROI macro-F1 are `0.4966025667`,
  `0.4727197983`, and `0.5082292933`. ROI minus zero is `+0.0116267266`,
  with interval `[-0.0398806556, 0.0906766805]`; ROI minus availability-only
  is `+0.0355094951`, with interval `[-0.0248897889, 0.0986581204]`.
- This is mixed evidence, not uniformly negative ROI evidence. Against zero,
  feeding-group macro-F1 rises by `+0.1796877378`; `drink` F1 rises from
  `0.3703703704` to `0.6486486486`, and `eat` F1 rises from `0.7906976744`
  to `0.8717948718` while retaining recall `1.0`.
- `playwithtoy` has only one validation unit. Recall is `1.0` in both modes,
  while F1 falls from `0.6666666667` to `0.3333333333` as false positives
  increase from one to four. Its ROI effect is not estimable on `legacy_16f`.
- The valid decision is
  `DO_NOT_EXPAND_ROI_RELATION_FROM_CURRENT_SHORT_EVIDENCE`; the full gain
  misses the margin and positive interval-low gate, and availability-only
  fails its bounded-difference check. Continue L6 numeric social relations
  from parameter-matched T6 zero; do not carry ROI values into the candidate.
- This evidence applies only to unreviewed `legacy_16f`. It is not reviewed,
  final, canonical OOF, Q2, or merged-data evidence.

## 2026-07-15 legacy L6 motion valid negative decision

- Commits `cc01582`, `96e25b4`, and `6447c94` implement the crash-bounded
  cached-motion trainer, three-control short matrix, and paired evaluator.
- Six fresh-process GPU runs passed with 30 steps each, deterministic repeats,
  73,400,320 peak reserved bytes, no OOM/retry, and zero cleanup residue.
- Across 245 native units and 33 video clusters, motion minus parameter-matched
  zero macro-F1 is `-0.0018138438`, with interval
  `[-0.0260250944, 0.0233049368]`.
- The valid decision is
  `DO_NOT_EXPAND_MOTION_FROM_CURRENT_SHORT_EVIDENCE`; no full motion run is
  authorized. Continue L6 with numeric social relations from the zero control.
- This applies only to unreviewed `legacy_16f`. Reassess motion on frozen
  merged-reviewed data, whose rare-class support is materially larger.

## 2026-07-14 one canonical engine and explicit review profiles

- Legacy-only development and mixed-source reviewed data use one canonical
  data engine. Context, geometry, ROI, motion, social, posture, temporal
  harmonization, cache, fold, and training modules must not be forked or
  copied for the 16-frame lane.
- `legacy-only-unreviewed-development` is a profile over that engine: select
  only `legacy_recovered`, record an explicit human-review waiver, build
  `T6/T8/T12/T16` views inside each 16-frame burst, and isolate outputs,
  hashes, metrics, and claims.
- `mixed-reviewed` selects legacy plus CVAT, requires both review layers under
  the currently frozen Q2 protocol, and uses `fixed6_observed_time` as the
  primary model view. Native units remain legacy 16-frame and CVAT 6-frame.
- Human review is not a universal technical prerequisite for every exploratory
  run. It is mandatory for artifacts called reviewed/final and for the active
  mixed-source Q2 lineage. A waived profile must remain visibly unreviewed and
  cannot inherit reviewed or Q2 authorization.
- Complete the scoped legacy L0-L8 goal in a new chat, emit the immutable
  handback audit, then return to the original chat and resume the blocked
  parent P0-P8 goal. Legacy completion never completes the parent implicitly.

## 2026-07-14 exact legacy temporal model inputs

- Commit `21b34fd` binds eight explicit legacy model views: `T6`, `T8`,
  `T12`, and `T16`, each with all-sliding event-balanced and centered-matched
  sampling.
- The common evaluation unit remains the complete 16-frame burst. The loader
  binds view name, selection column, slot manifest, exact observed timing, and
  `temporal_input_frames`; it rejects malformed lengths and timing.
- Actor and union-context tensors must already be exact `T`. Spatial tensors
  may have capacity 16 only when every post-`T` length-mask slot is false.
- Four synthetic tier forwards produce finite `[2, 10]` logits with the same
  25,115 parameters. Evidence is 438 classification tests passing with 181
  deselected, Ruff, compileall, and zero optimizer steps or weight downloads.
- This is `PASS IN CODE`. A versioned short real-data chain must pass before a
  full legacy rebuild, and neither action authorizes training or full OOF.

## 2026-07-14 legacy-only unreviewed development authorization

- The user authorizes bounded classifier development from the 72,864-row
  `legacy_recovered` reference without waiting for current Hidden or behavior
  review.
- Every artifact and metric must be labeled
  `legacy-only-unreviewed-development`; this lineage is for loader, model,
  runtime, controlled ablation, and historical-comparison work only.
- Recording/video-safe groups, one 16-frame native burst per unit, exact
  feature whitelists, immutable hashes, and short-before-full gates remain
  mandatory.
- Model inputs must expose controlled `T6`, `T8`, `T12`, and `T16` tiers built
  inside each burst after harmonization. Evaluate every tier at the common
  16-frame native-unit grain.
- This branch cannot be called reviewed/final train-ready data, replace the
  all-source reviewed evaluation, authorize a Q2 claim, or weaken the existing
  human-review and full-OOF gates.
- New chats must create the scoped goal from
  `plans/classification_v2-legacy-16f-development-goal-prompt.md` and track
  L0-L8 in its dedicated execution ledger.
- After L0-L8 genuinely complete, return the hash-bound handback to the parent
  Q2 chat. The parent goal remains incomplete until separately resumed.

## 2026-07-14 native source and missingness probes

- Commit `9b04209` replaces the positional, all-numeric window source probe
  with exact trainer-whitelist and ordered-window SHA256 binding.
- Repeated windows are averaged once per `temporal_unit_key`; preprocessing and
  the linear probe fit only grouped training roles, and every eligible native
  unit must appear in outer test exactly once.
- A separate availability-only behavior diagnostic accepts only
  `window_image_context_complete`, `scene_context_ready`, and
  `scene_partner_context_ready`. Label-gated `interaction_context_ready` fails.
- Evidence is 14 focused tests and 429 classification tests with 181 deselected,
  Ruff, compileall, import smoke, and zero overlong changed lines.
- This is fixture-only engineering evidence. It does not authorize active-data
  training while Hidden and behavior review remain incomplete.

## 2026-07-14 native-unit checkpoint selection

- Commit `abae856` replaces window-level early stopping with grouped inner
  validation at the native temporal-unit grain.
- The primary score is supported-class macro-F1; native-unit NLL breaks ties.
  Outer-test predictions are never eligible for model selection.
- Mean class probabilities collapse windows by `temporal_unit_key`. Blank or
  duplicate keys, target/fold/source/group conflicts, malformed probabilities,
  nonfinite losses, and native-unit row loss fail closed.
- New runs emit best-validation and outer-test evidence at both window and
  native-unit levels. Native output preserves `source_type` and
  `split_group_key` as audit-only metadata outside model X.
- Checkpoint v6, run identity v3, run manifest v3, prediction manifest v2,
  registry v5, and run audit v3 bind the exact selection policy.
- Evidence is 415 classification tests passing with 181 deselected, Ruff,
  compileall, zero overlong changed Python lines, and a passing synthetic
  checkpoint/resume audit. No project-data training or full OOF ran.
- Human Hidden and behavior review remain the active blockers. A bounded metric
  test should prefer a reviewed, grouped, native-safe legacy 16f slice and must
  be labeled `legacy-only`; it cannot support the all-source Q2 claim alone.

## 2026-07-14 audited visual freeze schedule

- Commit `2bd2fda` adds one versioned schedule for actor and union-context
  backbones: frozen warm-up, ResNet `layer4` only, then optional full fine-tune.
- All parameters enter stable optimizer groups before training. The backbone
  uses a lower LR, frozen BatchNorm stays in eval mode, and heads stay trainable.
- A no-download, zero-step V0/V1/V2 audit proves identical pretrained status,
  normalization, freeze schedule, and head signature across controlled changes.
- Checkpoint v5, run identity v2, run manifest v2, and registry v4 bind stage,
  optimizer groups, schedule, trainable counts, config, fold, and artifact hashes.
- Evidence is 39 focused tests, 391 classification tests with 181 deselected,
  compileall, Ruff, checkpoint resume, and zero project-data rows read.
- This is engineering PASS only. Hidden and behavior review still block the
  active snapshot, pretrained pilots, model selection, and full OOF.

## 2026-07-14 deterministic visual tiny-overfit gate

- Commit `3be22f8` adds a data-free ResNet18 actor-temporal correctness gate for
  one-batch gradients, 20-event ten-class overfit, and optimizer-state resume.
- Two deterministic CUDA runs share the same semantic SHA256. The persisted
  audit reports accuracy 1.0, loss ratio 0.076818, zero resume-logit drift,
  778,576,384 peak VRAM bytes, and finite nonzero backbone/head gradients.
- The gate caught train/eval BatchNorm drift during development. It now scores
  only after post-fit running-stat recalibration; that smoke-only policy does
  not select the final training BatchNorm policy.
- Evidence is 35 focused tests and 385 classification tests with 181 deselected.
  The audit explicitly denies snapshot and full-OOF authorization.

## 2026-07-14 audited ResNet backbone interface

- Commit `07ed768` adds versioned `smoke_cnn`, ResNet18, and ResNet34 frame
  encoders to the common mask-safe model factory.
- Exact ImageNet enums and RGB mean/std are part of the contract. Tests resolve
  pretrained metadata without downloading weights; actual forwards use
  `NONE_RANDOM_INIT` only.
- Controlled forwards pass for ResNet18 at 160 and 224 px and ResNet34 at
  224 px. Parameter counts are 11,185,658, 11,185,658, and 21,293,818.
- Evidence is 31 focused model tests, the dry-run backbone audit, and 381
  classification tests with 181 deselected. No optimizer or training ran.
- This closes the independent production-backbone interface only. Human Hidden
  and behavior review still block the active snapshot and every model pilot.

## 2026-07-14 historical target-independent Hidden v6 reference

- The technical reference is
  `outputs/classification_v2/rebuilds/hidden_review_v6_full_20260714`.
- Its 5,131 unique items comprise 4,122 Yes confirmations, 384 high-risk No,
  601 stratified-random No, and 24 clean controls.
- Selection excludes behavior and target-derived fields. The manifest SHA256 is
  `3e4fec14c466a89370a1e20d913cb024bd1dda1fa8db9c1fabdf8a51fa31072e`.
- Commit `f2179e3` binds media audit v2 to manifest/frame-context hashes. The
  24-item smoke and 5,131-item full resolver both have `media_missing=0`.
- Commits `32eaa2b` and `aaf8460` migrate v5 onto identifier v2, then carry 30
  payload rows with zero technical drift. User provenance is now unverified.
- Artifact coverage was 30/5,131, but verified human coverage is now 0/5,131.
  Random/high-risk support is zero, so the scientific gate is blocked.
- Evidence is 31 focused tests and 377 classification tests with 181 deselected.
  Hidden apply, reviewed snapshot, model training, and full OOF remain forbidden.

## 2026-07-14 historical baseline controls

- Commit `e5d6417` registers the commit-`18d6692` full OOF only as
  `HISTORICAL_ONLY` and the legacy sequence checkpoint only as
  `HISTORICAL_ARCHITECTURE_ONLY`.
- The audit reproduces 151,440/160,740 split-to-context positional mismatches;
  image and interaction manifests agree with each other at all positions.
- Registration-time hashes cover 34 artifacts and 527,948,648 bytes. They do
  not prove origin-time inputs because the old run did not bind those hashes.
- `pig_behavior_sequence.pt` is a ten-output ResNet34 architecture reference,
  not a performance baseline without dataset, grouped-split, config, and seed
  lineage.
- Evidence is 5 focused tests and 356 classification tests with 181 deselected.
  Training, model promotion, paired comparison, and Q2 claims remain forbidden.
- The next independent task is Hidden clustered uncertainty and
  target-independent prevalence gates. Human review remains the P0 blocker.

## 2026-07-14 native-unit paired evaluation contract

- Commit `1b6ba3d` preserves the complete fold-assignment universe while
  collapsing strict ten-class window probabilities to one prediction per
  eligible native temporal unit.
- Prediction labels and folds must match the authority manifest; malformed
  keys, probability vectors, conflicting targets, missing units, and split
  drift fail closed.
- Primary pooled macro-F1 uses the fixed global ten-class order. Supported-fold,
  behavior-group, source/video/recording, calibration, and class-fold evidence
  remain explicit.
- Paired uncertainty resamples recording clusters with identical unit, target,
  cluster, and fold mappings. Percentile bootstrap reports no pseudo p-value.
- Evidence is 31 focused tests, 351 classification tests with 181 deselected,
  and a passing four-unit synthetic checker with no full-data read or training.
- The next independent task is historical-baseline reconciliation as an
  engineering control. Human Hidden and behavior review remain blockers.

## 2026-07-14 strict fixed-six timing loader

- Commit `111f152` loads ordered real `time_delta` values for the primary
  fixed-six view while preserving the complete training-window universe.
- Missing, reordered, duplicate, negative, or contradictory slot records fail
  closed; unselected windows remain explicit NaN/mask rows instead of dropping.
- Actor, spatial, and union branches receive the same timing tensor while
  retaining branch-specific observation masks.
- Checkpoint schema v4 and registry v3 bind the separate temporal-manifest hash.
- Evidence is 334 classification tests, four passing synthetic checkers, no
  pretrained download, no full-data read, and no model training.
- The next independent task is native-unit collapse and paired evaluation on
  synthetic predictions. Human Hidden and behavior review remain blockers.

## 2026-07-14 mask-safe configurable model factory

- Commit `318bf58` adds ten exact model modes and four temporal encoders behind
  one validated factory; direct 10-class supervision remains mandatory.
- Availability and quality masks gate optional data without entering X as
  behavior evidence. Masked NaN values cannot change logits.
- Checkpoint schema v3 and registry v2 now bind `model_mode`; old configs are
  read compatibly and reserialize with an explicit mode.
- Evidence is 78 focused tests, 319 classification tests, model-factory and
  multitask dry-runs, zero weight downloads, and zero optimizer steps.
- ResNet backbones and strict Transformer training remain blocked. The next
  independent task is real fixed-six `time_delta` loading and hash lineage.

## 2026-07-14 fold-local training lineage contracts

- Commits `97f83c5` and `73b901d` add training-fold-only preprocessing and
  native-event mass weighting without reading validation/test statistics.
- Commit `16cdb93` adds immutable run identity, isolated `fold_id/run_id`
  artifacts, checkpoint schema v2, append-only registry rows, remote merge,
  resume/hash audits, and lineage-aware caller paths.
- Evidence is 33 focused tests, 292 classification tests with 181 deselected,
  zero overlong Python lines, and a passing checkpoint/resume smoke.
- No model training or OOF ran. Those historical payload counts are superseded;
  verified Hidden/behavior coverage is now 0/5,131 and 0/4,670.
- The next independent task is the configurable model factory with explicit
  modality availability/quality masks and forward-shape contracts.

## 2026-07-13 fixed-six temporal-view contract

- Commit `bb225ff` adds keyed manifests for `fixed6_observed_time`,
  `fixed6_normalized_phase`, and the `native6_16` ablation.
- The primary view reuses existing post-harmonization six-frame windows for
  both sources. It does not sample six quantiles across a legacy 16-frame burst.
- Every original window remains in a selection ledger; every native 6/16 unit
  remains in the native view, and missing observations remain explicit masks.
- Persisted manifests bind row counts and ordered stable-key hashes. Structural
  source/length/padding/timing/quality/availability shortcuts fail closed unless
  a separate valid mitigation artifact is supplied.
- Fixture evidence is 22/22 tests; the full classification regression is
  243 passed and 181 deselected. No real reviewed artifact, training, or OOF was
  run, so human Hidden and behavior review remain the active blockers.

## 2026-07-13 snapshot and launch-lineage hardening

- Commit `7cb4637` makes snapshot v2 fail closed on blank/duplicate keys,
  contract drift, invalid freeze, and ordered split/image/interaction mismatch.
- Commit `dd0e6ff` binds preflight, execution, and human authorization to the
  snapshot hash, lineage-audit hash, ordered `window_id` hash, config, and code.
- Interaction-context export now emits the same ordered-key audit and requires
  explicit `--overwrite` before replacing derived outputs.
- Regression evidence is 221 classification tests passing. No model training or
  full OOF was run.
- The bounded identifier-v2 packet remains technical-only because its Hidden
  and behavior-review authorizations are false.

## 2026-07-13 identifier-v2 positional lineage correction

- Commit `bfdf913` fixed a critical row-order defect: image and interaction
  windows had been sorted independently from split/target/spatial rows.
- In the historical full artifact, split-to-image and split-to-interaction
  positional mismatches were `151,440/160,740`; image and interaction happened
  to agree with each other. Historical OOF metrics are therefore runtime/debug
  evidence only, not classifier-quality evidence.
- Commit `a83d5a5` centralizes ordered `window_id` validation and adds fail-closed
  source-to-window identifier-v2 auditing plus overwrite guards.
- Current bounded authority is
  `outputs/classification_v2/rebuilds/scientific_smoke_identifier_v2_20260713`.
  It passes with 688 frame rows, 63 native units, 438 windows, exact 110-field X,
  zero trainable missing spatial slots, and 8/8 byte-identical repeat artifacts.
- Ordered sequence/image/train-ready/spatial hash is
  `05656ad78ecb65fac2341bc865f741039fe0c1e6b28211c65f2fc2c7973d7996`.
- Status remains `PASS_IDENTIFIER_V2_TECHNICAL_HUMAN_REVIEW_BLOCKED`; all
  training, full-OOF, reviewed-dataset, and Q2-claim authorizations are false.

## 2026-07-13 classification_v2 technical assurance gate

- Technical smoke authority is
  `outputs/classification_v2/rebuilds/scientific_smoke_v1/audits/technical_smoke_gate.json`.
- Commit `1679aca` reports `PASS_TECHNICAL_SMOKE_HUMAN_REVIEW_BLOCKED` with
  688 frame rows, 63 native/review units, 438 windows, both sources, and all
  10 behaviors.
- Tabular X matches the exact 110-column trainer whitelist; all 73 temporal
  evidence fields are present, while Hidden/review/target fields stay outside X.
- All 342 trainable smoke windows have complete spatial slots. The 544 missing
  slots belong only to 96 retained mask-false windows.
- Five repeated enhanced/harmonized/interval/window CSV pairs are byte-identical.
- Builders now exit nonzero on audit errors and require explicit `--overwrite`
  before replacing derived artifacts.
- This is code/data-generation assurance only. Current verified Hidden and
  behavior coverage is 0/5,131 and 0/4,670, so reviewed training remains blocked.

## 2026-07-13 authoritative classification_v2 state

- Use `docs/CLASSIFICATION_V2_CURRENT_STATE.md` as the status authority.
- The active path is the reviewed-data rebuild, not postrun promotion of the
  previous full OOF artifact.
- Historical v5/v6 templates have no user-verified decision authority.
- Current verified coverage is 0/5,131 Hidden and 0/4,670 behavior units.
- Therefore Hidden apply, behavior apply, the reviewed train-ready snapshot,
  model smoke on that snapshot, and a new full OOF are all blocked.
- The old commit-`18d6692` full OOF is historical engineering evidence only.

## 2026-07-13 classification_v2 Hidden review workload

- Current workload-policy implementation commit is `5212a59`.
- Hidden review now census-selects every untrusted CVAT Hidden=Yes, samples
  trusted legacy Yes by recording-date/behavior stratum, and caps initial
  high-risk Hidden=No review at one item per stratum.
- Versioned full evidence is
  `outputs/classification_v2/rebuilds/hidden_review_v5_full_20260713`.
- V5 has 5,171 unique items: 4,649 CVAT and 522 legacy. It has zero missing
  untrusted Yes, trusted-stratum quota mismatches, or high-risk cap violations.
- This only makes the review workload auditable. Human Hidden decisions and
  behavior decisions remain incomplete, so the dataset is not train-ready.

## Historical 2026-07-13 full OOF and workflow migration

- Full multimodal OOF training completed in
  `outputs/classification_v2/model_full/full_multimodal_oof/`.
- Verified full outputs contain 73,668 window predictions and 32,727 native
  temporal predictions; accuracy is `0.5216793473` and supported macro-F1 is
  `0.4156053847`.
- These metrics are engineering evidence from the previous data lineage. They
  cannot become the final Q2 result by postrun processing alone because current
  Hidden and behavior human-review gates are incomplete.
- All classification operator scripts now live only under
  `scripts/classification_v2/00_*` through `09_*`. The former split namespaces
  and compatibility wrappers were removed.
- Workflow migration commits are `d7d22a8` and `1491d78`. The structural audit
  is block `09` script `check_classification_v2_workflow_layout.py`.

## Historical 2026-07-13 pre-full hardening refresh

- The previous lineage recorded its verified HEAD in
  `outputs/classification_v2/model_design/q2_progress_report_audit.json` key
  `current_git_commit`.
- `q2_progress_report_audit.json` is valid with `PASS_PARTIAL_ROADMAP`,
  44/44 gates passing, clean git, `full_oof_execution_allowed=false`,
  `authorization_authorized=false`, and `q2_claim_allowed=false`.
- Full OOF execution gate is now hardened with 4 rejection cases, including a
  near-authorized authorization file that has all boolean approvals true but
  missing `reviewer` and `reviewed_at`.
- Preflight runtime benchmark drift now allows audit/auth-only changes without
  rebenchmarking, while keeping runtime/model/training changes fail-closed.
- This pre-full state was later followed by the historical full run. It does not
  describe the active reviewed-data rebuild and must not authorize another run.

## Architecture contract retained from 2026-07-12

- Active priority is `classification_v2` behavior recognition, not tracking
  ablation, unless the user explicitly switches back to tracking.
- Current target claim remains Q2-strong only: improved pig behavior recognition
  under recording-date/video-safe validation. Do not claim external farm,
  camera, or cohort generalization until external validation exists.
- Pipeline is built around multimodal spatio-temporal inputs:
  bbox/letterboxed actor image sequence, ROI relation features, motion,
  social/partner context, interaction visual context, event-balanced weights,
  native temporal OOF folds, and strict feature whitelist leakage guards.
- `pig_id` is annotation-local and must not be treated as the same animal across
  videos or sessions.
- Canonical actor image cache is letterboxed, not square-stretched:
  `outputs/classification_v2/image_cache_v2_letterbox/`.
- Every future full OOF launch still requires preflight plus explicit
  authorization bound to the active data/cache/config/code hashes.
- The previous 44/44 `PASS_PARTIAL_ROADMAP` report is stale for the current
  rebuild and must not be refreshed until human review and snapshot gates pass.
- Local RTX 3050 limits development batch size, not the research architecture;
  remote/rented GPU execution remains allowed after the same lineage gates.
- Postrun calibration, confusion analysis, ablation, registry, and completion
  checks remain required after a future reviewed-lineage full run.

## 2026-07-08 realtime full runtime chunk validation

- Runtime 13-video validation completed in two chunks: `outputs/eval/realtime/runtime_check_quality_delayed_simple_7video/iou0_area0_condarea0_merge0` plus `outputs/eval/realtime/runtime_check_quality_delayed_simple_remaining6/iou0_area0_condarea0_merge0`.
- Compared with `outputs/eval/realtime/realtime_balanced_13video/iou0_area0_condarea0_merge0`, per-video runtime total `remapped_idsw 75 -> 21`, `fp/fn` stayed `2320/1055`, no per-video IDSW regression, and `000302=0`.
- Remaining runtime IDSW: `000114=2`, `000231=6`, `000233=9`, `000263=2`, `000327=2`; all other 8 videos are `0`.

## 2026-07-08 realtime simple low-gain component pass

- Improved `realtime_quality_delayed` artifact candidate further by adding an opt-in second pass for simple motion components only: `realtime_motion_pair_simple_min_gain=0.005`, `realtime_motion_pair_simple_max_component_size=2`.
- Evidence artifact: `outputs/eval/realtime/probe_motion_pair_simple005_comp2_13video/iou0_area0_condarea0_merge0`.
- Compared with gain-gate candidate `outputs/eval/realtime/probe_motion_pair_gainmin004_edges2_13video/iou0_area0_condarea0_merge0`: `ALL remapped_idsw 27 -> 21`, `remapped_hota_pct 95.89 -> 96.60`, `remapped_idf1_pct 96.41 -> 97.02`, `fp/fn unchanged 2320/1055`.
- Per-video improvements with no IDSW regression on 13-video artifact probe: `000085 2 -> 0`, `000327 4 -> 2`, `000330 2 -> 0`; `000233` stayed `9`, `000302` stayed `0`.
- Runtime smoke check: `outputs/eval/realtime/runtime_check_quality_delayed_simple_233_302/iou0_area0_condarea0_merge0` using actual `realtime_quality_delayed` code path produced `000233 remapped_idsw=9` and `000302 remapped_idsw=0`, matching the artifact expectation for target/guardrail.
- Rejected probes: global `min_allowed_edge_gain=0.02` regressed `000114/000327`; global `max_jump=0.08/0.12` regressed `000233`; `memory_frames=20` regressed `000231`; `memory_frames=40` unchanged; global `min_gain=0.005` regressed `000233/000327`.
- Additional runtime smoke checks: `outputs/eval/realtime/runtime_check_quality_delayed_simple_263/iou0_area0_condarea0_merge0` produced `000263 remapped_idsw=2`; `outputs/eval/realtime/runtime_check_quality_delayed_simple_085/iou0_area0_condarea0_merge0` produced `000085 remapped_idsw=0`. Both match artifact expectations.

## 2026-07-08 realtime dense fallback gain gate

- Improved current `realtime_quality_delayed` motion-pair candidate by tightening dense-component fallback: `realtime_motion_pair_dense_fallback_max_edges=2`, `realtime_motion_pair_dense_fallback_min_median_gain=0.05`, `realtime_motion_pair_dense_fallback_min_edge_gain=0.04` while keeping `max_component_size=4`, `max_component_edges=3`, `max_support_ratio=0.35`.
- Evidence artifact: `outputs/eval/realtime/probe_motion_pair_gainmin004_edges2_13video/iou0_area0_condarea0_merge0`.
- Compared with previous dense candidate `outputs/eval/realtime/probe_motion_pair_comp4_edges3_dense_13video/iou0_area0_condarea0_merge0`: `ALL remapped_idsw 31 -> 27`, `000233 13 -> 9`, no per-video IDSW regression, `000302=0`, `fp/fn unchanged 2320/1055`; HOTA/IDF1 effectively unchanged at `95.89/96.41`.
- Runtime check on `000233` planned graph confirms allowed dense fallback edges become `{ID_1-ID_3, ID_1-ID_8}`, excluding weak low-min-gain `ID_2-ID_8` that caused extra switches.

## 2026-07-08 realtime motion-pair quality-delayed candidate

- Added opt-in `realtime_motion_pair_stabilizer` for `mode=realtime` only. It relabels short-memory motion-consistent ID attributes, then filters proposed relabel graph to small/sparse components. The current 13-video candidate uses `realtime_motion_pair_max_component_size=4`, `realtime_motion_pair_max_component_edges=3`, and dense-component rare-edge fallback (`max_edges=3`, `max_support_ratio=0.35`); this admits sparse four-ID episodes like `000327` and a limited rare-edge subset in dense `000233` while still blocking the dominant long cascade edge.
- Important implementation fix: the planning pass must use `deepcopy`; shallow `shape.copy()` mutates nested `attributes` and accidentally applies broad relabel before component filtering.
- Enabled the stabilizer in `realtime_quality_delayed`, not in `realtime_balanced`. Treat this as a quality-delayed candidate, not the pure causal realtime baseline.
- Validated runtime 5-video result: `outputs/eval/realtime/codex_motion_pair_quality_5video_fix/iou0_area0_condarea0_merge0`. Compared with `outputs/eval/realtime/realtime_balanced_5video/iou0_area0_condarea0_merge0`: `ALL remapped_idsw 43 -> 25`, `remapped_hota_pct 90.12 -> 92.20`, `remapped_idf1_pct 90.18 -> 92.50`, `fp/fn unchanged 849/661`.
- Per-video remapped IDSW in this candidate: `000231=8` (from `12`), `000233=15` (unchanged, no regression), `000263=2` (from `12`), `000328=0` (from `4`), `000302=0` (guardrail preserved).
- 13-video artifact probe with component size `4`, edge cap `3`, and rare-edge fallback: `outputs/eval/realtime/probe_motion_pair_comp4_edges3_dense_13video/iou0_area0_condarea0_merge0`. Compared with `outputs/eval/realtime/realtime_balanced_13video/iou0_area0_condarea0_merge0`: `ALL remapped_idsw 75 -> 31`, `remapped_hota_pct 92.77 -> 95.89`, `remapped_idf1_pct 93.12 -> 96.41`, `fp/fn unchanged 2320/1055`. No per-video remapped IDSW regression in this 13-video set; `000302` stayed `0`. `000327` improved `8 -> 4`; `000233` improved `15 -> 13` but remains the weakest realtime video.
- Do not promote this as realtime causal base without broader regression and explicit discussion that it is delayed/post-tracking stabilization. Next step should validate more videos and then design an online-buffer equivalent if true realtime latency is required.

## 2026-07-08 realtime profile cleanup and failed probes

- `realtime_balanced` was changed to inherit from a realtime-only eval base:
  `enable_offline_smoothing=false`, `identity_swap_guard=false`,
  `smooth_boxes=false`, and `refine_boxes=false`. Single `000233` with these
  offline flags forced off matched the prior metrics, so the current realtime
  failures are from online association/tracking behavior rather than offline
  smoothing.
- Added named realtime eval profiles:
  - `realtime_fast`: speed-oriented probe, `detect_every_n_frames=2`,
    `det_conf=0.25`, `max_raw_detections=32`, no offline smoothing.
  - `realtime_balanced`: current causal probe stack.
  - `realtime_quality_delayed`: finite-window local repair probe only; no
    suffix/long future repair.
- Rejected/neutral probes from this continuation:
  - `overlap_small_box_suppression=true` on realtime `000233`: no metric change
    (`remapped_idsw` stayed `15`).
  - hybrid causal guard stack on realtime `000233`: no improvement; FP slightly
    increased.
  - `tracker_type=botsort` on realtime `000233`: no metric change.
  - looser `realtime_visible_better_competitor_min_cost=0.28`,
    `min_gain=0.025` on `000263`: regressed `remapped_idsw 12 -> 16`, so do
    not promote.
  - `local_pair_swap_repair=true` with a 12-frame window on realtime `000263`:
    no metric change.
- Current conclusion: remaining realtime IDSW is not solved by porting existing
  hybrid causal guards or existing finite-window repair as-is. Next useful
  implementation should be a new online/short-buffer identity stabilizer, not a
  broad reject/hold guard and not offline suffix repair.

## 2026-07-08 realtime balanced profile

- Added `realtime_balanced` to `scripts/evaluate_tracking.py` as the current
  named realtime probe profile. It packages the useful opt-in realtime stack:
  `smooth_det020_loose` recovery settings, `occlusion_aware_matching=false`,
  `realtime_visible_close_competitor_guard=true`,
  `realtime_visible_better_competitor_reject=true`,
  `realtime_visible_better_competitor_prefer=true`, and
  `realtime_low_conf_recovery_guard=true`.
- `realtime_balanced` is still a probe profile, not a finished realtime
  baseline. It preserves the `000302` guardrail in the single-video check:
  `outputs/eval/realtime/realtime_balanced_302_guardrail/iou0_area0_condarea0_merge0`
  produced `remapped_idsw=0`, `remapped_hota_pct=99.38`,
  `remapped_idf1_pct=99.69`.
- 5-video validation with the named profile:
  `outputs/eval/realtime/realtime_balanced_5video/iou0_area0_condarea0_merge0`.
  This matches the prior long override candidate: `ALL remapped_idsw=43`,
  `fn=661`, `fp=849`, `remapped_hota_pct=90.12`,
  `remapped_idf1_pct=90.18`; per-video remapped IDSW remains
  `000231=12`, `000233=15`, `000263=12`, `000328=4`, `000302=0`.
- Rejected new probe: `realtime_reid_shadow_visible_hold`. Broad version on
  `000263` reduced remapped IDSW `12 -> 8` but badly damaged idmap coverage and
  HOTA/IDF1 (`remapped_hota_pct` about `81.70`). Narrowing to
  `max_missed=5` kept `000263` at `12` IDSW and did not improve HOTA. The guard
  was removed from code. Do not re-add a hold/consume duplicate-shadow reid
  guard without a fundamentally better discriminator.

## 2026-07-08 realtime failed guard probes

- Tried a narrow opt-in `realtime_occluded_reid_duplicate_guard` idea for
  `000263` reid switches. Default `min_iou=0.55` did not trigger. Lowering to
  `min_iou=0.45` reduced `000263` remapped IDSW `12 -> 8`, but badly damaged
  IDF1/HOTA/idmap coverage (`remapped_hota_pct` about `81.70` versus `90.62`),
  so this is not a promotion candidate. The underlying evidence is still useful:
  wrong `000263` reid detections around `792/846/865` overlap visible tracks by
  only about `0.46-0.49` IoU, so simple duplicate rejection is too blunt.
- Tried an opt-in visible row-regret reject for `000231` frame `1368`
  (`selected_cost≈0.668`, `track_best≈0.226`). It triggered exactly once but did
  not reduce remapped IDSW and slightly worsened FP/FN, so it was removed.
- Current realtime candidate remains the missed3 low-conf stack:
  `outputs/eval/realtime/probe_realtime_low_conf_recovery_missed3_5video/iou0_area0_condarea0_merge0`.
  Continue from diagnostics rather than re-adding the rejected duplicate or
  row-regret guards.

## 2026-07-08 realtime missed3 candidate

- Realtime baseline at `outputs/eval/realtime/baseline_current/iou0_area0_condarea0_merge0` remains the main realtime comparison point.
- Current useful realtime candidate keeps all new realtime guards opt-in:
  `occlusion_aware_matching=false`,
  `realtime_visible_close_competitor_guard=true`,
  `realtime_visible_better_competitor_reject=true`,
  `realtime_visible_better_competitor_prefer=true`,
  `realtime_low_conf_recovery_guard=true`.
- Tuned `realtime_low_conf_recovery_min_missed` default for the opt-in guard to `3`.
  Single `000233` improved versus the broad low-conf guard: `IDSW=15`,
  `FN=388`, `remapped_hota_pct=84.21` at
  `outputs/eval/realtime/probe_realtime_low_conf_recovery_233_missed3/iou0_area0_condarea0_merge0`.
- 5-video candidate
  `outputs/eval/realtime/probe_realtime_low_conf_recovery_missed3_5video/iou0_area0_condarea0_merge0`:
  `ALL remapped_idsw=43`, `fn=661`, `fp=849`, `remapped_hota_pct=90.12`,
  `remapped_idf1_pct=90.18`. Per-video remapped IDSW:
  `000231=12`, `000233=15`, `000263=12`, `000328=4`, `000302=0`.
  This is not final, but it is cleaner than the earlier broad low-conf guard
  because it preserves most IDSW gain while recovering FN.
- Rejected probe: `realtime_late_reid_guard` for stale occlusion reid on `000263`
  reduced `000263` IDSW `12 -> 10` but badly damaged IDF1/HOTA and idmap coverage
  (`remapped_hota_pct=81.66`), so it was removed from code.
- Diagnostics from `outputs/pred/realtime/probe_realtime_missed3_263_debug/.../association_debug_events.csv`:
  remaining `000263` switches are mostly `reid` from `OCCLUDED/occlusion_hold`
  despite low selected costs, e.g. frames `792` missed `3`, `846` missed `33`,
  `865` missed `5`. A simple late-missed gate is not safe; next direction should
  inspect whether these reid detections are duplicates/extra boxes or need a
  causal short-window identity stabilizer rather than a broad reject.

## 2026-07-08 realtime coverage candidate

- Baseline realtime at
  `outputs/eval/realtime/baseline_current/iou0_area0_condarea0_merge0` has a
  major coverage/FN problem: 13-video `ALL` `fn=72669`, `recall_pct=60.58`,
  `remapped_idsw=115`, `remapped_hota_pct=57.55`.
- Strongest realtime lever so far is `occlusion_aware_matching=false`; on the
  5-video guard set it reduces `fn` from about `32425` to `601`, but exposes
  visible close-competitor swaps.
- Added opt-in `realtime_visible_close_competitor_guard=true` for realtime only
  when `occlusion_aware_matching=false`. It resolves near-tie high-confidence
  visible assignments toward an otherwise unserved competitor track.
- The useful discriminator came from debug:
  - `000302` good trigger: frame `555`, selected `track 8` vs preferred
    `track 4`, costs `0.194358` vs `0.204259`, margin about `0.0099`.
  - `000263` false trigger with wider margin: frame `421`, costs `0.261192`
    vs `0.276337`, margin about `0.0151`.
  - Default `realtime_visible_close_competitor_margin=0.012` keeps the `000302`
    fix while blocking the `000263` false trigger.
- Current 5-video realtime candidate:
  `outputs/eval/realtime/probe_close_competitor_margin012_5video/iou0_area0_condarea0_merge0`.
  Compared with realtime baseline subset, it is a large coverage/HOTA
  improvement but not an IDSW win: `fn=601`, `fp=942`, `remapped_idsw=63`,
  `recall_pct=99.15`, `remapped_hota_pct=88.07`. Per-video remapped IDSW:
  `000231=27`, `000233=20`, `000263=12`, `000328=4`, `000302=0`.
- Do not promote this as the default realtime config yet. Next realtime work
  should reduce the remaining visible-swap IDSW on `000231/000233/000263/000328`
  without reintroducing hidden/occluded coverage loss.

## 2026-07-07 visible-start suffix gate full success

- New best full 12-video candidate:
  `outputs/eval/hybrid_bytetrack/codex_visible_suffix_gate_full/iou0_area0_condarea0_merge0`.
- Versus `outputs/eval/hybrid_bytetrack/Best_tracking/iou0_area0_condarea0_merge0`:
  `ALL` remapped IDSW improved `11 -> 0`; every per-video remapped IDSW is
  `0`.
- Guardrails stayed clean: `000085=0`, `000225=0`, `000231=0`, `000302=0`,
  `000328=0`.
- Targets fixed: `000233=0`, `000263=0`.
- Key code change: `suffix_pair_swap_repair=true` is now narrowed by requiring
  both shapes at `swap_start` to have `Hidden=No`. This blocks the false
  hidden-start suffix swaps previously seen on `000085` frame 17 and `000225`
  frame 264, while still allowing the visible-start `000263` suffix repair
  around frame 193.

- Do not hardcode `000263`/`000302` as optimizer target videos anymore.
- For optimizer ranking defaults, derive weak target videos from:
  `outputs/eval/hybrid_bytetrack/Tracking mới tắt smooth/yolov8/iou0_area0_condarea0_merge0/tracking_metrics.csv`
- Current weakest set from that file is:
  - `Pigs291119_000263_30fps`
  - `Pigs291119_000226_30fps`
  - `Pigs301119_000327_30fps`
  - `Pigs301119_000328_30fps`
- For `evaluate_tracking.py` metric comparisons, use commit `b697c4eba36db280cbf01f446873da17bcac509d` as the relevant historical reference instead of legacy 21/06.
- Critical IDSW-preserving tracking flow in `src/pig_behavior/tracking/runner.py`:
  - `apply_identity_swap_guard(...)` runs only when `cfg.enable_offline_smoothing and cfg.identity_swap_guard`.
  - `refine_shapes_temporally(...)` and then `stabilize_overlap_hidden_islands(...)` run only when `cfg.enable_offline_smoothing and (cfg.smooth_boxes or cfg.refine_boxes)`.
  - Do not change this back to `cfg.enable_offline_smoothing or cfg.mode == "hybrid_bytetrack"`; that drift was identified as a likely cause of worse IDSW versus commit `b697c4eba36db280cbf01f446873da17bcac509d`.
- Current tracking CLI flow:
  - Use `scripts/track_videos.py` for batch/single-video tracking.
  - `track_videos.py` calls `python -m pig_behavior.tracking.cli`; `src/pig_behavior/tracking/cli.py` must keep its `__main__` entrypoint.
  - `track_videos.py --eval-config <name>` reuses `evaluate_tracking.py` presets and passes them to tracking CLI as `--profile-override KEY=VALUE`.
  - `track_videos.py` must pass `src` via `PYTHONPATH` to the subprocess so module execution works without editable install.
  - `--no-emit-hidden-tracks` keeps tracker-maintained/interpolated boxes in the output but writes their `Hidden` attribute as `No` for CVAT relabeling; it does not disable internal tracking/association/occlusion state.
- Current runtime variants to compare are:
  - C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\eval\hybrid_bytetrack\Tracking moi tat smooth\yolov8
  - C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\eval\hybrid_bytetrack\Tracking moi bat smooth
- In reports, no smooth is currently worse than smooth; do not assume the unsmoothed runtime is the better baseline.
- Optimizer default scopes should stay tracking-focused.
- Detector-only presets were moved to explicit `--scope detector_probe` because `overnight_iou0` showed detector-only metrics were identical to `base`.
- Continue focusing on code and runtime behavior in association.py and runner.py, not detector weight.
- 2026-07-05 practical hard-set direction:
  - Use `hidden_owner_guard=true` + `hidden_owner_guard_hold_assignment=true` as the current practical opt-in base for hard-set work; it solves the `000231` frame-401 hidden-owner issue while preserving `000302=0` in later checks.
  - Do not keep tuning `reentry_ambiguous_hold` or `reentry_unowned_raw_mismatch_reject`/quarantine thresholds as the main path; those branches either damaged `000231`/`000302` or missed `000328`.
  - Next `000328` work should use a separate episode-level repeated unowned raw-ID mismatch detector, not per-assignment hold/reject rules.
# 2026-07-03 Best tradeoff found: lost-track reacquire split guards

- New strong 2-video tradeoff result:
  `outputs/eval/hybrid_bytetrack/20260703_193439/smooth_det020_loose/iou0_area0_condarea0_merge0/tracking_metrics.csv`
- Config was `smooth_det020_loose` plus `lost_track_reacquire_non_same_raw_distance_guard=false`.
- Key metrics:
  - `Pigs291119_000231_30fps`: IDSW `2`, HOTA `0.9705892717094201`, IDF1 `0.9847241970177549`.
  - `Pigs291119_000302_30fps`: IDSW `0`, HOTA `0.9930104703678451`, IDF1 `0.9964355605255801`.
  - `ALL`: IDSW `2`, HOTA `0.9820366705826231`, IDF1 `0.9907038986528682`.
- Preserve the current split `lost_track_reacquire_guard` design in `association.py` / `config.py`:
  - keep `lost_track_reacquire_guard=true`;
  - `lost_track_reacquire_non_same_raw_distance_guard=false` is now the default/base setting after the strong 9-video `20260703_194929` run;
  - do not disable raw-owner guard globally because it fixes `000302` but badly hurts `000231`;
  - preserve the conditional `lost_track_different_raw_hidden_owner_bypass` with `min_missed=2` and `min_center_gain=0.03`.
- `outputs/eval/hybrid_bytetrack/20260703_194929/smooth_det020_loose/iou0_area0_condarea0_merge0/` validated this as a good 9-video base; future tracking/eval/optimizer runs should not require a long override for this guard.

- 2026-07-05 successful candidate `20260705_220622`: hard-set remapped IDSW `000231=0`, `000263=2`, `000328=0`, `000302=0`, `ALL=2`. Candidate config: hidden-owner hold plus `reentry_unowned_raw_mismatch_episode_reject=true`, `reentry_unowned_raw_mismatch_episode_action=hold`, `episode_min_missed=1`, `episode_max_missed=20`, `episode_max_events=8`, `episode_max_cost=0.36`. Remaining `000263` frames `193/195` are track 3/4 fight/occlusion with raw IDs still consistent; user suspects possible GT ambiguity, so do not add broad guard before visual/GT confirmation.
- 2026-07-07 new 5-video candidate `outputs/eval/hybrid_bytetrack/20260707_082640/smooth_det020_loose/iou0_area0_condarea0_merge0`: improves weak `000233` without breaking hard guardrails. Remapped IDSW: `000231=0`, `000233=6`, `000263=2`, `000328=0`, `000302=0`, `ALL=8` versus `Best_tracking` `000233=9`, `ALL=11`. Add-on opt-in guard: `occlusion_reid_prefer_gap_over_bad_match=true`, `occlusion_reid_bad_match_action=reject`, raw mismatch + unowned raw + occlusion_hold only, `min_missed=7`, `max_missed=12`, `min_cost=0.55`, `max_cost=0.70`. This reject action intentionally does not consume the detection; the max-cost upper bound prevents `000231` frame-906 style regression.
- 2026-07-07 suffix repair candidate `outputs/eval/hybrid_bytetrack/codex_suffix_5video_min1500/iou0_area0_condarea0_merge0`: best current 5-video opt-in, remapped IDSW `000231=0`, `000233=6`, `000263=0`, `000328=0`, `000302=0`, `ALL=6`. Adds `suffix_pair_swap_repair=true` on top of protected practical config and 000233 guarded config. Key fix is suffix identity crossing for `000263` `Pig_3/Pig_4` after heavy overlap around frames `193/195`; default `suffix_pair_swap_min_suffix_frames=1500` avoids false suffix swaps seen with broad `60` frame setting on `000231/000233/000328/000302`. Keep opt-in until broader regression passes.
- 2026-07-07 new best 5-video opt-in candidate `outputs/eval/hybrid_bytetrack/codex_overlap_suppress_5video/iou0_area0_condarea0_merge0`: remapped IDSW `000231=0`, `000233=2`, `000263=0`, `000328=0`, `000302=0`, `ALL=2`. Adds `overlap_small_box_suppression=true` on top of the suffix candidate. This suppresses small low-confidence boxes in high-overlap frames (`min_iou=0.40`, `max_area_ratio=0.65`, `max_score=0.75`) and fixes the `000233` short box-crossing switches without breaking hard guardrails. Keep opt-in pending broader regression.
- 2026-07-07 current best 5-video opt-in candidate `outputs/eval/hybrid_bytetrack/codex_hidden_suffix_id_swap_5video/iou0_area0_condarea0_merge0`: remapped IDSW `000231=0`, `000233=0`, `000263=0`, `000328=0`, `000302=0`, `ALL=0`. Adds `hidden_suffix_id_swap_repair=true` on top of the overlap-suppress candidate. It catches the `000233` `ID_8/ID_1` low-confidence hidden suffix crossing around `1107-1118` using hidden-run length, max overlap, low median hidden score, and long suffix gates. Keep opt-in pending broader regression before base promotion.
- 2026-07-07 broader 12-video regression `outputs/eval/hybrid_bytetrack/20260707_174142/smooth_det020_loose/iou0_area0_condarea0_merge0` proved the full 5-video stack is not a safe common baseline. It improved `000233=0` and `000263=0`, but regressed previously clean videos: `000085: 0 -> 2` and `000225: 0 -> 2` remapped IDSW versus `Best_tracking`.
- 2026-07-07 ablation on `000085/000225/000233/000263`:
  - `ablate_control_assoc_occlusion_4video`: `000085=0`, `000225=0`, `000233=6`, `000263=2`.
  - `ablate_suffix_only_4video`: `000085=2`, `000225=2`, `000233=6`, `000263=0`; therefore current `suffix_pair_swap_repair=true` is unsafe and must not be promoted.
  - `ablate_overlap_only_4video`: `000085=0`, `000225=0`, `000233=2`, `000263=2`; `overlap_small_box_suppression=true` appears safe on this 4-video ablation.
  - `ablate_overlap_hidden_no_suffix_4video`: `000085=0`, `000225=0`, `000233=0`, `000263=2`; current safest common candidate is protected association/occlusion base plus `overlap_small_box_suppression=true` and `hidden_suffix_id_swap_repair=true`, explicitly with `suffix_pair_swap_repair=false`.
- 2026-07-07 full 12-video no-suffix common candidate `outputs/eval/hybrid_bytetrack/no_suffix_common_candidate_full/iou0_area0_condarea0_merge0`: remapped IDSW `ALL 11 -> 2` versus `Best_tracking`, with no per-video IDSW regression. Key per-video: `000085=0`, `000225=0`, `000231=0`, `000233 9 -> 0`, `000263=2`, `000302=0`, `000328=0`. This is now the safest broader candidate; remaining target is `000263=2` and must not be fixed with current `suffix_pair_swap_repair`.
