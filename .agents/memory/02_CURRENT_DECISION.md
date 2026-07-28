# Current Decision

## Confirmatory unseen method set frozen (2026-07-28)

- `PASS_UNSEEN_METHOD_SET_FROZEN`.
- Primary unseen method: frozen causal `realtime_fast`.
- Confirmatory technical baseline: frozen `bytetrack_raw`.
- `hybrid_bytetrack` and `rf_hybrid_offline` remain development-only
  ablations and are not authorized for unseen execution.
- Recommend revoking B1's legacy promotion in a separate lifecycle task;
  recommend not promoting R1. Runtime profiles are unchanged here.
- Future unseen comparison is confirmatory and whole-pipeline, including
  profile-specific detector cadence. It cannot support a pure
  association-core claim.
- Co-primary metrics are HOTA and IDF1. Primary identity-severity metrics are
  wrong-ID frames/seconds, terminal identity-error episodes, and persistent
  pairwise swaps.
- The next gate is a separate unseen-data authority freeze. No unseen access,
  execution, tuning, reselection, evaluation, or promotion is yet authorized.

## Development tracking 2x2 Standard-V2 complete (2026-07-28)

- `PASS_COMPLETE_DEVELOPMENT_2X2_AUTHORITY_ESTABLISHED`.
- B0/B1/R0 metric authorities were reused; only frozen R1 predictions were
  evaluated, twice, with `include_hidden=true`.
- R1: HOTA `0.878280776`, IDF1 `0.957881350`, IDSW `18`, wrong-ID frames
  `14,515`, and terminal identity-error episodes `14`.
- R0 remains strongest overall: it retains the best HOTA and IDF1, the lowest
  wrong-ID exposure, and the fewest terminal episodes.
- ByteTrack repair is `MIXED_TRADEOFF`; RF repair is `BROADLY_HARMFUL` under
  the predeclared metric hierarchy.
- B1 repair-event attribution is unavailable because its frozen promoted
  authority has no raw pre-repair output or ledger. R1 attribution is
  diagnostic only and is not used for the metric-level interaction.
- Cross-core results include profile-specific detector cadence and cannot
  support pure association-core or realtime claims.
- The next tracking gate is an unseen-method freeze decision. Unseen
  evaluation and promotion remain unauthorized.

## R1 frozen prediction authority established (2026-07-28)

- `R1_PREDICTION_GENERATION_DECISION=PASS_R1_PREDICTIONS_FROZEN`.
- Exact `rf_hybrid_offline` completed all 13 locked development videos from
  the frozen R0 even-frame cache, with no detector inference or evaluator.
- Frozen R0 public exports were not repair-contract complete, so generation
  used `EXACT_R1_PROFILE_EXECUTION`. A read-only guard proved raw RF-core
  parity with R0 before adapter and repair for every video.
- R1 retains 13 XML files and 187,200 prediction objects under
  `outputs/tracking/frozen_predictions_standard_v2_20260728_retry1/`
  `R1_rf_hybrid_offline/`.
- R1 prediction artifact authority is
  `40052f992871d50984fc4c0c839c4933b772bca2bfcaaaacafcde40d0e8a1800`.
- The frozen repair stack emitted 409 GT-free ledger events affecting 10,998
  frame records; 379 events used future frames. These are structural counts,
  not quality labels.
- R0 artifact modification count, detector inference, Standard-V2 and legacy
  metric runs, unseen access, and MP4 output were all zero.
- Structural parser, file-order, canonical prediction, ledger/event-ID, and
  inventory repeatability passed. Complete development 2x2 evaluation is now
  ready as a separate task; unseen evaluation and promotion remain forbidden.

## B0/B1/R0 Standard-V2 authorities established (2026-07-28)

- `REEVALUATION_DECISION=PASS_B0_B1_R0_STANDARD_V2_AUTHORITY_ESTABLISHED`
- Immutable B0, B1, and R0 predictions were evaluated twice with
  `TRACKING_EVALUATOR_STANDARD_V2` and `include_hidden=true`.
- Corrected aggregate HOTA is B0 `0.849511403`, B1 `0.849873389`, and
  R0 `0.888187232`; corrected IDF1 is B0 `0.920646368`,
  B1 `0.914081197`, and R0 `0.971892400`.
- Corrected `IDSW_STANDARD` is B0 `84`, B1 `64`, and R0 `29`.
- The old B1 > R0 > B0 headline ranking is not preserved. Current headline
  results must use Standard-V2; legacy HOTA/DetA/AssA remain historical
  non-standard diagnostics.
- B1 minus B0 remains the matched every-frame repair comparison. Comparisons
  with R0 are whole-pipeline effects including detector cadence, not pure
  association-core effects.
- Prediction immutability, TP/FP/FN conservation, wrong-ID conservation,
  pairwise-event conservation, two-pass repeatability, and input-order
  invariance all passed. Tracker/detector execution and unseen access were zero.
- Frozen metric artifacts are under
  `outputs/tracking/standard_v2_b0_b1_r0_reevaluation_20260728_retry1/`.
- R1 prediction generation may begin under its frozen method/cadence contract.
  Complete 2x2, unseen evaluation, runtime, promotion remain unauthorized.

## B0/B1 frozen prediction authorities established (2026-07-28)

- `PREDICTION_GENERATION_DECISION=PASS_B0_B1_PREDICTIONS_FROZEN`
- Exact `bytetrack_raw` and `hybrid_bytetrack` each completed all 13 locked
  development videos from the full-frame cache: 23,400 cache records per arm.
- B0 and B1 each retain 13 prediction XML files and 187,200 prediction objects
  under the non-disposable root
  `outputs/tracking/frozen_predictions_standard_v2_20260728_retry1/`.
- Detector inference, R0/R1 executions, evaluator/metric runs, unseen access,
  and MP4 output were all zero.
- B0 artifact authority is `13d9226c36141264cc33e4b498d38e5f3eaa9891cf32bc4c8fb87b01fd27d576`;
  B1 is `569c49e00905add068fac70c919fe21c10127e3ab773528a4ac44199fcb4835b`.
- B0/B1/R0 population, GT, source, detector model/config, and sequence
  boundaries pass the authority-level fairness gate. Detector row cadence is
  intentionally full-frame for B0/B1 and even-frame for R0.
- The next authorized tracking task is a separate Standard-V2 re-evaluation
  of frozen B0/B1/R0 outputs. Development 2x2, unseen evaluation, runtime,
  and promotion remain unauthorized.

## Full-frame development detector cache frozen (2026-07-28)

- `CACHE_DECISION=PASS_FULL_FRAME_DETECTOR_CACHE_FROZEN`
- The non-disposable cache authority contains 23,400 records: 1,800 frames
  for each of the locked 13 development videos.
- All 11,700 frozen R0 even-frame records remain byte-identical and
  `EVEN_SUBSET_PARITY=PASS`.
- The cache contains 11,700 unique newly inferred odd-frame records. Physical
  odd-frame calls were 12,100 because 400 identical odd-frame retries were
  required after a transient Windows heartbeat-file lock.
- Even-frame inference calls, tracker executions, metric runs, unseen-video
  accesses, and MP4 outputs were all zero.
- B0 and B1 may use the full-frame cache in a separately authorized prediction
  regeneration task. R0 remains bound to its frozen even-frame subset.
- Cross-core comparisons measure the whole pipeline, including detector
  cadence; a pure association-core effect claim is not authorized.
- Authority records are under
  `docs/tracking/full_frame_detector_cache/`.

## B0/B1 prediction regeneration blocked by detector cadence (2026-07-28)

- `PREDICTION_REGENERATION_DECISION=FAIL_COMMON_DETECTOR_REPLAY_CONTRACT`
- The surviving R0 cache authority remains valid and byte-verified.
- R0 cache evidence covers `0,2,...,1798` (900 records per video).
- Current `bytetrack_raw` and `hybrid_bytetrack` request detector evidence on
  every frame `0..1799` (1,800 records per video).
- No empty odd-frame evidence, cadence override, live inference fallback, or
  tracker change is scientifically authorized.
- B0/B1 tracker executions, detector inference calls, metric runs, unseen
  accesses, and MP4 outputs all remained zero.
- A future isolated task must first authorize full-frame detector evidence for
  the same locked 13-video development population.
- Evidence is under
  `docs/tracking/b0_b1_prediction_regeneration/`.

## Canonical tracking profile registry retired to three methods (2026-07-28)

- `ACTIVE_TRACKING_PROFILES=bytetrack_raw,realtime_fast,hybrid_bytetrack`
- `PRIMARY_REALTIME_PROFILE=realtime_fast`
- `PRIMARY_OFFLINE_PROFILE=hybrid_bytetrack`
- `RAW_BASELINE_PROFILE=bytetrack_raw`
- `realtime`, `realtime_balanced`, `realtime_quality_delayed`, and
  `realtime_fast_h1_r2` are unavailable for active execution.
- Historical manifests keep their stored profile names and all H1/H2
  scientific archives remain authoritative historical data.
- Retained algorithms, detector authority, and hybrid repair semantics did not
  change. The retirement evidence is under
  `docs/tracking/profile_retirement/`.
- The next tracking task must start from updated `main` in a new isolated
  worktree. Do not implement `rf_hybrid_offline` in the retirement worktree.

## Fresh current-main R0 baseline established (2026-07-28)

- `R0_BASELINE_AUTHORITY=ESTABLISHED`
- `R0_PROFILE=realtime_fast`
- `R0_CODE_SHA=64d835cbf1b25ecdef3a777a50f0b46db6c93f61`
- `R0_VIDEOS_COMPLETED=13/13`
- Aggregate: HOTA `0.9704398315450558`, IDF1 `0.9707702337312571`,
  IDSW `53`, FP `486`, FN `610`, fragments `107`.
- Exact detector-cache replay used `11,700` records; detector inference during
  tracking and run-root MP4 count were both zero.
- Per-video metrics and complete hash inventories are archived under
  `outputs/tracking/current_main_baseline_20260728`.

H2-CDSP is formally closed as `FAIL_NO_CURRENT_MAIN_STATE_LOSS`. The current
main error taxonomy, historical reconciliation, and next-hypothesis selection
are deferred. Do not implement `rf_hybrid_offline` in this worktree.

## H2-CDSP current-main shadow failed: no baseline state loss (2026-07-28)

- `H2_CURRENT_MAIN_SHADOW_DECISION=FAIL_NO_CURRENT_MAIN_STATE_LOSS`
- `CURRENT_MAIN_BOUNDED_REPRODUCTION=10/10_WINDOWS_COMPLETED`
- `H2_BASELINE_STATE_LOSS_POINTS=0`
- `H2_POSITIVE_WINDOWS_WITH_STATE_LOSS=0`
- `H2_EXTRA_USABLE_STATE_AT_REENTRY=0`
- `SHADOW_BASELINE_OUTPUT_EQUIVALENCE=PASS`
- `CACHE_FRAMES_LOADED=1049`
- Detector inference, GPU inference, validation execution, and MP4 output were
  all zero.
- Production implementation, association evaluation, validation, runtime, and
  promotion remain unauthorized.

The cache-authority and live-main tracking trees were byte-equivalent. Two
positive windows reproduced identity errors, but none reproduced the frozen
H2 baseline state-loss mechanism. All four controls also triggered the frozen
broad-preservation audit flag. This is bounded current-main evidence, not a
global prevalence claim. Historical `b0d9009` evidence remains
mechanism-discovery only.

## H2-CDSP design passed; shadow remains unauthorized (2026-07-27)

- `H2_HYPOTHESIS_NAME=CAUSAL_DROPOUT_STATE_PRESERVATION`
- `H2_DESIGN_CHECKER=PASS`
- `INDEPENDENT_REVIEW=PASS_DESIGN`
- `READY_FOR_CURRENT_MAIN_SHADOW_AUTHORIZATION=YES`
- `H2_IMPLEMENTATION_AUTHORIZED=NO`
- `H2_SHADOW_EXECUTION_AUTHORIZED=NO`
- Association evaluation, validation, runtime, and promotion are unauthorized.

H2 is a bounded state-preservation hypothesis, materially distinct from H1.
It exposes only a `PreservedStateEvidence` record; it does not reserve or
directly assign detections. The future current-main shadow must reproduce real
baseline state loss on at least two video keys and two recording sessions,
remain side-effect-free, and bind exact code, configuration, cache, source,
GT, and canonical-output lineage. Historical `b0d9009` evidence remains
mechanism-discovery only.

## Hidden-owner line closed; next design candidate only (2026-07-27)

- `HIDDEN_OWNER_PREFERENCE_FAMILY_STATUS=CLOSED_FOR_CURRENT_STUDY`
- `H1_R4_AUTHORIZED=NO`
- RF_ACC23 audit population: 10 events, 4,922 wrong-ID matched frames, and
  53 conserved ID-switch rows.
- Dominant mechanism: `OCCLUSION_OWNER_LOSS`, six events and 2,092 wrong-ID
  frames (42.503%).
- `NEXT_HYPOTHESIS_DECISION=PROPOSE_ONE_NEW_HYPOTHESIS`
- `NEXT_HYPOTHESIS_NAME=CAUSAL_DROPOUT_STATE_PRESERVATION`
- New implementation, tracking runs, validation, runtime, and promotion are
  unauthorized.

The next hypothesis is design-only. It must target bounded causal track-state
preservation through measured detector dropout or merge conditions and must
not become H1-r4 or another hidden-owner preference threshold. The recovered
RF_ACC23 lineage was measured at `b0d9009`; exact tracking-tree equivalence to
the later promoted main remains unproven.

## H1-r3 shadow prerequisite failed: no activation (2026-07-27)

- `H1_R3_SHADOW_PREREQUISITE_DECISION=FAIL_NO_SHADOW_ACTIVATION`
- `H1_R3_SHADOW_PAIR_CANDIDATES=1518`
- `H1_R3_SHADOW_CORE_ELIGIBLE_PAIRS=958`
- `H1_R3_SHADOW_SCORE_PAIRS=958`
- `H1_R3_SHADOW_WOULD_ACTIVATE=0`
- `H1_R3_IMPLEMENTATION_AUTHORIZATION_READY=NO`
- Association implementation, evaluation, validation, runtime evaluation, and
  promotion remain unauthorized.

The disabled observer did not change any development output. Its score maximum
was `0.59954303046`, below the frozen threshold `0.625`; no positive episode
activated. This closes the authorized prerequisite as a scientific failure,
not as evidence of tracking quality or safety under intervention. Do not tune
the frozen threshold or margin, and do not execute more association-changing
runs without new authority.

## H1-r3 design passed independent review (2026-07-27)

- Primary eligibility:
  `symmetric_iou_recency_core_with_conservative_optional_bounds`.
- `H1_R3_SCORE_NAME=owner_preference_lower_bound`
- `H1_R3_THRESHOLD=0.625`
- `H1_R3_MARGIN=0.25`
- `H1_R3_DESIGN_CHECKER=PASS`
- `INDEPENDENT_DESIGN_REVIEW=PASS_DESIGN`
- `H1_R3_IMPLEMENTATION_AUTHORIZED=NO`
- `H1_R3_EVALUATION_AUTHORIZED=NO`
- `H1_R3_RUNTIME_AUTHORIZED=NO`
- `H1_R3_PROMOTION_AUTHORIZED=NO`

The final design uses relative IoU and freshness as its nonredundant scored
core. Optional appearance and motion use conservative uncertainty intervals;
activation uses the worst-case lower bound, so masking cannot increase hidden
confidence. Golden features are recomputed from explicit realizable boxes,
descriptors, predictions, ages, and LK states.

A separately authorized telemetry-only phase must keep reservation disabled,
preserve assignments, and use score-blind development owner labels. If the
frozen conservative gate has no development operating region, close the
hidden-owner preference hypothesis rather than tune another iteration.

## H1-r2 missingness audit: multiple design failures (2026-07-27)

- `H1_R2_FROZEN_DEVELOPMENT_DECISION=FAIL_NO_ACTIVATION`
- `FEATURE_PLUMBING_DEFECT_FOUND=NO`
- `MISSINGNESS_CONTRACT_TOO_RESTRICTIVE=YES`
- `SCORE_OPERATING_RANGE_FEASIBLE=NO`
- `ROOT_DIAGNOSIS=MULTIPLE_DESIGN_FAILURES`
- `NEXT_ACTION=DESIGN_H1_R3`
- H1-r3 implementation, tracking evaluation, validation, runtime evaluation,
  and promotion remain unauthorized.

The immutable `770`-row export passed its hash and population checks. All eight
feature values and optional-evidence masks were valid on both sides in every
row. The `737` `missing_evidence` abstentions instead exactly equal the rows
whose present hidden overlap was below the hidden-only `0.50` eligibility
floor. This is a restrictive and misleadingly labeled contract gate, not
missing feature plumbing.

Of the `33` scored pairs, none passed the frozen score-plus-margin gate.
Diagnostic score-only cutoffs did not separate likely beneficial evidence from
harmful and ambiguous evidence, so the audit authorizes no threshold change.
Equal arm metrics remain no-effect evidence; safety under real activation is
`NOT_MEASURED`. Validation artifacts stayed hash-identical and validation was
not executed.

## H1-r2 development evaluation failed: no activation (2026-07-27)

- `H1_R1_STATUS=SCIENTIFICALLY_CLOSED`
- `GENERIC_CACHE_MERGE_STATUS=COMPLETED`
- `H1_R2_IMPLEMENTATION_COMPLETE=YES`
- `H1_R2_DEVELOPMENT_DECISION=FAIL_NO_ACTIVATION`
- `H1_R2_VALIDATION_EVALUATION_AUTHORIZED=NO`
- `H1_R2_RUNTIME_EVALUATION_AUTHORIZED=NO`
- `H1_R2_PROMOTION_AUTHORIZED=NO`

The frozen six-episode development population was evaluated with paired
`realtime_fast` and `realtime_fast_h1_r2` cache-only replay. All 770 candidate
pairs were exported: 737 abstained for missing evidence, 33 were valid but
below threshold, and none applied. All four positives and both controls had
zero real activation and zero association-output divergence.

Baseline and candidate quality metrics are therefore equal, but this is
no-effect evidence rather than H1-r2 benefit. Aggregate remapped IDSW is
`21` in both arms and wrong-ID matched frames are `135` in both arms.
Shared-cache parity, causal delay zero, prefix invariance, no-future-frame,
validation-hash blindness, and zero-MP4 checks pass. The exact repeat was not
run because the first run hit the frozen no-activation hard stop.

The score remains uncalibrated and is not a probability. Threshold `0.60`,
coefficients, missingness rules, production profiles, and source code were not
changed. Validation, runtime evaluation, threshold tuning, and promotion
remain unauthorized. The locked decision is
`docs/tracking/h1_r2/H1_R2_DEVELOPMENT_EVALUATION_DECISION_20260727.json`.

## Classification V2 correction accepted (2026-07-26)

- `SCIENTIFIC_ACCEPTED_SHA=a35e0b9aae8b55167b4562cfc7e26a45e2b4e312`
- `OPERATIONAL_FINAL_EXECUTION_SHA=PENDING_EXACT_SHA_AUDIT`
- Operational finalization is a separate execution/configuration authority;
  it does not alter the accepted scientific implementation.

- `PHASE4_IMPLEMENTATION_SHA=a35e0b9aae8b55167b4562cfc7e26a45e2b4e312`
- `ACCEPTED_IMPLEMENTATION_SHA=a35e0b9aae8b55167b4562cfc7e26a45e2b4e312`
- `OBJECT_TRACK_KEY_EXACT_SHA_REAUDIT=PASS`
- `PHASE1_4_INTEGRATED_ACCEPTANCE=PASS`
- `PHASE4_HUMAN_SIGNOFF=APPROVED`
- `REVIEWER=TuanHM`
- `REVIEW_DATE=2026-07-26`
- `REVIEWED_SHA=a35e0b9aae8b55167b4562cfc7e26a45e2b4e312`
- `MAIN_SYNC_STATUS=COMPLETE`
- `READY_FOR_LINEAGE_REBUILD_PLANNING=YES`
- `READY_TO_REBUILD_FRAME_LOCAL=NO`

The explicit object identity contract is
`schema.classification_v2.object_track_key`,
`classification_v2.object_track_key.v1`: escaped source, dataset, video,
then `track_id`, with `object_id` fallback, serialized using RFC3986 UTF-8
escaping. `PIG_ID_AUTHORITATIVE=NO`; production key bytes did not change.
The post-merge Group A, conformance, negative-control, two-root integrated,
semantic invalidation, and release-authority gates all passed.

The V6 ignored-input and legacy `G:\My Drive` failures remain pre-existing
environmental limitations. This status authorizes planning readiness only;
no lineage rebuild, GUI review, model execution, or training was started.

Current source-path authority is the completed P0-P10 run
`outputs/legacy_16f_rebuild/legacy_16f_rebuild_20260718_v2` plus the exact
12 behavior XMLs, ROI JSON and video root listed in README. The prior
`legacy_full_multigt_masked_nodup_16f` paths are superseded and are not active
source authority.

## Classification V2 acceptance reopened (2026-07-25)

- `PHASE4_IMPLEMENTATION_SHA=76a0458e39769d3e7fac865dd16439a0ed3c3a04`
- `ACCEPTED_IMPLEMENTATION_SHA=76a0458e39769d3e7fac865dd16439a0ed3c3a04`
- `PHASE4_EXACT_SHA_AUDIT=PASS`
- `PHASE1_4_INTEGRATED_ACCEPTANCE=REOPENED`
- `PHASE4_HUMAN_SIGNOFF=APPROVED`
- `REVIEWER=TuanHM`
- `REVIEW_DATE=2026-07-24`
- `REVIEWED_SHA=76a0458e39769d3e7fac865dd16439a0ed3c3a04`
- `MAIN_SYNC_STATUS=CODE_INTEGRATED_BUT_ACCEPTANCE_REOPENED`

The post-main differential gate collected the same 32 nodes on the accepted
SHA and the documentation SHA. Each baseline produced 22 passes and the same
10 failures:

- `test_frame_local_independent_checker_detects_content_drift`:
  `ACCEPTED_IMPLEMENTATION_DEFECT`. The production builder emits the
  contract `object_track_key`, while the independent checker derives an
  incompatible legacy-format key. It fails both in and outside the sandbox.
- `test_actual_v6_apply_and_independent_checker_pass`:
  `PREEXISTING_MISSING_IGNORED_ARTIFACT`. Its three mandatory V6 CSVs are
  Git-ignored and absent from both fresh worktrees. With the existing inputs
  available outside sandbox restrictions, the node passes.
- The eight initially failing `test_truy_nguon_multi_bbox.py` nodes:
  `PREEXISTING_ENVIRONMENT_OR_PATH`. Sandboxed Python cannot stat
  `G:\My Drive`; outside that isolation, both SHAs pass all 17 nodes in the
  file.

No failure was introduced by the five documentation files. The preserved
stash was not applied during this classification and remains non-authority.

Decision:
`ACCEPT_PHASE_4_AND_PHASE_1_4_INTEGRATED_IMPLEMENTATION`
was the historical human sign-off. The current controlling decision is
`REOPEN_PHASE1_4_INTEGRATED_ACCEPTANCE_FOR_IMPLEMENTATION_CORRECTION`.

The accepted implementation SHA remains the scientific implementation
authority. `MAIN_HEAD_AFTER_DOCUMENTATION_SYNC=THIS_DOCUMENTATION_COMMIT`
identifies the one later documentation-only commit; its exact SHA is recorded
by Git commit metadata and does not replace or alter accepted authority.

Original audit evidence:
`.codex_tmp/worktrees/phase1_4_final_acceptance_76a0458/outputs/`
`classification_v2/agent_audits/phase1_4_final_acceptance_76a0458/`.
Stable archive:
`C:\Users\ironh\Downloads\PIG_Behavior_Project_AUDIT_ARCHIVE\`
`phase1_4_final_acceptance_76a0458\`.
The package contains 87 files; source and archive SHA-256 inventories match.
The exact report is `final_phase1_4_acceptance_report.md`; the checklist is
`final_phase1_4_acceptance_checklist.md`. Audit target is the accepted SHA and
the verdict is `PASS_PHASE1_4_INTEGRATED_ACCEPTANCE`.

### Accepted integrated proof

- Production built 28 candidate manifests; all 28 validated.
- Audit harness assembled zero candidate manifests.
- All 26 upstream links were `CURRENT_AUTHORITATIVE`; zero noncurrent
  upstreams were accepted.
- Original negative controls passed 14/14 integrated and 15/15 builder.
- Extended negative controls passed 7/7.
- Transaction injections passed 9/9 without a prior candidate and 9/9 with a
  prior candidate.
- Zero new valid candidates and zero partial authoritative outputs survived
  failed transactions.
- LF/CRLF canonical equality and cross-root determinism passed.
- Runtime dependency closure passed 17/17.
- Phase 1, Phase 2, Phase 3, and Phase 4 invariants passed.
- Phase 2 schema hash remains
  `ec0c511b5f5198240492be49c0492e543c9e38eb4a4ff446259b958c2a59963b`.
- Every release authorization flag remained false.

### Current readiness

- `READY_FOR_LINEAGE_REBUILD_PLANNING=NO`
- `READY_TO_REBUILD_FRAME_LOCAL=NO`
- `READY_FOR_SOURCE_REBUILD=NO`
- `READY_FOR_HIDDEN_REVIEW=NO`
- `READY_FOR_TEMPORAL_HARMONIZATION=NO`
- `READY_FOR_NATIVE_EVIDENCE=NO`
- `READY_FOR_PIG_STRENET=NO`
- `READY_FOR_BEHAVIOR_GUI=NO`
- `READY_FOR_TRAIN_READY_EXPORT=NO`
- `READY_FOR_TENSOR_EXPORT=NO`
- `READY_FOR_MODEL_EXECUTION=NO`
- `READY_FOR_TRAINING=NO`

Lineage rebuild planning is blocked. A separate correction task must reconcile
the production and independent-checker `object_track_key` authority, then
repeat the affected exact-SHA acceptance gate. The planning paragraph below is
historical and superseded; this record authorizes neither that correction nor
any execution.

The next task is planning only: determine the authoritative rebuild start;
validate whether source merge remains `CURRENT_AUTHORITATIVE`; design an
isolated lineage run ID; lock input artifacts and manifests; define stage
output roots, dry-run and preflight gates, population reconciliation,
human-decision carry-forward, and rollback; then prepare a separate
frame-local execution authorization.

Pre-sync dirty work is preserved in a backup branch, external recovery
package, and unapplied stash. It was not applied to `main`.

Earlier Phase 4 candidates `c1df0cc4bc72f9f4a889a9fe8f5fe6b31cffb45a`,
`f42f0d3328700d1f3c2f6bcca01f147b145c8a07`, and
`99d63723bc1af8cc862551c045c9b2675076acce` remain historical records,
superseded by the final integrated accepted SHA.

## Active realtime runtime decision 2026-07-20

The common GPU harness is complete at instrumentation commit `7c9179e`.
`realtime_fast` and `realtime_balanced` both remain causal and prediction-
repeatable, but neither reaches the 30 FPS native gate. Fast reaches
`27.3398/27.6077` FPS (primary/repeat); Balanced reaches `28.7192/28.8994`.
Balanced has lower backlog and output age, while Fast retains the full-13
identity advantage. Therefore there is no native operational winner yet.
Do not claim a speed winner or silently label either profile native realtime;
future work needs a throughput improvement or an explicit drop policy.

The runtime decision is recorded in
`docs/TRACKING_REALTIME_RUNTIME_PROBE_DECISION_20260720.json`. The added
deadline, backlog and output-age telemetry is prediction-invariant and was
validated with 217 tracking tests. `bytetrack_raw` was not rerun.

## Active realtime Pareto decision 2026-07-20

Promote the validated far-right close-competitor guard in `realtime_fast`.
The full-13 candidate/repeat pass causal, prefix, lineage, repeatability and
zero-MP4 gates: IDSW `69 -> 59`, HOTA `94.35% -> 95.63%`, IDF1
`93.91% -> 95.37%`, FP/FN `506/630 -> 486/610`, and `000302` IDSW `6 -> 0`.
The profile commit is `74cad2b`, separate from algorithm commit `62f140b` and
the evidence commits.

Keep `bytetrack_raw` fixed at its existing authority (`145`, `88.91%`,
`88.47%`) and do not rerun it. Its runtime is historical labeled evidence;
the common-harness runtime gate now proves that neither causal profile is
native realtime. Balanced stays a valid quality reference, while Quality stays
in the Pareto comparison as delayed/finite-delay evidence; only its current
realtime-winner eligibility is withheld. Authority:
`docs/TRACKING_REALTIME_PARETO_SELECTION_DECISION_20260720.json`.

The first post-Fast Balanced family was screened only on difficult windows.
The far-right threshold candidate tied its parent on both W01 `000327` and W02
`000302`; therefore it was rejected before W03-W05, full video and full-13.
Keep `realtime_balanced` unchanged and retain the negative evidence at
`docs/TRACKING_BALANCED_FAR_RIGHT_GUARD_DECISION_20260720.json`.

## Active hybrid lane-completion decision 2026-07-19

Lock `hybrid_bytetrack_best` at IDSW `0`, HOTA `98.35062270%`, IDF1
`99.14903846%`, and FP/FN `1593/1593`. All 13 videos have zero IDSW, and the
remapped identity-event artifact has zero rows. The two frozen residual
clusters, `000233` H5b and `000328` H4, are resolved with repeatable,
hash-bound evidence and zero MP4.

Close the hybrid optimization lane. Remaining weak rows are localization and
continuity residuals, led by `000216`, rather than a bounded identity-switch
family. Reopen hybrid only for a predeclared failure on untouched sessions.

Hybrid is the first dependency and is complete. The realtime selection is not
pre-ranked as Fast, Balanced, then Quality: Fast is the causal operational
control, while every valid Fast, Balanced, and Quality authority enters the
same Pareto selection. Quality cannot be skipped; if a truthful causal or
finite-delay Quality candidate wins, it replaces Fast/Balanced in the paper.
RB3 Balanced is closed because repeat p95 failed, and the current finite-delay
Quality screen is now also closed without promotion.

The paper-critical comparison is `bytetrack_raw` -> one selected valid realtime
method -> `hybrid_bytetrack_best`, all under the same include-Hidden contract,
video universe, detector, GT, and evaluator. Three realtime profiles are not a
deliverable by themselves. Quality remains a selection challenger: if a causal
or finite-delay version passes prefix/latency gates and wins the declared
Pareto comparison, it replaces fast/balanced as the paper's realtime method and
the main comparison becomes raw -> Quality -> hybrid.
The current delay-`-1` global-post-video implementation is eligible only as a
separately labeled delayed method until that realtime contract is proved.

The no-regression rule protects accepted overall evidence, not an objectively
weak implementation. A profile whose implementation conflicts with its named
speed/quality role may be superseded or rebuilt. Keep its old result as labeled
negative/baseline evidence, freeze the replacement design before testing, and
promote only against the relevant raw and operational references through the
window -> video -> hard-set -> full-13 funnel.

## Active realtime Quality selection gate 2026-07-19

`realtime_quality` is a mandatory selection challenger before any realtime
winner is locked. This does not require all three realtime profiles in the
paper table; it requires a fair validation attempt under a truthful causal or
finite-delay contract.

Retain the existing Quality authority in the Pareto/report evidence as a
post-video upper bound (`IDSW 166`,
HOTA `97.66%`, IDF1 `97.58%`, delay `-1`), not as a realtime winner. RQ1 lags
`12/15/30` all failed frozen QW01: IDSW improved `36 -> 32`, but HOTA fell
`91.51% -> 91.12%` and IDF1 fell `91.17% -> 89.88%`. Do not run later RQ1
stages. RQ2 S1 then improved QW01 HOTA/IDF1, but QW02-QW04 produced no second
independent gain and its `18.99 FPS` missed the frozen `24.08 FPS` floor. RQ2 is
closed without promotion. RQ3 horizons `10/15/20` preserve the S1 prediction,
but the fastest horizon reaches only `17.20 FPS` and fails p95. RQ3 is closed.
RQ4 retains an output-equivalent `2.221x` copy-performance improvement, but its
QW01 primary/repeat reach only `19.42/16.91 FPS`, p95 `60.84/91.02 ms`, and a
repeat loop-FPS ratio of `0.856`. Quality therefore remains unpromoted; do not
run later windows, a full video, hard set, or full-13 from RQ4.

Authority:
`docs/TRACKING_REALTIME_QUALITY_SELECTION_GATE_20260719.json`.
RB3 decision:
`docs/TRACKING_RB3_RESERVED_REID_HOLD_DECISION_20260719.json`.
RQ1 decision and RQ2 plan:
`docs/TRACKING_RQ1_FIXED_LAG_DECISION_20260719.json` and
`docs/TRACKING_RQ2_QUALITY_ID_SAFE_PLAN_20260719.json`.
RQ2 decision and RQ3 plan:
`docs/TRACKING_RQ2_QUALITY_ID_SAFE_DECISION_20260719.json` and
`docs/TRACKING_RQ3_QUALITY_RUNTIME_HORIZON_PLAN_20260719.json`.
RQ3 decision and RQ4 plan:
`docs/TRACKING_RQ3_QUALITY_RUNTIME_HORIZON_DECISION_20260719.json` and
`docs/TRACKING_RQ4_QUALITY_COPY_PERFORMANCE_PLAN_20260719.json`.
RQ4 decision:
`docs/TRACKING_RQ4_QUALITY_COPY_PERFORMANCE_DECISION_20260719.json`.

## Active realtime Pareto selection 2026-07-19

Quality was screened as a mandatory challenger and is not silently omitted:
the delay-`-1` graph is post-video upper-bound evidence, while RQ4's finite
delay candidate fails effective FPS, p95, and repeat runtime ratio. RB1/RB2
were rejected at the first frozen window for IDF1/HOTA regression; RB3's
repeatable IDSW gain fails p95 and the locked Balanced identity target.

Select `realtime_fast` as the current causal reference because it has the
lowest zero-delay IDSW (`69` versus Balanced `121`) with repeatability PASS.
This is a multi-objective reference decision, not an accuracy-only ranking:
causal validity, prefix integrity, no-MP4 lineage, measured throughput,
latency, memory and delay remain selection dimensions. Fast's timing is
retained as evidence, but no speed advantage is claimed until raw/Fast/Balanced
are measured with one common harness and pass the frozen runtime gates.

Fast is not the final winner yet: its profile-specific `000302` guard is open
(`6` observed versus the frozen ceiling `2`). Run a targeted include-Hidden
follow-up before promoting it as the operational choice.

Balanced remains a useful causal quality reference (`HOTA 95.68%`,
`IDF1 95.76%`, FP/FN `448/586`) but its IDSW gap is still `52`; Quality has no
valid finite-delay runtime authority. The raw same-contract authority now
passes with IDSW `145`, HOTA `88.91%`, IDF1 `88.47%`, loop-FPS `22.65/27.03`,
repeatability PASS and zero MP4. Therefore raw -> Fast -> hybrid is the planned
paper chain, not yet a locked claim; the Fast guard and common runtime audit
must pass before a realtime method is selected.

Authority:
`docs/TRACKING_REALTIME_PARETO_SELECTION_DECISION_20260719.json`.

## Decision precedence

Only the active decision immediately below controls current work. All later
sections are historical records. The active tracking authority is
`docs/TRACKING_HYBRID_LANE_COMPLETION_DECISION_20260719.json`.
That authority closes hybrid and opens realtime planning. No realtime
candidate is authorized until a separate Pareto and funnel plan is frozen.
`docs/CLASSIFICATION_V2_CURRENT_STATE.md` applies only to the paused
classification workstream.

## Active source-lineage decision: reviewed mixed legacy + XML

The current rebuild target is one mixed source lineage containing the locked
legacy 16f P0-P10 export and the exact 12 behavior XML files under
`data/annotations/classification`. The merged source must retain source type,
input hashes, row counts and an output hash. XML files under
`data/annotations/tracking` are not behavior authority for this lineage.

Legacy and XML remain distinct native contracts during review: legacy native
units are 16-frame units and XML native targets are six-frame units. They are
merged for downstream review/training only after source audit; no legacy-only
metric or review decision transfers automatically. Mixed data remains blocked
from training until both review layers and all downstream gates pass.

## Active human-review execution authority

Hidden smoke uses `%HSM%` and `%HSMDEC%` only. `%HDEC%` is reserved for the
full `%HREV%` manifest; smoke decisions never satisfy or seed full authority.

Complete legacy 16f behavior review requires both
`--include-all-retained-legacy-units` at review-unit build time and
`--require-complete-legacy` at template coverage time. This binds every
retained 16-frame legacy native unit into the human review manifest. The
default selective queue remains valid for bounded diagnostics but cannot earn
complete legacy behavior-review status.

Behavior review consumes same-lineage Pig-STRENet evidence after temporal
harmonization. History transitions require complete history and target;
invalid history is masked and cannot create conflict evidence. The GUI may
show valid history before target frames, but evidence never auto-changes label.

The behavior queue is five-way: mandatory census, high-risk,
probability-sampled random residual audit, clean control and not-selected.
Only random supports an inverse-probability-weighted residual intervention
estimate. High-risk and clean-control results are separate; neither means
not-selected units are clean.

Write the exact behavior scientific design before the first decision. A final
training snapshot requires exact selected-unit coverage and a PASS gate with
source/video/native-unit clustered uncertainty. Every review_pig and sampling
field remains outside model-X.

## Active model-search authority: controlled screen, joint tune, confirm

Use this exact five-stage flow for classifier research and the future rented-
GPU lane:

1. Build a sufficiently strong and stable base, with only enough tuning to
   serve as a reliable measurement instrument.
2. Screen modalities by a predeclared ladder: seven singles, all 21 pairs,
   beam-search triples and larger subsets, then leave-one-out confirmation.
   Every subset uses parameter-matched-zero, availability-only, and real-value
   controls on identical folds, seeds, exposure, and optimization budget.
3. Freeze the selected modality set before comparing fusion structures. Do not
   change subset and fusion architecture in one uncontrolled comparison.
4. Jointly tune the visual backbone, temporal model, and selected fusion at
   larger scale on rented GPUs.
5. Repeat confirmatory modality ablations on the tuned strong model before
   locking the candidate.

Do not deeply tune a potentially information-incomplete base before modality
screening, and do not treat screening on a controlled base as the final model
decision. Local RTX 3050 limits correctness-batch placement only. Preserve all
valid older runs under their exact seed/config lineage and reuse their caches,
predictions, and diagnostics whenever contracts match.

`combined_all7__real` is a retained endpoint/stress test, not an optimization
result. It cannot close subset search or fusion-family search. A negative
modality result also cannot become `DROP` until diagnostics distinguish
bad/absent input, no extractable signal, actor redundancy, underpowered data,
optimization failure, and encoder/fusion-capacity failure. Required probes are
modality-only, actor-residual, within-stratum permutation, learning/gradient,
and at least one stronger mask-aware fusion control.

Compute boundary: legacy 16f is a method/correctness host, not the lineage for
exhaustive subset selection. Preserve its singles and all-seven endpoint; add
only synthetic or representative pair canaries needed to validate the ladder
and failure taxonomy. The full 21-pair, beam and leave-one-out search begins on
the frozen reviewed main lineage, where class support and source relevance are
adequate.

## Active Pig-STRENet artifact authority

Causal history uses only `[s-6..s-1] -> [s..s+5]`. Never read after `s+5`.
Legacy pair manifests must carry both relative burst coordinates and actual
source-frame coordinates. XML pairs use actual target coordinates directly.
Completeness, masks and gaps must be computed in one coordinate system.

Legacy 16f has persistence by construction because one label was propagated
through a burst. History conclusions therefore require XML-only, legacy-only,
source-balanced and history-by-source reports. XML reviewed targets are the
primary evidence for transfer; legacy is a development canary and supporting
training source only.

Derived windows conserve native-event mass: all pair weights from one native
event sum to `1.0`. Primary evaluation collapses predictions back to the native
event. Overlapping derived-window metrics are diagnostic only.

The predeclared controls are `T0/T1/H0/HA/HS/HR/HRev/PM`. Availability,
completeness, gap and duration fields are audit/mask metadata by default and
enter only the HA missingness control. Social partners are geometry-selected,
fixed top-K with explicit masks, and never selected by behavior labels.

Canary09 passed the exporter/audit contract under
`outputs/classification_v2/agent_audits/pig_strenet_20260719_canary09`.
This authorizes artifact reuse only. Trainer/fusion/GAT/ROI-Align integration,
training, OOF and model promotion remain unimplemented and unauthorized.
The media bridge now resolves lineage-bound scene frames from the real source
video (`video_key/source_video_path + frame_index`) and actor crops from either
existing crop files or the same video plus bbox. `background.png` and static
`Image #1` candidates are rejected as temporal scene media.

The real XML bounded canary at
`outputs/classification_v2/agent_audits/pig_strenet_xml_real_20260719_canary01/`
`07_pig_strenet_attempt2`
also passes the artifact schema: 2,400 native pairs, 28,800 slots, event mass
`1.0`, no future-frame use, and the same model-X/ROI/social tensor schemas as
Canary09. Eight frame-zero pairs correctly have unavailable six-frame history;
the other 2,392 pairs are complete. This remains
`xml-only-unreviewed-technical-canary`: it is export/lineage evidence only, not
accuracy evidence, review completion, training authorization or promotion.

The corrected media-bridge canaries are:

- legacy: `pig_strenet_media_bridge_legacy_20260719_canary11`, difference
  `PASS` for 96/96 actor slots and ROI pixels `PASS` for 288/288;
- XML: `pig_strenet_media_bridge_xml_20260719_canary02`, difference `PASS`
  for 2,400/2,400 pairs and ROI pixels `PASS` for 86,256/86,256 expected rows.

Both have `media_manifest.valid=true`, `background_as_temporal_scene_used=false`,
and no future-frame use. The 144 XML ROI rows outside the expected count are
natural frame-zero history slots, not unresolved media. The prior blocked result
was an exporter-resolution limitation, not evidence that source pixels were
absent. `max-native-events` now limits target units without truncating the full
frame table needed for causal history and scene/social context.

## Active decision: reviewed-data rebuild
## Completed tracking-only roadmap (merged 2026-07-20)

The user explicitly moved tracking work to `PIG_task_tracking` on branch
`task/update-tracking`. Do not touch classification code, data, or model work.

### Critical-path override 2026-07-18

Complete and authority-lock `hybrid_bytetrack` before any realtime transfer.
Near-wall Hidden bbox geometry is now promoted in `hybrid_bytetrack_best`:
IDSW stays `8`, FP/FN improve `1630 -> 1622`, HOTA improves
`98.31% -> 98.32%`, and raw IDF1 increases. Primary/repeat and all
non-geometry payload checks pass with zero MP4.

The promotion above closes only the near-wall experiment. Hybrid remains the
active optimization lane because its authority still has eight IDSW. Four are
the `000233` identity conflict at frames `1111-1114`; four are the `000328`
far-camera Hidden bbox conflict at frames `1347-1355`. Diagnose and test these
as separate families through hard-window, full-video, hard-set, and only then
full-13 gates. Realtime stays frozen until a separate hybrid lane-completion
decision records the residual audit and stop-gate evidence.

Realtime notes below remain evidence but no longer set task order. The next
realtime study must use `realtime_fast` as its operational reference.
`realtime_balanced` is not considered solved merely because it improves over
its older baseline; it must pass a predeclared identity-stability and latency
gate and provide material value relative to fast.

Tracking GT was seeded by an older tracker and manually corrected for bbox/ID.
Primary evaluation therefore uses `include_hidden=true`; the 1,930
tracker-derived `Hidden` values are not a visibility target. Exclude-Hidden
metrics are compatibility evidence only.

The causal `realtime_balanced` profile now promotes hidden-detection
reservation at `min_iom=0.96`, `min_gain=0.17`,
`max_alternative_cost=0.25`, visible hold enabled, and `hold_min_gain=0.17`.
Full-13 primary and repeat improve IDSW `133 -> 121`, IDF1 `93.71% -> 95.76%`,
HOTA `93.93% -> 95.68%`, FP/FN `449/587 -> 448/586`, and fragments
`130 -> 127`. Five videos improve and eight tie; no video regresses in IDSW,
IDF1, or HOTA. The small `000231` FP/FN trade-off is retained explicitly.

Reject `max_alternative_cost=0.30` even though aggregate IDSW reaches `119`:
it creates persistent `000216` ID 5/8 corruption and drops that video's
IDF1/HOTA from `99.55%/99.26%` to `90.10%/91.45%`. The selected `0.25`
threshold excludes cost `0.283780` while retaining the useful `000233` event
at `0.238421`. Promotion commit is `e8d39d7`; authority is
`docs/TRACKING_PROMOTION_DECISION_20260718_REALTIME_BALANCED_HIDDEN_RESERVATION.json`.
`realtime_quality_delayed` keeps its reservation-disabled semantics.

The H2 asymmetric refinement candidate at `e55973f` is rejected. Its full-13
run improved matches by 74 and aggregate HOTA by 0.0719 percentage points, but
increased IDSW from 10 to 14, including `000085: 0 -> 2` and `000328: 4 -> 6`.
It also changed 68 `Hidden/occluded` payloads after bbox refinement, so it was
not geometry-only through the complete post-processing chain. Do not repeat or
promote `refine_max_gap_frames=30` plus
`refine_max_previous_gap_frames=15`. Defaults remain unchanged at `15/0`.

The three hash-bound realtime baselines have now been replayed with
`include_hidden=true`. Aggregate IDSW is `87` for fast, `133` for balanced,
and `168` for the quality-delayed parent. These replace the exclude-Hidden
values only for primary scientific evaluation; the old reports remain
compatibility evidence.

Retain the existing quality-delayed default
`realtime_motion_pair_simple_min_gain=0.003`. Against the same-contract parent
at `0.005`, both primary and repeat improve aggregate IDSW `168 -> 166`, with
only `000263: 44 -> 42`; no video regresses, FP/FN and fragmentation are
unchanged, HOTA/IDF1 do not decrease, and the candidate repeatability audit
passes with 26 prediction XML files, 72 artifacts and `mp4_count=0`. Continue
to classify this profile as `post_video_global_graph` with delay `-1`; do not
claim fixed-delay causality or an absolute FPS improvement. The corrected
decision authority is
`docs/TRACKING_PROMOTION_DECISION_20260717_P2_QUALITY_DELAYED_INCLUDE_HIDDEN.json`.

The hybrid far-camera identity guard is now promoted. In two identical full-13
runs it reduces aggregate IDSW `10 -> 8`, entirely by fixing `000216: 2 -> 0`,
with no per-video IDSW regression. HOTA and IDF1 remain `98.31%` and `99.13%`
at report precision. The raw trade-off is FP/FN `1628 -> 1630`, concentrated in
`000302: 68 -> 75`, while `000216` improves `330 -> 325`. Judge this as an
overall tracking improvement, not by requiring every raw metric to be
monotonic. Commits are `7254670` for the algorithm and `e74a8fa` for the
separate profile promotion. The authoritative decision is
`docs/TRACKING_PROMOTION_DECISION_20260718_HYBRID_FAR_IDENTITY_GUARD.json`.

The causal `realtime_fast` profile now also promotes
`realtime_visible_better_competitor_prefer=true`. Full-13 primary and repeat
are semantically identical and improve aggregate IDSW `87 -> 69`, HOTA
`93.89% -> 94.35%`, IDF1 `93.21% -> 93.91%`, FP/FN `564/688 -> 506/630`, and
fragments `114 -> 110`. Only `000231` improves (`30 -> 12` IDSW); the other 12
videos are unchanged and no video regresses. The profile promotion is
`456fc97`; authority and rollback are recorded in
`docs/TRACKING_PROMOTION_DECISION_20260718_REALTIME_FAST_VISIBLE_PREFER.json`.
The timing contract remains causal framewise with delay `0`, and all prediction
and evaluation roots have `mp4_count=0`. Runtime differences between repeats
are recorded, so no speed claim is made.

Synthetic R0 causality, repeatability, baseline-lock and telemetry checks pass
`23/23` for the two causal realtime profiles. Runtime R0 also passes on the
`000263` hard event: prefix 210 versus extended 240 frames preserves all 1,680
flushed XML box payloads for both fast and balanced, with declared delay `0`
and `mp4_count=0`. Auditor commit is `f8e1b6e`; authority is
`docs/TRACKING_CAUSALITY_DECISION_20260718_R0.json`.

Freeze these 13 videos as development evidence. Any further change must start
with one isolated weak-event window, then a full difficult video and hard set;
run full-13 only after those gates pass. Final unbiased claims require
untouched sessions selected before more optimization. Every run keeps
`include_hidden=true`, a fresh output root, and recursive `mp4_count=0`.

### Main-versus-legacy source boundary

The user confirms that legacy 16f is a relatively clean standalone lineage,
not a source currently used by the main classification branch. The prior
`legacy-only-unreviewed-development` lane was created to exercise prompt/goal
orchestration and screen configurations or hypotheses for later main-branch
testing. It was not a data merge into the main branch.

Legacy goal completion and its handback do not activate, resume, authorize or
mark PASS any parent/main classifier goal. A candidate configuration must be
retested against the main branch's own versioned source manifest, reviewed
snapshot, folds, shortcut audits and short-run gates. Do not implicitly merge
the canonical legacy export into the main lineage.

Legacy 16f still requires its own two-sided frame/object Hidden review and
complete native-unit 16-frame behavior review before it can be called reviewed
or used as train-ready evidence. Those decisions neither satisfy nor block the
main branch while legacy remains outside the main source manifest.

### Legacy CVAT rebuild completed

The scientific legacy 16-frame rebuild completed P0-P10 in
`outputs/legacy_16f_rebuild/legacy_16f_rebuild_20260718_v2`. The canonical
export has 72,880 rows for 4,555 actors and 666 groups; every actor has six
native anchors and 16 frames. Behavior comes from the lowest CVAT task frame
per burst, whose suffix may be `k0..k5`.

The active mixed-format contract remains `task_0..task_2 = XML` and
`task_3 = JSON`. Task_3 is retained. Exactly three declared bad actor keys are
filtered before recovery, and all occur zero times after P2. The row equation
is `27,665 - 5 actor-policy - 330 video-policy = 27,330 anchors`.
P5 applies that same policy before authority/coverage and must report clean
`PASS`; `PASS_WITH_DECLARED_EXCLUSIONS` is not accepted for retained input.

The final audit is
`08_audits/legacy_16f_rebuild_completion_audit.json` with `status=PASS` and
`errors=[]`. This authorizes the canonical data handoff only. It does not call
the artifact human-reviewed, does not authorize training or OOF, and does not
unlock a Q2 claim. Hidden review and reviewed-lineage gates remain separate.

### C6 2026-07-19 full-development handback

The new rebuild-bound C6 configs supersede the earlier code-ready holds only
inside the isolated `legacy-only-unreviewed-development` lane. Temporal
controls passed 18/18 fresh repeats and froze A128 for modality screening.
The modality matrix then passed 22 modes x 2 repeats and 14 paired
comparisons. Process IDs were `29324` and `5064`; packet, config, cache and
claim audits passed. That promotion freeze authorized only ROI and union
context for a full-development confirmation.

The `full01` confirmation used 3,650 train and 241 validation native units and
completed all seven actor/ROI/union control modes at 345 optimizer steps each.
ROI did not meet the minimum gain or positive cluster-CI gates. Union met the
point-gain margins but failed positive-CI, NLL and rare-group guardrails. The
final decision is `RETAIN_A128_ACTOR_ONLY_FOR_C6_LEGACY_HANDBACK`; freeze
SHA256 is
`bf7cc849e49c56458af4ea91c1824ab46b839f527e799945b64dc96fc4d86e61`.

This evidence remains `TECHNICALLY_CLEAN_UNREVIEWED_DOUBLE_CHECK_PENDING`.
It closes C6 legacy development only. It does not authorize OOF, make
Hidden/behavior review complete, or transfer a result to the main source
manifest. The main reviewed lineage must retest A128, ROI and union under its
own snapshot, folds, availability controls and paired uncertainty.

### C6 temporal-control decision

The rebuild-bound temporal matrix is complete and its freeze keeps A128 for
the C6 legacy handback. The freeze SHA256 is
`150b61fddc42464d4d4767d55d5615d63507a02df6b74805c0ba85faffcf9a69`.
This is not the final main-data temporal choice. A128 and the simpler control
must be paired again on the frozen reviewed main lineage.

### Legacy C6 matrix closure

The legacy C6 short matrix, promotion freeze, full-development cache, `full01`
run, paired evaluation and final freeze are complete. No optional C6 modality
is retained for this legacy handback. Do not reinterpret zero/availability
controls as deployable candidates and do not generalize the legacy rejection
to reviewed full data. Full OOF and Q2 remain separately blocked.

This closure is a global legacy-handback decision, not a per-behavior modality
rejection. The short matrix covered geometry, motion, ROI, numeric social, pen,
union, and full-frame branches and retained all ten classes in per-class
metrics. ROI and union alone reached the legacy full-development confirmation;
geometry, motion, numeric social, pen, and full-frame context are deferred for
reviewed-lineage retest rather than removed from the model-design universe.

Descriptive short evidence assigns different hypotheses to different classes:
ROI to `drink/eat/lying/move/sitting`; motion and numeric social to
`drink/fight/social-nose/move`; pen context to
`eat/explore/lying/stand/sitting`; union to `eat/lying/sitting`; and geometry
mainly to `drink/move`. `fight`, `move`, and `playwithtoy` have low support, so
zero or positive point estimates cannot settle their utility.

After a behavior-complete main-lineage handoff, rerun all seven branches with
the same actor base, folds, seeds, temporal view, three modality controls, and
all ten behaviors. Promotion must combine behavior-specific paired uncertainty
and non-target harm bounds with the global NLL/calibration and missingness
guardrails. Only then may a predeclared behavior-conditional fusion candidate
enter short confirmation, finalist lock, and the separate full-OOF launch gate.

### Agent execution isolation

Until the user hands off a clean reviewed lineage, do not run review GUI,
apply, temporal rebuild, snapshot or model jobs against project data. Agent
audits use a fresh
`outputs/classification_v2/agent_audits/<AUDIT_RUN_ID>` root and may not write
canonical output folders.

After `REVIEW_STAGE=behavior_complete`, consume only the exact handed-off
`RUN_ID` as read-only input. Build the reviewed-Q2 artifact map, generated
contract, model-input manifest, snapshot and P0 report under that one agent
root. P0 can authorize model smoke only; it never authorizes full OOF.

### Clean human-review authority

The user confirms zero completed human decisions. Treat all existing 30-row
Hidden and 3-row behavior CSV payloads as unverified forensic/pilot artifacts,
not review evidence. Do not carry them into the next lineage.

Create the new operator-owned lineage only under
`human_review_workspace/classification_v2/<RUN_ID>`. Agent writes belong
under `outputs/classification_v2/agent_audits/<AUDIT_RUN_ID>`; no agent may open
a GUI or write the active human root. The operator owns apply/rebuild there;
after handoff, agent checks remain read-only on that root and write evidence to
the agent audit root.

### Full-data base selection boundary

Legacy Stage A v3 carries `SF128` only as the simplest control and marks
`A128` for a conditional mixed-reviewed retest. It does not select a final
base. The decision packet SHA256 is
`b3250ed5391d46e37469a22f16353bbc5f038fa250897c37056fe64a132a6910`.

After a clean behavior-complete handoff, compare at least `SF128` and `A128`
on the same frozen mixed-reviewed native-unit folds. The decision must report
pooled and per-source metrics, source-balanced support, missing-modality
strata, target behavior groups, paired video-cluster uncertainty, parameters,
runtime, and complete lineage hashes. A pooled gain driven by only one source
or by availability metadata cannot promote a candidate. Ambiguous evidence
retains the simpler control. Run the exact short gate before any authorized
full mixed-data confirmation; full OOF keeps its separate launch gate.

### Legacy one-sequence temporal sampling decision

The post-L8 controlled comparison keeps one complete 16-frame burst as the
native evaluation unit and feeds exactly one model sequence per unit. The views
are C6 `[5,6,7,8,9,10]`, C8 `[4,5,6,7,8,9,10,11]`, and S6
`[0,3,6,9,12,15]`; no contiguous T16 candidate is included.

On the fixed 245-unit, 33-video development validation set, C6, C8, and S6
macro-F1 are `0.3708555386`, `0.3588478457`, and `0.3334808033`. S6 minus C6
is `-0.0373747353`, with 2,000-video-cluster interval
`[-0.0871566209, 0.0263536824]`. C8 minus C6 is `-0.0120076929`, with interval
`[-0.0414907389, 0.0171363991]`.

Retain `c6_contiguous_centered` as the one-sequence working view. S6 has only
descriptive point gains for `drink` and `move`; both F1 intervals include zero,
and `move` has eight validation units. C8 has better NLL (`1.0379915527`) than
C6 (`1.0664057500`) but does not improve macro-F1. Do not promote either view.

This decision does not supersede the locked sliding-T6 L8 candidate because
that protocol uses four windows per native unit and different optimizer
exposure. It applies only to legacy unreviewed development and must be repeated
on merged reviewed data. Decision SHA256 is
`cdd24a27162ec46bc68214e6820e3aa41aebe86da53acd6903da175bcced2cfa`.

### Legacy post-handback pen-context decision

The parameter-matched `legacy_16f` pen-boundary experiment is closed as valid
short negative evidence. All cache, preflight, tiny-overfit, resume, six-run
repeat, and exact 245-native-unit/33-video pairing gates pass. Zero,
availability-only, and pen-context macro-F1 are `0.2774732055`, `0.2752044192`,
and `0.2773207312`.

Pen context minus zero changes macro-F1 by `-0.0001524743`, with the declared
2,000-sample video-cluster interval `[-0.0183285810, 0.0193813458]`. NLL
improves by `-0.0099586087`, and the `stand/move/explore` focus macro-F1
improves by `+0.0107610350`, but the global gain and positive interval-low
gates fail. Availability-only remains bounded at `-0.0022687862`, so no strong
missingness shortcut was detected in this short diagnostic.

A support-aware post-hoc diagnostic shows that this is conditional utility,
not evidence that pen context is generally useless. Pen-minus-zero F1 changes
are `drink +0.0334` (20 units), `eat +0.0133` (17), `move +0.0300` (8), and
`sitting +0.0306` (70), while `lying -0.0778` (62) is the main harm. The
`move` gain comes from reducing false positives from 39 to 29 without changing
its 3/8 recall; `sitting` gains five true positives, while `lying` loses six
correct units and its confusion into `sitting` rises from 16 to 23.

For 117 persistent-boundary units, pen context improves accuracy from
`0.4017` to `0.4188`, global-ten-class macro-F1 from `0.2461` to `0.2560`,
and NLL from `1.8372` to `1.7790`. For 110 interior-only units, it degrades
accuracy from `0.3545` to `0.3364`, macro-F1 from `0.2702` to `0.2601`, and
NLL from `1.8411` to `1.8809`, driven mainly by `lying`. The four T6 windows
expose 15 of the declared 16 frames and 14 unique frame pairs per native unit,
so this diagnostic is not a complete T16 exposure analysis. These results are
exploratory and do not change the locked promotion decision. The next legal
test is a predeclared boundary-gated or residual pen branch, with zero,
availability-only, and always-on controls, after the same short gates pass.

The decision is
`DO_NOT_EXPAND_PEN_CONTEXT_FROM_CURRENT_SHORT_EVIDENCE`. Do not run its full
legacy expansion or add pen values to the locked legacy candidate. The audit
artifact is `pen_context_short_decision.json`, SHA256
`673ddab840e5d69f984b47c9d832e2415147681f3df6b81448270766ab673e1c`.
This post-handback short experiment does not supersede the locked L0-L8 full
candidate, transfer to mixed-reviewed data, authorize Q2 claims, or prove that
pen context is generally useless. Reassess it only on a frozen reviewed
all-source lineage under the same parameter-matched controls.

### Legacy L0-L8 completion and parent handback

The scoped `legacy_16f` lane is complete at code commit `91a6c2a`. The locked
candidate is `legacy_16f_t6_sliding_event_balanced_v1`, with native macro-F1
`0.5343181014`, accuracy `0.6857142857`, and NLL `1.1206917661` on the fixed
245-unit, 33-video development validation set.

The L8 candidate-lock SHA256 is
`b91949711e15c493a07375c4f7fa5f44535220dfdbac68f095d2effee4be6ba6`.
The L0-L8 handback is
`outputs/classification_v2/legacy_only_unreviewed_development/`
`legacy_16f_goal_completion_audit.json`, with SHA256
`4b6bad32834fbede2001dee5627e5fbfa0005afb758f2c6a3cbfb125be3166f6`.
Every milestone is PASS, while human review, reviewed/final naming, canonical
full OOF, outer-holdout prediction, and Q2 claims remain unauthorized.

The next action is to resume the parent classification_v2 goal and re-audit
the canonical reviewed all-source P0-P8 blockers. Do not inherit a parent PASS
from this bounded legacy handback.

### Legacy L7 imbalance decision and L8 handoff

The three-policy `legacy_16f` L7 short matrix is complete with two fresh,
non-overlapping CUDA processes per policy. All repeat hashes are exact. Every
run used 30 optimizer steps, had no OOM or retry, peaked at `73,400,320`
reserved bytes on the local 4 GiB GPU, and cleaned allocated and reserved CUDA
memory to zero.

Event-balanced CE, effective-number CE, and Balanced Softmax native macro-F1
are `0.2717708642`, `0.1072693320`, and `0.1429984901`. Relative to
event-balanced CE, effective-number CE changes macro-F1 by `-0.1645015322`
with 33-video interval `[-0.1895271242, -0.0767137732]`; Balanced Softmax
changes it by `-0.1287723740` with interval
`[-0.1565980560, -0.0416499788]`. Their NLL values worsen from
`1.9439967908` to `3.3642298748` and `3.6577075450`; rare-group macro-F1
falls from `0.2382505739` to `0.0411892030` and `0.0713385243`. Balanced
Softmax also predicts `fight` for `70.2041%` of validation units and fails the
majority-collapse guard.

The valid decision is `RETAIN_EVENT_BALANCED_CE_REJECT_L7_ALTERNATIVES`.
Do not run a full confirmation for either rejected alternative. Start L8 from
the retained actor-only T6 event-balanced base. The decision artifact is
`l7_imbalance_decision_v1.json` with SHA256
`69c200e2b6d570b181423df30cc33cdbecb6686f175f7184b40067bd62ff1482`.
This remains bounded three-epoch evidence, not a full-convergence claim, and
does not transfer to merged-reviewed data, authorize reviewed/final naming,
canonical full OOF, or Q2 evidence.

### Legacy L6 full-frame decision and L7 handoff

The `legacy_16f` full-frame short matrix and paired evaluator are complete.
Zero, availability-only, and full-frame macro-F1 are `0.2697662759`,
`0.2721987509`, and `0.2942624204`. Full-frame minus zero is
`+0.0244961445`, with 33-video cluster interval
`[-0.0668714797, 0.0725200014]`; NLL worsens by `+0.2414525889`.
Full-frame minus availability-only is `+0.0220636696`, with interval
`[-0.0809709233, 0.0671747502]`; NLL worsens by `+0.3144303865`.

The valid decision is
`DO_NOT_EXPAND_FULL_FRAME_CONTEXT_FROM_CURRENT_SHORT_EVIDENCE`. The macro-F1
point estimates meet the margin, but both cluster intervals cross zero and
NLL worsens. Do not run a full confirmation or carry full-frame values into
the legacy candidate. The decision artifact SHA256 is
`e006dc6636ede5a35e71414448be1dc96f0f71e29f5f2a1b6d0230fa0c49c6bf`.

L6 is PASS with the parameter-matched T6 zero retained as the simplest bounded
base. Start L7 and compare event-balanced CE, effective-number CE, and Balanced
Softmax separately. These decisions apply only to unreviewed `legacy_16f` and
do not authorize reviewed/final naming, canonical full OOF, Q2 claims, or an
architecture conclusion for merged-reviewed data.

### Legacy L6 ROI relation decision

The `legacy_16f` ROI short matrix passed its predeclared paired promotion gate,
so one exact full confirmation was authorized. That confirmation is complete
at `l6r_full_decision_v1.json` under
`outputs/classification_v2/legacy_only_unreviewed_development/l6r_full_v1/`
with SHA256
`5a9a2b4b61b7ddeef0b5155ec69b678d73f0acd53917db98d1d6271cab5f1af3`.

Full zero, availability-only, and ROI macro-F1 are `0.4966025667`,
`0.4727197983`, and `0.5082292933`. ROI minus zero is `+0.0116267266`, with
33-video cluster interval `[-0.0398806556, 0.0906766805]`; ROI minus
availability-only is `+0.0355094951`, with interval
`[-0.0248897889, 0.0986581204]`. Availability-only minus zero is
`-0.0238827684`, with interval `[-0.0629523019, 0.0339059054]`.

Do not summarize this as uniformly negative ROI evidence. Relative to zero,
feeding-group macro-F1 improves by `+0.1796877378`; `drink` F1 improves from
`0.3703703704` to `0.6486486486`, and `eat` F1 improves from `0.7906976744`
to `0.8717948718`. `playwithtoy` has only one validation unit, so its ROI
effect is not estimable: recall is `1.0` in both modes, while false positives
rise from one to four and F1 falls from `0.6666666667` to `0.3333333333`.

The decision is `DO_NOT_EXPAND_ROI_RELATION_FROM_CURRENT_SHORT_EVIDENCE`.
The full ROI gain misses the required margin and positive interval-low gate,
and the availability diagnostic fails its bounded-difference check. Do not
carry ROI values into the next candidate. Continue L6 numeric social relations
from the parameter-matched T6 zero control. This does not authorize canonical
full OOF, reviewed/final naming, Q2 evidence, or any claim about merged-reviewed
data. Reassess ROI on merged-reviewed data, where rare-class support is
materially larger. L6 remains `IN_PROGRESS`.

### Legacy L6 numeric-social short decision

The ten-feature numeric-social cache and independent repeat gate pass for all
15,588 T6 sliding windows. The tensor shape is `[15588, 6, 10]`; 92,664 of
93,528 slots are available, with zero media reads and zero outer-holdout slots.
All four cache artifacts are byte-identical across independent roots. Partner
IDs remain audit metadata; top-K partner, geometry, motion, and ROI values are
excluded from model X.

The six-process short matrix is deterministic and crash-bounded. Zero,
availability-only, and numeric-social macro-F1 are `0.2620738697`,
`0.2621547321`, and `0.2624282011`. Numeric-social minus zero is
`+0.0003543314`, with 33-video cluster interval
`[-0.0342531654, 0.0398565230]`; accuracy falls by `0.0326530612` and NLL
worsens by `0.2248711988`.

The decision is `DO_NOT_EXPAND_SOCIAL_RELATION_FROM_CURRENT_SHORT_EVIDENCE`.
Do not run full numeric-social confirmation or carry its values forward.
Core-roadmap S2 requires S1 numeric-social PASS before top-K, so top-K is
`DEFERRED_NOT_AUTHORIZED`. Continue L6 actor-partner union-crop work from the
parameter-matched T6 zero because the interaction-context gap remains. This
does not transfer to merged-reviewed data or authorize reviewed/final naming,
canonical full OOF, or a Q2 claim.

### Legacy L6 motion decision

The `legacy_16f` motion short matrix is closed as valid negative evidence.
Motion does not meet the predeclared native-unit/video-cluster promotion gates,
so do not run full motion or add motion values to the next L6 candidate.
Continue from parameter-matched T6 with numeric social relations; do not carry
ROI or motion values into the next candidate. Geometry, motion, and ROI must be
reassessed on frozen merged-reviewed data; none of these legacy-only decisions
transfers to that lineage. The local 4 GiB GPU was only a correctness host and
was not the rejection reason.

### Canonical engine and review-policy boundary

There is one `classification_v2` data engine, not separate legacy and mixed
implementations. The legacy 16-frame lane differs only by source-selection,
review policy, temporal-view config, output namespace, lineage, and claim
flags. It must call the same canonical feature and harmonization modules.

The two current profiles are:

```text
legacy-only-unreviewed-development:
  sources = legacy_recovered
  review_policy = explicitly_waived_for_development
  temporal_views = T6/T8/T12/T16 within each native 16-frame burst

mixed-reviewed:
  sources = legacy_recovered + cvat_tracking_xml
  review_policy = required_by_current_Q2_protocol
  primary_temporal_view = fixed6_observed_time
```

Human review is configurable at the engine level. It is not an absolute
technical requirement for exploratory training when the user explicitly
accepts an unreviewed lineage. It is nevertheless a hard scientific gate for
the active `mixed-reviewed` snapshot, final/reviewed naming, and Q2 evidence.
Skipping it requires a new explicit unreviewed profile; it must never silently
convert incomplete decisions into authorization.

The separate legacy goal exists to isolate progress, artifacts, metrics, and
claims while the parent mixed-source goal is blocked. It is not permission to
duplicate context, ROI, motion, social, posture, or harmonization code. After
L0-L8 complete in the new chat, return its hash-bound handback to this original
chat and resume the parent P0-P8 goal.

The user grants standing authorization for a full data or model run when it is
necessary for the current milestone. Do not ask again solely because the run is
full or long. Every new lineage must first pass static/synthetic checks, the
exact short representative configuration, and schema/count/hash/output/runtime
audits. Stop before full on any failure. A semantic config change invalidates
the short evidence and requires the short gate again. This permission does not
bypass leakage, immutable lineage, full-OOF launch, or scientific claim gates.

For bounded model tests that report `accuracy` or `F1`, use an explicitly
declared `legacy_recovered` 16-frame development lineage when it is
scientifically compatible with the tested question. The user permits this
lineage to proceed without current human review because it is closest to the
older model lineage and currently less dirty than CVAT. It must be grouped by
recording/video, native-unit safe, hash-frozen, and labeled
`legacy-only-unreviewed-development` in every manifest, run, prediction, and
metric. It cannot replace the all-source reviewed evaluation, be called final
train-ready data, authorize full OOF, or support a Q2 claim by itself.

The reviewed all-source lineage remains independently blocked by incomplete
Hidden and behavior decisions. Its gates must not be weakened or reused to
misrepresent the legacy development branch as human reviewed.

Use `plans/classification_v2-legacy-16f-development-goal-prompt.md` to create a
separate goal for this branch and its dedicated L0-L8 ledger. On scoped goal
completion, return to the original Q2 chat with the immutable handback audit;
do not mark the parent P0-P8 objective complete automatically.

Temporal-length experiments must compare `T6`, `T8`, `T12`, and `T16` windows
generated only after harmonization and contained within one legacy burst. Use
the same native-burst folds and aggregate predictions back to the 16-frame
unit. Report an event-mass-balanced sliding-window view and a one-window-per-
burst matched view so sequence length is not confounded with sample count.

Commit `21b34fd` is the model-input authority for this ladder. It exposes eight
view/selection/slot contracts, binds exact observed-time tensors to each `T`,
and rejects mismatched config, image/context length, spatial padding, or timing.
Its 438-test regression is code evidence only. The short real-data packet,
cache alignment, and no-row-loss audit remain the next gates before any full
legacy rebuild or model execution.

Commit `9b04209` is now the source/missingness probe authority. A source probe
must use the exact ordered trainer whitelist, bind the train-ready ordered
window SHA256, collapse repeated windows to native units, fit only grouped
training roles, and test every eligible native unit once. Availability-only
behavior diagnostics may use only registered label-independent masks;
`interaction_context_ready` is forbidden because its current construction is
label-gated. This engineering PASS does not authorize active-data training.

Commit `abae856` freezes the model-selection grain for all new trainer runs.
Select checkpoints only from grouped inner-validation predictions collapsed by
`temporal_unit_key`, maximizing supported-class macro-F1 and using native-unit
NLL only as a tie-breaker. Remote pilots/full OOF require all 10 classes in the
inner-validation role; local synthetic or bounded smoke requires at least two.
Outer-test predictions remain evaluation-only. New window/native prediction,
aggregation, checkpoint, and registry artifacts must retain this policy and its
hash lineage. This engineering PASS does not authorize active-data training.

Commit `bb225ff` completes the temporal-view and structural shortcut contract
in code and on synthetic fixtures. The primary view now reuses harmonized
six-frame windows for both sources; legacy 16-frame quantile sampling is not
allowed. This is `PASS IN CODE`, not active-data evidence. Do not build the
reviewed temporal packet or run a model until both human review layers pass.

Commits `97f83c5`, `73b901d`, and `16cdb93` complete fold-local preprocessing,
native-event weighting, and run-lineage/registry contracts on fixtures. Run
artifacts now live under `output_root/fold_id/run_id`; downstream callers must
consume the returned lineage path. This remains engineering readiness only and
does not authorize model smoke on an unfrozen reviewed snapshot.

Commit `318bf58` completes the configurable model-factory contract in code.
Ten model modes and four temporal encoders pass mask, shape, missing-modality,
gradient, checkpoint, and lineage tests without downloading weights.

Commit `07ed768` extends that factory with audited ResNet18 and ResNet34 frame
encoders. The controlled interface separates ResNet18 160-to-224 resolution
from ResNet18-to-ResNet34 capacity, records exact ImageNet enum/normalization,
and passes random-init forwards without downloading weights or training. This
does not authorize a pretrained pilot before the reviewed snapshot is frozen.

Commit `2bd2fda` completes the independent visual fine-tuning schedule in code.
Actor and union-context ResNets share frozen, `layer4_only`, and full stages;
the backbone uses a lower LR while all optimizer parameters remain present for
stage-boundary resume. Checkpoint v5, run identity v2, run manifest v2, and
registry v4 bind this contract. The V0/V1/V2 checker is structural-only with
zero optimizer steps, zero project-data rows, and no weight download. It does
not authorize an active-data pilot or satisfy P1 performance PASS.

Commit `3be22f8` completes the independent synthetic visual correctness gate.
ResNet18-160 reaches ten-class tiny-event accuracy 1.0 with finite backbone and
head gradients, deterministic repeated evidence, and exact in-memory resume
parity. The audit remains `synthetic_only` and explicitly sets snapshot/full-OOF
authorization false; it cannot replace active reviewed-data smoke gates.

Commit `111f152` now loads real ordered fixed-six `time_delta` tensors into the
strict data module and binds the slot-manifest hash in checkpoint schema v4 and
registry v3. Corrupt order, slot identity, masks, or timing fail closed, and
unselected windows are retained as explicit masked rows. This remains fixture
evidence only: the reviewed snapshot is blocked, so ResNet training, pilot
training, and full OOF remain unauthorized.

Commit `1b6ba3d` completes native-unit collapse and paired evaluation in code.
Strict ten-class probabilities are reconciled against the complete fold
authority, pooled metrics use the fixed global class order, and paired
recording-cluster uncertainty requires identical units, targets, clusters, and
folds. Its synthetic checker does not replace the missing
human-reviewed snapshot or authorize training.

Commit `e5d6417` completes historical-baseline reconciliation as an engineering
control. Its audit reproduces 151,440/160,740 positional mismatches and marks
the old full OOF `HISTORICAL_ONLY`. It safely records the legacy ResNet34
sequence checkpoint as `HISTORICAL_ARCHITECTURE_ONLY`, not as a performance
baseline. The current regression is 385 passed and 181 deselected. Neither
historical artifact authorizes model selection, paired comparison, training,
full OOF, or a Q2 claim.

The identifier-v2 code/data chain passes at commit `a83d5a5`. Its bounded root
has 688 frame rows, 63 native/review units, 438 ordered windows, exact model-X
whitelisting, zero trainable missing spatial slots, and 8/8 deterministic stage
reruns. All reviewed-data, training, full-OOF, and Q2 authorizations remain false.

Snapshot/preflight contracts are hardened by `7cb4637` and `dd0e6ff`.
Future full-run evidence must bind exact ordered split, image, interaction,
spatial, snapshot, lineage-audit, config, and code hashes. Old v1 snapshots,
preflights, and authorization files are readable historical artifacts but
cannot authorize execution. This code gate does not replace human review.

Do not use the historical full OOF metrics to judge model quality. Commit
`bfdf913` proved that its split/target rows were positionally misaligned with
151,440 of 160,740 image and interaction windows. That run remains useful only
for compute, checkpoint, and pipeline-debug evidence.

The current `reviewed_frame_features.csv` is not human-review complete. Three
old behavior payload rows exist, but verified behavior coverage is 0/4,670.
Do not call this artifact clean final training data. Rebuild instructions are in
`docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`.

Keep the target-independent v6 Hidden manifest at
`outputs/classification_v2/rebuilds/hidden_review_v6_full_20260714` for the
technical reference only. Its 30 carried payload rows are unverified; clean
human coverage starts at 0/5,131 in a new
`human_review_workspace/classification_v2/<RUN_ID>` root.
Hash-bound media validation of the old reference resolves all items, but the
scientific gate remains BLOCKED until the clean authority is reviewed.

Do not resume full training from this decision alone. First create a versioned
reviewed-data lineage, pass complete-decision and leakage-safe fold gates, then
run model smoke gates on the frozen data/cache hashes.

## Historical 2026-07-13 post-full decision

For the previous artifact lineage, full OOF training completed and postrun
validation was the next gate. This no longer overrides the active rebuild.

- Do not rerun full training for the script migration; the completed artifacts
  remain the input to block `07` postrun evaluation.
- Run cross-fit calibration, confusion-focus comparison, ablation refresh,
  experiment registration, and block `09` completion gate in that order.
- Do not claim Q2 improvement until the completion gate reports
  `q2_claim_allowed=true`.
- Use only `scripts/classification_v2/00_*` through `09_*`; there are no wrapper
  commands under the former script namespaces.
- The claim boundary remains internal recording-date/video-safe improvement.
  No external farm, camera, cohort, or biological-identity generalization.

## Historical 2026-07-13 pre-full decision refresh

At that point, the previous lineage was pre-full ready, not Q2 complete.

- Current verified HEAD is the `current_git_commit` in
  `outputs/classification_v2/model_design/q2_progress_report_audit.json` after
  the latest pre-full refresh. Do not hard-code a commit here because memory
  commits intentionally move HEAD.
- Current progress is `PASS_PARTIAL_ROADMAP` with 44/44 gates passing.
- The execution gate now requires 4 rejection cases, including rejection of a
  near-authorized file missing `reviewer` and `reviewed_at`.
- Runtime preflight may allow audit/auth-only commit drift without rebenchmark,
  but must still fail closed for runtime/model/training-relevant changes.
- Do not run or claim full OOF until human authorization is explicitly valid and
  the execution gate allows it.

## Historical 2026-07-12 classification_v2 decision

The active project priority is `classification_v2` behavior recognition unless
the user explicitly switches back to tracking.

Decision recorded at that time:

- Treat the multimodal Q2 roadmap as pre-full ready, not complete.
- The accepted claim boundary is Q2 internal
  recording-date/video-safe improvement. Do not claim external farm, camera,
  cohort, or broad real-world generalization without external validation.
- The model direction is multimodal spatio-temporal:
  letterboxed actor bbox image sequence, ROI relation tensors, motion features,
  social/partner context, and interaction visual context.
- `pig_id` is annotation-local. Never use it as identity continuity across
  videos or sessions.
- Canonical actor visual cache:
  `outputs/classification_v2/image_cache_v2_letterbox/`.
- Historical full OOF output dir for that lineage:
  `outputs/classification_v2/model_full/full_multimodal_oof/`.
- The progress report then was `PASS_PARTIAL_ROADMAP` with 44/44 pre-full
  gates passing. It meant ready for authorization review, not ready to claim
  final Q2 results.
- Full OOF was fail-closed until
  `outputs/classification_v2/model_design/full_oof_authorization.json` was
  explicitly authorized with reviewer, long-run acknowledgement,
  no-Q2-claim acknowledgement, matching preflight config SHA256, and matching
  git commit.
- After full OOF finishes, run postrun calibration, confusion-focus comparison,
  ablation report refresh, experiment registry write, and completion gate before
  any Q2 claim.

Historical tracking decisions below are preserved for tracking work, but they
must not override the current `classification_v2` priority.

## 2026-07-07 current best full tracking candidate

Treat `outputs/eval/hybrid_bytetrack/codex_visible_suffix_gate_full/iou0_area0_condarea0_merge0`
as the current best validated full 12-video candidate.

Compared with `outputs/eval/hybrid_bytetrack/Best_tracking/iou0_area0_condarea0_merge0`:

- `ALL` remapped IDSW improved `11 -> 0`.
- Every per-video remapped IDSW is `0`.
- Clean guardrails remained clean: `000085=0`, `000225=0`, `000231=0`,
  `000302=0`, `000328=0`.
- Remaining targets are fixed: `000233=0`, `000263=0`.

The key correction after the failed `20260707_174142` full stack is that
`suffix_pair_swap_repair=true` now requires both shapes at the swap start frame
to have `Hidden=No`. This keeps the desired visible-start `000263` repair while
blocking the hidden-start false suffix swaps on `000085` and `000225`.

Current candidate stack:

- protected association/occlusion practical base.
- `occlusion_reid_prefer_gap_over_bad_match=true` with the proven unowned
  raw-mismatch occlusion-hold bounds.
- `overlap_small_box_suppression=true`.
- `hidden_suffix_id_swap_repair=true`.
- `suffix_pair_swap_repair=true`, but only with the visible-start gate in
  `repair_suffix_pair_swaps`.

## 2026-07-03 tracking decision

- Treat `outputs/eval/hybrid_bytetrack/20260703_193439/smooth_det020_loose/iou0_area0_condarea0_merge0/tracking_metrics.csv` as the current best 2-video tradeoff for `000231` + `000302`.
- Result:
  - `Pigs291119_000231_30fps`: IDSW `2`, HOTA `0.9705892717094201`, IDF1 `0.9847241970177549`.
  - `Pigs291119_000302_30fps`: IDSW `0`, HOTA `0.9930104703678451`, IDF1 `0.9964355605255801`.
  - `ALL`: IDSW `2`, HOTA `0.9820366705826231`, IDF1 `0.9907038986528682`.
- Keep the current split lost-track reacquire approach:
  - `lost_track_reacquire_guard=true`.
  - `lost_track_reacquire_non_same_raw_distance_guard=false` is the current default/base setting after 9-video run `20260703_194929`.
  - `lost_track_reacquire_raw_owner_guard=true`; do not turn it off globally.
  - Keep `lost_track_different_raw_hidden_owner_bypass=true`, `lost_track_different_raw_hidden_owner_min_missed=2`, and `lost_track_different_raw_hidden_owner_min_center_gain=0.03`.
- Ablation findings:
  - Turning off raw-owner guard globally gives `000302` IDSW `0` but makes `000231` much worse.
  - Turning off only non-same-raw distance guard gives `000231` IDSW `2` but still needs the hidden-owner bypass to recover `000302`.
  - Tightening only appearance threshold did not change the bad `000231=8`, `000302=0` result; owner state / center gain was the useful tightening.
- Default decision: tracking, evaluation, and optimizer should inherit this base from `TrackingConfig`; do not require callers to pass `--profile-override lost_track_reacquire_non_same_raw_distance_guard=false`.

## Current baseline

- Do not use legacy 21/06 as the primary comparison point anymore; when discussing `evaluate_tracking.py` metric drift, compare against commit `b697c4eba36db280cbf01f446873da17bcac509d`.
- Current accepted `hybrid_bytetrack` post-processing flow is the two-gate flow restored from `b697c4eba36db280cbf01f446873da17bcac509d`: identity guard requires `enable_offline_smoothing and identity_swap_guard`; temporal refinement plus `stabilize_overlap_hidden_islands` requires `enable_offline_smoothing and (smooth_boxes or refine_boxes)`.
- This flow is considered IDSW-critical and should be preserved unless an explicit ablation proves a replacement is better.
- Current tracking execution flow is `scripts/track_videos.py` -> `python -m pig_behavior.tracking.cli`.
- `track_videos.py --eval-config <name>` should stay aligned with `evaluate_tracking.py` named presets and forward them as `--profile-override KEY=VALUE`.
- `pig_behavior.tracking.cli` must keep the module entrypoint and `--profile-override` support; otherwise `track_videos.py --eval-config` either exits without running or fails argument parsing.
- `--no-emit-hidden-tracks` is an output-labeling control for CVAT relabeling: keep tracker-maintained/interpolated boxes, but export their `Hidden` attribute as `No`. It must not be treated as disabling internal hidden state, association, motion prediction, occlusion holding, or smoothing.
- Treat Tracking moi bat smooth as the current quality baseline when reading reports.
- Tracking moi tat smooth/yolov8 is still a relevant runtime variant, but its reported metrics are currently worse.
- For optimizer default target-video diagnostics, do not pin `000263`/`000302`.
- Instead derive the weak default target set from the current no-smooth baseline metrics file:
  `outputs/eval/hybrid_bytetrack/Tracking mới tắt smooth/yolov8/iou0_area0_condarea0_merge0/tracking_metrics.csv`
- Do not include detector-only presets (`det_conf`, `max_raw_detections`, `nms_iou` only) in the default optimizer scopes.
- Artifact `outputs/eval/hybrid_bytetrack/overnight_iou0/optimizer/tracking_optimizer_summary.csv` showed detector-only presets matched `base` metrics for both smooth and no-smooth.
- Detector-only checks now belong in explicit `--scope detector_probe` runs or explicit `--preset` runs.

## Investigation focus

- Keep focus on runtime and code-path differences inside hybrid_bytetrack.
- Primary suspects remain association.py raw_id owner/penalty/bypass logic and all_detection_indices matching.
- Secondary suspect remains forced post-processing in runner.py for hybrid_bytetrack.

## Guardrails

- Do not blame detector weight for the 000263 regression.
- Do not enable condarea by default without an explicit ablation.
- Prefer small, reversible patches.
## 2026-07-04 hard-scene improvement plan

User requested the plan be remembered and executed. Preserve current strong
baseline first: `hybrid_bytetrack + smooth_det020_loose +
iou0_area0_condarea0_merge0`, especially keeping `Pigs291119_000302_30fps = 0`
IDSW. Do not promote broad offline repair by default. Episode-level pair swap
repair remained opt-in and did not change the hard 4-video eval because the
remaining failures are not simple visible short-overlap geometry swaps:
`000231` involves Hidden/visible behavior, `000328` involves longer conflict,
and `000263` motion cost favors keeping current geometry.

Execution order:

1. Add opt-in association diagnostics first (`association_debug=true`) to record
   assignment accept/reject events, raw owner, top raw ID, split recovery,
   ambiguity, cost, threshold, and detection metadata. This must not change
   behavior when disabled.
2. Use diagnostics around IDSW frames to classify failures as
   `fight_rotate_bbox`, `long_occlusion_reentry`, `hidden_owner_steal`, or
   `raw_id_bypass_error`.
3. Patch only one narrow opt-in guard at a time in `association.py`:
   `ambiguity_owner_guard`, `hidden_owner_guard`, `raw_owner_quarantine`, then
   `long_occlusion_reentry_guard`.
4. Validate on hard set `000231/000263/000328/000302` first. Promote only if
   total hard-set IDSW drops, `000302` stays 0, and the 9-video baseline does
   not regress.

Implementation started:

- `association_debug=true` adds opt-in assignment diagnostics and remains off by
  default.
- `ambiguity_owner_guard=true` adds the first narrow opt-in guard: if a detection
  raw ID belongs to another candidate owner and that owner cost is close to the
  selected assignment, reject the likely raw-owner steal instead of letting a
  marginal assignment rewrite identity. This is intended for fighting/rotating
  bbox scenes and must be validated on the hard 4-video set before any broader
  promotion.
- User reported run `outputs/eval/hybrid_bytetrack/20260704_090756` had no
  meaningful metric change. Diagnostics under the matching prediction root show
  `assignment_reject_ambiguous_raw_owner = 0` for `iou0_area0_condarea0_merge0`,
  so the first guard did not trigger. Continue with `hidden_owner_guard=true`:
  when a detection raw ID belongs to a hidden/lost owner but is assigned to a
  different track, freeze identity learning for that assignment while still
  allowing bbox update. This remains opt-in and must be tested on the hard set.
- User reported `outputs/eval/hybrid_bytetrack/20260704_100102/.../merge0`
  unchanged. Diagnostics show `hidden_owner_freeze=True` triggered only once
  (`000231` frame 401), while `000263`, `000302`, and `000328` had zero hidden
  owner freezes. Because freezing identity learning did not change the exported
  bbox/label assignment, continue with a stricter opt-in
  `hidden_owner_guard_hold_assignment=true`: when the same hidden-owner conflict
  is detected, hold the assigned track instead of consuming the ambiguous
  detection. This is expected to affect at most the trigger frames and must be
  tested with `association_debug=true` before considering any promotion.
- User reported improvement on
  `outputs/eval/hybrid_bytetrack/20260704_103036/smooth_det020_loose/iou0_area0_condarea0_merge0`.
  Diagnostics show `assignment_hidden_owner_hold` triggered exactly once:
  `Pigs291119_000231_30fps` frame 401. The remapped IDSW events for `000231`
  disappeared; remaining switches are `000263` frames 193/195 and `000328`
  frames 1342/1360. `000302` remains clean in this hard-set run. Keep
  `hidden_owner_guard_hold_assignment` opt-in until 9-video regression is run.
  Next work should target `000263`/`000328` with a separate reentry/quarantine
  guard rather than broadening hidden-owner hold.
## 2026-07-04 reentry ambiguous hold candidate

After user reported improvement on
`outputs/eval/hybrid_bytetrack/20260704_103036/smooth_det020_loose/iou0_area0_condarea0_merge0`,
diagnostics confirmed `assignment_hidden_owner_hold` triggered once on
`Pigs291119_000231_30fps` frame 401 and removed the `000231` remapped IDSW
events. Remaining hard-set switches are `000263` frames 193/195 and `000328`
frames 1342/1360; `000302` remains clean. Keep hidden-owner hold opt-in until
9-video regression passes.

Next candidate added as opt-in only: `reentry_ambiguous_hold=true`. If a track is
OCCLUDED/LOST/MISSING or has enough missed frames and the assignment is already
marked ambiguous, hold the track instead of consuming the detection. Test this
separately from hidden-owner hold on the hard 4-video set.
## 2026-07-04 reentry hold retest result

User reported
`outputs/eval/hybrid_bytetrack/20260704_105654/smooth_det020_loose/iou0_area0_condarea0_merge0`
had real effect from `reentry_ambiguous_hold`. The old `000328` remapped IDSW
events at 1342/1360 disappeared and total remapped switch count dropped versus
the pre-guard baseline. However new remapped switches appeared (`000231` frame
325 and `000263` frames 475/1125), and debug showed reentry holds firing broadly
from early frames. Do not promote this broad version.

Narrowing applied: `reentry_ambiguous_hold` now requires prior stable detections
(`ever_detected` and at least `reentry_ambiguous_hold_min_hits`) and no longer
uses bare `MISSING` state as a trigger. Retest narrowed reentry hold alone before
combining with hidden-owner hold or running 9-video regression.
## 2026-07-04 reentry hold narrowed again

User provided
`outputs/eval/hybrid_bytetrack/20260704_112422/smooth_det020_loose/iou0_area0_condarea0_merge0`.
The narrowed reentry hold still fired far too broadly: thousands of
`assignment_reentry_ambiguous_hold` events per video starting at early frames
(e.g. `000231` from frame 3, `000328` from frame 7). Do not promote this
version. Tightened the helper again so `track.missed >=
reentry_ambiguous_hold_min_missed` is mandatory before OCCLUDED/LOST or
prediction/occlusion reason can trigger a hold. Retest this stricter version
alone; expected trigger count should drop from thousands to localized reentry
spans.

## 2026-07-05 practical hard-set config

Treat `hidden_owner_guard=true` plus `hidden_owner_guard_hold_assignment=true`
as the current practical hard-set improvement path. It preserved the clean
`000302` baseline and solved the known `000231` frame-401 hidden-owner failure
in the later 3-video/4-video checks. Keep it opt-in until broader regression
passes, but use it as the base when developing the next `000328` fix.

Do not continue tuning `reentry_ambiguous_hold` thresholds as the main path.
Runs through `20260705_152555` showed that hold-based reentry gates either fired
too broadly and damaged `000231`/`000302`, or became too narrow and missed the
`000328` switch. The `reentry_unowned_raw_mismatch_reject`/quarantine branch
also failed to recover `000328=0` without collateral effects: when broad enough
to affect `000328`, it damaged `000302`; when seed-gated, it no longer changed
`000328`. Treat those as diagnostic opt-ins, not promotion candidates.

Next direction: build a separate episode-level detector for `000328` style
failure. It should look for repeated unowned raw-ID mismatch conflicts over a
short window before taking action, rather than acting on each assignment
independently. Preserve `hidden_owner_guard_hold_assignment` as the `000231`
protection while testing this new branch.

## 2026-07-05 practical hard-set clarification

Use `hidden_owner_guard=true` plus `hidden_owner_guard_hold_assignment=true` as the current practical opt-in base for hard-set work. It fixed the known `000231` frame-401 hidden-owner failure and preserved `000302=0` in later checks.

Do not keep tuning `reentry_ambiguous_hold` or simple `reentry_unowned_raw_mismatch_reject`/quarantine thresholds as the main path. Those branches either damaged `000231`/`000302` when broad enough, or missed `000328` when narrowed.

The next branch is episode-level: detect repeated unowned raw-ID mismatch conflicts over a short frame window before rejecting. This is intended for the `000328` 1340-range failure while keeping hidden-owner hold as the `000231` protection.

## 2026-07-05 successful hard-set candidate

User reported and diagnostics confirmed `outputs/eval/hybrid_bytetrack/20260705_220622/smooth_det020_loose/iou0_area0_condarea0_merge0` is the current successful hard-set candidate.

Metrics: `000231=0`, `000263=2`, `000328=0`, `000302=0`, `ALL=2` remapped IDSW.

Candidate config for full-video validation before base promotion:

- `hidden_owner_guard=true`
- `hidden_owner_guard_hold_assignment=true`
- `reentry_unowned_raw_mismatch_episode_reject=true`
- `reentry_unowned_raw_mismatch_episode_action=hold`
- `reentry_unowned_raw_mismatch_episode_max_events=8`
- `reentry_unowned_raw_mismatch_episode_min_missed=1`
- `reentry_unowned_raw_mismatch_episode_max_missed=20`
- `reentry_unowned_raw_mismatch_episode_max_cost=0.36`
- `association_debug=true` for diagnostics only, not promotion behavior.

Observed guard effects: `000231` used `assignment_hidden_owner_hold` at frame `401`; `000328` used `assignment_hold_reentry_unowned_raw_mismatch_episode` at frame `1342`; `000302` had no guard trigger and stayed IDSW `0`.

Remaining `000263` switches are frames `193` and `195`, track `3/4` during fight/occlusion. Raw IDs are still consistent (`track 3 -> raw 6`, `track 4 -> raw 7`), so this is not the raw-ID mismatch failure class. User noted this may be GT ambiguity because visually the two pigs exchange IDs while fighting. Do not add a broad runtime guard for this before visual/GT confirmation.

## 2026-07-06 next weak-video tracking plan

Keep the current successful hard-set candidate as the protected base. Future
work is experimental until it proves no regression on the guardrail videos,
especially `Pigs291119_000302_30fps = 0` IDSW. The two remaining weak videos
should not be treated as one failure class.

For `Pigs291119_000263_30fps`, the remaining switches are around frames
`193/195` during close fight/occlusion between tracks `3/4`. Diagnostics showed
raw IDs remain consistent (`track 3 -> raw 6`, `track 4 -> raw 7`), so this is
not a raw-ID mismatch or hidden-owner steal. The next candidate should be a
very narrow visible-assignment guard, such as `visible + ambiguous + same_raw +
selected_cost high`, with a hold/freeze action over a short span. Do not use a
broad raw mismatch/reentry rule for this case.

Important clarification from the earlier read-only audit of
`notebooks/01_data_preparation/update_ids_for_annotation.ipynb`,
`DAT_Update_ID_For_Annotate.ipynb`, and early/stable tracker commits: the useful
lesson for `000263` is not raw ByteTrack ID ownership. The old annotation/update
flow stabilized identity with short-window local motion, roughly a 6-frame
window, and preferred a gap/prediction over accepting a bad high-cost match. The
notebook used a tighter matching threshold (`COST_THR = 0.60`) than the current
runtime reid/lost path (`lost_track_cost_threshold = 0.95`).

The key `000263` diagnostic sequence to preserve:

- frame `193`: track `3` misses assignment; track `4` accepts raw `7` with cost
  about `0.437596`.
- frame `194`: track `3` accepts raw `6` with cost about `0.743141`, which is
  high enough that the notebook-style logic would likely hold/predict instead
  of accepting.
- frame `195`: track `3` accepts raw `6` with cost about `0.177293`; track `4`
  accepts raw `7` with cost about `0.489884`.

Therefore the safest `000263` experiment is an opt-in
`occlusion_reid_prefer_gap_over_bad_match` style guard for fight/occlusion
geometry: `phase=reid`, track state `OCCLUDED/LOST`, `ambiguous=true`,
`same_raw_id=true`, short missed span, and `selected_cost > 0.60` or `0.65`.
The action should hold/predict/gap-fill instead of accepting the high-cost
detection. This should be tested separately from the `000233` different-raw
long-occlusion guard and validated carefully because a broad reid threshold
tightening can increase FN/fragments.

For `Pigs291119_000233_30fps`, the failures include short high-cost same-raw
confusions around `923/924` and `939/941`, plus longer mismatches after
occlusion around `1111-1242` and `1424+`. This looks like long-occlusion reid
accepting a bad high-cost target after `occlusion_hold`, often with different
or unowned raw IDs. The next candidate should target `phase=reid`,
`track_source=occlusion_hold`, enough `missed` frames, high selected cost, and
different/unowned raw ID, with an initial hold action rather than a broad reject.
Do not globally set broad `same_raw_only=false`; previous probes suggested it
would fire too often in other videos.

Validation order: test the `000263` and `000233` guards separately, then combine
only if each improves its target. The promotion gate remains the 5-video hard set
`000231/000233/000263/000328/000302`: `000231=0`, `000328=0`, `000302=0`,
`000263` does not regress and preferably improves, `000233` improves clearly,
and total remapped IDSW does not increase on the broader set. Frame/window gates
are acceptable for diagnosis only; promoted logic must be based on runtime
state, not hardcoded video IDs or frame numbers.

## 2026-07-07 000233 guarded improvement candidate

New best opt-in 5-video hard-set candidate:
`outputs/eval/hybrid_bytetrack/20260707_082640/smooth_det020_loose/iou0_area0_condarea0_merge0`.

Metrics versus `outputs/eval/hybrid_bytetrack/Best_tracking/iou0_area0_condarea0_merge0`:

- `Pigs291119_000231_30fps`: stayed `0` remapped IDSW.
- `Pigs291119_000233_30fps`: improved from `9` to `6` remapped IDSW.
- `Pigs291119_000263_30fps`: stayed `2` remapped IDSW.
- `Pigs301119_000328_30fps`: stayed `0` remapped IDSW.
- `Pigs291119_000302_30fps`: stayed `0` remapped IDSW.
- `ALL`: improved from `11` to `8` remapped IDSW on this 5-video set.

Winning add-on config on top of the protected practical base:

- `occlusion_reid_prefer_gap_over_bad_match=true`
- `occlusion_reid_bad_match_action=reject`
- `occlusion_reid_bad_match_same_raw_only=false`
- `occlusion_reid_bad_match_raw_mismatch_only=true`
- `occlusion_reid_bad_match_unowned_raw_only=true`
- `occlusion_reid_bad_match_occlusion_hold_only=true`
- `occlusion_reid_bad_match_min_missed=7`
- `occlusion_reid_bad_match_max_missed=12`
- `occlusion_reid_bad_match_min_cost=0.55`
- `occlusion_reid_bad_match_max_cost=0.70`

Diagnosis: for `000233`, the useful rejections are bad-but-plausible unowned
raw mismatch reid assignments around the long occlusion region, especially raw
`26` near frames `1114-1118`. A broader reject/hold version damaged metrics or
regressed `000231`. The max-cost upper bound is important: without it, a single
very high-cost reject around `000231` frame `906` caused new switches at
`909/912`. Keep this candidate opt-in until broader full-set regression passes.

Next remaining target is `000263=2`. Do not use the `000233` raw-mismatch guard
for `000263`; the `000263` failure remains same-raw fight/occlusion geometry
around frames `193/195`.

## 2026-07-07 suffix repair 000263 candidate

New best current 5-video opt-in candidate:
`outputs/eval/hybrid_bytetrack/codex_suffix_5video_min1500/iou0_area0_condarea0_merge0`.

Metrics:

- `Pigs291119_000231_30fps`: stayed `0` remapped IDSW.
- `Pigs291119_000233_30fps`: stayed `6` remapped IDSW versus the 000233 guarded candidate.
- `Pigs291119_000263_30fps`: improved from `2` to `0` remapped IDSW.
- `Pigs301119_000328_30fps`: stayed `0` remapped IDSW.
- `Pigs291119_000302_30fps`: stayed `0` remapped IDSW.
- `ALL`: improved from `8` to `6` remapped IDSW versus `20260707_082640`.

Winning add-on is `suffix_pair_swap_repair=true` on top of the protected
practical config and the 000233 guarded config. Keep it opt-in until broader
regression passes.

Diagnosis: `000263` is a suffix identity crossing after heavy overlap/fight,
not a raw-ID mismatch. The useful repair swaps the `Pig_3`/`Pig_4` suffix after
the uncertain overlap around frames `193/195`. The first broad suffix repair with
`suffix_pair_swap_min_suffix_frames=60` fixed `000263` but produced false suffix
swaps on guardrail videos (`000231`, `000233`, `000328`, `000302`). The current
default `suffix_pair_swap_min_suffix_frames=1500` is intentionally conservative
and removed those false swaps in the 5-video run.

Next validation step: run a broader regression/full set with this exact opt-in
candidate before any base promotion. The remaining weak target is `000233=6`;
do not weaken the suffix gate just to chase `000233`, because the broad version
already proved unsafe.

## 2026-07-07 000233 failed repair probes

Keep `outputs/eval/hybrid_bytetrack/codex_suffix_5video_min1500/iou0_area0_condarea0_merge0`
as the protected current best candidate. Do not promote the later 000233 probes:

- `20260707_122454`: enabling existing local/episode/long pair swap repairs on
  top of the best candidate did not change `000233`; remapped IDSW stayed `6`.
- `20260707_123316`: aggressively loosening local/episode/long repair thresholds
  also did not change `000233`; remapped IDSW stayed `6`.
- A new experimental hidden-overlap suffix repair was implemented and verified
  locally, but the single-video run `20260707_145820` worsened `000233` from
  `6` to `10` remapped IDSW, adding switches around `973/1081` and `1138/1144`.
  The code was reverted and must not be reintroduced without a stronger
  discriminator.
- Loosening existing suffix repair for overlapped suffixes
  (`suffix_pair_swap_min_suffix_frames=600`,
  `suffix_pair_swap_max_suffix_overlap_iou=1.0`) in `20260707_150456` also
  worsened `000233` from `6` to `10` and badly reduced IDF1/coverage.

Diagnostics: upper-bound GT-aware simulation shows that manually swapping
`ID_2/ID_8` at frame `923`, `ID_1/ID_8` at frames `939-940`, and `ID_1/ID_8`
from frame `1111` onward could make `000233` reach `0` IDSW without changing
FP/FN. However, those fixes rely on GT/evaluator knowledge: runtime motion gain,
raw IDs, and hidden-overlap signals are not distinctive enough. Hidden-overlap
runs similar to the desired `1111-1118` segment also occur earlier (`973-982`,
`1053-1062`) where swapping is harmful. Avoid hardcoded video/frame repair in
promotable tracking logic.

## 2026-07-07 overlap small-box suppression candidate

New best current 5-video opt-in candidate:
`outputs/eval/hybrid_bytetrack/codex_overlap_suppress_5video/iou0_area0_condarea0_merge0`.

Metrics:

- `Pigs291119_000231_30fps`: stayed `0` remapped IDSW.
- `Pigs291119_000233_30fps`: improved from `6` to `2` remapped IDSW.
- `Pigs291119_000263_30fps`: stayed `0` remapped IDSW.
- `Pigs301119_000328_30fps`: stayed `0` remapped IDSW.
- `Pigs291119_000302_30fps`: stayed `0` remapped IDSW.
- `ALL`: improved from `6` to `2` remapped IDSW versus the suffix candidate.

Winning add-on is `overlap_small_box_suppression=true` on top of the protected
practical config, the `000233` occlusion-reid guard, and
`suffix_pair_swap_repair=true`. Default thresholds are intentionally conservative:
`overlap_small_box_min_iou=0.40`,
`overlap_small_box_max_area_ratio=0.65`, and
`overlap_small_box_max_score=0.75`.

Diagnosis: the early `000233` switches at `923/924` and `939/941` are not raw-ID
owner failures. The runtime keeps the expected IDs, but the evaluator matches GT
`ID_8` to a neighboring smaller low-confidence box because its IoU is slightly
higher during heavy overlap. The new opt-in post-processing marks those small
low-confidence overlapped boxes Hidden, removing the short IDSW bounces. The
remaining `000233` switches are `1111/1119`, a harder `ID_1/ID_8` long conflict
that should not be fixed by broad suffix or GT-aware swaps.

Keep this candidate opt-in pending broader/full-set regression before base
promotion.

## 2026-07-07 hidden suffix ID-swap candidate

New best current 5-video opt-in candidate:
`outputs/eval/hybrid_bytetrack/codex_hidden_suffix_id_swap_5video/iou0_area0_condarea0_merge0`.

Metrics:

- `Pigs291119_000231_30fps`: stayed `0` remapped IDSW.
- `Pigs291119_000233_30fps`: improved from `2` to `0` remapped IDSW.
- `Pigs291119_000263_30fps`: stayed `0` remapped IDSW.
- `Pigs301119_000328_30fps`: stayed `0` remapped IDSW.
- `Pigs291119_000302_30fps`: stayed `0` remapped IDSW.
- `ALL`: improved from `2` to `0` remapped IDSW versus the overlap-suppress
  candidate.

Winning add-on is `hidden_suffix_id_swap_repair=true` on top of the protected
practical config, the `000233` occlusion-reid guard, `suffix_pair_swap_repair`,
and `overlap_small_box_suppression`.

Diagnosis: after the small-box suppression candidate, the only remaining
`000233` switches were `1111/1119` between `ID_1` and `ID_8`. Hide/unhide
simulations only moved the switch; only a suffix identity swap from frame `1111`
to the end removed both switches. The promotable discriminator is intentionally
narrow: a low-confidence hidden run that is long enough but not too long,
strongly overlaps one visible partner, then has a long common suffix. Defaults:

- `hidden_suffix_id_swap_min_hidden_frames=8`
- `hidden_suffix_id_swap_max_hidden_frames=15`
- `hidden_suffix_id_swap_min_overlap_iou=0.70`
- `hidden_suffix_id_swap_max_hidden_median_score=0.50`
- `hidden_suffix_id_swap_start_back_frames=7`
- `hidden_suffix_id_swap_min_suffix_frames=600`

On the 5-video run this detected the `000233 ID_8/ID_1` suffix crossing without
triggering regressions on `000231`, `000263`, `000328`, or `000302`. Keep this
opt-in pending broader/full-set regression before base promotion.

## 2026-07-07 broader regression correction

The broader regression run
`outputs/eval/hybrid_bytetrack/20260707_174142/smooth_det020_loose/iou0_area0_condarea0_merge0`
proved the previous full 5-video stack is not a safe common baseline. It fixed
the target videos (`000233=0`, `000263=0`) but regressed previously clean videos:

- `Pigs281119_000085_30fps`: `0 -> 2` remapped IDSW.
- `Pigs291119_000225_30fps`: `0 -> 2` remapped IDSW.

Ablation on `000085/000225/000233/000263` isolated the issue:

- `ablate_control_assoc_occlusion_4video`: `000085=0`, `000225=0`, `000233=6`, `000263=2`.
- `ablate_suffix_only_4video`: `000085=2`, `000225=2`, `000233=6`, `000263=0`.
- `ablate_overlap_only_4video`: `000085=0`, `000225=0`, `000233=2`, `000263=2`.
- `ablate_overlap_hidden_no_suffix_4video`: `000085=0`, `000225=0`, `000233=0`, `000263=2`.

Decision: do not promote `suffix_pair_swap_repair=true` in its current form. It
fixes `000263` but creates false suffix swaps on clean videos. The current safest
common candidate for broader validation is:

- protected association/occlusion practical base:
  `hidden_owner_guard=true`,
  `hidden_owner_guard_hold_assignment=true`,
  `reentry_unowned_raw_mismatch_episode_reject=true`,
  `reentry_unowned_raw_mismatch_episode_action=hold`,
  `reentry_unowned_raw_mismatch_episode_max_events=8`,
  `reentry_unowned_raw_mismatch_episode_min_missed=1`,
  `reentry_unowned_raw_mismatch_episode_max_missed=20`,
  `reentry_unowned_raw_mismatch_episode_max_cost=0.36`,
  `occlusion_reid_prefer_gap_over_bad_match=true`,
  raw-mismatch/unowned/occlusion-hold-only with `min_missed=7`,
  `max_missed=12`, `min_cost=0.55`, `max_cost=0.70`.
- add `overlap_small_box_suppression=true`.
- add `hidden_suffix_id_swap_repair=true`.
- explicitly keep `suffix_pair_swap_repair=false`.

Next step: run broader/full regression with this no-suffix common candidate. The
remaining `000263=2` should be addressed by a new, narrower discriminator rather
than by current suffix repair.

## 2026-07-07 no-suffix common candidate full regression

Full 12-video validation of the no-suffix common candidate passed:
`outputs/eval/hybrid_bytetrack/no_suffix_common_candidate_full/iou0_area0_condarea0_merge0`.

Compared with `outputs/eval/hybrid_bytetrack/Best_tracking/iou0_area0_condarea0_merge0`:

- `ALL` remapped IDSW improved `11 -> 2`.
- No video increased remapped IDSW.
- `Pigs291119_000233_30fps` improved `9 -> 0`.
- `Pigs291119_000263_30fps` stayed `2`; this is the remaining target.
- Guardrail videos stayed clean: `000085=0`, `000225=0`, `000231=0`,
  `000302=0`, `000328=0`.

Current safest broader candidate:

- protected association/occlusion practical base.
- `overlap_small_box_suppression=true`.
- `hidden_suffix_id_swap_repair=true`.
- `suffix_pair_swap_repair=false`.

Do not promote the previous full stack from `20260707_174142`; it included
`suffix_pair_swap_repair=true` and caused false switches on `000085` and
`000225`. Future `000263` work should either build a new narrower discriminator
or heavily gate suffix repair so it cannot trigger on clean videos.
