# Benchmark Notes

## 2026-07-29 historical H5b/H4 reproduction failure

- Frozen config/cache full-run repeatability: PASS across 13 videos twice.
- Historical prediction parity: FAIL (`5920` identity differences, `1959`
  Hidden-state differences, `187129` bbox tolerance violations).
- Reproduced Standard-V2 authority: HOTA `0.8498733886710031`, DetA
  `0.9044987387056114`, AssA `0.7995528004092259`, LocA
  `0.9239078255704691`, IDF1 `0.9140811965811966`, IDSW `64`.
- Identity severity: wrong-ID frames `32125`, episodes `68`, recovered `41`,
  terminal `26`, persistent pairwise swaps `11`.
- Classification: `FAIL_METRIC_PARITY`. The missing historical raw detector
  rows remain the principal unresolved evidence boundary.

## 2026-07-27 RF_ACC23 authority correction and artifact recovery

Evidence classes are marked explicitly. Nothing below is promoted beyond the
class stated.

`CODE_VERIFIED` (this repository, HEAD `5fa23de`):

- RF_ACC23 promotion SHA is `d925c9004e7aff5a3c8469b158d2428432c6031a`, which
  is also the last commit to touch `src/pig_behavior/tracking/`.
- The tracking source tree has been byte-identical since that commit:
  `d925c90:src/pig_behavior/tracking` and `HEAD:src/pig_behavior/tracking` both
  resolve to `8ba3d50d8322ab1c72b8f50b6ce4c9b1013f799a`. Semantic equivalence is
  exact at the source-tree level.
- Effective semantic config export SHA-256 (this repository's export payload):
  `9bf4ce6d07423ab517b4705c716e3eb012349b756b7c0591cc3458eac207808d`.
- `detect_every_n_frames=2`, `det_conf=0.25`, `occlusion_aware_matching=False`,
  `enable_offline_smoothing=False`; no post-video repair in `realtime_fast`.
- Timing contract resolves through `resolve_output_timing_contract` to
  `causal_framewise` with output delay `0`.
- `causal_hidden_detection_reservation` is **not** enabled in RF_ACC23. It
  exists only in `REALTIME_BALANCED_CONFIG` under a different confidence regime
  (`det_conf=0.20`, `min_iom=0.96`, `min_gain=0.17`).
- Correctness gates: `python -m pytest tests/test_tracking_profiles.py
  tests/test_tracking_prefix_invariance.py tests/test_tracking_repeatability.py
  tests/test_tracking_telemetry.py tests/test_tracking_no_mp4.py
  tests/test_tracking_baseline_lock.py tests/test_tracking_improvements.py -q`
  gives `180 passed`.

`ARTIFACT_VERIFIED` (recovered read-only from the `PIG_task_tracking` worktree,
branch `task/update-tracking`; hashes in
`docs/TRACKING_RF_ACC23_ARTIFACT_RECOVERY_20260727.md`):

- Full-13 quality, `run_id=20260723_rf_acc23_full13_instrumented_v1`, commit
  `b0d9009`: IDSW `53`, HOTA `0.9704398315450558`, IDF1 `0.9707702337312571`,
  FP/FN `486/610`, fragments `107`, quality gate `PASS`.
- Evaluation contract: `include_hidden=true`,
  `iou0_area0_condarea0_merge0`, `delay_frames=0`, `causal_framewise`,
  `detect_every_n_frames=2`, FP32, `uses_future_frames=false`.
- Hard6 identity: IDSW `55 -> 49`, HOTA `91.317 -> 94.487%`, IDF1
  `90.593 -> 94.289%`, wrong-ID matched-animal frames `7722 -> 4531`, with FP,
  FN and fragments unchanged.
- Full-13 runtime **FAIL** against frozen gates: loop `32.38` vs `>=36.84`,
  p95 `58.25 ms` vs `<=44.13`, end-to-end `23.37` vs `>=26.80`. The artifact
  attributes this to host power/clock drift, not to the algorithm. Native
  30 FPS was not achieved on full-13.
- No RF_ACC23 artifact authorized promotion: `promote_profile=false`,
  `full13_authorized=false`, `promotion_authorized=false` in all three decision
  documents.

`CONTRADICTED`:

- Hard6 membership is **not** `000114, 000231, 000233, 000263, 000327, 000302`.
  The run artifact records `000216, 000226, 000231, 000233, 000263, 000302`.
- The RF_ACC23 metrics did **not** come from `d925c90`. They come from
  `b0d9009` (full-13) and `0b40423` / `2bcdbfc` (Hard6). `d925c90` is the
  promotion commit only and produced no metric.

`UNRESOLVED`:

- Wrong-ID `8579 -> 5219` for RF_ACC23. The Hard6 artifact records
  `7722 -> 4531`; those digits appear elsewhere only in the 2026-07-19 hybrid
  lane decision, a different lineage.
- The `000302` RF_ACC23 guardrail. `000302` appears in the Hard6
  `unchanged_videos` list; no standalone RF_ACC23 IDSW for it was recorded.
- GT authority for `000216`.
- Whether the recovered lineage's tracking tree (`752d55d3…`) is semantically
  equivalent to the promoted tree on `main` (`8ba3d50d…`). Until that is shown,
  these numbers describe the recovered lineage, not `main`.

`MEMORY_VERIFIED_HISTORICAL`, not to be relabeled:

- The 2026-07-20 entry (`realtime_fast` far-right guard, IDSW `59`, HOTA
  `95.63%`, IDF1 `95.37%`, FP/FN `486/610`, fragments `107`) belongs to the
  **pre-RF_ACC23** `realtime_fast` lineage and is RF_ACC23's parent. It is not
  superseded and must not be relabeled as RF_ACC23.

`USER_SUPPLIED_UNVERIFIED`:

- `docs/external_audits/BAO_CAO_AUDIT_RF_ACC23_20260726.md`, SHA-256
  `bed2245df8cc85755a59b28223f2f9bd6797a8d8a897397f51c8d631c5942b0f`. Never
  present in repository history; imported as an external document, not as
  metric authority. Its lead to the `PIG_task_tracking` lineage was correct and
  enabled the artifact recovery above.

Locked input lineage (13 videos, GT, detector weight, config, environment):
`outputs/tracking/rf_acc23_lineage/rf_acc23_lineage_manifest_20260727_v1.json`,
SHA-256
`0cfb26acc7766e05c497d9efdfafa40dc92f2d5c527e0338b89602eef0838dfc`. The
`000263` ground truth uses the verified `Tracking_annotation_*` filename
variant. The 13-video set remains a development/evaluation set, not an unbiased
final test set.

## 2026-07-19 seed-matched base and all-seven fusion result

- Scope correction: this all-seven result is a naive-concatenation endpoint,
  not an optimized fusion result. The subset ladder, fixed-subset fusion-family
  comparison, and failure-attribution diagnostics remain unmeasured.
- Among seven measured frozen-ResNet18 temporal bases, A128 is the controlled
  measurement base: macro-F1 `0.4016091326`, accuracy `0.6680497925`, and NLL
  `0.9581952707`. This is not a global-best architecture claim.
- The seed-matched all-seven real fusion has macro-F1 `0.3990463886`, accuracy
  `0.6182572614`, and NLL `1.2583320608`. It beats same-width zero by
  `+0.0597244883` and availability-only by `+0.0475594846` in macro-F1, but
  both paired cluster intervals include zero.
- Against same-run actor-only, real fusion changes macro-F1 by
  `-0.0025627441`, with interval `[-0.0870213782, 0.0927085883]`, and worsens
  accuracy and NLL. Thus real values contain signal, while naive concatenation
  is not the final fusion structure.
- Per-class real-minus-actor F1 deltas are: `drink -0.047619`, `eat +0.078078`,
  `fight -0.400000`, `social-nose 0.000000`, `explore -0.065692`,
  `lying -0.084046`, `stand +0.285714`, `move +0.285714`,
  `sitting -0.077778`, and `playwithtoy 0.000000`.
- Continue with the five-stage authority: controlled strong base, matched
  modality/fusion screen, behavior-specific fusion selection, rented-GPU joint
  tuning, then confirmatory modality ablation on the tuned finalist.
- Preserve superseded seed runs and all valid caches, predictions, and
  diagnostics. Reuse them when contracts match; rerun only for semantic change
  or failed artifact integrity.

## 2026-07-19 C6 per-behavior modality screen

- The comparator below is short-run `actor_only`; values are descriptive F1
  point deltas from the two repeat packets, not reviewed-lineage confidence
  claims. All ten classes remain in scope.
- `drink` (`n=8`): ROI `+0.235`, geometry `+0.154`, motion `+0.145`, numeric
  social `+0.101`.
- `eat` (`n=17`): pen `+0.118`, ROI `+0.073`, union `+0.033`, motion and
  numeric social `+0.017`.
- `fight` (`n=4`): numeric social `+0.036`, motion `+0.016`; support is too
  small for a promotion claim.
- `social-nose` (`n=8`): motion and numeric social each `+0.063`.
- `explore` (`n=43`): pen `+0.032`.
- `lying` (`n=56`): union `+0.063`, ROI `+0.047`, numeric social `+0.044`,
  pen `+0.029`.
- `stand` (`n=11`): pen `+0.123`.
- `move` (`n=8`): motion `+0.076`, numeric social `+0.069`, ROI `+0.061`,
  geometry `+0.014`, union `+0.004`.
- `sitting` (`n=85`): union `+0.099`, ROI `+0.093`, pen `+0.055`, motion
  `+0.015`.
- `playwithtoy` (`n=1`): no positive point delta; support is insufficient.
- Full-frame context has no positive per-class delta against actor-only in this
  short packet. Some real-versus-zero effects differ because parameter-matched
  zero and availability controls answer different mechanism questions.
- These signals keep all seven branches eligible for predeclared
  behavior-conditional retest on the frozen reviewed main lineage. They do not
  override global NLL, calibration, availability, uncertainty, or harm gates.

## 2026-07-17 classification temporal base screening

- Stage A used seven controlled modes on 245 native validation units from 33
  video clusters, with matched folds, optimizer exposure, and paired
  video-cluster uncertainty.
- `SF128` macro-F1 is `0.355783`. `A128` macro-F1 is `0.372852`, but the
  operational `A128-SF128` interval crosses zero and locomotion macro-F1 drops
  by `0.086392`; therefore `A128` is a mixed-reviewed `RETEST`, not a carry.
- TCN and Transformer fail their parameter-matched mechanism controls. Do not
  spend legacy compute on their order/timing diagnostics.
- Stage C retains centered C6 as the one-sequence legacy view. S6 and C8 do
  not improve macro-F1 under matched exposure.
- Pen-boundary context is a valid short negative globally, with conditional
  target-group signals. Reassess it only as an isolated branch on reviewed
  mixed data.
- Final base selection requires paired pooled, per-source, source-balanced,
  missingness, target-group, uncertainty, resource, and lineage evidence on a
  frozen mixed-reviewed snapshot. No legacy-only rank is final.
- Stage A v3 SHA256:
  `b3250ed5391d46e37469a22f16353bbc5f038fa250897c37056fe64a132a6910`.
## 2026-07-20 Fast promotion and runtime boundary

- `realtime_fast` far-right guard full-13 primary/repeat: IDSW `59`, HOTA
  `95.63%`, IDF1 `95.37%`, FP/FN `486/610`, fragments `107`, gap-tolerant
  fragments `8`; no per-video IDSW regression and `000302=0`.
- The candidate/repeat loop-FPS is `19.08/25.48`; hard-7 candidate/parent
  runtime ratio `0.9588` passes its frozen screen, but full-13 timing is noisy.
  No speed advantage is claimed and native realtime is not yet locked.
- Raw authority remains `145/88.91%/88.47%` and is not rerun. Balanced keeps
  the quality trade-off (`121/95.68%/95.76%`); Quality remains delay-`-1`
  upper-bound evidence.
- Balanced far-right screening produced no gain in W01 `000327` or W02
  `000302`; both parent/candidate windows were perfect ties. The family is
  rejected without a full-video or full-13 run.
- Authority: `docs/TRACKING_REALTIME_PARETO_SELECTION_DECISION_20260720.json`.

## 2026-07-19 realtime selection metrics are multi-objective

- Accuracy is necessary but not sufficient for a realtime winner. The frozen
  comparison includes IDSW/per-video identity, HOTA/IDF1, FP/FN, fragments,
  effective FPS, loop-FPS, p50/p95 latency, stage timing, delay, memory,
  repeatability, lineage and zero-MP4 compliance.
- Raw authority PASS: IDSW `145`, HOTA `88.91%`, IDF1 `88.47%`, loop-FPS
  `22.65/27.03`, p95 `108.04/54.31 ms`, repeat loop-FPS ratio `1.193`, and
  recursive `mp4_count=0`.
- Fast is the current causal reference (IDSW `69`), but its profile-specific
  `000302` guard is open (`6` versus ceiling `2`) and its speed claim is not
  authorized until raw/Fast/Balanced use one comparable runtime harness.
  Balanced's higher HOTA/IDF1 does not dominate its IDSW gap; Quality's
  finite-delay runtime gate remains failed. No final operational winner is
  locked yet.
- Authority:
  `docs/TRACKING_REALTIME_PARETO_SELECTION_DECISION_20260719.json`.

## 2026-07-19 RQ4 Quality copy-performance decision

- Schema-aware motion-pair cloning preserves every tested global/fixed-lag
  output and improves the real-artifact median from `1.4471 s` to `0.6514 s`.
- QW01 primary/repeat are quality-identical: IDSW `32`, HOTA `94.75%`, IDF1
  `95.94%`, FP/FN `5/6`, semantic hash `8472ed87...86857`.
- Postprocess falls to `876.25/794.78 ms`, but end-to-end effective FPS is
  only `19.42/16.91`; p95 is `60.84/91.02 ms`. Repeat loop FPS is only
  `85.56%` of primary, below the frozen `90%` repeatability gate.
- The causal parent control also misses the session floor at `23.64 FPS` and
  p95 `47.41 ms`. This exposes host variation but does not authorize relaxing
  the frozen Quality thresholds after observing the result.
- All primary, repeat, and parent prediction/eval roots recursively contain
  zero MP4. Retain commit `1a1d008` as implementation improvement; reject RQ4
  profile promotion and all later funnel stages.
- Decision authority:
  `docs/TRACKING_RQ4_QUALITY_COPY_PERFORMANCE_DECISION_20260719.json`.

## 2026-07-19 realtime Pareto selection

- Fast is the current causal identity reference: IDSW `69`, HOTA `94.35%`,
  IDF1 `93.91%`, FP/FN `506/630`, and repeatability PASS. Its `000302` guard
  is still open (`6` versus ceiling `2`).
- Balanced is non-dominated on localization/coverage quality (HOTA `95.68%`,
  IDF1 `95.76%`, FP/FN `448/586`) but has IDSW `121`, a gap of `52` to Fast,
  and misses the locked Balanced target `86`.
- Quality has the highest post-video HOTA/IDF1 but is not a realtime authority:
  delay `-1` is global-graph evidence, and finite-delay RQ4 fails runtime.
- Therefore no final operational winner is locked. After the Fast guard and
  common-harness runtime audit pass, the comparison can be raw -> Fast ->
  hybrid under one contract.
- The three profiles remain available as evidence, but Quality cannot be
  skipped in future winner decisions and may replace Fast if a valid candidate
  later passes every frozen gate and wins the Pareto comparison.
- Authority:
  `docs/TRACKING_REALTIME_PARETO_SELECTION_DECISION_20260719.json`.

## 2026-07-19 hybrid lane completion

- The promoted `hybrid_bytetrack_best` resolves all full-13 identity residuals:
  13 videos have IDSW `0`, and the remapped identity-event CSV has zero rows.
- Final aggregate quality is HOTA `98.35062270%`, IDF1 `99.14903846%`,
  FP/FN `1593/1593`, strict fragments `426`, and gap-tolerant fragments `6`.
- Remaining weak videos are localization/continuity residuals rather than
  identity-switch clusters. The lowest HOTA is `000216` at `95.81526065%`,
  with FP/FN `325/325` and IDSW `0`.
- The predeclared aspirational IDSW target and every hybrid stop gate pass.
  Further tuning on these 13 development videos is stopped to limit overfit.
- Realtime planning is open with `realtime_fast` as the operational reference
  and `realtime_balanced` as the first optimization target. Full-13 remains
  gated by window, full-video, and hard-set evidence.

## 2026-07-19 hybrid H5b plus H4 promotion authority

- Paired full-13 control and combined candidate improve aggregate IDSW
  `8 -> 0`, HOTA `98.33687663% -> 98.35062270%`, IDF1
  `99.14102564% -> 99.14903846%`, and FP/FN
  `1603/1603 -> 1593/1593`.
- H5b alone changes only `000233`, fixing IDSW `4 -> 0`; the other 12 video
  metrics are identical. H4 then changes exactly 10 `ID_7` bbox rows on
  `000328`, fixing IDSW `4 -> 0`; the other 12 video geometries are unchanged.
- Primary and repeat combined metrics match across all 14 rows. The auditor
  verifies 26 semantic predictions, 46 artifacts, current input hashes, four
  zero-IDSW critical-video guards, and `mp4_count=0`.
- Repeatability authority SHA256 is
  `c88fe8241c85c609b67753ad229e3bcf23ac9f8ca8ac8d3485a8fcbafd8327bb`.
- Strict fragments and tracklets each increase by one, while gap-tolerant
  fragments stay at `6`; aggregate MOTP IoU decreases by
  `0.0000195086394853`. These are retained as non-blocking trade-offs.
- This is a post-video quality promotion with no speed claim. Exact config,
  lineage, rollback, and claim limits are recorded in
  `docs/TRACKING_PROMOTION_DECISION_20260719_HYBRID_H5B_H4.json`.

## 2026-07-18 hybrid near-wall Hidden bbox geometry promotion

- Parent/candidate aggregate: matches `185570 -> 185578`, FP/FN
  `1630/1630 -> 1622/1622`, IDSW `8 -> 8`, HOTA
  `98.31% -> 98.32%`, and MOTA `98.25% -> 98.26%`.
- Raw IDF1 improves by `0.0000427350`; both sides round to `99.13%`.
  Raw MOTP decreases by `0.0000055769` because eight additional difficult
  matches enter the matched population.
- Exactly 111 bbox rows change: `000114=5`, `000231=28`, `000233=78`.
  Ten videos are metric-identical. All 26 replay manifests pass shape-key and
  non-geometry payload equality.
- Strict fragments increase `426 -> 427`, while gap-tolerant fragments stay
  `6`. The known `000233` ID_6 frames 1140-1147 IoU decrease remains above
  `0.90` and creates no FP/FN or IDSW regression.
- Clean authority commit is `b66428e`; the repeatability authority SHA256 is
  `6b6899109ddbca43042645b503896e41c985bd838eb64513eeb72a9210015665`.
  It verifies 26 predictions, 46 artifacts, current input hashes, explicit
  geometry runtime absence, and `mp4_count=0`.
- Algorithm/profile commits are `3391dbd` and `4876217`. Full lineage is in
  `docs/TRACKING_PROMOTION_DECISION_20260718_HYBRID_NEAR_WALL_GEOMETRY.json`.

## 2026-07-18 realtime_balanced hidden-reservation promotion

- Parent and candidate roots are `20260718_80e4600_parent_eval_full13` and
  `20260718_80e4600_gain017_alt025_full13`; the independent repeat ends in
  `gain017_alt025_repeat_full13`, under `outputs/eval/tracking_windows`.
- The candidate uses `include_hidden=true`, start frame `0`,
  `iou0_area0_condarea0_merge0`, causal delay `0`, and no generated MP4.
- Aggregate IDSW improves `133 -> 121`, IDF1 `93.71% -> 95.76%`, HOTA
  `93.93% -> 95.68%`, FP/FN `449/587 -> 448/586`, and fragments `130 -> 127`.
  Five videos improve; eight tie; none regress in IDSW, IDF1, or HOTA.
- `max_alternative_cost=0.30` is negative evidence: it reaches IDSW `119` but
  drops `000216` IDF1/HOTA from `99.55%/99.26%` to `90.10%/91.45%` through
  persistent ID 5/8 corruption. The promoted `0.25` blocks cost `0.283780`
  and retains the useful `000233` cost `0.238421`.
- Repeatability authority SHA256 is
  `757d57b146b98d047b583f1e3025480b9646fda60a25543e29ee6df0d0b91429`;
  it verifies 26 predictions, 72 artifacts, matching semantic predictions,
  runtime guardrails, and `mp4_count=0`.
- Profile promotion commit is `e8d39d7`; exact config, hashes, and rollback
  are in
  `docs/TRACKING_PROMOTION_DECISION_20260718_REALTIME_BALANCED_HIDDEN_RESERVATION.json`.

## 2026-07-18 realtime_fast visible-competitor preference promotion

- Parent authority:
  `outputs/eval/tracking_candidate_locks/`
  `20260717_94a8232_includehidden_realtime_fast_baseline_repeatability_v1.json`.
- Candidate primary and repeat:
  `20260718_7f36b57_r1_visible_prefer_fast_full13_primary_v1` and
  `20260718_7f36b57_r1_visible_prefer_fast_full13_repeat_v2` under
  `outputs/eval/tracking_candidates`.
- The only profile change is
  `realtime_visible_better_competitor_prefer=true` in `realtime_fast`.
  Both runs use `include_hidden=true`, the corrected rule combo, causal delay
  `0`, and no generated MP4.
- Aggregate quality changes are IDSW `87 -> 69`, HOTA `93.89% -> 94.35%`,
  IDF1 `93.21% -> 93.91%`, FP/FN `564/688 -> 506/630`, and fragments
  `114 -> 110`. `000231` improves IDSW `30 -> 12`; all other videos tie and
  no video regresses. Primary/repeat semantic prediction hashes match.
- Recursive audits report zero MP4 in both prediction and evaluation roots.
  Primary/repeat tracking totals are about `860.27/831.58` seconds; mean
  per-video effective FPS is `27.40/28.16`. Runtime is not promoted as a
  speed claim because the repeats differ.
- The profile promotion is commit `456fc97`; the complete decision and
  rollback are in
  `docs/TRACKING_PROMOTION_DECISION_20260718_REALTIME_FAST_VISIBLE_PREFER.json`.

## 2026-07-18 realtime R0 runtime causality gate

- `realtime_fast` and `realtime_balanced` were run on `000263` through frame
  210 and independently through frame 240, covering the hard occlusion at
  frames `193/195` before appending 30 future frames.
- Both profiles preserve all 1,680 already-flushed XML box payloads, declare
  `causal_framewise` with delay `0`, and have zero MP4 across prediction and
  evaluation roots.
- Auditor and tests are committed at `f8e1b6e`. The two immutable audit JSONs
  are under `outputs/eval/tracking_causality/`
  `20260718_f8e1b6e_r0_prefix_invariance_audit_v1`.
- Hashes, run IDs, claim boundary and rollback are locked in
  `docs/TRACKING_CAUSALITY_DECISION_20260718_R0.json`.

## 2026-07-18 hybrid far-camera identity guard promotion

- Parent authority:
  `outputs/eval/tracking_candidates/20260717_6e65b22_baseline_include_hidden_audit_v1`.
- Candidate primary and repeat:
  `20260717_6e65b22_identity_guard_far_full13_primary_v1` and
  `20260717_6e65b22_identity_guard_far_full13_repeat_v1` under
  `outputs/eval/tracking_candidates`.
- Both full-13 runs use `include_hidden=true` and return IDSW `8`, versus parent
  `10`. The only identity change is `000216: 2 -> 0`; no video regresses.
- Report-precision HOTA/IDF1 remain `98.31%/99.13%`. Raw FP/FN change
  `1628 -> 1630`; `000216` improves by five each and `000302` worsens by seven
  each. This is accepted as an overall quality improvement with the local
  geometry trade-off retained for the next isolated ablation.
- Recursive no-output-video audit across target, guardrail, primary, repeat,
  and prediction roots found `mp4_count=0`.
- Algorithm and profile commits are `7254670` and `e74a8fa`. Roll back the
  profile first, then the algorithm, using the exact order recorded in
  `docs/TRACKING_PROMOTION_DECISION_20260718_HYBRID_FAR_IDENTITY_GUARD.json`.
- Runtime timing varied between repeats, so this decision authorizes a quality
  promotion but no speed claim.
## 2026-07-16 tracking P0 historical baseline lock

- Generator commit: `6265f1b3f0d80d622a5d2727cfb6cb1d62aca5d7`.
- Audit:
  `outputs/eval/baseline_locks/20260716_p0_tracking_baselines.json`.
- Audit status is `INCOMPLETE`, not promotion-grade `PASS`.
- Hybrid `20260707_230230` and all five mode-comparison branches have the same
  13-video/GT universe SHA256:
  `30cc7ea80fb1d0a23bc1e3c4d38e15b76e3bf86b8cc244f65afe7e0fb38af980`.
- Metrics match the accepted baselines, all report roots have MP4 count `0`,
  and detector/mask SHA256 values are locked.
- All historical prediction roots are absent: hybrid is missing 13 XML files
  and each mode-comparison branch is missing 13 XML files.
- Therefore these runs remain metrics/report evidence only. Do not claim
  prediction-byte lineage or use them as a repeatability PASS.
- Before ablation, regenerate a no-MP4 baseline under fresh prediction/eval
  roots with commit-bound run and artifact manifests.
- Audit SHA256:
  `6ce129972b7bdb422d51edbf91ef88cf1a886814964bb344c46661949580da53`.

## 2026-07-16 legacy L6 full-frame-context short decision

- The parameter-matched zero, availability-only, and full-frame modes use the
  same 245 native units, 33 video clusters, seed, 134,924 parameters, and 30
  optimizer steps. All repeats are deterministic.
- Native macro-F1 is `0.2697662759`, `0.2721987509`, and `0.2942624204`.
- Full-frame minus zero is `+0.0244961445`, with cluster interval
  `[-0.0668714797, 0.0725200014]`; accuracy changes by `+0.0122448980` and
  NLL worsens by `+0.2414525889`.
- Full-frame minus availability-only is `+0.0220636696`, with interval
  `[-0.0809709233, 0.0671747502]`; NLL worsens by `+0.3144303865`.
- Decision: `DO_NOT_EXPAND_FULL_FRAME_CONTEXT_FROM_CURRENT_SHORT_EVIDENCE`.
  Do not run a full confirmation or carry full-frame values into the candidate.
  This is mixed legacy-only evidence, not a merged-data architecture rejection.
- Decision artifact SHA256:
  `e006dc6636ede5a35e71414448be1dc96f0f71e29f5f2a1b6d0230fa0c49c6bf`.

## 2026-07-15 legacy L6 ROI relation full confirmation

- The zero, availability-only, and ROI controls share the same T6 native-unit
  universe, seed, 70,704 parameters, and 30 optimizer steps.
- Native macro-F1 is `0.2420943922`, `0.2405788407`, and `0.2886109023`,
  respectively; all repeat artifacts are deterministic.
- ROI minus zero is `+0.0465165101`, with 33-video cluster interval
  `[0.0134524177, 0.0768469401]` and NLL delta `-0.0912769139`.
- ROI minus availability-only is `+0.0480320616`, with interval
  `[0.0178011460, 0.0771756533]`. Availability-only minus zero is effectively
  null, with interval `[-0.0122444323, 0.0100985531]`.
- The exact full confirmation is complete. Full zero, availability-only, and
  ROI macro-F1 are `0.4966025667`, `0.4727197983`, and `0.5082292933`.
- ROI minus zero is `+0.0116267266`, with 33-video cluster interval
  `[-0.0398806556, 0.0906766805]`; ROI minus availability-only is
  `+0.0355094951`, with interval `[-0.0248897889, 0.0986581204]`.
- Availability-only minus zero is `-0.0238827684`, with interval
  `[-0.0629523019, 0.0339059054]`.
- ROI has positive class-specific feeding evidence against zero: feeding-group
  macro-F1 rises by `+0.1796877378`, `drink` F1 rises from `0.3703703704` to
  `0.6486486486`, and `eat` F1 rises from `0.7906976744` to `0.8717948718`.
- `playwithtoy` support is one validation unit. Recall stays `1.0`, but F1
  falls from `0.6666666667` to `0.3333333333` as false positives rise from
  one to four; this class cannot support a `legacy_16f` ROI conclusion.
- Decision: `DO_NOT_EXPAND_ROI_RELATION_FROM_CURRENT_SHORT_EVIDENCE`.
  The full ROI gain misses the required margin and positive interval-low gate;
  availability-only also fails its bounded-difference check. Do not carry ROI
  values into the next candidate. Reassess ROI on merged-reviewed data, whose
  rare-behavior support is materially larger.
- Full artifact:
  `l6r_full_decision_v1.json` under
  `outputs/classification_v2/legacy_only_unreviewed_development/l6r_full_v1/`.
  SHA256: `5a9a2b4b61b7ddeef0b5155ec69b678d73f0acd53917db98d1d6271cab5f1af3`.

## 2026-07-15 legacy L6 motion benchmark

- The parameter-matched zero, availability-only, and motion controls use the
  same T6 native-unit universe, seed, 69,664 parameters, and 30 steps.
- Native macro-F1 is `0.2620738697`, `0.2572327329`, and `0.2602600258`,
  respectively; all repeat artifacts are deterministic.
- Motion minus zero is `-0.0018138438` with 33-video cluster interval
  `[-0.0260250944, 0.0233049368]`; motion is not promoted to full legacy
  development despite its NLL improvement.
- Preserve this as `legacy_16f` negative evidence only. Reassess on merged data
  with materially greater rare-behavior support.

## 2026-07-13 focused classifier critical path

Canonical plan:
`plans/classification_v2-core-classifier-roadmap.md`.

- P0-P8 optimize and validate the 10-class classifier only.
- Five-class, paper reproduction, publication, and integration are optional P9.
- Visual pilots separate resolution from backbone:
  `R18/160 -> R18/224 -> R34/224`.
- Architecture search uses the selected ResNet18 view where practical; finalists
  are confirmed with the selected stronger visual backbone.
- Temporal controls are pooling, TCN, then a small Transformer if justified.
- Primary temporal view has six slots; native 6/16 is a source-shortcut ablation.
- Context gains require matched-subset, availability-only, modality-dropout,
  all-source, and source-probe controls.
- Initial imbalance choices are event-balanced CE, effective-number CE, and
  Balanced Softmax; select one policy.
- Hierarchy follows strong-baseline error analysis and reviewed attributes.
- Full OOF is limited to F0, F1, F2, plus no-hierarchy when scientifically needed.

## 2026-07-13 Bergamini five-class comparison protocol

Use three separately named tracks:

1. `strict5_learned`: canonical `stand`, `lying`, `move`, `eat`, and `drink`
   only, using identical grouped native-unit folds across learned models.
2. `coarse5_attribute`: operational mapping from reviewed posture and locomotion
   attributes; never silently map `sitting` or `explore` by class name.
3. Paper-aligned hybrid: movement threshold, ROI/orientation, and binary
   lying-standing ResNet18. Call it exact reproduction only after physical and
   temporal calibration plus primary-source protocol verification pass.

The 10-class task remains the thesis primary result. ResNet18 `160x160` is the
engineering pilot; pretrained ResNet34 `224x224` is the main visual baseline
after runtime gates. Offline longitudinal analysis is primary, while causal
near-real-time is a separately evaluated secondary protocol.

## 2026-07-19 Pig-STRENet artifact canary

Canary09 used input SHA256
`d76a0b5b12e1c38eb050e7787776e51df8110114bf8afb1c2ecff847918e5805`
and passed with 8 native events, 8 causal pairs, 96 slots, 288 all-class ROI
rows and 288 fixed top-K social edges. Native-event mass min/max was `1.0`.
Stabilized actor-crop differences had shape `[8,11,32,32]` with zero skipped
pairs. No training, OOF or data writes occurred.

The social tensor contract was fixed `K=3` for all 96 pair-slot groups. ROI
dynamics packed to `[288,11]`; social edges packed to `[288,19]`. Geometry
selection was available for all 288 actor-ROI rows. Full-scene ROI union pixel
patches were correctly masked unavailable because no scene-image path exists in
the input CSV. This is not evidence against ROI visual context.

This result validates export, leakage controls, masks, event weighting and
lineage only. It is not an accuracy result and cannot select a modality,
fusion architecture or classifier.

## 2026-07-19 Pig-STRENet real XML technical canary

The bounded real-CVAT source `data/annotations/classification/
Pigs281119_000085.xml` was processed under
`xml-only-unreviewed-technical-canary` at
`outputs/classification_v2/agent_audits/pig_strenet_xml_real_20260719_canary01/`
`07_pig_strenet_attempt2`.
The run passed with 2,400 native actor-target pairs, 28,800 slots, 86,400
all-class ROI rows, and 86,400 fixed-K social edges. Event mass was exactly
`1.0`; causal checks found no future-frame use and all tensor/model-X schemas
matched Canary09.

Eight pairs at target frame zero have unavailable history, as required by the
causal boundary; 2,392 pairs have complete six-frame history. Difference maps
were explicitly skipped and full-scene ROI pixels remained blocked because
the XML-derived table has no scene-image path. The result is technical export
evidence only: no training, OOF, accuracy claim, review completion or
promotion is authorized.

## Active classification_v2 benchmark direction

Current behavior-recognition benchmark target:

- Claim level: Q2 internal recording-date/video-safe improvement only.
- Immediate prerequisite: complete Hidden and behavior review, then freeze a
  new reviewed train-ready snapshot. No architecture promotion uses the current
  incomplete review payload.
- Required validation: full multimodal OOF with native folds, not random
  frame/window split.
- Main comparison axis: multimodal learned model versus registered native,
  tabular linear, and tabular nonlinear controls.
- Required postrun outputs: metrics, predictions, prediction schema audit,
  calibration audit, calibrated predictions, confusion-focus comparison,
  high-confidence hard errors, source-balanced report, and experiment registry.
- Do not use pilot or smoke metrics as paper-facing full results.
- The old commit-`18d6692` full run (`macro-F1=0.4156053847`) is a diagnostic
  engineering baseline only because its data lineage predates current review
  gates. It is not a promotion target for the final reviewed evaluation.

Historical tracking benchmark notes below are preserved only for tracking work.

## Important conclusions

- `000302` improvement is attributed mainly to the new detector weight.
- `000263` IDSW regression is not caused by detector weight.
- User confirmed both old and new weights produce IDSW ≈ 6 on current code for `000263`.
- Legacy 21/06 had IDSW ≈ 2 on `000263`.

## Current preferred baseline

```text
hybrid_bytetrack
iou0_area0_condarea0_merge0
```

## Metrics to prioritize

- IDSW
- IDF1
- HOTA
- fragments
- gap-tolerant fragments
- FP/FN

## Common comparison set

```text
Pigs281119_000085_30fps
Pigs291119_000263_30fps
Pigs291119_000302_30fps
```
