# Benchmark Notes

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
