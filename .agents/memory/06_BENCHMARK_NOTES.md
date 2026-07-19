# Benchmark Notes

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
