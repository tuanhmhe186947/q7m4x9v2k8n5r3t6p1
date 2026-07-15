# Benchmark Notes

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
