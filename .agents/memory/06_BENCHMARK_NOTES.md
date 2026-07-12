# Benchmark Notes

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
- Required validation: full multimodal OOF with native folds, not random
  frame/window split.
- Main comparison axis: multimodal learned model versus registered native,
  tabular linear, and tabular nonlinear controls.
- Required postrun outputs: metrics, predictions, prediction schema audit,
  calibration audit, calibrated predictions, confusion-focus comparison,
  high-confidence hard errors, source-balanced report, and experiment registry.
- Do not use pilot or smoke metrics as paper-facing full results.

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
