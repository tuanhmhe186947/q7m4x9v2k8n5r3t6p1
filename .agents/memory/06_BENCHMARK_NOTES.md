# Benchmark Notes

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
