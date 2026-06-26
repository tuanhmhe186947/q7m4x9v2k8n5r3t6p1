# Current Technical Decision

## Settled facts

- `Pigs291119_000302_30fps` improved mainly because of the new detector weight.
- `Pigs291119_000263_30fps` IDSW increase is not caused by weight.
- User confirmed both old and new weights produce IDSW ≈ 6 on current code for `000263`.
- Legacy 21/06 produced IDSW ≈ 2 for `000263`.
- Therefore `000263` regression must be investigated as code/config/runtime behavior regression.

## Current baseline

```text
tracking_mode = hybrid_bytetrack

USE_IOU_FALLBACK = False
USE_AREA_OCCLUSION_FREEZE = False
USE_CONDITIONAL_AREA_OCCLUSION_FREEZE = False
USE_MERGED_BOX_SPLIT = False

config name = iou0_area0_condarea0_merge0
```

## Current hypothesis

Most likely cause of `000263` IDSW regression:

1. `hybrid_bytetrack` currently uses raw ByteTrack ID owner/penalty/bypass logic.
2. `hybrid_bytetrack` matches `all_detection_indices` too early.
3. `hybrid_bytetrack` is not equivalent to legacy one-way `tracking_engine.py` from 21/06.
4. Forced post-processing by mode may also contribute.

## Next recommended patch

Patch `association.py` first:

- Do not apply raw_id owner/penalty/bypass logic to `hybrid_bytetrack`.
- Keep raw_id logic only for `bytetrack_raw` if needed.
- Let `hybrid_bytetrack` use safer high-confidence / low-confidence matching, closer to legacy/non-ByteTrack behavior.
- Do not change weight, detector, condarea, evaluation, or thresholds in the first patch.
