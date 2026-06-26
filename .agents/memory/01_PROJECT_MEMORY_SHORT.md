# Current Short Memory

- Do not blame weight for `Pigs291119_000263_30fps` IDSW increase.
- User confirmed both old and new weights produce IDSW ≈ 6 on current code for `000263`.
- Legacy 21/06 produced IDSW ≈ 2 for `000263`.
- Therefore focus on code/config/runtime behavior.
- `Pigs291119_000302_30fps` improvement is due to the new detector weight; do not use it as proof that `hybrid_bytetrack` matches legacy.
- Current preferred baseline: `hybrid_bytetrack + iou0_area0_condarea0_merge0`.
- Do not enable `condarea` by default without ablation.
- Primary suspect: `association.py` raw_id owner/penalty/bypass and `all_detection_indices` matching for `hybrid_bytetrack`.
- Secondary suspect: `runner.py` forced post-processing for `hybrid_bytetrack`.
- Patch in small steps. Do not change weight/detector/evaluation first.
