# Project Memory - Medium Term

This week the main focus is recovering and tuning the tracking pipeline after the architecture was split from legacy `tracking_engine.py` into `src/pig_behavior/tracking/*`.

Key points:

1. Legacy 21/06 had one one-way tracking flow and no `cfg.mode`.
2. Current code has multiple modes, and `hybrid_bytetrack` is not fully equivalent to legacy.
3. Current best baseline is `hybrid_bytetrack + iou0_area0_condarea0_merge0`.
4. `Pigs291119_000302_30fps` improved mainly because of the new detector weight.
5. `Pigs291119_000263_30fps` increased IDSW from ≈2 to ≈6 with both old and new weights on current code, so the cause is code/pipeline behavior.
6. Main suspect: `association.py`, especially raw_id logic and `all_detection_indices` matching.
7. Secondary suspects: forced post-processing by mode and detection filtering differences from legacy.
