# Project Memory - Medium Term

## 2026-07-12 active classification_v2 focus

The current medium-term project focus is the `classification_v2` Q2 behavior
recognition roadmap, not tracking ablation.

Current state:

- The data/review/train-ready path has been upgraded toward a multimodal
  spatio-temporal design using bbox actor images, ROI relations, motion,
  social/partner context, and leakage-safe tabular features.
- Canonical image cache is letterboxed:
  `outputs/classification_v2/image_cache_v2_letterbox/`.
- Packed actor and visual-context caches are expected to be reused for full OOF.
- The latest pre-full progress report is `PASS_PARTIAL_ROADMAP` with all
  pre-full gates passing. This is ready for human authorization review, not a
  completed paper result.
- Full OOF is intentionally blocked until explicit authorization is written and
  the execution gate passes.
- After full OOF, required postrun artifacts are calibration, confusion-focus
  comparison, ablation reporting, experiment registry, and completion gate.

The tracking notes below are preserved because they still matter if the user
returns to tracking. They are not the active workstream.

## Historical tracking memory

This week the main focus is recovering and tuning the tracking pipeline after the architecture was split from legacy `tracking_engine.py` into `src/pig_behavior/tracking/*`.

Key points:

1. Legacy 21/06 had one one-way tracking flow and no `cfg.mode`.
2. Current code has multiple modes, and `hybrid_bytetrack` is not fully equivalent to legacy.
3. Current best baseline is `hybrid_bytetrack + iou0_area0_condarea0_merge0`.
4. `Pigs291119_000302_30fps` improved mainly because of the new detector weight.
5. `Pigs291119_000263_30fps` increased IDSW from ≈2 to ≈6 with both old and new weights on current code, so the cause is code/pipeline behavior.
6. Main suspect: `association.py`, especially raw_id logic and `all_detection_indices` matching.
7. Secondary suspects: forced post-processing by mode and detection filtering differences from legacy.
