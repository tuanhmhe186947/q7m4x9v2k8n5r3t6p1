# Project Memory - Medium Term

## 2026-07-12 active classification_v2 focus

The current medium-term project focus is the `classification_v2` Q2 behavior
recognition roadmap, not tracking ablation.

Current state:

- The data/review/train-ready path has been upgraded toward a multimodal
  spatio-temporal design using bbox actor images, ROI relations, motion,
  social/partner context, and leakage-safe tabular features.
- The active rebuild has 245,664 enhanced rows and a valid 5,171-item Hidden v5
  template. Human Hidden decisions are incomplete.
- Behavior review coverage is 3/4,670 units, with 4,667 missing and one pending.
- Therefore no reviewed train-ready snapshot is currently valid for new model
  experiments. Complete review and freeze new hashes before model smoke.
- A full 13-fold engineering OOF run exists at commit `18d6692`, but it belongs
  to the previous unreviewed lineage and is not the final Q2 result.
- Canonical actor cache remains letterboxed. Rebuild it or verify its hash
  against the future reviewed snapshot before reuse.
- Current status authority is `docs/CLASSIFICATION_V2_CURRENT_STATE.md`.

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
