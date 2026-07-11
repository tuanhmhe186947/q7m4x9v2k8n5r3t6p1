# Project Rules

## 2026-07-03 IDSW guard rules

1. Preserve the split lost-track reacquire guard implementation that produced:
   `outputs/eval/hybrid_bytetrack/20260703_193439/smooth_det020_loose/iou0_area0_condarea0_merge0/tracking_metrics.csv`.
2. Current best tradeoff for `000231` + `000302` requires:
   - `lost_track_reacquire_guard=true`
   - `lost_track_reacquire_non_same_raw_distance_guard=false` as the default/base setting
   - `lost_track_reacquire_raw_owner_guard=true`
   - `lost_track_different_raw_hidden_owner_bypass=true`
   - `lost_track_different_raw_hidden_owner_min_missed=2`
   - `lost_track_different_raw_hidden_owner_min_center_gain=0.03`
3. Do not turn off `lost_track_reacquire_raw_owner_guard` globally; it fixes `000302` but damages `000231`.
4. Do not remove the conditional different-raw hidden-owner bypass without an ablation against `000231` and `000302`.
5. Do not assume appearance threshold tuning alone solves this tradeoff; tested `0.15` did not change the `000231=8`, `000302=0` result.
6. Do not reintroduce the need for `--profile-override lost_track_reacquire_non_same_raw_distance_guard=false`; this is now the base/default so tracking/eval/optimizer use it automatically.

## General rules

1. Always preserve the user's current experimental conclusion unless existing files clearly contradict it.
2. Do not repeatedly reopen settled hypotheses.
3. Do not blame weight for `000263` IDSW increase.
4. Prefer small, reversible patches.
5. Do not mix unrelated changes in one patch.
6. Do not run long benchmark/tracking unless the user explicitly requests.
7. When asked to audit, do not modify code.
8. When asked to patch, modify only the requested scope.
9. Always state which files were changed.
10. Always state which behavior changed and which behavior was intentionally not changed.
11. Always report which memory files were read before making changes.
12. Keep code lines within the repository formatter/linter limit before commit. Wrap long conditions, strings, and argument lists proactively so pre-commit does not fail on line length.

## Tracking-specific rules

1. For `evaluate_tracking.py` behavior and metric comparisons, treat commit `b697c4eba36db280cbf01f446873da17bcac509d` as the main historical reference unless the user explicitly asks for another snapshot.
2. Do not assume `hybrid_bytetrack` is already legacy-compatible.
3. Do not assume folder name `iou0_area0_condarea0_merge0` proves runtime flags were correct; inspect config/runtime path if needed.
4. Preserve the `runner.py` post-processing gates that improved IDSW:
   - identity guard: `cfg.enable_offline_smoothing and cfg.identity_swap_guard`
   - temporal refinement and overlap hidden island stabilization: `cfg.enable_offline_smoothing and (cfg.smooth_boxes or cfg.refine_boxes)`
   - `stabilize_overlap_hidden_islands(shapes, cfg)` must run after `refine_shapes_temporally(...)` in that second block.
4. Keep `hybrid_bytetrack` default rule flags OFF unless explicitly requested.
5. Do not enable `condarea` by default unless the user asks or ablation proves it.
6. Be careful with raw ByteTrack IDs; they may be unstable after occlusion.
7. For `000263` IDSW, inspect association logic before changing detector.
8. For `000302` improvement, remember it is attributed to weight, not necessarily tracking logic.
9. XML CVAT export is a support output, not the main objective.
10. Main objective is stable identity tracking for 8 pigs.

## Code-change rules

1. If changing `association.py`, isolate one behavior at a time:
   - raw_id logic
   - matching phase
   - lost/reid handling
2. If changing `runner.py`, do not silently force offline smoothing by mode.
3. If changing `detections.py`, document how it differs from legacy `tracking_engine.py`.
4. If changing `config.py`, document default mode/rule behavior clearly.
5. If changing evaluation path, ensure stale XML cannot be confused with fresh XML.

## Verification rules

When user permits running checks, verify in this order:

1. Static/syntax/import check.
2. Single video `Pigs291119_000263_30fps`.
3. Single video `Pigs291119_000302_30fps`.
4. 3-video common set:
   - `Pigs281119_000085_30fps`
   - `Pigs291119_000263_30fps`
   - `Pigs291119_000302_30fps`
5. 7-video full set.

Metrics to watch:

- `remapped_idsw`
- `remapped_idf1_pct`
- `remapped_hota_pct`
- `remapped_fragments`
- `gap_tolerant_fragments`
- `fp`
- `fn`

## Preserved legacy agent rules

The previous `.agents/AGENTS.md` remains in place and contains broader repository coding standards for PyTorch, OpenCV, GPU fallback, quality checks, and command formatting. Root `AGENTS.md` is now the main Codex entrypoint; use the preserved file as supplemental implementation guidance when relevant.
