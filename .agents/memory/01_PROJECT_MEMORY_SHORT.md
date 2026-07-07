# Project Memory Short

- Do not hardcode `000263`/`000302` as optimizer target videos anymore.
- For optimizer ranking defaults, derive weak target videos from:
  `outputs/eval/hybrid_bytetrack/Tracking mới tắt smooth/yolov8/iou0_area0_condarea0_merge0/tracking_metrics.csv`
- Current weakest set from that file is:
  - `Pigs291119_000263_30fps`
  - `Pigs291119_000226_30fps`
  - `Pigs301119_000327_30fps`
  - `Pigs301119_000328_30fps`
- For `evaluate_tracking.py` metric comparisons, use commit `b697c4eba36db280cbf01f446873da17bcac509d` as the relevant historical reference instead of legacy 21/06.
- Critical IDSW-preserving tracking flow in `src/pig_behavior/tracking/runner.py`:
  - `apply_identity_swap_guard(...)` runs only when `cfg.enable_offline_smoothing and cfg.identity_swap_guard`.
  - `refine_shapes_temporally(...)` and then `stabilize_overlap_hidden_islands(...)` run only when `cfg.enable_offline_smoothing and (cfg.smooth_boxes or cfg.refine_boxes)`.
  - Do not change this back to `cfg.enable_offline_smoothing or cfg.mode == "hybrid_bytetrack"`; that drift was identified as a likely cause of worse IDSW versus commit `b697c4eba36db280cbf01f446873da17bcac509d`.
- Current tracking CLI flow:
  - Use `scripts/track_videos.py` for batch/single-video tracking.
  - `track_videos.py` calls `python -m pig_behavior.tracking.cli`; `src/pig_behavior/tracking/cli.py` must keep its `__main__` entrypoint.
  - `track_videos.py --eval-config <name>` reuses `evaluate_tracking.py` presets and passes them to tracking CLI as `--profile-override KEY=VALUE`.
  - `track_videos.py` must pass `src` via `PYTHONPATH` to the subprocess so module execution works without editable install.
  - `--no-emit-hidden-tracks` keeps tracker-maintained/interpolated boxes in the output but writes their `Hidden` attribute as `No` for CVAT relabeling; it does not disable internal tracking/association/occlusion state.
- Current runtime variants to compare are:
  - C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\eval\hybrid_bytetrack\Tracking moi tat smooth\yolov8
  - C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\eval\hybrid_bytetrack\Tracking moi bat smooth
- In reports, no smooth is currently worse than smooth; do not assume the unsmoothed runtime is the better baseline.
- Optimizer default scopes should stay tracking-focused.
- Detector-only presets were moved to explicit `--scope detector_probe` because `overnight_iou0` showed detector-only metrics were identical to `base`.
- Continue focusing on code and runtime behavior in association.py and runner.py, not detector weight.
- 2026-07-05 practical hard-set direction:
  - Use `hidden_owner_guard=true` + `hidden_owner_guard_hold_assignment=true` as the current practical opt-in base for hard-set work; it solves the `000231` frame-401 hidden-owner issue while preserving `000302=0` in later checks.
  - Do not keep tuning `reentry_ambiguous_hold` or `reentry_unowned_raw_mismatch_reject`/quarantine thresholds as the main path; those branches either damaged `000231`/`000302` or missed `000328`.
  - Next `000328` work should use a separate episode-level repeated unowned raw-ID mismatch detector, not per-assignment hold/reject rules.
# 2026-07-03 Best tradeoff found: lost-track reacquire split guards

- New strong 2-video tradeoff result:
  `outputs/eval/hybrid_bytetrack/20260703_193439/smooth_det020_loose/iou0_area0_condarea0_merge0/tracking_metrics.csv`
- Config was `smooth_det020_loose` plus `lost_track_reacquire_non_same_raw_distance_guard=false`.
- Key metrics:
  - `Pigs291119_000231_30fps`: IDSW `2`, HOTA `0.9705892717094201`, IDF1 `0.9847241970177549`.
  - `Pigs291119_000302_30fps`: IDSW `0`, HOTA `0.9930104703678451`, IDF1 `0.9964355605255801`.
  - `ALL`: IDSW `2`, HOTA `0.9820366705826231`, IDF1 `0.9907038986528682`.
- Preserve the current split `lost_track_reacquire_guard` design in `association.py` / `config.py`:
  - keep `lost_track_reacquire_guard=true`;
  - `lost_track_reacquire_non_same_raw_distance_guard=false` is now the default/base setting after the strong 9-video `20260703_194929` run;
  - do not disable raw-owner guard globally because it fixes `000302` but badly hurts `000231`;
  - preserve the conditional `lost_track_different_raw_hidden_owner_bypass` with `min_missed=2` and `min_center_gain=0.03`.
- `outputs/eval/hybrid_bytetrack/20260703_194929/smooth_det020_loose/iou0_area0_condarea0_merge0/` validated this as a good 9-video base; future tracking/eval/optimizer runs should not require a long override for this guard.

- 2026-07-05 successful candidate `20260705_220622`: hard-set remapped IDSW `000231=0`, `000263=2`, `000328=0`, `000302=0`, `ALL=2`. Candidate config: hidden-owner hold plus `reentry_unowned_raw_mismatch_episode_reject=true`, `reentry_unowned_raw_mismatch_episode_action=hold`, `episode_min_missed=1`, `episode_max_missed=20`, `episode_max_events=8`, `episode_max_cost=0.36`. Remaining `000263` frames `193/195` are track 3/4 fight/occlusion with raw IDs still consistent; user suspects possible GT ambiguity, so do not add broad guard before visual/GT confirmation.
- 2026-07-07 new 5-video candidate `outputs/eval/hybrid_bytetrack/20260707_082640/smooth_det020_loose/iou0_area0_condarea0_merge0`: improves weak `000233` without breaking hard guardrails. Remapped IDSW: `000231=0`, `000233=6`, `000263=2`, `000328=0`, `000302=0`, `ALL=8` versus `Best_tracking` `000233=9`, `ALL=11`. Add-on opt-in guard: `occlusion_reid_prefer_gap_over_bad_match=true`, `occlusion_reid_bad_match_action=reject`, raw mismatch + unowned raw + occlusion_hold only, `min_missed=7`, `max_missed=12`, `min_cost=0.55`, `max_cost=0.70`. This reject action intentionally does not consume the detection; the max-cost upper bound prevents `000231` frame-906 style regression.
- 2026-07-07 suffix repair candidate `outputs/eval/hybrid_bytetrack/codex_suffix_5video_min1500/iou0_area0_condarea0_merge0`: best current 5-video opt-in, remapped IDSW `000231=0`, `000233=6`, `000263=0`, `000328=0`, `000302=0`, `ALL=6`. Adds `suffix_pair_swap_repair=true` on top of protected practical config and 000233 guarded config. Key fix is suffix identity crossing for `000263` `Pig_3/Pig_4` after heavy overlap around frames `193/195`; default `suffix_pair_swap_min_suffix_frames=1500` avoids false suffix swaps seen with broad `60` frame setting on `000231/000233/000328/000302`. Keep opt-in until broader regression passes.
