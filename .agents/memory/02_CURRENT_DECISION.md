# Current Decision

## 2026-07-03 tracking decision

- Treat `outputs/eval/hybrid_bytetrack/20260703_193439/smooth_det020_loose/iou0_area0_condarea0_merge0/tracking_metrics.csv` as the current best 2-video tradeoff for `000231` + `000302`.
- Result:
  - `Pigs291119_000231_30fps`: IDSW `2`, HOTA `0.9705892717094201`, IDF1 `0.9847241970177549`.
  - `Pigs291119_000302_30fps`: IDSW `0`, HOTA `0.9930104703678451`, IDF1 `0.9964355605255801`.
  - `ALL`: IDSW `2`, HOTA `0.9820366705826231`, IDF1 `0.9907038986528682`.
- Keep the current split lost-track reacquire approach:
  - `lost_track_reacquire_guard=true`.
  - `lost_track_reacquire_non_same_raw_distance_guard=false` is the current default/base setting after 9-video run `20260703_194929`.
  - `lost_track_reacquire_raw_owner_guard=true`; do not turn it off globally.
  - Keep `lost_track_different_raw_hidden_owner_bypass=true`, `lost_track_different_raw_hidden_owner_min_missed=2`, and `lost_track_different_raw_hidden_owner_min_center_gain=0.03`.
- Ablation findings:
  - Turning off raw-owner guard globally gives `000302` IDSW `0` but makes `000231` much worse.
  - Turning off only non-same-raw distance guard gives `000231` IDSW `2` but still needs the hidden-owner bypass to recover `000302`.
  - Tightening only appearance threshold did not change the bad `000231=8`, `000302=0` result; owner state / center gain was the useful tightening.
- Default decision: tracking, evaluation, and optimizer should inherit this base from `TrackingConfig`; do not require callers to pass `--profile-override lost_track_reacquire_non_same_raw_distance_guard=false`.

## Current baseline

- Do not use legacy 21/06 as the primary comparison point anymore; when discussing `evaluate_tracking.py` metric drift, compare against commit `b697c4eba36db280cbf01f446873da17bcac509d`.
- Current accepted `hybrid_bytetrack` post-processing flow is the two-gate flow restored from `b697c4eba36db280cbf01f446873da17bcac509d`: identity guard requires `enable_offline_smoothing and identity_swap_guard`; temporal refinement plus `stabilize_overlap_hidden_islands` requires `enable_offline_smoothing and (smooth_boxes or refine_boxes)`.
- This flow is considered IDSW-critical and should be preserved unless an explicit ablation proves a replacement is better.
- Current tracking execution flow is `scripts/track_videos.py` -> `python -m pig_behavior.tracking.cli`.
- `track_videos.py --eval-config <name>` should stay aligned with `evaluate_tracking.py` named presets and forward them as `--profile-override KEY=VALUE`.
- `pig_behavior.tracking.cli` must keep the module entrypoint and `--profile-override` support; otherwise `track_videos.py --eval-config` either exits without running or fails argument parsing.
- `--no-emit-hidden-tracks` is an output-labeling control for CVAT relabeling: keep tracker-maintained/interpolated boxes, but export their `Hidden` attribute as `No`. It must not be treated as disabling internal hidden state, association, motion prediction, occlusion holding, or smoothing.
- Treat Tracking moi bat smooth as the current quality baseline when reading reports.
- Tracking moi tat smooth/yolov8 is still a relevant runtime variant, but its reported metrics are currently worse.
- For optimizer default target-video diagnostics, do not pin `000263`/`000302`.
- Instead derive the weak default target set from the current no-smooth baseline metrics file:
  `outputs/eval/hybrid_bytetrack/Tracking mới tắt smooth/yolov8/iou0_area0_condarea0_merge0/tracking_metrics.csv`
- Do not include detector-only presets (`det_conf`, `max_raw_detections`, `nms_iou` only) in the default optimizer scopes.
- Artifact `outputs/eval/hybrid_bytetrack/overnight_iou0/optimizer/tracking_optimizer_summary.csv` showed detector-only presets matched `base` metrics for both smooth and no-smooth.
- Detector-only checks now belong in explicit `--scope detector_probe` runs or explicit `--preset` runs.

## Investigation focus

- Keep focus on runtime and code-path differences inside hybrid_bytetrack.
- Primary suspects remain association.py raw_id owner/penalty/bypass logic and all_detection_indices matching.
- Secondary suspect remains forced post-processing in runner.py for hybrid_bytetrack.

## Guardrails

- Do not blame detector weight for the 000263 regression.
- Do not enable condarea by default without an explicit ablation.
- Prefer small, reversible patches.
## 2026-07-04 hard-scene improvement plan

User requested the plan be remembered and executed. Preserve current strong
baseline first: `hybrid_bytetrack + smooth_det020_loose +
iou0_area0_condarea0_merge0`, especially keeping `Pigs291119_000302_30fps = 0`
IDSW. Do not promote broad offline repair by default. Episode-level pair swap
repair remained opt-in and did not change the hard 4-video eval because the
remaining failures are not simple visible short-overlap geometry swaps:
`000231` involves Hidden/visible behavior, `000328` involves longer conflict,
and `000263` motion cost favors keeping current geometry.

Execution order:

1. Add opt-in association diagnostics first (`association_debug=true`) to record
   assignment accept/reject events, raw owner, top raw ID, split recovery,
   ambiguity, cost, threshold, and detection metadata. This must not change
   behavior when disabled.
2. Use diagnostics around IDSW frames to classify failures as
   `fight_rotate_bbox`, `long_occlusion_reentry`, `hidden_owner_steal`, or
   `raw_id_bypass_error`.
3. Patch only one narrow opt-in guard at a time in `association.py`:
   `ambiguity_owner_guard`, `hidden_owner_guard`, `raw_owner_quarantine`, then
   `long_occlusion_reentry_guard`.
4. Validate on hard set `000231/000263/000328/000302` first. Promote only if
   total hard-set IDSW drops, `000302` stays 0, and the 9-video baseline does
   not regress.

Implementation started:

- `association_debug=true` adds opt-in assignment diagnostics and remains off by
  default.
- `ambiguity_owner_guard=true` adds the first narrow opt-in guard: if a detection
  raw ID belongs to another candidate owner and that owner cost is close to the
  selected assignment, reject the likely raw-owner steal instead of letting a
  marginal assignment rewrite identity. This is intended for fighting/rotating
  bbox scenes and must be validated on the hard 4-video set before any broader
  promotion.
- User reported run `outputs/eval/hybrid_bytetrack/20260704_090756` had no
  meaningful metric change. Diagnostics under the matching prediction root show
  `assignment_reject_ambiguous_raw_owner = 0` for `iou0_area0_condarea0_merge0`,
  so the first guard did not trigger. Continue with `hidden_owner_guard=true`:
  when a detection raw ID belongs to a hidden/lost owner but is assigned to a
  different track, freeze identity learning for that assignment while still
  allowing bbox update. This remains opt-in and must be tested on the hard set.
- User reported `outputs/eval/hybrid_bytetrack/20260704_100102/.../merge0`
  unchanged. Diagnostics show `hidden_owner_freeze=True` triggered only once
  (`000231` frame 401), while `000263`, `000302`, and `000328` had zero hidden
  owner freezes. Because freezing identity learning did not change the exported
  bbox/label assignment, continue with a stricter opt-in
  `hidden_owner_guard_hold_assignment=true`: when the same hidden-owner conflict
  is detected, hold the assigned track instead of consuming the ambiguous
  detection. This is expected to affect at most the trigger frames and must be
  tested with `association_debug=true` before considering any promotion.
- User reported improvement on
  `outputs/eval/hybrid_bytetrack/20260704_103036/smooth_det020_loose/iou0_area0_condarea0_merge0`.
  Diagnostics show `assignment_hidden_owner_hold` triggered exactly once:
  `Pigs291119_000231_30fps` frame 401. The remapped IDSW events for `000231`
  disappeared; remaining switches are `000263` frames 193/195 and `000328`
  frames 1342/1360. `000302` remains clean in this hard-set run. Keep
  `hidden_owner_guard_hold_assignment` opt-in until 9-video regression is run.
  Next work should target `000263`/`000328` with a separate reentry/quarantine
  guard rather than broadening hidden-owner hold.
## 2026-07-04 reentry ambiguous hold candidate

After user reported improvement on
`outputs/eval/hybrid_bytetrack/20260704_103036/smooth_det020_loose/iou0_area0_condarea0_merge0`,
diagnostics confirmed `assignment_hidden_owner_hold` triggered once on
`Pigs291119_000231_30fps` frame 401 and removed the `000231` remapped IDSW
events. Remaining hard-set switches are `000263` frames 193/195 and `000328`
frames 1342/1360; `000302` remains clean. Keep hidden-owner hold opt-in until
9-video regression passes.

Next candidate added as opt-in only: `reentry_ambiguous_hold=true`. If a track is
OCCLUDED/LOST/MISSING or has enough missed frames and the assignment is already
marked ambiguous, hold the track instead of consuming the detection. Test this
separately from hidden-owner hold on the hard 4-video set.
## 2026-07-04 reentry hold retest result

User reported
`outputs/eval/hybrid_bytetrack/20260704_105654/smooth_det020_loose/iou0_area0_condarea0_merge0`
had real effect from `reentry_ambiguous_hold`. The old `000328` remapped IDSW
events at 1342/1360 disappeared and total remapped switch count dropped versus
the pre-guard baseline. However new remapped switches appeared (`000231` frame
325 and `000263` frames 475/1125), and debug showed reentry holds firing broadly
from early frames. Do not promote this broad version.

Narrowing applied: `reentry_ambiguous_hold` now requires prior stable detections
(`ever_detected` and at least `reentry_ambiguous_hold_min_hits`) and no longer
uses bare `MISSING` state as a trigger. Retest narrowed reentry hold alone before
combining with hidden-owner hold or running 9-video regression.
## 2026-07-04 reentry hold narrowed again

User provided
`outputs/eval/hybrid_bytetrack/20260704_112422/smooth_det020_loose/iou0_area0_condarea0_merge0`.
The narrowed reentry hold still fired far too broadly: thousands of
`assignment_reentry_ambiguous_hold` events per video starting at early frames
(e.g. `000231` from frame 3, `000328` from frame 7). Do not promote this
version. Tightened the helper again so `track.missed >=
reentry_ambiguous_hold_min_missed` is mandatory before OCCLUDED/LOST or
prediction/occlusion reason can trigger a hold. Retest this stricter version
alone; expected trigger count should drop from thousands to localized reentry
spans.
