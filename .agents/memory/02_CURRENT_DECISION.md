# Current Decision

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
## 2026-07-03 - 000216 lost/hidden reacquire status

- User confirmed the new lost/hidden reacquire guard solved the severe ID jump cascade on `Pigs291119_000216_30fps` around frames 486-499 when running:
  `python scripts\track_videos.py -v C:\Users\ironh\Downloads\PIG_Behavior_Project\data\videos\Pigs291119_000216_30fps.mp4 --mode hybrid_bytetrack --eval-config smooth_det020_loose --no-emit-hidden-tracks`
- Current patch scope that fixed it: `src/pig_behavior/tracking/association.py` lost-track reacquire plausibility gate plus `src/pig_behavior/tracking/config.py` thresholds:
  `lost_track_reacquire_guard`, `lost_track_reacquire_max_center_jump`, `lost_track_reacquire_same_raw_max_center_jump`, `lost_track_reacquire_raw_owner_grace`.
- New unresolved issue: same `000216` run has an easier-case regression from frame 1584 onward where IDs `4` and `7` swap when the pigs merely walk close together, without full overlap/occlusion.
- Next investigation should preserve the hard-case fix while avoiding over-aggressive association changes that break easy close-neighbor cases. Focus likely on visible-track close-pair identity stability / swap guard, not on weakening the lost/hidden teleport guard blindly.
- The first follow-up visible close-pair cost guard did not fix the `000216` frame 1584+ ID 4/7 swap and was removed. Current follow-up direction is a narrower visible raw-owner transfer guard: if a detection raw-id belongs to another still-visible fixed track, do not let a different visible track take it unless it is clearly closer by `visible_raw_owner_transfer_min_gain`.
