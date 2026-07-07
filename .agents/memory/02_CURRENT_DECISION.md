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

## 2026-07-05 practical hard-set config

Treat `hidden_owner_guard=true` plus `hidden_owner_guard_hold_assignment=true`
as the current practical hard-set improvement path. It preserved the clean
`000302` baseline and solved the known `000231` frame-401 hidden-owner failure
in the later 3-video/4-video checks. Keep it opt-in until broader regression
passes, but use it as the base when developing the next `000328` fix.

Do not continue tuning `reentry_ambiguous_hold` thresholds as the main path.
Runs through `20260705_152555` showed that hold-based reentry gates either fired
too broadly and damaged `000231`/`000302`, or became too narrow and missed the
`000328` switch. The `reentry_unowned_raw_mismatch_reject`/quarantine branch
also failed to recover `000328=0` without collateral effects: when broad enough
to affect `000328`, it damaged `000302`; when seed-gated, it no longer changed
`000328`. Treat those as diagnostic opt-ins, not promotion candidates.

Next direction: build a separate episode-level detector for `000328` style
failure. It should look for repeated unowned raw-ID mismatch conflicts over a
short window before taking action, rather than acting on each assignment
independently. Preserve `hidden_owner_guard_hold_assignment` as the `000231`
protection while testing this new branch.

## 2026-07-05 practical hard-set clarification

Use `hidden_owner_guard=true` plus `hidden_owner_guard_hold_assignment=true` as the current practical opt-in base for hard-set work. It fixed the known `000231` frame-401 hidden-owner failure and preserved `000302=0` in later checks.

Do not keep tuning `reentry_ambiguous_hold` or simple `reentry_unowned_raw_mismatch_reject`/quarantine thresholds as the main path. Those branches either damaged `000231`/`000302` when broad enough, or missed `000328` when narrowed.

The next branch is episode-level: detect repeated unowned raw-ID mismatch conflicts over a short frame window before rejecting. This is intended for the `000328` 1340-range failure while keeping hidden-owner hold as the `000231` protection.

## 2026-07-05 successful hard-set candidate

User reported and diagnostics confirmed `outputs/eval/hybrid_bytetrack/20260705_220622/smooth_det020_loose/iou0_area0_condarea0_merge0` is the current successful hard-set candidate.

Metrics: `000231=0`, `000263=2`, `000328=0`, `000302=0`, `ALL=2` remapped IDSW.

Candidate config for full-video validation before base promotion:

- `hidden_owner_guard=true`
- `hidden_owner_guard_hold_assignment=true`
- `reentry_unowned_raw_mismatch_episode_reject=true`
- `reentry_unowned_raw_mismatch_episode_action=hold`
- `reentry_unowned_raw_mismatch_episode_max_events=8`
- `reentry_unowned_raw_mismatch_episode_min_missed=1`
- `reentry_unowned_raw_mismatch_episode_max_missed=20`
- `reentry_unowned_raw_mismatch_episode_max_cost=0.36`
- `association_debug=true` for diagnostics only, not promotion behavior.

Observed guard effects: `000231` used `assignment_hidden_owner_hold` at frame `401`; `000328` used `assignment_hold_reentry_unowned_raw_mismatch_episode` at frame `1342`; `000302` had no guard trigger and stayed IDSW `0`.

Remaining `000263` switches are frames `193` and `195`, track `3/4` during fight/occlusion. Raw IDs are still consistent (`track 3 -> raw 6`, `track 4 -> raw 7`), so this is not the raw-ID mismatch failure class. User noted this may be GT ambiguity because visually the two pigs exchange IDs while fighting. Do not add a broad runtime guard for this before visual/GT confirmation.

## 2026-07-06 next weak-video tracking plan

Keep the current successful hard-set candidate as the protected base. Future
work is experimental until it proves no regression on the guardrail videos,
especially `Pigs291119_000302_30fps = 0` IDSW. The two remaining weak videos
should not be treated as one failure class.

For `Pigs291119_000263_30fps`, the remaining switches are around frames
`193/195` during close fight/occlusion between tracks `3/4`. Diagnostics showed
raw IDs remain consistent (`track 3 -> raw 6`, `track 4 -> raw 7`), so this is
not a raw-ID mismatch or hidden-owner steal. The next candidate should be a
very narrow visible-assignment guard, such as `visible + ambiguous + same_raw +
selected_cost high`, with a hold/freeze action over a short span. Do not use a
broad raw mismatch/reentry rule for this case.

Important clarification from the earlier read-only audit of
`notebooks/01_data_preparation/update_ids_for_annotation.ipynb`,
`DAT_Update_ID_For_Annotate.ipynb`, and early/stable tracker commits: the useful
lesson for `000263` is not raw ByteTrack ID ownership. The old annotation/update
flow stabilized identity with short-window local motion, roughly a 6-frame
window, and preferred a gap/prediction over accepting a bad high-cost match. The
notebook used a tighter matching threshold (`COST_THR = 0.60`) than the current
runtime reid/lost path (`lost_track_cost_threshold = 0.95`).

The key `000263` diagnostic sequence to preserve:

- frame `193`: track `3` misses assignment; track `4` accepts raw `7` with cost
  about `0.437596`.
- frame `194`: track `3` accepts raw `6` with cost about `0.743141`, which is
  high enough that the notebook-style logic would likely hold/predict instead
  of accepting.
- frame `195`: track `3` accepts raw `6` with cost about `0.177293`; track `4`
  accepts raw `7` with cost about `0.489884`.

Therefore the safest `000263` experiment is an opt-in
`occlusion_reid_prefer_gap_over_bad_match` style guard for fight/occlusion
geometry: `phase=reid`, track state `OCCLUDED/LOST`, `ambiguous=true`,
`same_raw_id=true`, short missed span, and `selected_cost > 0.60` or `0.65`.
The action should hold/predict/gap-fill instead of accepting the high-cost
detection. This should be tested separately from the `000233` different-raw
long-occlusion guard and validated carefully because a broad reid threshold
tightening can increase FN/fragments.

For `Pigs291119_000233_30fps`, the failures include short high-cost same-raw
confusions around `923/924` and `939/941`, plus longer mismatches after
occlusion around `1111-1242` and `1424+`. This looks like long-occlusion reid
accepting a bad high-cost target after `occlusion_hold`, often with different
or unowned raw IDs. The next candidate should target `phase=reid`,
`track_source=occlusion_hold`, enough `missed` frames, high selected cost, and
different/unowned raw ID, with an initial hold action rather than a broad reject.
Do not globally set broad `same_raw_only=false`; previous probes suggested it
would fire too often in other videos.

Validation order: test the `000263` and `000233` guards separately, then combine
only if each improves its target. The promotion gate remains the 5-video hard set
`000231/000233/000263/000328/000302`: `000231=0`, `000328=0`, `000302=0`,
`000263` does not regress and preferably improves, `000233` improves clearly,
and total remapped IDSW does not increase on the broader set. Frame/window gates
are acceptable for diagnosis only; promoted logic must be based on runtime
state, not hardcoded video IDs or frame numbers.

## 2026-07-07 000233 guarded improvement candidate

New best opt-in 5-video hard-set candidate:
`outputs/eval/hybrid_bytetrack/20260707_082640/smooth_det020_loose/iou0_area0_condarea0_merge0`.

Metrics versus `outputs/eval/hybrid_bytetrack/Best_tracking/iou0_area0_condarea0_merge0`:

- `Pigs291119_000231_30fps`: stayed `0` remapped IDSW.
- `Pigs291119_000233_30fps`: improved from `9` to `6` remapped IDSW.
- `Pigs291119_000263_30fps`: stayed `2` remapped IDSW.
- `Pigs301119_000328_30fps`: stayed `0` remapped IDSW.
- `Pigs291119_000302_30fps`: stayed `0` remapped IDSW.
- `ALL`: improved from `11` to `8` remapped IDSW on this 5-video set.

Winning add-on config on top of the protected practical base:

- `occlusion_reid_prefer_gap_over_bad_match=true`
- `occlusion_reid_bad_match_action=reject`
- `occlusion_reid_bad_match_same_raw_only=false`
- `occlusion_reid_bad_match_raw_mismatch_only=true`
- `occlusion_reid_bad_match_unowned_raw_only=true`
- `occlusion_reid_bad_match_occlusion_hold_only=true`
- `occlusion_reid_bad_match_min_missed=7`
- `occlusion_reid_bad_match_max_missed=12`
- `occlusion_reid_bad_match_min_cost=0.55`
- `occlusion_reid_bad_match_max_cost=0.70`

Diagnosis: for `000233`, the useful rejections are bad-but-plausible unowned
raw mismatch reid assignments around the long occlusion region, especially raw
`26` near frames `1114-1118`. A broader reject/hold version damaged metrics or
regressed `000231`. The max-cost upper bound is important: without it, a single
very high-cost reject around `000231` frame `906` caused new switches at
`909/912`. Keep this candidate opt-in until broader full-set regression passes.

Next remaining target is `000263=2`. Do not use the `000233` raw-mismatch guard
for `000263`; the `000263` failure remains same-raw fight/occlusion geometry
around frames `193/195`.

## 2026-07-07 suffix repair 000263 candidate

New best current 5-video opt-in candidate:
`outputs/eval/hybrid_bytetrack/codex_suffix_5video_min1500/iou0_area0_condarea0_merge0`.

Metrics:

- `Pigs291119_000231_30fps`: stayed `0` remapped IDSW.
- `Pigs291119_000233_30fps`: stayed `6` remapped IDSW versus the 000233 guarded candidate.
- `Pigs291119_000263_30fps`: improved from `2` to `0` remapped IDSW.
- `Pigs301119_000328_30fps`: stayed `0` remapped IDSW.
- `Pigs291119_000302_30fps`: stayed `0` remapped IDSW.
- `ALL`: improved from `8` to `6` remapped IDSW versus `20260707_082640`.

Winning add-on is `suffix_pair_swap_repair=true` on top of the protected
practical config and the 000233 guarded config. Keep it opt-in until broader
regression passes.

Diagnosis: `000263` is a suffix identity crossing after heavy overlap/fight,
not a raw-ID mismatch. The useful repair swaps the `Pig_3`/`Pig_4` suffix after
the uncertain overlap around frames `193/195`. The first broad suffix repair with
`suffix_pair_swap_min_suffix_frames=60` fixed `000263` but produced false suffix
swaps on guardrail videos (`000231`, `000233`, `000328`, `000302`). The current
default `suffix_pair_swap_min_suffix_frames=1500` is intentionally conservative
and removed those false swaps in the 5-video run.

Next validation step: run a broader regression/full set with this exact opt-in
candidate before any base promotion. The remaining weak target is `000233=6`;
do not weaken the suffix gate just to chase `000233`, because the broad version
already proved unsafe.
