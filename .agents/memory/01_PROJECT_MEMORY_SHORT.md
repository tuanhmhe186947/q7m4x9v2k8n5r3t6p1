# Project Memory Short

## 2026-07-13 authoritative classification_v2 state

- Use `docs/CLASSIFICATION_V2_CURRENT_STATE.md` as the status authority.
- The active path is the reviewed-data rebuild, not postrun promotion of the
  previous full OOF artifact.
- Hidden v5 has a valid 5,171-item template but no complete human decisions.
- Behavior coverage is 3/4,670 units, with 4,667 missing and one pending.
- Therefore Hidden apply, behavior apply, the reviewed train-ready snapshot,
  model smoke on that snapshot, and a new full OOF are all blocked.
- The old commit-`18d6692` full OOF is historical engineering evidence only.

## 2026-07-13 classification_v2 Hidden review workload

- Current workload-policy implementation commit is `5212a59`.
- Hidden review now census-selects every untrusted CVAT Hidden=Yes, samples
  trusted legacy Yes by recording-date/behavior stratum, and caps initial
  high-risk Hidden=No review at one item per stratum.
- Versioned full evidence is
  `outputs/classification_v2/rebuilds/hidden_review_v5_full_20260713`.
- V5 has 5,171 unique items: 4,649 CVAT and 522 legacy. It has zero missing
  untrusted Yes, trusted-stratum quota mismatches, or high-risk cap violations.
- This only makes the review workload auditable. Human Hidden decisions and
  behavior decisions remain incomplete, so the dataset is not train-ready.

## Historical 2026-07-13 full OOF and workflow migration

- Full multimodal OOF training completed in
  `outputs/classification_v2/model_full/full_multimodal_oof/`.
- Verified full outputs contain 73,668 window predictions and 32,727 native
  temporal predictions; accuracy is `0.5216793473` and supported macro-F1 is
  `0.4156053847`.
- These metrics are engineering evidence from the previous data lineage. They
  cannot become the final Q2 result by postrun processing alone because current
  Hidden and behavior human-review gates are incomplete.
- All classification operator scripts now live only under
  `scripts/classification_v2/00_*` through `09_*`. The former split namespaces
  and compatibility wrappers were removed.
- Workflow migration commits are `d7d22a8` and `1491d78`. The structural audit
  is block `09` script `check_classification_v2_workflow_layout.py`.

## Historical 2026-07-13 pre-full hardening refresh

- The previous lineage recorded its verified HEAD in
  `outputs/classification_v2/model_design/q2_progress_report_audit.json` key
  `current_git_commit`.
- `q2_progress_report_audit.json` is valid with `PASS_PARTIAL_ROADMAP`,
  44/44 gates passing, clean git, `full_oof_execution_allowed=false`,
  `authorization_authorized=false`, and `q2_claim_allowed=false`.
- Full OOF execution gate is now hardened with 4 rejection cases, including a
  near-authorized authorization file that has all boolean approvals true but
  missing `reviewer` and `reviewed_at`.
- Preflight runtime benchmark drift now allows audit/auth-only changes without
  rebenchmarking, while keeping runtime/model/training changes fail-closed.
- This pre-full state was later followed by the historical full run. It does not
  describe the active reviewed-data rebuild and must not authorize another run.

## Architecture contract retained from 2026-07-12

- Active priority is `classification_v2` behavior recognition, not tracking
  ablation, unless the user explicitly switches back to tracking.
- Current target claim remains Q2-strong only: improved pig behavior recognition
  under recording-date/video-safe validation. Do not claim external farm,
  camera, or cohort generalization until external validation exists.
- Pipeline is built around multimodal spatio-temporal inputs:
  bbox/letterboxed actor image sequence, ROI relation features, motion,
  social/partner context, interaction visual context, event-balanced weights,
  native temporal OOF folds, and strict feature whitelist leakage guards.
- `pig_id` is annotation-local and must not be treated as the same animal across
  videos or sessions.
- Canonical actor image cache is letterboxed, not square-stretched:
  `outputs/classification_v2/image_cache_v2_letterbox/`.
- Every future full OOF launch still requires preflight plus explicit
  authorization bound to the active data/cache/config/code hashes.
- The previous 44/44 `PASS_PARTIAL_ROADMAP` report is stale for the current
  rebuild and must not be refreshed until human review and snapshot gates pass.
- Local RTX 3050 limits development batch size, not the research architecture;
  remote/rented GPU execution remains allowed after the same lineage gates.
- Postrun calibration, confusion analysis, ablation, registry, and completion
  checks remain required after a future reviewed-lineage full run.

## 2026-07-08 realtime full runtime chunk validation

- Runtime 13-video validation completed in two chunks: `outputs/eval/realtime/runtime_check_quality_delayed_simple_7video/iou0_area0_condarea0_merge0` plus `outputs/eval/realtime/runtime_check_quality_delayed_simple_remaining6/iou0_area0_condarea0_merge0`.
- Compared with `outputs/eval/realtime/realtime_balanced_13video/iou0_area0_condarea0_merge0`, per-video runtime total `remapped_idsw 75 -> 21`, `fp/fn` stayed `2320/1055`, no per-video IDSW regression, and `000302=0`.
- Remaining runtime IDSW: `000114=2`, `000231=6`, `000233=9`, `000263=2`, `000327=2`; all other 8 videos are `0`.

## 2026-07-08 realtime simple low-gain component pass

- Improved `realtime_quality_delayed` artifact candidate further by adding an opt-in second pass for simple motion components only: `realtime_motion_pair_simple_min_gain=0.005`, `realtime_motion_pair_simple_max_component_size=2`.
- Evidence artifact: `outputs/eval/realtime/probe_motion_pair_simple005_comp2_13video/iou0_area0_condarea0_merge0`.
- Compared with gain-gate candidate `outputs/eval/realtime/probe_motion_pair_gainmin004_edges2_13video/iou0_area0_condarea0_merge0`: `ALL remapped_idsw 27 -> 21`, `remapped_hota_pct 95.89 -> 96.60`, `remapped_idf1_pct 96.41 -> 97.02`, `fp/fn unchanged 2320/1055`.
- Per-video improvements with no IDSW regression on 13-video artifact probe: `000085 2 -> 0`, `000327 4 -> 2`, `000330 2 -> 0`; `000233` stayed `9`, `000302` stayed `0`.
- Runtime smoke check: `outputs/eval/realtime/runtime_check_quality_delayed_simple_233_302/iou0_area0_condarea0_merge0` using actual `realtime_quality_delayed` code path produced `000233 remapped_idsw=9` and `000302 remapped_idsw=0`, matching the artifact expectation for target/guardrail.
- Rejected probes: global `min_allowed_edge_gain=0.02` regressed `000114/000327`; global `max_jump=0.08/0.12` regressed `000233`; `memory_frames=20` regressed `000231`; `memory_frames=40` unchanged; global `min_gain=0.005` regressed `000233/000327`.
- Additional runtime smoke checks: `outputs/eval/realtime/runtime_check_quality_delayed_simple_263/iou0_area0_condarea0_merge0` produced `000263 remapped_idsw=2`; `outputs/eval/realtime/runtime_check_quality_delayed_simple_085/iou0_area0_condarea0_merge0` produced `000085 remapped_idsw=0`. Both match artifact expectations.

## 2026-07-08 realtime dense fallback gain gate

- Improved current `realtime_quality_delayed` motion-pair candidate by tightening dense-component fallback: `realtime_motion_pair_dense_fallback_max_edges=2`, `realtime_motion_pair_dense_fallback_min_median_gain=0.05`, `realtime_motion_pair_dense_fallback_min_edge_gain=0.04` while keeping `max_component_size=4`, `max_component_edges=3`, `max_support_ratio=0.35`.
- Evidence artifact: `outputs/eval/realtime/probe_motion_pair_gainmin004_edges2_13video/iou0_area0_condarea0_merge0`.
- Compared with previous dense candidate `outputs/eval/realtime/probe_motion_pair_comp4_edges3_dense_13video/iou0_area0_condarea0_merge0`: `ALL remapped_idsw 31 -> 27`, `000233 13 -> 9`, no per-video IDSW regression, `000302=0`, `fp/fn unchanged 2320/1055`; HOTA/IDF1 effectively unchanged at `95.89/96.41`.
- Runtime check on `000233` planned graph confirms allowed dense fallback edges become `{ID_1-ID_3, ID_1-ID_8}`, excluding weak low-min-gain `ID_2-ID_8` that caused extra switches.

## 2026-07-08 realtime motion-pair quality-delayed candidate

- Added opt-in `realtime_motion_pair_stabilizer` for `mode=realtime` only. It relabels short-memory motion-consistent ID attributes, then filters proposed relabel graph to small/sparse components. The current 13-video candidate uses `realtime_motion_pair_max_component_size=4`, `realtime_motion_pair_max_component_edges=3`, and dense-component rare-edge fallback (`max_edges=3`, `max_support_ratio=0.35`); this admits sparse four-ID episodes like `000327` and a limited rare-edge subset in dense `000233` while still blocking the dominant long cascade edge.
- Important implementation fix: the planning pass must use `deepcopy`; shallow `shape.copy()` mutates nested `attributes` and accidentally applies broad relabel before component filtering.
- Enabled the stabilizer in `realtime_quality_delayed`, not in `realtime_balanced`. Treat this as a quality-delayed candidate, not the pure causal realtime baseline.
- Validated runtime 5-video result: `outputs/eval/realtime/codex_motion_pair_quality_5video_fix/iou0_area0_condarea0_merge0`. Compared with `outputs/eval/realtime/realtime_balanced_5video/iou0_area0_condarea0_merge0`: `ALL remapped_idsw 43 -> 25`, `remapped_hota_pct 90.12 -> 92.20`, `remapped_idf1_pct 90.18 -> 92.50`, `fp/fn unchanged 849/661`.
- Per-video remapped IDSW in this candidate: `000231=8` (from `12`), `000233=15` (unchanged, no regression), `000263=2` (from `12`), `000328=0` (from `4`), `000302=0` (guardrail preserved).
- 13-video artifact probe with component size `4`, edge cap `3`, and rare-edge fallback: `outputs/eval/realtime/probe_motion_pair_comp4_edges3_dense_13video/iou0_area0_condarea0_merge0`. Compared with `outputs/eval/realtime/realtime_balanced_13video/iou0_area0_condarea0_merge0`: `ALL remapped_idsw 75 -> 31`, `remapped_hota_pct 92.77 -> 95.89`, `remapped_idf1_pct 93.12 -> 96.41`, `fp/fn unchanged 2320/1055`. No per-video remapped IDSW regression in this 13-video set; `000302` stayed `0`. `000327` improved `8 -> 4`; `000233` improved `15 -> 13` but remains the weakest realtime video.
- Do not promote this as realtime causal base without broader regression and explicit discussion that it is delayed/post-tracking stabilization. Next step should validate more videos and then design an online-buffer equivalent if true realtime latency is required.

## 2026-07-08 realtime profile cleanup and failed probes

- `realtime_balanced` was changed to inherit from a realtime-only eval base:
  `enable_offline_smoothing=false`, `identity_swap_guard=false`,
  `smooth_boxes=false`, and `refine_boxes=false`. Single `000233` with these
  offline flags forced off matched the prior metrics, so the current realtime
  failures are from online association/tracking behavior rather than offline
  smoothing.
- Added named realtime eval profiles:
  - `realtime_fast`: speed-oriented probe, `detect_every_n_frames=2`,
    `det_conf=0.25`, `max_raw_detections=32`, no offline smoothing.
  - `realtime_balanced`: current causal probe stack.
  - `realtime_quality_delayed`: finite-window local repair probe only; no
    suffix/long future repair.
- Rejected/neutral probes from this continuation:
  - `overlap_small_box_suppression=true` on realtime `000233`: no metric change
    (`remapped_idsw` stayed `15`).
  - hybrid causal guard stack on realtime `000233`: no improvement; FP slightly
    increased.
  - `tracker_type=botsort` on realtime `000233`: no metric change.
  - looser `realtime_visible_better_competitor_min_cost=0.28`,
    `min_gain=0.025` on `000263`: regressed `remapped_idsw 12 -> 16`, so do
    not promote.
  - `local_pair_swap_repair=true` with a 12-frame window on realtime `000263`:
    no metric change.
- Current conclusion: remaining realtime IDSW is not solved by porting existing
  hybrid causal guards or existing finite-window repair as-is. Next useful
  implementation should be a new online/short-buffer identity stabilizer, not a
  broad reject/hold guard and not offline suffix repair.

## 2026-07-08 realtime balanced profile

- Added `realtime_balanced` to `scripts/evaluate_tracking.py` as the current
  named realtime probe profile. It packages the useful opt-in realtime stack:
  `smooth_det020_loose` recovery settings, `occlusion_aware_matching=false`,
  `realtime_visible_close_competitor_guard=true`,
  `realtime_visible_better_competitor_reject=true`,
  `realtime_visible_better_competitor_prefer=true`, and
  `realtime_low_conf_recovery_guard=true`.
- `realtime_balanced` is still a probe profile, not a finished realtime
  baseline. It preserves the `000302` guardrail in the single-video check:
  `outputs/eval/realtime/realtime_balanced_302_guardrail/iou0_area0_condarea0_merge0`
  produced `remapped_idsw=0`, `remapped_hota_pct=99.38`,
  `remapped_idf1_pct=99.69`.
- 5-video validation with the named profile:
  `outputs/eval/realtime/realtime_balanced_5video/iou0_area0_condarea0_merge0`.
  This matches the prior long override candidate: `ALL remapped_idsw=43`,
  `fn=661`, `fp=849`, `remapped_hota_pct=90.12`,
  `remapped_idf1_pct=90.18`; per-video remapped IDSW remains
  `000231=12`, `000233=15`, `000263=12`, `000328=4`, `000302=0`.
- Rejected new probe: `realtime_reid_shadow_visible_hold`. Broad version on
  `000263` reduced remapped IDSW `12 -> 8` but badly damaged idmap coverage and
  HOTA/IDF1 (`remapped_hota_pct` about `81.70`). Narrowing to
  `max_missed=5` kept `000263` at `12` IDSW and did not improve HOTA. The guard
  was removed from code. Do not re-add a hold/consume duplicate-shadow reid
  guard without a fundamentally better discriminator.

## 2026-07-08 realtime failed guard probes

- Tried a narrow opt-in `realtime_occluded_reid_duplicate_guard` idea for
  `000263` reid switches. Default `min_iou=0.55` did not trigger. Lowering to
  `min_iou=0.45` reduced `000263` remapped IDSW `12 -> 8`, but badly damaged
  IDF1/HOTA/idmap coverage (`remapped_hota_pct` about `81.70` versus `90.62`),
  so this is not a promotion candidate. The underlying evidence is still useful:
  wrong `000263` reid detections around `792/846/865` overlap visible tracks by
  only about `0.46-0.49` IoU, so simple duplicate rejection is too blunt.
- Tried an opt-in visible row-regret reject for `000231` frame `1368`
  (`selected_cost≈0.668`, `track_best≈0.226`). It triggered exactly once but did
  not reduce remapped IDSW and slightly worsened FP/FN, so it was removed.
- Current realtime candidate remains the missed3 low-conf stack:
  `outputs/eval/realtime/probe_realtime_low_conf_recovery_missed3_5video/iou0_area0_condarea0_merge0`.
  Continue from diagnostics rather than re-adding the rejected duplicate or
  row-regret guards.

## 2026-07-08 realtime missed3 candidate

- Realtime baseline at `outputs/eval/realtime/baseline_current/iou0_area0_condarea0_merge0` remains the main realtime comparison point.
- Current useful realtime candidate keeps all new realtime guards opt-in:
  `occlusion_aware_matching=false`,
  `realtime_visible_close_competitor_guard=true`,
  `realtime_visible_better_competitor_reject=true`,
  `realtime_visible_better_competitor_prefer=true`,
  `realtime_low_conf_recovery_guard=true`.
- Tuned `realtime_low_conf_recovery_min_missed` default for the opt-in guard to `3`.
  Single `000233` improved versus the broad low-conf guard: `IDSW=15`,
  `FN=388`, `remapped_hota_pct=84.21` at
  `outputs/eval/realtime/probe_realtime_low_conf_recovery_233_missed3/iou0_area0_condarea0_merge0`.
- 5-video candidate
  `outputs/eval/realtime/probe_realtime_low_conf_recovery_missed3_5video/iou0_area0_condarea0_merge0`:
  `ALL remapped_idsw=43`, `fn=661`, `fp=849`, `remapped_hota_pct=90.12`,
  `remapped_idf1_pct=90.18`. Per-video remapped IDSW:
  `000231=12`, `000233=15`, `000263=12`, `000328=4`, `000302=0`.
  This is not final, but it is cleaner than the earlier broad low-conf guard
  because it preserves most IDSW gain while recovering FN.
- Rejected probe: `realtime_late_reid_guard` for stale occlusion reid on `000263`
  reduced `000263` IDSW `12 -> 10` but badly damaged IDF1/HOTA and idmap coverage
  (`remapped_hota_pct=81.66`), so it was removed from code.
- Diagnostics from `outputs/pred/realtime/probe_realtime_missed3_263_debug/.../association_debug_events.csv`:
  remaining `000263` switches are mostly `reid` from `OCCLUDED/occlusion_hold`
  despite low selected costs, e.g. frames `792` missed `3`, `846` missed `33`,
  `865` missed `5`. A simple late-missed gate is not safe; next direction should
  inspect whether these reid detections are duplicates/extra boxes or need a
  causal short-window identity stabilizer rather than a broad reject.

## 2026-07-08 realtime coverage candidate

- Baseline realtime at
  `outputs/eval/realtime/baseline_current/iou0_area0_condarea0_merge0` has a
  major coverage/FN problem: 13-video `ALL` `fn=72669`, `recall_pct=60.58`,
  `remapped_idsw=115`, `remapped_hota_pct=57.55`.
- Strongest realtime lever so far is `occlusion_aware_matching=false`; on the
  5-video guard set it reduces `fn` from about `32425` to `601`, but exposes
  visible close-competitor swaps.
- Added opt-in `realtime_visible_close_competitor_guard=true` for realtime only
  when `occlusion_aware_matching=false`. It resolves near-tie high-confidence
  visible assignments toward an otherwise unserved competitor track.
- The useful discriminator came from debug:
  - `000302` good trigger: frame `555`, selected `track 8` vs preferred
    `track 4`, costs `0.194358` vs `0.204259`, margin about `0.0099`.
  - `000263` false trigger with wider margin: frame `421`, costs `0.261192`
    vs `0.276337`, margin about `0.0151`.
  - Default `realtime_visible_close_competitor_margin=0.012` keeps the `000302`
    fix while blocking the `000263` false trigger.
- Current 5-video realtime candidate:
  `outputs/eval/realtime/probe_close_competitor_margin012_5video/iou0_area0_condarea0_merge0`.
  Compared with realtime baseline subset, it is a large coverage/HOTA
  improvement but not an IDSW win: `fn=601`, `fp=942`, `remapped_idsw=63`,
  `recall_pct=99.15`, `remapped_hota_pct=88.07`. Per-video remapped IDSW:
  `000231=27`, `000233=20`, `000263=12`, `000328=4`, `000302=0`.
- Do not promote this as the default realtime config yet. Next realtime work
  should reduce the remaining visible-swap IDSW on `000231/000233/000263/000328`
  without reintroducing hidden/occluded coverage loss.

## 2026-07-07 visible-start suffix gate full success

- New best full 12-video candidate:
  `outputs/eval/hybrid_bytetrack/codex_visible_suffix_gate_full/iou0_area0_condarea0_merge0`.
- Versus `outputs/eval/hybrid_bytetrack/Best_tracking/iou0_area0_condarea0_merge0`:
  `ALL` remapped IDSW improved `11 -> 0`; every per-video remapped IDSW is
  `0`.
- Guardrails stayed clean: `000085=0`, `000225=0`, `000231=0`, `000302=0`,
  `000328=0`.
- Targets fixed: `000233=0`, `000263=0`.
- Key code change: `suffix_pair_swap_repair=true` is now narrowed by requiring
  both shapes at `swap_start` to have `Hidden=No`. This blocks the false
  hidden-start suffix swaps previously seen on `000085` frame 17 and `000225`
  frame 264, while still allowing the visible-start `000263` suffix repair
  around frame 193.

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
- 2026-07-07 new best 5-video opt-in candidate `outputs/eval/hybrid_bytetrack/codex_overlap_suppress_5video/iou0_area0_condarea0_merge0`: remapped IDSW `000231=0`, `000233=2`, `000263=0`, `000328=0`, `000302=0`, `ALL=2`. Adds `overlap_small_box_suppression=true` on top of the suffix candidate. This suppresses small low-confidence boxes in high-overlap frames (`min_iou=0.40`, `max_area_ratio=0.65`, `max_score=0.75`) and fixes the `000233` short box-crossing switches without breaking hard guardrails. Keep opt-in pending broader regression.
- 2026-07-07 current best 5-video opt-in candidate `outputs/eval/hybrid_bytetrack/codex_hidden_suffix_id_swap_5video/iou0_area0_condarea0_merge0`: remapped IDSW `000231=0`, `000233=0`, `000263=0`, `000328=0`, `000302=0`, `ALL=0`. Adds `hidden_suffix_id_swap_repair=true` on top of the overlap-suppress candidate. It catches the `000233` `ID_8/ID_1` low-confidence hidden suffix crossing around `1107-1118` using hidden-run length, max overlap, low median hidden score, and long suffix gates. Keep opt-in pending broader regression before base promotion.
- 2026-07-07 broader 12-video regression `outputs/eval/hybrid_bytetrack/20260707_174142/smooth_det020_loose/iou0_area0_condarea0_merge0` proved the full 5-video stack is not a safe common baseline. It improved `000233=0` and `000263=0`, but regressed previously clean videos: `000085: 0 -> 2` and `000225: 0 -> 2` remapped IDSW versus `Best_tracking`.
- 2026-07-07 ablation on `000085/000225/000233/000263`:
  - `ablate_control_assoc_occlusion_4video`: `000085=0`, `000225=0`, `000233=6`, `000263=2`.
  - `ablate_suffix_only_4video`: `000085=2`, `000225=2`, `000233=6`, `000263=0`; therefore current `suffix_pair_swap_repair=true` is unsafe and must not be promoted.
  - `ablate_overlap_only_4video`: `000085=0`, `000225=0`, `000233=2`, `000263=2`; `overlap_small_box_suppression=true` appears safe on this 4-video ablation.
  - `ablate_overlap_hidden_no_suffix_4video`: `000085=0`, `000225=0`, `000233=0`, `000263=2`; current safest common candidate is protected association/occlusion base plus `overlap_small_box_suppression=true` and `hidden_suffix_id_swap_repair=true`, explicitly with `suffix_pair_swap_repair=false`.
- 2026-07-07 full 12-video no-suffix common candidate `outputs/eval/hybrid_bytetrack/no_suffix_common_candidate_full/iou0_area0_condarea0_merge0`: remapped IDSW `ALL 11 -> 2` versus `Best_tracking`, with no per-video IDSW regression. Key per-video: `000085=0`, `000225=0`, `000231=0`, `000233 9 -> 0`, `000263=2`, `000302=0`, `000328=0`. This is now the safest broader candidate; remaining target is `000263=2` and must not be fixed with current `suffix_pair_swap_repair`.
