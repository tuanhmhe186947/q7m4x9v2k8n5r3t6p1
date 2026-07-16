# Current Decision

## Decision precedence

Only the active decision immediately below controls current work. All later
sections are historical records. Current gate status is centralized in
`docs/CLASSIFICATION_V2_CURRENT_STATE.md`.

## Active decision: reviewed-data rebuild

### Clean human-review authority

The user confirms zero completed human decisions. Treat all existing 30-row
Hidden and 3-row behavior CSV payloads as unverified forensic/pilot artifacts,
not review evidence. Do not carry them into the next lineage.

Create the new operator-owned lineage only under
`human_review_workspace/classification_v2/<RUN_ID>`. Agent writes belong
under `outputs/classification_v2/agent_audits/<AUDIT_RUN_ID>`; no agent may open
a GUI or write the active human root. The operator owns apply/rebuild there;
after handoff, agent checks remain read-only on that root and write evidence to
the agent audit root.

### Legacy L0-L8 completion and parent handback

The scoped `legacy_16f` lane is complete at code commit `91a6c2a`. The locked
candidate is `legacy_16f_t6_sliding_event_balanced_v1`, with native macro-F1
`0.5343181014`, accuracy `0.6857142857`, and NLL `1.1206917661` on the fixed
245-unit, 33-video development validation set.

The L8 candidate-lock SHA256 is
`b91949711e15c493a07375c4f7fa5f44535220dfdbac68f095d2effee4be6ba6`.
The L0-L8 handback is
`outputs/classification_v2/legacy_only_unreviewed_development/`
`legacy_16f_goal_completion_audit.json`, with SHA256
`4b6bad32834fbede2001dee5627e5fbfa0005afb758f2c6a3cbfb125be3166f6`.
Every milestone is PASS, while human review, reviewed/final naming, canonical
full OOF, outer-holdout prediction, and Q2 claims remain unauthorized.

The next action is to resume the parent classification_v2 goal and re-audit
the canonical reviewed all-source P0-P8 blockers. Do not inherit a parent PASS
from this bounded legacy handback.

### Legacy L7 imbalance decision and L8 handoff

The three-policy `legacy_16f` L7 short matrix is complete with two fresh,
non-overlapping CUDA processes per policy. All repeat hashes are exact. Every
run used 30 optimizer steps, had no OOM or retry, peaked at `73,400,320`
reserved bytes on the local 4 GiB GPU, and cleaned allocated and reserved CUDA
memory to zero.

Event-balanced CE, effective-number CE, and Balanced Softmax native macro-F1
are `0.2717708642`, `0.1072693320`, and `0.1429984901`. Relative to
event-balanced CE, effective-number CE changes macro-F1 by `-0.1645015322`
with 33-video interval `[-0.1895271242, -0.0767137732]`; Balanced Softmax
changes it by `-0.1287723740` with interval
`[-0.1565980560, -0.0416499788]`. Their NLL values worsen from
`1.9439967908` to `3.3642298748` and `3.6577075450`; rare-group macro-F1
falls from `0.2382505739` to `0.0411892030` and `0.0713385243`. Balanced
Softmax also predicts `fight` for `70.2041%` of validation units and fails the
majority-collapse guard.

The valid decision is `RETAIN_EVENT_BALANCED_CE_REJECT_L7_ALTERNATIVES`.
Do not run a full confirmation for either rejected alternative. Start L8 from
the retained actor-only T6 event-balanced base. The decision artifact is
`l7_imbalance_decision_v1.json` with SHA256
`69c200e2b6d570b181423df30cc33cdbecb6686f175f7184b40067bd62ff1482`.
This remains bounded three-epoch evidence, not a full-convergence claim, and
does not transfer to merged-reviewed data, authorize reviewed/final naming,
canonical full OOF, or Q2 evidence.

### Legacy L6 full-frame decision and L7 handoff

The `legacy_16f` full-frame short matrix and paired evaluator are complete.
Zero, availability-only, and full-frame macro-F1 are `0.2697662759`,
`0.2721987509`, and `0.2942624204`. Full-frame minus zero is
`+0.0244961445`, with 33-video cluster interval
`[-0.0668714797, 0.0725200014]`; NLL worsens by `+0.2414525889`.
Full-frame minus availability-only is `+0.0220636696`, with interval
`[-0.0809709233, 0.0671747502]`; NLL worsens by `+0.3144303865`.

The valid decision is
`DO_NOT_EXPAND_FULL_FRAME_CONTEXT_FROM_CURRENT_SHORT_EVIDENCE`. The macro-F1
point estimates meet the margin, but both cluster intervals cross zero and
NLL worsens. Do not run a full confirmation or carry full-frame values into
the legacy candidate. The decision artifact SHA256 is
`e006dc6636ede5a35e71414448be1dc96f0f71e29f5f2a1b6d0230fa0c49c6bf`.

L6 is PASS with the parameter-matched T6 zero retained as the simplest bounded
base. Start L7 and compare event-balanced CE, effective-number CE, and Balanced
Softmax separately. These decisions apply only to unreviewed `legacy_16f` and
do not authorize reviewed/final naming, canonical full OOF, Q2 claims, or an
architecture conclusion for merged-reviewed data.

### Legacy L6 ROI relation decision

The `legacy_16f` ROI short matrix passed its predeclared paired promotion gate,
so one exact full confirmation was authorized. That confirmation is complete
at `l6r_full_decision_v1.json` under
`outputs/classification_v2/legacy_only_unreviewed_development/l6r_full_v1/`
with SHA256
`5a9a2b4b61b7ddeef0b5155ec69b678d73f0acd53917db98d1d6271cab5f1af3`.

Full zero, availability-only, and ROI macro-F1 are `0.4966025667`,
`0.4727197983`, and `0.5082292933`. ROI minus zero is `+0.0116267266`, with
33-video cluster interval `[-0.0398806556, 0.0906766805]`; ROI minus
availability-only is `+0.0355094951`, with interval
`[-0.0248897889, 0.0986581204]`. Availability-only minus zero is
`-0.0238827684`, with interval `[-0.0629523019, 0.0339059054]`.

Do not summarize this as uniformly negative ROI evidence. Relative to zero,
feeding-group macro-F1 improves by `+0.1796877378`; `drink` F1 improves from
`0.3703703704` to `0.6486486486`, and `eat` F1 improves from `0.7906976744`
to `0.8717948718`. `playwithtoy` has only one validation unit, so its ROI
effect is not estimable: recall is `1.0` in both modes, while false positives
rise from one to four and F1 falls from `0.6666666667` to `0.3333333333`.

The decision is `DO_NOT_EXPAND_ROI_RELATION_FROM_CURRENT_SHORT_EVIDENCE`.
The full ROI gain misses the required margin and positive interval-low gate,
and the availability diagnostic fails its bounded-difference check. Do not
carry ROI values into the next candidate. Continue L6 numeric social relations
from the parameter-matched T6 zero control. This does not authorize canonical
full OOF, reviewed/final naming, Q2 evidence, or any claim about merged-reviewed
data. Reassess ROI on merged-reviewed data, where rare-class support is
materially larger. L6 remains `IN_PROGRESS`.

### Legacy L6 numeric-social short decision

The ten-feature numeric-social cache and independent repeat gate pass for all
15,588 T6 sliding windows. The tensor shape is `[15588, 6, 10]`; 92,664 of
93,528 slots are available, with zero media reads and zero outer-holdout slots.
All four cache artifacts are byte-identical across independent roots. Partner
IDs remain audit metadata; top-K partner, geometry, motion, and ROI values are
excluded from model X.

The six-process short matrix is deterministic and crash-bounded. Zero,
availability-only, and numeric-social macro-F1 are `0.2620738697`,
`0.2621547321`, and `0.2624282011`. Numeric-social minus zero is
`+0.0003543314`, with 33-video cluster interval
`[-0.0342531654, 0.0398565230]`; accuracy falls by `0.0326530612` and NLL
worsens by `0.2248711988`.

The decision is `DO_NOT_EXPAND_SOCIAL_RELATION_FROM_CURRENT_SHORT_EVIDENCE`.
Do not run full numeric-social confirmation or carry its values forward.
Core-roadmap S2 requires S1 numeric-social PASS before top-K, so top-K is
`DEFERRED_NOT_AUTHORIZED`. Continue L6 actor-partner union-crop work from the
parameter-matched T6 zero because the interaction-context gap remains. This
does not transfer to merged-reviewed data or authorize reviewed/final naming,
canonical full OOF, or a Q2 claim.

### Legacy L6 motion decision

The `legacy_16f` motion short matrix is closed as valid negative evidence.
Motion does not meet the predeclared native-unit/video-cluster promotion gates,
so do not run full motion or add motion values to the next L6 candidate.
Continue from parameter-matched T6 with numeric social relations; do not carry
ROI or motion values into the next candidate. Geometry, motion, and ROI must be
reassessed on frozen merged-reviewed data; none of these legacy-only decisions
transfers to that lineage. The local 4 GiB GPU was only a correctness host and
was not the rejection reason.

### Canonical engine and review-policy boundary

There is one `classification_v2` data engine, not separate legacy and mixed
implementations. The legacy 16-frame lane differs only by source-selection,
review policy, temporal-view config, output namespace, lineage, and claim
flags. It must call the same canonical feature and harmonization modules.

The two current profiles are:

```text
legacy-only-unreviewed-development:
  sources = legacy_recovered
  review_policy = explicitly_waived_for_development
  temporal_views = T6/T8/T12/T16 within each native 16-frame burst

mixed-reviewed:
  sources = legacy_recovered + cvat_tracking_xml
  review_policy = required_by_current_Q2_protocol
  primary_temporal_view = fixed6_observed_time
```

Human review is configurable at the engine level. It is not an absolute
technical requirement for exploratory training when the user explicitly
accepts an unreviewed lineage. It is nevertheless a hard scientific gate for
the active `mixed-reviewed` snapshot, final/reviewed naming, and Q2 evidence.
Skipping it requires a new explicit unreviewed profile; it must never silently
convert incomplete decisions into authorization.

The separate legacy goal exists to isolate progress, artifacts, metrics, and
claims while the parent mixed-source goal is blocked. It is not permission to
duplicate context, ROI, motion, social, posture, or harmonization code. After
L0-L8 complete in the new chat, return its hash-bound handback to this original
chat and resume the parent P0-P8 goal.

The user grants standing authorization for a full data or model run when it is
necessary for the current milestone. Do not ask again solely because the run is
full or long. Every new lineage must first pass static/synthetic checks, the
exact short representative configuration, and schema/count/hash/output/runtime
audits. Stop before full on any failure. A semantic config change invalidates
the short evidence and requires the short gate again. This permission does not
bypass leakage, immutable lineage, full-OOF launch, or scientific claim gates.

For bounded model tests that report `accuracy` or `F1`, use an explicitly
declared `legacy_recovered` 16-frame development lineage when it is
scientifically compatible with the tested question. The user permits this
lineage to proceed without current human review because it is closest to the
older model lineage and currently less dirty than CVAT. It must be grouped by
recording/video, native-unit safe, hash-frozen, and labeled
`legacy-only-unreviewed-development` in every manifest, run, prediction, and
metric. It cannot replace the all-source reviewed evaluation, be called final
train-ready data, authorize full OOF, or support a Q2 claim by itself.

The reviewed all-source lineage remains independently blocked by incomplete
Hidden and behavior decisions. Its gates must not be weakened or reused to
misrepresent the legacy development branch as human reviewed.

Use `plans/classification_v2-legacy-16f-development-goal-prompt.md` to create a
separate goal for this branch and its dedicated L0-L8 ledger. On scoped goal
completion, return to the original Q2 chat with the immutable handback audit;
do not mark the parent P0-P8 objective complete automatically.

Temporal-length experiments must compare `T6`, `T8`, `T12`, and `T16` windows
generated only after harmonization and contained within one legacy burst. Use
the same native-burst folds and aggregate predictions back to the 16-frame
unit. Report an event-mass-balanced sliding-window view and a one-window-per-
burst matched view so sequence length is not confounded with sample count.

Commit `21b34fd` is the model-input authority for this ladder. It exposes eight
view/selection/slot contracts, binds exact observed-time tensors to each `T`,
and rejects mismatched config, image/context length, spatial padding, or timing.
Its 438-test regression is code evidence only. The short real-data packet,
cache alignment, and no-row-loss audit remain the next gates before any full
legacy rebuild or model execution.

Commit `9b04209` is now the source/missingness probe authority. A source probe
must use the exact ordered trainer whitelist, bind the train-ready ordered
window SHA256, collapse repeated windows to native units, fit only grouped
training roles, and test every eligible native unit once. Availability-only
behavior diagnostics may use only registered label-independent masks;
`interaction_context_ready` is forbidden because its current construction is
label-gated. This engineering PASS does not authorize active-data training.

Commit `abae856` freezes the model-selection grain for all new trainer runs.
Select checkpoints only from grouped inner-validation predictions collapsed by
`temporal_unit_key`, maximizing supported-class macro-F1 and using native-unit
NLL only as a tie-breaker. Remote pilots/full OOF require all 10 classes in the
inner-validation role; local synthetic or bounded smoke requires at least two.
Outer-test predictions remain evaluation-only. New window/native prediction,
aggregation, checkpoint, and registry artifacts must retain this policy and its
hash lineage. This engineering PASS does not authorize active-data training.

Commit `bb225ff` completes the temporal-view and structural shortcut contract
in code and on synthetic fixtures. The primary view now reuses harmonized
six-frame windows for both sources; legacy 16-frame quantile sampling is not
allowed. This is `PASS IN CODE`, not active-data evidence. Do not build the
reviewed temporal packet or run a model until both human review layers pass.

Commits `97f83c5`, `73b901d`, and `16cdb93` complete fold-local preprocessing,
native-event weighting, and run-lineage/registry contracts on fixtures. Run
artifacts now live under `output_root/fold_id/run_id`; downstream callers must
consume the returned lineage path. This remains engineering readiness only and
does not authorize model smoke on an unfrozen reviewed snapshot.

Commit `318bf58` completes the configurable model-factory contract in code.
Ten model modes and four temporal encoders pass mask, shape, missing-modality,
gradient, checkpoint, and lineage tests without downloading weights.

Commit `07ed768` extends that factory with audited ResNet18 and ResNet34 frame
encoders. The controlled interface separates ResNet18 160-to-224 resolution
from ResNet18-to-ResNet34 capacity, records exact ImageNet enum/normalization,
and passes random-init forwards without downloading weights or training. This
does not authorize a pretrained pilot before the reviewed snapshot is frozen.

Commit `2bd2fda` completes the independent visual fine-tuning schedule in code.
Actor and union-context ResNets share frozen, `layer4_only`, and full stages;
the backbone uses a lower LR while all optimizer parameters remain present for
stage-boundary resume. Checkpoint v5, run identity v2, run manifest v2, and
registry v4 bind this contract. The V0/V1/V2 checker is structural-only with
zero optimizer steps, zero project-data rows, and no weight download. It does
not authorize an active-data pilot or satisfy P1 performance PASS.

Commit `3be22f8` completes the independent synthetic visual correctness gate.
ResNet18-160 reaches ten-class tiny-event accuracy 1.0 with finite backbone and
head gradients, deterministic repeated evidence, and exact in-memory resume
parity. The audit remains `synthetic_only` and explicitly sets snapshot/full-OOF
authorization false; it cannot replace active reviewed-data smoke gates.

Commit `111f152` now loads real ordered fixed-six `time_delta` tensors into the
strict data module and binds the slot-manifest hash in checkpoint schema v4 and
registry v3. Corrupt order, slot identity, masks, or timing fail closed, and
unselected windows are retained as explicit masked rows. This remains fixture
evidence only: the reviewed snapshot is blocked, so ResNet training, pilot
training, and full OOF remain unauthorized.

Commit `1b6ba3d` completes native-unit collapse and paired evaluation in code.
Strict ten-class probabilities are reconciled against the complete fold
authority, pooled metrics use the fixed global class order, and paired
recording-cluster uncertainty requires identical units, targets, clusters, and
folds. Its synthetic checker does not replace the missing
human-reviewed snapshot or authorize training.

Commit `e5d6417` completes historical-baseline reconciliation as an engineering
control. Its audit reproduces 151,440/160,740 positional mismatches and marks
the old full OOF `HISTORICAL_ONLY`. It safely records the legacy ResNet34
sequence checkpoint as `HISTORICAL_ARCHITECTURE_ONLY`, not as a performance
baseline. The current regression is 385 passed and 181 deselected. Neither
historical artifact authorizes model selection, paired comparison, training,
full OOF, or a Q2 claim.

The identifier-v2 code/data chain passes at commit `a83d5a5`. Its bounded root
has 688 frame rows, 63 native/review units, 438 ordered windows, exact model-X
whitelisting, zero trainable missing spatial slots, and 8/8 deterministic stage
reruns. All reviewed-data, training, full-OOF, and Q2 authorizations remain false.

Snapshot/preflight contracts are hardened by `7cb4637` and `dd0e6ff`.
Future full-run evidence must bind exact ordered split, image, interaction,
spatial, snapshot, lineage-audit, config, and code hashes. Old v1 snapshots,
preflights, and authorization files are readable historical artifacts but
cannot authorize execution. This code gate does not replace human review.

Do not use the historical full OOF metrics to judge model quality. Commit
`bfdf913` proved that its split/target rows were positionally misaligned with
151,440 of 160,740 image and interaction windows. That run remains useful only
for compute, checkpoint, and pipeline-debug evidence.

The current `reviewed_frame_features.csv` is not human-review complete. Three
old behavior payload rows exist, but verified behavior coverage is 0/4,670.
Do not call this artifact clean final training data. Rebuild instructions are in
`docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`.

Keep the target-independent v6 Hidden manifest at
`outputs/classification_v2/rebuilds/hidden_review_v6_full_20260714` for the
technical reference only. Its 30 carried payload rows are unverified; clean
human coverage starts at 0/5,131 in a new
`human_review_workspace/classification_v2/<RUN_ID>` root.
Hash-bound media validation of the old reference resolves all items, but the
scientific gate remains BLOCKED until the clean authority is reviewed.

Do not resume full training from this decision alone. First create a versioned
reviewed-data lineage, pass complete-decision and leakage-safe fold gates, then
run model smoke gates on the frozen data/cache hashes.

## Historical 2026-07-13 post-full decision

For the previous artifact lineage, full OOF training completed and postrun
validation was the next gate. This no longer overrides the active rebuild.

- Do not rerun full training for the script migration; the completed artifacts
  remain the input to block `07` postrun evaluation.
- Run cross-fit calibration, confusion-focus comparison, ablation refresh,
  experiment registration, and block `09` completion gate in that order.
- Do not claim Q2 improvement until the completion gate reports
  `q2_claim_allowed=true`.
- Use only `scripts/classification_v2/00_*` through `09_*`; there are no wrapper
  commands under the former script namespaces.
- The claim boundary remains internal recording-date/video-safe improvement.
  No external farm, camera, cohort, or biological-identity generalization.

## Historical 2026-07-13 pre-full decision refresh

At that point, the previous lineage was pre-full ready, not Q2 complete.

- Current verified HEAD is the `current_git_commit` in
  `outputs/classification_v2/model_design/q2_progress_report_audit.json` after
  the latest pre-full refresh. Do not hard-code a commit here because memory
  commits intentionally move HEAD.
- Current progress is `PASS_PARTIAL_ROADMAP` with 44/44 gates passing.
- The execution gate now requires 4 rejection cases, including rejection of a
  near-authorized file missing `reviewer` and `reviewed_at`.
- Runtime preflight may allow audit/auth-only commit drift without rebenchmark,
  but must still fail closed for runtime/model/training-relevant changes.
- Do not run or claim full OOF until human authorization is explicitly valid and
  the execution gate allows it.

## Historical 2026-07-12 classification_v2 decision

The active project priority is `classification_v2` behavior recognition unless
the user explicitly switches back to tracking.

Decision recorded at that time:

- Treat the multimodal Q2 roadmap as pre-full ready, not complete.
- The accepted claim boundary is Q2 internal
  recording-date/video-safe improvement. Do not claim external farm, camera,
  cohort, or broad real-world generalization without external validation.
- The model direction is multimodal spatio-temporal:
  letterboxed actor bbox image sequence, ROI relation tensors, motion features,
  social/partner context, and interaction visual context.
- `pig_id` is annotation-local. Never use it as identity continuity across
  videos or sessions.
- Canonical actor visual cache:
  `outputs/classification_v2/image_cache_v2_letterbox/`.
- Historical full OOF output dir for that lineage:
  `outputs/classification_v2/model_full/full_multimodal_oof/`.
- The progress report then was `PASS_PARTIAL_ROADMAP` with 44/44 pre-full
  gates passing. It meant ready for authorization review, not ready to claim
  final Q2 results.
- Full OOF was fail-closed until
  `outputs/classification_v2/model_design/full_oof_authorization.json` was
  explicitly authorized with reviewer, long-run acknowledgement,
  no-Q2-claim acknowledgement, matching preflight config SHA256, and matching
  git commit.
- After full OOF finishes, run postrun calibration, confusion-focus comparison,
  ablation report refresh, experiment registry write, and completion gate before
  any Q2 claim.

Historical tracking decisions below are preserved for tracking work, but they
must not override the current `classification_v2` priority.

## 2026-07-07 current best full tracking candidate

Treat `outputs/eval/hybrid_bytetrack/codex_visible_suffix_gate_full/iou0_area0_condarea0_merge0`
as the current best validated full 12-video candidate.

Compared with `outputs/eval/hybrid_bytetrack/Best_tracking/iou0_area0_condarea0_merge0`:

- `ALL` remapped IDSW improved `11 -> 0`.
- Every per-video remapped IDSW is `0`.
- Clean guardrails remained clean: `000085=0`, `000225=0`, `000231=0`,
  `000302=0`, `000328=0`.
- Remaining targets are fixed: `000233=0`, `000263=0`.

The key correction after the failed `20260707_174142` full stack is that
`suffix_pair_swap_repair=true` now requires both shapes at the swap start frame
to have `Hidden=No`. This keeps the desired visible-start `000263` repair while
blocking the hidden-start false suffix swaps on `000085` and `000225`.

Current candidate stack:

- protected association/occlusion practical base.
- `occlusion_reid_prefer_gap_over_bad_match=true` with the proven unowned
  raw-mismatch occlusion-hold bounds.
- `overlap_small_box_suppression=true`.
- `hidden_suffix_id_swap_repair=true`.
- `suffix_pair_swap_repair=true`, but only with the visible-start gate in
  `repair_suffix_pair_swaps`.

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

## 2026-07-07 000233 failed repair probes

Keep `outputs/eval/hybrid_bytetrack/codex_suffix_5video_min1500/iou0_area0_condarea0_merge0`
as the protected current best candidate. Do not promote the later 000233 probes:

- `20260707_122454`: enabling existing local/episode/long pair swap repairs on
  top of the best candidate did not change `000233`; remapped IDSW stayed `6`.
- `20260707_123316`: aggressively loosening local/episode/long repair thresholds
  also did not change `000233`; remapped IDSW stayed `6`.
- A new experimental hidden-overlap suffix repair was implemented and verified
  locally, but the single-video run `20260707_145820` worsened `000233` from
  `6` to `10` remapped IDSW, adding switches around `973/1081` and `1138/1144`.
  The code was reverted and must not be reintroduced without a stronger
  discriminator.
- Loosening existing suffix repair for overlapped suffixes
  (`suffix_pair_swap_min_suffix_frames=600`,
  `suffix_pair_swap_max_suffix_overlap_iou=1.0`) in `20260707_150456` also
  worsened `000233` from `6` to `10` and badly reduced IDF1/coverage.

Diagnostics: upper-bound GT-aware simulation shows that manually swapping
`ID_2/ID_8` at frame `923`, `ID_1/ID_8` at frames `939-940`, and `ID_1/ID_8`
from frame `1111` onward could make `000233` reach `0` IDSW without changing
FP/FN. However, those fixes rely on GT/evaluator knowledge: runtime motion gain,
raw IDs, and hidden-overlap signals are not distinctive enough. Hidden-overlap
runs similar to the desired `1111-1118` segment also occur earlier (`973-982`,
`1053-1062`) where swapping is harmful. Avoid hardcoded video/frame repair in
promotable tracking logic.

## 2026-07-07 overlap small-box suppression candidate

New best current 5-video opt-in candidate:
`outputs/eval/hybrid_bytetrack/codex_overlap_suppress_5video/iou0_area0_condarea0_merge0`.

Metrics:

- `Pigs291119_000231_30fps`: stayed `0` remapped IDSW.
- `Pigs291119_000233_30fps`: improved from `6` to `2` remapped IDSW.
- `Pigs291119_000263_30fps`: stayed `0` remapped IDSW.
- `Pigs301119_000328_30fps`: stayed `0` remapped IDSW.
- `Pigs291119_000302_30fps`: stayed `0` remapped IDSW.
- `ALL`: improved from `6` to `2` remapped IDSW versus the suffix candidate.

Winning add-on is `overlap_small_box_suppression=true` on top of the protected
practical config, the `000233` occlusion-reid guard, and
`suffix_pair_swap_repair=true`. Default thresholds are intentionally conservative:
`overlap_small_box_min_iou=0.40`,
`overlap_small_box_max_area_ratio=0.65`, and
`overlap_small_box_max_score=0.75`.

Diagnosis: the early `000233` switches at `923/924` and `939/941` are not raw-ID
owner failures. The runtime keeps the expected IDs, but the evaluator matches GT
`ID_8` to a neighboring smaller low-confidence box because its IoU is slightly
higher during heavy overlap. The new opt-in post-processing marks those small
low-confidence overlapped boxes Hidden, removing the short IDSW bounces. The
remaining `000233` switches are `1111/1119`, a harder `ID_1/ID_8` long conflict
that should not be fixed by broad suffix or GT-aware swaps.

Keep this candidate opt-in pending broader/full-set regression before base
promotion.

## 2026-07-07 hidden suffix ID-swap candidate

New best current 5-video opt-in candidate:
`outputs/eval/hybrid_bytetrack/codex_hidden_suffix_id_swap_5video/iou0_area0_condarea0_merge0`.

Metrics:

- `Pigs291119_000231_30fps`: stayed `0` remapped IDSW.
- `Pigs291119_000233_30fps`: improved from `2` to `0` remapped IDSW.
- `Pigs291119_000263_30fps`: stayed `0` remapped IDSW.
- `Pigs301119_000328_30fps`: stayed `0` remapped IDSW.
- `Pigs291119_000302_30fps`: stayed `0` remapped IDSW.
- `ALL`: improved from `2` to `0` remapped IDSW versus the overlap-suppress
  candidate.

Winning add-on is `hidden_suffix_id_swap_repair=true` on top of the protected
practical config, the `000233` occlusion-reid guard, `suffix_pair_swap_repair`,
and `overlap_small_box_suppression`.

Diagnosis: after the small-box suppression candidate, the only remaining
`000233` switches were `1111/1119` between `ID_1` and `ID_8`. Hide/unhide
simulations only moved the switch; only a suffix identity swap from frame `1111`
to the end removed both switches. The promotable discriminator is intentionally
narrow: a low-confidence hidden run that is long enough but not too long,
strongly overlaps one visible partner, then has a long common suffix. Defaults:

- `hidden_suffix_id_swap_min_hidden_frames=8`
- `hidden_suffix_id_swap_max_hidden_frames=15`
- `hidden_suffix_id_swap_min_overlap_iou=0.70`
- `hidden_suffix_id_swap_max_hidden_median_score=0.50`
- `hidden_suffix_id_swap_start_back_frames=7`
- `hidden_suffix_id_swap_min_suffix_frames=600`

On the 5-video run this detected the `000233 ID_8/ID_1` suffix crossing without
triggering regressions on `000231`, `000263`, `000328`, or `000302`. Keep this
opt-in pending broader/full-set regression before base promotion.

## 2026-07-07 broader regression correction

The broader regression run
`outputs/eval/hybrid_bytetrack/20260707_174142/smooth_det020_loose/iou0_area0_condarea0_merge0`
proved the previous full 5-video stack is not a safe common baseline. It fixed
the target videos (`000233=0`, `000263=0`) but regressed previously clean videos:

- `Pigs281119_000085_30fps`: `0 -> 2` remapped IDSW.
- `Pigs291119_000225_30fps`: `0 -> 2` remapped IDSW.

Ablation on `000085/000225/000233/000263` isolated the issue:

- `ablate_control_assoc_occlusion_4video`: `000085=0`, `000225=0`, `000233=6`, `000263=2`.
- `ablate_suffix_only_4video`: `000085=2`, `000225=2`, `000233=6`, `000263=0`.
- `ablate_overlap_only_4video`: `000085=0`, `000225=0`, `000233=2`, `000263=2`.
- `ablate_overlap_hidden_no_suffix_4video`: `000085=0`, `000225=0`, `000233=0`, `000263=2`.

Decision: do not promote `suffix_pair_swap_repair=true` in its current form. It
fixes `000263` but creates false suffix swaps on clean videos. The current safest
common candidate for broader validation is:

- protected association/occlusion practical base:
  `hidden_owner_guard=true`,
  `hidden_owner_guard_hold_assignment=true`,
  `reentry_unowned_raw_mismatch_episode_reject=true`,
  `reentry_unowned_raw_mismatch_episode_action=hold`,
  `reentry_unowned_raw_mismatch_episode_max_events=8`,
  `reentry_unowned_raw_mismatch_episode_min_missed=1`,
  `reentry_unowned_raw_mismatch_episode_max_missed=20`,
  `reentry_unowned_raw_mismatch_episode_max_cost=0.36`,
  `occlusion_reid_prefer_gap_over_bad_match=true`,
  raw-mismatch/unowned/occlusion-hold-only with `min_missed=7`,
  `max_missed=12`, `min_cost=0.55`, `max_cost=0.70`.
- add `overlap_small_box_suppression=true`.
- add `hidden_suffix_id_swap_repair=true`.
- explicitly keep `suffix_pair_swap_repair=false`.

Next step: run broader/full regression with this no-suffix common candidate. The
remaining `000263=2` should be addressed by a new, narrower discriminator rather
than by current suffix repair.

## 2026-07-07 no-suffix common candidate full regression

Full 12-video validation of the no-suffix common candidate passed:
`outputs/eval/hybrid_bytetrack/no_suffix_common_candidate_full/iou0_area0_condarea0_merge0`.

Compared with `outputs/eval/hybrid_bytetrack/Best_tracking/iou0_area0_condarea0_merge0`:

- `ALL` remapped IDSW improved `11 -> 2`.
- No video increased remapped IDSW.
- `Pigs291119_000233_30fps` improved `9 -> 0`.
- `Pigs291119_000263_30fps` stayed `2`; this is the remaining target.
- Guardrail videos stayed clean: `000085=0`, `000225=0`, `000231=0`,
  `000302=0`, `000328=0`.

Current safest broader candidate:

- protected association/occlusion practical base.
- `overlap_small_box_suppression=true`.
- `hidden_suffix_id_swap_repair=true`.
- `suffix_pair_swap_repair=false`.

Do not promote the previous full stack from `20260707_174142`; it included
`suffix_pair_swap_repair=true` and caused false switches on `000085` and
`000225`. Future `000263` work should either build a new narrower discriminator
or heavily gate suffix repair so it cannot trigger on clean videos.
