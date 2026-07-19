# Workflow

## Active mixed source workflow (2026-07-20)

Build the mixed source from the locked legacy P0-P10 export and the 12 XML
behavior files in `data/annotations/classification`. Never substitute the
older `data/annotations/tracking` directory. Hash every source and the merged
CSV before Hidden review. Preserve `legacy_recovered` and
`cvat_tracking_xml` provenance through temporal harmonization, review units,
folds and the training snapshot.

## Human-review evidence and sampling workflow

Use this order for each exact source lineage:

1. Finish two-sided frame/object Hidden review and apply it.
2. Harmonize native temporal units and build unreviewed windows.
3. Build causal Pig-STRENet artifacts from the harmonized frame table.
4. Exact-join validity-masked review evidence and assign behavior cohorts.
5. Write the immutable behavior scientific design before any decision.
6. Run GUI smoke, then resume full review on the same decision roots.
7. Require exact coverage and PASS the behavior scientific gate before
   authorizing a reviewed training snapshot.

History-to-target features require complete history and target. Random behavior
audit estimates only the post-high-risk residual intervention rate using exact
sampling weights. High-risk yield and clean controls remain diagnostic;
unselected units are never renamed human-verified clean. Review and sampling
fields are prohibited from model-X.

## Authoritative two-pass classifier research flow

1. Establish a stable measurement base. Search credible sequence heads and
   visual backbones with grouped inner validation, progressive budgets, and
   modest tuning; stop when rank and calibration are stable enough to measure
   input effects rather than spending the final compute budget.
2. Freeze its data, folds, seeds, preprocessing, optimizer exposure, capacity
   envelope, and metrics. Run seven singles and all 21 pairs, then use a
   predeclared beam to add one modality per level. Stop on a frozen no-gain rule.
3. Confirm the selected set with leave-one-out and the all-seven endpoint. Each
   subset uses parameter-matched-zero, availability-only, and real controls.
   Freeze the subset before comparing fusion architectures separately.
4. On rented GPUs, jointly tune the selected visual backbone, temporal model,
   and fusion. Increase budgets progressively from correctness and short pilots
   to multi-seed development; never select from outer-fold predictions.
5. On the tuned strong finalist, repeat matched zero/availability/real
   ablations for every retained modality. Lock a candidate only after global
   and behavior-specific confirmatory gates pass.

Candidate families may include strong 2D image encoders with sequence heads
and end-to-end video backbones; the RTX 3050 does not prune this search space.
Use the local GPU for semantic/correctness gates and bind remote runs to the
same manifest and implementation SHA. Reuse valid caches, predictions,
checkpoints, and diagnostics; rerun only after semantic changes or failed
artifact audits. Existing legacy results remain screening evidence, not proof
that high-capacity joint tuning or confirmatory ablation has been executed.

The existing all-seven run is only the ladder endpoint/reference. A negative
subset result enters failure attribution before rejection: input/availability
audit, modality-only probe, actor-residual probe, within-stratum permutation,
learning curves and gradient health, then a stronger mask-aware fusion control.
Classify the outcome as `NO_SIGNAL`, `REDUNDANT_WITH_ACTOR`, `UNDERPOWERED`,
`OPTIMIZATION_FAILURE`, `FUSION_CAPACITY_FAILURE`, or `DATA_QUALITY_FAILURE`.

Run only synthetic and representative subset canaries on legacy 16f. Its rare
class support and unreviewed source make exhaustive ranking low-value. Execute
the complete 21-pair and beam ladder after the reviewed main snapshot is frozen.

## Active worktree routing rule

Use the current main worktree by default. When the user starts two concurrent
sessions and explicitly assigns a worktree/branch to one session, that
assignment is binding for that session only. Do not assume that
`PIG_task_model` or `PIG_task_tracking` is permanent. Verify the assigned
repository root and branch before every implementation session; if no separate
assignment was made, remain in `C:\Users\ironh\Downloads\PIG_Behavior_Project`.

Creating a worktree does not merge or copy uncommitted changes. Do not merge,
copy, stash, commit, or apply changes between worktrees unless the user
explicitly requests that operation. Tracking remains separate only when the
user assigns `C:\Users\ironh\Downloads\PIG_task_tracking` to a session.

## Legacy CVAT correction to recovered 16f

Use the canonical classification source lane, not a separate model pipeline:

```text
task_0..task_3 annotations + manifest
  -> versioned provenance scaffold + explicit source-video policy
  -> duplicate preview and nodup scaffold with row accounting
  -> fail-closed CVAT anchor audit
  -> CVAT-derived center/scaffold and six-anchor bbox tables
  -> one complete-group dense recovery smoke
  -> post-recovery behavior/bbox/frame/key checker
  -> versioned full dense recovery
  -> frame-object export + independent native-CVAT k0 authority audit
  -> existing classification_v2 merge/features/review flow
```

Resolve each shape through `manifest.jsonl`. For each actor, propagate only
the `k0` behavior to `k1..k5` and all 16 dense frames. Preserve six independent
CVAT bboxes; recover only the ten intervening frames. Hidden is separate from
behavior propagation. Repeat audit and smoke after any annotation hash change.
Never overwrite raw `data/`, the old dense reference, or canonical outputs.
`PASS_WITH_DECLARED_EXCLUSIONS` is not a clean pass: inspect and explicitly
approve every excluded actor key or complete its six anchors before recovery.

The executable clean-root command sequence is
`docs/LEGACY_16F_REBUILD_FROM_SCRATCH_RUNBOOK.md`. Root CSV paths are
historical and must not be used. `exclude_source_videos.csv` is a reviewed
policy input; duplicate preview/filter artifacts are derived outputs.

## C6 temporal-control matrix activation

The code-ready temporal matrix is fail-closed. While legacy data is being
cleaned, only static and synthetic commands are legal:

```bat
set PY=C:\Users\ironh\anaconda3\envs\pig_project\python.exe
set PYTHONPATH=%CD%\src
set C6TDIR=scripts\classification_v2\04_baselines_smokes
set C6T=%C6TDIR%\check_c2v2_c6_temporal_controls.py
set C6TCFG=configs\classification_v2\legacy_development_c6_temporal_controls_code_ready_v1.json
%PY% %C6T% --config %C6TCFG% --static-preflight
%PY% %C6T% --config %C6TCFG% --synthetic-preflight
```

After a clean lineage handoff, create a new versioned short config with the
handoff ID, exact clean input hashes, fresh output root, and
`data_run_authorized=true`. Run `--data-preflight`; do not run constant or
shuffled-delta modes when the identifiability gate says they are equivalent to
real timing. Run each authorized `--run-mode` twice using `repeat01` and
`repeat02`, then run `--audit-short-gate`.

Full development requires a separate config with scope
`full_development_confirmation`, explicit development authorization, and the
exact path/hash/status/config hash of the PASS short gate. It never authorizes
full OOF. Sequence shuffle must use one shared permutation across actor,
geometry, motion, ROI, social, pen, union/full-frame content and their aligned
availability/quality masks.

## C6 2026-07-19 rebuild screening record

After the clean technical rebuild handoff, the required order was completed:

1. C6 temporal controls: 9 modes x 2 fresh repeats, then A128 freeze.
2. C6 modality inputs/cache from the new rebuild lineage.
3. C6 modality matrix: 22 modes x 2 fresh processes, then paired evaluate.

The modality decision is `PASS` with 44 valid packets, 14 paired comparisons,
2,000 video-cluster bootstrap draws per comparison, and zero errors. It is
still `legacy-only-unreviewed-development`; Hidden and behavior review remain
double-check pending. The actor/context pipeline uses hash-bound `.npy`
features and records zero source-media reads. No full development, Q2 or
main-branch authorization follows from this screening.

Interpret this matrix at two levels. The global gate may defer a branch from
legacy full-development, but it must not erase its per-class evidence. Preserve
all ten behavior rows and classify every branch as `retained`, `deferred for
reviewed-lineage retest`, or `unsupported because of insufficient class
support`; never rewrite `deferred` as generally useless.

For the future main reviewed lineage, run behavior-conditional modality work in
this order:

1. Freeze the reviewed source manifest, snapshot, native units, folds, actor
   base, temporal view, feature whitelist, seeds, and metric contract.
2. Run the seven-single, 21-pair, beam and leave-one-out subset ladder with all
   three controls; report all ten classes and behavior groups.
3. Add paired intervals, calibration/NLL, source/availability strata, harm
   bounds and the failure-attribution probes before rejecting a branch.
4. Freeze the selected subset, then compare fusion families from predeclared
   class-modality hypotheses or a declared factorial design.
5. Repeat static, synthetic, tiny-overfit, resume, and representative short
   gates after every semantic change. Run bounded full-development only for
   candidates that pass both behavior-specific and global safety gates.
6. Lock finalists before requesting the separate full-OOF authorization. Do
   not use outer-fold predictions to choose class-modality weights.

## C6 modality matrix activation

The matrix has one direct actor-only arm and three controls for each optional
branch: parameter-matched zero, availability-only, and real values. The
branches are geometry, motion, ROI, numeric social, pen context, union context,
and full-frame context. Union and full-frame are always separate experiments.

While the bound legacy lineage is dirty, only these commands are legal:

```bat
set C6DIR=scripts\classification_v2\04_baselines_smokes
set C6=%C6DIR%\check_classification_v2_legacy_c6_modality_matrix.py
set C6CFG=configs\classification_v2\legacy_development_c6_modality_matrix_code_ready_v1.json
python %C6% --config %C6CFG% --action static-preflight
python %C6% --action synthetic-preflight
```

The first project-data action remains fail-closed until the user supplies a
clean lineage. After handoff, create a new config with updated hashes, a
nonblank handoff ID, a new output root, and explicit authorization. Then run
`build-cache`, two separate `run-repeat` commands, and `evaluate` in that order.
Do not start full development until this short matrix passes. Never infer a
full-data model decision from the current dirty lineage.

## Legacy 16-frame native unit versus model input

For bounded legacy development, preserve the complete 16-frame burst as the
grouping, split, support, and evaluation unit. A model does not need to consume
all 16 contiguous frames. The current one-sequence contract is:

```text
C6 contiguous centered: offsets 5,6,7,8,9,10
C8 contiguous centered: offsets 4,5,6,7,8,9,10,11
S6 uniform span-16:     offsets 0,3,6,9,12,15
```

Use one sequence per native unit. Compare S6 with C6 to isolate temporal span
at fixed six-frame input, then compare C8 with C6 to isolate sequence length.
Keep the cache, native-unit set, fold, model, loss, seed, epoch count, and
optimizer exposure identical. Preserve real elapsed deltas: each S6 step spans
three original frame intervals.

The 2026-07-17 paired decision retains C6 for this one-sequence profile. S6 and
C8 remain registered ablations, not promoted defaults. The older sliding-T6
candidate has four windows per native unit and different optimizer exposure;
report it only as historical context, never as a causal paired comparison.
This legacy choice does not alter the main branch's source manifest or primary
temporal view and does not authorize Q2 or full OOF.

## Pen-boundary context is an isolated model candidate

The enhanced frame-feature step now derives label-independent pen context from
`data/annotations/scene/mask.png`. Canonical runs must bind mask SHA-256
`b59b998ef49335b730c5f117e7161f24ccd277d3b5130c0e640dab7bbb980658`,
threshold at 127, and use nearest-neighbor only when frame-size resizing is
required. Mask paths, hashes, availability, quality, inward normals,
`pen_center_inside` and binary `pen_near_boundary` stay outside model-X.

Spatial export may emit `pen_boundary_context`, but the current trainer
whitelist and full model do not enable it. The first promotion experiment
changes one family only: paired `actor_geometry_motion` versus
`actor_geometry_motion_pen` on the same native units, folds, temporal view,
seed, backbone, loss and sampler. Both modes receive `motion_delta`, so gain
cannot be attributed merely to adding generic movement. Run synthetic and short
real feature gates before any bounded model pilot. No full OOF or
external-camera claim is authorized by feature availability alone.

## Canonical Hidden evidence tiers

Hidden remains a frame/object review decision. After apply, sequence windows
must be rebuilt from frame rows; do not fast-reuse a pre-review window manifest
while the default Hidden quality policy is enabled.

The canonical policy is:

```text
main_train:
  hidden_burden_ratio <= 0.25
  hidden_longest_run_ratio <= 0.20

robust_train_only:
  hidden_burden_ratio <= 0.50
  hidden_longest_run_ratio <= 0.40

exclude:
  either robust limit is exceeded
  window_sample_weight = 0.0
```

Apply this contract independently to T6, T8, T12 and T16. Hidden burden uses
the current frame-level `hidden` values after review apply, including untrusted
Hidden=Yes conservatively. Hidden ratios, run lengths, trust and policy tiers
are audit/mask metadata and must never enter model-X. The generated
`--no-exclude-high-hidden-from-main` CLI option is ablation-only.

## Isolated reviewed-Q2 execution roots

Operator commands for source rebuild, Hidden review, behavior review and apply
write only below
`human_review_workspace/classification_v2/<RUN_ID>`. Agent commands never write
that root and never reuse current canonical output folders. Every agent audit
or downstream artifact uses one unique
`outputs/classification_v2/agent_audits/<AUDIT_RUN_ID>` root.

The operator sends `RUN_ID`, `REVIEW_STAGE`, reviewer and review-code SHA at
handoff. Until `REVIEW_STAGE=behavior_complete`, the agent stops before
post-review rebuild, snapshot and project-data model smoke. After handoff, the
artifact map, generated contract, model-input manifest, snapshot and P0 audit
must all stay under the same agent root. Follow runbook section 17.2.1.

## One engine, isolated lineage profiles

Do not maintain an independent 16-frame feature implementation. Shared modules
may be reused, but source manifests, data hashes, reviews, folds, snapshots,
goals, authorizations, metrics and claims remain isolated by lineage:

```text
canonical classification_v2 engine
  + legacy-only-unreviewed-development profile
      -> legacy source only
      -> historical prompt/goal and configuration-screening lane
      -> T6/T8/T12/T16 inside each native 16-frame burst
      -> isolated development artifacts and claims
      -> no automatic activation or PASS transfer to the main goal
  + main classification profile
      -> source set bound only by its own versioned manifest
      -> legacy 16f currently excluded; no implicit legacy merge
      -> Hidden and behavior review required for its own selected data
      -> fixed6_observed_time primary view
      -> reviewed final artifacts after all gates pass
```

Legacy 16f remains unreviewed even though P0-P10 is technically clean. It needs
its own Hidden and behavior review before reviewed/train-ready use. While it is
outside the main source manifest, those decisions do not replace or block the
main profile's review coverage.

Review is a profile-level scientific policy, not an unavoidable code-path
dependency. A user-authorized exploratory profile may bypass human review only
when its manifests and outputs explicitly remain unreviewed. The current
mixed-source Q2 lineage is review-required and stays blocked until both review
layers pass; this does not block the separate legacy development profile.

Run the legacy L0-L8 scoped goal in a new chat. On completion, write and verify
its immutable handback audit, return to the original chat, and resume the
parent mixed-source P0-P8 goal. Never treat legacy completion as parent-goal
completion.

## Reviewed-data rebuild gate

For a new `classification_v2` data lineage, follow
`docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`. Full runs are
authorized only after the same semantic config passes static checks, a short
legacy+CVAT chain, and schema/count/hash/output/runtime audits.

This is standing user authorization: once those gates pass, proceed with a
necessary full run without asking again only because it is full or long. If a
gate fails or any semantic input changes, stop and repeat the short chain. Full
OOF must also pass its immutable launch packet and execution gate.

Required frame-data order:

```text
enhanced frame features
  -> two-sided Hidden manifest and media gate
  -> human Hidden decisions and fail-closed coverage
  -> hidden_reviewed_frame_features.csv
  -> temporal harmonization and sequence windows
  -> behavior review units and behavior decision apply
```

CVAT Hidden is tracking-derived/untrusted before review. The Hidden GUI must
show full-frame context, write decision CSV only, and apply each decision to
its declared frame/object key. Do not use the legacy GUI that writes corrected
source copies for a new lineage.

Detailed settled Hidden policy and validation evidence are in
`.agents/memory/09_HIDDEN_REVIEW.md`.

The old v6 root is a technical template/media reference only. Its 30 carried
payload rows are unverified because the user confirms no review has started;
do not continue from or carry that CSV. Build a clean root under
`human_review_workspace/classification_v2/<RUN_ID>` and start at zero.
Decision outputs live only below that root, while agent audits use
`outputs/classification_v2/agent_audits/<AUDIT_RUN_ID>`. During review, agents
may read but must not write the selected human root or launch either GUI. The
same read-only rule remains after handoff; agent evidence stays in its audit
root.

Separate roots prevent artifact collisions, but not code-version races. The
operator starts only after `READY_FOR_HUMAN_REVIEW` handoff with an exact Git
SHA and a short-gate-passing semantic configuration.

The current identifier-v2 short chain is independently verified under
`outputs/classification_v2/rebuilds/scientific_smoke_identifier_v2_20260713`.
After changing source/features/temporal/image/train-ready ordering, rebuild a
new bounded root and run both lineage and consolidated gates:

```bat
set S9=scripts\classification_v2\09_final_release_audit
set BASE=outputs\classification_v2\rebuilds
set ROOT=%BASE%\scientific_smoke_identifier_v2_20260713
set REPEAT=%BASE%\scientific_smoke_identifier_v2_repeat_20260713
%PY% %S9%\check_classification_v2_identifier_v2_lineage.py ^
  --root %ROOT% ^
  --repeat-root %REPEAT% ^
  --overwrite
%PY% %S9%\check_classification_v2_technical_smoke_gate.py ^
  --root %ROOT% ^
  --repeat-root %REPEAT% ^
  --overwrite
```

Expected statuses are `PASS_IDENTIFIER_V2_TECHNICAL_HUMAN_REVIEW_BLOCKED` and
`PASS_TECHNICAL_SMOKE_HUMAN_REVIEW_BLOCKED`, never training authorization.
Use both sources and all 10 behaviors. Builders must exit nonzero on audit
errors. Existing derived outputs require explicit `--overwrite`; prefer a new
versioned directory for changed semantics.

The temporal-view code contract is `PASS IN CODE` at `bb225ff`. After reviewed
windows exist, build `fixed6_observed_time`, `fixed6_normalized_phase`, and
`native6_16` with the block `02` temporal-view builder, then run its structural
shortcut checker. The fixed-six view reuses existing harmonized six-frame
windows; never sample six quantiles across a legacy burst. Keep all original
windows in the selection ledger and keep source/native-length metadata outside
model tensors. An unmitigated source shortcut is a training hard stop.

For Pig-STRENet artifact work, build the causal pair manifest before any model
integration. Legacy windows use explicit relative starts but export actual
frame boundaries; XML uses its native six-frame target and preceding same-track
history. Reject mixed-coordinate completeness calculations.

Run the control matrix in this order:

```text
T0 -> T1 -> H0 -> HA -> HS -> HR -> HRev -> PM
```

All derived views from one native event must have weights summing to `1.0`.
Pack ROI dynamics and fixed top-K social edges with deterministic row indexes
and masks. Partner routing is geometry-only. Keep availability/provenance out
of default model X and expose it only to HA. A bounded artifact canary must bind
input, code, config, environment and artifact hashes before trainer work.

The corrected media-bridge canary is
`pig_strenet_media_bridge_legacy_20260719_canary11`. It is an exporter/audit
PASS only: actor-crop difference maps are materialized from legacy crop files,
and full-scene ROI patches are decoded from the bound source video. The XML
follow-up `pig_strenet_media_bridge_xml_20260719_canary02` also passes both
pixel branches. The resolver rejects static `background.png`/`Image #1`, binds
video path plus frame index, records source hashes and writes per-pixel
provenance. Repeat the short gate after any semantic change.

The earlier XML follow-up canary at
`pig_strenet_xml_real_20260719_canary01/07_pig_strenet_attempt2` is retained as
pre-bridge evidence. Its scene-pixel block was an exporter-resolution issue,
not a source-data absence. The corrected XML run remains
`xml-only-unreviewed-technical-canary`: it cannot be treated as reviewed data
or used to claim accuracy, promotion or training readiness. `max-native-events`
must select target keys without truncating the full frame table, so causal
history and scene/social context remain available.

Training-contract code now uses fold-local preprocessing, native-event mass
weighting, and immutable lineage. A requested `output_dir` is an output root,
not the artifact directory. The trainer owns this exact layout:

```text
output_root\fold_id\run_id
```

Downstream Python callers must use `training_run_dir(audit)`. Check a completed
packet with block `04` `check_classification_v2_run_lineage.py`. Independent
remote fold rows are merged only through block `06`
`classification_v2_merge_run_registry.py`; never concatenate or overwrite the
central registry manually. These contracts pass at `16cdb93`, but no real run
is allowed before the reviewed snapshot and smoke gates pass.

Commit `abae856` adds the model-selection layer on top of this lineage. Each
epoch writes window and native-unit validation predictions, but only grouped
inner-validation native-unit supported macro-F1 may select a checkpoint; NLL
is the deterministic tie-breaker. The selected packet contains:

```text
best_validation_predictions.csv
best_validation_native_unit_predictions.csv
best_validation_aggregation_audit.json
best_validation.pt
oof_test_predictions.csv
oof_test_native_unit_predictions.csv
oof_test_aggregation_audit.json
```

Outer-test artifacts are evaluation-only. Native prediction rows retain source
and split-group metadata for later grouped reports, while those fields remain
outside model X. Resume must match the native-selection policy in checkpoint
v6 and run identity v3; policy drift is a hard error.

For a reviewed full-multimodal candidate, rerun the lineage checker with
`--require-interaction-lineage`. Snapshot v2 must show one ordered hash for
split, image-window, and interaction-window manifests; exporter audits must
match the same hash. Full preflight additionally requires an explicit
`--lineage-audit-json` and binds snapshot, lineage, ordered-window, config, and
code hashes. Bounded technical audits keep training authorization false and
must therefore be rejected by this preflight.

The current canonical reviewed artifact is not human-review complete. Complete
all mandatory review units, pass the fail-closed decision-coverage audit, then
rebuild reviewed windows with `--disable-fast-reuse`. Use recording-date or
validated session groups; never random-split frames or overlapping windows.

Do not launch model training from a new rebuild until its versioned
data/cache/fold hashes are frozen and all local model smoke gates pass.

## Legacy-only unreviewed development lane

The user separately authorizes bounded development on the legacy 16-frame
source without waiting for current human review. Use a new versioned root under
`outputs/classification_v2/legacy_only_unreviewed_development`; never write into
the reviewed rebuild, canonical train-ready, or historical full-OOF folders.

The lane must preserve one complete 16-frame burst as the native unit, group
splits by recording date or video, keep all overlapping windows from a burst in
one role, and bind source, feature whitelist, cache, fold, and config hashes.
Every artifact and metric must carry the exact scope label
`legacy-only-unreviewed-development` and `human_review_complete=false`.

Build model-input tiers for window lengths `6`, `8`, `12`, and `16` only after
temporal harmonization. All tiers inherit the same burst-level split. Compare
both all-sliding windows with per-burst event-mass normalization and a
deterministic one-window-per-burst matched view. Keep model settings fixed and
change only temporal length; aggregate evaluation to the 16-frame native unit.

Commit `21b34fd` implements the exact model-input boundary for this ladder.
The builder emits one full-universe selection ledger and one observed-time slot
manifest for each of the eight tier/view combinations. Training config must
bind the matching view, selection column, manifest filename, and exact input
length. Actor/context inputs are never truncated; padded spatial capacity is
sliced only after post-tier masks are proven false.

Run the ladder in this order: read-only source audit, complete-unit short chain,
full legacy data rebuild, leakage-safe snapshot freeze, loader sample,
one-batch forward/backward, tiny overfit, resume, then one short development
fold. A full or long run is permitted only after the exact short configuration
passes and receives the existing explicit authorization. Results from this lane
cannot support a reviewed main-branch Q2 claim.

## Active classification_v2 Workflow Override

Use this section as the current workflow. Older tracking/RGB-D/FastAPI notes in
this file are historical unless the user explicitly switches workstreams.

Current state:

- `classification_v2` behavior recognition is the active project priority.
- Status authority is `docs/CLASSIFICATION_V2_CURRENT_STATE.md`.
- The identifier-v2 technical chain passes at commit `a83d5a5`: 688 frame rows,
  63 native/review units, 438 ordered windows, exact X whitelist, and 8/8
  source-to-window repeatability.
- Temporal-view manifests and structural shortcut checks pass 22 synthetic
  tests at `bb225ff`; no active reviewed packet has been built from them yet.
- Fold-local preprocessing, native-event weighting, and immutable run lineage
  pass at `97f83c5`, `73b901d`, and `16cdb93`.
- The mask-safe factory at `318bf58` exposes ten exact model modes and four
  temporal encoders.
- The visual-backbone contract at `07ed768` supports audited ResNet18 160/224
  and ResNet34 224 controls. Unit tests use random init and do not download
  pretrained weights; active-data pilots remain blocked by the snapshot.
- The visual schedule at `2bd2fda` applies frozen, `layer4_only`, and optional
  full stages to actor and union-context ResNets. Backbone/head optimizer groups
  are stable across resume and bind checkpoint v5, run identity v2, run
  manifest v2, and registry v4. Its V0/V1/V2 audit has zero optimizer steps.
- Native-unit checkpoint selection at `abae856` supersedes those lineage schema
  versions for new runs with checkpoint v6, identity v3, manifest v3,
  prediction manifest v2, registry v5, and run audit v3.
- Native source/missingness probes at `9b04209` require the exact ordered trainer
  whitelist and train-ready window SHA256, aggregate to `temporal_unit_key`,
  fit grouped training roles only, and emit each eligible outer-test unit once.
  The availability probe permits only label-independent registered masks.
- The synthetic-only visual gate at `3be22f8` passes deterministic ResNet18-160
  gradient, ten-class tiny-overfit, eval, and in-memory resume checks. It never
  authorizes an active-data run.
- The strict loader at `111f152` aligns real fixed-six `time_delta` tensors to
  the complete window universe. Its checkpoint v4/registry v3 contract is
  superseded for new runs by the native-selection v6/v5 schemas above.
- Current classification regression is 429 passed and 181 deselected. This is
  fixture evidence, not training authorization.
- Transformer timing plumbing now passes in code, but every model run remains
  blocked until the reviewed snapshot and its exact hashes are frozen.
- The active lineage stops at block `01`: the reference Hidden design passes,
  but verified human coverage is 0/5,131 and apply is incomplete.
- Behavior review also starts at 0/4,670 verified decisions in the clean root.
- Do not rebuild train-ready exports, refresh model preflight, or launch model
  training until both review layers pass and versioned hashes are frozen.
- The 73,668-window/32,727-native-unit full OOF at commit `18d6692` had
  split-to-multimodal positional misalignment. Keep it only as historical
  compute/pipeline evidence, never as model-performance evidence.
- Use only the numbered script workflow under
  `scripts/classification_v2/00_*` through `09_*`.
- Q2 claim remains locked until a new reviewed-lineage full run and block `09`
  completion gate both pass.

Command conventions:

1. Work from `C:\Users\ironh\Downloads\PIG_Behavior_Project`.
2. Use CMD semantics for project commands:
   `cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project`
3. Set `PYTHONPATH` before classification commands:
   `set PYTHONPATH=%CD%\src`
4. Prefer:
   `C:\Users\ironh\anaconda3\envs\pig_project\python.exe`
5. Use packed letterboxed actor and visual-context caches for full experiments.
6. Do not repeat seek/crop/resize frame loops when packed caches already exist.

Edit and commit rules:

1. Before edits, read root `AGENTS.md` plus:
   `01_PROJECT_MEMORY_SHORT.md`, `02_CURRENT_DECISION.md`,
   `03_PROJECT_RULES.md`, and `08_WORKFLOW.md`.
2. Use `apply_patch` for source/config/docs edits.
3. Avoid redirects, heredocs, here-strings, `cat`, or ad hoc write scripts for
   manual edits.
4. For Markdown memory/workflow files, do not delete/recreate the file first.
   Read exact nearby text, patch a small hunk, and preserve historical content
   unless the user explicitly asks to remove it.
5. If an `apply_patch` hunk fails, re-read the nearby lines and retry with a
   smaller context-matched patch instead of shell-writing the file.
6. For every Markdown edit, choose a stable heading or nearby anchor before
   patching. If the anchor is missing, add a small dated section near the top;
   do not use redirects, temporary overwrite files, or delete-add rewrites.
7. After a Markdown edit, inspect `git diff -- <file>`, run `git diff --check`,
   and scan changed `.md` files for overlong lines before staging.
8. Wrap long Markdown command lines with CMD continuation `^`.
9. For Markdown append/update work, follow this exact failure-prevention
   protocol:
   - Re-read the target section immediately before editing.
   - Patch under a stable heading or add one dated section near the top.
   - Keep each hunk scoped to one section and fewer than about 40 changed lines.
   - Never use `>>`, `Set-Content`, `Add-Content`, heredoc, here-string,
     `cat`, or temporary overwrite files for manual Markdown edits.
   - If a hunk fails, re-read 20-40 nearby lines and retry with a smaller hunk.
   - Verify with `git diff -- <file>`, `git diff --check`, and an overlong-line
     scan before staging.
10. Markdown failure-stop rule:
    - Treat `.md` files as hand-edited project memory, not generated output.
    - If two `apply_patch` attempts fail for the same Markdown target, stop and
      re-read the exact file section before trying again.
    - Do not recover from a failed Markdown patch by using PowerShell writers,
      shell redirects, temporary files, or whole-file replacement.
    - For append-like changes, insert under an existing heading or add one
      small dated heading near the top with `apply_patch`.
    - If the target location is ambiguous after re-reading, ask the user or
      report the ambiguity instead of guessing with a broad rewrite.

Future full-run refresh sequence after snapshot readiness:

Do not run this sequence now. It becomes active only after the reviewed data,
cache, whitelist, and fold hashes are frozen and all model smoke gates pass.

```bat
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src
set PY=C:\Users\ironh\anaconda3\envs\pig_project\python.exe
set S5=scripts\classification_v2\05_preflight_authorization
set S8=scripts\classification_v2\08_publication_reporting
set S9=scripts\classification_v2\09_final_release_audit
%PY% %S5%\preflight_classification_v2_full_multimodal_oof.py ^
  --snapshot-json outputs\classification_v2\training_snapshots\c2v2_27ed5c9963904c52.json ^
  --runtime-benchmark-audit-json ^
  outputs\classification_v2\model_benchmarks_visual_v3\summary_head\runtime_benchmark_audit.json
%PY% %S5%\write_classification_v2_full_oof_authorization_template.py
%PY% %S5%\write_classification_v2_full_oof_authorization_file.py
%PY% %S5%\check_classification_v2_full_oof_authorization_template.py
%PY% %S5%\check_classification_v2_full_oof_authorization_file.py
%PY% %S5%\check_classification_v2_full_oof_authorization_writer.py
%PY% %S5%\check_classification_v2_full_oof_preflight_freshness.py
%PY% %S5%\write_classification_v2_full_oof_launch_packet.py
%PY% %S5%\check_classification_v2_full_oof_launch_packet.py
%PY% %S5%\check_classification_v2_full_oof_execution_gate.py
%PY% %S5%\write_classification_v2_full_oof_postrun_registration_packet.py
%PY% %S5%\check_classification_v2_full_oof_postrun_registration_packet.py
%PY% %S8%\classification_v2_write_q2_progress_report.py
%PY% %S8%\check_classification_v2_q2_progress_report.py
%PY% %S9%\check_classification_v2_full_oof_completion_gate.py
```

Full OOF authorization rule:

- Do not run full OOF until `full_oof_authorization.json` has
  `authorized=true`.
- Require `acknowledges_long_run=true`.
- Require `acknowledges_no_q2_claim_until_verified=true`.
- Require non-empty `reviewer`.
- Require matching preflight config SHA256 and git commit.
- Require `check_classification_v2_full_oof_execution_gate.py` to allow
  execution.

Use these generated files as the source of truth:

```text
outputs/classification_v2/model_design/full_oof_launch_packet.md
outputs/classification_v2/model_design/full_oof_launch_packet.json
outputs/classification_v2/model_design/full_oof_authorization.json
outputs/classification_v2/model_design/full_oof_postrun_registration_packet.md
outputs/classification_v2/model_design/full_oof_postrun_registration_packet.json
```

Post-full required order:

1. Cross-fit calibration.
2. Confusion-focus comparison.
3. Ablation report refresh.
4. Experiment registry registration.
5. Completion gate.
6. Q2 progress report refresh.

Only after the completion gate reports `q2_claim_allowed=true` may the result
be described as a Q2 internal improvement candidate.

Memory refresh after full OOF:

1. Update `01_PROJECT_MEMORY_SHORT.md` with final PASS/FAIL, key metrics,
   output paths, and claim boundary.
2. Update `02_CURRENT_DECISION.md` with accepted result decision and blockers.
3. Update `06_BENCHMARK_NOTES.md` with final OOF/control metrics and confusion
   findings.
4. Update `08_WORKFLOW.md` only if launch/postrun command sequence changed.
5. Commit the memory refresh separately.

## Preserved Historical Workflow

## Historical classification_v2 pre-full workflow

Use this workflow when continuing the behavior-recognition roadmap.

1. Work from project root:
   `C:\Users\ironh\Downloads\PIG_Behavior_Project`
2. For CMD execution, set:
   `set PYTHONPATH=%CD%\src`
3. Prefer Python:
   `C:\Users\ironh\anaconda3\envs\pig_project\python.exe`
4. Before any code edit, read root `AGENTS.md` and memory files
   `01_PROJECT_MEMORY_SHORT.md`, `02_CURRENT_DECISION.md`,
   `03_PROJECT_RULES.md`, and `08_WORKFLOW.md`.
5. Use `apply_patch` for source/config/docs edits. Do not write source files
   with redirects, heredocs, here-strings, `cat`, or ad hoc scripts.
6. Before every code commit, run an overlong-line scan on changed code files,
   for example `rg -n "^.{101,}$" <changed-files>`, and run
   `git diff --check`.

### Historical classification_v2 pre-full gates

Run these checks before any full OOF launch:

```bat
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src
set PY=C:\Users\ironh\anaconda3\envs\pig_project\python.exe
set S5=scripts\classification_v2\05_preflight_authorization
set S8=scripts\classification_v2\08_publication_reporting
set S9=scripts\classification_v2\09_final_release_audit
%PY% %S5%\preflight_classification_v2_full_multimodal_oof.py ^
  --snapshot-json outputs\classification_v2\training_snapshots\c2v2_27ed5c9963904c52.json ^
  --runtime-benchmark-audit-json ^
  outputs\classification_v2\model_benchmarks_visual_v3\summary_head\runtime_benchmark_audit.json
%PY% %S5%\write_classification_v2_full_oof_authorization_template.py
%PY% %S5%\write_classification_v2_full_oof_authorization_file.py
%PY% %S5%\check_classification_v2_full_oof_authorization_template.py
%PY% %S5%\check_classification_v2_full_oof_authorization_file.py
%PY% %S5%\check_classification_v2_full_oof_authorization_writer.py
%PY% %S5%\check_classification_v2_full_oof_preflight_freshness.py
%PY% %S5%\write_classification_v2_full_oof_launch_packet.py
%PY% %S5%\check_classification_v2_full_oof_launch_packet.py
%PY% %S5%\check_classification_v2_full_oof_execution_gate.py
%PY% %S5%\write_classification_v2_full_oof_postrun_registration_packet.py
%PY% %S5%\check_classification_v2_full_oof_postrun_registration_packet.py
%PY% %S8%\classification_v2_write_q2_progress_report.py
%PY% %S8%\check_classification_v2_q2_progress_report.py
%PY% %S9%\check_classification_v2_full_oof_completion_gate.py
```

For that historical lineage, the expected pre-full status was
`PASS_PARTIAL_ROADMAP` with fail-closed full OOF authorization. It was not a
completed Q2 result and is not the expected status of the active rebuild.

### Full OOF authorization rule

Do not run the full OOF command until:

- `full_oof_authorization.json` has `authorized=true`.
- `acknowledges_long_run=true`.
- `acknowledges_no_q2_claim_until_verified=true`.
- reviewer is non-empty.
- preflight config hash and git commit match the current clean preflight.

After full OOF completes, run the postrun commands from
`outputs/classification_v2/model_design/full_oof_postrun_registration_packet.json`
in order: calibration, confusion focus comparison, ablation report refresh,
registry registration, and completion gate.

### Full OOF launch packet

Use the generated launch packet as the single source of truth:

```text
outputs/classification_v2/model_design/full_oof_launch_packet.md
outputs/classification_v2/model_design/full_oof_launch_packet.json
```

The historical packet targeted:

```text
outputs/classification_v2/model_full/full_multimodal_oof
```

Do not reuse this path or packet for a future reviewed-lineage run.

The full run must use cached letterboxed actor images and packed visual context.
Do not run ad hoc full loops that repeatedly seek, crop, resize, and convert
frames when the packed caches are available.

### Post-full memory refresh

After a successful full OOF and postrun completion gate:

1. Update `01_PROJECT_MEMORY_SHORT.md` with the final PASS/FAIL state, key
   metrics, output paths, and claim boundary.
2. Update `02_CURRENT_DECISION.md` with the accepted result decision and any
   remaining blockers.
3. Update `06_BENCHMARK_NOTES.md` with final OOF/control metrics and confusion
   findings.
4. Update `08_WORKFLOW.md` if the launch or postrun command sequence changed.
5. Commit the memory refresh separately from code changes.

## Before every coding task

1. Read root `AGENTS.md`.
2. Read `.agents/memory/01_PROJECT_MEMORY_SHORT.md`.
3. Read `.agents/memory/02_CURRENT_DECISION.md`.
4. Read `.agents/memory/03_PROJECT_RULES.md`.
5. State which memory files were read.

## Audit task

If the user asks to audit/check/diff:

- Do not modify code.
- Do not run tracking/evaluation/inference.
- Use read-only commands only.
- Report findings with file/function/behavior/risk.

## Patch task

If the user asks to patch:

- Modify only requested scope.
- Prefer one small patch at a time.
- State exact files changed.
- State behavior changed.
- State behavior intentionally not changed.
- Do not run long benchmarks unless requested.
- Before every commit that changes code, scan changed files for overlong lines:
  `rg -n "^.{101,}$" <changed-files>`.
- Wrap long conditions, strings, comprehensions, function calls, and argument
  lists proactively so formatter/linter hooks do not fail the commit.

File-write safety:

- For source/config/docs edits, use `apply_patch` with small hunks.
- Do not use shell redirects, heredocs, here-strings, `cat`, or ad hoc write
  scripts unless the file is a generated artifact.
- Inspect `git diff -- <file>` after editing and fix overlong lines before
  staging.

## Verification task

When user allows verification, run in this order:

1. Static/syntax/import check.
2. Single video `Pigs291119_000263_30fps`.
3. Single video `Pigs291119_000302_30fps`.
4. 3-video common set.
5. 7-video full set.

## Preserved legacy workflow notes from previous `.agents/WORKFLOW.md`

- ROI definitions live in `data/annotations/roi/ROI_annotations.coco.json`
  with related scene background and mask assets.
- CVAT XML annotations are processed into classification datasets and feature tables.
- Detection and tracking previously centered around `pig-track-for-annotation`
  workflows with GPU fallback to CPU.
- RGB-D occlusion handling used depth calibration files such as
  `depth_scale.npy`, `inverse_intrinsic.npy`, and `rot.npy`.
- Behavior training, export, inference, and FastAPI dashboard flows remain
  documented in `.agents/WORKFLOW.md`.
- CI/quality gates previously documented there remain preserved as legacy workflow guidance.

## Historical 2026-07-12 classification_v2 override

This records the old pre-full contract and must not be executed for the active
reviewed-data rebuild.

- `classification_v2` behavior recognition became the active priority.
- That lineage reported `PASS_PARTIAL_ROADMAP` with 44/44 pre-full gates; this
  was not a completed Q2 result.
- Its full OOF remained fail-closed until `full_oof_authorization.json` was
  authorized with reviewer, acknowledgements, config hash, and git commit.
- Before that full run, the workflow refreshed the preflight, template/file
  checks, authorization writer check, preflight freshness check, launch packet
  check, execution gate, completion gate, postrun packet check, and Q2 progress
  report.
- After each commit, it refreshed the sequence before requesting authorization.
- It launched only when `check_classification_v2_full_oof_execution_gate.py`
  allowed execution.
- After full OOF, it required calibration, confusion comparison, ablation report
  refresh, experiment registry registration, and completion gate before making
  any Q2 result claim.
