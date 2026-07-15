# Classification V2 Legacy 16F Development Goal Prompt

Version: 1.1

Status: scoped goal authority for a separate chat

Use this prompt to create a project goal dedicated to the
`legacy-only-unreviewed-development` lane. It does not replace the canonical
reviewed all-source P0-P8 goal.

## Goal Request

```text
Bạn là senior machine-learning, computer-vision và scientific-audit agent cho:

C:\Users\ironh\Downloads\PIG_Behavior_Project

Hãy tạo goal với objective:

Hoàn thiện lane legacy-only-unreviewed-development từ dữ liệu legacy 16-frame
đến một classifier 10 lớp có bounded development evidence, bằng các input
T6/T8/T12/T16, recording/video-safe validation, reusable letterboxed image
caches, exact feature whitelists, immutable lineage và ablation một biến.
Không gọi lane này là reviewed/final, không dùng nó cho Q2 claim và không chạy
canonical full OOF.

Luôn chạy project command trong CMD:

cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src
set PY=C:\Users\ironh\anaconda3\envs\pig_project\python.exe
```

## Authority Order

1. `AGENTS.md`.
2. `.agents/memory/01_PROJECT_MEMORY_SHORT.md`.
3. `.agents/memory/02_CURRENT_DECISION.md`.
4. `.agents/memory/03_PROJECT_RULES.md`.
5. `.agents/memory/08_WORKFLOW.md`.
6. `plans/classification_v2-core-classifier-roadmap.md`.
7. `plans/classification_v2-core-execution-ledger.md`.
8. This prompt and its dedicated execution ledger.

The core roadmap wins on scientific invariants. The dedicated ledger wins only
for progress inside this legacy development lane. Historical plans never
override current memory, code contracts, or fresh audit evidence.

## Scope And Claim Boundary

The direct target is the canonical 10-class order:

```text
drink, eat, fight, social-nose, explore,
lying, stand, move, sitting, playwithtoy
```

Permitted work:

- full legacy-derived data and cache builds after exact short gates pass;
- local or remote bounded model pilots after pre-experiment readiness passes;
- T6/T8/T12/T16, visual, temporal, spatial, ROI, social, and imbalance
  ablations when each comparison changes one scientific family;
- accuracy and F1 reports labeled `legacy-only-unreviewed-development`;
- comparison with the historical legacy architecture as engineering evidence.

Forbidden claims and actions:

- do not call this human-reviewed or final train-ready data;
- do not use it as all-source Q2 evidence or external generalization evidence;
- do not weaken Hidden or behavior-review blockers in the canonical lineage;
- do not run canonical full OOF or mark the parent P0-P8 goal complete;
- do not mutate, rename, or overwrite files under `data/`.

### Legacy-To-Merged Interpretation Boundary

- Class counts, rare-class failures, and metric ceilings observed in this
  legacy 16-frame lane describe only `legacy_recovered`.
- The user confirms that the merged dataset contains substantially more rare
  behavior examples. Do not use legacy-only support to reject an architecture
  or estimate the attainable performance of the merged lineage.
- Use this lane for controlled engineering and architecture evidence. Reassess
  every retained architecture on the frozen merged-reviewed lineage before any
  final, Q2, or general model conclusion.

## Starting Evidence

The starting implementation commit is `c41f1ed`. Earlier binding commits are
`21b34fd`, `ef0b3bd`, and `2049a2d`.

The bounded packet root is:

```text
outputs/classification_v2/legacy_only_unreviewed_development/
short_temporal_tiers_v2_20260714
```

Current bounded evidence:

- 496 frame/object rows and 31 complete native bursts;
- all 10 classes represented;
- 310 all-sliding windows and 124 centered matched windows;
- eight strict model views covering T6/T8/T12/T16;
- zero duplicate keys, cross-burst windows, row loss, or label changes;
- loader audit PASS for shape, timing, observation, and unselected masks;
- 442 classification tests pass with 181 deselected;
- no optimizer step and no pretrained-weight download.

Expected full legacy counts:

| Artifact | Expected rows |
|---|---:|
| frame/object rows | 72,864 |
| native 16-frame bursts | 4,554 |
| T6 all-sliding windows | 18,216 |
| T8 all-sliding windows | 13,662 |
| T12 all-sliding windows | 9,108 |
| T16 all-sliding windows | 4,554 |
| all-sliding universe | 45,540 |
| centered matched universe | 18,216 |

Treat these as expected contract values, not results. Recompute and fail closed
when observed values differ.

Preserve unrelated starting worktree changes unless the user explicitly changes
scope:

```text
.tokensave/config.json
outputs/classification_v2/train_ready_windows/feature_semantics_audit.json
scripts/diagnostics/detect_single_frame.py
```

## Immutable Temporal Contract

- One native/evaluation unit is one complete 16-frame legacy burst.
- T6/T8/T12/T16 windows are cut only inside that burst after harmonization.
- Do not interpolate, resample, cross bursts, or infer review decisions.
- All windows from one burst remain in the same recording-safe split role.
- Aggregate predictions back to one 16-frame burst before primary metrics.

Two paired protocols are mandatory:

1. `all_sliding_event_balanced`, stride 3, with total loss mass one per
   burst/tier.
2. `one_centered_window_matched`, exactly one window per burst/tier.

Only temporal input length changes in the T comparison. Keep fold manifest,
backbone, resolution, temporal encoder, loss, sampler, optimizer, augmentation,
and seed fixed.

Tensor contract:

```text
actor/context RGB  [B, T, 3, H, W]
spatial group      [B, T, D_group]
time and masks     [B, T]
behavior logits    [B, 10]
```

Actor and context tensors must already have exact T. Spatial capacity 16 may be
sliced only after proving every length-mask slot after T is false.

## Scientific And Leakage Invariants

1. Do not select every numeric column automatically.
2. Exclude labels, targets, paths, IDs, folds, policies, `manual_*`, and
   `review_*` from model X.
3. Fit normalization, priors, class weights, thresholds, and calibration only
   from each training fold.
4. Treat `pig_id` as annotation-local and never as cross-video identity.
5. Preserve every frame, burst, window, and exclusion with an explicit reason.
6. Partner and ROI routing must not inspect target behavior.
7. Missing modality requires masks, but availability is not behavior evidence.
8. Use letterbox with inspectable metadata; never square-stretch pig crops.
9. Training must use audited packed caches with zero source-media fallback.
10. Use the same eligible native bursts and folds for every paired comparison.
11. Outer or held-out predictions never tune architecture or thresholds.
12. A result without immutable data, cache, fold, whitelist, config, and code
    hashes is not promotable evidence.

## Ordered Milestones

### L0 - Reconcile State

- Read all authority files and the dedicated ledger.
- Check active goal and worktree without reverting unrelated changes.
- Verify starting commits, packet paths, counts, and audit hashes.
- Record any documentation drift before changing code.

### L1 - Complete The Short Pre-Experiment Packet

- bind every selected window to exact image-context slots;
- prove every image-context ID exists in the cache and packed index;
- verify letterbox policy, source aspect ratio, padding, dtype, and shape;
- prove zero source reads through the strict dataset loader;
- build recording-date/video-safe burst folds and class-by-fold support;
- prove every window inherits exactly one native-burst fold;
- run an independent repeat and compare ordered hashes byte-for-byte.

L1 PASS requires cache, slot, row, fold, support, and hash audits with no errors.
It authorizes the equivalent full legacy data build, not model training.

### L2 - Build The Full Legacy Lineage

- select legacy rows through a versioned source-selection module;
- rebuild context, geometry, ROI, enhanced, harmonized, interval, and window
  artifacts using canonical modules;
- create full T6/T8/T12/T16 tier manifests and strict loader evidence;
- verify expected counts, all 10 labels, no silent loss, and deterministic repeat;
- do not use ad hoc row filtering as the lineage authority.

### L3 - Freeze Development Inputs

- build versioned 160-pixel letterboxed actor cache and inspectable previews;
- build 224-pixel cache only when the controlled visual matrix reaches it;
- freeze recording-safe folds, X whitelist/blacklist, temporal views, cache
  indexes, source probes, and all hashes;
- verify source, length, padding, availability, and missingness shortcuts;
- write an immutable legacy development snapshot and lineage manifest.

L3 PASS is the first authorization for model correctness gates.

### L4 - Model Correctness Ladder

Run in order:

1. import, compile, shape, mask, and leakage tests;
2. one-batch forward/backward with finite nonzero gradients;
3. deterministic repeat;
4. checkpoint and optimizer resume equivalence;
5. overfit 16-64 unique native bursts;
6. cache-only I/O and bounded GPU memory profile;
7. one fold and one epoch with class/source support audit.

No Accuracy/F1 development comparison starts before L4 PASS.

### L5 - Controlled Core Baselines

Run the visual controls separately:

- V0: ResNet18 at 160;
- V1: ResNet18 at 224;
- V2: ResNet34 at 224.

Use V1 ResNet18/224 masked mean as the efficient temporal-length search control
and retain V2 ResNet34/224 as a capacity reference. Close every temporal-head
control with exact paired native-unit, per-class, rare-group, runtime, parameter,
and recording-cluster-bootstrap evidence before promotion or rejection.

Preserve a masked-TCN result as a negative legacy control when it does not pass
those guardrails, then continue T6/T8/T12/T16 with V1 under both temporal
sampling protocols. A legacy rejection does not reject TCN on merged-reviewed
data. Test a small Transformer only if a temporal-capacity hypothesis remains
after the TCN decision; lack of local VRAM is never the rejection reason.

Report metrics only after burst-level aggregation and keep native-unit
predictions paired. Reassess retained and rejected architecture families on the
frozen merged-reviewed lineage before any final architecture conclusion.

### L6 - Input And Context Improvement Loop

Starting from the retained actor-temporal baseline, add one family at a time:

1. geometry;
2. motion;
3. all-class ROI relations;
4. numeric social relations;
5. top-K partner set;
6. union crop;
7. full frame only if earlier interaction context remains insufficient.

Every added modality requires a parameter-matched control, availability-only
control, source probe, missing-modality test, and confusion-group report.

### L7 - Select One Imbalance Policy

Compare event-balanced CE, effective-number CE, and Balanced Softmax separately.
Never combine a new loss and sampler in the same unexplained experiment. Balance
native bursts before dividing mass among overlapping windows.

### L8 - Lock The Legacy Development Candidate

- retain the simplest candidate with stable paired gains;
- write immutable config, checkpoint, predictions, metrics, and model card;
- report global, rare, interaction, ROI, posture, locomotion, per-class,
  recording, runtime, and calibration evidence;
- preserve negative experiments in the registry;
- emit the dedicated completion and handback audit.

## Short-Before-Full Authorization

The user grants standing permission for a necessary full legacy data or model
run after the exact semantic configuration passes its short gate. Do not ask
again only because the run is long.

Before every full expansion, require:

1. static and synthetic checks;
2. the exact bounded real-data configuration;
3. schema, count, key, hash, output, cache-I/O, runtime, and VRAM audits;
4. no semantic change since the short evidence.

Any change to data, crop/resize, cache, temporal view, fold, model, loss,
sampler, or augmentation invalidates dependent short evidence. Never use a full
run as the first correctness test.

### Hardware Placement Policy

- The local RTX 3050 4 GiB GPU is a correctness and bounded-smoke host, not a
  research-capacity limit.
- Do not reject or weaken an architecture solely to fit local VRAM. Place a
  gated pilot or full development run on a rented GPU when scientifically
  justified.
- Before a remote expansion, rerun the exact short runtime gate on the target
  environment and bind GPU, driver, CUDA, precision, config, code, and artifact
  hashes to the same run schema.
- Hardware placement does not relax one-variable ablations, outer-holdout
  isolation, review boundaries, or canonical full-OOF authorization.

## Experiment Discipline

- One experiment changes one principal family.
- Register the hypothesis, parent config, semantic diff, expected class effect,
  compute cap, and stop rule before execution.
- Use the same folds, seeds, eligible bursts, aggregation, and metrics for paired
  candidates.
- Use ResNet18 for efficient search where it preserves the hypothesis, but do
  not reject ResNet34, end-to-end, multimodal, or video architectures solely
  because the local GPU cannot host their scientifically justified pilot.
- Report pooled burst-level macro-F1 as primary development metric, with
  accuracy and window metrics only as secondary diagnostics.
- Do not promote gains explained by source, availability, parameter count, or
  extra short-window placements.

## Stop Conditions

Stop the current milestone on any of these conditions:

- raw data changes;
- missing, duplicate, reordered, or cross-fold native bursts;
- cache key, slot, tensor, or packed-index mismatch;
- source-media fallback during cache-required loading;
- target/review/path/ID/fold leakage into X;
- fold-local statistics reading validation or held-out data;
- NaN/Inf, constant logits, resume drift, or unexplained row loss;
- an uncontrolled multi-family experiment;
- an attempt to call legacy evidence reviewed, final, Q2, or external.

Do not hide failures with broad exceptions, row deletion, stale fallback
artifacts, automatic relabeling, or threshold changes after held-out inspection.

## Commit And Evidence Policy

For every milestone:

1. mark one ledger item `IN_PROGRESS`;
2. patch the canonical module with tests;
3. write a versioned audit artifact;
4. run line-length, diff, compile, focused, and classification regression gates;
5. commit exactly one achievement;
6. mark the ledger `PASS`, `FAIL`, or `BLOCKED` with commit and rollback notes.

Do not stage unrelated user changes. Large caches, checkpoints, and predictions
remain derived artifacts unless repository policy explicitly tracks them.

## Goal Completion Criteria

This scoped goal is complete only when L0-L8 PASS and all required work in this
legacy lane is finished. Completion requires:

- a full, deterministic 72,864-frame and 4,554-burst legacy lineage;
- immutable folds, cache, whitelist, temporal views, and lineage hashes;
- strict T6/T8/T12/T16 actor/spatial/timing alignment;
- model correctness, resume, cache-only I/O, runtime, and bounded-pilot evidence;
- controlled visual, temporal, modality, and imbalance ablations;
- one locked legacy development candidate with burst-level paired metrics;
- explicit `legacy-only-unreviewed-development` claim flags;
- no canonical reviewed-data or Q2 gate weakened.

Write the final handback to:

```text
outputs/classification_v2/legacy_only_unreviewed_development/
legacy_16f_goal_completion_audit.json
```

The audit must bind the final code SHA, configs, data/cache/fold/whitelist
hashes, candidate artifacts, metrics, unresolved risks, and rollback path.

## Return To The Parent Goal

After this scoped goal is genuinely complete, return to the original chat and
send:

```text
Resume full classification_v2 Q2 goal using the completed legacy 16f handback
and re-audit the canonical reviewed all-source P0-P8 blockers.
```

The parent goal remains blocked/incomplete until resumed. Legacy completion
does not mark parent P0-P8 complete, replace human review, or authorize canonical
full OOF.

## Required Milestone Report

After each achievement report status, hypothesis/contract, changed files,
commands, tests, row/key/hash evidence, commit SHA, rollback, next dependency,
and any human action. Do not stop at a plan when an independent safe task remains.
