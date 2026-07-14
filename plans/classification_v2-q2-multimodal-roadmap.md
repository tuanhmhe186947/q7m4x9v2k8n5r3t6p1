# Classification V2 Q2 Multimodal Roadmap

Version: 1.0

Date: 2026-07-10

Scope: `classification_v2` only. This plan does not modify raw data and does not authorize a full training run by itself.

## 1. Objective and Claim Boundary

Build and evaluate an auditable multimodal pig-behavior classifier that combines:

- actor bbox image sequences;
- normalized bbox geometry and posture proxies;
- temporal motion features;
- class-specific ROI relations for feeder, drinker, and toy;
- label-independent full-frame and partner context;
- social pair and group context;
- hierarchical auxiliary heads for posture, motion/context, ROI intent, and interaction.

Primary paper claim:

> Improved pig behavior recognition under recording/session-safe validation using reviewed multimodal spatio-temporal data.

Claim limits:

- Do not claim external-farm, external-camera, external-cohort, or biological-subject generalization.
- `pig_id` is a within-video/session annotation identity, not a biological identity across videos.
- The primary evaluation unit is the native temporal/review unit, not an overlapping sequence window.
- Window-level metrics are diagnostic only because overlapping windows are not independent observations.
- A PASS in smoke tests proves software and data-contract readiness, not scientific superiority.

## 2. Current Evidence Baseline

The following items are already implemented and must not be rebuilt under a new parallel path:

| Surface | Current evidence | Status |
|---|---:|---|
| Reviewed sequence windows | 160,740 rows | PASS |
| Native temporal units | 33,354 rows, 33,353 valid | PASS |
| Grouped window split | 680 groups, zero reported leakage groups | PASS for smoke use |
| Publication grouping | 13 inferred recording dates | PASS contract, protocol needs strengthening |
| Actor image frame index | 245,664 loadable rows, zero unloadable | PASS |
| Spatial sequence export | bbox, motion, ROI, social, quality arrays at `[N,16,D]` | PASS |
| Multimodal forward | logits `[4,10]`, masked-padding delta `0.0` | PASS |
| Tiny multimodal overfit | 10 train and 10 evaluation rows, loss reduced | PASS smoke only |
| Auxiliary targets | 160,740 rows, zero duplicate `window_id` | PASS target contract |
| Interaction context audit | 14,790 interaction windows, 13,456 context-ready | PASS audit only |
| Interaction context gaps | 1,290 missing full-frame context, 44 missing frame context | OPEN |
| Source shortcut probe | balanced accuracy `1.0` from tabular features | CRITICAL RISK |
| Spatial source probe | balanced accuracy about `0.995` | CRITICAL RISK |
| Current publication split | 11 train dates, 1 validation date, 1 test date | INSUFFICIENT as sole paper evidence |

Existing readiness report:

`outputs/classification_v2/train_ready_windows/model_readiness_report.md`

Existing machine-readable blueprint:

`outputs/classification_v2/train_ready_windows/model_upgrade_blueprint.json`

## 3. Non-Negotiable Invariants

Every future criterion must preserve these rules:

1. Never write into or mutate `data/`.
2. Never silently drop a frame, native unit, review unit, or window.
3. Every exclusion remains in an audit artifact with an explicit reason, mask, action, and count.
4. Never silently change labels. Label changes only come from review-unit decisions and preserve before/after columns.
5. `review_unit_id`, `window_id`, `temporal_unit_key`, source IDs, path fields, label fields, policy text, and `manual_*` or `review_*` fields never enter model X.
6. No trainer may infer features by selecting all numeric columns.
7. All table joins use a declared stable key and validate uniqueness. Row-order alignment alone is insufficient.
8. Any row-order tensor artifact must carry a row-index manifest, source file hashes, schema version, and row count.
9. Context availability, branch gating, partner selection, sampling, and inference routing must not use the true behavior label.
10. Test-fold labels and examples remain unseen during model choice, threshold selection, active learning, and manual error-driven relabeling.
11. Fit normalization, imputation, class weights, calibration, and thresholds on training data only within each fold.
12. `fight` applies only to directly involved pigs; `social-nose` remains actor-only by default.
13. `stand` belongs to motion/context, not posture. `fight` does not belong to motion. `playwithtoy` remains in ROI intent and mandatory review policy.
14. Hidden trusted annotations are not auto-rejected or auto-downweighted solely because they are hidden.
15. New output identifiers use `window_id`, never `window_uid`.
16. Large full training or benchmark runs require an explicit execution gate and a declared compute budget.

## 4. Target Architecture

```text
Actor RGB crop sequence --------------------> actor visual encoder ---------+
Full-frame RGB sequence --------------------> scene context encoder --------+---> gated fusion
Top-K partner crop sequence + masks --------> partner set encoder ----------+
BBox/shape/motion/ROI/social sequences -----> spatial temporal encoder -----+
Window numeric whitelist -------------------> tabular context encoder ------+
                                                                            |
                                                                            v
                                                                 temporal fusion
                                                                            |
                         +------------------+----------------+---------------+---------------+
                         v                  v                v               v               v
                    behavior head      posture head     motion head      ROI head      interaction head
                         |
                         v
                calibrated native-unit predictions
```

The scene and partner branches are available by asset/geometry, not by ground-truth behavior. The model must receive the same routing information at training and inference time.

## 5. Dependency Graph

```text
S0A Contract freeze
  +--> S0B Scientific spatio-temporal feature upgrade -----+
  +--> S1 Label-independent scene/partner visual branch ---+--> S3 Strict training system
  +--> S2 Multitask output and masked loss ----------------+
  +--> S4 Publication fold and metric protocol ------------+
  +--> S5 Source/domain controls ---------------------------+
                                                           +--> S6 Controlled baselines and ablations
                                                                   |
                                                                   +--> S7 Hard-negative and active-review loop
                                                                   |
                                                                   +--> S8 Advanced temporal/social experiments
                                                                           |
                                                                           +--> S9 Long-term model and paper package
```

Safe parallel work:

- S0B, S1, and S2 may be implemented in parallel after S0A when their changed files do not overlap.
- S4 and S5 may be implemented in parallel after S0A, but both must be complete before S6.
- S3 can begin with config/schema work while S1 and S2 are in progress, then integrate only after both APIs stabilize.
- S7 requires out-of-fold predictions from S6. It must not inspect a confirmatory test fold.
- S8 and S9 are evidence-driven. They do not start merely because the preceding code compiles.

## 6. Commit and Change Protocol

For each criterion `Sx`:

1. Confirm a clean worktree and record the current commit.
2. Create a checkpoint commit: `checkpoint start classification v2 Sx <short-name>`.
3. Implement one behavior or contract at a time.
4. Run static checks and criterion-specific checks.
5. Generate an audit JSON with `errors`, `warnings`, row counts, key-duplicate counts, and source/label distributions.
6. Inspect `git diff --check` and the exact changed-file list.
7. Commit the completed criterion: `classification v2 Sx <short-name>`.
8. Do not combine unrelated tracking, GUI, dataset, and model changes in one commit.
9. Do not commit large `.csv`, `.npz`, `.pt`, video, or image artifacts unless repository policy explicitly tracks them. Commit code, small configs, checks, and concise reports.

Rollback rule:

- Every new branch, loss term, sampler, and domain-control mechanism is opt-in until its own PASS gate succeeds.
- Rollback means disabling or reverting the isolated criterion, not rewriting prior reviewed data.

## 7. S0A: Freeze the Reproducible Data and Prediction Contract

Status: NEXT, no full training.

### Context

Current artifacts have strong row-count checks, but the scientific pipeline also needs an immutable dataset snapshot and explicit key-based alignment across CSV and NPZ artifacts. S0A first freezes the current baseline. After S0B and S1 add approved artifacts, the same tool creates a new candidate snapshot ID; it never overwrites the baseline snapshot.

### Deliverables

Add:

- `src/pig_behavior/classification_v2/contracts/training_snapshot.py`
- `src/pig_behavior/classification_v2/contracts/model_io.py`
- `scripts/behavior_review_tools/classification_v2_freeze_training_snapshot.py`
- `scripts/dev_tools/check_classification_v2_training_snapshot.py`
- `configs/classification_v2/data_contract_v1.json`

The snapshot manifest must contain:

- contract version;
- creation timestamp;
- git commit;
- absolute project root only in runtime audit, not portable config;
- relative artifact path;
- file size and SHA-256;
- row count and ordered column names for tables;
- dtype summary and allowed-null policy;
- NPZ array names, shapes, dtypes, and finite-value counts;
- ordered `window_id` digest for every row-aligned artifact;
- label order;
- train mask, sample weight, event weight, and split artifact hashes;
- source and behavior distributions;
- forbidden-X patterns and explicit feature whitelists.

### Required behavior

- Fail closed when a required artifact is missing, duplicated, reordered, or hash-mismatched.
- Detect duplicate `window_id` and duplicate native temporal keys.
- Join CSV artifacts by `window_id`; do not accept accidental row alignment as proof.
- For NPZ artifacts, require a companion ordered `window_id` array or row-index manifest.
- Preserve all rows and report invalid masks separately.
- Treat snapshots as immutable. A changed file hash always produces a new snapshot ID and output directory.
- Require S3 to reference the post-S0B/S1 candidate snapshot explicitly; the baseline snapshot remains available for comparison.

### Verification

```bat
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src
rtk python -m py_compile src\pig_behavior\classification_v2\contracts\training_snapshot.py src\pig_behavior\classification_v2\contracts\model_io.py scripts\behavior_review_tools\classification_v2_freeze_training_snapshot.py scripts\dev_tools\check_classification_v2_training_snapshot.py
rtk python scripts\behavior_review_tools\classification_v2_freeze_training_snapshot.py
rtk python scripts\dev_tools\check_classification_v2_training_snapshot.py
```

### PASS gate

- All required artifact hashes are present.
- `window_id` alignment is proven across X, y, masks, weights, split, image, and auxiliary targets.
- Duplicate key count is zero.
- Forbidden input scan returns zero model-input violations.
- Audit `errors` is empty.

### FAIL gate

- Any consumer still relies on unverified row order.
- Any feature whitelist is inferred from dtype.
- A snapshot can be regenerated after source changes without changing snapshot ID.

## 8. S0B: Upgrade Scientific Spatio-Temporal Feature Semantics

Status: REQUIRED BEFORE freezing the final training snapshot, no full training.

### Purpose

The current spatial export is smoke-ready, but a paper-grade feature branch must prove that every value has correct time units, prediction-time availability, missingness semantics, and train/inference parity. This criterion improves the main feature modules rather than creating an output-only workaround.

### Modules to review and extend

- `src/pig_behavior/classification_v2/features/geometry.py`
- `src/pig_behavior/classification_v2/features/motion.py`
- `src/pig_behavior/classification_v2/features/roi.py`
- `src/pig_behavior/classification_v2/features/social.py`
- `src/pig_behavior/classification_v2/features/spatiotemporal.py`
- `src/pig_behavior/classification_v2/features/aggregate.py`
- `src/pig_behavior/classification_v2/spatial_sequence_export.py`
- `scripts/behavior_review_tools/classification_v2_build_enhanced_spatiotemporal_features.py`
- `scripts/behavior_review_tools/classification_v2_export_spatial_sequences.py`
- `scripts/dev_tools/check_classification_v2_spatial_sequences.py`

Add a dedicated semantic checker:

- `scripts/dev_tools/check_classification_v2_spatiotemporal_feature_semantics.py`

### Geometry contract

- Preserve raw bbox coordinates for audit.
- Export normalized center, width, height, area, and aspect ratio using verified frame dimensions.
- Export clamp distance and out-of-frame fraction instead of silently replacing the bbox.
- Distinguish invalid, degenerate, clamped, interpolated, inherited, and directly observed boxes.
- Ensure posture proxies are functions of geometry only and never behavior labels.
- Check scale invariance by resizing a synthetic frame and bbox together.

### Motion contract

- Compute within `object_track_key`, `video_key`, and source boundary only.
- Use real `delta_frame` and `effective_fps`; do not assume adjacent rows are adjacent frames.
- Export time-normalized center velocity, size velocity, acceleration, and optional jerk.
- Export displacement, path length, straightness/path efficiency, direction change, speed variance, motion energy, and burstiness.
- Mark gaps and insufficient-history rows explicitly.
- Do not forward-fill motion across a missing or cross-interval gap without an audit flag.
- Define whether features are offline-window or causal. A future realtime claim requires a separate causal export and must not reuse centered/future-aware aggregates.

### ROI contract

- Export feeder, drinker, and toy relations for every sample independently of its behavior label.
- Include overlap, edge distance, center distance, near/contact flags, and valid-ROI masks per class.
- Add window-level occupancy, contact ratio, entry count, exit count, minimum distance, and distance trend where mathematically defined.
- Keep behavior-derived `target_roi_*` and `roi_target_*` fields audit-only and forbidden from X.
- Do not remove rows when ROI is unavailable; emit quality masks and reasons.
- Verify that `playwithtoy` maps to toy in review policy while model X still contains all ROI classes symmetrically.

### Social contract

- Compute peers only within the same video and frame.
- Export nearest and top-K center/edge distances, IoU/contact, local density, relative scale, and partner count.
- Export pair persistence and approach/separation velocity using stable object-pair keys.
- Keep pair selection label-independent.
- Prevent `fight` propagation to bystanders and preserve actor-only `social-nose` semantics.
- Missing partner context is a mask/review reason, not a row-deletion reason.

### Quality and prediction-time availability

For every candidate feature, classify it as:

- available at inference;
- training-only audit;
- unavailable without an additional detector/tracker signal.

In particular, audit `Hidden` carefully. If it comes only from human annotation and the deployment path cannot reproduce it, exclude it from X or replace it with an inference-available visibility/track-quality estimate. Annotation-only visibility must not create train/serve leakage.

Other quality signals must distinguish:

- observed versus padded frame;
- bbox valid versus invalid;
- ROI available versus missing;
- partner context available versus missing;
- frame gap and time delta;
- crop/video decode success;
- clamp amount and image-boundary truncation.

### Numerical verification

Build hand-calculated fixtures for:

- constant position, constant velocity, and constant acceleration;
- one missing frame and one large frame gap;
- two tracks from different videos with adjacent row indices;
- exact ROI overlap, touching edge, near but noncontact, and missing ROI;
- two pigs approaching, stationary, and separating;
- out-of-frame and zero-area boxes;
- 6-frame CVAT and 16-frame legacy intervals.

The checker must compare expected values with tight tolerances, verify finite outputs where defined, and verify masks where values are undefined.

### PASS gate

- Hand-calculated geometry, motion, ROI, and social fixtures pass.
- No cross-video, cross-track, or cross-temporal-unit contamination exists.
- Every X feature has a prediction-time availability classification.
- Annotation-only fields are excluded from X.
- No behavior-derived target ROI field enters X.
- Enhanced frame and sequence row counts are preserved.
- Audit reports feature ranges, null/nonfinite counts, source distributions, and mask distributions.

### FAIL gate

- A feature requires the true behavior to be computed.
- Missing frames are treated as uniform time steps.
- Clamped boxes overwrite raw coordinates without a flag.
- Annotation-only Hidden status is used as if it were available at inference.
- Feature upgrades change labels or remove rows.

## 9. S1: Build Label-Independent Full-Frame and Partner Visual Context

Status: HIGHEST PRIORITY, no full training.

### Scientific hypothesis

`fight` and `social-nose` are poorly identified from actor crops alone because the relevant evidence includes another pig, relative orientation, contact, and local group context.

### Critical leakage rule

Do not use `behavior_label`, `is_interaction_window`, `review_template`, or an interaction target to decide whether scene/partner tensors are loaded. Such routing would reveal the label at inference time.

### Deliverables

Add:

- `src/pig_behavior/classification_v2/datasets/scene_partner_index.py`
- `src/pig_behavior/classification_v2/datasets/scene_partner_dataset.py`
- `src/pig_behavior/classification_v2/models/interaction_context_encoder.py`
- `scripts/behavior_review_tools/classification_v2_build_scene_partner_index.py`
- `scripts/dev_tools/check_classification_v2_scene_partner_index.py`
- `scripts/dev_tools/check_classification_v2_scene_partner_loader.py`
- `scripts/dev_tools/check_classification_v2_interaction_context_forward.py`

Modify only after the new APIs pass independently:

- `src/pig_behavior/classification_v2/models/multimodal_fusion.py`
- `src/pig_behavior/classification_v2/datasets/image_sequence_dataset.py`

### Index design

For every loadable window, not only interaction-labeled windows, export:

- `window_id`;
- ordered frame indices;
- actor bbox and actor availability mask;
- full-frame video path reference for CVAT rows;
- top-K partner object keys and bboxes per frame;
- partner availability mask;
- normalized actor-partner delta position, distance, IoU, overlap/contact proxy, and area ratio;
- context quality status and explicit missing reason;
- source type for audit only.

Partner ranking must be deterministic and label-independent:

1. same `video_key` and frame;
2. exclude actor object key;
3. valid bbox only;
4. sort by normalized edge distance, then center distance, then stable object key;
5. select top K, initially `K=3`;
6. pad with zeros and a false mask when fewer partners exist.

### Tensor contract

- Actor crops: `[B,T,3,H_actor,W_actor]`.
- Scene frames: `[B,T,3,H_scene,W_scene]` with aspect-preserving letterbox.
- Partner crops: `[B,T,K,3,H_partner,W_partner]`.
- Partner geometry: `[B,T,K,D_pair]`.
- Separate `length_mask`, `observed_mask`, `scene_mask`, and `partner_mask`.
- Never infer missingness from all-zero RGB values.

### Legacy policy

- Keep legacy rows in the dataset.
- Set scene/partner branch masks false when original full-frame assets cannot be resolved.
- Do not synthesize fake full-frame context from the actor crop.
- Report legacy interaction performance separately and state its context limitation.
- Do not claim a context benefit on legacy crop-only rows unless valid source imagery is recovered through an audited resolver.

### Model design

- Use a lightweight shared visual backbone for actor and partner crops initially.
- Use a lower-resolution scene encoder to control memory.
- Aggregate partners with masked attention or DeepSets, never by concatenating a variable number of partners.
- Fuse scene/partner embeddings only through availability masks.
- Keep context encoder opt-in until mask invariance and missing-context tests pass.
- Treat branch availability masks as potentially source-predictive and include them in source-probe diagnostics.
- Evaluate training-time modality dropout as an ablation to reduce over-reliance on source-specific context availability; dropout is random and never label-conditioned.

### Multimodal image preprocessing

- Use aspect-preserving crop resize/letterbox and record the transform.
- Evaluate a small actor-context margin such as 1.10 to 1.25 only on inner validation; keep the raw actor bbox for audit.
- Apply one consistent random transform across all frames in a sequence.
- Coordinate geometric augmentation across actor image, scene image, partner image, bbox geometry, and ROI geometry.
- Do not horizontally flip only the RGB branch while leaving spatial/ROI coordinates unchanged.
- Keep color augmentation modest and source-aware; verify it does not make one source easier to identify.
- Use pretrained-backbone normalization consistently in train and inference.
- Bbox jitter is training-only, bounded, and never changes y or audit bboxes.
- Temporal reversal, frame dropping, and speed perturbation remain disabled until their behavior semantics and all-branch consistency are validated.

### Review GUI integration

Extend the main GUI rather than introducing a separate decision schema:

- `scripts/behavior_review_tools/review_temporal_unit_gui.py`

Add reusable rendering support:

- `src/pig_behavior/classification_v2/review/context_renderer.py`
- `scripts/dev_tools/check_classification_v2_review_context_rendering.py`

Interaction view:

- show the full frame for CVAT units;
- highlight the actor bbox and top-K candidate partner bboxes with stable colors;
- show actor-only versus group annotation policy;
- retain the actor crop strip for fine detail;
- keep all decision fields and apply scope unchanged.

ROI view:

- overlay feeder, drinker, and toy polygons/boxes;
- show actor bbox and ROI relation without using the current label to hide other ROI classes;
- flag missing/invalid ROI assets explicitly.

Motion view:

- overlay the actor center trajectory for the temporal unit;
- mark observed, interpolated, and missing frames distinctly;
- do not change decision scope from native review unit.

Rendering checks must include the known `_30fps` video alias case and must never write corrected behavior for pending decisions.

### Verification

Required test cases:

- CVAT actor with 0, 1, 2, and at least 3 neighbors.
- Missing frame, invalid bbox, out-of-frame bbox, and partial sequence.
- Mixed CVAT 6-frame and legacy 16-frame batch.
- Partner order unchanged across repeated runs.
- Replacing padded partner pixels changes logits by exactly zero within tolerance.
- Removing ground-truth behavior columns from the input manifest does not change context indexing.
- Context loader can run on unlabeled inference rows.

### PASS gate

- Zero duplicate context keys.
- No context-index field depends on behavior/review labels.
- All available CVAT context samples load without errors.
- Missing legacy context remains explicit and row-preserving.
- Masked padding invariance is at most `1e-6`.
- Interaction context audit reports source, label, and readiness slices without feeding those columns into X.

### FAIL gate

- Branch routing uses the true behavior label.
- Fight partner choice uses annotation semantics unavailable at inference.
- Missing legacy scene context causes rows to disappear.
- Full-frame preprocessing differs between train and evaluation.

## 10. S2: Add Multitask Heads and Masked Hierarchical Loss

Status: HIGH PRIORITY, no full training.

### Scientific interpretation

The auxiliary targets are deterministic decompositions of the behavior label. They are an inductive-bias and regularization experiment, not independent extra annotation. The paper must not describe them as new supervision.

### Deliverables

Add:

- `src/pig_behavior/classification_v2/models/multitask_fusion.py`
- `src/pig_behavior/classification_v2/training/multitask_loss.py`
- `scripts/dev_tools/check_classification_v2_multitask_forward.py`
- `scripts/dev_tools/check_classification_v2_multitask_loss.py`
- `scripts/behavior_review_tools/classification_v2_multitask_smoke_train.py`
- `scripts/dev_tools/check_classification_v2_multitask_smoke_train.py`

### Output contract

Return a typed output object with logits for:

- `behavior`: 10 classes;
- `posture`: lying, sitting, standing-or-other;
- `motion_context`: move, explore, stand, other;
- `roi_intent`: eat, drink, playwithtoy, none;
- `interaction`: fight, social-nose, none.

### Loss contract

```text
L_total = L_behavior
        + lambda_posture * mask_posture * L_posture
        + lambda_motion * mask_motion * L_motion
        + lambda_roi * mask_roi * L_roi
        + lambda_interaction * mask_interaction * L_interaction
        + lambda_consistency * L_hierarchy_consistency
```

Rules:

- Normalize each masked auxiliary loss by its active-mask sum.
- A zero-active mask produces a zero loss, not NaN.
- Derive class weights from training-fold targets only.
- Keep auxiliary lambdas in versioned config.
- Start with fixed lambdas and ablate them; uncertainty weighting is a later experiment.
- Never feed auxiliary labels, masks, logits, or losses back into X.
- A consistency penalty may align behavior and hierarchy probabilities, but must not read test labels.

### Verification

- Shape and label-order tests for every head.
- Gradient reaches shared encoders and each active head.
- Inactive head has zero contribution.
- Shuffling an inactive target leaves total loss unchanged.
- Auxiliary artifact joins by unique `window_id`.
- Tiny overfit smoke lowers total and behavior loss on a balanced subset.
- Behavior-only mode reproduces the existing fusion output shape.

### PASS gate

- All loss terms are finite.
- Mask and class-weight handling is deterministic.
- No target column appears in model inputs.
- Smoke audit records per-head losses, active rows, label counts, and errors.

### FAIL gate

- Reported improvement is based only on training loss.
- Auxiliary heads are described as independent labels.
- Head masks are created from source or true label during inference.

## 11. S3: Build a Strict Reproducible Training System

Status: REQUIRED BEFORE ANY FULL TRAINING.

### Deliverables

Add:

- `src/pig_behavior/classification_v2/training/config.py`
- `src/pig_behavior/classification_v2/training/data_module.py`
- `src/pig_behavior/classification_v2/training/trainer.py`
- `src/pig_behavior/classification_v2/training/checkpoint.py`
- `configs/classification_v2/baseline_spatial_tcn.json`
- `configs/classification_v2/baseline_actor_image.json`
- `configs/classification_v2/multimodal_context_multitask.json`
- `scripts/behavior_review_tools/classification_v2_train.py`
- `scripts/dev_tools/check_classification_v2_training_config.py`
- `scripts/dev_tools/check_classification_v2_training_reproducibility.py`

### Configuration contract

Each run records:

- dataset snapshot ID and hashes;
- code commit and dirty-worktree flag;
- fold ID and split manifest hash;
- model architecture and parameter count;
- exact feature whitelist by branch;
- label order;
- normalization/imputation state;
- seed and deterministic settings;
- optimizer, scheduler, epoch, batch size, gradient clipping, and precision;
- class, sample, and event-weight policies;
- early-stopping metric and patience;
- hardware and package versions;
- checkpoint and prediction artifact paths.

### Training behavior

- Train selection is `split == train AND train_mask == true`.
- Validation and test remain separate. Never merge them for convenience.
- Early stopping uses validation native-unit macro F1 or a predeclared proxy, never test performance.
- Save last and best-validation checkpoints.
- Resume validates snapshot, config, fold, and label-order compatibility.
- Every prediction CSV includes `window_id`, fold, split, true label, predicted label, confidence, class probabilities, model version, and snapshot ID.
- Audit and identifiers remain in prediction outputs, not in model tensors.
- Data-loader worker and augmentation seeds are reproducible.

### Unit and smoke tests

- Forbidden columns injected into a mock input cause a hard failure.
- Reordered rows still join correctly by `window_id` or fail with a clear digest mismatch.
- Same seed and CPU smoke produce matching sampled IDs and near-identical losses.
- Resume from checkpoint preserves optimizer/scheduler state.
- Empty or all-invalid batch fails clearly.
- Mixed sequence lengths remain mask-stable.

### PASS gate

- One tiny run completes from config with no implicit defaults outside the saved config.
- Re-running the same smoke config reproduces selected IDs and metric values within declared tolerance.
- Experiment registry entry contains hashes, metrics, configs, and artifact references.
- No all-numeric feature selection exists anywhere in the trainer path.

## 12. S4: Replace the Single-Holdout Paper Protocol with Grouped Out-of-Fold Evaluation

Status: CRITICAL SCIENTIFIC GATE.

### Problem

The current publication split holds out one recording date for validation and one for test. With only 13 recording-date groups, one fixed test date cannot support a strong general conclusion and can make results highly split-dependent.

### Primary protocol

Use deterministic outer grouped cross-validation at `recording_date` level:

- Primary: 5-fold stratified-group assignment over 13 recording dates.
- Inner selection: grouped validation from the outer-train dates only.
- Produce exactly one out-of-fold prediction per valid native temporal unit.
- Keep all windows from a native unit, video, and recording date in the same outer fold.
- Never group by `pig_id` across videos.

If five folds cannot preserve required class coverage, the fold builder must report this and choose the largest predeclared feasible fold count. It must never silently move rows between groups.

### Sensitivity protocols

- Leave-one-recording-date-out evaluation for split-sensitivity analysis.
- CVAT-only and legacy-only grouped evaluations.
- Matched 6-frame subset across sources to test interval-length/source confounding.
- Context-ready interaction subset and all-interaction subset.

### Deliverables

Add:

- `src/pig_behavior/classification_v2/evaluation/grouped_folds.py`
- `src/pig_behavior/classification_v2/evaluation/native_unit_metrics.py`
- `src/pig_behavior/classification_v2/evaluation/statistics.py`
- `scripts/behavior_review_tools/classification_v2_build_q2_folds.py`
- `scripts/behavior_review_tools/classification_v2_evaluate_oof_predictions.py`
- `scripts/dev_tools/check_classification_v2_q2_folds.py`
- `scripts/dev_tools/check_classification_v2_oof_metrics.py`

### Fold audit

For each fold, record:

- recording dates, videos, sources, native units, windows, and valid rows;
- behavior counts and class-coverage flags;
- source-by-behavior contingency table;
- context readiness counts;
- zero overlap for recording date, video, native unit, review unit, and window IDs;
- fold assignment seed and algorithm version.

### Metrics

Primary:

- native-unit macro F1 from pooled out-of-fold predictions.

Secondary:

- balanced accuracy;
- per-class precision, recall, and F1;
- macro recall;
- multiclass MCC;
- log loss;
- Brier score;
- expected calibration error;
- confusion matrix;
- coverage and abstention rate if abstention is enabled.

Rare-class reporting rules:

- pooled out-of-fold macro F1 uses the fixed 10-class label order;
- report exact support and confidence interval for `playwithtoy` in every fold and source;
- never remove an absent fold/class from a mean without showing the support rule;
- include one-versus-rest precision-recall curves for rare classes when probability predictions are available;
- do not oversample, duplicate, or synthesize validation/test units.

Required slices:

- source type;
- recording date;
- behavior;
- window length;
- hidden/context quality;
- interaction context-ready versus context-missing;
- ROI-required versus not required;
- reviewed accept/corrected/excluded provenance for audit only.

### Statistical comparison

- Compare models on identical out-of-fold native units.
- Bootstrap clusters by recording date or video, not individual overlapping windows.
- Report 95% confidence intervals and paired differences versus baseline.
- Use a predeclared multiple-comparison correction for many ablations, such as Holm.
- Report effect size and uncertainty, not only p-values.
- Do not tune thresholds on pooled test predictions.

### PASS gate

- Every valid native unit receives exactly one out-of-fold prediction.
- Zero group leakage across outer train/test.
- All metrics are reproducible from saved predictions.
- Fold-level and pooled results are both reported.
- Unsupported class/fold combinations are explicit.

## 13. S5: Control Source and Domain Shortcuts

Status: CRITICAL SCIENTIFIC GATE.

Engineering update: commit `9b04209` implements the exact-whitelist,
ordered-window-hash-bound source probe and the label-independent
availability-only behavior probe at native temporal-unit grain. This closes the
probe correctness contract only; active reviewed-data results, learned-embedding
probes, and source-balanced candidate metrics remain pending.

### Problem

Current clean features predict `legacy_recovered` versus `cvat_tracking_xml` almost perfectly. A behavior model can therefore exploit source-correlated geometry, sequence length, missingness, ROI availability, or label prevalence instead of learning behavior.

### Required controls

1. Report combined, CVAT-only, legacy-only, and source-balanced metrics.
2. Report per-class metrics inside each source where class support exists.
3. Add a matched-length control using 6-frame windows for both sources.
4. Fit normalization and imputation inside each training fold only.
5. Audit feature missingness and distribution shift by source and behavior.
6. Train a linear source probe on learned embeddings. Lower source predictability is diagnostic, not automatically proof of better behavior representation.
7. Compare uniform, class-balanced, event-balanced, and source-class-balanced weighting.
8. Add source-adversarial learning only as an ablation after simpler controls.
9. Compare ROI-only, geometry-only, and image-only probes to quantify camera/location shortcuts.
10. Report whether each recording date contains the same camera/ROI layout; do not call a date-safe split camera-safe unless metadata proves it.

### Deliverables

Add:

- `src/pig_behavior/classification_v2/evaluation/domain_controls.py`
- `src/pig_behavior/classification_v2/training/samplers.py`
- `scripts/behavior_review_tools/classification_v2_build_source_matched_views.py`
- `scripts/behavior_review_tools/classification_v2_evaluate_domain_controls.py`
- `scripts/dev_tools/check_classification_v2_source_matched_views.py`
- `scripts/dev_tools/check_classification_v2_domain_controls.py`

Optional experiment, disabled by default:

- `src/pig_behavior/classification_v2/models/domain_adversarial.py`

### Interpretation rules

- Cross-source transfer is an internal domain-shift diagnostic, not external generalization.
- Removing source information is not always beneficial because frame cadence and availability masks are operationally real. The correct goal is behavior performance that remains stable across source slices.
- A domain adversary is rejected if it improves pooled metrics while materially damaging a supported behavior/source slice.
- Report label-source confounding before interpreting model gains.

### PASS gate

- Source-balanced and per-source native-unit metrics exist for every candidate model.
- Matched-length results are available.
- Source-probe results are reported on held-out grouped folds.
- No source identifier enters X.
- No source slice regresses beyond a predeclared tolerance without explicit discussion.

## 14. S6: Run Controlled Baselines and Factorial Ablations

Status: BLOCKED until S0A, S0B, and S1-S5 pass and the user authorizes full training.

### Baselines

Train all candidates on identical folds, snapshots, weights, seeds, and prediction units:

| ID | Model | Purpose |
|---|---|---|
| B0 | Fold-specific majority and prior baseline | Minimum sanity baseline |
| B1 | Whitelisted tabular linear/tree baseline | Static handcrafted signal baseline |
| B2 | Spatial TCN | Bbox, motion, ROI, social temporal baseline |
| B3 | Actor crop temporal encoder | Image-only baseline |
| B4 | Actor image + spatial TCN | Core multimodal baseline |
| B5 | B4 + scene/partner context | Test interaction-context hypothesis |
| B6 | B5 + multitask heads | Test hierarchical regularization |
| B7 | Full proposed model | Candidate paper model |

### Minimum ablation matrix

- remove bbox geometry;
- remove motion;
- remove ROI relation;
- remove social relation;
- remove actor image;
- remove full-frame scene;
- remove partner crops;
- remove partner geometry;
- remove auxiliary heads;
- remove event balancing;
- matched 6-frame source control;
- behavior-only versus multitask loss;
- actor-only versus actor-plus-context for `fight` and `social-nose`.

### Run policy

- Start with one fold and one seed as an engineering run.
- Promote to all folds only after finite loss, nonempty predictions, valid masks, and native collapse pass.
- Use at least three seeds for the final candidate and strongest baseline if compute permits.
- Fix model choice and hyperparameter search space before opening confirmatory outer-fold results.
- Log runtime, peak memory, parameter count, and inference throughput.

### Promotion gate

A candidate advances only if:

- pooled out-of-fold native-unit macro F1 improves over B4 with a positive paired confidence interval or a clearly meaningful effect size;
- no critical source/behavior slice has unexplained severe regression;
- interaction context improves the context-ready interaction slice;
- calibration is no worse after post-hoc calibration;
- all results are reproducible from registry artifacts.

The exact numerical improvement threshold must be frozen before S6 begins, based on baseline variance and practical error cost rather than chosen after seeing test results.

## 15. S7: Hard-Negative Mining and Active Review Without Test Leakage

Status: AFTER initial out-of-fold baseline predictions.

### Candidate generation

Use training-pool or development-fold predictions only. Rank native review units by:

- predictive entropy;
- top-1/top-2 margin;
- disagreement across seeds or folds;
- confusion-focus pair probability;
- model/data-source disagreement;
- missing ROI or interaction context quality;
- rare-class coverage;
- repeated error across overlapping windows collapsed to one review unit.

Focus pairs:

- fight versus social-nose, stand, move;
- eat or drink versus stand, explore;
- playwithtoy versus explore, stand, move;
- lying versus sitting;
- move versus explore, stand.

### Deliverables

Add:

- `src/pig_behavior/classification_v2/review/active_learning.py`
- `scripts/behavior_review_tools/classification_v2_build_active_review_shortlist.py`
- `scripts/dev_tools/check_classification_v2_active_review_shortlist.py`

Outputs:

- standard review-unit template CSV;
- shortlist audit JSON;
- selection-reason distribution;
- label/source/context coverage table;
- duplicate and overlap audit.

### Review protocol

- Never auto-correct a label.
- Never emit `corrected_behavior` for pending rows.
- Select one row per `review_unit_id`.
- Reserve 10-20% of shortlisted units for blinded double review.
- Report Cohen's kappa or Krippendorff's alpha where appropriate.
- Adjudicate disagreements and version the reviewed dataset.
- Once a confirmatory test fold has been inspected for error selection, it is no longer an untouched test fold.

### PASS gate

- Zero duplicate review units.
- All decision-schema columns are present.
- Test-fold units are excluded from active selection.
- Applying decisions preserves frame row count and emits before/after audit.
- A new data snapshot ID is created after accepted corrections/exclusions.

## 16. S8: Evidence-Driven Advanced Temporal and Social Experiments

Status: ONLY after S6 error analysis identifies a plausible missing signal.

### S8.1 Graph social branch

Represent each pig in a frame as a node. Use label-independent edges for the top-K geometric neighbors.

Node inputs:

- actor visual embedding;
- normalized bbox and shape;
- motion state;
- ROI relations;
- quality masks.

Edge inputs:

- relative position and scale;
- edge and center distance;
- IoU/contact proxy;
- approach/separation velocity;
- pair persistence;
- partner availability.

Candidate modules:

- `src/pig_behavior/classification_v2/models/social_graph.py`
- `src/pig_behavior/classification_v2/models/temporal_graph_encoder.py`
- `scripts/dev_tools/check_classification_v2_social_graph_forward.py`

Gate: graph input creation must be label-independent and must not propagate `fight` to bystanders.

### S8.2 Better temporal encoder

Compare, under the same folds and inputs:

- temporal convolution network;
- bidirectional GRU/LSTM for offline classification;
- Transformer encoder with masks and relative time encoding.

Do not compare architectures with different context availability or preprocessing. Report compute and latency with quality.

### S8.3 Inter-frame visual motion

Evaluate RGB frame differences or compact optical-flow features as a separate branch. Flow is fitted/generated without labels and carries explicit missing/frame-gap masks.

Gate: demonstrate benefit specifically on move/explore/stand or fight dynamics, not only pooled accuracy.

### S8.4 Pose/keypoint branch

Proceed only if a pose model or annotation source has measurable pig-keypoint quality on this domain.

Required before use:

- keypoint visibility audit;
- per-joint missingness;
- source/camera robustness;
- no pseudo-label use on confirmatory test data;
- posture and interaction ablation.

### S8.5 Calibration and abstention

- Fit temperature scaling on inner validation only.
- Report ECE, Brier score, and reliability diagrams.
- Tune per-class thresholds only for a declared operational objective.
- An abstention policy must report coverage versus accuracy and preserve an auditable review path.

## 17. S9: Long-Term Unified Model and Paper-Ready Package

Status: LONG TERM. Start only after simpler S6/S8 ablations justify the added complexity.

### Proposed unified model

1. Shared visual backbone for actor and partner crops.
2. Low-resolution full-scene encoder.
3. Spatial TCN or temporal Transformer for bbox, motion, ROI, and quality signals.
4. Masked set/graph encoder for social partners.
5. Optional validated pose and inter-frame motion branches.
6. Gated multimodal fusion conditioned only on observed availability masks.
7. Behavior head plus hierarchical auxiliary heads.
8. Calibrated native-unit prediction and optional abstention.
9. Optional domain-adversarial regularizer, retained only if source-slice evidence supports it.

### Training schedule

Stage A:

- initialize/freeze most of the visual backbone;
- train spatial, fusion, and heads;
- verify no branch collapse and finite gradients.

Stage B:

- unfreeze the final visual blocks with a lower learning rate;
- retain event and class weighting;
- monitor per-source and rare-class validation metrics.

Stage C:

- fine-tune calibration on inner validation;
- freeze all decisions before outer-fold evaluation.

### Required paper tables

- dataset and reviewed-label distribution by source and behavior;
- fold composition and leakage audit;
- main native-unit comparison table with confidence intervals;
- per-class precision/recall/F1;
- source-specific and source-balanced results;
- context-ready interaction results;
- ablation table by modality and loss;
- calibration and computational-cost table;
- reviewer agreement and active-review accounting;
- failure taxonomy with representative review units.

### Required paper figures

- end-to-end audited data/model pipeline;
- multimodal architecture;
- grouped validation protocol;
- confusion matrices for baseline and proposed model;
- per-class paired improvement plot;
- calibration plot;
- interaction-context qualitative examples with actor and partner boxes;
- error clusters by source, context quality, and behavior pair.

### Non-model paper requirements

- Freeze a protocol document before confirmatory S6 runs: primary endpoint, fold algorithm, candidate models, seeds, ablations, exclusion rules, and statistical tests.
- Maintain a literature matrix covering pig behavior datasets, image/video baselines, temporal models, social-context models, ROI-aware methods, and leakage-safe validation practice.
- Publish or archive a dataset card with source provenance, annotation grain, review process, class/source distribution, known ambiguity, missing-context policy, and intended use.
- Document behavior definitions and actor/group annotation rules sufficiently for an independent reviewer.
- Report reviewer count, double-review fraction, agreement statistic, adjudication rule, and decision counts.
- Include animal-welfare/ethics approval, data ownership, and recording-permission statements where applicable; do not invent missing approval identifiers.
- Include hardware, software versions, seed policy, runtime, parameter count, and model-selection budget.
- State whether code, derived manifests, weights, and restricted data can be released, with a reproducibility path when raw videos cannot be shared.
- Distinguish exploratory analyses from confirmatory analyses in the manuscript.

### Paper claim gate

PASS for a strong Q2-oriented submission requires:

- reviewed immutable dataset snapshot;
- zero demonstrated train/test group leakage;
- grouped out-of-fold native-unit results;
- statistically defensible paired comparison;
- source/domain controls;
- modality and loss ablations;
- context-ready interaction analysis;
- reproducible config, code SHA, predictions, and registry entries;
- explicit limitations about external generalization and biological identity.

Q1-style external generalization remains BLOCKED until a genuinely external farm, camera setup, and/or cohort is collected and evaluated without adapting to its test labels.

## 18. Global PASS/FAIL Dashboard

| Criterion | Current status | PASS evidence required |
|---|---|---|
| S0A Snapshot and alignment | NOT STARTED | Hashes, key digests, whitelist, zero alignment errors |
| S0B Spatio-temporal semantics | NOT STARTED | Hand-calculated fixtures, inference availability, row preservation |
| S1 Scene/partner context | NOT STARTED | Label-independent loader/model smoke and context audit |
| S2 Multitask loss | NOT STARTED | Mask/gradient/overfit checks and no target leakage |
| S3 Training system | NOT STARTED | Reproducible config run, checkpoint, registry, strict whitelist |
| S4 Q2 folds and metrics | NOT STARTED | Zero grouped leakage and one OOF prediction per native unit |
| S5 Domain controls | PARTIAL IN CODE | Native probes pass; candidate evidence pending |
| S6 Controlled experiments | BLOCKED BY S0A-S5 | Baseline/ablation OOF evidence with uncertainty |
| S7 Active review | BLOCKED BY OOF PREDICTIONS | Train-pool shortlist, review audit, new snapshot |
| S8 Advanced branches | EVIDENCE-GATED | Error-driven ablation improvement on intended slices |
| S9 Unified model/paper | LONG TERM | Full reproducibility and Q2 paper claim gate |
| External Q1 generalization | BLOCKED | External farm/camera/cohort test data |

## 19. Mandatory Data and Review Regression Gates

Run these gates after any change to parsing, temporal harmonization, spatial export, image resolution, review units, review GUI, decision application, or sequence rebuilding.

### CVAT anchor gate

For `Pigs281119_000085_30fps`, pig `ID_4`, anchor `1020`:

- enhanced frame behavior is `social-nose`;
- temporal interval `1020..1025` final behavior is `social-nose`;
- review unit behavior is `social-nose`;
- review template is `interaction`;
- no downstream module rewrites it as `stand` or routes it to motion review.

### GUI video resolver gate

For `video_key=Pigs291119_000231`, pig `ID_4`, frames `678..683`:

- resolve `data/videos/Pigs291119_000231_30fps.mp4`;
- read all requested frames through OpenCV;
- obtain valid bboxes and nonblank actor crops;
- render the interaction/full-frame view without `missing_video_or_crop`;
- preserve case-insensitive extension and recursive-search fallback behavior.

### Review policy gate

- ROI: eat, drink, playwithtoy.
- Interaction: fight, social-nose.
- Motion/context: explore, move, stand.
- Posture: lying, sitting only.
- `playwithtoy` remains in the full review manifest.
- No duplicate `review_unit_id`.
- No new output contains `window_uid`.

### Decision and apply gate

- All four decision CSVs retain the canonical schema.
- Pending decisions do not apply corrected labels.
- Accept keeps behavior, corrected changes the whole review unit, exclude preserves rows with zero weight and false include mask.
- Duplicate decisions fail clearly or follow an explicitly audited deterministic policy.
- `reviewed_frame_features` row count equals enhanced frame-feature row count.
- `behavior_before_review` or an equivalent before/after audit remains available.

### Reviewed sequence gate

- Rebuild lengths 6, 8, 12, and 16 from `reviewed_frame_features.csv`.
- Stable, mixed, and transition windows remain counted and flagged.
- Invalid/excluded windows are masked, not silently deleted.
- Manifest, feature table, and audit row/key counts agree.
- Train-ready feature whitelist remains unchanged unless its contract version is explicitly incremented.

Any failure here blocks S3-S9 even when a model smoke test passes.

## 20. Global Failure Conditions

Stop and mark the current criterion FAIL if any of these occurs:

- row counts change without an explicit row-level audit;
- a key is duplicated and resolved by keeping the first/last row silently;
- a ground-truth label controls feature loading or model branch routing;
- a test fold is used for thresholding, early stopping, shortlist selection, or manual correction;
- source, path, ID, review, policy, or label columns enter X;
- full model claims rely on window-level samples as independent observations;
- results are reported only pooled while one source or rare class collapses;
- the publication model is selected after repeated inspection of the same held-out date;
- auxiliary targets are presented as independent annotation;
- external generalization is claimed without external data.

## 21. Plan Mutation Protocol

This roadmap is evidence-driven and may change, but changes must be auditable:

1. Record the proposed change, reason, evidence, affected dependencies, and claim impact.
2. Never weaken a leakage or row-preservation invariant to unblock implementation.
3. Split a criterion when it begins changing unrelated modules or cannot be reviewed in one commit.
4. Insert a new criterion when a failed gate reveals a missing prerequisite.
5. Mark a criterion skipped only with a written scientific justification and downstream claim reduction.
6. Update the dependency graph and PASS/FAIL dashboard in the same commit.
7. Preserve rejected experiment configs and concise audit results; do not silently overwrite them as successful runs.

## 22. Recommended Immediate Execution Order

1. S0A training snapshot and key-alignment contract.
2. S0B spatio-temporal semantic fixtures and prediction-time availability audit.
3. S1 label-independent scene/partner index, loader, and GUI context rendering.
4. S2 multitask forward and masked-loss smoke.
5. S3 strict config-driven trainer integration.
6. S4 five-fold grouped publication protocol and native-unit metrics.
7. S5 source/domain control views and reports.
8. Request explicit approval and compute budget before S6 full controlled training.
9. Use S6 out-of-fold errors to decide whether S7 or a specific S8 branch is justified.
10. Enter S9 only after ablations prove that each retained branch contributes reproducibly.

The first implementation criterion should be S0A. It reduces the risk that later model improvements are artifacts of row misalignment, stale outputs, or uncontrolled dataset changes.

## 23. Criterion Command Gates

All commands run from CMD with the required project environment:

```bat
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src
```

### S0A

```bat
rtk python -m py_compile src\pig_behavior\classification_v2\contracts\training_snapshot.py src\pig_behavior\classification_v2\contracts\model_io.py scripts\behavior_review_tools\classification_v2_freeze_training_snapshot.py scripts\dev_tools\check_classification_v2_training_snapshot.py
rtk python scripts\behavior_review_tools\classification_v2_freeze_training_snapshot.py --config configs\classification_v2\data_contract_v1.json
rtk python scripts\dev_tools\check_classification_v2_training_snapshot.py
```

### S0B

```bat
rtk python -m py_compile src\pig_behavior\classification_v2\features\geometry.py src\pig_behavior\classification_v2\features\motion.py src\pig_behavior\classification_v2\features\roi.py src\pig_behavior\classification_v2\features\social.py src\pig_behavior\classification_v2\features\spatiotemporal.py src\pig_behavior\classification_v2\spatial_sequence_export.py scripts\dev_tools\check_classification_v2_spatiotemporal_feature_semantics.py
rtk python scripts\dev_tools\check_classification_v2_spatiotemporal_feature_semantics.py
rtk python scripts\behavior_review_tools\classification_v2_export_spatial_sequences.py
rtk python scripts\dev_tools\check_classification_v2_spatial_sequences.py
```

After S0B and S1 artifacts stabilize, create a new immutable candidate snapshot:

```bat
rtk python scripts\behavior_review_tools\classification_v2_freeze_training_snapshot.py --config configs\classification_v2\data_contract_v1.json --snapshot-name candidate_multimodal_v1
rtk python scripts\dev_tools\check_classification_v2_training_snapshot.py --snapshot-name candidate_multimodal_v1
```

### S1

```bat
rtk python -m py_compile src\pig_behavior\classification_v2\datasets\scene_partner_index.py src\pig_behavior\classification_v2\datasets\scene_partner_dataset.py src\pig_behavior\classification_v2\models\interaction_context_encoder.py src\pig_behavior\classification_v2\review\context_renderer.py scripts\behavior_review_tools\classification_v2_build_scene_partner_index.py scripts\dev_tools\check_classification_v2_scene_partner_index.py scripts\dev_tools\check_classification_v2_scene_partner_loader.py scripts\dev_tools\check_classification_v2_interaction_context_forward.py scripts\dev_tools\check_classification_v2_review_context_rendering.py
rtk python scripts\behavior_review_tools\classification_v2_build_scene_partner_index.py
rtk python scripts\dev_tools\check_classification_v2_scene_partner_index.py
rtk python scripts\dev_tools\check_classification_v2_scene_partner_loader.py
rtk python scripts\dev_tools\check_classification_v2_interaction_context_forward.py
rtk python scripts\dev_tools\check_classification_v2_review_context_rendering.py
```

### S2

```bat
rtk python -m py_compile src\pig_behavior\classification_v2\models\multitask_fusion.py src\pig_behavior\classification_v2\training\multitask_loss.py scripts\behavior_review_tools\classification_v2_multitask_smoke_train.py scripts\dev_tools\check_classification_v2_multitask_forward.py scripts\dev_tools\check_classification_v2_multitask_loss.py scripts\dev_tools\check_classification_v2_multitask_smoke_train.py
rtk python scripts\dev_tools\check_classification_v2_multitask_forward.py
rtk python scripts\dev_tools\check_classification_v2_multitask_loss.py
rtk python scripts\behavior_review_tools\classification_v2_multitask_smoke_train.py --steps 8 --per-class-train 1 --per-class-eval 1
rtk python scripts\dev_tools\check_classification_v2_multitask_smoke_train.py
```

### S3

```bat
rtk python -m py_compile src\pig_behavior\classification_v2\training\config.py src\pig_behavior\classification_v2\training\data_module.py src\pig_behavior\classification_v2\training\trainer.py src\pig_behavior\classification_v2\training\checkpoint.py scripts\behavior_review_tools\classification_v2_train.py scripts\dev_tools\check_classification_v2_training_config.py scripts\dev_tools\check_classification_v2_training_reproducibility.py
rtk python scripts\dev_tools\check_classification_v2_training_config.py --config configs\classification_v2\multimodal_context_multitask.json
rtk python scripts\behavior_review_tools\classification_v2_train.py --config configs\classification_v2\multimodal_context_multitask.json --fold 0 --seed 123 --smoke
rtk python scripts\dev_tools\check_classification_v2_training_reproducibility.py
```

### S4

```bat
rtk python -m py_compile src\pig_behavior\classification_v2\evaluation\grouped_folds.py src\pig_behavior\classification_v2\evaluation\native_unit_metrics.py src\pig_behavior\classification_v2\evaluation\statistics.py scripts\behavior_review_tools\classification_v2_build_q2_folds.py scripts\behavior_review_tools\classification_v2_evaluate_oof_predictions.py scripts\dev_tools\check_classification_v2_q2_folds.py scripts\dev_tools\check_classification_v2_oof_metrics.py
rtk python scripts\behavior_review_tools\classification_v2_build_q2_folds.py --outer-folds 5 --group-level recording_date
rtk python scripts\dev_tools\check_classification_v2_q2_folds.py
rtk python scripts\dev_tools\check_classification_v2_oof_metrics.py --fixture-only
```

### S5

```bat
rtk python -m py_compile src\pig_behavior\classification_v2\evaluation\domain_controls.py src\pig_behavior\classification_v2\training\samplers.py scripts\behavior_review_tools\classification_v2_build_source_matched_views.py scripts\behavior_review_tools\classification_v2_evaluate_domain_controls.py scripts\dev_tools\check_classification_v2_source_matched_views.py scripts\dev_tools\check_classification_v2_domain_controls.py
rtk python scripts\behavior_review_tools\classification_v2_build_source_matched_views.py
rtk python scripts\dev_tools\check_classification_v2_source_matched_views.py
rtk python scripts\dev_tools\check_classification_v2_domain_controls.py --fixture-only
```

### S6

Before any full run, execute only a one-fold engineering smoke:

```bat
rtk python scripts\behavior_review_tools\classification_v2_train.py --config configs\classification_v2\multimodal_context_multitask.json --fold 0 --seed 123 --smoke
```

The all-fold command is intentionally not a default gate. Record explicit user approval, compute budget, frozen snapshot, protocol hash, and model matrix before running it.

### S7

```bat
rtk python -m py_compile src\pig_behavior\classification_v2\review\active_learning.py scripts\behavior_review_tools\classification_v2_build_active_review_shortlist.py scripts\dev_tools\check_classification_v2_active_review_shortlist.py
rtk python scripts\behavior_review_tools\classification_v2_build_active_review_shortlist.py --pool-split train --unit-key review_unit_id
rtk python scripts\dev_tools\check_classification_v2_active_review_shortlist.py
```

### S8

Run only the checker for the specific branch justified by S6 error analysis. Example graph branch:

```bat
rtk python -m py_compile src\pig_behavior\classification_v2\models\social_graph.py src\pig_behavior\classification_v2\models\temporal_graph_encoder.py scripts\dev_tools\check_classification_v2_social_graph_forward.py
rtk python scripts\dev_tools\check_classification_v2_social_graph_forward.py
```

### Mandatory pipeline regressions

```bat
rtk python scripts\dev_tools\diagnose_cvat_unit_label_mismatch.py
rtk python scripts\dev_tools\diagnose_gui_video_loading.py
rtk python scripts\dev_tools\check_review_unit_template_coverage.py
rtk python scripts\dev_tools\check_review_unit_gui_decisions.py
rtk python scripts\dev_tools\check_apply_review_unit_decisions_output.py
rtk python scripts\dev_tools\check_classification_v2_train_ready_windows.py
rtk python scripts\dev_tools\check_classification_v2_native_temporal_units.py
```

### Final change gate for every criterion

```bat
rtk git diff --check
rtk git status --short
```

## 24. Adversarial Review Record

The roadmap was challenged against the main ways a multimodal behavior paper could produce an optimistic but invalid result.

| Finding | Severity | Resolution in this roadmap |
|---|---|---|
| Loading partner context only for known interaction labels leaks y | Critical | S1 builds context for every loadable window and gates only by asset/geometry availability |
| One fixed test date is split-sensitive | Critical | S4 requires grouped out-of-fold native-unit evaluation plus leave-one-date sensitivity |
| Source is almost perfectly predictable from clean features | Critical | S5 requires per-source, source-balanced, matched-length, and learned-embedding probes |
| Annotation `Hidden` may not exist at inference | Critical | S0B requires prediction-time availability classification and exclusion/replacement of annotation-only fields |
| Freezing before adding new feature/context artifacts makes the snapshot stale | High | S0A freezes an immutable baseline and then creates a distinct post-S0B/S1 candidate snapshot |
| Overlapping windows inflate effective sample size | High | Event weights are training-only; claims use one out-of-fold prediction per native unit |
| Active review can contaminate the test set | High | S7 selects only from training/development pools and invalidates any inspected test fold |
| Auxiliary heads are deterministic functions of behavior y | Medium | S2 frames them as inductive bias and mandates behavior-only ablation |
| Missing legacy context can become a source shortcut | High | S1 preserves masks and rows; S5 probes availability masks and tests modality dropout |
| Rare `playwithtoy` support can disappear inside folds | High | S4 fixes the global label order, reports support, and audits fold feasibility |
| ROI/location cues can encode camera/session identity | High | S5 adds ROI-only and geometry-only probes and forbids camera-safe claims without metadata |
| Complex long-term model can hide which modality helps | High | S6 requires a fixed baseline and modality/loss ablation matrix before S9 |

No unresolved critical design finding remains in the plan. Empirical unknowns remain intentionally open: hardware budget, achievable fold class coverage, reviewer agreement, and the actual effect size of each model branch.

## 25. Self-Evaluation

| Axis | Score | Evidence and remaining gap |
|---|---:|---|
| Accuracy | 4.5/5 | Current row counts, split structure, interaction readiness, and source-shortcut values are taken from local audit artifacts. Planned CLI options are contracts to implement, not claims that the commands already exist. |
| Completeness | 4.5/5 | Covers reviewed data, spatial semantics, RGB/context loading, multitask learning, leakage-safe folds, domain controls, active review, ablations, long-term model, and paper requirements. External validation data and ethics identifiers cannot be supplied from current evidence. |
| Clarity | 4.0/5 | Dependency graph, milestone dashboard, and command gates provide navigation. The document is deliberately long because each milestone must be executable after context loss. |
| Actionability | 4.5/5 | Every criterion names modules, scripts, artifacts, checks, PASS/FAIL conditions, dependencies, and commit/rollback rules. Numerical promotion tolerance remains to be frozen from baseline variance before S6. |
| Conciseness | 3.5/5 | Detail is high and some global invariants recur inside milestones. The repetition is retained where it prevents leakage or silent data loss during multi-session execution. |

Overall self-evaluation: `4.2/5`.

Highest-impact remaining decisions before S6:

1. Declare available GPU/CPU memory and maximum experiment budget.
2. Freeze the practical minimum effect size after baseline repeated-seed variance is known.
3. Confirm camera/session metadata and whether an independent second reviewer is available.
