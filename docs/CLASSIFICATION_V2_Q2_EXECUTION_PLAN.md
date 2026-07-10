# Classification V2 Q2 Execution Plan

## 1. Scope and Claim Boundary

This plan turns the roadmap and protocol into the next executable checklist for
`classification_v2`. The target framework is multimodal spatio-temporal behavior
recognition with actor bbox image sequences, ROI all-class relations, social
partner context, whitelisted spatial features, and native temporal-unit
evaluation.

Allowed claim after all gates pass:

- Q2-strong in-domain claim: improved pig behavior recognition under
  session/video-safe native temporal validation.

Not allowed without new evidence:

- Q1-style cross-farm, cross-camera, cross-cohort, or unseen-animal
  generalization.
- Treating `pig_id` as the same biological pig across videos.
- Treating overlapping windows as independent publication samples.
- Treating pilot/smoke results as paper-facing learned model results.

## 2. Current State

Completed engineering evidence:

- Review-to-train data lineage preserves rows from reviewed frame features to
  sequence windows and train-ready artifacts.
- Feature input uses a whitelist and denies review/manual/source/path/ID/label
  columns from model X.
- Native OOF folds exist for 13 recording groups.
- B0/B1/B2 native OOF baselines are recorded.
- Learned multimodal OOF runner has a bounded pilot, prediction schema checks,
  registry record, and resumable per-fold artifact support.

Current blocking condition:

- Full learned native OOF multimodal evaluation is not recorded yet at
  `outputs/classification_v2/experiment_registry/full_multimodal_oof_record.json`.

## 3. End-to-End Data Logic

The pipeline must keep these units distinct:

| Unit | Meaning | Used for |
|---|---|---|
| `frame_uid` | One pig/object observation at one frame | feature/image lookup |
| `review_unit_id` | Human review unit, CVAT 6f or legacy 16f | label decision lineage |
| `temporal_unit_key` | Native annotation interval identity | primary evaluation unit |
| `window_id` | Training sequence window | model input join key |
| `recording_group_id` | session/video/date group | leakage-safe OOF split |

Rules:

- CVAT anchor `k` labels frames `k..k+5`.
- Legacy unit labels the full 16-frame burst.
- Review decisions apply to the full review unit, not to a single frame only.
- `exclude` never drops rows; it sets mask/weight/action.
- `pending` never applies `manual_corrected_behavior`.
- Every model prediction used for paper must collapse to one native temporal
  unit prediction before metrics.

## 4. Spatial and Image Feature Plan

### 4.1 Geometry and Motion

Keep model-usable geometry and motion as normalized, label-free signals:

- bbox center, width, height, area, aspect ratio.
- bbox deltas, scale deltas, velocity, acceleration, path length, net
  displacement, tortuosity, stationary ratio, turn angle, and frame gaps.
- bbox validity, truncation, hidden/interpolation, and missing-frame quality
  masks.

PASS:

- No cross-video or cross-track feature computation.
- Missing frames are masked, not silently assumed continuous.
- Padding values cannot change model logits when masks are applied.

FAIL:

- Raw absolute camera coordinates are used without normalization or audit.
- Label/review columns influence any motion feature.

### 4.2 ROI Relations

Model input must use all-class ROI relations, not target ROI selected by label:

- feeder, drinker, and toy distance/inside/overlap/near/contact channels.
- approach speed/angle, dwell ratio, entry/exit count per ROI class.
- `roi_available_mask` and quality flags.

PASS:

- `target_roi_*` remains audit/policy-only.
- Missing ROI never deletes a sample.
- `playwithtoy` has toy ROI logic represented without label leakage.

FAIL:

- Model X receives `target_roi_*`, `behavior_label`, or policy text.
- ROI missing samples are dropped silently.

### 4.3 Social Context

Social/interaction context is geometry-based:

- top-k neighbor relations, relative position, distance, IoU/contact, relative
  velocity, approach/separation speed, density, and partner validity masks.
- full-frame or partner context is required for interaction review/training
  analysis.

PASS:

- `social-nose` remains actor-only.
- `fight` applies only to directly involved pigs.
- bystanders are not propagated as fight labels.

FAIL:

- Partner is selected using ground-truth behavior.
- Interaction context crosses video/frame boundaries.

### 4.4 Image Views

Image sequence input should be built from label-independent views:

- `actor_view`: bbox crop with stable expansion.
- `local_context_view`: larger crop to include nearby ROI/partner context.
- optional `scene_view`: low-resolution full frame with actor marker/mask.

PASS:

- CVAT loads from video+bbox, legacy loads from crop resolver.
- `Pigs291119_000231` resolves `_30fps.mp4` and frame `678..683`.
- Letterbox/normalization policy is recorded in config/audit.

FAIL:

- Missing image is silently replaced by blank tensor without audit.
- Crops are chosen based on target behavior label.

## 5. Model and Training Architecture

The model family should be built as fair, ablatable branches:

| Branch | Input | Purpose |
|---|---|---|
| B0 | class prior | sanity floor |
| B1 | whitelisted tabular linear | linear baseline |
| B2 | whitelisted tabular nonlinear | strong tabular control |
| B3 | actor image sequence | appearance/posture signal |
| B4 | actor image + temporal encoder | temporal image signal |
| B5 | spatial masked TCN | geometry/motion/ROI/social signal |
| P1 | actor image + spatial + tabular fusion | primary proposed model |
| P2 | P1 + local/full context | ROI/interaction hypothesis |
| P3 | P2 + auxiliary heads | posture/motion/ROI multitask test |
| P4 | P2 + social graph | long-term interaction extension |

Training rules:

- Split/fold manifest is read, never generated ad hoc by trainer.
- Scalers, clipping, class weights, calibration, and thresholds are fit inside
  each outer-train fold only.
- Sample weights combine review include mask, review/sample weight, class policy,
  and event/window de-duplication policy.
- Primary metric is native temporal-unit macro-F1 with recording-cluster
  uncertainty, not raw window accuracy.

## 6. Immediate Execution Checklist

### S31. Execution Plan and Full OOF Readiness

Deliverables:

- This execution plan.
- Clean preflight audits:
  - `check_classification_v2_full_learned_oof_contract.py`
  - `check_classification_v2_ablation_shortcut_contract.py`
  - `check_classification_v2_full_multimodal_oof.py` for pilot.

PASS:

- Contract valid.
- Only blocker is missing full learned OOF record.
- Worktree is clean before full run.

### S32. Full Learned Native OOF Run

Command:

```bat
python scripts\behavior_review_tools\classification_v2_run_full_multimodal_oof.py ^
  --full ^
  --output-dir outputs\classification_v2\model_full\full_multimodal_oof ^
  --image-size 64 ^
  --hidden-dim 48 ^
  --steps-per-fold 6 ^
  --train-batch-size 32 ^
  --eval-batch-size 64 ^
  --bootstrap-iterations 200
```

PASS:

- `run_mode=full`.
- All planned folds produce fold artifacts.
- Predictions and native temporal metrics pass schema checks.
- No forbidden leakage columns appear in prediction schema.

FAIL:

- A reduced/pilot run is registered as full paper evidence.
- A fold failure is hidden by silently dropping that fold.

### S33. Full Record Registration

Register only after S32 post-run gates pass:

```bat
python scripts\behavior_review_tools\classification_v2_register_experiment.py ^
  --name full_multimodal_oof ^
  --metrics-json outputs\classification_v2\model_full\full_multimodal_oof\full_multimodal_oof_metrics.json ^
  --artifact outputs\classification_v2\model_full\full_multimodal_oof\full_multimodal_oof_audit.json ^
  --artifact outputs\classification_v2\model_full\full_multimodal_oof\full_multimodal_oof_prediction_schema_audit.json ^
  --notes "full learned multimodal native OOF evaluation" ^
  --experiment-stage paper_facing_candidate ^
  --paper-facing ^
  --result-kind model_evaluation
```

PASS:

- Registry record has `git_dirty=false`.
- Native temporal metrics gate is valid.
- `external_generalization_claim=false`.

### S34. Claim Gate Recheck

Run:

- full learned OOF contract checker.
- ablation/shortcut contract checker.
- paper-grade protocol checker.

PASS:

- Missing-full-OOF blocker is removed.
- Remaining warnings are scientific caveats, not data leakage or missing primary
  prediction schema.

## 7. Next Scientific Upgrades After Full OOF

Priority A, before any manuscript claim:

- Full source/session/behavior slice metrics.
- Paired comparison against B0/B1/B2 with recording-cluster CI.
- Confusion-focused report for fight/social-nose, eat/drink/explore/stand,
  playwithtoy/explore/stand/move, lying/sitting, and move/explore/stand.
- Calibration and confidence/error analysis.
- Reviewer coverage and inter-rater plan for rare/confusing classes.

Priority B, model improvement:

- E1 spatial masked TCN full OOF.
- E3 actor-image-only OOF.
- E4 actor plus local-context OOF.
- E5 fusion ablations: minus ROI, minus social, minus motion, image-only,
  spatial-only.
- Source shortcut controls from learned embeddings.

Priority C, long-term research:

- Multi-task heads for posture, motion, ROI-intent, and interaction group.
- Graph social branch with top-k neighbor edges and bystander FP audit.
- Pose/keypoint branch if reliable keypoints can be generated.
- Hard-negative mining from reviewed confusion clusters.
- Active learning loop using uncertainty and disagreement.

## 8. Commit and Artifact Policy

Commit after each completed criterion:

- Source code, configs, docs, small JSON audits, and registry records.
- Do not force-add large CSV, NPZ, tensor cache, checkpoint, or model artifacts
  unless explicitly required.
- For large artifacts, record path, size, checksum, command, commit SHA, and
  dirty state in JSON audit/registry.

Every added module/script should include docstrings or comments for:

- why a unit/key exists,
- how leakage is prevented,
- how masks/weights affect training,
- why a decision is deterministic,
- what is audit-only versus model input.

## 9. PASS/FAIL Summary

Overall PASS for the next goal requires:

- Full learned native OOF record exists and validates.
- Reviewed/train-ready data gates still preserve rows and forbid leakage.
- Metrics are native temporal-unit metrics, not independent window metrics.
- Claims remain Q2 in-domain and do not imply external generalization.
- Follow-up ablations and shortcut controls are planned from the full OOF
  result, not retrofitted to justify a preferred outcome.

Current status at the time of this plan:

- PASS: engineering pipeline, train-ready contracts, pilot learned OOF, resumable
  fold artifacts, pilot registry.
- FAIL/blocked for paper-facing model claim: full learned OOF record missing.
