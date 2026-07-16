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

Current active lineage:

- A 245,664-row enhanced technical reference exists, but it is not the clean
  review authority. The new lineage rebuilds from immutable source inputs under
  `outputs/classification_v2/human_review_runs/<RUN_ID>`.
- The v6 reference design has 5,131 target-independent Hidden items, but its
  carried payload is unverified. Clean human coverage starts at 0/5,131.
- Clean behavior coverage starts at 0/4,670 user-verified units.
- Hidden apply, reviewed temporal rebuild, behavior apply, and a frozen
  train-ready snapshot are therefore blocked.

Historical engineering evidence:

- A 13-fold full OOF run exists at commit `18d6692` with 73,668 window and
  32,727 native-unit predictions.
- It belongs to the previous unreviewed lineage and is not the final Q2 result.
- Old preflight, launch, and progress reports are not valid authorization for
  the active rebuild.

Use `CLASSIFICATION_V2_CURRENT_STATE.md` for the current PASS/FAIL matrix. The
next executable stage is human Hidden review, not model postrun promotion.

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

### S31. Complete Hidden v5 Review

Deliverables:

- Complete decision CSV for all 5,171 v5 review items.
- Media-resolution and decision-schema audits.
- Separate counts for trusted-Yes audit, untrusted-Yes census, high-risk No,
  random No, and clean controls.

PASS:

- Every mandatory item has one resolved decision.
- No unresolved pending row or contradictory decision payload remains.
- Decision apply preserves source rows and writes a new derived artifact.

### S32. Apply Hidden and Rebuild Temporal Artifacts

Use the data rebuild runbook commands for the versioned v5 lineage. Temporal
harmonization must consume the Hidden-reviewed frame artifact, never the
pre-review enhanced CSV.

PASS:

- Hidden apply has equal input/output row counts and explicit action counts.
- CVAT anchors still represent `k..k+5`; legacy bursts still contain 16 frames.
- Rebuilt intervals and windows have deterministic unique keys.
- No row, label, or trust state changes without an audit record.

FAIL:

- Temporal windows are reused from a different Hidden lineage.
- An excluded review item is physically dropped instead of masked/audited.
- Temporal harmonization runs before Hidden apply.

### S33. Complete Behavior Review and Apply

Rebuild behavior review units from the v5 Hidden-reviewed temporal lineage.
Complete all required review decisions before apply; the existing 3-row
decision payload is evidence of an incomplete review, not a reusable result.

PASS:

- Required and decided `review_unit_id` sets match exactly.
- No pending decision carries active correction or exclusion payload.
- Apply preserves frame rows and records corrected/excluded counts.
- Reviewed labels remain within the canonical 10-class vocabulary.

### S34. Freeze Train-Ready Lineage and Run Bounded Smokes

Rebuild reviewed windows, native units, X/y/masks/weights, image indexes,
feature whitelist, and recording-safe folds. Freeze their hashes before any
model comparison.

PASS:

- Snapshot, cache, whitelist, config, and fold hashes are mutually linked.
- One-batch, tiny-overfit, resume, runtime, and one-fold tests pass.
- Source, sequence length, padding, and context availability shortcut audits
  have explicit results and mitigation where required.

### S35. Authorize Future Finalist OOF

Only after S34 passes, generate a new run ID, output directory, preflight, and
human authorization bound to the frozen lineage and clean code SHA. Use the
future full-OOF runbook; never overwrite or promote the commit-`18d6692`
artifact.

## 7. Scientific Upgrades After the Reviewed Baseline

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

Overall PASS for the active data-readiness goal requires:

- Hidden and behavior decision coverage is complete and fail-closed.
- Reviewed/train-ready data gates preserve rows and forbid leakage.
- Snapshot, cache, whitelist, and fold hashes are frozen together.
- All bounded model smoke gates pass before full-run authorization.
- Claims remain Q2 in-domain and do not imply external generalization.

Current status at the time of this plan:

- PASS: enhanced rows and Hidden v5 template coverage.
- FAIL: complete Hidden decisions and complete behavior decisions.
- BLOCKED: reviewed snapshot, matching caches/folds, model smokes, and a new
  reviewed-lineage full OOF.
- Historical only: the commit-`18d6692` OOF run and its lineage-local reports.
