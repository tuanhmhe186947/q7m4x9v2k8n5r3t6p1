# Classification V2 Scientific Performance Upgrade Roadmap

Version: 2.1-working

Date: 2026-07-13

Status: extended research reference; not the core classifier critical path.

Scope: `classification_v2` data, model, validation, and publication workflow.

This plan supersedes the model-training direction in
`classification_v2-q2-multimodal-roadmap.md`. The older document remains the
historical record of data and audit construction. This plan does not authorize
a long OOF run and never changes raw files under `data/`.

Use `classification_v2-core-classifier-roadmap.md` as the canonical P0-P8
execution plan. Five-class comparison, paper reproduction, publication, and
integration work in this document are optional Phase 9 activities.

## 1. Objective And Claim Boundary

Primary objective:

> Improve 10-class pig behavior recognition under recording-date and video-safe
> validation using actor appearance, bbox dynamics, ROI relations, social
> context, and temporally supervised hierarchical behavior representations.

Secondary objective:

> Build a strict five-class branch for `standing`, `lying`, `moving`, `eating`,
> and `drinking`, evaluated on the same native temporal units and grouped folds
> as both the old-compatible and modern models.

The intended paper claim is Q2-strong internal improvement. Do not claim
external farm, camera, cohort, or biological-individual generalization without
an independent external dataset. `pig_id` remains annotation-local.

### 1.1 Three Distinct Scientific Tracks

1. Main thesis track: 10-class session-safe learned recognition.
2. Comparability track: strict five-class ResNet18, ResNet34, and enhanced-model
   experiments under the same thesis protocol.
3. Optional reproduction track: paper-aligned heuristic movement and ROI logic
   plus a binary ResNet18 lying-standing branch.

Never combine results from these tracks into one model claim. The strict-5
learned classifier is a paper-aligned label-space comparison, not a reproduction
of the VISAPP pipeline.

## 2. Current Evidence And Feasibility

Current native-unit OOF evidence has 32,727 predictions:

| Metric | Current value |
|---|---:|
| Accuracy | 0.5217 |
| Macro-F1 | 0.4156 |
| Macro recall | 0.4518 |
| 95% cluster-bootstrap CI for macro-F1 | 0.3840 to 0.4614 |

The current model is an engineering baseline, not a capacity-matched successor
to the old checkpoint. It uses a three-layer CNN, `64x64` actor images, a small
temporal convolution, and about 108,653 trainable parameters. The old checkpoint
is consistent with ResNet34, CBAM, a two-layer Transformer, `224x224` images,
and about 24.0 million parameters. It is roughly 221 times larger.

The upgrade is feasible on the RTX 3050 Laptop GPU if experiments are staged:

- begin with pretrained ResNet18 at `160x160`, AMP, and bounded fold pilots;
- benchmark memory and throughput before ResNet34 or `224x224` experiments;
- use reusable letterboxed packed caches at each approved resolution;
- never launch all 13 folds before overfit, one-fold, and runtime gates pass.

Higher scores are a research target, not a guaranteed outcome. A gain is valid
only if it survives the same grouped folds, native-unit aggregation, paired
uncertainty analysis, leakage checks, and predeclared promotion thresholds.

## 3. Current Failure Hypotheses

1. Visual capacity and resolution are too low for posture and head-object cues.
2. Actor-only evidence is insufficient for `fight` and `social-nose`.
3. Window imbalance overcounts overlapping windows from the same event.
4. Current auxiliary targets are decompositions of final labels, not independent
   frame-level attributes, so they regularize but do not add new supervision.
5. `lying` versus `sitting`, `move` versus `explore`, and interaction pairs need
   explicit confusion-focused data and losses.
6. Source is highly predictable from current features, creating domain shortcuts.

## 4. Scientific Invariants

1. Preserve every row and record masks or exclusions; never silently drop data.
2. Derive weights, priors, normalization, thresholds, and calibration from each
   training fold only.
3. Evaluate overlapping windows diagnostically, but use native temporal units
   and recording groups as the scientific units.
4. Keep `manual_*`, `review_*`, identifiers, paths, labels, and policy text out
   of model X. Never select all numeric columns automatically.
5. Build actor, scene, partner, ROI, and social routing without consulting y.
6. Do not use annotation-only `Hidden` as a deployable model feature unless an
   equivalent inference-time visibility signal is defined and tested.
7. Keep the final 10-class head directly supervised in every hierarchy model.
8. Do not hard-propagate `fight` to bystanders or `social-nose` to partners.
9. Treat `pig_id` as local to its annotation video or session.
10. Record dataset hash, code SHA, config hash, seed, fold, and cache lineage.
11. Use the same fold manifest for every model in a paired comparison.
12. Do not tune architecture or thresholds on the final outer-fold predictions.

## 5. Hierarchical Behavior Ontology

The hierarchy is multi-attribute, not a single tree. Complex behaviors can
share posture and motion states, so one exclusive parent would be biologically
incorrect. Each head has an `unknown/not-observed` mask when evidence is absent.

| Head | Labels | Evidence and purpose |
|---|---|---|
| Posture | recumbent, sitting, upright, unknown | body shape and pose |
| Locomotion | stationary, moving, transition, unknown | trajectory over time |
| ROI state | feeder, drinker, toy, none, unknown | contact and proximity |
| Social state | contact, approach, separation, none, unknown | partner relations |
| Interaction | fight, social-nose, none, unknown | actor and partner context |
| Final behavior | canonical 10 labels | direct task supervision |

### 5.1 Supervision Policy

Canonical labels provide only partial auxiliary supervision:

- `lying` strongly supervises recumbent posture;
- `sitting` strongly supervises sitting posture;
- `stand` supports upright and mostly stationary, subject to review;
- `move` supports moving, but does not define posture;
- `eat`, `drink`, and `playwithtoy` supervise ROI intent, not guaranteed contact;
- `fight` and `social-nose` supervise interaction type, not one posture state;
- `explore` is not deterministically equal to moving or standing.

Therefore, generate a target plus confidence and mask for every head. Do not
invent hard labels where the canonical behavior is insufficient. Derived labels
are weak or auxiliary targets and must be identified as such in reports.

### 5.2 Recommended Hierarchy Mechanism

Use a shared encoder with masked auxiliary heads and soft gated fusion:

```text
actor + spatial + ROI + social + scene tokens
                  |
           shared temporal encoder
                  |
     +------------+-----------+-----------+
     |            |           |           |
 posture      locomotion   ROI/social   interaction
     +------------+-----------+-----------+
                  |
       soft attribute embeddings
                  |
          final 10-class head
```

The final head consumes shared features and optional soft attribute embeddings.
It never consumes argmax auxiliary labels. A hard two-stage cascade is retained
only as an ablation because early errors can block rare final behaviors.

### 5.3 Attribute Review Design

Review approximately 600-1000 unique review units, subject to a power and
coverage audit. One unit may receive all applicable attributes:

- posture: lying, sitting, standing, transition, unclear;
- locomotion: stationary, locomoting, turning, transition, unclear;
- ROI: target class, far/near/contact, head engagement, contact duration;
- interaction: none, social-nose, fight, other-contact, role, partner and count;
- quality: occlusion, bbox completeness, ID confidence, temporal consistency.

Sample by source, recording session, original class, confusion group,
occlusion/quality, and class frequency. Keep the existing review-unit grain:
16 frames for legacy and six frames for CVAT. Interaction review also shows the
full frame, actor box, partner boxes, and context-sufficiency state.

Double-review a stratified subset and report agreement overall and by attribute.
Attribute decisions apply through `review_unit_id`; they do not overwrite raw
labels or silently force a five-class mapping.

## 6. Target Model Family

The first credible candidate is a pretrained actor model plus audited spatial
and context branches. It must be capacity-matched against a strong actor-only
baseline before multimodal gains are claimed.

### 6.1 Visual And Temporal Backbone

- Engineering pilot: ImageNet-pretrained ResNet18 at `160x160`.
- Main thesis visual baseline: ImageNet-pretrained ResNet34 at `224x224`, only
  after the ResNet18 data, leakage, memory, and runtime gates pass.
- Optional later candidate: EfficientNet-B0 or ConvNeXt-Tiny, one at a time.
- Frame features: one embedding per observed actor crop with a padding mask.
- Temporal baseline: masked TCN with dilations and attention pooling.
- Main temporal candidate: two to four Transformer layers with relative time
  encoding, real frame deltas, and key-padding masks.
- Sequence lengths: compare native 6 and 16 frames first; use 8 and 12 only as
  controlled context-length ablations.

ResNet18 and ResNet34 comparisons must use the same ImageNet-1K pretrained
status. Record the exact `torchvision` weight enum, package versions, mean/std,
resize and letterbox contract, frozen stages, freeze duration, and checkpoint
hash. Train-from-scratch remains a separate ablation.

### 6.2 Multimodal Branches

| Branch | Input | Initial encoder |
|---|---|---|
| Actor | letterboxed bbox RGB sequence | pretrained CNN plus temporal model |
| Geometry | normalized bbox and shape | small MLP plus masked TCN |
| Motion | delta-frame-aware kinematics | small MLP plus masked TCN |
| ROI | all feeder, drinker, toy relations | class-symmetric MLP plus TCN |
| Social | nearest and top-K partner relations | set encoder plus temporal model |
| Context | actor-plus-partner/full-frame sequence | separate pretrained CNN |
| Quality | inference-available masks only | gated numeric embedding |

Every branch emits an availability mask. Fusion uses gated late fusion or
cross-attention, and missing context must produce a masked embedding rather than
a synthetic zero interpreted as real evidence.

### 6.3 Interaction Context

For every sample, select partners without labels using same-frame geometry and
stable tie-breaking. Use the nearest top-K pigs, actor-partner union crops, and
an optional downsampled full frame. Do not enable this branch only for known
interaction labels. Record availability by source and class for audit.

### 6.4 Training Stages

1. Warm up new heads with the pretrained backbone frozen.
2. Unfreeze the last visual stage using a lower learning rate.
3. Fine-tune the full backbone only if validation and runtime evidence support it.
4. Apply early stopping to native-unit macro-F1, not window-level macro-F1.

## 7. Severe Class Imbalance Plan

Current training windows range from 37,053 `sitting` samples to 472
`playwithtoy` samples, about a 78.5:1 ratio. At the native OOF unit, the ratio
is 11,269 `sitting` to 143 `playwithtoy`, about 78.8:1. `lying` is important and
confused with `sitting`, but it is not the rarest class. Counts must be reported
at frame, window, review-unit, native-event, video, and recording-date levels.

### 7.1 Unit Of Balancing

Balance unique native events first. Overlapping windows from one event share an
event mass of one, then divide that mass among their windows. Never oversample
rare overlapping windows as if each were an independent rare event.

For event `e` with windows `W_e`:

```text
base_window_weight(e, w) = review_weight(e, w) / sum(review_weight(e, j))
```

Class correction is then fitted from total training-fold event mass, not global
window counts. The audit records min, max, mean, quantiles, class mass, and
effective sample size before and after weighting.

### 7.2 Controlled Loss Candidates

Evaluate each method independently before combinations:

| ID | Method | Scientific role |
|---|---|---|
| L0 | event-balanced standard CE | reference |
| LP | inverse-frequency CE | paper-aligned historical control |
| L1 | effective-number class-balanced CE | smooth rare-class correction |
| L2 | Balanced Softmax | training-prior-aware logits |
| L3 | logit adjustment | separate prior correction ablation |
| L4 | deferred reweighting | learn representation before strong weights |
| L5 | focal loss | hard-example emphasis, isolated ablation only |

For effective-number weighting, compute `E_n = (1-beta^n)/(1-beta)` from
training-fold native-event mass. Tune `beta` only on inner validation groups,
normalize weights to mean one, and cap extreme weights.

The inverse-frequency paper control is fitted from each training fold, capped,
and reported separately. It is not automatically combined with event oversampling
or focal loss because that can overcorrect rare classes.

### 7.3 Sampler And Batch Construction

- Sample unique events first, then one or more windows inside the event.
- Use class-aware event sampling only in training, never validation or test.
- Limit repeated use of the same event within an epoch.
- Require minimum source and class diversity when the fold supports it.
- Track unique events seen, event reuse, and effective sample size per epoch.
- Do not use both aggressive class sampling and large class weights initially.

### 7.4 Confusion-Aware Learning

After clean OOF predictions exist, create fold-safe hard-negative queues for:

- `lying` versus `sitting`;
- `move` versus `explore` and `stand`;
- `fight` versus `social-nose`, `move`, and `stand`;
- `eat` or `drink` versus `explore` and `stand`;
- `playwithtoy` versus `explore`, `stand`, and `move`.

Mine only from predictions generated without training on that event. Use
pair-specific contrastive or margin losses as a later ablation. Do not inspect
or manually relabel confirmatory test errors during candidate development.

### 7.5 Multitask Loss

The initial objective is:

```text
L = L_behavior
  + lambda_p * L_posture
  + lambda_m * L_locomotion
  + lambda_r * L_roi
  + lambda_s * L_social
  + lambda_i * L_interaction
  + lambda_c * L_soft_consistency
```

Start auxiliary lambdas at `0.1` or `0.2`, not all at `0.25` by default. Tune
one family at a time. Auxiliary losses use masks and confidence weights. The
consistency term must remain soft and cannot force an invalid deterministic
mapping from a complex behavior to a single posture or motion state.

### 7.6 Data-Level Remediation

Loss reweighting cannot create missing behavioral diversity. Before promoting a
rare-class model, audit unique events, videos, dates, sources, duration, context
availability, and reviewer confidence for every class.

Priority actions:

- re-review high-impact confusion pairs at review-unit granularity;
- double-review a stratified subset and report Cohen's kappa or Krippendorff's
  alpha with classwise disagreement;
- collect additional independent events for `playwithtoy`, `social-nose`,
  `fight`, `stand`, `move`, and `drink` where source diversity is weak;
- preserve natural event duration and avoid manufacturing independent samples
  by sliding the same rare interval many times;
- use only label-preserving visual augmentation: mild photometric jitter,
  small scale/translation, and carefully audited horizontal flip;
- do not use temporal reversal for direction-sensitive attributes unless target
  semantics remain valid;
- record augmentation parameters and apply them only to training folds.

Synthetic or generative augmentation is exploratory only. It requires a
real-only test and an ablation proving gains do not come from synthetic shortcuts.

## 8. Bergamini VISAPP 2021 Comparison Branch

### 8.1 Reference And Reproduction Boundary

Reference paper:

> Bergamini et al., "Extracting Accurate Long-term Behavior Changes from a
> Large Pig Dataset," VISAPP 2021, pp. 524-533.

DOI: `10.5220/0010288405240533`.

The reported five-behavior result is a hybrid pipeline, not a direct five-way
ResNet classifier:

1. `move` from a 2.5 cm center-displacement threshold over two seconds;
2. `eat` and `drink` from feeder/drinker geometry and body orientation;
3. remaining cases classified as lying or standing by ResNet18;
4. inverse class weighting for imbalance.

Record these details as user-supplied protocol until checked against the PDF and
any available source code. Do not claim exact reproduction without equivalent
time sampling, metric calibration, ROI geometry, orientation logic, preprocessing,
and evaluation units.

### 8.2 Strict Mapping Contract

| Canonical label | Five-class label |
|---|---|
| `stand` | `standing` |
| `lying` | `lying` |
| `move` | `moving` |
| `eat` | `eating` |
| `drink` | `drinking` |

Do not silently map `sitting` to standing, `explore` to moving, or interaction
classes to standing or moving. Preserve every native unit and add
`five_class_eligible` plus `five_class_exclusion_reason` for noneligible rows.

### 8.3 Strict And Coarse Analyses

1. Strict-5 analysis: train and evaluate a direct five-class head on
   the strict subset using grouped folds.
2. Paper-style coarse analysis: derive locomotion and posture from reviewed
   attributes, retain `sitting` separately, and report operational mappings.
3. Hybrid reproduction: run only after the calibration gate in section 8.5.

### 8.4 Required Model Comparisons

| ID | Model | Purpose |
|---|---|---|
| P0 | old checkpoint inference adapter | preserve historical artifact evidence |
| P1 | old architecture retrained | isolate architecture from old split |
| P2 | ResNet18 actor-only | strong modern visual control |
| P3 | actor plus temporal model | isolate temporal gain |
| P4 | full multimodal model | isolate nonvisual spatial/context gain |
| P5 | hierarchical multimodal model | test hierarchy hypothesis |

All trainable comparisons use identical eligible units, fold manifests, seeds,
augmentation policy, and native-unit metrics. The old random split may be
reproduced only as a labeled historical-comparability analysis, never as the
primary scientific result.

### 8.5 Hybrid Reproduction Calibration Gate

The optional hybrid track must verify:

- source PDF and any code-derived protocol details;
- effective frame rate and the exact 12-frame or equivalent two-second window;
- conversion from image displacement to centimeters using depth, ground-plane,
  or another documented physical calibration;
- feeder and drinker ROI coordinates in the same geometric coordinate system;
- body-orientation estimation and its failure behavior;
- binary lying-standing crop, normalization, weights, and decision threshold;
- whether paper metrics use ground-truth boxes, tracked boxes, or both;
- evaluation grain, class support, exclusions, and weighting.

Do not infer a 6 FPS stream from the CVAT six-frame anchor interval. Verify video
FPS and timestamps independently, then express the movement window in seconds.

If centimeter calibration or protocol equivalence is unavailable, rename the
experiment `paper_inspired_hybrid`, tune thresholds only on training groups, and
do not compare its accuracy as an exact reproduction of the reported result.

### 8.6 Claim-Safe Names

- `strict5_learned`: direct learned strict five-class benchmark.
- `coarse5_attribute`: operational mapping from reviewed attributes.
- `paper_inspired_hybrid`: incomplete heuristic reproduction.
- `bergamini2021_reproduction`: allowed only after every gate above passes.

## 9. Leakage-Safe Validation And Statistical Protocol

### 9.1 Grouping

- Outer grouping: recording date or verified session; video remains nested.
- Never split frames, windows, review units, or the same video across roles.
- `pig_id` does not connect animals across videos.
- Keep the existing 13-group OOF manifest as a paired baseline artifact.
- Add metadata audits for camera, pen, date, source, and duplicate visual events.

Create `recording_metadata.csv` with `video_key`, `farm_id`, `pen_id`,
`camera_id`, `cohort_id`, `recording_date`, `session_id`, `session_start_time`,
`metadata_source`, and `metadata_confidence`. Derived dates or sessions must cite
the filename, `times.txt`, or another source; unknown values remain explicit.

Current user-supplied dataset context is one research pen, one D435i camera, one
eight-pig cohort, recordings across 23 days in six weeks, and a 6 FPS scientific
source stream. Verify these values from primary documentation before using them
in a table or split. They imply temporal variation within one acquisition domain,
not multi-domain generalization.

Because all current groups have already informed development, the next result is
an internally validated comparative study, not a pristine external confirmation.
For stronger evidence, use nested grouped CV or collect untouched sessions.

An untouched same-domain session must have no prior use in detector or behavior
training, threshold or ROI calibration, tracking tuning, review-template design,
error analysis, or model selection. It supports a same-domain session-held-out
claim, not external-domain generalization. External claims require a new pen,
camera, cohort, or farm.

### 9.2 Model Selection

Recommended practical protocol:

1. Designate development groups for software, runtime, and ablation pilots.
2. Use grouped inner validation within each training partition for early stopping.
3. Lock architecture, loss, augmentation, and thresholds before full OOF.
4. Execute one authorized full OOF for the locked candidate and controls.
5. Do not tune from the resulting outer OOF errors.

If compute permits, use nested grouped CV for hyperparameter selection. If not,
declare the fixed pilot-group protocol and its limitation in the paper.

### 9.3 Metrics

Primary metric: native-unit macro-F1 across all supported classes.

Secondary metrics:

- macro recall, balanced accuracy, and accuracy;
- per-class precision, recall, F1, and support;
- confusion-pair counts and normalized confusion rates;
- NLL, Brier score, ECE, and reliability curves;
- source-balanced and per-source metrics;
- per-video and per-recording metrics;
- inference latency, throughput, peak VRAM, parameters, and FLOPs.

### 9.4 Statistical Comparison

- Use paired predictions on the same native units.
- Bootstrap by recording/session cluster, not by individual windows.
- Report absolute delta, relative delta, 95% CI, and fold-level distributions.
- Add a paired permutation or exact sign-based test across recording groups.
- Correct secondary multiple comparisons with Holm correction.
- Report at least three seeds for shortlisted pilot configurations.
- Predeclare a smallest effect of interest for 10-class macro-F1.

Initial promotion target: macro-F1 delta at least `+0.02` versus the current
same-protocol baseline, with a paired CI that does not indicate material harm.
For a paper-facing superiority claim, prefer a positive lower CI bound. Also
require no severe degradation in rare-class recall, source balance, or calibration.

### 9.5 Calibration

Fit temperature scaling or classwise calibration only from inner validation or
cross-fitted predictions. Never calibrate on the held-out outer group. Report
uncalibrated and calibrated metrics, and use calibrated probabilities for
threshold or abstention studies only.

### 9.6 PASS And FAIL Rules

PASS requires:

- zero group, video, temporal-unit, and cache-lineage leakage;
- all 10 classes supported in every report where mathematically possible;
- paired macro-F1 gain meeting the predeclared effect target;
- no catastrophic per-class recall collapse greater than 0.05 absolute without
  an explicitly accepted tradeoff;
- interaction gains evaluated on context-ready and all-source slices;
- source-balanced result consistent with the aggregate direction;
- complete reproducibility, calibration, runtime, and ablation artifacts.

FAIL or HOLD applies when a gain exists only at window level, only under random
split, only after test-informed tuning, or only because noneligible rows were
removed without an audit trail.

### 9.7 Performance Target Tiers

Targets are evaluated only under the frozen grouped native-unit protocol:

| Tier | 10-class macro-F1 target | Interpretation |
|---|---:|---|
| Minimum promotion | current baseline plus at least 0.02 | useful improvement |
| Research target | at least 0.50 | strong practical gain |
| Stretch target | at least 0.55 | high internal performance |

The research and stretch values are goals, not guaranteed PASS thresholds.
Promotion also requires confidence intervals and class guardrails. A higher
accuracy caused by predicting `sitting` or `explore` more often is not success.
The five-class target will be set after the paper's exact metric, support,
ethogram ambiguity, and evaluation pipeline are verified from primary sources.

### 9.8 Offline And Causal Protocols

Offline longitudinal analysis is the primary deployment claim. It may use
centered windows and declared post-processing because accuracy and long-term
behavior profiles are the priority.

Causal near-real-time is a separate secondary experiment. It may use only the
current and past frames, cannot use undeclared delayed identity repair or future
smoothing, and reports model-only plus end-to-end latency, throughput, frame
delay, macro-F1, and degradation from offline mode.

For a 6 FPS source, the deployment target is end-to-end throughput of at least
6 FPS. Failure to meet it does not invalidate an offline ResNet34 result, but it
blocks a near-real-time claim.

## 10. Dependency Graph

```text
M0 baseline reconciliation
 |
 +--> M1 ontology and five-class contract
 |     +--> M2 event-level imbalance audit
 |     +--> M5 hierarchical supervision
 |
 +--> M3 resolution cache and pretrained actor baseline
 |     +--> M4 temporal backbone
 |             +--> M6 multimodal and interaction fusion
 |
 +--> M7 five-class benchmark branch

M2 + M4 + M5 + M6 + M7
             |
             v
       M8 bounded smokes
             |
             v
       M9 controlled pilots
             |
             v
       M10 candidate lock and authorization
             |
             v
       M11 full grouped OOF
             |
             v
       M12 statistics and paper package
```

M2, M3, and M7 may proceed in parallel after M0/M1 if their files do not
overlap. No full run occurs before M10. Every milestone has one reversible
commit, a checker, an audit JSON, and an explicit rollback path.

## 11. Implementation Milestones

### M0. Reconcile And Freeze The Current Baseline

Purpose: establish a trustworthy comparison point before changing semantics.

Tasks:

- repair the stale `_runtime_match_errors` pytest import or restore the intended
  public helper contract;
- run the current block `07` calibration, confusion, and ablation reports;
- refresh block `08` registry and block `09` completion audit;
- reconcile stale architecture audits that still say full OOF was not run;
- freeze current predictions, metrics, config, cache hashes, and git SHA;
- write a baseline limitations note covering model size and old split leakage.

Primary files:

- `src/pig_behavior/classification_v2/training/full_run_preflight.py`
- `tests/test_classification_v2_full_run_preflight.py`
- `scripts/classification_v2/07_postrun_evaluation/`
- `scripts/classification_v2/08_publication_reporting/`
- `scripts/classification_v2/09_final_release_audit/`

PASS: pytest collection succeeds, postrun artifacts are current, baseline hashes
are immutable, and no Q2 superiority claim is made. Commit boundary:
`test: reconcile classification v2 full OOF baseline`.

### M1. Freeze Ontology, Attribute Targets, And Five-Class Contract

Purpose: define labels before writing model code.

Tasks:

- create a versioned ontology with label definitions and ambiguous cases;
- define strong, weak, unknown, and missing auxiliary supervision;
- define strict five-class eligibility without deleting noneligible units;
- add reviewer guidance for posture, locomotion, ROI, and social attributes;
- build a literature evidence table for the old paper, hierarchical behavior
  recognition, long-tail losses, temporal models, and grouped validation;
- preregister hypotheses, primary metric, comparison models, and claim boundary;
- build `recording_metadata.csv` with field-level provenance and confidence;
- produce counts by label, source, video, recording group, and review status.

Primary files:

- `configs/classification_v2/behavior_ontology_v2.json`
- `src/pig_behavior/classification_v2/ontology.py`
- `configs/classification_v2/recording_metadata.csv`
- `scripts/classification_v2/02_train_ready_exports/`
  `classification_v2_build_hierarchical_targets.py`
- `scripts/classification_v2/02_train_ready_exports/`
  `classification_v2_build_five_class_manifest.py`
- matching checkers in `scripts/classification_v2/09_final_release_audit/`

PASS: zero duplicate keys, all original rows retained, every auxiliary value has
a confidence/mask, and every five-class exclusion has a reason. Commit:
`feat: freeze classification v2 hierarchy and five class ontology`.

### M2. Implement Fold-Local Event Balancing And Loss Policies

Purpose: correct imbalance without window duplication or test-prior leakage.

Tasks:

- add native-event and effective-mass diagnostics;
- implement effective-number CE, Balanced Softmax, logit adjustment, deferred
  reweighting, and focal loss behind separate config values;
- implement an event-first sampler with deterministic seed and event reuse audit;
- fit priors and weights only from each training fold;
- add synthetic tests for exact weights, caps, ESS, and zero-support failures.

Primary files:

- `src/pig_behavior/classification_v2/training/samplers.py`
- `src/pig_behavior/classification_v2/training/imbalance_losses.py`
- `src/pig_behavior/classification_v2/training/config.py`
- `tests/test_classification_v2_imbalance_policies.py`
- checker under `scripts/classification_v2/04_baselines_smokes/`

PASS: held-out labels cannot change weights, overlapping windows conserve event
mass, all weights are finite, and ESS is reported by class/source. Commit:
`feat: add fold local event imbalance policies`.

### M3. Build Inspectable Resolution Caches And Pretrained Actor Baseline

Purpose: increase visual information without repeating video seek and crop work.

Tasks:

- create canonical `160x160` letterboxed actor and context caches once;
- preserve original crop aspect ratio, padding, bbox, frame, and source metadata;
- store packed tensors plus a one-to-one CSV index, SHA-256, shape, dtype, and
  per-cache preview contact sheets for human inspection;
- fail on duplicate cache key, missing tensor, nonfinite value, or stretched crop;
- implement ResNet18 pretrained and random-init controls;
- benchmark batch sizes under AMP on RTX 3050 before pilot training;
- after ResNet18 PASS, build the versioned `224x224` cache and ResNet34 control
  without changing split, normalization, augmentation, or pretrained status.

Primary files:

- `src/pig_behavior/classification_v2/datasets/image_cache.py`
- `src/pig_behavior/classification_v2/models/pretrained_visual.py`
- `scripts/classification_v2/03_image_cache_context/`
- `scripts/classification_v2/04_baselines_smokes/`
- `tests/test_classification_v2_pretrained_visual.py`

Output roots use semantic names such as `actor_rgb_letterbox_160_v1` and
`interaction_rgb_letterbox_160_v1`; do not create repeated `smoke` or `resume`
cache folders. PASS: zero source reads during training smoke, aspect-ratio tests
pass, previews match source crops, and runtime audit selects a safe batch size.
Commit: `feat: add pretrained actor cache and baseline`.

### M4. Add Strong Masked Temporal Encoders

Purpose: model behavior duration and transitions beyond frame appearance.

Tasks:

- retain the current TCN as a reproducible temporal control;
- add a masked Transformer with relative frame-time encoding;
- test invariance to padded values and sensitivity to frame order;
- support native 6-frame CVAT and 16-frame legacy intervals explicitly;
- add causal versus offline mode flags and prevent accidental future use.

Primary files:

- `src/pig_behavior/classification_v2/models/temporal_transformer.py`
- `src/pig_behavior/classification_v2/models/spatial_tcn.py`
- `src/pig_behavior/classification_v2/training/trainer.py`
- `tests/test_classification_v2_temporal_encoders.py`

PASS: padding delta is zero within tolerance, order reversal changes the output,
real frame deltas affect timing, and tiny-set overfit succeeds. Commit:
`feat: add masked temporal transformer control`.

### M5. Upgrade Hierarchical Targets, Heads, And Soft Fusion

Purpose: learn basic attributes while preserving direct final classification.

Tasks:

- replace unconditional deterministic decompositions with confidence and masks;
- add locomotion and social-state heads to posture, ROI, and interaction heads;
- feed soft attribute embeddings to the final head behind an opt-in gate;
- keep behavior-only, auxiliary-only, soft-fusion, and hard-cascade ablations;
- validate that masked targets contribute exactly zero loss and gradient;
- build the stratified 600-1000 unit attribute-review set from section 5.3;
- double-review a subset and keep agreement metrics beside target confidence.

Primary files:

- `src/pig_behavior/classification_v2/models/multitask_heads.py`
- `src/pig_behavior/classification_v2/models/multitask_fusion.py`
- `src/pig_behavior/classification_v2/training/multitask_loss.py`
- `src/pig_behavior/classification_v2/review/attribute_review.py`
- `tests/test_classification_v2_hierarchical_model.py`

PASS: final logits remain directly supervised, argmax auxiliary predictions are
never model inputs, unknown targets are masked, and hierarchy consistency cannot
force invalid mappings. Commit: `feat: add soft hierarchical behavior fusion`.

### M6. Upgrade Multimodal ROI And Social Context Fusion

Purpose: add context where appearance alone is structurally insufficient.

Tasks:

- encode all ROI classes symmetrically so y never selects a target ROI feature;
- add top-K partner set tokens, pair persistence, and relative motion;
- add actor-partner union crops and optional full-frame context for all samples;
- gate missing branches with explicit availability embeddings;
- test bystander fight protection and actor-only social-nose semantics;
- add ROI, social, context, and fusion ablations.

Primary files:

- `src/pig_behavior/classification_v2/models/multimodal_fusion_v2.py`
- `src/pig_behavior/classification_v2/models/partner_set_encoder.py`
- `src/pig_behavior/classification_v2/datasets/visual_context_dataset.py`
- `tests/test_classification_v2_multimodal_context.py`

PASS: routing is label-independent, missing context remains auditable, no target
ROI fields enter X, and full-frame/partner evidence is available in interaction
reports. Commit: `feat: add label independent multimodal context fusion`.

### M7. Implement The Five-Class Benchmark Pipeline

Purpose: make the old-paper comparison reproducible and separate from 10-class
development.

Tasks:

- build strict eligible manifests and grouped fold audits;
- add a direct five-class output head and loss configuration;
- add attribute-based coarse mapping with explicit unknown and exclusion states;
- adapt old checkpoint inference when its preprocessing can be reproduced;
- retrain an old-architecture control under safe folds;
- report old-random-split reproduction only as a separate appendix result;
- implement the hybrid movement/ROI/binary posture path only after section 8.5;
- add strict, coarse, and hybrid reports without merging their claims.

Primary files:

- `src/pig_behavior/classification_v2/benchmarks/five_class.py`
- `src/pig_behavior/classification_v2/benchmarks/bergamini_hybrid.py`
- `src/pig_behavior/classification_v2/benchmarks/legacy_model.py`
- `scripts/classification_v2/04_baselines_smokes/`
- `scripts/classification_v2/07_postrun_evaluation/`
- `tests/test_classification_v2_five_class_contract.py`

PASS: identical units and folds are used across learned models, noneligible rows
remain in audit output, and paper-inspired results cannot be labeled exact
reproduction or session-safe without their corresponding gates. Commit:
`feat: add strict five class paper benchmark`.

### M8. Run Bounded Correctness And Overfit Gates

Purpose: prove implementation correctness before expensive comparisons.

Run in this order:

1. import, compile, and focused pytest;
2. one batch forward/backward for every branch and loss;
3. deterministic repeat test with fixed seed;
4. overfit 16 to 64 unique native events;
5. one-fold, one-epoch smoke with source and class coverage;
6. cache-only IO and GPU utilization audit;
7. prediction-schema and native-collapse audit.

PASS: finite loss, decreasing overfit loss, nonconstant logits, exact resume,
zero source frame reads, valid native predictions, and no leakage columns.
Commit: `test: pass classification v2 scientific model smokes`.

### M9. Run Controlled Development Pilots

Purpose: select one candidate with interpretable ablations.

Pilot matrix, in order:

- B0 current tiny full multimodal baseline;
- B1 ResNet18 actor-only, event-balanced CE;
- B2 B1 plus TCN or Transformer;
- B3 ResNet34 `224x224` actor-temporal baseline after runtime PASS;
- B4 B3 plus spatial geometry and motion;
- B5 B4 plus ROI;
- B6 B5 plus social and visual context;
- B7 B6 plus soft hierarchy, the initial enhanced-model candidate;
- B8 best B1-B7 plus one imbalance method;
- B9 second imbalance method only if B8 evidence is insufficient;
- B10 strict five-class ResNet18, ResNet34, and enhanced-model controls;
- B11 optional paper-inspired hybrid after calibration PASS.

Use a small fixed development fold set and at least three seeds for finalists.
Compare native-unit macro-F1, class slices, calibration, runtime, and paired
predictions. Do not combine hierarchy, new loss, new sampler, and new backbone
in one unexplained jump.

PASS: select one 10-class candidate and one five-class candidate with complete
ablation evidence and no leakage/runtime blocker. Commit:
`research: select classification v2 locked pilot candidate`.

### M10. Lock Candidate, Snapshot, Runtime, And Authorization

Purpose: prevent late configuration drift before expensive OOF.

Tasks:

- freeze data snapshot, ontology, folds, cache hashes, model config, loss,
  augmentation, seeds, and metric contract;
- benchmark selected batch sizes and resolutions on the actual GPU;
- estimate fold and total runtime from measured full-epoch coverage;
- regenerate preflight, launch packet, and explicit user authorization;
- fail closed on code, config, cache, or snapshot drift.

PASS: every preflight check is current and the user explicitly approves the
long run for the exact config hash. Commit:
`chore: lock classification v2 scientific OOF candidate`.

### M11. Execute Full Grouped OOF

Purpose: generate paired predictions for the locked scientific comparisons.

Minimum full comparisons:

- current tiny baseline artifact, reused rather than retrained if compatible;
- pretrained ResNet18 pilot control if it remains scientifically informative;
- pretrained ResNet34 `224x224` actor-temporal thesis baseline;
- selected 10-class multimodal/hierarchical candidate;
- strict-5 ResNet18, ResNet34, and selected enhanced-model controls;
- optional hybrid only if its separate reproduction gate passes.

Every fold writes progress, checkpoint, train/eval indices, class priors,
weights, unique-event coverage, losses, predictions, runtime, VRAM, and hashes.
Resume is allowed only when the exact signature matches. No source frame seek is
allowed when a required packed cache exists.

PASS: all folds complete full epoch coverage, all expected native units have one
OOF prediction, row loss is zero, duplicate keys are zero, and audits are valid.
Commit only code and concise registry metadata, not large checkpoints:
`research: register classification v2 scientific OOF run`.

### M12. Postrun Statistics, Error Analysis, And Paper Package

Purpose: determine whether the improvement claim is scientifically supported.

Tasks:

- collapse window probabilities to native units with the frozen policy;
- run paired cluster bootstrap, permutation/sign tests, and Holm correction;
- calibrate with cross-fitted temperatures and report calibration change;
- produce per-class, per-source, per-video, and per-recording slices;
- review confusion pairs without changing confirmatory labels;
- build tables for 10-class, five-class, ablations, runtime, and limitations;
- register every run and run the final completion gate.

Required paper artifacts:

- data and ontology table;
- grouped split diagram and leakage audit;
- architecture diagram and parameter table;
- imbalance method and event-mass table;
- 10-class and five-class confusion matrices;
- paired effect sizes with confidence intervals;
- calibration and source-balanced results;
- qualitative successes, failures, and annotation ambiguity;
- explicit internal-only claim and external-validation limitation.

PASS: the completion gate states whether the claim is allowed, all reported
numbers trace to immutable artifacts, and negative results remain recorded.
Commit: `docs: publish classification v2 scientific evaluation package`.

## 12. Workflow Placement

New scripts stay inside the existing numbered structure:

| Stage | New responsibility |
|---|---|
| `02_train_ready_exports` | ontology, hierarchy, five-class manifests |
| `03_image_cache_context` | versioned 160/224 caches and previews |
| `04_baselines_smokes` | actor, temporal, loss, hierarchy smokes |
| `05_preflight_authorization` | snapshot, runtime, authorization gates |
| `06_full_oof_training` | locked 10-class and five-class runners |
| `07_postrun_evaluation` | paired statistics and confusion analysis |
| `08_publication_reporting` | registry, tables, and paper package |
| `09_final_release_audit` | cross-stage completion and claim gate |

Do not reintroduce `scripts/behavior_review_tools`, `scripts/dev_tools`, or
wrapper scripts. Shared logic belongs under `src/pig_behavior/classification_v2`.

## 13. Compute And Cache Policy

Default hardware-aware ladder:

| Gate | Resolution/model | Scope |
|---|---|---|
| G0 | synthetic tensors | forward and loss only |
| G1 | ResNet18 `160x160` | 16-64 event overfit |
| G2 | ResNet18 `160x160` | one fold, one epoch |
| G3 | ResNet34 `224x224` | one fold, one epoch after G2 PASS |
| G4 | shortlisted model | fixed development folds |
| G5 | locked model | full grouped OOF after authorization |

Each cache has one stable root with resumable state inside it. A cache smoke
uses a row limit but the same root and manifest schema; it does not create a new
folder. Resume verifies hashes before continuing. Rebuilding a cache creates a
new version only when preprocessing semantics or resolution changes.

Store arrays as `.npy` or another memory-mappable tensor format for fast reuse,
but pair them with:

- a readable CSV index;
- shape, dtype, min/max, checksum, and row-count audit;
- a preprocessing JSON contract;
- preview PNG contact sheets sampled across source, label, and aspect ratio;
- a loader that proves key-to-tensor alignment.

The packed binary tensor is not expected to be human-readable by itself. The
manifest and previews make its meaning inspectable and auditable.

## 14. Standard Verification Template

Every milestone runs from CMD with the project Python and `PYTHONPATH` set.
The exact focused tests depend on changed files, but the minimum order is:

1. overlong-line scan and `git diff --check`;
2. `py_compile` or `compileall` on changed Python files;
3. focused pytest for the changed contract;
4. checker script that writes an audit JSON;
5. bounded GPU smoke when model/runtime behavior changed;
6. worktree and artifact-lineage review before commit.

## 15. Resolved Working Decisions

- Reference: Bergamini et al., VISAPP 2021, DOI `10.5220/0010288405240533`.
- Primary task: 10-class session-safe recognition.
- Secondary task: strict five-class comparison.
- Optional track: calibrated paper-aligned hybrid reproduction.
- ImageNet-1K pretrained weights are allowed and must be matched across models.
- Attribute review of about 600-1000 unique units is feasible.
- Structured recording metadata must be created from documented provenance.
- Offline longitudinal analysis is primary; causal near-real-time is secondary.
- ResNet34 at `224x224` is the main thesis baseline after ResNet18 gates pass.
- A same-domain untouched session is useful Q2 evidence but not external-domain
  evidence; its current availability is unconfirmed.

## 16. Open Evidence Gates

- Verify paper details from the primary PDF and any available source code.
- Recover or document the paper ethogram ambiguity for sitting and exploration.
- Prove centimeter displacement and two-second timing equivalence.
- Verify whether the paper evaluated ground-truth or tracked boxes.
- Determine whether a truly untouched same-domain session can be locked.
- Confirm structured metadata values and confidence before split construction.
- Measure ResNet34 memory, runtime, and causal throughput on the RTX 3050.
- Define the proposed enhanced model architecture before using a paper-facing
  name such as `Pig-STRENet`.

## 17. Change And Rollback Protocol

Before each milestone, record the clean starting SHA and commit a small plan
ledger update marking that milestone `IN_PROGRESS`. After its PASS gate, commit
the completed implementation and evidence. Keep each completion commit limited
to one contract or scientific hypothesis.

New public modules and nontrivial functions require concise docstrings stating
inputs, outputs, masks, and leakage assumptions. Add comments only where timing,
key alignment, or mathematical logic is not self-evident.

Before committing code, scan changed files for lines over 100 characters, run
`git diff --check`, compile changed Python, and run focused tests. Large caches,
predictions, and checkpoints stay as derived outputs unless repository policy
explicitly tracks them.

If a milestone fails, disable or revert only its opt-in config and preserve its
audit as a negative result. Do not rewrite reviewed data, remove failed evidence,
or alter a prior baseline to make a later model look better.
