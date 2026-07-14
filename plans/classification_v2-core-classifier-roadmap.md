# Classification V2 Core Classifier Roadmap

Version: 1.0

Date: 2026-07-13

Status: canonical critical path for the next 10-class classifier lineage.

This roadmap is limited to `classification_v2`. It aims to find, validate, and
package the strongest scientifically defensible 10-class behavior classifier.
It does not evaluate detection, tracking, or the end-to-end runtime pipeline.

The extended research and publication ideas remain in
`classification_v2-scientific-performance-upgrade-roadmap.md`, but they do not
block the core phases defined here.

## 1. Scope And Success Definition

Primary task:

```text
drink, eat, fight, social-nose, explore,
lying, stand, move, sitting, playwithtoy
```

Success means a reproducible gain under recording/session-safe native-unit OOF,
with interpretable ablations, shortcut controls, rare-class guardrails, and an
exported inference-compatible checkpoint. A high random-window score is not
success.

### 1.1 Non-Goals For The Critical Path

- detector or tracker benchmarking;
- end-to-end causal latency or throughput claims;
- strict five-class and Bergamini reproduction experiments;
- alternative-farm, camera, pen, or cohort claims;
- publication formatting before the candidate is locked.

These tasks move to Phase 9 or a later integration roadmap. The classifier must
only expose stable inputs and outputs so it can be integrated later.

## 2. Current Evidence And Main Risks

The historical full OOF artifact has 32,727 native temporal predictions and
recorded the following outputs:

| Metric | Value |
|---|---:|
| Accuracy | 0.5217 |
| Macro-F1 | 0.4156 |
| Macro recall | 0.4518 |

These are not valid classifier-baseline metrics. Commit `bfdf913` found
151,440 positional mismatches across 160,740 split-to-image and
split-to-interaction rows. Keep the run only for compute/checkpoint debugging.
The next correctly aligned short pilot must establish the reconciled baseline
before any architecture or promotion comparison.

The next lineage must address five primary risks:

1. exact positional alignment across targets and every multimodal branch;
2. insufficient image resolution and visual capacity;
3. source shortcuts from domain, sequence length, and class distribution;
4. missing-context shortcuts across legacy and CVAT sources;
5. severe event-level class imbalance and label ambiguity.

## 3. Non-Negotiable Scientific Invariants

1. Never mutate raw files under `data/`.
2. Never silently drop frames, review units, native units, or windows.
3. Keep labels, identifiers, paths, `manual_*`, `review_*`, and policy text out
   of model X.
4. Never infer the feature set from all numeric columns.
5. Fit normalization, priors, weights, calibration, and thresholds from each
   training fold only.
6. Keep frames and units from one video in one split role.
7. Use one frozen fold manifest for paired model comparisons.
8. Report the primary result on pooled OOF native-unit predictions.
9. Use partner and context routing that is independent of the true label.
10. Preserve actor-only `social-nose` and direct-participant `fight` policy.
11. Treat `pig_id` as annotation-local, never as cross-video identity.
12. Change one principal experiment family at a time.

## 4. Classifier Integration Contract

The classifier is developed independently, but its interface must be usable by
the future pipeline.

### 4.1 Inputs

- actor RGB sequence with letterbox metadata;
- timestamps or `frame_delta_sec`;
- geometry and motion tensors;
- all-class ROI relation tensors;
- optional social and visual-context tensors;
- length, observation, quality, and modality-availability masks.

### 4.2 Outputs

- ordered 10-class logits and probabilities;
- optional auxiliary logits;
- confidence and calibration metadata;
- model, config, preprocessing, and label-order versions.

### 4.3 Inference Constraints

Every X feature must have an inference-time computation path. Ground-truth-only
visibility, behavior-derived target ROI fields, review decisions, and source IDs
cannot enter X. The model must accept missing modalities and support deterministic
batch inference from an immutable checkpoint package.

## 5. Metric Contract

Primary metric:

- macro-F1 on pooled OOF native-unit predictions using the global 10-class set.

Required secondary metrics:

- accuracy, balanced accuracy, macro recall, NLL, Brier score, and ECE;
- per-class precision, recall, F1, support, and confusion matrix;
- per-source, video, recording/session, and quality-slice metrics;
- parameter count, peak VRAM, throughput, and training runtime.

Required grouped metrics:

| Group | Labels |
|---|---|
| Rare | fight, social-nose, playwithtoy, move |
| Interaction | fight, social-nose |
| ROI behavior | eat, drink, playwithtoy |
| Posture | lying, sitting |
| Locomotion/context | move, explore, stand |

Promotion requires higher global macro-F1, nondecreasing rare-class macro-F1,
and improvement in at least one predeclared target confusion group. Exact class
guardrail thresholds are frozen after Phase 0 baseline reconciliation.

### 5.1 Fold Support Rules

- Build and publish a `class x fold` native-unit support matrix.
- Per-fold macro-F1 uses only classes with true support in that fold.
- Pooled OOF macro-F1 always uses the global 10-class order.
- Inner early stopping uses supported-class macro-F1 with NLL as tie-breaker.
- An inner validation role with inadequate class support is not used alone for
  architecture selection.

Implementation status: commit `abae856` enforces this policy in config,
trainer, checkpoint/resume, prediction manifests, and the run registry. Window
probabilities are averaged by `temporal_unit_key`; source and split-group
metadata remain audit-only. Remote pilots/full OOF require all 10 inner classes,
while local smoke requires at least two. Outer-test predictions are explicitly
ineligible for model selection. This is fixture-level engineering PASS only;
human review still blocks active-data performance evidence.

## 6. Shortcut-Control Contract

### 6.1 Source And Sequence-Length Controls

`source_type` never enters X. The primary temporal view is
`fixed6_observed_time`:

- CVAT reuses each harmonized six-frame anchor-interval window;
- legacy reuses existing harmonized six-frame subwindows within its burst;
- do not sample six quantiles across a legacy 16-frame burst;
- a legacy native unit may contribute multiple windows, but it never crosses a
  split and event weighting controls repeated event mass;
- both sources expose six keyed slots with identical length/padding semantics;
- timestamps and frame deltas remain only when shortcut audits permit them;
- every original window stays in a selection ledger and every native unit stays
  in audit/native-ablation artifacts.

Compare this primary view with `native6_16` only as an ablation. Also build a
`fixed6_normalized_phase` diagnostic that removes absolute duration. These views
separate padding-length shortcuts from legitimate temporal information.

Do not infer 6 FPS from a six-frame CVAT interval. Observed time uses verified
timestamps; normalized phase removes absolute duration as a diagnostic.

### 6.2 Required Source Probes

- Train a source probe from raw model inputs.
- Train the same probe from frozen learned embeddings.
- Report source predictability by modality and temporal view.
- Report all classifier metrics per source and on a source-balanced subset.
- Use source-balanced event batches when each fold has sufficient support.

High source-probe accuracy is a warning, not automatic proof of leakage. The
promotion question is whether behavior gains remain on matched, balanced slices.

### 6.3 Missing-Context Controls

Every context experiment reports:

1. actor-only on the context-ready matched subset;
2. actor plus availability only on the same subset and all data;
3. actor plus real context on the same subset and all data;
4. final model on all sources with natural missingness.

Training uses label-independent modality dropout. Its probability is fitted from
training data or predeclared, then applied without consulting y. The real context
branch must outperform the availability-only control; otherwise the apparent
gain is treated as a missingness shortcut.

Availability masks gate missing tensors but are not automatically accepted as
behavior evidence. Report context availability by label, source, video, and fold.

### 6.4 Augmentation Contract

- Actor-only RGB may test horizontal flip.
- Multimodal flip requires synchronized image, bbox, ROI, and relation transforms.
- Fixed-camera full-frame context is not flipped by default.
- Temporal reversal is disabled by default.
- Crop jitter is mild and cannot remove critical head or body evidence.
- Every augmentation is training-only, versioned, and recorded in run metadata.

## 7. Experiment Discipline

For each ablation:

- keep folds, eligible units, seed set, preprocessing, optimizer, and metric
  contract fixed unless that item is the tested variable;
- change one primary family: visual, temporal, spatial, context, imbalance, or
  hierarchy;
- use paired native-unit predictions;
- record config hash, code SHA, data/cache hash, device, seed, and runtime;
- require at least three seeds for shortlisted development candidates;
- stop a branch when its gain is absent, unstable, shortcut-driven, or too costly.

When a modality increases parameter count, add an actor-temporal wider-MLP
control with parameters within a predeclared tolerance, initially `+/- 5%`.

## 8. Core Dependency Graph

```text
P0 freeze and shortcut audit
 -> P1 visual baseline
 -> P2 temporal baseline
 -> P3 geometry, motion, and ROI
 -> P4 social context
 -> P5 imbalance selection
 -> P6 confusion-driven hierarchy
 -> P7 candidate lock
 -> P8 full OOF and model package
 -> P9 optional comparisons and integration
```

P9 never blocks P0-P8. Event-balanced CE starts in P1 and remains the reference
through P4; P5 selects whether a different imbalance loss is justified.

## 9. Phase 0: Freeze Data, Baseline, And Shortcut Evidence

### P0.1 Deliverables

- immutable train-ready snapshot and source hashes;
- reviewed native-unit and window manifests;
- one frozen recording/video-safe fold manifest;
- strict X feature whitelist and forbidden-field audit;
- ordered split/image/interaction/spatial lineage and snapshot-v2 binding;
- current OOF baseline reconciliation;
- class, source, length, context, and quality distributions;
- `class x fold` support matrix;
- source and missingness probe reports;
- versioned temporal-view manifests.

### P0.2 Baseline Reconciliation

Do not use the historical full-OOF metrics as a model-quality control because
its modalities were positionally misaligned. Register it only as
compute/checkpoint evidence. After the reviewed snapshot and P0 contracts pass,
run a correctly aligned short actor-temporal baseline to establish the first
reconciled performance control; outer predictions never select architecture.

### P0.3 Temporal Views

Create keyed, audited manifests for:

- `fixed6_observed_time`, the primary model-selection view;
- `fixed6_normalized_phase`, a source-shortcut diagnostic;
- `native6_16`, a secondary information-preserving ablation.

No view deletes native units. Each records selected frame indices, timestamps,
source interval length, missing frames, and deterministic sampling version.

Commit `bb225ff` implements this contract and structural shortcut audits on
synthetic fixtures. Active-lineage manifests remain blocked by human review.

Commit `9b04209` implements the keyed native-unit tabular source probe and
label-independent availability-only behavior diagnostic. It binds the exact
trainer whitelist and ordered-window hash, fits training roles only, and keeps
source/readiness metadata outside classifier X. Active-lineage evidence remains
blocked by human review and context-manifest rebuild.

### P0 PASS

Zero key leakage or row loss; all hashes and support tables exist; source and
context shortcuts are quantified; current baseline artifacts are internally
consistent. No learned-model change begins before this PASS.

## 10. Phase 1: Strong Visual Baseline

Use `fixed6_observed_time`, masked mean pooling, event-balanced CE, identical
augmentations, optimizer, folds, and seeds.

### P1 Visual Matrix

| ID | Backbone | Input | Purpose |
|---|---|---:|---|
| V0 | ResNet18 | `160x160` | engineering baseline |
| V1 | ResNet18 | `224x224` | isolate resolution gain |
| V2 | ResNet34 | `224x224` | isolate backbone gain |

All use the same ImageNet-1K pretrained status, exact weight enum, normalization,
freeze schedule, and trainable-head design. Train-from-scratch is not critical.

Implementation status: commit `2bd2fda` provides the shared frozen,
`layer4_only`, and optional full schedule, stable differential-LR optimizer
groups, BatchNorm policy, checkpoint/resume lineage, and a zero-step V0/V1/V2
checker. The checker proves structural controls without downloading weights or
reading project data. It is not the bounded reviewed-data pilot required for
P1 performance PASS.

### P1 Cache And Runtime Rules

- Build versioned letterboxed `160` and `224` caches once.
- Store packed tensors, key index, hashes, dtype/shape audit, and preview sheets.
- Never repeatedly seek, crop, resize, and convert source video during training.
- Benchmark safe batch size with AMP before each visual pilot.
- Local RTX 3050 validates correctness; external GPU may run larger pilots.

### P1 Selection

Run bounded development-fold pilots. `V0 -> V1` estimates resolution effect;
`V1 -> V2` estimates backbone effect. Only V1 and V2 need advance when V0 has
already served its engineering purpose.

Use ResNet18 at the selected resolution as the default P2-P6 architecture-search
backbone. V2 establishes the capacity reference; it does not make every modality
pilot pay the ResNet34 compute cost.

### P1 PASS

At least one pretrained baseline materially exceeds the current tiny model on
paired development units without source, rare-class, or runtime guardrail failure.

## 11. Phase 2: Temporal Baseline

Use ResNet18 at the selected Phase 1 resolution. Keep all other settings fixed.

### P2 Temporal Matrix

| ID | Encoder | Initial design |
|---|---|---|
| T0 | masked mean pooling | fixed Phase 1 control |
| T0A | masked attention pooling | learned pooling only |
| T1 | masked TCN | small dilated residual stack |
| T2 | small Transformer | one or two layers only |

T2 uses moderate `d_model`, real frame-delta encoding, key-padding masks, and
low-to-moderate dropout. Do not increase to four layers without T2 evidence.

### P2 Correctness Gates

- masked padding values cannot change valid outputs;
- reversing valid frames changes T1/T2 outputs;
- changing frame deltas changes time-aware outputs;
- one-batch forward/backward remains finite;
- 16-64 unique native events can be overfit;
- resume reproduces optimizer and prediction state exactly.

### P2 View Ablation

Compare the chosen temporal encoder on `fixed6_observed_time` and `native6_16`.
Include `fixed6_normalized_phase` as a shortcut diagnostic. Select native length
only if its gain survives source probes and source-balanced evaluation.

### P2 Legacy Temporal-Length Ladder

The separate `legacy-only-unreviewed-development` lane compares exact actor,
spatial, and observed-time inputs at `T6`, `T8`, `T12`, and `T16`. Every tier
is cut inside one complete 16-frame burst after harmonization. No temporal
resampling, interpolation, cross-burst window, or review inference is allowed.

Use two paired views:

| View | Windows per 16-frame burst | Scientific role |
|---|---:|---|
| all sliding, stride 3 | T6=4, T8=3, T12=2, T16=1 | primary development comparison |
| one centered window | exactly one for every tier | sample-count sensitivity control |

For each configured `T`, the model contract is:

```text
actor RGB             [B, T, 3, H, W]
spatial feature group [B, T, D_group]
observed time delta   [B, T]
actor/union context   [B, T, 3, H, W]
length/quality masks  [B, T]
final behavior logits [B, 10]
```

Spatial exports may be stored with capacity 16, but the loader must prove that
all length-mask values after `T` are false before slicing. Actor and context
sequences must already have exact length `T`; the loader does not truncate or
interpolate them to make a malformed sample fit.

For the all-sliding view, the total loss mass of each burst within each tier is
one. All windows from a burst stay in the same recording-safe fold. Window
predictions are aggregated to the common 16-frame native burst before metrics.
Backbone, resolution, temporal encoder, loss, sampler, fold manifest, and seed
remain fixed; only temporal input length changes. The centered control prevents
extra placements at shorter tiers from being mistaken for a sequence-length
gain.

These legacy-only metrics support architecture development and comparison with
the historical legacy model. They are not reviewed all-source evidence and
cannot authorize a Q2 claim or full OOF run.

### P2 PASS

Select the simplest temporal encoder whose paired gain is stable. Transformer is
rejected when TCN or pooling performs equivalently within the uncertainty margin.

## 12. Phase 3: Geometry, Motion, And ROI

Use the selected actor-temporal model. Add one feature family per experiment.

### P3 Matrix

| ID | Inputs added | Main hypothesis |
|---|---|---|
| A0 | actor-temporal only | control |
| A1 | geometry | posture and body-scale cues |
| A2 | geometry plus motion | locomotion separation |
| A3 | A2 plus all-class ROI | eat, drink, toy intent |

### P3 Feature Rules

- Geometry is normalized with verified frame dimensions.
- Motion uses real frame deltas and never crosses video or track boundaries.
- Feeder, drinker, and toy relations are exported for every sample.
- Behavior-derived target ROI fields remain audit-only.
- Missing ROI emits masks and reasons, never row deletion.
- Training-fold normalization is fitted separately for every fold.

### P3 Controls

For each added family, compare paired predictions and run a capacity-matched
actor-temporal wider-MLP control. Report source probes and class-group metrics.

### P3 PASS

Retain only feature families with stable gains on their hypothesized classes and
no evidence that source, missingness, or parameter count explains the gain.

## 13. Phase 4: Social And Interaction Context

Add social evidence incrementally. Do not begin with full-frame cross-attention.

### P4 Matrix

| ID | Context added | Advancement rule |
|---|---|---|
| S0 | no social branch | control |
| S1 | numeric pair geometry | first required test |
| S2 | top-K partner set encoder | only after S1 PASS |
| S3 | actor-partner union crop | only after S2 or S1 gap remains |
| S4 | full-frame context CNN | only when S3 remains insufficient |

Numeric relations include distance, overlap, contact proxy, relative motion,
approach/separation speed, persistence, partner count, and masks. Partner ranking
is deterministic and label-independent.

### P4 Shortcut Controls

Every S1-S4 comparison includes the matched context-ready subset, all-source
evaluation, availability-only control, label-independent modality dropout, and
source probes. Union-crop context is preferred before full frame because it has
less background and bystander capacity.

### P4 Interaction Audit

Report `fight`, `social-nose`, interaction macro-F1, actor/partner availability,
bystander hard negatives, multiple partners, and missing-partner cases.

### P4 PASS

Retain the smallest context stage whose real-context gain exceeds availability
only, survives matched-subset analysis, and improves interaction behavior without
materially reducing global or rare-class metrics.

## 14. Phase 5: Select One Imbalance Policy

Event-balanced standard CE is L0 and is used from Phase 1 onward. Phase 5 asks
whether one alternative improves the already-selected multimodal architecture.

### P5 Loss Matrix

| ID | Loss | Notes |
|---|---|---|
| L0 | event-balanced CE | reference |
| L1 | effective-number CE | fold-local event mass |
| L2 | Balanced Softmax | fold-local training prior |

Do not implement focal loss, deferred reweighting, or another sampler until
L0-L2 are insufficient and error analysis gives a falsifiable reason.

### P5 Weight Rules

- Balance unique native events before windows.
- Divide each event's mass among its overlapping windows.
- Fit class mass from training-fold events only.
- Normalize weights, cap extremes, and report effective sample size.
- Do not combine aggressive sampling with class loss in the first comparison.

### P5 Selection

Keep backbone, modalities, folds, seeds, and optimizer fixed. Select exactly one
policy using global, rare-class, class-group, calibration, and stability metrics.

### P5 PASS

The chosen policy improves or preserves rare-class macro-F1 without causing a
majority-class collapse or unstable seed dependence. Otherwise retain L0.

## 15. Phase 6: Confusion-Driven Attribute Hierarchy

Hierarchy starts only after Phases 1-5 establish the strong nonhierarchical
model and its development-fold errors.

The shortlist may use development OOF predictions only. Outer confirmatory
predictions remain unseen until the Phase 8 candidate and review data are frozen.

### P6 Review Shortlist

Sample 600-1000 unique review units with overlapping quotas for:

- lying versus sitting and stand versus sitting;
- move versus explore and move versus stand;
- fight versus social-nose, move, and stand;
- eat versus explore/stand and drink versus sitting/explore;
- playwithtoy versus explore/stand;
- interaction roles, receivers, bystanders, and close-contact hard negatives;
- high occlusion, missing partner, multiple partners, and transitions.

Double-review 20% of high-impact confusion and interaction groups and at least
10% of other reviewed groups. Report agreement by attribute and confusion group.

### P6 Target Policy

Use independently reviewed posture, locomotion, ROI, and interaction attributes
with confidence and masks. Do not pseudo-label the full dataset in version one.
Derived labels from final behavior remain weak targets and are not treated as
independent annotation evidence.

### P6 Model Matrix

| ID | Model |
|---|---|
| H0 | selected model without auxiliary heads |
| H1 | masked auxiliary heads, no feedback to final head |
| H2 | H1 plus soft attribute fusion |

The final behavior head remains directly supervised. Hard cascade is optional
research only and does not enter the critical path.

### P6 PASS

Hierarchy advances only when H1 or H2 improves the final 10-class target over H0
on identical inputs, folds, and seeds. Auxiliary accuracy alone is insufficient.

## 16. Phase 7: Lock Full-OOF Candidates

### P7.1 Capacity Confirmation

Transfer only the retained ResNet18 architecture to ResNet34:

| ID | Candidate |
|---|---|
| C0 | ResNet34 actor-temporal baseline |
| C1 | ResNet34 retained geometry, motion, and ROI candidate |
| C2 | ResNet34 final retained social candidate |
| C2-H0 | C2 without hierarchy when applicable |

Use identical inputs, temporal encoder, loss, optimizer policy, folds, and seeds.
This confirms that modality gains survive the stronger backbone without rerunning
every rejected P2-P6 branch on ResNet34.

### P7.2 Full-OOF Candidate Limit

Lock no more than three primary configurations plus one mandatory hierarchy
ablation:

| ID | Candidate |
|---|---|
| F0 | strongest actor-temporal baseline |
| F1 | actor plus retained geometry, motion, and ROI |
| F2 | final retained multimodal and social candidate |
| F2-H0 | F2 without hierarchy when F2 uses hierarchy |

If social context or hierarchy was rejected, F2 is the best simpler candidate
and the unnecessary ablation is omitted.

### P7.3 Lock Packet

Freeze model graph, feature groups, temporal view, loss, normalization,
augmentation, modality dropout, folds, seeds, cache hashes, class order, early
stopping, aggregation, and metric contract. Record measured runtime and compute
budget before requesting full-run authorization.

### P7.4 Promotion Gate

- all correctness and shortcut checks PASS;
- development gains are paired and stable across finalist seeds;
- global and rare-class guardrails PASS;
- each retained modality has an interpretable ablation;
- parameter-matched controls do not explain the main gains;
- full-run config is immutable and explicitly authorized.

## 17. Phase 8: Full Grouped OOF And Model Package

### P8 Execution

Run each locked candidate on the same outer folds. Independent fold jobs may run
on different GPU instances, but every job must validate the same snapshot,
config, and cache hashes before training.

Each fold records:

- train, inner-validation, and held-out group identities;
- train/eval native units and ordered checksums;
- fold-local class prior, weights, and normalization;
- seed, GPU, VRAM, CUDA, PyTorch, and torchvision versions;
- optimizer coverage, losses, checkpoint signature, and runtime;
- window and native-unit predictions.

### P8 Evaluation

- pooled OOF global 10-class metrics;
- class-group and confusion-pair guardrails;
- class-by-fold support and supported-class fold metrics;
- per-source, video, recording, context, and quality slices;
- paired cluster bootstrap and effect sizes;
- cross-fitted calibration;
- parameter, VRAM, throughput, and runtime table.

### P8 Package

Export immutable checkpoint, config, label order, preprocessing contract, input
schema, feature whitelist, calibration artifact, prediction schema, model card,
and a loader smoke test. Batch output must match direct model output.

### P8 PASS

All expected native units have exactly one OOF prediction, no rows are lost,
artifacts are hash-linked, and the selected model passes global, rare-class,
shortcut, calibration, and reproducibility gates.

## 18. Phase 9: Optional Work After Candidate Lock

These tasks cannot delay or alter the locked 10-class result:

- strict five-class ResNet18/ResNet34 comparison;
- Bergamini paper-inspired or calibrated reproduction;
- alternative backbones such as EfficientNet or ConvNeXt;
- hard cascade or graph-social research;
- detector/tracker noise robustness;
- causal windows, end-to-end latency, and deployment integration;
- publication formatting and extended statistical comparisons.

Any Phase 9 finding that motivates a new classifier starts a new versioned
lineage; it does not retroactively tune the Phase 8 held-out predictions.

## 19. Hardware-Agnostic Compute Policy

### Local Development Machine

- compile and focused tests;
- cache previews and alignment audits;
- one-batch forward/backward;
- tiny native-event overfit;
- one-fold one-epoch and inference smoke;
- resume and artifact-schema checks.

### External Or Rented GPU

- ResNet34 `224x224` pilots;
- multi-seed development experiments;
- union/full-context visual branches;
- full grouped OOF and heavy finalist ablations.

Hardware limits experiment placement, not scientific architecture. Every run
records device model, VRAM, software versions, hashes, fold, seed, runtime, and
checkpoint signature.

## 20. Module And Script Placement

Shared implementation belongs under `src/pig_behavior/classification_v2`:

```text
contracts/temporal_views.py
contracts/classifier_io.py
datasets/versioned_image_cache.py
models/pretrained_visual.py
models/temporal_pooling.py
models/temporal_transformer.py
models/partner_set_encoder.py
models/multimodal_fusion_v2.py
training/imbalance_losses.py
training/event_sampler.py
evaluation/shortcut_probes.py
evaluation/class_group_metrics.py
```

Operator scripts stay in the numbered workflow:

| Stage | Core responsibility |
|---|---|
| `02_train_ready_exports` | temporal views and contracts |
| `03_image_cache_context` | versioned caches and context indices |
| `04_baselines_smokes` | P1-P6 bounded experiments |
| `05_preflight_authorization` | candidate lock and runtime estimate |
| `06_full_oof_training` | independent locked fold jobs |
| `07_postrun_evaluation` | native metrics and shortcut reports |
| `08_publication_reporting` | model card and registry only after P8 |
| `09_final_release_audit` | classifier package completion gate |

Do not add wrapper scripts or restore former script namespaces. New public
functions require concise docstrings describing keys, masks, shapes, timing, and
leakage assumptions.

## 21. Standard Verification Ladder

Every implementation milestone runs:

1. changed-file overlong-line scan and `git diff --check`;
2. compile/import checks;
3. focused unit tests with hand-calculated or synthetic fixtures;
4. one-batch forward/backward;
5. deterministic repeat and resume test;
6. 16-64 native-event overfit when model behavior changed;
7. one-fold one-epoch smoke;
8. prediction, native-collapse, cache-I/O, and shortcut audits;
9. artifact hash and worktree review before commit.

### 21.1 Stop Conditions

Stop and investigate before the next phase when:

- any row or key disappears;
- source or availability-only controls explain a claimed gain;
- loss is nonfinite, logits collapse, or resume diverges;
- rare-class macro-F1 materially falls;
- a new modality does not outperform its parameter-matched control;
- an expensive stage has no stable paired gain;
- fold or cache lineage cannot be reproduced.

### 21.2 Full-Run Authorization

No full OOF starts from a pilot result alone. The exact snapshot, candidate,
runtime estimate, code SHA, config hash, and no-claim acknowledgement require
explicit authorization through the existing fail-closed gate.

## 22. Commit And Rollback Protocol

For each phase:

1. record the clean starting SHA;
2. commit a small ledger update marking the phase `IN_PROGRESS`;
3. implement one contract or experiment family;
4. run the complete applicable verification ladder;
5. commit the PASS evidence and implementation separately from large outputs;
6. update benchmark memory only for settled findings.

Caches, predictions, and checkpoints remain derived artifacts unless repository
policy explicitly tracks them. Failed experiments remain registered as negative
evidence. Rollback disables or reverts only the isolated opt-in feature; it never
rewrites reviewed data or historical baselines.

## 23. Core Classifier Completion Checklist

- [ ] Phase 0 snapshot, folds, baseline, and shortcut audits PASS.
- [ ] Resolution and backbone effects are separately measured.
- [ ] The simplest effective temporal encoder is selected.
- [ ] Geometry, motion, and ROI gains have isolated ablations.
- [ ] Social context advances incrementally and passes missingness controls.
- [ ] Exactly one declared imbalance policy is selected.
- [ ] Hierarchy is tested only after confusion-driven review.
- [ ] Full OOF candidates are limited and immutable.
- [ ] Global, rare-class, group, fold-support, and source metrics are reported.
- [ ] Parameter-matched controls are included where modalities add capacity.
- [ ] No full run uses source reads when a canonical cache exists.
- [ ] The checkpoint package passes deterministic batch inference.
- [ ] Integration inputs and outputs are versioned and leakage-safe.
- [ ] Optional five-class, paper, and deployment tasks did not block selection.

Completion means the strongest supported 10-class classifier is ready for later
integration. It does not imply detector/tracker or end-to-end pipeline readiness.
