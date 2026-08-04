# Classification V2 Thesis Blueprint and Evidence Map

**Document status:** Working blueprint v0.2  
**Language:** English  
**Document type:** Thesis, using the `MiniMedMind_thesis.docx` structure  
**Scope:** Classification V2 pig-behavior recognition project  
**Created:** 2026-08-02

## 1. Purpose and writing contract

This document is the planning authority for the first English thesis draft.
It maps each thesis section to project evidence and records the boundary
between established protocol, ongoing optimization, and claims that must wait
for a later artifact. It also treats real study images, pipeline diagrams, and
reproducible plots as part of the evidence plan rather than as decoration.

The retained reference template is:

```text
C:\Users\ironh\Downloads\Đồ án\Paper\Mẫu viết paper_thesis\MiniMedMind_thesis.docx
```

The thesis should preserve the template's broad sequence:

1. Acknowledgment
2. Abstract
3. Table of contents, lists, and abbreviations
4. Introduction
5. Methodology
6. Experiment
7. Conclusion and Future Work
8. References

The thesis may be longer than a journal article and may include implementation
detail, audit tables, and appendices. It must not present a development smoke,
unreviewed lineage, or optimization `keep` decision as final scientific
evidence.

## 2. Working research framing

### 2.1 Working title

**An AI Spatio-Temporal Framework for Dynamic Behavior Profiling and Anomaly
Detection in Group-Housed Pigs**

The title is user-confirmed and should be used consistently across the thesis,
figures, cover page, and future thesis artifacts.

### 2.2 Research questions

**RQ1.** How can native pig-behavior bursts be converted into a reproducible,
reviewed, and leakage-safe learning dataset when annotations include identity,
bounding-box, source, and temporal inconsistencies?

**RQ2.** Which actor, motion, region-of-interest, and social-context signals
contribute to ten-class behavior recognition under grouped video-safe
evaluation?

**RQ3.** Can an independent posture target improve the interpretation of
behavior states and support posture-aware error analysis under grouped
evaluation?

**RQ4.** How can long-term individual and group behavior profiles be used for
behavioral-deviation screening without presenting the result as a supervised
abnormality diagnosis?

RQ2, RQ3, and RQ4 remain open for quantitative results until the current
optimization, posture authority, and train-ready rebuild produce bound
evaluation artifacts.

### 2.3 System scope and anomaly semantics

The thesis describes a three-layer pipeline:

1. detection and identity tracking of individual pigs;
2. ten-class behavior classification from RGB-derived multimodal
   spatio-temporal inputs;
3. long-term behavior profiling and potential-deviation screening.

Behavior classification is the central scientific component. Detection and
tracking provide identity-preserving inputs, while profiling and screening are
downstream application layers.

The Intel RealSense camera recorded both RGB and depth streams. The current
behavior-classification pipeline, however, uses RGB-derived inputs:
actor appearance, bounding-box geometry, motion, ROI, pen context, and social
relations. Because no registered experiment has yet demonstrated a benefit
from depth, depth is treated as a recorded acquisition modality and a future
work direction, not as a current model feature.

Anomaly detection is not a frame-level `abnormal` class and does not have a
separate veterinary ground truth in the current dataset. The precise thesis
term is **behavioral deviation screening** or **potential behavioral anomaly
detection**. It aggregates behavior sequences over an hour, session, or day
using duration, frequency, mean bout length, transition counts, time-of-day
distribution, and deviation from an individual or group baseline.

The system therefore screens cases for further inspection. It does not diagnose
disease, stress, injury, or welfare status. Future anomaly ground truth may be
linked to health records, treatment, veterinary assessment, environmental
events, or expert review.

The intended deployment has two modes:

- offline analysis for long-term profiles and deviation screening;
- causal near-real-time alerts for high-priority events such as fight.

The offline mode may use longer aggregation and post-processing. The online
mode must use only current and past frames.

### 2.4 Provisional contributions

1. An end-to-end framework that connects detection, identity tracking,
   RGB-derived multimodal behavior classification, and long-term behavioral
   analysis.
2. A versioned review and corrected-source workflow for native pig-behavior
   bursts, including residual controls and a fixed-point temporal audit.
3. A leakage-safe RGB-derived multimodal spatio-temporal data contract and
   grouped evaluation protocol for ten-class behavior recognition.
4. A main posture experiment that separates lying, sitting, and standing for
   posture-aware interpretation and error analysis.
5. A downstream behavioral-deviation screening layer that uses long-term
   profiles without overclaiming supervised anomaly diagnosis.
6. An empirical analysis of multimodal feature families and their failure
   modes, once the optimization and evaluation artifacts are complete.

Contributions 4–6 must not be written as supported empirical results until the
corresponding authority, optimization, and evaluation artifacts exist.

### 2.5 Personal contribution boundary

The thesis inherits the source videos, pen context, and part of the behavior
taxonomy from prior work. The personal technical contribution is the redesign
and evaluation of the pipeline around that inherited material, including:

- data selection, recovery, harmonization, and label restructuring;
- annotation quality control and human-review workflow;
- detector and identity-tracking evaluation;
- geometry, motion, ROI, social, and temporal feature construction;
- sequence preparation and behavior-model experimentation;
- the posture experiment and long-term behavior analysis layers;
- leakage control, reproducibility, and evaluation protocol design.

The thesis must distinguish inherited data and taxonomy from the pipeline
components designed, implemented, or experimentally evaluated in this work.

## 3. Evidence status at blueprint creation

The latest user-confirmed review artifacts provide the current provisional
data layer:

| Evidence item | Current value | Status | Use in thesis |
| --- | --- | --- | --- |
| Human-reviewed units | 3,243 total | FROZEN artifact | Dataset/review protocol |
| Primary reviewed units | 3,123 | FROZEN artifact | Main reviewed scope |
| Independent controls | 120, seed `20260801` | FROZEN artifact | Residual-control method |
| Accepted decisions | 2,750 | FROZEN artifact | Review summary |
| Corrected source labels | 493 | FROZEN artifact | Correction analysis |
| Technical exclusions | 0 | FROZEN artifact | Coverage statement |
| Fixed-point HIGH targets | 0 | FROZEN artifact | Review-closure evidence |
| Corrected-source authority | Frozen | FROZEN artifact | Source lineage |
| Source context | SRUC, near Edinburgh, UK | USER-CONFIRMED | Dataset description |
| Recording period | 5 Nov–11 Dec 2019 | USER-CONFIRMED | Dataset description |
| Group composition | 8 growing pigs in one research pen | USER-CONFIRMED | Dataset description |
| Capture | D435i; RGB/depth recorded | USER-CONFIRMED | Acquisition |
| Native timing | 6 fps, daytime 07:00–19:00 | USER-CONFIRMED | Time conversion |
| Processed video | 30 fps packaging of 6 fps data | USER-CONFIRMED | Timing caveat |
| Final model/config/metrics | Not yet bound here | IN PROGRESS | Do not claim yet |
| Real posture authority | Planned; not executed | IN PROGRESS | Posture gate |

The direct evidence files are:

- `review_close_authority.json`
  Directory: `outputs/classification_v2/review_authority/`
  `review_close_behavior_3243_faee589_20260802_082500_v1/`
- `corrected_source_authority.json`
  Directory: `outputs/classification_v2/review_authority/`
  `review_close_behavior_3243_faee589_20260802_082500_v1/`
- `reviewed_training_application_authority.json`
  Directory: `outputs/classification_v2/review_authority/`
  `reviewed_training_application_3243_672eef4_20260802_v1/`
- `post_review_residual_suspicion_audit.json`
  Directory: `outputs/classification_v2/review_authority/`
  `post_review_fixed_point_final_3243_386c304_20260802_081500_v1/`

The reviewed-training application is currently an overlay authority. It does
not by itself prove that corrected frame features, adjusted-ROI features, and
all required T6/T8/T12/T16 windows have been rebuilt and audited.

The source context above is user-confirmed study knowledge and must be bound to
the original dataset description or prior publication before final submission.
The processed 30 fps files retain 1800 frames but represent five minutes of
real time because the source acquisition rate was 6 fps. Duration, frequency,
and biological speed must therefore use source-time conversion rather than
the playback FPS alone.

### 3.1 Authority reconciliation note

Some project memory and current-state Markdown summaries still describe the
earlier two-unit HIGH blocker. The latest generated authorities above describe
the subsequent `3,243`-unit closure. Until `project-state-steward` reconciles
those summaries and the claim registry, this blueprint treats the latest
artifacts as **provisional paper evidence**, not as permission to promote final
scientific claims.

### 3.2 Visual evidence requirement

Every major methodology or result subsection must have a planned visual
anchor: a real study frame, a system diagram, a reproducible plot, a table, or
an explicitly marked pending visual. The figure and table inventory is kept in:

`docs/thesis_drafts/THESIS_FIGURE_AND_TABLE_PLAN.md`

Text and visuals will be developed together in Vietnamese first, then rewritten
into original English after the user confirms the interpretation.

## 4. Thesis outline mapped to the template

### Front matter

#### Acknowledgment

Write only after authorship, supervisor, funding, and institutional wording are
confirmed.

#### Abstract

Draft last. It must state the problem, data protocol, method, evaluation design,
main quantitative findings, limitations, and conclusion. Do not insert metric
values before the final evaluation artifact is registered.

#### Lists and abbreviations

Recommended initial abbreviations:

- CVAT: Computer Vision Annotation Tool
- ROI: Region of Interest
- IDSW: Identity Switch
- OOF: Out-of-Fold
- RGB: Red, Green, Blue
- CNN: Convolutional Neural Network
- RNN: Recurrent Neural Network

The final list must be generated from terms actually used in the thesis.

### Chapter 1. Introduction

#### 1.1 Motivation

Explain why automatic pig-behavior monitoring requires temporal context,
identity continuity, and robust annotation quality. Introduce the practical
problem without claiming external-farm generalization.

#### 1.2 Problem definition and scope

Define the direct ten-class behavior target, the native annotated burst, the
single-pen/single-camera development scope, and the distinction between
annotation-local `pig_id` and biological identity. The target classes are:

- `drink` and `eat`: use of the drinker and feeder resources;
- `fight`: antagonistic interaction;
- `social-nose`: non-antagonistic nose-based social interaction;
- `explore`: investigation of the pen, substrate, or objects;
- `lying`, `stand`, `move`, and `sitting`: posture and locomotion states;
- `playwithtoy`: interaction with the enrichment device or toy.

These labels represent activities and states over a native burst, not merely a
single-frame pose. Their long-term distribution is the input to behavior
profiling.

The dataset and pipeline are designed for eight growing pigs housed together in
a research pen. The thesis must state that the current evidence is bounded to
the recorded pen and camera geometry.

#### 1.3 Related work

Organize the review around:

1. video-based animal-behavior recognition;
2. spatio-temporal and multimodal representation learning;
3. identity-aware tracking and interaction context;
4. annotation quality, temporal consistency, and data leakage;
5. grouped evaluation for correlated video data.

Every external statement requires a verified citation. The literature set is
not yet frozen in this evidence map.

#### 1.4 Contributions

Use the provisional contributions in Section 2.4, then narrow them to the
claims admitted by the final claim registry.

### Chapter 2. Methodology

#### 2.1 Overview pipeline

Present the pipeline as:

```text
raw video and annotations
  -> detection and identity tracking
  -> source harmonization and native-unit construction
  -> Hidden/Behavior/identity quality review
  -> frozen corrected-source authority
  -> review-independent feature and temporal-window rebuild
  -> grouped behavior and posture training/evaluation
  -> long-term behavior profiling and deviation screening
```

The model-input view should describe actor-centred temporal windows rather than
the annotation-rebuild workflow: the planned configurations are 6, 8, 12, and
16 frames, together with a six-frame sampled configuration at positions 0, 3,
6, 9, 12, and 15. This is a protocol description; final window manifests and
evaluation artifacts are still required before reporting comparative results.

Candidate evidence:

- `docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`
- `docs/CLASSIFICATION_V2_POST_REVIEW_LEARNING_PIPELINE.md`
- `docs/CLASSIFICATION_V2_GUI_OPERATOR_GUIDE.md`

#### 2.2 Data sources and native units

Describe the CVAT-derived six-frame units, legacy sixteen-frame units where
applicable, source provenance, frame indexing, and native-unit identity.
State explicitly that a training window is anchored to one native unit and is
not a substitute for the annotation unit. Record that the source acquisition
was 6 fps and that processed files are packaged at 30 fps while retaining the
same 1800-frame, five-minute real-time content.

Evidence to bind:

- source merge manifest and hashes;
- corrected-source authority;
- frame-feature manifest;
- final T6/T8/T12/T16 manifests after rebuild.

The real-time conversion must be explicit: 30 processed frames correspond to
approximately five seconds of real time, and one playback second represents
approximately five seconds in the pen. Biological duration and frequency must
not be computed from the 30 fps packaging rate alone.

The study pen is approximately 5.8 m × 1.9 m and contains a three-space
feeder, two nipple drinkers, a suspended enrichment device, and straw or paper
substrate over partly slatted flooring. Recording was daytime-only, from 07:00
to 19:00. These environmental details should be connected to the original
dataset source when the final thesis prose is written.

#### 2.3 Behavior annotation and review protocol

Describe the primary review, consistency re-review, targeted residual review,
independent 120-unit control, and fixed-point HIGH-gap audit. State that review
selection metadata and quality fields are audit-only and are excluded from
model-X.

Evidence:

- frozen composite behavior decisions and quality ledgers;
- `review_close_authority.json`;
- fixed-point audit JSON;
- GUI operator guide and review scripts.

The review protocol is part of data quality control, not a supervised anomaly
labeling process. Anomaly-oriented outputs are computed downstream from the
time-ordered behavior predictions.

#### 2.4 Identity, bounding-box, and corrected-source lineage

Explain the Mini-CVAT sidecar process, explicit editable IDs, sequential source
apply manifests, before/after hashes, rollback evidence, and the rule that
source corrections invalidate affected spatial and motion features.

Evidence:

- `scripts/classification_v2/01_review_units_gui/review_identity_continuity_gui_v2.py`
- `review_close_behavior_3243_faee589_20260802_082500_v1/corrected_source_authority.json`
- identity apply manifests listed by that authority.

#### 2.5 Long-term behavior profiling and deviation screening

Define the profile window as an hour, session, or day, depending on the
application. For each tracked individual and, where appropriate, the group,
aggregate:

- total duration per behavior;
- behavior frequency and mean bout duration;
- continuous-bout duration;
- behavior-transition counts and sequence structure;
- time-window behavior proportions;
- deviation from an individual or group baseline.

Candidate screening signals include prolonged lying, reduced eating or drinking,
reduced movement, unusually frequent antagonistic behavior, and a substantial
change in the normal behavior distribution. These are screening indicators for
human or veterinary follow-up, not diagnostic labels.

The offline profile uses longer aggregation and post-processing. The causal
near-real-time branch is reserved for urgent event alerts, such as fight, and
uses only current and past frames.

This layer requires a separate profile/evaluator artifact before quantitative
anomaly-screening claims are admitted.

#### 2.6 Feature construction and leakage controls

Describe actor RGB evidence, ROI/geometry evidence, motion, social context,
padding masks, and feature whitelists only after the final rebuilt manifest is
available. Explain the forbidden fields:

- review decisions and reviewer metadata;
- selection reasons, ranks, and quality flags;
- target-bearing fields and source labels;
- future-frame evidence outside model support;
- random or overlapping split identifiers.

Evidence:

- feature-builder source and manifests;
- model-input whitelist;
- target-leakage and split-leakage audits;
- `dataset-contract-leakage-guard` validation results.

#### 2.7 Model architecture and loss

Describe the final selected architecture only after optimization freezes the
config. The implementation currently contains multimodal and multitask
components, but the thesis must distinguish implemented infrastructure from
the promoted finalist.

Candidate code locations:

- `src/pig_behavior/classification_v2/models/multitask_heads.py`
- `src/pig_behavior/classification_v2/training/multitask_loss.py`
- `src/pig_behavior/classification_v2/training/trainer.py`
- `src/pig_behavior/classification_v2/training/multitask_smoke.py`

The posture head is a main thesis experiment, not a claim that can be assumed
from the presence of code. Its reader-facing states are `lying`, `sitting`, and
`standing`; the current machine contract represents `standing` as `upright`.
The main thesis requires a posture authority, grouped evaluation,
confusion matrix, and error analysis before reporting posture results.

#### 2.8 Reproducibility and implementation environment

Record the final code SHA, dirty-worktree status, semantic configuration hash,
data/artifact hashes, seed, Python/dependency environment, hardware, and exact
commands. This section cannot be finalized from the current blueprint alone.

### Chapter 3. Experiment

#### 3.1 Dataset summary and split design

Report source counts, native-unit counts, behavior support, reviewed and
unreviewed policy, and grouped split assignments only from the final rebuilt
snapshot. State that unreviewed units retain their original source label under
the frozen overlay policy; do not call them human-reviewed.

#### 3.2 Baselines and optimization protocol

Describe the frozen baseline, predeclared modality/feature families, matched
controls, seed policy, acceptance metrics, and stopping rules. The optimization
campaign must be represented by reproducible manifests, not narrative intuition.

Current status: optimization is ongoing and no final selected configuration is
admitted by this blueprint.

The experiment chapter must also contain a distinct posture experiment using
the `lying`, `sitting`, and `standing` targets. Report posture-specific support,
macro-F1 or balanced accuracy, per-class precision/recall/F1, confusion matrix,
and behavior-conditioned error analysis. Posture must remain an independent
target rather than being silently inferred from every behavior label.

#### 3.3 Evaluation metrics

Predeclare metrics before reading final predictions. Candidate metrics include
macro-F1, balanced accuracy, per-class precision/recall/F1, calibration, NLL,
confusion matrices, availability strata, and grouped uncertainty intervals.
The final metric set must match the evaluation contract.

#### 3.4 Quantitative results

Reserved for final registered predictions and evaluator artifacts. This section
must include the code SHA, split manifest, seed, configuration hash, and data
snapshot hash for every table or figure.

#### 3.5 Qualitative examples and error analysis

Use examples tied to review-independent evidence and preserve provenance. Do
not expose private reviewer notes or use selection metadata as model evidence.

#### 3.6 Ablation and robustness analysis

Report one declared scientific family per ablation. Separate single-camera,
single-pen findings from any later transfer experiment. No cross-pen or
external-farm claim is allowed without a distinct authority and evaluation.

#### 3.7 Long-term profiling and deviation screening

Evaluate this layer as an application analysis, not as supervised anomaly
classification. Report the profile window, baseline construction, thresholds or
outlier rule, alert rate, and representative cases. Without anomaly ground
truth, report descriptive screening behavior and expert-review requirements
rather than accuracy, sensitivity, or disease-detection claims.

### Chapter 4. Conclusion and Future Work

#### 4.1 Conclusion

Summarize only claims admitted by the final claim registry. The review protocol
may be concluded once the authority reconciliation and data rebuild gates pass;
model conclusions must wait for registered evaluation.

#### 4.2 Limitations and future work

Expected limitations to document, subject to evidence:

- single-pen and single-camera geometry;
- no demonstrated cross-pen transfer yet;
- dependence on corrected source and ROI authority;
- finite reviewed residual-control coverage;
- native-burst target semantics versus longer model context;
- posture authority, transition strata, and the absence of anomaly ground truth;
- optimization and compute constraints.

## 5. Evidence matrix for drafting order

| Thesis unit | Draft now? | Required evidence before final wording |
| --- | --- | --- |
| Introduction motivation | Yes | Project scope and verified literature |
| Related work | Partly | DOI-verified bibliography |
| Review protocol | Yes | Review-close and fixed-point artifacts |
| Corrected-source lineage | Yes | Corrected-source authority and apply chain |
| Behavior ontology and study context | Yes | User narrative plus source citation |
| Feature construction | Skeleton only | Rebuilt feature manifest and whitelist audit |
| Final architecture | Skeleton only | Frozen optimization config and code SHA |
| Posture experiment | Design now; results later | Posture authority and grouped evaluation |
| Dataset counts | Skeleton only | Final train-ready snapshot manifest |
| Metrics | Protocol now; values later | Frozen evaluator contract and predictions |
| Behavior profiling | Design now; values later | Profile manifest and time conversion |
| Deviation screening | Design now; descriptive results later | Baseline and screening evaluator |
| Results | No final claims yet | Registered evaluation artifacts |
| Limitations | Draft provisional list | Final scope and transfer status |
| Abstract | No | Final method and results |

## 6. Immediate writing sequence

1. Reconcile the latest review authorities with project memory and claim
   registry.
2. Freeze the evidence manifest for the final reviewed lineage.
3. Draft Chapter 2 Sections 2.1–2.5 from existing protocol artifacts and the
   user-confirmed study context.
4. Bind the rebuilt feature/window manifests and complete Sections 2.6–2.8.
5. Draft Chapter 1 around the actual contribution boundary.
6. Write Chapter 3 protocol sections before reading final results.
7. Add the main posture experiment after its authority and grouped evaluation
   are registered.
8. Add long-term profiles and deviation screening as a downstream application
   analysis with explicit no-ground-truth limitations.
9. Insert quantitative results only after optimization and evaluator artifacts
   are registered.
10. Finish Chapter 4, Abstract, lists, and front matter last.

## 7. Open decisions before thesis finalization

- Final thesis title and supervisor-approved terminology.
- Final citation style and bibliography source set.
- Completion of the posture authority and its executed evaluation.
- Profile window and baseline definition for deviation screening.
- Near-real-time alert policy versus offline screening policy.
- Exact train-ready snapshot, split, seed, and evaluator authority.
- Final optimization winner and matched ablation family.
- Ethics, data-availability, funding, conflict-of-interest, and authorship
  statements required by the institution.

## 8. Claim boundary

This blueprint is a planning artifact. It does not promote any claim to
`SUPPORTED`, does not authorize training, and does not replace the project
authority files or the final evaluation registry. It also does not treat
behavioral-deviation screening as supervised anomaly classification or as a
diagnosis of disease, stress, injury, or welfare status.
