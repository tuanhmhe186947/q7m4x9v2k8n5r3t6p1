# An AI Spatio-Temporal Framework for Dynamic Behavior Profiling and Anomaly
Detection in Group-Housed Pigs

**Document:** Master thesis outline, evidence-bounded revision V2  
**Working language:** Vietnamese meaning pass first; original academic English
after technical confirmation  
**Status:** Authoritative writing structure; not a results report
**Title status:** The approved thesis title is retained unchanged. The term
“anomaly detection” is operationally bounded in the thesis to
behavioral-deviation screening, not supervised abnormal classification or
clinical diagnosis.

## 1. Purpose and scientific centre

This document is the master structure for the thesis. It separates the
scientific method from implementation history and keeps each claim tied to the
current project authority. The thesis presents one end-to-end system, but its
scientific centre is identity-conditioned, multimodal spatio-temporal behavior
classification. Detection and tracking provide the identity-bearing inputs;
behavior profiling and deviation screening use the resulting predictions.

The study should be described as a controlled single-pen, single-camera study
of group-housed growing pigs. It should not claim cross-farm, cross-camera or
clinical generalisation without a registered experiment. The title uses
“anomaly detection” for potential behavioral-deviation screening, not for a
supervised `abnormal` class or veterinary diagnosis.

## 2. Research questions

**RQ1.** How can a reproducible and leakage-safe learning dataset be
constructed from heterogeneous pig-video annotations?

**RQ2.** How can individual identity continuity be maintained under occlusion,
re-entry and variable visible-pig counts, and how do causal and offline
tracking modes differ in quality and processing cost?

**RQ3.** Which visual, geometric, motion, ROI and social-context signals
contribute to ten-class behavior recognition?

**RQ4.** How accurately can lying, sitting and standing be recognised as an
independent posture target, and which behavior-classification errors can be
better characterised through posture-aware analysis?

This question is answered with posture Macro-F1, per-class recall, a confusion
matrix, unresolved/masked-sample reporting and behavior-error strata by
posture. A stronger claim that posture supervision improves behavior
classification requires a matched auxiliary-task ablation and is not assumed.

**RQ5.** How can identity-conditioned behavior profiles be used for behavioral
deviation screening without presenting the result as a supervised diagnosis?

RQ1--RQ5 define the thesis scope. They do not authorize quantitative claims
until the corresponding data, split, evaluator and artifact lineage is frozen.

## 3. Contribution and inheritance boundaries

The inherited material comprises the source videos, pen and camera context,
and part of the behavior taxonomy from earlier work using the same dataset.
The thesis contribution is the controlled reconstruction and evaluation of the
pipeline around that material:

- source-time detection-data selection and detector-data QC;
- identity-aware tracking, correction and separate tracking evaluation;
- mixed CVAT/legacy behavior-source harmonization with provenance;
- Hidden visibility and behavior human-review protocols;
- RGB-derived geometric, motion, ROI, temporal and social features;
- sequence-model and independent posture experiments;
- identity-conditioned profiling and potential-deviation screening; and
- grouped leakage control, reproducibility and error analysis.

The contribution claim is integration, control and evaluation of the complete
framework. It is not a claim that every detector, tracker or neural component
is individually novel.

## 4. Thesis structure

### Front matter

Acknowledgement; Abstract; Table of Contents; List of Figures; List of Tables;
List of Abbreviations.

### Chapter 1. Introduction

1.1. Motivation and Research Context  
1.2. Problem Definition and Study Scope  
1.3. Related Work  
1.4. Research Gaps  
1.5. Research Questions  
1.6. Contributions and Contribution Boundaries  
1.7. Thesis Organisation

Chapter 1 introduces the need for identity-aware temporal behavior monitoring,
states the single-pen scope, and defines deviation screening without clinical
overclaiming. Related work should be organised around pig detection and
detection-data construction, tracking and identity continuity, posture and
behavior recognition, spatio-temporal/context-aware modelling, and behavior
profiling or deviation analysis. Research gaps should be synthesised after the
literature review rather than mixed into every literature subsection.

### Chapter 2. Methodology

2.1. Overview of the Proposed Framework  
2.2. Study Data and Source-Time Representation  
2.3. Detection Dataset Construction and Pig Detection  
2.4. Identity Tracking and Continuity  
2.5. CVAT Behavior and Visibility Annotation  
2.6. Source Harmonization and Corrected-Source Lineage  
2.7. Spatio-Temporal Feature Construction  
2.8. Evidence-Guided Human Review  
2.9. Temporal Windows and Leakage Controls  
2.10. Behavior and Posture Model  
2.11. Behavior Profiling and Deviation Screening  
2.12. Reproducibility and Implementation Environment

### Chapter 3. Experiments and Results

This chapter currently defines result containers and reporting contracts; it
does not claim that training or multi-day/video evaluation has been completed.
The post-training results will be written only after the reviewed data,
training runs and grouped evaluators are frozen. The planned cross-day and
cross-video behavior results have an explicit placeholder in Section 3.6.1.

3.1. Dataset Composition and Grouped Splits  
3.1.1. Detection Dataset and Date-Grouped Split  
3.1.2. Tracking Ground-Truth Evaluation Population  
3.1.3. Mixed-Source Behavior Dataset  
3.1.4. Posture Target Support  
3.1.5. Cross-Task Leakage and Overlap Audits  
3.2. Experimental Protocol and Baselines  
3.3. Evaluation Metrics and Statistical Reporting  
3.4. Pig Detection Results  
3.5. Identity Tracking Results  
3.6. Ten-Class Behavior Classification Results  
3.7. Independent Posture Experiment  
3.8. Feature and Temporal Ablations  
3.9. Qualitative and Failure Analysis  
3.10. Long-Term Profiles and Deviation-Screening Demonstration  
3.11. Cross-Component Discussion and Answers to the Research Questions  
3.11.1. Answers to the Research Questions  
3.11.2. Detection--Tracking--Behavior Dependencies  
3.11.3. Scientific and Practical Implications  
3.11.4. Validity Threats

### Chapter 4. Conclusion and Future Work

4.1. Conclusions  
4.2. Limitations  
4.3. Future Work

References and appendices should contain data contracts, review schemas,
detector/tracker configurations, additional grouped metrics and the
reproducibility checklist.

## 5. Chapter 2 writing contracts

### 2.1. Overview of the Proposed Framework

Present the framework as a flow from RGB video to detection, identity-bearing
trajectories, actor-centred temporal inputs, behavior/posture outputs and two
downstream uses: long-term profile/deviation screening and causal event alerts.
The current model branch is RGB-derived. Depth is acquisition context and
future work unless a registered ablation demonstrates a contribution.

The prose should explain why identity continuity is required before duration,
bout, transition and individual-profile statistics can be interpreted. It
should identify behavior classification as the central component without
minimising the scientific role of tracking.

### 2.2. Study Data and Source-Time Representation

Describe the SRUC research pen near Edinburgh, recorded from 5 November to 11
December 2019, with approximate dimensions 5.8 m × 1.9 m. The pen contains a
three-space feeder, two nipple drinkers, suspended enrichment and straw/paper
substrate over partly slatted flooring. The D435i records RGB and depth during
daytime observation, and the group contains up to eight growing pigs. State
that the visible-pig count varies by frame; the pipeline does not assume eight
visible actors at every time point.

Each processed clip contains 1,800 frames. The source acquisition is about
6 fps and is recorded during the daytime window, approximately 07:00--19:00,
so a clip represents approximately five minutes of real observation, although
the MP4 is packaged for playback at 30 fps and lasts about one minute.
All biological duration, frequency and motion interpretations use source-time
timestamps or the source clock, not the playback rate alone.

Native behavior units must be distinguished from model windows. The CVAT source
uses an anchor interval `k..k+5`; the recovered legacy source retains a dense
sixteen-frame burst. Both sources can contribute to the mixed training dataset
and legacy is reported as additional temporal diversity, while source type,
video, actor, frame, native-unit key and timestamp remain traceable.

### 2.3. Detection Dataset Construction and Pig Detection

The concrete frame-selection authority is
`notebooks/01_data_preparation/video_to_frame_phase_1.ipynb`. It selects
source-time frame candidates from the declared recording-day groups and writes
the selected-image manifest used for manual detection annotation. The detector
annotation contract contains only the class `pig` and its bounding box.

This section explains how detector data were constructed and how detector
outputs are handed to tracking. It does not repeat behavior-review rules.

#### 2.3.1. Source-time candidate selection

The notebook ranks candidates using activity relative to the preceding frame
and, when available, a background reference, then applies temporal-window,
image-hash and within-video spacing filters. Exact thresholds and target counts
belong in the run manifest and Chapter 3 rather than in the general method
description.

Describe the notebook implementation: background and valid-pen mask, activity
relative to the preceding frame and background, source-time candidate ranking,
aHash/Hamming near-duplicate filtering, temporal spacing and date/video
coverage. Exact thresholds and target counts belong in the run manifest and
Chapter 3; any relaxed fill pass must retain its sampling provenance.

#### 2.3.2. Detection annotation and training

Bounding boxes are created manually on the selected frames. The detection
records do not contain `pig_id`, `track_id` or behavior labels; identity and
behavior are assigned in later pipeline stages.

Describe manual bounding-box annotation and quality control as the detector-data
authority. Report the detector architecture, configuration, split, evaluator,
and counts only after their manifests and hashes are bound. A detector box is a
frame-object observation; it is not a behavior label or a tracking identity.

#### 2.3.3. Detector method and output contract

Describe the detector method using the manually verified `pig` bounding boxes
as training targets. The architecture, training settings and exact counts are
reported with the registered experiment in Chapter 3. The output contract
contains the box coordinates, confidence, frame index and source-time key before
the ordered detections are handed to tracking. Detection output is not identity
ground truth and is not a behavior label.

#### 2.3.4. Split, leakage and output contract

Define grouped split units and audit duplicate, near-duplicate, neighboring
source-time and leaf/video overlap before reporting detector performance. The
detector output contract must state box coordinates, confidence and the exact
handoff to the tracker. Detection metrics and tracking metrics remain separate.

### 2.4. Identity Tracking and Continuity

#### 2.4.1. Identity Semantics and Tracking Scope

Distinguish frame-level detections, annotation identities, video-local track
identities, trajectories and biological identities. State that a track identity
is local to its video and is not a claim of one permanent biological identity
across all videos or the six-week recording period.

#### 2.4.2. Tracking Ground-Truth Construction

Describe the detector-assisted tracking scaffold, CVAT tracking annotations,
identity adjudication, source correction and provenance audit. A corrected
source becomes tracking ground truth only after the declared video/frame scope
has been applied and audited. The ground-truth population and evaluator are
separate from detector and behavior evaluation.

#### 2.4.3. Tracking Input and Common Evidence Contract

Define the detector-to-tracker record using box coordinates, confidence, frame
index and source time. State the method-specific frame cadence, valid-pen mask
and temporary hidden/missing-observation policy. The tracker output must retain
the identity, box and source-time information needed for trajectory evaluation.

#### 2.4.4. ByteTrack and RealTime-Fast Causal Cores

Describe *ByteTrack-Raw* as the un-repaired ByteTrack baseline and
*RealTime-Fast* as the causal association method. Summarize track lifecycle,
motion/appearance evidence, guards, tie-breaks and ambiguity handling. Neither
causal core may use future frames.

#### 2.4.5. Offline Trajectory Repair

Describe *Hybrid-ByteTrack* and *RF-Hybrid* as post-video methods. Define their
tracklet inputs, use of future context, repair candidates, accepted repairs and
output audit. Do not represent all modes with one shared additive association
cost.

#### 2.4.6. Development and Configuration Selection

Define the frozen 13-video development population, error strata, predeclared
metrics and configuration-freeze rule. Place detailed metric values and
configuration comparisons in Chapter 3; do not claim unseen generalization or
retune from the unseen population.

#### 2.4.7. Downstream Role and Failure Semantics

Explain how trajectories support duration, bout, transition and individual
profiles. Define fragmentation, identity switch, wrong-identity episodes and
re-entry as distinct failure semantics, and state the limitation when error
propagation into behavior statistics has not been measured.

### 2.5. CVAT Behavior and Visibility Annotation

#### 2.5.1. Ten-class ontology

Define `drink`, `eat`, `fight`, `social-nose`, `explore`, `lying`, `stand`,
`move`, `sitting` and `playwithtoy`. The classes describe activity, posture,
resource use and social interaction over a temporal unit, not an isolated frame
pose. Posture is later evaluated as an independent target.

#### 2.5.2. CVAT behavior-unit and visibility schema

Describe the CVAT annotation unit, actor association, behavior label, frame
anchor and visibility fields. `Hidden` is a frame-object visibility attribute,
not a behavior label, posture label or anomaly label. Keep the annotation
schema separate from the later model-window schema.

#### 2.5.3. Independent Hidden attribute

Introduce `Hidden` as a frame-object visibility attribute, independent of
behavior and posture. It is not silently propagated across a six-frame or
sixteen-frame unit. Target-independent cohorts include untrusted `Yes`,
high-risk `No`, stratified-random `No` and low-risk controls. Review decisions
are `Yes`, `No` or `Unclear` with confidence and provenance.

The current authority records zero verified Hidden decisions out of 5,131
review items. Therefore the thesis may describe the review protocol and its
scientific gates, but must not call the current data Hidden-reviewed or
train-ready.

#### 2.5.4. Annotation evidence fields

Review candidates are generated from ROI/resource, motion, posture/shape,
interaction/partner, visibility, box/identity and temporal evidence. Candidate
generation is an annotation-quality control layer. It does not create labels,
change weights or diagnose abnormal behavior. The exact
`fight`--interruption--`fight` rule is a conservative contiguous-unit
candidate, not an automatic relabeling rule.

#### 2.5.5. Annotation/review boundary

The reviewer can accept the source label, correct it, or technically exclude
the unit, with original label, corrected label, reviewer and apply scope bound
to the decision. Review-unit scope is not the same as a model-window scope.
Legacy complete review, targeted residual review and independent controls must
be described according to the current authority, not older PASS-looking files.

### 2.6. Source Harmonization and Corrected-Source Lineage

Explain how CVAT and legacy behavior units are mapped into a common schema for
the classification dataset while retaining source type, video, actor, frame,
native-unit key and timestamp. Legacy units are pooled with the CVAT source to
add day/video temporal diversity; they are not a separate evaluation label or
an automatically excluded training source. Native source length and model
window length remain distinct throughout harmonization.

Describe the immutable sequence: review decision → corrected-source authority
→ row/key/provenance audit → source-feature rebuild → temporal-window rebuild →
leakage audit. Source corrections invalidate affected geometry, motion, ROI and
social evidence until those features are rebuilt. Review notes, risk scores,
sampling strata and reviewer metadata remain audit fields and are excluded from
model X.

The current residual closure requires the targeted 39 units, the frozen
120-unit control and one subsequent fixed-point audit. Until that closure and
the required feature/window rebuild pass, Chapter 3 must not report a final
reviewed train-ready composition. The current handoff identifies one remaining
HIGH fight-bounded run in `Pigs291119_000225_30fps.mp4`, track 7, at anchors 204
and 210; this is a review blocker, not a relabeling instruction.

### 2.7. Spatio-Temporal Feature Construction

Define the model-facing feature families: actor RGB appearance, bounding-box
geometry, motion, functional ROI context, social/partner context and temporal
history. State the feature availability and missingness policy. Review reasons,
Hidden risk, reviewer notes, source labels, target-derived fields and future
frames outside the declared support are not model inputs.

The primary feature branch is RGB-derived. Depth is not to be described as a
validated model signal in this thesis unless a separate, registered ablation is
completed.

### 2.8. Evidence-Guided Human Review

Review candidates are generated from ROI/resource, motion, posture/shape,
interaction/partner, visibility, box/identity and temporal evidence. Candidate
generation is an annotation-quality control layer. It does not create labels,
change weights or diagnose abnormal behavior. The exact
`fight`--interruption--`fight` rule is a conservative contiguous-unit
candidate, not an automatic relabeling rule.

The reviewer can accept the source label, correct it or technically exclude
the unit, with the original label, corrected label, reviewer and apply scope
bound to the decision. Review-unit scope is not the same as model-window
scope. Legacy complete review, targeted residual review and independent
controls must be described according to the current authority, not older
PASS-looking files.

The current residual closure requires the targeted 39 units, the frozen
120-unit control and one subsequent fixed-point audit. Until that closure and
the required feature/window rebuild pass, Chapter 3 must not report a final
reviewed train-ready composition. The current handoff identifies one remaining
HIGH fight-bounded run in `Pigs291119_000225_30fps.mp4`, track 7, at anchors 204
and 210; this is a review blocker, not a relabeling instruction.

### 2.9. Temporal Windows and Leakage Controls

The six-frame view is a planning reference until the experiment manifest is
frozen. In the final thesis, replace “current primary view” with “the selected
primary temporal configuration” and link that wording to the corresponding
protocol and results.

Define the distinction between native annotation units and derived model
windows. The planned window experiments are 6, 8, 12 and 16 frames, together
with the sampled-six configuration at positions `0, 3, 6, 9, 12, 15`. The
current primary view is the post-harmonization six-frame representation for
both sources; other lengths are ablations unless a later authority promotes a
different primary contract. The final convention must be stated once the
window manifest is frozen.

Leakage control must cover grouped recording/day/session or video roles,
near-duplicate images, neighboring source-time intervals, overlapping native
units, actor/track identity leakage and ordered window IDs. A random frame split
is not an acceptable substitute for the declared grouped evaluation.

### 2.10. Behavior and Posture Model

Describe the promoted model only after its configuration, feature whitelist,
training lineage and evaluator are frozen. The model receives RGB-derived
multimodal temporal inputs and predicts the ten-class behavior target.

Posture is a main experiment, not a footnote. It uses an independent target with
`lying`, `sitting` and `standing` as reader-facing states; the current machine
contract may encode `standing` as `upright`. Posture labels must be independently
masked and evaluated. Behavior names must not be silently replaced by posture
names, and interaction labels such as `fight` or `social-nose` must not imply a
posture without evidence.

### 2.11. Behavior Profiling and Behavioral Deviation Screening

Aggregate identity-conditioned predictions over an hour, session or day. A
profile may include duration, frequency, mean and continuous bout length,
transition counts, time-window proportions and deviation from an individual or
group baseline.

Use the terms `behavioral deviation screening` or `potential behavioral anomaly
detection`. Candidate signals can include prolonged lying, reduced eating or
drinking, reduced movement, unusually frequent antagonistic interaction or a
substantial distribution shift. These signals require further inspection and
are not disease, stress, injury or welfare diagnoses. The offline branch may use
long aggregation and post-processing; a near-real-time branch must be causal.

### 2.12. Reproducibility and Implementation Environment

Record code SHA, dirty-worktree declaration, semantic configuration hash, data
and artifact hashes, random seed, Python/dependency environment, hardware and
exact commands. Distinguish code-verified contracts from artifact-verified
results and human-verified review decisions. Historical runs remain useful for
lineage or debugging but cannot be promoted to current scientific results when
their input or split lineage is incomplete.

## 6. Chapter 3 experiment contracts

### 3.1. Dataset composition and grouped splits

Report the final composition only from the rebuilt mixed-source snapshot. Show
source type, recording-day/video coverage, native-unit counts, ten-class support,
review status and split roles. Do not call unreviewed source rows human-reviewed.

#### 3.1.1. Detection dataset and date-grouped split

Define the detection population as frame-level pig bounding-box observations.
Its grouping key is the source recording date and video, with duplicate and
near-duplicate controls applied before the split. Detection image counts must
not be reused as behavior-sample counts.

#### 3.1.2. Tracking ground-truth evaluation population

Define the tracking population as the videos and frame intervals that have
tracking ground truth or an explicitly registered identity evaluator. Report
the video-level grouping key, visible-pig strata, occlusion/re-entry episodes
and causal/offline evaluation roles. Tracking videos are not automatically the
behavior-classification test set.

#### 3.1.3. Mixed-source behavior dataset

Define the behavior population as the pooled CVAT and legacy actor-centered
temporal units that pass the declared source, label and leakage contracts.
Report coverage by recording day and video, ten-class support, source
contribution and native-unit provenance. Legacy units are an additional source
of temporal diversity within the pooled training dataset, not a separate
population that is excluded by definition.

#### 3.1.4. Posture target support

Define posture as an independent target attached only where the posture label
is supported and not unresolved or masked. Report its usable, unresolved and
masked strata separately from the ten-class behavior counts. Posture support
must not be inferred from the behavior label name.

#### 3.1.5. Cross-task leakage and overlap audits

Audit overlap across detection frames, tracking videos, behavior units and
posture targets using source video, recording day, frame interval, actor or
track key and ordered-window identity. State the split group key for each task;
one generic word such as “test set” is not sufficient. Any post-training
cross-day or cross-video result must use the same declared grouping contract.

### 3.2. Experimental protocol and baselines

State the predeclared baseline, modality and feature families, training roles,
seeds, stopping criteria and acceptance metrics. Keep detector, tracker,
behavior model and downstream profile experiments separate so that a detector
metric is not mistaken for a behavior or identity metric.

### 3.3. Evaluation metrics and uncertainty

Use task-matched metrics: detection precision/recall or mAP when registered;
tracking HOTA/IDF1/IDSW and episode summaries; behavior macro-F1, per-class
metrics and confusion matrices; posture metrics with masked uncertainty; and
profile-level descriptive deviation measures. Report grouped uncertainty and
the evaluation population. No metric may enter the thesis before its evaluator,
inputs and hashes are bound.

### 3.4. Pig detection results

Report detector performance on the grouped detection split, with qualitative
examples for dense scenes, occlusion, low visible-pig counts and empty or
ambiguous frames. The figure and table must identify the source-time basis and
annotation version.

### 3.5. Identity tracking results

Report aggregate and per-video continuity, occlusion/re-entry episodes,
causal/offline quality and runtime. Include a fairness matrix fixing detector
weights, cadence, thresholds, GT population and evaluator. Discuss downstream
identity error propagation or state it as a limitation if not measured.

### 3.6. Ten-class behavior classification results

Report grouped performance for the ten classes and analyze confusion among
resource, interaction, motion and posture-related behaviors. Include feature
family and temporal-window ablations only when the associated train-ready
manifests and run registry are valid.

#### 3.6.1. Post-training multi-day and multi-video evaluation (PENDING)

After the reviewed data, training configuration and evaluator are frozen, add
the results from videos spanning multiple recording days. Report the number of
videos and days, class distribution by day and video, source contribution,
temporal diversity and the grouped cross-day/cross-video metrics. This section
is deliberately a results placeholder: no accuracy, macro-F1, confusion matrix
or generalisation claim may be written here before the corresponding training
and evaluation artifacts exist.

### 3.7. Independent posture experiment

Report posture metrics, confusion matrix, masked/unresolved handling and
posture-aware behavior errors. Explain whether posture helps interpretation;
do not infer this from the existence of a posture head alone.

### 3.8. Feature and temporal ablations

Compare actor RGB, geometry, motion, ROI, social and temporal families under
matched grouped splits. Depth remains a future-work branch unless validated.
Native six-/sixteen-frame source differences must not be confused with model
window length.

### 3.9. Qualitative and failure analysis

Use source-bound examples of detector failures, identity switches, occlusion,
ROI contradiction, interaction ambiguity, posture confusion and temporal
islands. A review candidate is evidence of uncertainty, not evidence that the
original label was wrong.

### 3.10. Long-term profiles and deviation screening

Demonstrate profile construction and deviation indicators using review-independent
predictions and explicit source-time windows. Present this as a screening
demonstration and limitation-aware analysis, not as supervised anomaly accuracy.

## 7. Figures and tables

Every visual must be introduced or interpreted in surrounding prose. A visual
is included only when it explains the method or evidence; it is not a layout
placeholder. Every image or plot needs source video/frame or artifact identity,
source-time basis, caption and status.

### 7.1. Recommended figures

The numbering below identifies retained or conditional candidates; it is not a
requirement to place every candidate in the main text. These are working
identifiers and will be renumbered contiguously after the retained set is
selected. Related panels may be
combined, moved to an appendix or omitted when the surrounding prose and tables
are sufficient.

1. Optional opening study image; deferred unless Chapter 1 needs it.  
2. End-to-end framework with the two downstream branches.  
3. Study pen, camera, resources and RGB acquisition context.  
4. Conditional behavior and visibility review lineage.  
Deferred candidate: source-time conversion, normally described in prose or a
table rather than a separate figure.  
6. Detection-data selection, annotation and grouped split workflow.  
7. Conditional qualitative detection panels.  
8. Conditional identity-conditioned behavior profiles.  
9. Conditional behavioral-deviation screening example.  

Figures 1, 4 and 7--9 remain `PENDING` or `DEFERRED` until their source or
evaluation artifacts are bound. Real RGB images may be taken from
`data/raw/images_clean/`; individual crops require source video, frame interval
and lineage binding before publication.

### 7.2. Recommended tables

The table content follows the existing figure/table plan; final numbering must
remain one master sequence after the chapter drafts are merged.

1. Study, camera, pen and source-time specifications.  
2. Detection candidate-selection stages and provenance fields.  
3. Detection split and annotation counts.  
4. Detector configuration and output contract.  
5. Tracking identity semantics, modes and evaluation population.  
6. Tracking metrics, episode strata and runtime.  
7. Ten behavior classes and definitions.  
8. Behavior/visibility annotation and review decisions.  
9. Source harmonization, review and corrected-lineage composition.  
10. Feature families, model-X exclusions and temporal windows.  
11. Behavior/posture metrics and grouped uncertainty.  
12. Profile variables, deviation rules and limitations.

Counts, metrics and composition tables are `PENDING` until the relevant
authority artifact is current. The manuscript should not include internal
evidence-status columns unless the supervisor requests them; those columns are
editorial controls used while drafting.

### 7.3. Section-to-visual binding

| Thesis section | Required visual anchor | What the surrounding prose must establish |
|---|---|---|
| 2.1 | Figure 2 | framework and downstream branches |
| 2.2 | Figure 3 | study context, camera position and functional pen regions |
| 2.3 | Figures 6 and 7 | detection selection, annotation and split logic |
| 2.4 | None required | identity lifecycle and causal/offline modes described in prose |
| 2.5--2.6 | Figure 4 (conditional) | review branches and corrected-source lineage |
| 2.7--2.9 | None required initially | text/table-led inputs, posture target and evaluation |
| 2.10 | Figures 8--9 (conditional) | profiles and screening interpretation |

The first paragraph that follows or precedes each visual must explain why it
is included and what evidence it establishes. A caption cannot substitute for
that textual cross-reference.

## 8. Current evidence status and claim boundaries

| Component | Status | Safe wording |
|---|---|---|
| Detection selection | Historical pathway | Protocol; final counts pending. |
| Detection annotation/training | Result artifact pending | Contract and evaluation design. |
| Tracking | Method authority frozen | Semantics and metrics; result pending. |
| Mixed sources | Schema and lineage implemented | Pooled sources with provenance. |
| Hidden review | `0/5,131` verified | Protocol and blocker, not completion. |
| Behavior review | Residual closure pending | Review lineage and current blocker. |
| Corrected rebuild | Blocked | No final feature/window claim. |
| Temporal windows | Builders exist | Report only bound manifests. |
| Behavior model | Data/evaluator pending | No final performance claim. |
| Cross-day/video evaluation | Training and grouped results pending | No generalisation claim. |
| Posture | Authority pending | Target and evaluation design. |
| Profiling | Downstream protocol | Descriptive screening, not diagnosis. |

The current authority is `docs/CLASSIFICATION_V2_CURRENT_STATE.md` together
with `02_CURRENT_DECISION.md` and the scope-specific authority index. Older
blueprint counts, smoke outputs and historical PASS-looking payloads remain
lineage evidence only. They must not override current blockers.

## 9. Evidence and exclusion rules for writing

Every factual sentence in the thesis must have one of four evidence classes:

- `SOURCE_FACT`: acquisition, pen, timing or inherited taxonomy;
- `PROTOCOL`: implemented data, review, tracking or model contract;
- `ARTIFACT_RESULT`: immutable evaluator output with data/config/code hashes;
- `HUMAN_REVIEW`: explicit reviewer decision with key, scope and provenance.

If no current evidence class is available, write the item as `PENDING`,
`LIMITATION` or `FUTURE WORK`. Never replace missing authority with a number
from a draft, a smoke run, a historical model output or a conversation memory.

The following must remain outside model X unless a new contract explicitly
proves otherwise: reviewer identity and notes, review reasons and priorities,
Hidden sampling strata, corrected labels used as leakage-bearing metadata,
future-frame evidence, split identifiers and any target-derived field.

## 10. Drafting workflow

1. Freeze the Vietnamese technical meaning for one subsection and its evidence
   ledger before writing English.
2. Write prose around the scientific question, not around implementation logs.
3. Introduce each figure or table in surrounding prose and state what it
   establishes.
4. Separate inherited data, model-assisted annotation, manual correction,
   derived evidence and final model input.
5. Use source-time for every duration or frequency statement.
6. Keep Hidden review and behavior review as annotation-quality controls, not
   anomaly labels.
7. Convert the approved Vietnamese meaning into original academic English;
   do not translate sentence by sentence.
8. Before adding results, check current authority, evaluator, split, hashes and
   claim-registry status.

## 11. Authoritative references

- `docs/CLASSIFICATION_V2_CURRENT_STATE.md`
- `.agents/memory/02_CURRENT_DECISION.md`
- `.agents/memory/18_AUTHORITY_INDEX.md`
- `docs/CLASSIFICATION_V2_THESIS_BLUEPRINT_EVIDENCE_MAP.md`
- `docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`
- `docs/CLASSIFICATION_V2_GUI_OPERATOR_GUIDE.md`
- `docs/thesis_drafts/THESIS_FIGURE_AND_TABLE_PLAN.md`
- `src/pig_behavior/classification_v2/`
- `scripts/classification_v2/`
- `notebooks/01_data_preparation/`

The previous file
`PIG_BEHAVIOR_COMPLETE_THESIS_OUTLINE_WITH_DETECTION_V1.md` is retained as
historical planning material. This V2 document is the master writing outline
for the thesis title shown at the top of this file.
