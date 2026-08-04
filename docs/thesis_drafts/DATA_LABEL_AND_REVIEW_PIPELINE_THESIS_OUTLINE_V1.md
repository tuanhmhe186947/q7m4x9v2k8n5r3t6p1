# Thesis Outline — Data, Labeling, and Human-Review Pipeline

**Working language:** Vietnamese first, then academic English

**Thesis title:** *An AI Spatio-Temporal Framework for Dynamic Behavior
Profiling and Anomaly Detection in Group-Housed Pigs*

**Purpose:** provide a compact, evidence-bounded structure for Chapter 2.3 and
the related data-methodology sections. The outline separates what the model
receives from what is retained only as quality-control evidence.

## 1. Methodological claim boundary

The pipeline converts source video into identity-bearing frame observations,
reviewed native behavior units, and finally declared model windows. Detection
boxes, source-local tracks, behavior targets, visibility metadata, and review
decisions have different semantic grains. The chapter should state these
boundaries before describing any model input.

The review process is an annotation-quality protocol. It does not provide
veterinary labels, an `abnormal` frame class, or anomaly ground truth. Long-term
behavioral deviation is computed later from identity-conditioned predictions.

## 2. Source lineage and label granularity

Describe the historical detector-data selection only to the extent needed to
explain source provenance: pen masking, activity and background differences,
perceptual de-duplication, and video/day balancing. State that detector-assisted
tracking is a proposal and that CVAT correction supplies source-local boxes and
track continuity.

Define the two behavior-source contracts in one paragraph. A CVAT anchor `k`
represents `k..k+5`; the recovered legacy source retains sixteen-frame native
bursts. The sources are pooled to increase temporal and day-level diversity,
not because one source replaces the other. Their provenance and native lengths
remain available for audit and ablation.

## 3. Independent Hidden-visibility quality control

Present `Hidden` as a frame--object visibility attribute, independent of
behavior and posture. Explain that row-level visibility is not silently
propagated across a native unit. Any window-level burden or exclusion is an
explicit aggregation performed after review.

Describe the four target-independent cohorts in prose: untrusted Yes census,
risk-enriched No, stratified-random No, and low-risk No controls. Explain that
random sampling supports false-negative estimation with inverse-probability
weights, whereas high-risk correction yield is a prioritisation result and not
a population prevalence estimate.

Define the review conditions: exact adjacency is required for temporal
evidence; behavior and target fields cannot influence selection; the risk score
selects candidates but never changes a label. The reviewer chooses Yes, No, or
Unclear in the full-frame GUI. Unclear is unresolved and resolved decisions
must satisfy confidence-compatibility and coverage audits.

State the present status without ambiguity: the technical manifest and media
checks exist, but verified current human Hidden coverage is `0/5,131`.
Historical or carried payloads are not current review evidence; Hidden apply,
temporal rebuild, and training are blocked.

## 4. Behavior human review

After native temporal evidence is constructed, behavior units are selected for
context review using identity or box discontinuity, visibility burden, missing
partner evidence, ROI or motion disagreement, abrupt geometry, short runs, and
temporal label conflicts. These rules create inspection candidates rather than
new labels.

The reviewer accepts the source label, corrects it, or records a technical
exclusion. Corrections are applied through a declared source-authority step and
must be followed by row-count, key, provenance, feature-rebuild, and leakage
checks. Candidate scores, reviewer notes, and Hidden risk fields remain audit
metadata and are excluded from model input.

## 5. Figure 4 and text integration

**Figure 4 — Data-label and review lineage.** The surrounding text should
introduce the figure before describing the ordering. The figure should show:

```text
video and recovered crops
        |
detector-data selection -> detector-assisted tracks -> CVAT correction
        |
frame/object features and target-independent Hidden cohorts
        |--------------------------|
Hidden Yes/No/Unclear review   native behavior-unit review
        |--------------------------|
reviewed visibility metadata   accepted/corrected/excluded behavior units
        \__________________________/
          corrected source and audits
                    |
          harmonized windows -> model input
```

The figure must distinguish the model-input path from audit-only metadata. The
text should refer to Figure 4 in the opening paragraph, when explaining the
two review branches, and when describing the corrected-source gate. Do not
place unverified review counts or historical PASS status in the figure.

## 6. Evidence anchors and status labels

Use `docs/CLASSIFICATION_V2_CURRENT_STATE.md` as current status authority and
`.agents/memory/09_HIDDEN_REVIEW.md` for the Hidden contract. Use the runbook,
GUI guide, review scripts, and preparation notebooks as supporting evidence.
Label each statement as current protocol, implemented technical component,
historical lineage, or blocked scientific result. Do not cite the technical
Hidden reference as completed human review.

## 7. English conversion gate

Before conversion, confirm four meanings with the author: (i) CVAT and legacy
source roles, (ii) Hidden as independent frame/object visibility, (iii) the
reviewer decisions and unresolved `Unclear` rule, and (iv) the boundary between
annotation-quality review and behavioral-deviation screening. Convert the
confirmed Vietnamese prose as thesis writing rather than translating each
sentence literally.

## 8. Full-outline alignment (authoritative correction)

The previous version of this outline was too focused on Hidden review. The
current thesis structure is the consolidated outline in
`PIG_BEHAVIOR_THESIS_MASTER_OUTLINE_V2.md`. The following
mapping is the writing contract to use when drafting Chapter 2.

### 8.1 Sections 2.3--2.6: data and review

**Section 2.3 — Detection Dataset Construction and Pig Detection** must explain
the source-time frame-selection pipeline: empty-pen/background reference,
valid-pen mask, frame-to-frame and frame-to-background activity, one-second
candidate ranking, 64-bit average-hash filtering, temporal/video/day balancing,
fallback or emergency fill, manual Roboflow boxes and QC, grouped split,
duplicate/leakage audits, detector training configuration and the detector
output contract. The historical YOLO-assisted scaffold belongs in a bounded
subsection and must not be presented as final behavior or tracking truth.

**Section 2.4 — Identity Tracking** must distinguish detections, provisional
associations, source-local `track_id`/`pig_id`, trajectory continuity and
biological identity. It must describe CVAT tracking-source correction, the
sidecar identity-adjudication boundary, causal versus offline semantics,
occlusion/re-entry and variable visible-pig counts, separate tracking metrics,
runtime cost and downstream propagation. Detector accuracy must not be used as a
proxy for identity continuity.

**Section 2.5 — Behavior Data Construction, Annotation, and Human Review** must
state the ten-class ontology, the mixed role of CVAT and legacy sources, the
six-frame CVAT and sixteen-frame legacy native contracts, and the evidence-based
review candidate families. It must state that legacy is pooled with the current
source to add temporal day/video diversity, not excluded from training by
definition. `Hidden` must be introduced as an independent frame-object
visibility attribute, with its target-independent review cohorts and current
coverage status.

**Section 2.6 — Corrected-Source and Identity Lineage** must describe the
explicit accept/corrected/exclude decisions, apply scope, row/key/provenance
audits, feature rebuild and fixed-point checks. It must keep candidate scores,
review notes and Hidden sampling fields outside model X. The section may
describe the safe dependency order, but it must not claim a final train-ready
snapshot while current authority remains blocked.

### 8.2 Sections 2.7--2.10: model-facing data and downstream use

**Section 2.7 — Feature Construction** should cover only RGB-derived actor
appearance, box geometry, motion, ROI, social and temporal signals that are
declared in the feature whitelist. Depth is acquisition context/future work
unless a registered ablation proves its value.

**Section 2.8 — Temporal Sequence Construction and Leakage Controls** should
define windows of 6/8/12/16 frames and the sampled-six configuration, while
keeping native review units conceptually separate from model windows. It should
state the grouped split and overlap checks required before export.

**Section 2.9 — Behavior and Posture Model Architecture** should present the
multimodal RGB-derived architecture and posture as a main experiment. Review
metadata, Hidden risk, reviewer notes and candidate scores must not be described
as input features.

**Section 2.10 — Behavior Profiling and Behavioral Deviation Screening** should
aggregate identity-conditioned predictions over longer source-time periods.
The output is a potential behavioral-deviation signal, not a supervised
`abnormal` label, clinical diagnosis or welfare ground truth.

### 8.3 Visual and table binding

Use the consolidated outline's master numbering. Figure 6 should introduce the
detection-data workflow; Figures 8--12 should support tracking; Figure 5 should
show annotation/review lineage; Figures 13--16 should support model, posture,
profiling and deviation screening. Each visual must be named in the prose
immediately before or after it and bound to source-time, file/hash and evaluator
artifacts. Do not place a figure in a section as an isolated instruction or
editorial note. Tables 2, 6--8, 13--14 and 18 should carry the corresponding
protocol and feature boundaries; final counts and metrics remain pending until
their authority artifacts are frozen.

### 8.4 Status language for the manuscript

Use “implemented protocol” for code-backed workflow, “historical pathway” for
the notebooks and YOLO scaffold, “provisional” for detector/tracker outputs,
and “pending authority” for review-close, corrected-source, split and metric
artifacts. Do not write that Hidden review, behavior review or training is
complete merely because a manifest or historical PASS-looking payload exists.
The exact title remains *An AI Spatio-Temporal Framework for Dynamic Behavior
Profiling and Anomaly Detection in Group-Housed Pigs*.
