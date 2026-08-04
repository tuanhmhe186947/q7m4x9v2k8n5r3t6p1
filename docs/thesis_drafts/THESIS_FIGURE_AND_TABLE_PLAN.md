# Thesis Figure and Table Plan

**Thesis:** *An AI Spatio-Temporal Framework for Dynamic Behavior Profiling
and Anomaly Detection in Group-Housed Pigs*  
**Status:** Minimal planning inventory v0.2  
**Numbering:** Figure numbers are working identifiers until the retained set is
selected; the final thesis sequence will be renumbered contiguously at
assembly.  
**Rule:** Figures are retained only when they explain a structure, show study
context, or provide evidence that cannot be conveyed as effectively by prose or
a table. A section is not assigned a figure by default.

## 1. Visual narrative

The thesis may use real study media and reproducible plots to connect the main
levels of the research story:

```text
real pen and recorded video
  -> tracked individuals and reviewed behavior intervals
  -> multimodal behavior/posture model
  -> long-term profile and deviation screening
```

Figures should not be decorative screenshots or duplicated explanations. Each
retained visual must have a source path or reproducible generator, a caption
that states its purpose, and a clear relationship to the surrounding paragraph.
Unbound or redundant candidates remain deferred and are not counted in the
thesis figure sequence.

## 2. Core figures

### Figure 1. Optional opening study image

**Purpose:** Provide a single representative view of the study pen only if
Chapter 1 requires a visual introduction.

**Status:** Optional and deferred. Do not combine an unbound tracking example,
behavior timeline and deviation profile merely to fill an opening figure slot.
The study context is already covered by Figure 3 and the framework by Figure 2.

### Figure 2. End-to-end framework overview

**Purpose:** Explain the three-layer architecture in one view.

**Paradigm:** Solution-overview flow diagram.

**Panels/modules:** RGB video input, detection and identity tracking,
individual-centred temporal windows, RGB-derived feature families, behavior and
posture heads, profile aggregation, and the two downstream alert/screening
branches. Annotation review and corrected-source lineage are not repeated in
this overview.

**Tool:** PowerPoint or Figma for the first layout; export as PDF or SVG.

**Status:** Can be drafted from the blueprint; final module names must match
the implemented pipeline.

### Figure 3. Study pen, camera, and data acquisition

**Purpose:** Give physical context for the dataset and explain why pen/camera
geometry limits transfer claims.

**Planned panels:** pen layout, feeder, nipple drinkers, enrichment device,
camera position, and one representative RGB scene frame. A short acquisition
annotation may state that the Intel RealSense D435i recorded both RGB and depth,
but the current behavior-classification path uses RGB-derived inputs;
depth must not be drawn as a current model branch unless a separate experiment
later demonstrates its contribution.

**Known facts to show:** approximately 5.8 m × 1.9 m pen, up to eight growing pigs,
Intel RealSense D435i, 1280×720, 6 fps source timing, daytime recording.

**Source:** user-confirmed study description plus original dataset metadata.
Candidate RGB frame:
`data/raw/images_clean/burst_color_012ef1fa_633_f10_k0.jpg`.

**Status:** Real RGB media located; source-layout confirmation is still
required. A depth visual is not needed for this figure.

### Figure 4. Behavior and visibility review lineage

**Purpose:** Show how annotated video segments pass through the two review
branches and become a corrected data source for later model windows.

**Planned stages:** annotated video segments, independent visibility review,
behavior-label review, accepted/corrected/excluded decisions, corrected source,
and the final consistency checks before window construction.

**Source:** review scripts, GUI guide, review-close authority, and corrected
source authority.

**Status:** Conditional shared figure for Sections 2.5--2.6. Retain only if the
two review branches and their different outputs cannot be explained clearly in
one paragraph and a compact table.

### Deferred candidate. Source-time conversion

**Purpose:** None required for the current draft. The relationship between the
6-fps acquisition clock and 30-fps playback packaging is sufficiently clear in
Section 2.2 and does not need a dedicated diagram.

**Status:** Deferred; use prose or a small data-specification table if the
source-time convention needs emphasis later.

### Figure 6. Detection-data construction and grouped split

**Purpose:** Explain the source-frame selection, duplicate filtering, manual
bounding-box annotation and date-grouped split used to construct detector data.

**Source:** selection notebook, detector annotation records and split manifest.

**Status:** Retained for Section 2.3, subject to final source and split binding.

### Figure 7. Representative detector annotations

**Purpose:** Show verified pig boxes in dense, occluded, low-occupancy and
empty-pen scenes. The panels are qualitative examples, not a substitute for
detection metrics.

**Source:** final manually checked detector frames and annotation records.

**Status:** Retained only if the selected panels are source-bound and add
information beyond Figure 6.

### Classifier performance visual (table-first; no fixed figure number)

**Purpose:** Report classifier and posture evidence in tables and confusion
matrices when the registered evaluator is available. A separate performance
figure is optional and should be added only if it improves interpretation.

**Source:** registered evaluator predictions only.

**Status:** Deferred until evaluation; Table 6 is the default presentation.

### Figure 8. Long-term behavior profiles

**Purpose:** Show how frame/burst predictions become individual or group
behavior profiles over an hour, session, or day.

**Recommended plots:** stacked behavior proportions, behavior transition graph,
and time-series profile for selected individuals.

**Source:** review-independent time-ordered predictions and profile builder.

**Status:** Conditional on a bound profile artifact and time-conversion audit;
the section may use a table instead.

### Figure 9. Behavioral-deviation screening examples

**Purpose:** Demonstrate screening outputs without implying a disease label.

**Recommended plots:** baseline versus observed profile, deviation score or
rule output, and a human-review flag timeline.

**Caption boundary:** use “potential behavioral deviation” or “screening
signal,” never “diagnosed abnormality.”

**Status:** Conditional on a bound baseline and screening-rule definition; not
required if only the screening protocol is reported.

## 3. Recommended tables

| Table | Content | Evidence required |
| --- | --- | --- |
| Table 1 | Study and recording specifications | Dataset metadata |
| Table 2 | Ten behavior classes and definitions | User-confirmed ontology |
| Table 3 | Review scope, controls, accepts, corrections | Review-close authority |
| Table 4 | Feature families and model-X exclusions | Feature whitelist audit |
| Table 5 | Model configurations and optimization contract | Frozen config manifests |
| Table 6 | Behavior and posture evaluation metrics | Registered predictions |
| Table 7 | Profile variables and screening rules | Profile evaluator |
| Table 8 | Limitations and transfer boundaries | Final authority and scope |

## 4. Universal visual rules

- Prefer PDF, SVG, or other vector export for diagrams and plots.
- Keep text readable after thesis-page scaling; target at least 8 pt.
- Use colour-blind-safe palettes and do not encode meaning by colour alone.
- Use honest axis ranges and show units, source-time conversion, and sample
  support.
- Make every caption self-contained; its first sentence should state what the
  figure establishes.
- Keep real frames representative and anonymized only if required by the data
  or institution.
- Never fabricate a profile, anomaly case, annotation correction, or metric to
  fill a visual slot.

## 5. Writing integration rule

For each subsection, first decide whether a visual adds information beyond the
prose or a table. If it does, the workflow is: write the Vietnamese explanation,
select or specify the supporting media, bind it to an evidence path and caption,
obtain confirmation of the scientific meaning, and then write the English
paragraph and caption together. If it does not, record the visual as deferred
and continue with the text.

The figure plan is a living inventory. A candidate remains `PENDING` or
`DEFERRED` until its source and scientific purpose are established; it is not
automatically included in the final thesis.
