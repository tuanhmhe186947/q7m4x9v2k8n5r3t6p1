---
name: thesis-evidence-writing
description: >-
  Project-local workflow for drafting this pig-behavior thesis from verified
  evidence. Use when planning or writing a methodology, data, experiment,
  figure, table, caption, or conclusion subsection; when turning Vietnamese
  notes into original English thesis prose; or when checking that prose,
  visuals, claims, and project artifacts remain aligned.
---

# Thesis Evidence Writing

## Purpose

Keep the author's scientific meaning in control while turning project material
into an English thesis. Work in Vietnamese first when the user is still
deciding the content, obtain confirmation of the technical interpretation, and
only then write original English prose. Every factual sentence and visual must
have an evidence path or an explicit `PENDING` status.

## Required context

Before drafting or revising, read the relevant sections of:

- `docs/CLASSIFICATION_V2_THESIS_BLUEPRINT_EVIDENCE_MAP.md`
- `docs/thesis_drafts/THESIS_FIGURE_AND_TABLE_PLAN.md`
- the current draft for the target section;
- the applicable project authority and memory files, especially
  `18_AUTHORITY_INDEX`, `02_CURRENT_DECISION`, and `04_PROJECT_MEMORY_MEDIUM`.

Use the newest authoritative artifact for data or results. If two sources claim
current authority for the same scope, stop and surface the conflict.

## Workflow

### 1. Fix the subsection contract

State the subsection's one job, its expected reader, and its evidence boundary.
Make a compact ledger with: claim, evidence path or hash, status
(`CONFIRMED`, `PROTOCOL`, `IN PROGRESS`, or `PENDING`), visual anchor, and open
question. Do not convert a protocol or planned experiment into a result.

### 2. Draft the Vietnamese meaning

Write or refine the content in Vietnamese when the user has not yet confirmed
the scientific interpretation. Keep technical names stable: the ten behavior
classes are `drink`, `eat`, `fight`, `social-nose`, `explore`, `lying`, `stand`,
`move`, `sitting`, and `playwithtoy`; reader-facing posture states are lying,
sitting, and standing.

Respect these project boundaries:

- behavior classification is the central layer between identity tracking and
  long-term profiling;
- anomaly output is `behavioral deviation screening` or `potential behavioral
  anomaly detection`, not a supervised `abnormal` label or diagnosis;
- processed 30-fps playback retains source 6-fps timing, so biological duration
  uses source-time conversion;
- inherited video/taxonomy must be separated from the author's pipeline,
  review, tracking, feature, model, and evaluation contributions;
- posture is a main experiment only after its authority and grouped evaluation
  exist.

### 3. Obtain the meaning checkpoint

Return the Vietnamese draft, evidence ledger, unresolved terms, and planned
visual to the user. Ask for confirmation of meaning before producing English.
If the user has already approved the Vietnamese content, record that approval
and proceed without re-opening settled terminology.

### 4. Write original English prose

Rewrite by meaning, not sentence-by-sentence translation. Use plain academic
English, stable terminology, and verbs whose strength matches the evidence.
Never invent a number, citation, mechanism, result, scenario, or identifier.
Separate observed results (`we observed`, `the evaluation reports`) from plans
(`will be evaluated`, `is intended to`). Keep project-specific claims bounded
to the recorded pen, camera, data contract, and evaluation authority.

Use `academic-paper` for general chapter structure and citation work. Apply
conservative language editing only after the scientific content is settled;
language improvement must not silently change scientific meaning.

### 5. Bind the visual anchor

Every major subsection must have one of: a real study image, an implementation
diagram, a reproducible plot, a table, or an explicit `PENDING` visual. Record
the source artifact and what the visual establishes. Never fabricate a frame,
profile, correction, metric, or anomaly case to fill a slot.

For a figure or table, provide a self-contained caption, units and time basis,
sample support, and a first sentence that states its purpose. Use
`figure-designer` when the visual still needs layout or chart-type design.

### 6. Run the thesis alignment check

Before delivery, verify:

1. title and terminology match the confirmed thesis title;
2. every factual claim has a valid evidence path and status;
3. results are not reported before the bound evaluator artifact exists;
4. source-time conversion is not confused with 30-fps playback;
5. anomaly language does not imply veterinary diagnosis;
6. posture claims match the actual `lying`/`sitting`/`standing` authority;
7. visual anchors, captions, tables, and prose describe the same scope;
8. inherited material and personal contributions remain distinct.

If a check fails, label the subsection `needs user attention` and state the
smallest missing artifact or decision. Do not silently fill the gap.

## Output contract

For a Vietnamese working pass, deliver: `Draft prose`, `Evidence anchors`,
`Visual anchor`, `Open questions`, and `English conversion status`.

For an approved English pass, deliver: `English prose`, `caption or table
text`, `evidence ledger`, and a short `meaning-risk / pending-items` note.
Keep internal planning tables out of the thesis manuscript unless the user
explicitly requests them as a thesis table.

## Hard stops and handoffs

Stop before writing when authority conflicts, a required file is unreadable, a
metric or model is not bound, or a visual would require fabrication. Hand off
to `academic-paper-reviewer` or `pre-submission-reviewer` for manuscript-level
review, and to the project data/lineage skills for dataset, split, feature, or
experiment-authority questions. This skill does not authorize source-data
edits, relabeling, model runs, or scientific claim promotion.
