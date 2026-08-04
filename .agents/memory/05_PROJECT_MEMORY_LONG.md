# Project Memory - Long Term

## Lifecycle contract

- Scope: project charter, stable contracts, accepted facts, and a complete
  project-wide overview.
- Admit only validated information with source authority, scope, acceptance
  date, and invalidation condition.
- Do not store daily progress, unresolved blockers, hypotheses, raw failures,
  or temporary next actions here.
- Stable mission authority is `12_PROJECT_CHARTER.md`.
- Scientific claim status is `14_CLAIM_REGISTRY.md`.
- Evidence maturity authority is `21_MEMORY_MATURITY.json`; the living dossier
  below is generated and must not be edited manually.
- Current task state belongs in `01`, `02`, and `04`, not this file.
- Superseded long-term facts remain traceable but must be marked historical.

<!-- memory-maturity:dossier:start -->
## Living Project Dossier

This is a generated, evidence-bound reading surface. Its machine authority is
`21_MEMORY_MATURITY.json`. Elapsed inactivity never promotes knowledge;
acceptance and revalidation events do.

### Project Abstract and Scope

No maturity-registry entry is currently promoted here.

### Scientific Questions and Boundaries

No maturity-registry entry is currently promoted here.

### Data and System Architecture

No maturity-registry entry is currently promoted here.

### Validated Methods and Contracts

No maturity-registry entry is currently promoted here.

### Supported Results and Claims

No maturity-registry entry is currently promoted here.

### Reusable Corrective Methods

No maturity-registry entry is currently promoted here.

### Limitations and Non-Generalization Boundaries

No maturity-registry entry is currently promoted here.

### Reproducibility and Governance

#### Evidence-based project memory maturity

Completed reusable knowledge moves to the living dossier only after typed evidence, deliberate
acceptance, source disposition, and event-based revalidation gates pass.

- Scope: Project-local agent memory lifecycle and long-memory dossier generation.
- Authority: `.agents/memory/03_PROJECT_RULES.md`, `.agents/memory/18_AUTHORITY_INDEX.json`,
  `.agents/memory/21_MEMORY_MATURITY.json`
- Evidence: `.agents/skills/project-state-steward/scripts/manage_memory_maturity.py`
  (da594c0d52cb), `tests/test_memory_maturity_manager.py` (a98ba864cb15)
- Accepted: 2026-08-03T08:39:40+07:00 by project-state-steward under explicit user direction
- Limitations: Agent fixture results do not prove live-agent compliance.; Existing scientific and
  operational history is not bulk-promoted.
- Invalidation: The manager, evidence schema, authority scope, or method state changes without
  revalidation.; A regression permits promotion by age or leaves invalidated knowledge current.

### Knowledge Awaiting Revalidation

None.

### Dossier Reading Contract

Entries are current only while their registered authority, method, claim,
artifact, and invalidation triggers remain satisfied. Superseded or
contradicted knowledge remains in registry history, not as current truth.

<!-- memory-maturity:dossier:end -->

## Archived overview snapshot: classification_v2 behavior recognition

As of 2026-07-12, the active workstream is `classification_v2` behavior
recognition.

Long-term target:

- Build a paper-facing Q2-strong internal evaluation for pig behavior
  recognition under recording-date/video-safe validation.
- Use multimodal spatio-temporal inputs:
  letterboxed actor bbox images, ROI relation tensors, social/partner context,
  interaction visual context, and leakage-safe engineered features.
- Keep the claim boundary conservative. Do not claim external farm, camera, or
  cohort generalization until an external validation set exists.
- Treat `pig_id` as annotation-local, not persistent identity across videos.
- Keep review/audit identifiers and label-derived policy fields out of model X.
- Use cached packed images for repeatable full OOF runs.

Current lineage status:

- The active task is rebuilding reviewed data, not promoting the old model run.
- The v6 Hidden design is a technical reference only. Existing 30-row Hidden
  and 3-row behavior payloads are unverified and cannot be carried.
- Verified coverage starts at 0/5,131 Hidden and 0/4,670 behavior units under
  `human_review_workspace/classification_v2/<RUN_ID>`.
- Agent evidence uses `outputs/classification_v2/agent_audits/<AUDIT_RUN_ID>`
  and must not write the operator-owned review root.
- The commit-`18d6692` 13-fold run is historical engineering evidence from the
  previous lineage; it is not the final thesis evaluation.
- A future reviewed-lineage full OOF requires frozen data/cache/fold hashes,
  local smoke gates, explicit authorization, and all postrun completion gates.
- Use `docs/CLASSIFICATION_V2_CURRENT_STATE.md` for current PASS/FAIL status.

Historical tracking memory below is preserved for tracking-specific tasks.

## A. Project overview

Project: `PIG_Behavior_Project`

Goal:

Build an RGB/RGB-D tracking and behavior profiling pipeline for 8 group-housed pigs.

Practical priorities:

1. Stable ID tracking for each pig.
2. Reduce ID switches during crowding/occlusion.
3. Run reliably on laptop hardware.
4. Support near-realtime/realtime modes when configured.
5. Export CVAT XML to support ground-truth labeling.

CVAT XML is a support output, not the main objective.

## B. Important videos

### `Pigs291119_000263_30fps`

- Sensitive video for IDSW regression.
- Legacy 21/06 achieved IDSW ≈ 2.
- Current code gives IDSW ≈ 6 with both old and new weights.
- Therefore this regression is not attributed to weight.

### `Pigs291119_000302_30fps`

- Difficult video that previously had low identity quality.
- Improved strongly with the new detector weight.
- Do not use this improvement as proof that current `hybrid_bytetrack` equals legacy behavior.

### `Pigs281119_000085_30fps`

- Common comparison video for 3-video benchmark set.

## C. Legacy tracking reference

Legacy reference around 21/06/2026, commit/snapshot around `1649aa2`.

Main old file:

```text
src/pig_behavior/data_preparation/tracking_engine.py
```

Legacy characteristics:

- No `cfg.mode`.
- No `hybrid_bytetrack`.
- No `bytetrack_raw`.
- One one-way tracking pipeline.
- Called `model.track(...)`, then parsed/filtered/matched in the same pipeline.

## D. Current tracking architecture

Current modules:

```text
src/pig_behavior/tracking/config.py
src/pig_behavior/tracking/runner.py
src/pig_behavior/tracking/detections.py
src/pig_behavior/tracking/association.py
src/pig_behavior/tracking/refinement.py
```

Current modes:

```text
realtime
bytetrack_raw
hybrid_bytetrack
bytetrack
gt_export
```

Important mapping:

```text
bytetrack -> hybrid_bytetrack
gt_export -> hybrid_bytetrack
```

## E. Current baseline

```text
tracking_mode = hybrid_bytetrack
config = iou0_area0_condarea0_merge0

USE_IOU_FALLBACK = False
USE_AREA_OCCLUSION_FREEZE = False
USE_CONDITIONAL_AREA_OCCLUSION_FREEZE = False
USE_MERGED_BOX_SPLIT = False
```

Do not enable `condarea` by default without ablation.

## F. Known regression

`hybrid_bytetrack` currently is not equivalent to legacy one-way tracking from 21/06.

Even with the same detector weight, `Pigs291119_000263_30fps` increased from IDSW ≈ 2 to IDSW ≈ 6.

## G. Most suspicious code differences

1. `association.py` uses raw ByteTrack ID signal in `hybrid_bytetrack`:

   - raw_id bypass gate for lost tracks
   - raw_id owner
   - raw_id penalty
   - raw_id preference

2. `association.py` lets `hybrid_bytetrack` match `all_detection_indices` too early.

3. `runner.py` may force post-processing for `hybrid_bytetrack`:

   - `apply_identity_swap_guard`
   - `refine_shapes_temporally`

4. `detections.py` has mode-specific filtering:

   - `hybrid_bytetrack` uses duplicate suppression/adaptive confidence ladder
   - `realtime` and `bytetrack_raw` use simpler `det_conf` filtering

5. Output path/export changed and may cause stale XML confusion if not checked.

## H. Patch priority

1. Patch `association.py`: remove raw_id owner/penalty/bypass from `hybrid_bytetrack`.
2. Patch matching phase: make `hybrid_bytetrack` use safer high-confidence /
   low-confidence matching.
3. Patch `runner.py`: do not force post-processing by mode.
4. Only then test `condarea` through ablation.
5. Check evaluation/prediction XML path for stale output confusion.

## I. What not to do

- Do not blame `000263` regression on weight.
- Do not change detector/weight to fix `000263`.
- Do not enable `condarea` by default without evidence.
- Do not add new complex heuristics before restoring legacy-compatible behavior.
- Do not run long benchmarks unless requested.
- Do not modify multiple unrelated components in one patch.

## J. Preserved notes from previous `.agents/PROJECT_MEMORY.md`

- The project is an AI-powered system for pig detection, tracking, and behavior classification.
- The core stack includes Python, PyTorch, OpenCV, ultralytics, FastAPI,
  numpy, pandas, scikit-learn, pytest, ruff, and MyPy.
- RGB-D occlusion inference exists as an auxiliary capability, while RGB-only
  tracking remains a core path.
- Behavior modeling uses temporal crops plus tabular sequence features.
- The prior memory file also recorded processed dataset paths, ROI
  annotations, and classifier feature conventions; that file remains
  preserved at `.agents/PROJECT_MEMORY.md`.
