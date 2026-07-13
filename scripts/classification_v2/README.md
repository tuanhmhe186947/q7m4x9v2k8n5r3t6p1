# classification_v2 Script Workflow

This directory is the single operator namespace for `classification_v2`.
There are no compatibility wrappers in the former script folders. Run every
command from the project root with `PYTHONPATH=%CD%\src`.

## Current execution point

The active lineage is in block `01`, not block `07`: complete the versioned
Hidden review, rebuild temporal/review artifacts, and complete behavior review
before train-ready exports. The prior commit-`18d6692` full OOF has positional
multimodal misalignment and is only compute/debug evidence.

The bounded technical chain under
`outputs/classification_v2/rebuilds/scientific_smoke_identifier_v2_20260713`
passes at commit `a83d5a5`. Block `09` identifier-v2 and technical smoke
checkers verify exact frame lineage, ordered window hashes, cross-stage counts,
feature semantics, spatial completeness, and 8/8 deterministic reruns. PASS is
technical evidence only; final lineage promotion remains blocked until both
human-review layers pass.

Status authority: `docs/CLASSIFICATION_V2_CURRENT_STATE.md`.

Data rebuild commands:
`docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`.

| Order | Folder | Workflow responsibility |
|---|---|---|
| 00 | `00_source_feature_temporal` | source merge, features, temporal units |
| 01 | `01_review_units_gui` | review units, GUI, decisions, review audits |
| 02 | `02_train_ready_exports` | X/y/masks/weights, folds, temporal views, leakage checks |
| 03 | `03_image_cache_context` | actor and interaction cache plus loaders |
| 04 | `04_baselines_smokes` | model contracts, baselines, bounded smokes |
| 05 | `05_preflight_authorization` | preflight, authorization, launch packets |
| 06 | `06_full_oof_training` | full OOF runner and training-output checks |
| 07 | `07_postrun_evaluation` | calibration, metrics, confusion, ablation |
| 08 | `08_publication_reporting` | registry, Q2 reports, paper package checks |
| 09 | `09_final_release_audit` | aggregate readiness and completion gates |

Each checker lives beside the workflow stage it validates. A stage may read
artifacts from earlier stages, but it must not invoke a later stage implicitly.

Folder numbers express ownership, not a strict one-pass order. The required
data sequence is `00 source/features -> 01 Hidden review/apply -> 00 temporal ->
01 behavior review/apply -> 02-09`. Hidden review must precede temporal
harmonization even though temporal scripts are owned by block `00`.
