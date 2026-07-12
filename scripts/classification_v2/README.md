# classification_v2 Script Workflow

This directory is the single operator namespace for `classification_v2`.
There are no compatibility wrappers in the former script folders. Run every
command from the project root with `PYTHONPATH=%CD%\src`.

| Order | Folder | Workflow responsibility |
|---|---|---|
| 00 | `00_source_feature_temporal` | source merge, features, temporal units |
| 01 | `01_review_units_gui` | review units, GUI, decisions, review audits |
| 02 | `02_train_ready_exports` | X/y/masks/weights, folds, leakage checks |
| 03 | `03_image_cache_context` | actor and interaction cache plus loaders |
| 04 | `04_baselines_smokes` | model contracts, baselines, bounded smokes |
| 05 | `05_preflight_authorization` | preflight, authorization, launch packets |
| 06 | `06_full_oof_training` | full OOF runner and training-output checks |
| 07 | `07_postrun_evaluation` | calibration, metrics, confusion, ablation |
| 08 | `08_publication_reporting` | registry, Q2 reports, paper package checks |
| 09 | `09_final_release_audit` | aggregate readiness and completion gates |

Each checker lives beside the workflow stage it validates. A stage may read
artifacts from earlier stages, but it must not invoke a later stage implicitly.
The only authoritative order is `00` through `09`.
