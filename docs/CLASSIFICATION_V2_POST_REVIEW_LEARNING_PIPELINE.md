# Classification V2 post-review learning pipeline

## Purpose

This pipeline turns the completed Behavior review into two distinct products:

1. corrected label authority for the reviewed dataset; and
2. scientific diagnostics for the selector and spatiotemporal feature contract.

The second product does not become model input. Review decisions, error flags,
selection reasons, ranks, paths, IDs, and label-quality fields are forbidden
from model-X.

The pipeline can be installed while review is active, but it cannot read review
outcomes until a frozen `review_close_authority.json` exists.

## Fixed workflow

1. Finish the primary 2,729-item Behavior review.
2. Finish and source-apply mini-CVAT bbox, identity, and Hidden corrections.
3. Review at least 120 controls sampled from an explicit frozen population
   after subtracting the 2,729 primary scope.
4. Export frozen copies of both scopes, decisions, and label-quality sidecars.
5. Create `review_close_authority.json`. This fails if coverage is incomplete.
6. Analyse original-to-reviewed transitions and named spatiotemporal features.
7. Resolve any Behavior or Hidden conflicts introduced by mini-CVAT.
8. Rebuild source/frame features with the adjusted ROI authority:
   `data/annotations/roi/ROI_annotations.toy_adjusted.coco.json`.
9. Apply the frozen Behavior authority to the rebuilt frame authority.
10. Full-recompute unified T6, T8, T12, and T16 windows.
11. Run lineage, target-leakage, split-leakage, mask, and label audits.
12. Freeze a reviewed-Q2 snapshot only after every gate passes.

No old window structure may be reused in step 10.

## Stage A: predeclare the residual control scope

This is the only stage that may run before primary review is complete. Its two
inputs must not contain decision or label-quality outcome columns.

```bat
set "POST_REVIEW=scripts\classification_v2\02_train_ready_exports"
python "%POST_REVIEW%\build_post_review_control_scope.py" ^
  --population-csv <FROZEN_EXPLICIT_PARENT_POPULATION.csv> ^
  --primary-scope-csv <FROZEN_PRIMARY_2729_SCOPE.csv> ^
  --output-dir <CONTROL_AUTHORITY_OUTPUT_DIR> ^
  --target-count 120 --seed 20260801
```

The parent population is explicit. The implementation does not assume that it
is the 6,061 candidate population or the 27,294 auto-carry population. The
primary scope must be a strict subset, and control keys and temporal-unit keys
must have zero overlap with it.

Sampling is deterministic and stratified by behavior, source provenance, and
review-unit type. Recording date can be added explicitly as another stratum
when support is sufficient. Every selected row retains the source columns
unchanged and receives only `post_review_control_*` metadata, including sampling
probability and inverse-probability weight.

## Stage B: freeze completed review authority

Run only from durable frozen exports outside the active Behavior ledger path.

```bat
set "POST_REVIEW=scripts\classification_v2\02_train_ready_exports"
python "%POST_REVIEW%\freeze_post_review_authority.py" ^
  --primary-scope-csv <FROZEN_PRIMARY_SCOPE.csv> ^
  --primary-decisions-csv <FROZEN_PRIMARY_DECISIONS.csv> ^
  --primary-quality-csv <FROZEN_PRIMARY_QUALITY.csv> ^
  --control-scope-csv <FROZEN_CONTROL_SCOPE.csv> ^
  --control-decisions-csv <FROZEN_CONTROL_DECISIONS.csv> ^
  --control-quality-csv <FROZEN_CONTROL_QUALITY.csv> ^
  --output-json <FROZEN_OUTPUT_DIR/review_close_authority.json>
```

The gate requires one resolved decision and one quality row for every scope
item. It also verifies these semantics:

- `accept` means unchanged and non-technical;
- `corrected` means a changed label and confirmed source-label error;
- `exclude` means a technical exclusion;
- technical exclusions do not count as label changes.

All six files are path/hash-bound. Active Behavior ledger paths are rejected
before any read.

## Stage B2: freeze the mini-CVAT corrected-source chain

Identity apply manifests must be supplied in application order. The final CSV
and XML files must match the last after-hash in each uninterrupted chain.

```bat
set "POST_REVIEW=scripts\classification_v2\02_train_ready_exports"
python "%POST_REVIEW%\freeze_corrected_source_authority.py" ^
  --identity-apply-manifest-json <FIRST_APPLY_MANIFEST.json> ^
  --identity-apply-manifest-json <SECOND_APPLY_MANIFEST.json> ^
  --source-target <FINAL_SOURCE.csv> --source-target <FINAL_SOURCE.xml> ^
  --output-json <FROZEN_OUTPUT_DIR/corrected_source_authority.json>
```

If multiple corrections touched the same source, every next before-hash must
equal the previous after-hash. Only the last after-hash must equal the final
source file. This preserves valid sequential edits without falsely requiring
every historical intermediate hash to equal the final file.

## Stage C: learn from changed and unchanged labels

The feature list is explicit. There is no automatic selection of numeric
columns.

```bat
set "POST_REVIEW=scripts\classification_v2\02_train_ready_exports"
python "%POST_REVIEW%\analyze_post_review_learning.py" ^
  --review-close-authority-json <review_close_authority.json> ^
  --primary-scope-csv <FROZEN_PRIMARY_SCOPE.csv> ^
  --primary-quality-csv <FROZEN_PRIMARY_QUALITY.csv> ^
  --control-scope-csv <FROZEN_CONTROL_SCOPE.csv> ^
  --control-quality-csv <FROZEN_CONTROL_QUALITY.csv> ^
  --frame-features-csv <REVIEW_INDEPENDENT_FRAME_FEATURES.csv> ^
  --feature-column <FEATURE_1> --feature-column <FEATURE_2> ^
  --output-dir <POST_REVIEW_LEARNING_OUTPUT_DIR>
```

The output contains:

- original-to-reviewed transition counts;
- changed, unchanged, and technical-exclusion counts;
- behavior/source/review-template/error-pattern strata;
- primary selector precision;
- inverse-probability-weighted residual false-negative estimate;
- estimated selector recall within the explicit parent population;
- weighted Wilson intervals using Kish effective sample size;
- per-feature support, mean, median, missingness, and standardized difference
  between changed and unchanged labels.

These statistics diagnose selector and feature semantics. They do not prove
that a threshold or representation should change. Any proposed change needs an
independent validation or ablation on frozen data.

## Stage D: final integration preflight

The preflight is non-executing. It binds frozen review files, adjusted ROI,
corrected source authority, rebuilt frame features, and identity apply
manifests. Use repeated `--artifact NAME=PATH` arguments for all required names.

Required names are:

- `primary_scope`
- `primary_decisions`
- `primary_quality`
- `control_scope`
- `control_decisions`
- `control_quality`
- `adjusted_roi`
- `corrected_source_authority`
- `rebuilt_frame_features`

Mini-CVAT Behavior or Hidden updates block the preflight until an explicit
resolution is supplied. Bbox-only and identity-only corrections do not create
a label-authority conflict. Every identity target's after-hash must match the
`target_after_hashes` map in corrected source authority.

The successful status is `READY_FOR_REVIEWED_WINDOW_REBUILD`. It authorizes
only the next audited reconstruction step, not training or model promotion.

## Non-interference boundary

The code rejects any path resolving beneath:

```text
human_review_workspace\classification_v2\*\human_decisions\behavior\
```

No stage opens a GUI, changes active manifests, changes review membership,
applies decisions automatically, rebuilds canonical outputs automatically, or
starts training.
