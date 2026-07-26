# Classification V2 — Balanced Causal Model: Implementation Readiness

Scaffolding for `BALANCED_CAUSAL_MAIN_MODEL`. This document describes what was
built, how to run it, and what remains blocked by the Behavior review gate.

| Field | Value |
|---|---|
| BASE_GIT_SHA | `4fc94c2d3c9ebba5d6dc2ba6ad0aa5e7149e9c98` |
| Branch | `agent/balanced_model_scaffold_r1` |
| LABEL_AUTHORITY_STATUS | `PROVISIONAL_TRUSTED_PRE_BEHAVIOR_REVIEW` |
| PAPER_GRADE_USE | **NO** |
| Production data used | **NO** — synthetic tensors and in-test fixtures only |
| Real model metrics produced | **NO** |

Design authority: `CLASSIFICATION_V2_BALANCED_MODEL_SCIENTIFIC_PROTOCOL.md`.
This document is the implementation counterpart, not a second specification.

---

## 1. Worktree isolation

The work was developed on a separate branch in a separate Git worktree so that
the active production lineage in the primary worktree was never checked out,
reset, stashed, or edited.

```
BASE_GIT_SHA   4fc94c2d3c9ebba5d6dc2ba6ad0aa5e7149e9c98
BRANCH_NAME    agent/balanced_model_scaffold_r1
WORKTREE_PATH  C:\Users\ironh\Downloads\PIG_Behavior_Project_balanced_scaffold
```

Everything added is a **new file**. No existing module, config, manifest,
review decision, or production artifact was modified. No production run root
was read or written, and no production pipeline stage was started.

---

## 2. Package structure

New code lives in subpackages so it cannot be confused with the existing
production modules that carry similar names.

```
src/pig_behavior/classification_v2/
  models/balanced/
    contracts.py        schema-driven batch/tensor contract + validator
    temporal.py         causal temporal primitives (TCN / GRU / transformer)
    visual.py           shared per-frame visual encoder interface
    numeric.py          grouped numeric encoders + control-mask handling
    fusion.py           fusion modes, 10-class head, masked auxiliary head
    balanced_model.py   the composable model used by every baseline
    baselines.py        B0-B3 configurations
    registry.py         model registry
    synthetic.py        deterministic synthetic batches for tests
  training/balanced/
    class_priors.py     train-fold native-unit priors (fail-closed on role)
    losses.py           loss registry L0-L7 + component harness
    sampling.py         natural / class-aware / native-unit-balanced sampling
    training_mass.py    overlapping-window training-mass correction + audit
  splits/
    date_grouped_split.py   FOUR_FOLD_DATE_GROUPED_OUTER_CV builder
    split_audit.py          the eight split audits
  temporal_views/
    registry.py           canonical model-side view names and semantics
    builder_contract.py   what a view builder must satisfy
    matched_cohort.py     ALL_ELIGIBLE / COMMON_MATCHED_COHORT utilities

scripts/classification_v2/model_research/
  snapshot_completed_run.py   read-only snapshot of a COMPLETED stage
```

Two naming clarifications, because the repository already has modules with
similar names:

* `classification_v2/temporal_views/` is the **model-side** canonical view
  registry (`T6_TARGET_CONTIGUOUS`, …). `features/temporal_views.py` remains the
  **data-side** fixed-six slot manifest builder used by the lineage.
* `models/balanced/` does not replace `models/model_factory.py`. The existing
  factory keeps serving the legacy mode registry; the balanced package is the
  causal research ladder.

Schema authorities are reused, never re-declared:
`models/balanced/contracts.py` imports `MOTION_FEATURE_NAMES` from
`features/motion_schema.py` and the remaining group names from
`spatial_sequence_export.SPATIAL_FRAME_FEATURES`.

---

## 3. Model registry

`models.balanced.registry` maps a scientific name to a validated configuration.

| Name | Implemented | Temporal view |
|---|---|---|
| `B0_ACTOR_SINGLE_FRAME` | yes | endpoint frame only |
| `B1_ACTOR_T6_SEQUENCE` | yes | `T6_TARGET_CONTIGUOUS` |
| `B2_ACTOR_T6_PLUS_GEOMETRY` | yes | `T6_TARGET_CONTIGUOUS` |
| `B3_ACTOR_T6_PLUS_GEOMETRY_MOTION` | yes | `T6_TARGET_CONTIGUOUS` |
| `BALANCED_CAUSAL_MAIN_MODEL` | **no** | `T6_TARGET_PLUS_H12` |

`build_model("BALANCED_CAUSAL_MAIN_MODEL")` raises
`FusionExtensionPointError` naming the modules it still needs. That is
deliberate: an unbuilt module must fail loudly rather than silently degrade to
concatenation. The declared extension points are ROI-conditioned FiLM,
actor–partner relation tokens, two-timescale causal history, and quality-aware
gated fusion.

---

## 4. Tensor contract

`ModelBatch` keeps the target and history segments separate by construction, so
the two masks can never be conflated. Each segment carries a `valid_mask` and an
integer `frame_offsets` tensor expressed **relative to the prediction
endpoint**: the endpoint is offset `0`, earlier frames are strictly negative,
and any positive offset is future information.

Validator checks, all reported by name:

```
MOTION_DIMENSION_CONTRACT   width == len(canonical motion names); sidecar
                            schema hash/version must match
FEATURE_ORDER_CONTRACT      missing / extra / duplicated / reordered names
MASK_SHAPE_CONTRACT         mask ranks, binary values, control widths,
                            missing-modality availability
TARGET_LENGTH_CONTRACT      declared length, endpoint semantics, no future frame
HISTORY_LENGTH_CONTRACT     declared length, history strictly before the target
FORBIDDEN_FEATURE_CONTRACT  labels, ids, paths, split/review/target-ROI fields
FINITE_VALUE_CONTRACT       no NaN/Inf anywhere in a predictive tensor
BATCH_ALIGNMENT_CONTRACT    consistent B across tensors, metadata and labels
```

Rules the contract enforces rather than assumes:

* The motion dimension is `len(MOTION_FEATURE_NAMES)`. A test walks the AST of
  `contracts.py` and fails if a literal `12` appears anywhere in it.
* Quality and availability masks are controls. They are encoded through a
  separate `ControlMaskEncoder` and never counted in the predictive width.
  `SPATIAL_PREDICTIVE_DIMENSION == 46` = `4 + 2 + 12 + 18 + 10`.
* A maskable modality that is absent must come with an explicit availability
  mask; a missing modality is never readable as valid zero evidence.
* Labels, `native_unit_id`, and `window_id` travel as metadata and are never
  routed into any encoder.

---

## 5. Baselines B0–B3

A baseline is a configuration, not a class, so a measured ladder difference
cannot be an implementation difference.

| Baseline | Inputs | Numeric groups |
|---|---|---|
| B0 | actor image at the causal endpoint | — |
| B1 | `T6_TARGET_CONTIGUOUS` actor images | — |
| B2 | B1 + geometry | `bbox_xywh_n`, `bbox_shape_n` |
| B3 | B2 + motion | + `motion_delta` |

Geometry and motion pass through `GroupedNumericEncoder`
(`Linear → LayerNorm → GELU → Dropout` per group) before fusion; nothing is
concatenated raw into the classifier. No baseline uses ROI, social, causal
history, or gated fusion. Any baseline other than B0 can be re-instantiated at
T8/T12/T16 with `target_length=` without changing another component.

---

## 6. Loss registry

`training.balanced.losses` implements a common interface returning separated
components:

```
unreduced_per_sample_loss -> class_weight -> native_unit_mass_weight
                          -> final_sample_weight -> reduced_loss
```

| Candidate | Status | Required explicit configuration |
|---|---|---|
| `L0_STANDARD_CROSS_ENTROPY` | implemented | — |
| `L1_WEIGHTED_CROSS_ENTROPY` | implemented | `weighting_strategy`, `weight_cap` when capped |
| `L2_EFFECTIVE_NUMBER_CLASS_BALANCED` | implemented | `effective_number_beta` |
| `L3_BALANCED_SOFTMAX` | implemented | (tau fixed at 1.0) |
| `L3_LOGIT_ADJUSTMENT` | implemented | `tau` |
| `L4_FOCAL_LOSS` | implemented | `focal_gamma`; alpha only when configured |
| `L5_LDAM_DRW` | implemented | `ldam_max_margin`, `drw_start_epoch`, `drw_beta` |
| `L6_CLASSIFIER_RETRAINING_SUPPORT` | support only | `ClassifierRetrainingConfig`, `tau_normalize_classifier` |
| `L7_CONTROLLED_SAMPLING_SUPPORT` | support only | `training.balanced.sampling` |

No beta is globally selected for you: `L2` refuses to build without an explicit
value, so the beta grid stays a documented experimental choice.

---

## 7. Class priors and native-unit training mass

`compute_class_priors` accepts only `role=TRAIN_FOLD_NATIVE_UNITS`. Validation,
test, all-data, source-box counts and uncorrected window rows are each rejected
by name with the reason. Duplicated `native_unit_id` values are rejected, which
is what stops overlapping windows from being counted as independent evidence.

Two permitted corrections are implemented and audited:

* `FIXED_WINDOWS_PER_NATIVE_UNIT` — a fixed window budget per unit per epoch.
* `PER_WINDOW_WEIGHTING` —
  `sample_weight = native_unit_class_weight / windows_from_the_same_unit`.

`training_mass_audit` reports `WINDOWS_PER_NATIVE_UNIT_DISTRIBUTION`,
`NATIVE_UNIT_TRAINING_MASS_BEFORE/AFTER_CORRECTION`,
`CLASS_TRAINING_MASS_BEFORE/AFTER_CORRECTION` and
`MAX_MIN_NATIVE_UNIT_MASS_RATIO`. A test builds two native units of the same
class with seven and two windows and asserts their intended training mass is
equal after correction (ratio 3.5 before, 1.0 after).

---

## 8. Split authority

`OUTER_PROTOCOL = FOUR_FOLD_DATE_GROUPED_OUTER_CV`. This is **not** full
leave-one-date-out, and the builder refuses any other protocol name.

```
FOLD_1  291119
FOLD_2  301119
FOLD_3  281119
FOLD_4  the remaining small legacy dates, pooled
```

The builder emits `outer_date_group_id`, `session_group_id`,
`inner_recording_group_id`, `outer_fold_id` and `inner_fold_id` from metadata
tables only. `101219a` and `101219b` collapse to calendar group `101219`;
cross-source dates `281119` and `291119` keep all their CVAT and legacy units in
one fold. Inner validation buckets recording groups through a stable SHA-256 of
the group id and the seed, so the assignment depends on metadata, seed and
configuration but never on the Git SHA or row order.

Fourteen date tokens collapse to thirteen calendar groups; eleven small legacy
tokens (ten calendar groups) pool into `FOLD_4`. Per-date descriptive
statistics are produced for every calendar group with
`high_power_independent_test = False`, because four outer folds is low power and
the tiny dates are not independent high-power tests.

All eight audits (`NO_CALENDAR_DATE_SPANS_OUTER_FOLDS` …
`FOLD_SUPPORT_REPORTED`) run in `split_audit.audit_split_authority`.

---

## 9. Temporal-view registry

Exact canonical names, with offsets relative to the prediction endpoint:

| View | Family | Target | History | Notes |
|---|---|---:|---:|---|
| `T6/T8/T12/T16_TARGET_CONTIGUOUS` | `TARGET_CONTIGUOUS` | 6/8/12/16 | 0 | trailing causal |
| `T6_TARGET_PLUS_H6/H12/H24` | `TARGET_PLUS_CAUSAL_HISTORY` | 6 | 6/12/24 | label from target only |
| `S6_AT_16_SPARSE` | `LEGACY_SPARSE_ABLATION` | 6 | 0 | burst offsets `[0,3,6,9,12,15]`, pair deltas `[3,3,3,3,3]` |
| `HISTORICAL_C6_SCREEN` | `HISTORICAL_SCREEN` | 6 | 0 | burst offsets `[5,6,7,8,9,10]`, metrics not transferable |

The two legacy views are distinguishable only at burst level — relative to its
own endpoint the historical screen looks contiguous — so the registry stores
`legacy_burst_offsets` and `views_are_distinct` compares them. The ambiguous
name `6c` is not used anywhere.

`builder_contract.validate_window` checks exact offsets, strictly increasing
real-elapsed timestamps, absence of future frames, single-label targets, history
strictly preceding the target start, stable actor authority and split group,
deterministic window ids, and that no label or review field appears in
`model_input_fields`. Insufficient history produces a masking warning rather
than silent padding.

`matched_cohort` builds `ALL_ELIGIBLE` per view and `COMMON_MATCHED_COHORT` as
the exact intersection. `length_conclusion_guard` refuses to call a longer view
successful without a matched-cohort gain that clears the preregistered minimum
effect, and raises when someone tries to compare a target-view length against a
causal-history length — those are different claims and are never pooled.

---

## 10. Commands (Windows)

All commands assume the repository root. The project virtual environment is
`.venv` (Python 3.11, torch CPU build).

**Activate the environment (PowerShell):**

```powershell
cd C:\Users\ironh\Downloads\PIG_Behavior_Project
.\.venv\Scripts\Activate.ps1
```

**Run the new targeted tests:**

```powershell
python -m pytest tests/test_classification_v2_balanced_model_tensor_contract.py `
                 tests/test_classification_v2_balanced_model_baselines.py `
                 tests/test_classification_v2_balanced_model_causality.py `
                 tests/test_classification_v2_balanced_motion_schema_binding.py `
                 tests/test_classification_v2_balanced_imbalance_losses.py `
                 tests/test_classification_v2_balanced_native_unit_training_mass.py `
                 tests/test_classification_v2_balanced_date_grouped_split.py `
                 tests/test_classification_v2_balanced_temporal_view_contract.py `
                 tests/test_classification_v2_balanced_temporal_matched_cohort.py `
                 tests/test_classification_v2_balanced_synthetic_tiny_overfit.py `
                 tests/test_classification_v2_balanced_run_snapshot.py -q
```

Or, more briefly:

```powershell
python -m pytest tests/test_classification_v2_balanced_*.py -q
```

**Lint the changed files:**

```powershell
.\ruff.exe check src\pig_behavior\classification_v2\models\balanced `
                 src\pig_behavior\classification_v2\training\balanced `
                 src\pig_behavior\classification_v2\splits `
                 src\pig_behavior\classification_v2\temporal_views `
                 scripts\classification_v2\model_research `
                 tests\test_classification_v2_balanced_*.py
```

**Overlong-line scan before commit (repository rule):**

```powershell
rg -n "^.{101,}$" --glob "*.py" src\pig_behavior\classification_v2\models\balanced `
   src\pig_behavior\classification_v2\training\balanced `
   src\pig_behavior\classification_v2\splits `
   src\pig_behavior\classification_v2\temporal_views
```

**Type check.** No type checker is configured in `pyproject.toml` or
`.pre-commit-config.yaml`; `TYPECHECK_STATUS=NOT_CONFIGURED`. If one is adopted
later, the balanced package is fully annotated and ready for it.

**CPU synthetic smoke (forward/backward for the whole ladder):**

```powershell
python -c "import torch; from pig_behavior.classification_v2.models.balanced.registry import build_model; from pig_behavior.classification_v2.models.balanced.baselines import BASELINE_NAMES; from pig_behavior.classification_v2.models.balanced.synthetic import SyntheticBatchSpec, synthetic_batch; [print(n, tuple(build_model(n, hidden_dim=32)(synthetic_batch(SyntheticBatchSpec(contract=build_model(n, hidden_dim=32).config.batch_contract, batch_size=2, image_size=16)))['logits'].shape)) for n in BASELINE_NAMES]"
```

**Optional small CUDA smoke — only after the production Pig-STRENet process has
finished.** The CUDA test is skipped automatically when CUDA is unavailable
(which is the case on the current CPU-only torch build):

```powershell
python -m pytest tests/test_classification_v2_balanced_synthetic_tiny_overfit.py::test_optional_small_cuda_smoke -q
```

**Generate a run snapshot from an explicitly supplied COMPLETED manifest:**

```powershell
python scripts\classification_v2\model_research\snapshot_completed_run.py `
  --run-root "<COMPLETED_RUN_ROOT>" `
  --manifest-path "<COMPLETED_RUN_ROOT>\manifest.json" `
  --log-path "<COMPLETED_RUN_ROOT>\stage.log" `
  --command-line "<exact command that produced the run>" `
  --expected-stage native_evidence `
  --output snapshot_native_evidence.json
```

The tool exits non-zero with `SNAPSHOT_REFUSED` when the stage is still
`RUNNING`. `--allow-running-metadata-only` produces a metadata-only record that
is never hashed, never marked PASS, and never canonical.

No command in this document starts `behavior_review_units`, the production
Behavior GUI, behavior decision apply, production training, or a final
temporal-view export.

---

## 11. Integration by the production-lineage owner

Everything added is additive. To adopt it:

1. Point the loader at `models.balanced.contracts.ModelBatch` and call
   `require_batch(batch, model.config.batch_contract)` once per epoch on the
   first batch. The validator names the failing contract, so a schema drift
   report is actionable without reading model code.
2. Feed the reviewed metadata table to `splits.build_split_authority` and store
   `outer_date_group_id`, `session_group_id`, `inner_recording_group_id`
   alongside the train-ready rows. Run `require_split_authority` before fitting.
3. Fit `compute_class_priors` per outer fold on that fold's training native
   units, then build the loss from `LossConfig`.
4. Build a `WindowInventory` from the training-fold windows and apply either
   mass-correction strategy; attach `training_mass_audit` to the run manifest.
5. Take the target/history frame indices from
   `temporal_views.expected_frame_indices` so the production builder and the
   model agree on the same definition of a view.

---

## 12. What remains blocked

**Blocked until Pig-STRENet passes:** the production behavior-review population
build. Nothing in this package depends on that run completing.

**Blocked until Behavior review decisions are applied:**

* final train-ready data;
* production temporal-view exports (T6/T8/T12/T16 and the legacy ablations);
* real exploratory training on the reviewed lineage;
* any paper-grade experiment or headline number.

The synthetic tiny-overfit test proves the optimization path runs. It says
nothing about model quality: the data is separable by construction, and no
production label, media, or run root is involved.
