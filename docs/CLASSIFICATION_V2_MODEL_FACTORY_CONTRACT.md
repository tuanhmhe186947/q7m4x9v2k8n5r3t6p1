# Classification V2 Model Factory Contract

## Scope

This contract covers synthetic model construction and forward correctness. It
does not authorize training, one-fold evaluation, or full OOF. The production
visual backbone and temporal-view tensor loader remain separate milestones.

The final behavior head always emits ten logits in canonical label order. An
auxiliary head never replaces direct behavior supervision and its argmax never
enters the final head.

## Model Modes

| Mode | Actor | Spatial groups | Numeric partner | Union crop | Auxiliary |
|---|---:|---|---:|---:|---:|
| `actor_only` | yes | none | no | no | no |
| `actor_temporal` | yes | none | no | no | no |
| `actor_geometry` | yes | geometry and quality | no | no | no |
| `actor_geometry_motion` | yes | geometry, motion, quality | no | no | no |
| `actor_geometry_roi` | yes | geometry, motion, ROI, quality | no | no | no |
| `actor_geometry_roi_social` | yes | all spatial groups | no | no | no |
| `actor_partner_union` | yes | none | no | yes | no |
| `full_multimodal` | yes | all spatial groups | yes | yes | no |
| `full_multimodal_hierarchy` | yes | all spatial groups | yes | yes | yes |
| `spatial_only_control` | no | all spatial groups | no | no | no |

`actor_only` is the masked-mean non-temporal visual control. The remaining
modes may use `masked_mean`, `masked_attention`, `masked_tcn`, or
`small_transformer` when their input contract is satisfied.

The factory validates the exact branch flags and ordered spatial groups. A
config cannot declare one mode and silently enable another modality.

## Tensor And Mask Rules

Actor and union images use `[B,T,3,H,W]`. Spatial groups use `[B,T,D]`.
Partner features use `[B,D]` or `[B,K,D]`. Every temporal branch has structural
length, observed, available, and quality masks shaped `[B,T]`. Partner masks
use `[B]` or `[B,K]`.

Observed slots cannot extend past length. Available and quality slots cannot
extend past observation. Values in an effective masked slot are zeroed before
any convolution, projection, or attention operation. NaN values in valid slots
fail immediately; NaN values in masked slots cannot change logits.

Availability masks only gate an embedding. They are not concatenated into X.
Missingness shortcut controls and label-independent modality dropout remain
required before a context candidate can be promoted.

## Temporal Contract

`masked_mean`, `masked_attention`, and `masked_tcn` do not invent timing.
`small_transformer` requires a real non-negative `time_delta` tensor matching
each branch mask. Missing-slot deltas may be NaN because those slots are
explicitly masked.

The current model API supports branch-specific timing. The strict training data
module does not yet load `time_delta` from the fixed-six slot manifest. Until
that loader and its hash lineage pass, Transformer model tests are technical
evidence only and Transformer training must stay blocked.

## Visual Backbone Contract

Only `smoke_cnn` is implemented in the current factory. A ResNet name fails
before construction, so the system cannot silently substitute the smoke CNN or
download a weight. ResNet18/160, ResNet18/224, and ResNet34/224 interfaces are
the next P1 visual milestone and must record an exact weight enum.

## Lineage

The current strict checkpoint schema is
`classification_v2_training_checkpoint_v3`. The run identity and append-only
registry include `model_mode`; the full resolved config hash also binds temporal
encoder parameters and every legacy branch flag. Registry v2 writes to
`runs_registry_v2.csv`, preserving the prior registry as historical evidence.

## Validation

Run the synthetic factory audit from CMD:

```bat
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src
set PY=C:\Users\ironh\anaconda3\envs\pig_project\python.exe
%PY% scripts\classification_v2\04_baselines_smokes\check_classification_v2_model_factory.py ^
  --dry-run
```

The checker performs zero optimizer steps, downloads no weights, builds all ten
modes and four temporal encoders, and requires finite `[2,10]` behavior logits.
