# behavior_review_tools Script Index

This folder contains executable pipeline builders and training/postrun commands
for `classification_v2`. Paths are intentionally preserved before full OOF so
launch packets and existing audits do not break.

Use this README to find the right script by workflow block.

## Block 1: Source, Feature, And Temporal Build

Purpose: build derived data from raw annotations and reviewed frame features.

Key scripts:

- `classification_v2_merge_sources.py`
- `classification_v2_build_geometry_features.py`
- `classification_v2_build_roi_features.py`
- `classification_v2_build_enhanced_spatiotemporal_features.py`
- `classification_v2_build_temporal_harmonization.py`
- `classification_v2_build_sequence_windows.py`

Typical outputs:

- `outputs/classification_v2/frame_features/*`
- `outputs/classification_v2/sequence_features/*`
- `outputs/classification_v2/sequence_features_reviewed/*`

## Block 2: Review Units And Decisions

Purpose: create review units, run GUI review, and apply decisions without row
loss or raw-data modification.

Key scripts:

- `classification_v2_build_review_units.py`
- `build_behavior_review_templates.py`
- `review_temporal_unit_gui.py`
- `classification_v2_apply_review_unit_decisions.py`
- `classification_v2_apply_review_policy.py`

Typical outputs:

- `outputs/classification_v2/review_units/*`
- `outputs/classification_v2/review_policy/reviewed_frame_features.csv`
- `outputs/classification_v2/review_policy/apply_review_unit_decisions_audit.json`

## Block 3: Train-Ready Exports

Purpose: export leakage-safe model inputs, labels, masks, weights, and splits.

Key scripts:

- `classification_v2_export_train_ready_windows.py`
- `classification_v2_export_spatial_sequences.py`
- `classification_v2_build_class_weights.py`
- `classification_v2_build_event_weights.py`
- `classification_v2_build_grouped_splits.py`
- `classification_v2_build_native_oof_folds.py`

Typical outputs:

- `outputs/classification_v2/train_ready_windows/*`
- `outputs/classification_v2/native_temporal_units_oof_folds/*`

## Block 4: Image And Context Cache

Purpose: prepare reusable image tensors so full training does not repeatedly
seek video frames, crop bboxes, resize, or convert BGR/RGB.

Key scripts:

- `classification_v2_build_image_cache.py`
- `classification_v2_build_packed_image_cache.py`
- `classification_v2_build_image_cache_integrity.py`
- `classification_v2_build_image_context_index.py`
- `classification_v2_build_visual_interaction_cache.py`

Canonical outputs:

- `outputs/classification_v2/image_cache_v2_letterbox/*`
- `outputs/classification_v2/visual_interaction_cache/*`

Policy:

- Actor crops are letterboxed, not square-stretched.
- Cache artifacts are derived outputs under `outputs/`, never raw data.

## Block 5: Baselines, Smoke, And Ablations

Purpose: run bounded controls and engineering checks before full OOF.

Key scripts:

- `classification_v2_run_native_majority_baseline.py`
- `classification_v2_run_tabular_linear_baseline.py`
- `classification_v2_run_tabular_nonlinear_baseline.py`
- `classification_v2_run_q2_baseline_smokes.py`
- `classification_v2_multimodal_smoke_train.py`
- `classification_v2_run_multimodal_ablation_pilot.py`
- `classification_v2_multitask_smoke_train.py`
- `classification_v2_spatial_tcn_smoke_train.py`

Policy:

- Smoke and pilot metrics are engineering evidence only.
- Do not register a reduced run as full paper-facing evidence.

## Block 6: Full OOF And Postrun

Purpose: run the paper-facing learned multimodal OOF path after authorization.

Key scripts:

- `classification_v2_run_full_multimodal_oof.py`
- `classification_v2_cross_fit_calibration.py`
- `classification_v2_compare_confusion_focus.py`
- `classification_v2_build_source_balanced_report.py`
- `classification_v2_register_experiment.py`

Canonical output:

- `outputs/classification_v2/model_full/full_multimodal_oof/*`

Required order:

1. One-shot readiness audit from `scripts/dev_tools`.
2. Human authorization.
3. Short full-like smoke.
4. Full OOF.
5. Calibration, confusion comparison, registry, completion gate.

## Future Folder Migration

Do not move these scripts before the first full-like smoke path is stable.
Future wrappers may be added under `scripts/classification_v2/`, but old paths
must remain valid until all packet writers and docs are updated.
