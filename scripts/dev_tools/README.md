# dev_tools Script Index

This folder contains validators, audits, and diagnostic checks. Most scripts
read artifacts and fail closed; they should not modify raw data.

Use this README to locate checks by workflow block.

## One-Shot Pre-Full Gate

Use this first when asking whether full OOF can run:

```bat
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src
C:\Users\ironh\anaconda3\envs\pig_project\python.exe ^
  scripts\dev_tools\check_classification_v2_full_readiness_once.py
```

Expected pre-authorization status:

- `valid=true`
- `status=PASS_PRE_FULL_READY_AUTHORIZATION_REQUIRED`
- `errors=[]`

If this passes, do not rerun every individual pre-full checker unless a source
artifact changed.

## Data And Review Checks

Purpose: verify data lineage, review decisions, GUI decisions, and template
coverage.

Representative scripts:

- `check_review_unit_template_coverage.py`
- `check_review_unit_gui_decisions.py`
- `check_apply_review_unit_decisions_output.py`
- `check_classification_v2_review_unit_contracts.py`
- `diagnose_cvat_unit_label_mismatch.py`
- `diagnose_gui_video_loading.py`

Use when:

- changing review policy
- changing GUI writer/apply script
- rebuilding review unit manifests
- debugging CVAT anchor or video resolver behavior

## Train-Ready And Leakage Checks

Purpose: verify X/y/mask/weight/split contracts and leakage-safe model inputs.

Representative scripts:

- `check_classification_v2_training_snapshot.py`
- `check_classification_v2_data_module.py`
- `check_classification_v2_loader_inputs.py`
- `check_classification_v2_feature_semantics.py`
- `check_classification_v2_event_weights.py`
- `check_classification_v2_native_oof_folds.py`
- `check_classification_v2_q2_progress_report.py`

Use when:

- adding features
- changing train-ready exports
- changing fold/split logic
- changing the feature whitelist or denylist

## Image, Cache, And Loader Checks

Purpose: verify actor image cache, visual context cache, and sequence loaders.

Representative scripts:

- `check_classification_v2_image_cache.py`
- `check_classification_v2_image_cache_integrity.py`
- `check_classification_v2_image_cache_inventory.py`
- `check_classification_v2_image_loader.py`
- `check_classification_v2_image_sequence_loader.py`
- `check_classification_v2_image_tensor_loader.py`
- `check_classification_v2_visual_interaction_cache.py`
- `check_classification_v2_interaction_context_loader.py`

Use when:

- modifying video/crop resolvers
- changing letterbox/image preprocessing
- rebuilding packed actor or visual-context caches
- debugging missing image/context rows

## Model, Baseline, And Smoke Checks

Purpose: validate architecture contracts, baseline artifacts, and bounded smoke
training results.

Representative scripts:

- `check_classification_v2_baseline_configs.py`
- `check_classification_v2_behavior_only_baselines.py`
- `check_classification_v2_multimodal_forward.py`
- `check_classification_v2_multimodal_smoke_train.py`
- `check_classification_v2_multimodal_ablation_pilot.py`
- `check_classification_v2_multitask_forward.py`
- `check_classification_v2_checkpoint.py`
- `check_classification_v2_model_architecture_contract.py`

Use when:

- changing model architecture
- changing smoke configs
- changing checkpoint or prediction schemas

## Full OOF Gate Checks

Purpose: keep the long full OOF run fail-closed until human authorization and
postrun artifacts are valid.

Representative scripts:

- `preflight_classification_v2_full_multimodal_oof.py`
- `check_classification_v2_full_readiness_once.py`
- `check_classification_v2_full_oof_authorization_file.py`
- `check_classification_v2_full_oof_authorization_writer.py`
- `check_classification_v2_full_oof_preflight_freshness.py`
- `check_classification_v2_full_oof_launch_packet.py`
- `check_classification_v2_full_oof_execution_gate.py`
- `check_classification_v2_full_oof_completion_gate.py`
- `check_classification_v2_full_oof_postrun_registration_packet.py`

Use `check_classification_v2_full_readiness_once.py` as the summary checker.
Run the individual scripts only after changing their corresponding packet,
writer, or gate logic.

## Postrun, Registry, And Paper-Safety Checks

Purpose: verify metrics, experiment registration, ablations, and claim gates.

Representative scripts:

- `check_classification_v2_full_multimodal_oof.py`
- `check_classification_v2_native_temporal_metrics.py`
- `check_classification_v2_source_balanced_report.py`
- `check_classification_v2_experiment_registry.py`
- `check_classification_v2_ablation_reporting.py`
- `check_classification_v2_ablation_shortcut_contract.py`
- `check_classification_v2_paper_grade_protocol.py`

Use after:

- full OOF prediction artifacts exist
- calibration and confusion outputs exist
- registry record is written

## Future Folder Migration

Do not rename or move these checks before the full-like smoke path is stable.
When migration starts:

1. Add new wrappers under `scripts/classification_v2/audits/`.
2. Keep old paths working.
3. Update launch/postrun packet writers.
4. Re-run `check_classification_v2_full_readiness_once.py`.
5. Remove wrappers only after grep confirms no old path references remain.
