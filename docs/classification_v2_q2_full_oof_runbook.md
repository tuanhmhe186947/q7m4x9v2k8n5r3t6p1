# classification_v2 Q2 Full OOF Runbook

This runbook defines the next paper-facing execution step after the current
engineering pilots. It keeps the claim boundary explicit: Q2 evidence requires
full native-temporal OOF learned evaluation, not smoke or pilot metrics.

## Current State

- B0 native majority baseline is recorded.
- B1 linear tabular whitelist native OOF baseline is recorded.
- B2 nonlinear tabular whitelist native OOF baseline is recorded.
- Ablation/shortcut reporting is recorded for comparable B0/B1/B2 metrics.
- Learned multimodal OOF runner has a bounded pilot record only.
- Full paper-facing learned multimodal record is still missing:
  `outputs/classification_v2/experiment_registry/full_multimodal_oof_record.json`

## Claim Boundary

Allowed claim after current state:

- Engineering readiness for multimodal actor image + spatial sequence +
  interaction-context training under native OOF bookkeeping.

Not allowed yet:

- Paper-facing learned-model performance claim.
- Q1/external generalization claim.
- Treating window rows as independent publication samples.
- Treating smoke or pilot metrics as full OOF metrics.

Allowed target claim after full run passes:

- Q2-strong: improved pig behavior recognition under session/video-safe
  native-temporal validation.

## Pre-Run Gates

Run these before full training:

```bat
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src

python scripts\classification_v2\06_full_oof_training\ ^
  check_classification_v2_full_learned_oof_contract.py
python scripts\classification_v2\07_postrun_evaluation\ ^
  check_classification_v2_ablation_reporting.py
python scripts\classification_v2\07_postrun_evaluation\ ^
  check_classification_v2_ablation_shortcut_contract.py ^
  --contract-json configs\classification_v2\ablation_shortcut_contract_v1.json ^
  --output-json outputs\classification_v2\model_design\ablation_shortcut_contract_audit.json
```

Expected before full run:

- `full_learned_oof_contract_audit.json`: `valid=true`, `paper_ready=false`.
- `ablation_reporting_audit.json`: `valid=true`.
- `ablation_shortcut_contract_audit.json`: `valid=true`.
- Remaining blocker is only full learned OOF record missing.

## Full Run Command

The command below intentionally requires `--full`; without it the runner stays
bounded and cannot be registered as full evidence.

```bat
python scripts\classification_v2\06_full_oof_training\ ^
  classification_v2_run_full_multimodal_oof.py ^
  --full ^
  --output-dir outputs\classification_v2\model_full\full_multimodal_oof ^
  --image-size 64 ^
  --hidden-dim 48 ^
  --epochs-per-fold 3 ^
  --bootstrap-iterations 200
```

Hardware notes:

- CPU is supported but may be slow because CVAT crops are read from video.
- GPU should be used when available through `--device auto`.
- Full mode ignores bounded pilot step counts and trains complete shuffled epochs.
- Confirm `train_row_coverage_ratio=1.0` for every fold before interpreting a run as paper-facing.
- If runtime is too high, reduce only training steps for a documented
  engineering run. Do not register a reduced/bounded run as full paper evidence.

## Post-Run Gates

Run:

```bat
python scripts\classification_v2\06_full_oof_training\ ^
  check_classification_v2_full_multimodal_oof.py ^
  --audit-json outputs\classification_v2\model_full\full_multimodal_oof\full_multimodal_oof_audit.json ^
  --predictions-csv outputs\classification_v2\model_full\full_multimodal_oof\full_multimodal_oof_predictions.csv ^
  --metrics-json outputs\classification_v2\model_full\full_multimodal_oof\full_multimodal_oof_metrics.json

python scripts\classification_v2\02_train_ready_exports\ ^
  check_classification_v2_native_temporal_metrics.py ^
  --metrics-json outputs\classification_v2\model_full\full_multimodal_oof\full_multimodal_oof_metrics.json ^
  --result-kind model_evaluation
```

Pass criteria:

- `run_mode=full`.
- `paper_facing_result=true`.
- Prediction schema valid.
- Native temporal metrics payload valid.
- Native temporal rows are nonzero and fold coverage is complete.
- No manual/review/source/path/ID/split/label leakage columns in prediction CSV.

## Registration

Register only after all post-run gates pass:

```bat
python scripts\classification_v2\08_publication_reporting\ ^
  classification_v2_register_experiment.py ^
  --name full_multimodal_oof ^
  --metrics-json outputs\classification_v2\model_full\full_multimodal_oof\full_multimodal_oof_metrics.json ^
  --artifact outputs\classification_v2\model_full\full_multimodal_oof\full_multimodal_oof_audit.json ^
  --artifact outputs\classification_v2\model_full\full_multimodal_oof\full_multimodal_oof_prediction_schema_audit.json ^
  --notes "full learned multimodal native OOF evaluation" ^
  --experiment-stage paper_facing_candidate ^
  --paper-facing ^
  --result-kind model_evaluation
```

Then verify:

```bat
python scripts\classification_v2\08_publication_reporting\ ^
  check_classification_v2_experiment_registry.py ^
  --record-json outputs\classification_v2\experiment_registry\full_multimodal_oof_record.json
```

The registry record must show:

- `git_dirty=false`.
- `paper_facing=true`.
- native temporal metrics gate valid.
- external generalization claim false.

## Artifact Policy

Commit:

- Source code and config changes.
- Small JSON audits and registry records.
- Small prediction schema audits.

Do not commit by default:

- Large full prediction CSVs.
- Large unit prediction CSVs.
- Model checkpoint files.
- Large tensor or cache outputs.

If a large artifact must be preserved, record its path, size, checksum, and
generation command in the registry/audit instead of force-adding it.

## Final Contract Update

After a clean full record exists:

1. Re-run `check_classification_v2_full_learned_oof_contract.py`.
2. Re-run `check_classification_v2_ablation_shortcut_contract.py`.
3. Confirm the full OOF blocker is removed only if the checker explicitly
   validates `full_multimodal_oof_record.json`.

Do not edit blockers manually to make the contract pass.

## Remaining Scientific Upgrades

- Full source-balanced metrics by source type and recording group.
- Branch ablation full OOF deltas: image only, spatial only, interaction
  context off, ROI/social feature subsets.
- Calibration/threshold analysis per behavior.
- Confusion-focused analysis for fight/social-nose, ROI-intent, posture, and
  move/explore/stand pairs.
- Reviewer-agreement audit before claiming label quality beyond single-review
  correction.
