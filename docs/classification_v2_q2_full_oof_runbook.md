# classification_v2 Q2 Full OOF Runbook

This is a future execution runbook, not the current next step. The active
lineage must first complete Hidden and behavior human review and freeze new
data/cache/fold hashes. See `CLASSIFICATION_V2_CURRENT_STATE.md`.

The commit-`18d6692` full run used the previous artifact lineage. Do not rerun,
register, or promote it as a substitute for reviewed-lineage evaluation.

## Current State

- B0 native majority baseline is recorded.
- B1 linear tabular whitelist native OOF baseline is recorded.
- B2 nonlinear tabular whitelist native OOF baseline is recorded.
- Ablation/shortcut reporting is recorded for comparable B0/B1/B2 metrics.
- A historical 13-fold learned run exists with 73,668 window predictions and
  32,727 native-unit predictions.
- The active Hidden v5 and behavior-review decisions are incomplete, so no new
  reviewed train-ready snapshot is available.
- This runbook becomes executable only after the current-state data and smoke
  gates pass and a new authorization is bound to their hashes.

## Claim Boundary

Allowed current statement:

- Hidden template coverage is audited, but human-review and train-ready gates
  remain incomplete.
- The old full OOF proves engineering wiring only for its historical lineage.

Not allowed yet:

- Paper-facing learned-model performance claim.
- Q1/external generalization claim.
- Treating window rows as independent publication samples.
- Treating smoke or pilot metrics as full OOF metrics.

Allowed target claim after full run passes:

- Q2-strong: improved pig behavior recognition under session/video-safe
  native-temporal validation.

## Pre-Run Gates

Run these only after the reviewed-data runbook and model smoke gates pass:

```bat
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src
set RUN_ID=reviewed_YYYYMMDD_HHMMSS
set OOF_OUT=outputs\classification_v2\model_full\%RUN_ID%

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

- Hidden and behavior decision coverage are complete and fail-closed audits pass.
- The reviewed snapshot, caches, whitelist, and folds share frozen hashes.
- One-batch, tiny-overfit, resume, runtime, and one-fold smoke gates pass.
- `full_learned_oof_contract_audit.json`: `valid=true`, `paper_ready=false`.
- `ablation_reporting_audit.json`: `valid=true`.
- `ablation_shortcut_contract_audit.json`: `valid=true`.
- A new authorization matches the reviewed hashes, clean code SHA, and run ID.

## Full Run Command

Do not run this command for the current incomplete rebuild. When all upstream
gates pass, generate a new output directory and authorization packet rather
than overwriting the historical artifact.

The command below intentionally requires `--full`; without it the runner stays
bounded and cannot be registered as full evidence.

```bat
python scripts\classification_v2\06_full_oof_training\ ^
  classification_v2_run_full_multimodal_oof.py ^
  --full ^
  --output-dir %OOF_OUT% ^
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
  --audit-json %OOF_OUT%\full_multimodal_oof_audit.json ^
  --predictions-csv %OOF_OUT%\full_multimodal_oof_predictions.csv ^
  --metrics-json %OOF_OUT%\full_multimodal_oof_metrics.json

python scripts\classification_v2\02_train_ready_exports\ ^
  check_classification_v2_native_temporal_metrics.py ^
  --metrics-json %OOF_OUT%\full_multimodal_oof_metrics.json ^
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
  --name %RUN_ID% ^
  --metrics-json %OOF_OUT%\full_multimodal_oof_metrics.json ^
  --artifact %OOF_OUT%\full_multimodal_oof_audit.json ^
  --artifact %OOF_OUT%\full_multimodal_oof_prediction_schema_audit.json ^
  --notes "full learned multimodal native OOF evaluation" ^
  --experiment-stage paper_facing_candidate ^
  --paper-facing ^
  --result-kind model_evaluation
```

Then verify:

```bat
python scripts\classification_v2\08_publication_reporting\ ^
  check_classification_v2_experiment_registry.py ^
  --record-json outputs\classification_v2\experiment_registry\%RUN_ID%_record.json
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
   validates the new `%RUN_ID%_record.json` and its bound artifact hashes.

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
