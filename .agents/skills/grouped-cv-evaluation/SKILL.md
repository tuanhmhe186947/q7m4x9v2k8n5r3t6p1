---
name: grouped-cv-evaluation
description: >-
  Evaluate classification_v2 with one recording/video-safe fold manifest and
  native temporal-unit predictions. Use for OOF metrics, prediction comparison,
  paired uncertainty, confusion analysis, and fold-support audits.
---

# Grouped Cv Evaluation

## Purpose

Produce comparable 10-class evidence without treating overlapping windows as
independent observations or hiding unsupported fold classes.

## When to use

Invoke for OOF aggregation, two-model prediction comparison, metric generation,
confidence intervals, calibration review, or class/fold coverage analysis.

## Project context

Evaluate ten labels on CVAT six-frame and legacy 16-frame native units. Window
predictions are training artifacts; the primary metric is pooled OOF native-unit
macro-F1 using the global class order.

## Required inputs

- immutable fold manifest shared by all compared models;
- native-unit and window manifests with explicit key lineage;
- prediction files linked to checkpoint, fold, config, and label order;
- recording/session, video, source, quality, and class metadata;
- predeclared confusion pairs and aggregation policy.

## Scientific invariants

- Use identical held-out units and fold roles for paired comparisons.
- Aggregate windows to exactly one prediction per native unit.
- Report missing fold classes rather than imputing or hiding them.
- Compute fold macro-F1 only over supported classes.
- Compute primary pooled macro-F1 over all ten global classes.
- Cluster uncertainty by recording or native event, not overlapping window.
- Never use outer-fold predictions for architecture or threshold tuning.

## Ordered procedure

1. Validate model lineage and exact fold-manifest equality.
2. Validate prediction schemas, class order, unique keys, and expected counts.
3. Collapse window predictions using the frozen aggregation policy.
4. Reconcile every native unit and unresolved multi-label conflict.
5. Build pooled global, per-class, grouped, source, video, and session metrics.
6. Publish the class-by-fold support matrix and supported-class fold metrics.
7. Compute confusion matrices and predeclared confusion-pair statistics.
8. Run paired cluster bootstrap or another predeclared paired analysis.
9. Record warnings, missing predictions, and aggregation coverage.

## Required outputs

Produce `oof_predictions.csv`, `native_unit_predictions.csv`,
`metrics_global.json`, `metrics_per_class.csv`, `metrics_per_group.csv`,
`confusion_matrix.csv`, `class_fold_support.csv`, `paired_comparison.json`, and
`evaluation_audit.json`.

## Validation commands

Use [prediction-count audit](../checks/audit_prediction_count.py),
[native-unit audit](../checks/audit_native_unit_uniqueness.py), and
[split audit](../checks/audit_split_overlap.py). Use bounded synthetic files for
skill installation checks; do not start evaluation training.

## Stop conditions

Stop when fold manifests differ, prediction counts mismatch, aggregation loses a
native unit, one unit has unresolved ground-truth labels, or outer predictions
have influenced model selection.

## Forbidden actions

Do not random-split predictions, compare unequal held-out sets, report only
supported easy classes, bootstrap windows independently, tune on outer OOF, or
drop missing predictions to improve metrics.

## Completion report format

Use the shared [completion report](../templates/skill_completion_report.md) and
local [metric contract](templates/metric_contract.json). Include pooled and
supported-fold definitions, counts, uncertainty method, warnings, and PASS/FAIL.
