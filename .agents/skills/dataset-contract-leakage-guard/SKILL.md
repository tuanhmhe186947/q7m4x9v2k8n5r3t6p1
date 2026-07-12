---
name: dataset-contract-leakage-guard
description: >-
  Audit classification_v2 data, folds, temporal units, and explicit model-X
  features. Use before training, baseline, prediction comparison, dataset export,
  split changes, normalization, weights, source controls, or leakage review.
---

# Dataset Contract And Leakage Guard

## Purpose

Fail closed when data lineage, grouped splits, temporal-unit semantics, or
inference-safe feature selection are not proven.

## When to use

Invoke before creating folds, windows, loaders, normalization, weights, model
inputs, predictions, or training runs. Invoke again after any schema change.

## Project context

The target is the canonical 10-class classifier. Legacy native units contain 16
frames; CVAT anchor intervals contain six. Harmonize first, then create training
windows. Treat `pig_id` as annotation-local, never as identity across videos.

## Required inputs

- reviewed native-unit and temporal-interval manifests;
- candidate fold and training-window manifests;
- explicit feature whitelist and blacklist;
- source, recording/session, video, and context availability columns;
- fit scope for normalization, priors, weights, thresholds, and calibration.

## Scientific invariants

- Keep every recording date, session, and video in one split role per fold.
- Keep each review/native unit and all overlapping windows in one fold.
- Never select all numeric columns automatically.
- Exclude labels, review fields, target-derived fields, paths, IDs, and folds.
- Fit every learned transform from training-fold data only.
- Quantify source, length, padding, context, and missing-modality shortcuts.
- Never silently drop a row or change a label.
- Require 16-frame legacy and six-frame CVAT native-unit contracts.

## Ordered procedure

1. Verify source hashes and temporal harmonization status.
2. Reconcile frame, interval, native-unit, review-unit, and window key counts.
3. Validate source-specific lengths and native-unit uniqueness.
4. Audit recording/session/video membership for every split role.
5. Prove review units and overlapping windows never cross split boundaries.
6. Validate the whitelist against forbidden patterns and inference paths.
7. Record fold-local fitting rules for every data-derived parameter.
8. Produce class/source support and shortcut diagnostics by fold.
9. Compare output keys and labels with inputs; explain every exclusion.
10. Stop before training unless the audit has zero errors.

## Required outputs

Produce `feature_whitelist.json`, `feature_blacklist.json`,
`fold_manifest.csv`, `leakage_audit.json`, `class_by_fold_support.csv`,
`source_by_fold_support.csv`, and `temporal_unit_audit.json`.

## Validation commands

Use [split audit](../checks/audit_split_overlap.py),
[feature audit](../checks/audit_feature_leakage.py), and
[native-unit audit](../checks/audit_native_unit_uniqueness.py). Run only on
synthetic fixtures or bounded manifests unless broader I/O is authorized.

## Stop conditions

Stop on split overlap, duplicate fold assignment, missing unexplained keys,
target-derived X fields, incomplete harmonization, wrong 16/6 lengths, or a
near-direct source/length-to-label shortcut without mitigation.

## Forbidden actions

Do not edit `data/`, infer cross-video identity from `pig_id`, random-split
rows, fit transforms globally, hide unsupported folds, delete rows to pass
checks, or authorize training while the audit is invalid.

## Completion report format

Use the shared [completion report](../templates/skill_completion_report.md) and
the local [output contract](templates/required_outputs.json). Report key counts,
support tables, errors, warnings, mitigations, and a PASS/FAIL decision.
