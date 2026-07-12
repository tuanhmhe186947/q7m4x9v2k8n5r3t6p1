---
name: imbalanced-classification-evaluator
description: >-
  Future opt-in evaluator for event-balanced CE, effective-number CE, and
  Balanced Softmax after a one-fold classification_v2 pilot. Use explicitly to
  compare one imbalance policy with fixed data, model, folds, and sampling.
---

# Imbalanced Classification Evaluator

## Purpose

Evaluate one imbalance intervention at a time using native-event mass and
rare-class guardrails rather than overlapping-window frequency.

## When to use

Invoke explicitly only after one-fold correctness and baseline pilots pass and
error analysis shows class imbalance is a plausible limiting factor.

## Project context

Status is `future`; implicit invocation is disabled. Initial candidates are
event-balanced CE, effective-number CE, and Balanced Softmax only. Rare classes
include `fight`, `social-nose`, `playwithtoy`, and `move`.

## Required inputs

- frozen model, folds, seed set, preprocessing, and eligible native events;
- row, frame, window, native-unit, and unique-event counts;
- fold-local class prior and event-mass computation;
- baseline paired native-unit predictions and group metrics;
- one predeclared candidate imbalance policy.

## Scientific invariants

- Balance unique native events before overlapping windows.
- Divide event mass among windows from the same event.
- Estimate priors and weights from training folds only.
- Change loss or sampler policy, never both in the first comparison.
- Report global and rare-class performance together.
- Select exactly one policy or retain event-balanced CE.

## Ordered procedure

1. Confirm the one-fold pilot and grouped evaluation contracts pass.
2. Reconcile all five count levels and event-to-window multiplicity.
3. Compute fold-local event mass, prior, caps, and effective sample size.
4. Keep the model and sampler fixed while testing one declared loss.
5. Run bounded paired development comparisons with identical seeds.
6. Report global, rare, interaction, class recall, and confusion-pair metrics.
7. Reject unstable or majority-collapse policies.
8. Select one policy only when paired evidence and calibration support it.

## Required outputs

Produce count-level audits, fold-local weight/prior manifests, paired global and
rare-class metrics, per-class recall, confusion-pair results, stability evidence,
and one selected or retained imbalance policy.

## Validation commands

Reuse grouped native-unit evaluation and event-weight checks in
`scripts/classification_v2/02_train_ready_exports` and
`07_postrun_evaluation`. Run only bounded one-fold pilots under explicit use.

## Stop conditions

Stop before one-fold PASS, on count ambiguity, global prior fitting, simultaneous
loss/sampler changes, majority collapse, unstable seeds, or reduced rare-class
macro-F1 without a predeclared tradeoff.

## Forbidden actions

Do not invoke implicitly, add focal/deferred losses initially, combine multiple
losses and samplers, oversample overlapping windows as independent events, tune
on outer folds, or start full OOF.

## Completion report format

Use the shared [completion report](../templates/skill_completion_report.md) and
local [imbalance contract](templates/imbalance_contract.json). Report all count
levels, fold-local policy, paired metrics, stability, selection, and PASS/FAIL.
