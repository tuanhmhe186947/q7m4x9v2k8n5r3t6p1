---
name: scientific-ablation-controller
description: >-
  Control classification_v2 experiments so each comparison changes one declared
  scientific family and passes promotion gates. Use for baseline design,
  ablations, finalist selection, rejection, or full-OOF authorization review.
---

# Scientific Ablation Controller

## Purpose

Prevent uninterpretable multi-variable changes and keep expensive confirmatory
evaluation limited to predeclared, evidence-backed finalists.

## When to use

Invoke when proposing a baseline, changing resolution/backbone/temporal encoder,
adding a modality, changing imbalance policy, promoting a candidate, or planning
any OOF run.

## Project context

The critical path targets the ten-class classifier. Five-class comparison,
paper reproduction, deployment, and end-to-end pipeline work remain optional and
cannot block or tune the locked ten-class result.

## Required inputs

- reconciled baseline, frozen feature whitelist, folds, and metric contract;
- parent and candidate configs with semantic diff;
- changed scientific family and falsifiable hypothesis;
- bounded compute budget, seeds, development folds, and stop rule;
- paired native-unit predictions and all gate evidence.

## Scientific invariants

- Change one principal family per experiment unless using a declared factorial design.
- Separate `R18/160 -> R18/224` from `R18/224 -> R34/224`.
- Keep folds, units, preprocessing, seeds, and metrics paired.
- Never select architecture from outer-fold predictions.
- Require correctness before performance and paired evidence before promotion.
- Lock at most F0, F1, F2, plus F2-no-hierarchy when scientifically required.
- Require explicit human authorization before full OOF.

## Ordered procedure

1. Reconcile the current baseline and freeze the comparison universe.
2. Write one hypothesis and one changed-family declaration.
3. Reject any semantic diff that changes uncontrolled families.
4. Pass schema/correctness, one-batch, tiny-overfit, and resume gates.
5. Pass bounded runtime/VRAM and development-fold gates.
6. Compare paired native-unit predictions and class-group guardrails.
7. Record promotion or rejection with evidence and uncertainty.
8. Repeat only for the next isolated family.
9. Lock finalist configs, hashes, seeds, folds, and compute estimate.
10. Keep full OOF blocked until the existing authorization gate passes.

## Required outputs

Produce `experiment_matrix.csv`, `ablation_registry.csv`,
`promotion_decisions.json`, `rejected_experiments.json`, and
`finalist_lock.json`.

## Validation commands

Use the shared [experiment matrix](../templates/experiment_matrix.example.csv)
and [promotion template](../templates/promotion_decision.example.json). Validate
config semantic diffs and existing bounded smoke gates before promotion.

## Stop conditions

Stop when multiple families change without factorial design, the baseline or
whitelist is unresolved, correctness fails, outer predictions informed design,
or an agent attempts an unapproved batch of full OOF runs.

## Forbidden actions

Do not bundle backbone, resolution, temporal, modality, loss, and sampler
changes; cherry-pick favorable folds; suppress rejected evidence; bypass gates;
or put five-class, publication, or deployment work on the critical path.

## Completion report format

Use the shared [completion report](../templates/skill_completion_report.md) and
local [gate contract](templates/promotion_gates.json). State the single changed
family, paired evidence, every gate, decision, rollback, and PASS/FAIL.
