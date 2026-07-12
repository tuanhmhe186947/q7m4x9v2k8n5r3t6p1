---
name: multimodal-sequence-model-builder
description: >-
  Build configurable classification_v2 visual, temporal, geometry, ROI, social,
  fusion, auxiliary, and final-head modules. Use for model factories, tensor
  contracts, ablation modes, missing modalities, smokes, or checkpoint schemas.
---

# Multimodal Sequence Model Builder

## Purpose

Build one inference-compatible multimodal codebase whose branches can be enabled
by config and evaluated through isolated scientific ablations.

## When to use

Invoke when adding or changing encoders, fusion, model modes, shape contracts,
missing-modality behavior, visual baselines, temporal baselines, or checkpoints.

## Project context

The direct target is ten behavior classes. Actor RGB, real timing, geometry,
motion, all-class ROI relations, partner sets, and optional union/full context
must remain usable later by inference without review or label-derived fields.

## Required inputs

- versioned tensor names, dtypes, shapes, masks, and timing semantics;
- explicit modality set and model mode;
- selected visual and temporal baseline controls;
- ten-class label order and optional reviewed attribute masks;
- inference-time availability rules and checkpoint schema version.

## Scientific invariants

- Always supervise the final ten-class head directly.
- Never hard-cascade auxiliary argmax predictions into the final head by default.
- Select partners without target behavior labels.
- Never propagate `fight` to bystanders or `social-nose` to receivers.
- Gate every optional branch with availability and quality masks.
- Do not treat availability alone as behavioral evidence.
- Run with missing modalities and exclude all review or GT-only fields.
- Separate resolution and backbone effects with controlled comparisons.

## Ordered procedure

1. Freeze the classifier input/output and label-order contracts.
2. Implement `ActorEncoder` and masked mean/attention temporal controls.
3. Add masked TCN, then a small Transformer only with evidence.
4. Add `GeometryMotionEncoder`, `ROIEncoder`, and mask-aware fusion separately.
5. Add `PartnerSetEncoder`, then `UnionCropEncoder`; defer full frame.
6. Add `AvailabilityEncoder` only with availability-only shortcut controls.
7. Add masked auxiliary heads without removing direct final supervision.
8. Expose all supported modes through one validated model factory.
9. Run shape, missing-modality, parameter-count, and deterministic smokes.
10. Serialize an inference-compatible, hash-linked checkpoint contract.

## Required outputs

Produce module contracts, input/output shape specifications, a configurable
factory, ablation configs, parameter-count report, forward and missing-modality
tests, and an inference-compatible checkpoint schema.

## Validation commands

Use [shape audit](../checks/audit_model_forward_shapes.py) and
[mask audit](../checks/audit_missing_modality_masks.py). Also run the existing
bounded forward checker in `scripts/classification_v2/04_baselines_smokes` when
the relevant model code exists; never start full training from this skill.

## Stop conditions

Stop on shape drift, nonfinite outputs, mask-dependent logits for absent data,
target-dependent partner selection, unsupported inference inputs, unresolved
visual confounds, or failure of one-batch and missing-modality smokes.

## Forbidden actions

Do not introduce label routing, hard auxiliary cascades, unmasked optional
tensors, source IDs, global numeric feature discovery, simultaneous uncontrolled
architecture changes, pretrained downloads, or full OOF execution.

## Completion report format

Use the shared [completion report](../templates/skill_completion_report.md) and
local [model mode contract](templates/model_modes.json). Report shapes, masks,
parameters, enabled branches, test cases, inference constraints, and PASS/FAIL.
