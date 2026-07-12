---
name: experiment-lineage-reproducibility
description: >-
  Record classification_v2 lineage for local smoke, remote pilot, resume, and
  remote full-OOF modes. Use whenever a run, checkpoint, prediction, cache, fold,
  environment, or experiment registry artifact is created or compared.
---

# Experiment Lineage Reproducibility

## Purpose

Make every bounded or remote classifier run reproducible and link every
prediction unambiguously to data, code, config, fold, and checkpoint.

## When to use

Invoke before any smoke, pilot, fold run, resume, checkpoint export, prediction
merge, metric generation, or experiment registry update.

## Project context

Support `local_smoke`, `remote_pilot`, and `remote_full_oof`. Local RTX 3050 is
a correctness host, not a research limit. Each fold must write to an isolated
directory so independent remote GPU jobs can be merged safely.

## Required inputs

- run name, execution mode, fold, seed, and immutable config;
- code SHA plus dirty-worktree status;
- dataset, cache, fold, and feature-whitelist hashes;
- architecture, pretrained enum, resolution, temporal view, modalities, loss;
- environment and hardware inventory;
- declared checkpoint, prediction, metric, and registry destinations.

## Scientific invariants

- Hash semantic inputs before execution and verify them before resume.
- Never overwrite another fold or run directory.
- Link predictions to exactly one checkpoint, config, and fold.
- Record exact pretrained weight enum and preprocessing.
- Preserve failures with reason instead of deleting failed run evidence.
- Keep remote and local runs under the same manifest schema.
- Require immutable fold outputs before any cross-GPU merge.

## Ordered procedure

1. Create a unique run ID and fail if its output directory already contains data.
2. Capture code SHA, dirty state, resolved config, and all input hashes.
3. Capture Python, PyTorch, torchvision, CUDA, cuDNN, OS, GPU, and VRAM.
4. Freeze architecture, preprocessing, temporal view, modalities, loss, and seed.
5. Write a planned run manifest before model execution.
6. Write fold outputs atomically inside their own directories.
7. Record runtime, peak VRAM, status, and failure reason.
8. Hash checkpoint, prediction, metric, and environment artifacts.
9. Validate resume equivalence and artifact links before continuing.
10. Append one immutable registry row; never rewrite previous lineage.

## Required outputs

Produce `run_manifest.json`, `environment.json`, `artifact_manifest.json`,
`checkpoint_manifest.json`, `prediction_manifest.json`, and
`runs_registry.csv` with all mandatory run, software, hardware, path, hash,
runtime, status, and failure fields.

## Validation commands

Use [artifact hash audit](../checks/audit_artifact_hashes.py) and
[run manifest template](../templates/run_manifest.example.json). Run resume and
merge checks only on synthetic or bounded smoke artifacts during installation.

## Stop conditions

Stop when hashes differ, checkpoint config is absent, a fold path would be
overwritten, resume config differs, pretrained weights are ambiguous, or a
prediction cannot be linked to one checkpoint and held-out fold.

## Forbidden actions

Do not reuse run IDs, overwrite folds, resume across semantic config changes,
omit dirty state, register unverifiable artifacts, delete failure evidence, or
run full OOF without the existing explicit authorization gate.

## Completion report format

Use the shared [completion report](../templates/skill_completion_report.md) and
local [lineage fields](templates/lineage_fields.json). Include hashes, execution
mode, fold isolation, environment, artifacts, failures, and PASS/FAIL.
