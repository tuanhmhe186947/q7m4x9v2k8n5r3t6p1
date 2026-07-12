---
name: gpu-training-profiler
description: >-
  Future opt-in classification_v2 GPU profiler after a one-fold pilot. Use
  explicitly for AMP correctness, bounded batch search, peak VRAM, throughput,
  dataloader bottlenecks, OOM, resume, or local/remote execution planning.
---

# Gpu Training Profiler

## Purpose

Measure a fixed classifier configuration safely and choose an execution profile
without weakening the scientific model merely to fit the local GPU.

## When to use

Invoke explicitly after one-fold correctness passes and before authorizing an
expensive remote pilot or full-fold execution estimate.

## Project context

Status is `future`; implicit invocation is disabled. The RTX 3050 Laptop GPU is
for local correctness and bounded profiling. Remote or rented GPUs may run the
same immutable configuration and independent fold jobs.

## Required inputs

- frozen model, resolution, sequence view, modalities, loss, and cache;
- representative bounded batch and dataloader configuration;
- target execution mode, hardware, fold size, and cost assumptions;
- checkpoint/resume contract and deterministic seed;
- explicit maximum search steps, wall time, and memory safety margin.

## Scientific invariants

- Test AMP numerical agreement before using AMP performance results.
- Bound batch search and release tensors between trials.
- Separate model throughput from data-loading and end-to-end throughput.
- Preserve effective batch through declared gradient accumulation.
- Record OOM state and recover without changing architecture silently.
- Prefer cache, AMP, accumulation, checkpointing, or larger remote VRAM before
  reducing the selected model.
- Keep independent fold outputs isolated and hash-compatible.

## Ordered procedure

1. Confirm the one-fold pilot and lineage manifests pass.
2. Run CPU and CUDA one-batch correctness with and without AMP.
3. Measure host-to-device, forward, backward, optimizer, and loader time.
4. Run a bounded batch-size search with a fixed safety margin.
5. Test gradient accumulation for the target effective batch.
6. Capture peak allocated/reserved VRAM and OOM memory snapshots.
7. Test checkpoint save/resume on the bounded workload.
8. Estimate runtime and cost per fold with startup and evaluation overhead.
9. Compare local, remote, and parallel-fold execution profiles.
10. Recommend placement without authorizing full training.

## Required outputs

Produce AMP correctness, batch-size bounds, accumulation config, loader profile,
throughput, peak VRAM, OOM snapshot when applicable, resume result, per-fold
runtime/cost estimate, and local/remote execution profiles.

## Validation commands

Use the existing bounded runtime tools in
`scripts/classification_v2/04_baselines_smokes` only after explicit activation.
During skill installation, compile and inspect this contract without CUDA work.

## Stop conditions

Stop before one-fold PASS, on AMP divergence, nonfinite gradients, unbounded
search, unreleased OOM state, resume mismatch, cache mismatch, output collision,
or any attempt to convert a profile into an unauthorized full run.

## Forbidden actions

Do not invoke implicitly, start full OOF, silently reduce model capacity, search
without limits, compare different configs as hardware results, overwrite folds,
or omit startup, loader, evaluation, and checkpoint overhead from estimates.

## Completion report format

Use the shared [completion report](../templates/skill_completion_report.md) and
local [profile contract](templates/profile_contract.json). Report bounds,
hardware, software, hashes, timing components, memory, cost, risks, and PASS/FAIL.
