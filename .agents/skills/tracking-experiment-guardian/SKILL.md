---
name: tracking-experiment-guardian
description: >-
  Guard pig multi-object tracking evaluations, mode comparisons, ablations,
  metric promotion, artifact lineage, and no-video-output experiments. Use
  before changing or evaluating realtime, ByteTrack, hybrid association,
  refinement, detection, tracking profiles, or HOTA/IDF1/IDSW metrics.
---

# Tracking Experiment Guardian

## Overview

Execute tracking changes as isolated, reproducible experiments. Preserve strong
identity results, fail closed on lineage drift, and never emit MP4 artifacts
during analysis.

## Required skill selection

Before code changes or evaluation, inspect the available skill catalog and
record the selected skills in the working plan.

- Use `computer-vision-opencv` for image/video and OpenCV behavior.
- Use `safe-refactor-test-guardian` for small reversible code changes.
- Use `scientific-ablation-controller` for one-family comparisons.
- Use `experiment-lineage-reproducibility` for run IDs and hashes.
- Use `find-skills` when the installed catalog has a real capability gap.
- Use `skill-creator` to add or improve a project-local skill only after that
  gap is demonstrated. Validate it before relying on it.

## Ordered workflow

1. Read `AGENTS.md`, memory `01`, `02`, `03`, `08`, and tracking memory
   `04` through `07`.
2. Read the active addendum in `Kế Hoạch Tương Lai.md`.
3. Record the starting commit, dirty files, detector hash, exact video/GT
   manifest, mode profile, semantic config, and unique output roots.
4. Declare one parent, one candidate, one changed family, and one hypothesis.
5. Reject diffs that change another family or settled detector conclusions.
6. Pass static, synthetic, no-MP4, and prediction-integrity checks.
7. Run one target video, then its guardrail set, before a full comparison.
8. Compare paired per-video metrics and repeat the exact candidate.
9. Record promotion or rejection; never suppress negative evidence.
10. Promote profile defaults only in a separate reversible commit.

## Baseline and output invariants

- Treat `outputs/eval/mode_compare/20260709_040751` as the current five-mode
  comparison until a hash-bound replacement is promoted.
- Preserve the three existing realtime profiles; do not create duplicates.
- Treat `realtime_fast` and `realtime_balanced` as causal zero-delay profiles.
- Treat `realtime_quality_delayed` as `post_video_global_graph` with delay `-1`
  until a prefix-invariance test proves a finite flush boundary.
- For any causal or fixed-delay claim, append future frames at every declared
  flush boundary and require already-flushed XML payloads to remain identical.
- Preserve per-video hybrid remapped IDSW `0` as a hard guardrail.
- Keep `000302` at IDSW `0` for realtime candidates.
- Do not blame detector weight for the current-code `000263` regression.
- Keep `iou0_area0_condarea0_merge0`; never enable `condarea` without an
  isolated ablation.
- Treat input MP4 files as read-only. Never generate MP4, preview, overlay, or
  event-clip artifacts during evaluation, probe, optimization, or benchmark.
- Fail if an experiment output root already exists or contains any MP4.
- Retain XML, CSV, JSON, Markdown, and small logs needed for evidence.

## Promotion gates

Promote a candidate only when all applicable gates pass:

- Bind parent and candidate to exact commits, inputs, detector, profile, and
  semantic-config hashes.
- Produce no MP4 anywhere under the fresh experiment root.
- Improve the declared weak metric in both identical runs, not only aggregate.
- Preserve every declared per-video IDSW and prediction-integrity guardrail.
- Stay within the latency and memory budget of the affected realtime profile.
- Record frame, detector, association and postprocess timing, p50/p95 latency,
  effective FPS, peak process RSS, peak CUDA memory, output contract and delay.
- Show telemetry consistent with the declared changed family and hypothesis.
- Keep the algorithm or profile-default promotion in a separate commit.

Reject ties, unexplained prediction drift, mixed-family changes, missing
artifacts, or improvements that depend on changing the evaluation contract.

## Required outputs

- A run manifest with selected skills, lineage hashes, profile, hypothesis,
  changed family, commands, environment, and output roots.
- Per-video predictions and HOTA, IDF1, IDSW, detection, and runtime metrics.
- A per-video `tracking_runtime_telemetry.csv` linked to hashed quality reports.
- A paired baseline/candidate delta table for every evaluated video.
- A recursive artifact audit proving the experiment root contains no MP4.
- A signed promotion decision that records every passed or failed gate.

## Stop conditions

Stop before further tracking work when:

- The output root is reused, non-empty, or contains an MP4.
- An input, detector, GT, profile, config, or commit cannot be hash-bound.
- Static, synthetic, prediction-integrity, or focused regression tests fail.
- A candidate changes more than its declared family or evaluation contract.
- A required guardrail regresses, even when the aggregate metric improves.
- Future frames change already-flushed output for a causal/fixed-delay profile.
- Telemetry cannot explain the observed prediction or identity change.

Fix the gate and restart with a new run ID. Do not reinterpret a failed run as
promotion evidence.

## Completion report

Report:

1. Parent and candidate commits, run IDs, hashes, and selected skills.
2. Hypothesis, changed family, target weak metric, and guardrails.
3. Per-video and aggregate HOTA, IDF1, IDSW, detection, and runtime deltas.
4. Repeatability, prediction-integrity, and recursive no-MP4 audit results.
5. Promotion or rejection, failed gates, residual risks, and the exact next
   reversible step.
