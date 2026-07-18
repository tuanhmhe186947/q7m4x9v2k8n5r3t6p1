---
name: tracking-experiment-guardian
description: >-
  Guard pig multi-object tracking evaluations, mode comparisons, ablations,
  metric promotion, artifact lineage, and no-video-output experiments. Use
  before changing or evaluating realtime, ByteTrack, hybrid association,
  refinement, detection, tracking profiles, or HOTA/IDF1/IDSW metrics.
---

# Tracking Experiment Guardian

## Purpose

Execute tracking changes as isolated, reproducible experiments. Preserve strong
identity results, fail closed on lineage drift, and never emit MP4 artifacts
during analysis.

## When to use

Before code changes or evaluation, inspect the available skill catalog and
record the selected skills in the working plan.

- Use `computer-vision-opencv` for image/video and OpenCV behavior.
- Use `safe-refactor-test-guardian` for small reversible code changes.
- Use `scientific-ablation-controller` for one-family comparisons.
- Use `experiment-lineage-reproducibility` for run IDs and hashes.
- Use `find-skills` when the installed catalog has a real capability gap.
- Use `skill-creator` to add or improve a project-local skill only after that
  gap is demonstrated. Validate it before relying on it.

## Project context

Guard `hybrid_bytetrack`, `realtime_fast`, `realtime_balanced`, and
`realtime_quality_delayed`. The existing detector conclusions, profile
semantics, baseline locks, and no-MP4 constraint are settled inputs.

Tracking GT was seeded by an older tracker and manually corrected for bbox and
ID. Treat corrected bbox/ID as authoritative. Treat the 1,930 `Hidden` values
as tracker-derived and not human-confirmed visibility.

## Required inputs

Require the parent commit, dirty-file inventory, video/GT and detector hashes,
resolved semantic config, selected profile, fresh output roots, hypothesis,
single changed family, target weak metric, guardrails, and runtime budget.

## Scientific invariants

Change one family per candidate. Compare the same videos, GT, detector,
evaluation contract, and hardware policy. Preserve negative evidence, raw
artifact hashes, and canonical prediction hashes that exclude only declared
volatile CVAT metadata.

Use `include_hidden=true` for primary geometry and identity metrics. This keeps
corrected bbox/ID rows without making `Hidden` a target. Use exclude-Hidden
metrics only as a separately labeled compatibility replay.

## Ordered procedure

1. Read `AGENTS.md`, memory `01`, `02`, `03`, `08`, and tracking memory
   `04` through `07`.
2. Read the active addendum in `Kế Hoạch Tương Lai.md`.
3. Record the starting commit, dirty files, detector hash, exact video/GT
   manifest, mode profile, semantic config, and unique output roots.
4. Declare one parent, one candidate, one changed family, and one hypothesis.
5. Reject diffs that change another family or settled detector conclusions.
6. Pass static, synthetic, no-MP4, and prediction-integrity checks.
7. Freeze difficult event windows from parent artifacts only. Keep tracker
   warm-up frames separate from the scored interval.
8. Screen in order: event windows, one full target video, a multi-video hard
   set, then full-13 only after the hard-set gate passes.
9. Compare paired per-video metrics and repeat the exact finalist.
10. Run `scripts/audit_tracking_repeatability.py` on every completed
   primary/repeat pair. Keep input rehashing enabled for authority evidence.
11. Record promotion or rejection; never suppress negative evidence.
12. Promote profile defaults only in a separate reversible commit.

### Staged evidence funnel

- Build windows from the locked parent's remapped switch events, never from a
  candidate. Use `scripts/build_hard_event_windows.py` to freeze the window
  manifest and its source hashes.
- Start tracking before the scored interval so causal state can warm up. Score
  only `[score_start_frame, score_end_frame]`; never score warm-up frames or
  reset the tracker at the first difficult frame.
- For a strictly post-video geometry family, use
  `scripts/replay_post_video_geometry.py` when a tracker-reset window cannot
  reproduce the locked parent's state. Replay only from the parent's hashed
  `annotations_cvat_shapes.json` and matching XML. Require equal shape keys and
  non-geometry payload, then score the frozen window before any full-video
  metric evaluation.
- Never use geometry replay for detection, association, identity, visibility,
  causal, or runtime claims. Creating a full-length replay artifact is not a
  full-video evaluation and does not advance the funnel by itself.
- Treat window results as screening evidence, not promotion evidence. Reject a
  candidate immediately when it misses its target mechanism or creates a
  severe local failure.
- Advance to a full target video only after improvement appears in at least
  two independent episodes. Advance to the hard set only after the full target
  video improves.
- Require the hard set to contain at least three difficult videos plus every
  declared guardrail. Advance to full-13 only when the hard-set aggregate
  improves and gains occur on at least two difficult videos.
- Judge non-regression on the declared aggregate and critical guardrails. A
  bounded per-video trade-off is allowed when its limit was declared before
  execution, the multi-video aggregate improves, and the trade-off is reported.
- Freeze gates before execution. Never change windows, local regression
  budgets, or guardrails after seeing candidate results.

### Baseline and output invariants

- Treat `outputs/eval/mode_compare/20260709_040751` as the current five-mode
  comparison until a hash-bound replacement is promoted.
- Preserve the three existing realtime profiles; do not create duplicates.
- Treat `realtime_fast` and `realtime_balanced` as causal zero-delay profiles.
- Treat `realtime_quality_delayed` as `post_video_global_graph` with delay `-1`
  until a prefix-invariance test proves a finite flush boundary.
- For any causal or fixed-delay claim, append future frames at every declared
  flush boundary and require already-flushed XML payloads to remain identical.
- Preserve the same-contract aggregate quality and every predeclared critical
  guardrail. Report all paired per-video changes, including accepted local
  trade-offs.
- Preserve or improve `000302` IDSW for each realtime profile against its
  include-Hidden baseline. Historical fixed limits from exclude-Hidden reports
  are compatibility evidence only.
- Do not blame detector weight for the current-code `000263` regression.
- Keep `iou0_area0_condarea0_merge0`; never enable `condarea` without an
  isolated ablation.
- Treat input MP4 files as read-only. Never generate MP4, preview, overlay, or
  event-clip artifacts during evaluation, probe, optimization, or benchmark.
- Fail if an experiment output root already exists or contains any MP4.
- Retain XML, CSV, JSON, Markdown, and small logs needed for evidence.

### Promotion gates

Promote a candidate only when all applicable gates pass:

- Bind parent and candidate to exact commits, inputs, detector, profile, and
  semantic-config hashes.
- Require `include_hidden=true` in the primary baseline, candidate, and repeat.
  Never promote an improvement that depends on excluding tracker-derived
  `Hidden` rows.
- Produce no MP4 anywhere under the fresh experiment root.
- Improve the declared aggregate weak metric in both identical confirmation
  runs and pass every predeclared critical guardrail.
- Keep every local regression within its frozen budget and report it plainly.
- For a geometry-only family, require equal track IDs, shape keys, Behavior,
  `Hidden`, `occluded`, and other non-geometry payload before full or repeat.
- Stay within the latency and memory budget of the affected realtime profile.
- Require repeat effective FPS to be at least 90% of primary effective FPS.
- Require repeat peak RSS and CUDA memory to stay within 110% of primary.
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
- For geometry replay, a hash-bound replay manifest and per-box geometry delta
  CSV proving parent JSON/XML integrity and non-geometry payload equality.
- A recursive artifact audit proving the experiment root contains no MP4.
- An immutable repeatability audit that rehashes inputs and artifacts, checks
  canonical predictions, and compares metrics exactly outside `pred_xml`.
- A signed promotion decision that records every passed or failed gate.

## Validation commands

Run the project skill-pack validator and tracking discovery scenario before
using this skill. For code changes, run Ruff, compileall, focused tracking
tests, `git diff --check`, and the changed-file line-length scan. Audit every
fresh root recursively for MP4. Require prediction artifact records to retain
raw SHA256 and
`cvat_xml_c14n_without_created_updated_dumped_v1` semantic SHA256. Use the
[check contract](checks/check_manifest.json), [scenario](examples/scenario.md),
and [promotion template](templates/promotion_decision.example.json).
Run the geometry replay script with `--dry-run` against the exact parent before
its first artifact-producing invocation.
The repeatability checker must run from a clean, commit-bound worktree and PASS
before a run becomes baseline, candidate, or promotion authority.
Keep its default FPS and peak-memory ratio guards enabled for authority runs.
`--skip-input-rehash` and `--allow-dirty-auditor` are test-only and cannot
support promotion evidence.

## Stop conditions

Stop before further tracking work when:

- The output root is reused, non-empty, or contains an MP4.
- An input, detector, GT, profile, config, or commit cannot be hash-bound.
- Static, synthetic, prediction-integrity, or focused regression tests fail.
- A candidate changes more than its declared family or evaluation contract.
- A critical guardrail fails or a local regression exceeds its frozen budget.
- Future frames change already-flushed output for a causal/fixed-delay profile.
- Telemetry cannot explain the observed prediction or identity change.

Fix the gate and restart with a new run ID. Do not reinterpret a failed run as
promotion evidence.

## Forbidden actions

Do not reuse roots, overwrite evidence, generate video artifacts, compare
different inputs or evaluation contracts, tune on outer/full results, bundle
families, enable `condarea` implicitly, relax a locked guardrail after seeing a
failure, or promote a default in the same commit as the candidate algorithm.

## Completion report format

Report:

1. Parent and candidate commits, run IDs, hashes, and selected skills.
2. Hypothesis, changed family, frozen windows, stage gates, target metric, and
   guardrails.
3. Per-video and aggregate HOTA, IDF1, IDSW, detection, and runtime deltas.
4. Repeatability, prediction-integrity, and recursive no-MP4 audit results.
5. Promotion or rejection, failed gates, residual risks, and the exact next
   reversible step.
