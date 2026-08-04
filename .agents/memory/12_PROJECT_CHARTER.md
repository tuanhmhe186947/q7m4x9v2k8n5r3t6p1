# Project Charter

> Charter version: `1.1`. The canonical file hash is recorded in
> `12_PROJECT_CHARTER.sha256`.

## Authority

- Class: long-term, stable project authority.
- Change only after explicit user direction or accepted evidence changes scope.
- Record a superseding charter version; never silently rewrite prior intent.
- Daily state, blockers, experiments, and raw metrics do not belong here.

## Scientific Question

Can a reproducible, leakage-safe pig tracking and behavior-recognition
pipeline improve identity stability and internal behavior recognition under
recording-date/video-safe evaluation, while retaining evidence lineage and
practical execution constraints?

## Non-Goals

- External farm, camera, cohort, or biological-identity generalization.
- Treating annotation-local `pig_id` as a cross-video identity.
- Promoting an unreviewed dataset, diagnostic run, or historical artifact.
- Optimizing a single video or metric at the expense of declared guardrails.

## Mission

Build a scientifically defensible pig behavior system that connects:

1. reliable per-pig tracking,
2. leakage-safe behavior recognition,
3. reproducible evaluation and artifact lineage,
4. practical execution on available hardware, and
5. reviewable outputs for annotation, analysis, and research reporting.

## Scientific Outcome

- Support internal recording-date and video-safe claims first.
- Bind every promoted result to code, config, data, evaluator, and limitations.
- Separate engineering evidence from scientific performance evidence.
- Require human-reviewed authority where labels or review decisions are material.
- Do not claim external farm, camera, or cohort generalization without evidence.

## Success Criteria

- The selected pipeline has immutable input, code, config, split, seed, and
  evaluator lineage.
- Scientific claims are admitted only when the claim registry is complete.
- Tracking and behavior gates pass with repeatability and per-video limits.
- Review, leakage, schema, and no-output-video audits pass where applicable.
- A new agent session can recover current state without reading historical logs.

## Allowed And Forbidden Claims

- Allowed: `CODE_VERIFIED`, `ARTIFACT_VERIFIED`, `RUN_VERIFIED`, or
  `HUMAN_VERIFIED` claims within their registered scope and limitations.
- Forbidden: calling diagnostic, pilot, unverified, or incomplete-lineage output
  a final scientific result.
- Forbidden: generalizing beyond the registered data population or claiming
  causality from whole-pipeline comparisons without a causal control.

## Project Invariants

- Never use future or evaluation-only information in model inputs.
- Treat annotation identity as source-scoped unless a stronger contract exists.
- Preserve accepted artifacts and their lineage even when methods are retired.
- Prefer small reversible changes and controlled comparisons.
- A contradiction reopens the affected claim; it does not rewrite history.
- Long runs require their declared static, synthetic, short-run, and permission gates.

## Stop Criteria

- Stop at any failed lineage, leakage, split, schema, repeatability, or review
  gate.
- Stop when two current authorities conflict or a transition skips a gate.
- Stop before cleanup when ownership or rebuildability is uncertain.
- Stop before promotion when the claim manifest is incomplete.

## Promotion And Reject Criteria

- Promote only after `VALIDATED -> FROZEN -> PROMOTED`, complete lineage,
  repeatability, limitations, and the declared guardrail set.
- Reject as `REJECTED`, `BLOCKED`, `SUPERSEDED`, or `NOT_REPRODUCIBLE` when the
  matching terminal condition is evidenced.
- A metric improvement alone never authorizes promotion.

## Completion Standard

The project is complete only when the selected pipeline is reproducible from
registered inputs, passes its scientific gates, has bounded limitations, and
produces the required tracking, behavior, evaluation, and review artifacts.
