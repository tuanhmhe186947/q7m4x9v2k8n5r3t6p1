# Tracking Metric Mathematical Contract

Date: 2026-07-28

Scope: static source review and hand-authored synthetic cases only. No tracker,
detector, GPU, real-video, or historical-output execution was performed.

## Evaluator call path

The active evaluation path is:

`tracking.cli` -> `tracking.pipeline.run_pipeline` ->
`tracking.evaluator.evaluate_pair` -> `tracking.cvat_io.parse_cvat_video_xml`
-> `tracking.evaluator.evaluate_tracking`.

Each frame is assigned by `tracking.matching.match_frame`. Dataset aggregation
is performed by `tracking.metrics.aggregate_metrics`. Per-video and `ALL` rows
are converted to percentage display columns by
`tracking.evaluator.metrics_to_dataframe`, then reported by
`tracking.reporting.build_markdown_report`.

The separate permanent/terminal analysis is implemented by
`scripts/tracking/build_rf_acc23_error_taxonomy.py`; it is not part of the core
metric evaluator.

## Population, matching, and units

- CVAT boxes marked `outside` are excluded.
- `include_hidden=false` excludes `Hidden=Yes` boxes symmetrically from GT and
  prediction XML. `include_hidden=true` retains them.
- The hidden flag changes the evaluated population; it does not change the
  matching formula.
- Boxes use floating-point `(xtl, ytl, xbr, ybr)` coordinates and ordinary
  intersection over union.
- `match_frame` builds the full IoU matrix, runs Hungarian assignment to
  maximize total IoU, and then removes assigned pairs below the configured IoU
  threshold.
- Applying the threshold after unconstrained assignment is defective: a
  low-IoU assigned pair can prevent a higher-cardinality valid gated matching.
  The production implementation remains unchanged in this audit.
- All internal ratios are in `[0, 1]` unless noted. Display columns ending in
  `_pct` multiply the ratio by 100 and round to two decimals.
- Empty denominators return `0.0`; empty sequences are not reported as perfect.

## Standard and HOTA-style metrics

- `TP / matches` is the retained frame-assignment count. It is nonnegative
  and summed by video. It is a standard count affected by the matching defect.
- `FP = Pred - TP`. It is a nonnegative standard count; lower is better.
  Dataset aggregation sums video counts.
- `FN = GT - TP`. It is a nonnegative standard count; lower is better.
  Dataset aggregation sums video counts.
- Detection precision is `TP / Pred`. It is a standard `[0,1]` ratio; higher
  is better. The dataset value is recomputed from summed counts.
- Detection recall is `TP / GT`. It is a standard `[0,1]` ratio; higher is
  better. The dataset value is recomputed from summed counts.
- MOTA is `1 - (FN + FP + IDSW) / GT`. It is a standard CLEAR MOT metric,
  at most one and unbounded below; higher is better. It is recomputed from
  summed counts.
- MOTP IoU is matched-IoU sum divided by TP. It is a standard MOTP-style
  `[0,1]` localization value; higher is better. Dataset aggregation is
  match-weighted. It is not reported as LocA.
- IDTP is the assigned overlap count from the whole-video maximum one-to-one
  GT/predicted identity assignment. It is a nonnegative standard count;
  higher is better. Per-video assignments are summed, preserving boundaries.
- `IDFP = Pred - IDTP` and `IDFN = GT - IDTP`. Both are nonnegative standard
  counts; lower is better. Dataset aggregation sums video counts.
- IDF1 is `2 IDTP / (2 IDTP + IDFP + IDFN)`. It is a standard `[0,1]`
  identity-mass ratio; higher is better. The aggregate is recomputed from
  summed per-video ID counts.
- ID precision would be `IDTP / (IDTP + IDFP)`. It is a standard `[0,1]`
  ratio with higher better, but it is not currently reported.
- ID recall would be `IDTP / (IDTP + IDFN)`. It is a standard `[0,1]`
  ratio with higher better, but it is not currently reported.
- DetA is `TP / (TP + FP + FN)` at one configured threshold. It is a `[0,1]`
  detection-intersection ratio; higher is better. It is recomputed from summed
  counts, but is not standard threshold-averaged DetA.
- AssA is the match-weighted mean of `TPA / (TPA + FNA + FPA)` across matched
  identity pairs. It is a `[0,1]` association Jaccard; higher is better.
  Video aggregation is match-weighted, but threshold averaging is absent.
- HOTA is `sqrt(DetA * AssA)` at one threshold. It is a `[0,1]` HOTA-style
  value; higher is better. Standard HOTA instead averages alpha-specific HOTA
  over `0.05` through `0.95`.
- LocA is not reported. Standard HOTA LocA is mean localization over
  true-positive matches at each alpha; it is `[0,1]` with higher better.

The current HOTA, DetA, and AssA labels do not satisfy standard TrackEval
headline semantics. The evaluator normally uses IoU `0.5`, while TrackEval HOTA
reports the mean across `0.05, 0.10, ..., 0.95`. For one correct-identity pair
with IoU `0.6`, the current evaluator reports HOTA `1.0` at alpha `0.5`; the
manually calculated threshold-averaged value is `12/19`, approximately
`0.6315789474`.

At a fixed threshold, `ALL` DetA is count-micro-aggregated and AssA is
match-weighted. The evaluator does not take an unjustified unweighted mean of
per-video percentages. However, the missing threshold dimension means the
reported aggregate is not standard threshold-averaged HOTA.

## Identity continuity and project diagnostics

- IDSW is a standard-style transition event. It increments when a GT's current
  retained predicted ID differs from its previous retained predicted ID.
  Previous identity persists across gaps; counts sum by video and reset there.
- Fragments are standard-style continuity counts. A previously matched GT
  becomes unmatched and is later matched. Counts sum by video.
- Gap-tolerant fragments are project-specific. Runs split only when a gap
  exceeds the configured tolerance, which defaults to 15 frames.
- A wrong-ID matched row is project-specific. It is a retained match whose
  post-remap predicted ID differs from GT; unmatched rows do not contribute.
- An identity-error episode is a project-specific connected pair-event.
  Wrong-ID rows up to 15 frames apart can connect when identities overlap.
- A permanent swap is project-specific. The current taxonomy marks terminal
  events or spans of at least 60 frames; this is an implementation bug.
- A terminal swap is project-specific. The current taxonomy tests the last
  video frame, not the affected GT's final authoritative matched frame.

`IDSW_COUNT != IDENTITY_ERROR_SEVERITY`.

A conventional IDSW is an identity transition event for one GT trajectory under
the evaluator's matching and gap policy. A one-frame wrong assignment followed
by recovery can create two identity switch events. One wrong transition that
persists to the end can create only one. Therefore IDSW alone cannot distinguish
a temporary identity-error episode from a persistent or terminal identity
swap. The audit avoids the non-standard phrase `permanent IDSW`; it uses
identity switch event, temporary identity-error episode, persistent/permanent
identity swap, terminal identity swap, and wrong-ID matched duration.

## Permanent and terminal implementation classification

Both current implementations are classified `IMPLEMENTATION_BUG`.

- Connected components use identity overlap and a 15-frame gap, so a pair swap
  can be one event even though it contains two GT-level identity episodes.
- `duration_frames` is `end_frame - start_frame + 1`; it is not the number of
  wrong-ID matched rows or authoritative wrong-ID time intervals.
- `self_recovers` is inferred from whether the component ends before
  `evaluated_frames - 1`. It does not verify a later correct match for each
  affected GT identity.
- `terminal_swap` consequently tests the video boundary, not the final
  authoritative matched frame of each affected GT.
- `permanent_swap` includes every terminal event and every event whose span is
  at least 60 frames. Terminal events are intentionally an overlapping subset,
  but the summary does not state this clearly.
- GT-ambiguous events are preserved and classified, yet summary counts do not
  automatically exclude them from authoritative permanent/terminal ranking.

The grouping function does conserve every input mismatch row exactly once
inside its connected pair-event, and it prevents cross-video components. Those
properties do not repair the duration and recovery semantics above.

## Boundary and missingness policies

- Per-video evaluation resets IDSW, fragment, and global identity-assignment
  state.
- Aggregate IDF1 sums per-video IDTP/IDFP/IDFN, so identical predicted labels
  in separate videos are not globally coupled.
- Unmatched GT contributes FN and may end a strict tracklet. Unmatched
  prediction contributes FP.
- Gaps do not reset `last_match_for_gt`, so current IDSW is gap-persistent.
- Hidden boxes contribute exactly when retained by `include_hidden`.
- The current Hidden labels are not fully human-confirmed scientific authority;
  code parity does not resolve that data-authority limitation.
- Input frame keys are sorted. Identity sets are sorted before global
  assignment, and synthetic permutation checks are deterministic.
