# Development identity diagnostic metric specification

This document restates, without changing, `TRACKING_EVALUATOR_STANDARD_V2`,
`TRACKING_MATCHING_STANDARD_V2`, and `IDENTITY_ERROR_EPISODES_V2`.

## Population and GT–prediction matching

Outside boxes are excluded. `include_hidden` is applied symmetrically before
generic matching. The primary study uses `include_hidden=true`. Detection,
CLEAR continuity, and episode rows use IoU 0.50 eligibility-constrained
assignment: ineligible edges are removed first, then eligible cardinality is
maximized before total IoU. HOTA separately follows TrackEval's prescribed
global-alignment assignment at each of 19 alphas from 0.05 through 0.95.

## Identity authority and wrong-ID rows

Identity global metrics use one sequence-local TrackEval Identity assignment.
Episode severity instead freezes `IDENTITY_AUTHORITY_FIRST_OBSERVATION_V2`:
the first unambiguous eligible match binds an unmapped prediction identity to
an unmapped GT identity and never changes. A matched resolved GT row is wrong
when the observed prediction identity is not the frozen expected identity.
One video frame can contribute multiple wrong animal-frame observations. An
unmatched GT or prediction is not a wrong-ID row; it contributes FN or FP.

Wrong-ID seconds sum `1/FPS` per wrong animal-frame, using each video's proven
positive FPS. Thus the unit is animal-seconds, not elapsed wall-clock span.

## Episodes

The primary key is `(sequence_key, gt_id)`. Wrong rows join when no correct
matched row intervenes and the frame delta is at most 15. Unmatched rows add no
wrong exposure and do not recover an episode. A later authoritative correct
match makes the episode recovered. An episode is terminal when it contains the
GT trajectory's final authoritative matched observation and has no recovery.
Recovered and terminal are mutually exclusive. A gap larger than 15 splits an
episode and censors the earlier component unless later correct evidence proves
recovery.

## Pairwise swaps, Hidden, fragments, and edge cases

A pairwise swap requires reciprocal wrong ownership of two GT identities in
the same frame. Unordered pairs are counted once. Persistence requires at least
60 direct joint observations, or two terminal linked GT episodes that still
target each other at their final authoritative observations. Three-way cycles
are not pairwise swaps.

With `include_hidden=false`, Hidden rows are excluded before matching and
cannot create matches, misses, fragments, switches, or episodes. A fragment is
a resolved GT trajectory returning to a matched state after one or more
authoritative evaluated GT observations were unmatched. Missing predictions
may create FN and fragmentation, but not wrong-ID exposure by themselves.

Tied first identity authority is retained as ambiguous and excluded from
authoritative severity totals. Duplicate identities in one frame, invalid
boxes, non-positive FPS, malformed authority maps, and mixed sequence keys fail
closed. Every sequence resets authority, switch memory, fragments, and episode
state. A global permutation of prediction labels is harmless when the frozen
one-to-one mapping is consistent throughout the sequence.
