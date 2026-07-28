# Tracking Evaluator Standard V2 Contract

Date: 2026-07-28

Contract IDs:

- `TRACKING_EVALUATOR_STANDARD_V2`
- `TRACKING_MATCHING_STANDARD_V2`
- `IDENTITY_ERROR_EPISODES_V2`

This contract governs new tracking evaluations only. Historical reports remain
under `TRACKING_EVALUATOR_LEGACY_V1`; their stored fields are not renamed or
reinterpreted. Headline comparisons across the two evaluator versions are
invalid until frozen outputs are re-evaluated under this contract.

## Population and sequence boundaries

One authoritative sequence is one video/session pair. Metric state, identity
assignment, switch memory, fragments, and episodes reset at every sequence.
Dataset aggregation combines sequence sufficient statistics; it never
concatenates identity namespaces.

The evaluator receives all rows, validates them, then applies `include_hidden`
symmetrically to GT and predictions. It records:

- `hidden_gt_rows` and `visible_gt_rows` before filtering;
- `hidden_prediction_rows` before filtering;
- `evaluated_hidden_rows` and `evaluated_visible_rows` for the GT population.

With `include_hidden=false`, hidden rows do not match, create misses, switches,
fragments, or episodes. With `include_hidden=true`, they participate normally.
An empty or all-filtered population is not treated as perfect.

## Matching contracts

Detection, CLEAR-style identity continuity, and episode construction use
eligibility-constrained assignment at IoU `0.50`. Objects are canonically
ordered before assignment. Ineligible edges are excluded before the solution;
the objective maximizes eligible cardinality first and total IoU second.
Duplicate object identities within a frame and object/frame-key mismatches
fail validation.

Standard HOTA is a separate matching contract. It uses the official TrackEval
global-alignment score multiplied by localization similarity, performs the
official per-frame Hungarian assignment, and applies each alpha acceptance
test to that assignment. It must not reuse the generic detection matcher.
This reference-prescribed behavior is the sole exception to generic pre-gating.

Identity-global assignment follows TrackEval Identity at IoU `0.50`: every
eligible GT/prediction identity co-occurrence contributes to the sequence
matrix, dummy rows and columns represent unmatched identities, and one global
minimum IDFP+IDFN assignment is solved per sequence.

## Standard HOTA family

The canonical alpha set is `0.05, 0.10, ..., 0.95` (19 values). At each alpha:

- `TP`, `FP`, and `FN` are accepted HOTA matches and unmatched detections.
- `DetA = TP / (TP + FP + FN)`.
- Pair association is `TPA / (TPA + FPA + FNA)`.
- `AssA` is that association Jaccard averaged over true-positive detections.
- `LocA` is mean IoU over true-positive detections.
- `HOTA = sqrt(DetA * AssA)`.

All five ratios are dimensionless in `[0,1]`; higher is better. Headline
`hota`, `deta`, `assa`, and `loca` are arithmetic means of their 19
dataset-combined alpha values. A fixed-threshold value is diagnostic only and
must include its alpha in the field name.

For zero TP, HOTA, DetA, and AssA are `0`. LocA follows TrackEval's explicit
empty convention of `1`; this does not make the empty sequence perfect because
HOTA remains `0`.

## Identity metrics

TrackEval-compatible global identity assignment yields `IDTP`, `IDFP`, and
`IDFN` per sequence. The reported ratios are:

- `IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)`;
- `ID precision = IDTP / (IDTP + IDFP)`;
- `ID recall = IDTP / (IDTP + IDFN)`.

They are dimensionless `[0,1]` ratios; higher is better. Zero denominators
produce `0`, and dataset values recompute from summed per-sequence counts.

`IDSW_STANDARD` follows the pinned TrackEval CLEAR sequence policy. Assignment
first preserves an eligible pairing from the immediately previous timestep,
then maximizes remaining eligible IoU. Switch memory retains the last matched
prediction identity for a GT across any unmatched gap within that sequence.
It has no maximum-gap cutoff and resets at the sequence boundary.

The previous project implementation used the same gap-persistent memory, so V2
does not publish a duplicate `IDSW_GAP_PERSISTENT_PROJECT` value. IDSW remains
an event count, not an identity-error severity score.

## Detection and continuity metrics

At IoU `0.50`, `TP` is eligible matched detections, `FN = GT - TP`, and
`FP = prediction - TP`. Counts are nonnegative and lower is better for FP/FN.
Detection precision is `TP / (TP + FP)` and recall is
`TP / (TP + FN)`; both are `[0,1]`, higher is better, and zero denominators
produce `0`.

`fragments` is the standard sequence-local count of a GT trajectory resuming a
matched state after one or more authoritative GT observations were unmatched.
Frames on which that GT has no evaluated annotation do not themselves create a
fragment. The count resets at each sequence boundary.

For every sequence and eligible threshold:

- `TP + FN` equals the authoritative evaluated GT detection count;
- `TP + FP` equals the authoritative evaluated prediction count.

## Identity-error episodes

Episode severity uses a frozen sequence mapping distinct from IDF1's global
assignment. An explicit one-to-one authority map may be supplied. Otherwise,
`IDENTITY_AUTHORITY_FIRST_OBSERVATION_V2` maps an unmapped prediction identity
to an unmapped GT at their first unambiguous eligible spatial match, then never
changes that mapping. A tied first assignment is audit-ambiguous. This prevents
a long later swap from redefining the expected identity.

For a resolved GT, a matched row is wrong when its prediction identity differs
from that GT's frozen expected prediction identity. An unmapped new prediction
matched to a resolved GT is wrong but has no pairwise target. Rows for GT
without resolved authority are preserved in an ambiguity table and excluded
from authoritative severity totals.

The primary partition key is `(sequence_key, gt_id)`. Consecutive wrong rows
join one episode only when no correct matched observation intervenes and their
frame delta is at most 15. An unmatched row neither adds duration nor recovers
an episode. A larger delta splits the component and censors the earlier one
unless a subsequent correct match proves recovery.

A recovered episode has a later authoritative correct match. A terminal
episode contains that GT's final authoritative matched observation and has no
recovery. Recovered and terminal are mutually exclusive. Every authoritative
wrong row belongs to exactly one GT episode.

`persistent_pairwise_identity_swap` is a secondary cross-link. At one frame,
GT A must carry B's frozen prediction identity while GT B carries A's. The
unordered pair is counted once and may never duplicate the two primary
GT-level episodes or their wrong rows. Reciprocal observations connect at the
same maximum frame delta of 15.

A reciprocal pair event is persistent when it has at least 60 direct joint
observations, or when both linked GT episodes are terminal and continue to
target each other at their final authoritative observations. Three-way cycles
are not pairwise swaps. All event IDs and ordering are deterministic.

Episode duration in frames is the number of contributing GT-match
observations, measured in animal-frames, never the frame span. With valid FPS,
seconds equal summed per-row exposure `1/FPS`, measured in animal-seconds.
Without valid time authority, seconds and recovery-latency fields are null.
Recovery latency runs from the final wrong observation to the first later
correct authoritative match.

## Exposure-normalized diagnostics

The denominator for frame-normalized diagnostics is the count of authoritative
matched GT observations. V2 reports:

- `IDSW_STANDARD` per 1,000 authoritative matched GT frames;
- wrong-ID animal-frames per 1,000 authoritative matched GT frames;
- percentage of resolved GT trajectories with any identity-error episode;
- percentage of evaluated videos with any terminal episode;
- percentage of evaluated videos with any persistent pairwise swap.

Seconds-based exposure and episode summaries are reported only with valid
positive FPS. Median uses the ordinary midpoint definition. P95 uses the
deterministic nearest-rank definition. No composite severity score is formed.

## Aggregation

HOTA aggregation follows TrackEval: sum per-alpha TP/FP/FN across sequences,
TP-weight AssA and LocA at each alpha, recompute DetA and HOTA, then average
the 19 combined alpha values. It never averages per-video headline HOTA.

Identity aggregation sums per-sequence IDTP/IDFP/IDFN and recomputes IDF1,
ID precision, and ID recall. Detection and event counts sum. Frame-normalized
rates recompute from summed numerators and denominators. Video and trajectory
percentages use unique authoritative sequence/trajectory denominators.

The per-video table is mandatory. Mean-video, median-video, best-video, and
worst-video values are descriptive secondary statistics and cannot replace the
dataset-authoritative aggregate.

## Zero denominators and validation

Unless the TrackEval LocA empty convention is stated above, a zero denominator
produces `0`, not a perfect score. Percentage fields are explicitly `[0,100]`;
all other ratios remain `[0,1]`.

Validation fails closed on duplicate `(frame, identity)` rows, frame-key/object
frame disagreement, non-finite or inverted boxes, non-positive FPS, malformed
authority maps, and mixed sequence keys. Canonical object and sequence ordering
makes all metrics and event IDs invariant to input order.

Every alpha must conserve GT and prediction counts. Every authoritative wrong
row must be classified exactly once; ambiguous rows are retained separately;
terminal is a subset of all episodes; pairwise links contain exactly two
distinct GT identities and are unique under symmetric ordering.

## Versioned output

Every row and report carries the evaluator, matching, and episode contract IDs;
alpha set; hidden policy; sequence-boundary policy; IDSW policy; identity
authority policy; reference parity status; evaluator Git SHA; and canonical
metric-config SHA-256.

New legacy report generation is forbidden. A historical unversioned row may be
read only when explicitly classified as `TRACKING_EVALUATOR_LEGACY_V1`. A
checker rejects unversioned V2, mixed legacy/V2 rows, inconsistent V2 metadata,
legacy `permanent_swap` or `terminal_swap` fields in V2, and simultaneous
legacy remapped HOTA/AssA with standard V2 HOTA/AssA.

Corrected V2 fields never silently overwrite historical files. Frozen-output
re-evaluation must write a new run directory and preserve source hashes.

## Reference authority

Executable parity authority is the official TrackEval repository at commit
`12c8791b303e0a0b50f753af204249e622d0281a`, licensed MIT by Jonathon
Luiten (2020). It is stored reference-only outside production imports. V2 must
match its HOTA, DetA, AssA, LocA, IDF1, ID precision, ID recall, and supporting
counts within the declared numerical tolerance.

TrackEval HOTA's official assignment occurs once per frame before alpha
acceptance. That behavior is reproduced only in the HOTA implementation; it
does not authorize post-assignment gating in detection, CLEAR, or episode
matching.
