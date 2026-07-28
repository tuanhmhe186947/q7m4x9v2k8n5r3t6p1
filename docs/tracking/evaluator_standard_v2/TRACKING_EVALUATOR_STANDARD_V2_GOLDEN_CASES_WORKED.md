# Tracking Evaluator Standard V2 Golden Cases

Date: 2026-07-28

Contract: `TRACKING_EVALUATOR_STANDARD_V2`

Identity contract: `IDENTITY_ERROR_EPISODES_V2`

## Authority and notation

Every expected value below was derived by hand from the V2 contract. No
production evaluator was called to generate an expected value.

Unless a case says otherwise, timestamps are exact, FPS is 10, boxes for an
intended spatial match have IoU 1, and all rows are visible. Therefore one
wrong matched animal-frame contributes 0.1 animal-seconds.

The CSV uses these compact tuples:

- `det_counts_050`: detection `TP/FP/FN` at IoU 0.50.
- `hota_counts`: HOTA `TP/FP/FN` for the stated alpha band.
- `identity`: `IDF1/ID precision/ID recall`.
- `continuity`: `IDSW_STANDARD/fragments`.
- `severity`: wrong animal-frames, wrong seconds, all episodes, recovered
  episodes, terminal episodes, and persistent pairwise swaps.
- `hidden_population`: hidden GT, visible GT, hidden prediction, evaluated
  hidden rows, and evaluated visible rows.

`all19` means each alpha in `0.05, 0.10, ..., 0.95`. Headline HOTA-family
values are arithmetic means over those 19 alpha values.

`IDSW_STANDARD` retains the last matched prediction identity across any
unmatched gap in the same sequence. It has no maximum gap and resets at a
sequence boundary. V2 does not publish a duplicate gap-persistent IDSW field.

## Fixture inputs

### C01 — Perfect single trajectory

One GT identity `g1` and prediction `p1` coincide on frames 1 through 3.
All 19 HOTA slices and all identity ratios are 1. There is no error episode.

### C02 — Perfect two-trajectory sequence

`g1/p1` and `g2/p2` coincide on frames 1 and 2. The four matched detections
give perfect detection, association, localization, and identity results.

### C03 — One missed detection

`g1` exists on frames 1 and 2; `p1` exists only on frame 1. At every alpha,
`DetA=1/2` and `AssA=1/2`, so `HOTA=sqrt(1/4)=1/2`. The global identity
counts are `IDTP=1`, `IDFP=0`, and `IDFN=1`.

### C04 — One false positive

Frame 1 contains coincident `g1/p1` plus one disjoint `p2`. At every alpha,
`DetA=1/2`, `AssA=1`, and `HOTA=sqrt(1/2)`. Identity counts are
`IDTP=1`, `IDFP=1`, and `IDFN=0`.

### C05 — Eligibility-before-assignment counterexample

One frame has GT rows `g1,g2`, prediction rows `p1,p2`, and IoU matrix
`[[0.90,0.60],[0.60,0.49]]`. At IoU 0.50, cardinality-first constrained
matching selects the two off-diagonal eligible edges. An unrestricted
Hungarian solution would select the diagonal and lose one match after gating.

### C06 — One-frame identity error and recovery

For `g1`, matched prediction IDs are `p1,p2,p1` on frames 1 through 3.
The frozen expected identity is `p1`. Frame 2 is one recovered episode;
IDSW counts the transition to `p2` and the transition back to `p1`.

### C07 — Multi-frame identity error and recovery

For `g1`, IDs are `p1,p1,p2,p2,p2,p1`. Frames 3 through 5 form one
three-animal-frame recovered episode. The recovery latency is 0.1 seconds.

### C08 — Error episode persisting to the final observation

For `g1`, IDs are `p1,p1,p2,p2,p2,p2`. The four wrong rows form one
terminal episode. There is one IDSW event and no recovery latency.

### C09 — Reciprocal pairwise swap persisting to the end

Two GT tracks are correct on frames 1 and 2, then exchange prediction IDs on
frames 3 and 4. This creates two GT-level terminal episodes but exactly one
unordered persistent pairwise swap. Terminal reciprocity satisfies persistence
even though there are fewer than 60 direct joint observations.

### C10 — Reciprocal pairwise swap followed by recovery

Two GT tracks are correct on frames 1 and 2, exchanged on frames 3 and 4,
and correct on frame 5. There are two recovered GT-level episodes and four
IDSW events. Two direct joint swap observations are below the persistence
horizon, so the pairwise persistent count is zero.

### C11 — Fragmentation without a wrong identity

`g1` exists on frames 1 through 4. `p1` matches frames 1, 2, and 4.
The return after the missed frame creates one fragment, but no IDSW or
wrong-ID row. `DetA=AssA=HOTA=3/4` at every alpha.

### C12 — Longer unmatched gap followed by the original identity

`g1` exists on frames 1 through 5. `p1` matches frames 1, 4, and 5.
The two-frame miss creates one fragment. Retained switch memory sees the
same prediction identity after the gap, so IDSW remains zero.

### C13 — Reappearance under a new identity

`g1` is matched to `p1` on frame 1, unmatched on frame 2, and matched to
`p2` on frame 3. Gap-persistent standard memory gives one IDSW. The final
`p2` row is one wrong animal-frame and one terminal episode.

### C14 — Hidden interval excluded

Frames 1 and 3 are visible and frame 2 has hidden `g1/p1` rows. With
`include_hidden=false`, only the two visible matches are evaluated. The
filtered hidden frame is not a miss and does not create a fragment.

### C15 — Hidden interval included

The raw rows are identical to C14, but `include_hidden=true`. All three
matches are evaluated, including one hidden GT and one hidden prediction.
The evaluated result remains perfect while the population counters differ.

### C16 — Empty prediction sequence

One visible GT row exists and no prediction exists. HOTA, DetA, and AssA
are zero at all alphas. LocA uses the TrackEval empty-TP convention of 1.
All identity ratios are zero; the empty prediction is not a perfect result.

### C17 — Empty GT sequence

One visible prediction exists and no GT row exists. HOTA, DetA, AssA, and
all identity ratios are zero. LocA is 1 by the empty-TP convention.

### C18 — Multiple videos and reused labels

Sequence `v1` contains one correct `g1/p1` match. Sequence `v2` contains
one correct `g1/p2` match. The reused GT label and changed prediction label
cannot create an IDSW or episode because all authority resets at the boundary.

### C19 — Input-order permutation

This is C02 with frame rows, GT rows, and prediction rows presented in reverse
order. Its canonical report must be byte-equivalent to C02, including empty
event tables and deterministic metadata.

### C20 — Duplicate identity validation failure

One sequence/frame contains two GT rows both labeled `g1`. Validation must
fail deterministically before matching. No partial metric report is emitted.

### C21 — Sixty-observation pairwise persistence horizon

Two GT tracks are correct on frame 1, reciprocally exchanged on frames 2
through 61, and recovered on frame 62. The 60 direct joint observations meet
the frozen horizon. Count one persistent unordered pair even though both
GT-level episodes later recover.

There are 120 wrong animal-frames, not 60 video frames. At 10 FPS they
contribute 12 animal-seconds. Four IDSW events record the two transitions
into the exchange and the two transitions out.

### C22 — Ambiguous first identity authority

One frame uses a tied all-ones `2 x 2` IoU matrix. Canonical matching remains
deterministic and yields two detections, but both first identity links are
audit-ambiguous. The two rows remain in the audit population and are excluded
from authoritative identity-severity ranking.

## Worked metric calculations

For all fixtures except C05, every accepted spatial match has IoU 1.
Consequently each per-alpha value is constant across the 19 alphas.

### Miss and false-positive cases

For C03, one TP and one FN give `DetA=1/(1+0+1)=1/2`. The only identity pair
has one true-positive association and one GT-side miss, so `AssA=1/2`.
Thus `HOTA=sqrt((1/2)(1/2))=1/2`.

For C04, one TP and one FP give `DetA=1/2`. The matched pair has perfect
association, so `AssA=1` and `HOTA=sqrt(1/2)`.

### C05 HOTA bands

HOTA intentionally uses its own official global-alignment assignment. For
C05 it selects the diagonal, separately from the generic constrained matcher.

- Alpha 0.05 through 0.45: both diagonal edges pass. The result is
  `HOTA=DetA=AssA=1` and `LocA=(0.90+0.49)/2=0.695`.
- Alpha 0.50 through 0.90: only the 0.90 edge passes. Therefore
  `DetA=1/(1+1+1)=1/3`, `AssA=1`, `LocA=0.9`, and `HOTA=sqrt(1/3)`.
- Alpha 0.95: no edge passes. HOTA, DetA, and AssA are 0; LocA is 1.

There are nine alpha values in each of the first two bands and one in the
last band. The manually derived headlines are:

- `HOTA=(9 + 9*sqrt(1/3))/19 = 0.747165917`.
- `DetA=(9 + 9/3)/19 = 12/19 = 0.631578947`.
- `AssA=18/19 = 0.947368421`.
- `LocA=(9*0.695 + 9*0.9 + 1)/19 = 0.808157895`.

### Single-GT identity episodes

For one GT with prediction-fragment lengths `a` and `b`, perfect detection
gives `AssA=(a^2+b^2)/(a+b)^2`.

- C06 uses `a=2,b=1`: `AssA=5/9` and `HOTA=sqrt(5)/3`.
- C07 uses `a=3,b=3`: `AssA=1/2` and `HOTA=sqrt(1/2)`.
- C08 uses `a=2,b=4`: `AssA=5/9` and `HOTA=sqrt(5)/3`.

The C06 and C07 switchbacks each create two IDSW events. C08 has one initial
transition and no recovery, so it has one IDSW despite longer wrong duration.

### Reciprocal swaps

For two length-`N` tracks with `k` correct and `b=N-k` exchanged matches,
the correct-pair Jaccard is `k/(2N-k)` and the exchanged-pair Jaccard is
`b/(2N-b)`. Their TP-weighted average is AssA.

C09 has `N=4,k=2,b=2`; every association Jaccard is `1/3`. Hence
`AssA=1/3`, `HOTA=sqrt(1/3)`, and global identity assignment retains four
of eight detections, giving `IDF1=IDP=IDR=1/2`.

C10 has `N=5,k=3,b=2`. Its AssA is
`(3*(3/7)+2*(1/4))/5=5/14`, so `HOTA=sqrt(5/14)`. Global identity
assignment retains six of ten detections, giving all three identity ratios
as `3/5`.

C21 has `N=62,k=2,b=60`. Its AssA is
`(2*(1/61)+60*(15/16))/62=13733/15128`. Therefore
`HOTA=sqrt(13733/15128)=0.952778508`. Global identity assignment retains
120 of 124 detections, so all three identity ratios are `30/31`.

### Gaps and identity assignment

C11 has three matches from four GT rows. Both DetA and pair association are
`3/4`, so HOTA is `3/4`. Its identity counts are
`IDTP=3,IDFP=0,IDFN=1`.

C12 analogously has three matches from five GT rows. DetA, AssA, and HOTA
are `3/5`, with identity counts `IDTP=3,IDFP=0,IDFN=2`.

C13 has one observation under each of two prediction IDs and one miss.
Thus `DetA=2/3`, `AssA=1/3`, and `HOTA=sqrt(2)/3`. A one-to-one global
identity assignment retains one row: `IDTP=1,IDFP=1,IDFN=2`.

## Episode and duration expectations

Wrong duration is summed exposure, not the inclusive frame span. C09 and C10
each have two animals wrong on two frames, which is four animal-frames and
0.4 animal-seconds. C21 has two animals wrong on 60 frames, which is
120 animal-frames and 12 animal-seconds.

Each C06, C07, C10, and C21 recovered episode has a 0.1-second latency from
its final wrong observation to its first later correct match. Terminal cases
C08, C09, and C13 have null recovery latency.

The pairwise event is a secondary unordered cross-link. It never consumes or
duplicates the two primary GT-level episodes. Persistence requires either
60 direct reciprocal joint observations or reciprocal terminal episodes at
both final authoritative observations.

## Required invariants

Each metric implementation must satisfy all of these assertions:

- At every alpha, `TP+FN` equals evaluated GT detections.
- At every alpha, `TP+FP` equals evaluated prediction detections.
- All HOTA-family ratios remain in `[0,1]`.
- Every authoritative wrong row belongs to exactly one GT-level episode.
- Recovered and terminal are mutually exclusive.
- A pairwise event references exactly two GT trajectories and is counted once.
- No IDSW, fragment, episode, or identity authority crosses C18's boundary.
- C19 is invariant to every declared input permutation.
- C20 fails before emitting a report.
- C22 preserves ambiguous rows in audit data but excludes them from ranking.

## Test-use rules

Exact integer, null, validation, and canonical-output assertions allow no
tolerance. Rational and square-root ratios use absolute tolerance `1e-12`
when expressed exactly and `1e-9` when the CSV gives nine decimal places.

Tests may materialize these hand-authored inputs, but expected outputs must
remain literal fixture data. Updating expected values from production output
would invalidate this golden-case authority.
