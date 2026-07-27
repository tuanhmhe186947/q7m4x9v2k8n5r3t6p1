# H2-CDSP scientific design contract

Status: frozen design only  
Hypothesis: `H2_CAUSAL_DROPOUT_STATE_PRESERVATION`  
Short name: `H2-CDSP`

## 1. Authority and evidence language

The hidden-owner preference family is closed for the current study:

- H1-r1: `REJECTED_CONFIGURATION`;
- H1-r2: `FAIL_NO_ACTIVATION`;
- H1-r3: `FAIL_NO_SHADOW_ACTIVATION`;
- H1-r4: not authorized.

Historical RF_ACC23 evidence was produced at `b0d9009`. Its role is
`MECHANISM_DISCOVERY_ONLY`. It does not measure current-main failure
prevalence or establish implementation benefit. The included historical
mechanistic population contains six authoritative `OCCLUSION_OWNER_LOSS`
events with 2,092 wrong-ID frames, 42.503048% of the historical audited
wrong-ID population. Event `000216` has unresolved GT authority and is
excluded.

Before production implementation can be authorized, a separately authorized
side-effect-free current-main shadow study must reproduce the relevant state
loss and measure the non-intervening H2 operating population.

## 2. Hypothesis

Some short causal detector dropouts or merge-like occlusions prematurely
discard or degrade otherwise trustworthy track state. Preserving a bounded,
uncertainty-aware snapshot may leave better causal evidence available at a
later ordinary re-entry association.

H2 does not assert that an old or hidden track owns any detection. It does not
reserve or directly assign a detection. Its intervention is state retention,
not owner preference.

## 3. Intervention boundary

H2 may preserve:

- last trusted bbox geometry and normalized center/scale;
- bounded causal velocity and log-scale displacement;
- motion uncertainty and reliability;
- last trusted appearance prototype, quality, and age;
- last confirmed detector frame;
- dropout age, state confidence, and invalidation reason.

H2 must not:

- fabricate a detection or create a track;
- add an emitted object while no detector match exists;
- refresh state from an unassigned or weak detection;
- reserve a detection or block a visible assignment;
- override a strong visible assignment because a track is old;
- transfer state across videos;
- use GT, video key, episode ID, date, frame range, or role at runtime;
- use a future frame, offline repair, or smoothing.

The four stages are separate:

1. **state memory preservation** copies a trusted snapshot without changing an
   assignment;
2. **causal state propagation** predicts bounded geometry and grows
   uncertainty from past state only;
3. **association evidence at re-entry** exposes a reference plus quality and
   uncertainty to an explicitly authorized ordinary association path;
4. **assignment decision** remains outside H2-CDSP and cannot consume an H2
   diagnostic as a command.

The sole permitted future association consumer is a bounded, track-local
substitution. A `PreservedStateEvidence` record may replace a missing or
baseline-degraded bbox, motion reference, or appearance reference for the same
still-live track. The ordinary baseline candidate set, costs, weights, gates,
and assignment solver remain unchanged. The substitution adds no penalty,
bonus, owner score, reservation, veto, candidate, or direct assignment. Its
maximum influence is the influence already permitted to that same track-local
evidence channel by ordinary association. Any other consumer is outside this
contract. If this bounded substitution produces no real association
divergence, the future result is `FAIL_NO_ASSOCIATION_EFFECT`.

## 4. Trusted snapshot

A snapshot becomes trusted only after a current-frame detector match that the
ordinary baseline association has already accepted and all of these hold:

- detection confidence meets the selected profile's existing
  `track_high_conf` threshold;
- `FixedTrack.state == "VISIBLE"`, `last_source == "detected"`,
  `last_ambiguous == false`, and `hits >= 4`;
- bbox and frame index are finite and valid;
- the baseline assignment is unique rather than tied or ambiguous;
- the match belongs to the current video and current frame.

On a trusted match, `initial_state_confidence=1` and normalized initial
uncertainty is exactly `U_0=0.10`. Optional evidence may be updated only from
that assigned detection. H2 does not introduce a new detector threshold.
Weak, ambiguous, or unassigned detections are a non-transition invariant: they
cannot refresh the snapshot, appearance, motion, confidence, or uncertainty.

## 5. Frozen preservation semantics

Let `a` be integer frames since the trusted detector frame. Constants are:

- confidence half-life `H_c=6` frames;
- appearance half-life `H_a=8` frames;
- motion half-life `H_m=4` frames;
- base uncertainty growth `g_0=0.05` per frame;
- missing/weak-motion growth `g_m=0.10` per frame;
- camera-boundary penalty `b=0.15`;
- maximum preservation age `A_max=10` frames;
- minimum usable confidence `C_min=0.30`;
- maximum usable uncertainty `U_max=0.75`.

The exact formulas are frozen in
`H2_CDSP_PRESERVATION_FORMULAS.md` and the state registry. A preserved state is
usable only when every core validity condition holds, `a <= A_max`,
`C(a) >= C_min`, and `U(a) <= U_max`.

Optional appearance and motion are not required for a short preservation.
Their absence contributes zero reliability, cannot raise confidence, and may
shorten usable duration through uncertainty growth.

## 6. Geometry normalization

Center displacement is divided by the trusted bbox diagonal. Width and height
changes use log ratios. Per-frame normalized center velocity is clipped to
Euclidean magnitude `0.25`; each absolute log-scale rate is clipped to
`log(1.25)`. These bounds are identical for small and large boxes.

When motion is unavailable, geometry remains at its last trusted value and
uncertainty grows at the maximum missing-motion rate. Once any propagated box
at age `1..a` crosses the image boundary, the frozen boundary uncertainty
penalty persists for all later ages in that snapshot. Clipping or later
re-entry into the image does not remove the penalty or reduce uncertainty.

## 7. State machine

The declared states are:

- `VISIBLE_CONFIRMED`;
- `DROPOUT_GRACE`;
- `OCCLUSION_PRESERVED`;
- `STALE_PRESERVED`;
- `INVALIDATED`;
- `TERMINATED`.

All transitions and their exact guards are in
`H2_CDSP_STATE_MACHINE.json`. `TERMINATED` is absorbing. `INVALIDATED` exposes
no H2 association evidence. It may return to `VISIBLE_CONFIRMED` only after
the ordinary baseline independently accepts a new trusted match to the same
still-live track.

The H2 lifecycle maps to current `FixedTrack` state as follows:

- `VISIBLE_CONFIRMED` requires the trusted-match predicate above;
- `DROPOUT_GRACE` requires a live track in baseline `MISSING` or `OCCLUDED`
  state at dropout age 1 or 2;
- `OCCLUSION_PRESERVED` requires a live baseline `OCCLUDED` track at age 3
  through 6 plus a causal occlusion signal: `last_source=="occlusion_hold"`,
  `is_area_occluded`, or `last_merged_split`;
- `STALE_PRESERVED` covers a live track at age 3 through 10 without qualifying
  occlusion support, or any live preserved track at age 7 through 10;
- `INVALIDATED` disables only H2 evidence; the baseline track may still exist;
- `TERMINATED` means the `fixed_id` has been removed from the active baseline
  track dictionary or explicitly removed by the baseline lifecycle.

Baseline `LOST` alone is not H2 termination. It routes to `INVALIDATED` while
the object remains in the active track dictionary. Transition precedence is:
terminal absorption; sequence reset; baseline removal; an ordinary trusted
match from `INVALIDATED`; invalidation of any other unsafe state; an ordinary
trusted match from a non-invalidated state; deterministic age/occlusion
routing; then fail-closed invalidation. An `INVALIDATED` state without a new
ordinary trusted match remains `INVALIDATED`.

H2 never grants emission permission. Existing baseline output behavior remains
outside the preservation state machine.

## 8. Safety invariants

- Age alone never increases confidence.
- Uncertainty never decreases without a new trusted match.
- Missing evidence never becomes positive evidence.
- Repeated skipped frames monotonically weaken preserved state.
- Long dropout becomes unusable no later than frame age 11.
- Invalid or non-finite state fails closed.
- A terminal track never silently revives.
- Cross-video state is cleared before the first frame of the next video.
- Only an ordinary trusted match can reset confidence or uncertainty.
- Preserved state is evidence, never an assignment or reservation command.

## 9. Difference from H1

| Property | H1-r1 | H1-r2 | H1-r3 | H2-CDSP |
|---|---|---|---|---|
| Intervention time | re-entry | re-entry | re-entry shadow | before re-entry |
| Primary object | owner claim | pair score | lower-bound pair score | track state |
| Heterogeneous raw cost | yes | no | no | no |
| Hidden-only gate | yes/effective | yes | no | no |
| Owner score threshold | yes | yes | yes | none |
| Reserves detection | intended | intended | shadow only | never |
| Blocks visible assignment | intended | intended | never in shadow | never |
| Direct assignment command | reservation | reservation | none | none |
| Preserves causal evidence | incidental | incidental | diagnostic only | primary |

If a future specification introduces an owner/competitor score, detection
reservation, or visible-assignment veto, it is not H2-CDSP and this design
checker must fail.

## 10. Causality and runtime exclusions

All H2 inputs are from the current or earlier frame. The design is
`causal_framewise` with output delay 0. It has no future-frame dependency,
post-video pass, or smoothing. Runtime scoring cannot read GT or offline
development roles.

## 11. Current-main shadow prerequisite

The future shadow study is mandatory and frozen in
`H2_CDSP_CURRENT_MAIN_SHADOW_PREREQUISITE.json`. It must:

- run current-main `realtime_fast` with H2 diagnostics side-effect-free;
- include the six authoritative historical positive events and four
  predeclared controls from the historical mechanism-discovery population;
- reproduce current-main baseline state loss rather than assume historical
  prevalence;
- change no assignment, track state, output, cost matrix, or profile default;
- use no H1/H2 validation window.

Passing requires at least two independent positive events spanning at least
two video keys and two recording sessions, with a reproduced baseline
state-loss point, usable H2 state past that point and through a relevant
re-entry opportunity, bounded control activity, and exact shadow output
equivalence. If the video/session independence requirement is unmet, the
result is `INCONCLUSIVE`, never pass.

## 12. Validation policy

Historical taxonomy events and all shadow-development windows are development
evidence. They cannot serve as independent final validation. A new untouched
H2 validation population must be frozen before implementation, disjoint by
video, recording date/session, source video, and preferably pen/session. No H2
validation output may be inspected before separate authorization.

## 13. Future implementation boundary

Any future implementation must be opt-in, disabled by default, telemetry
independent, causal framewise, delay 0, and unable to change baseline behavior
unless explicitly selected. No future profile name is declared here.

This contract authorizes neither shadow execution nor production
implementation, association evaluation, validation, runtime testing, or
promotion.
