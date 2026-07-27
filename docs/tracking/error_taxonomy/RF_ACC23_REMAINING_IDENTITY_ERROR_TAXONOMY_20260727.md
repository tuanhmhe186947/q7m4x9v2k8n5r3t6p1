# RF_ACC23 remaining identity-error taxonomy — 2026-07-27

## Decision

The bounded audit supports one design-stage hypothesis:
`CAUSAL_DROPOUT_STATE_PRESERVATION`.

This is not an implementation or run authorization. It is deliberately not
another hidden-owner preference score. It would preserve established track
state through a predeclared, bounded detector-dropout or merge condition and
would target state lifecycle rather than rank a hidden and visible owner.

H1-r1, H1-r2, and H1-r3 remain closed for the current study. H1-r3 repaired
support density but produced zero activation at its frozen gate. Its maximum
uncalibrated `owner_preference_lower_bound` of `0.59954303046` does not justify
lowering the frozen `0.625` threshold.

## Scope and lineage

No tracking, detector inference, GPU inference, validation, Hard6, full-13, or
runtime benchmark was run. The audit reads the recovered locked RF_ACC23
full-13 artifacts only.

The quality artifacts were produced at
`b0d90098b2ae1fcdcfe8ca4faaca7a215631ec66`. Byte equivalence between that
tracking tree and the later promoted main tracking tree has not been proven.
The findings therefore describe the measured recovered RF_ACC23 lineage.

Locked input integrity:

- recovered evaluation artifact manifest:
  `f94b1613c86b8248fc608094e3f1183da53b56d00a3d97c789713adb1785cdce`;
- remapped identity events:
  `85a4a983d87f598c5ca2f9a5b67af42de2e09464f2f0f844e82abf2a3f5a8f48`;
- tracking metrics:
  `46f34d4ebc898c5cd22555d5c6d3e8bcb8e3faa681cd03ef84b9547ffd9b84e1`;
- H1-r3 shadow candidate pairs:
  `1503a361ae88ad28167fa07e45b7fb08ed4ff40cc428cf209d1c7acef427d1c8`.

The artifact manifest records zero MP4 outputs.

## Event definition

The unit is a connected wrong-identity episode, not one row per animal.
Simultaneous paired or cascading mappings are merged when they share an
identity. A gap of at most 15 frames may bridge the episode only when identity
connectivity remains. The procedure conserved:

- all `4,922` wrong-ID matched-animal rows;
- all `53` ID-switch rows;
- ten distinct error events.

`terminal` means the event is still wrong on the final evaluated frame.
`permanent` means terminal or at least 60 frames long. This explicit duration
definition distinguishes consequential long-lived errors from short
self-recovering swaps without counting both animals as separate events.

## Primary taxonomy

| Primary mechanism | Events | Wrong-ID frames | Share | Permanent | Terminal |
|---|---:|---:|---:|---:|---:|
| OCCLUSION_OWNER_LOSS | 6 | 2,092 | 42.503% | 2 | 2 |
| TRACK_BIRTH_OR_DUPLICATE_TRACK | 2 | 1,619 | 32.893% | 2 | 0 |
| REENTRY_AFTER_LONG_HIDDEN_DURATION | 1 | 868 | 17.635% | 1 | 1 |
| GT_OR_EVALUATION_AMBIGUITY | 1 | 343 | 6.969% | 1 | 0 |

No event was forced into an unsupported appearance, LK, motion-propagation, or
visible-visible category. LK success and motion-history state were not
exported, so those fields are `NOT_EXPORTED`. Finite association costs establish
that association evidence was exported; they do not independently establish
appearance-descriptor availability.

The `000216` event remains `GT_OR_EVALUATION_AMBIGUITY` because its source GT
authority is unresolved in project memory. It contributes to the conserved
population but not to a mechanistic association conclusion.

## Event-level findings

- Two long initial identity substitutions account for 1,619 wrong-ID frames.
  They begin at frame zero, where no causal history exists.
- Six occlusion-owner-loss events begin with a missing detection, a hidden or
  occlusion-held track, overlapping tracks, and available association rows.
- One terminal `000231` event begins with a missing detection, lost-track state,
  and a re-identification phase but without exported hidden state at onset.
- Three terminal events remain wrong through frame 1799: `000231`, `000233`,
  and `000327`.

The event CSV records affected identities, duration, onset evidence, recovery,
terminal/permanent status, GT authority, and hashes of every per-video quality
report, debug trace, and prediction XML used.

## Was hidden-owner preference aimed at the real problem?

Partly. Genuine hidden-owner contention accounts for six events and 2,092
wrong-ID frames, or 42.503% of the audited wrong-ID duration. That is the
largest primary category, but 57.497% lies outside it.

H1-r3 shadow coverage exists for only two of the six error events because the
shadow audit was intentionally limited to frozen development windows:

- both measured events contained core-eligible H1-r3 rows;
- one event (`000263`, frames 798–837) contained two rows within the
  predeclared descriptive band of 0.05 below the frozen gate;
- neither measured event had a frozen-gate activation;
- no row disagreed with the baseline because no activation occurred;
- the other four events are `NOT_MEASURED`, not zero.

Thus H1-r3 theoretically addressed the dominant category but only partially
covered it empirically and never reached an intervention. Score magnitude alone
does not establish that an assignment would have changed or improved.

## Opportunity ranking

The ranking multiplies normalized error impact, frequency, causal-evidence
availability, intervention specificity, and evaluation feasibility. It
penalizes frame-zero proposals because they lack causal history.

1. `CAUSAL_DROPOUT_STATE_PRESERVATION` targets the six dominant events at their
   shared, measured onset: bounded detector dropout or merge plus existing
   track state.
2. `REENTRY_SPECIFIC_IDENTITY_CONTINUITY` targets one terminal event.
3. Track-birth protection has high frame impact but no pre-event causal history.
4. The `000216` ambiguity requires GT-authority resolution, not a tracker rule.

The selected hypothesis has a non-empty observed operating condition and can
predeclare positive episodes from dropout-associated identity-loss events and
controls from dropout/overlap episodes that retain identity. Its design must
specify a bounded hold, release conditions, duplicate-track risk, and hard stop
rules before any implementation. It must preserve the untouched validation
population.

## Authorization status

- hidden-owner preference family: `CLOSED_FOR_CURRENT_STUDY`;
- H1-r4: not authorized;
- new implementation: not authorized;
- new tracking run: not authorized;
- validation: not executed and not authorized;
- runtime and promotion: not authorized.

Detailed generated evidence is under
`outputs/tracking/rf_acc23_error_taxonomy_20260727/`. The deterministic
inventory binds every generated artifact.
