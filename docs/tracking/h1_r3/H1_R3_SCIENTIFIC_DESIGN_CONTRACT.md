# H1-r3 Scientific Design Contract

Date frozen: 2026-07-27

Status: design only. Implementation, evaluation, runtime benchmarking, merge
of an implementation, profile creation, and promotion are not authorized.

## Locked prior conclusions

- `H1_R1_STATUS=REJECTED_CONFIGURATION`
- `H1_R2_STATUS=FAIL_NO_ACTIVATION`
- `H1_R2_FEATURE_PLUMBING_DEFECT=NO`
- `H1_R2_MISSINGNESS_CONTRACT_TOO_RESTRICTIVE=YES`
- `H1_R2_SCORE_OPERATING_RANGE_FEASIBLE=NO`
- `H1_R2_QUALITY_EFFECT=NO_EFFECT_AT_FROZEN_GATE`
- `H1_R2_SAFETY_UNDER_REAL_ACTIVATION=NOT_MEASURED`

H1-r1 and H1-r2 remain closed. Their thresholds are not reopened. The H1-r2
profile remains unpromoted.

## Scientific hypothesis

A causal hidden owner should reserve a contested detection only when the owner
has a sufficiently stronger combination of normalized geometry, recency, and
comparable optional evidence than the visible competitor. Candidate evidence
must be defined on one common scale, and missing evidence must never create a
signed advantage.

The primary eligibility design is
`symmetric_iou_recency_core_with_conservative_optional_bounds`.

For a track candidate `t` and contested detection `d`, the same feature map is
used on both sides:

```text
x_hidden = F(hidden, detection)
x_visible = F(visible, detection)
```

The pair is eligible only when the detection is valid and both candidates pass
the same IoU, recency, reference-box provenance, LK validity, and numerical
rules. There is no hidden-only overlap floor.

## Eligibility alternatives

### No hard overlap floor

Overlap remains a bounded soft feature. This removes the H1-r2 failure in
which a present value below `0.50` was mislabeled as missing evidence. Alone,
however, this option does not state the minimum evidence needed to compare two
tracks.

### Symmetric evidence-group eligibility

Both candidates must have valid reference geometry and a causal recency value.
Appearance and motion are not mandatory. This is the selected eligibility
design because it expresses the minimum information logically needed for a
pairwise spatial ownership comparison without requiring all eight fields.

### Relative overlap evidence

Overlap contributes through
`overlap_hidden - overlap_visible`, never through an absolute hidden-only gate.
This is incorporated into the selected design.

### Quality-weighted availability

Observed appearance and motion use the lower side quality. If either side is
missing, the score carries the full bounded uncertainty interval for that
channel. Activation uses the worst-case lower bound. This prevents masking
adverse competitor evidence from making the hidden owner more favorable.

## Core and optional evidence

Core required features are:

- `overlap_similarity`;
- `track_freshness`.

They are defined identically for hidden and visible tracks. Each candidate must
have a finite positive-area causal reference box and an age from zero through
eight detector opportunities. The same detection confidence floor `0.25` is
checked once on the shared contested detection.

`normalized_center_similarity` and `scale_similarity` remain symmetric
diagnostics with zero score weight. They are not counted as independent
evidence because all three geometry quantities are correlated functions of the
same two boxes.

Both candidates use this reference-box cascade: causally valid LK propagation,
then a finite causal prediction, then the last confirmed finite box. LK uses at
least four attempted and three valid points with forward-backward error at most
`1.5` pixels. No candidate receives a different provenance or validity rule.

Optional features are:

- `appearance_similarity` with `appearance_available`;
- `motion_consistency` with `motion_available`.

When both sides are available, an optional contribution is the quality-weighted
hidden-minus-visible difference. When either side is missing, the channel
contributes its full possible interval: appearance `[-0.15,+0.15]` and motion
`[-0.10,+0.10]`. Activation uses the lower endpoint. Masking an observed
channel therefore cannot raise hidden-owner confidence.

Appearance quality decays as:

```text
q_app(t) = 2^(-descriptor_age_detection_opportunities / 4)
q_app_pair = min(q_app(hidden), q_app(visible))
```

Motion quality decays as:

```text
q_motion(t) =
    base_quality * 2^(-prediction_age_frames / 2)

base_quality =
    LK_valid_fraction, for causal LK
    0.5, for another valid causal predictor

q_motion_pair = min(q_motion(hidden), q_motion(visible))
```

Every quality lies in `[0,1]`. A future implementation must use current or past
frames only.

## Pairwise score

Define core differences:

```text
delta_overlap = overlap_hidden - overlap_visible
delta_freshness = freshness_hidden - freshness_visible
core_support = 0.60 * delta_overlap + 0.15 * delta_freshness
```

For an observed optional channel:

```text
appearance_support =
    0.15 * q_app_pair * (appearance_hidden - appearance_visible)

motion_support =
    0.10 * q_motion_pair * (motion_hidden - motion_visible)
```

If either side is missing, use the corresponding full uncertainty interval.
Sum interval endpoints:

```text
relative_owner_support_lower =
    core_support + appearance_lower + motion_lower

relative_owner_support_upper =
    core_support + appearance_upper + motion_upper

owner_preference_lower_bound =
    0.5 + 0.5 * relative_owner_support_lower

owner_preference_upper_bound =
    0.5 + 0.5 * relative_owner_support_upper
```

The owner-preference bounds are uncalibrated and are not probabilities.

Weights sum to one. IoU has the largest influence; freshness, appearance, and
motion are supporting evidence. Center and scale have zero score weight.

## Required mathematical behavior

- Identical observed evidence gives support interval `[0,0]` and score `0.5`.
- Swapping hidden and visible maps `[lower,upper]` to `[-upper,-lower]`.
- Stronger hidden evidence increases preference.
- Stronger visible evidence decreases preference.
- Equivalent normalized geometry is invariant to uniform bbox scale.
- Masking evidence cannot increase the hidden-owner lower bound.
- Longer hidden age lowers freshness and cannot improve preference by itself.
- All terms remain finite and bounded.

## Frozen activation rule

Reserve for the hidden owner only when:

```text
pair is eligible
delta_overlap >= 0.10
relative_owner_support_lower >= 0.25
owner_preference_lower_bound >= 0.625
```

The score threshold is exactly `0.5 + 0.5 * 0.25`. It and the support margin
are two representations of one gate, not independent restrictions. Equality
activates within numeric tolerance `1e-12`.

The relative geometry requirement prevents appearance or motion alone from
causing reservation, while permitting overlap below the former absolute
`0.50` floor when hidden geometry is stronger than visible geometry.

An eligible pair with `delta_overlap <= -0.10` and upper support at most
`-0.25` supports retaining the visible competitor. Every other eligible pair
abstains as weak or ambiguous.

## Feasibility

Weights sum to one and every normalized difference lies in `[-1,1]`, so:

```text
relative_owner_support in [-1,1]
owner_preference bounds in [0,1]
core support in [-0.75,0.75]
```

With both optional channels missing, the lower support can still reach `0.50`:
maximum core `0.75` minus uncertainty `0.25`. The selected margin `0.25` is
strictly inside that conservative region.

Golden features are recomputed from explicit boxes, descriptors, motion
predictions, and ages. They include non-perfect activation, visible support,
ambiguity, role-swap complements, and uniform bbox scaling.

The selected gate was not derived from validation results or from optimizing a
tracking metric. H1-r1/H1-r2 evidence is used only to reject their impossible
absolute gates.

## Development and validation

The six H1-r1/H1-r2 episodes remain development and audit evidence. They are
not eligible for independent validation.

The four existing validation windows are copied byte-for-byte into the H1-r3
manifest. Their boundaries, source hashes, GT hashes, roles, and assignment
artifact hashes are unchanged. Role knowledge exists because assignments were
made before H1-r2 implementation. Outcome leakage does not exist because no
H1-r3 implementation or output exists and no validation window was executed.

No threshold, coefficient, feature, eligibility rule, or window may change
after validation begins. Validation requires separate authorization.

Before any reservation-enabled development run, a separately authorized
telemetry-only phase must keep assignments unchanged. Candidate-level owner
labels must be assigned from development GT, source, and parent evidence
without viewing H1-r3 scores. The predeclared false-reservation cost is four
times a missed reservation; abstention is preferred when the conservative
lower bound does not clear the frozen gate. This phase may test whether the
frozen operating region exists, but may not change its weights or threshold.
If no conservative development operating region exists, close the hidden-owner
preference hypothesis rather than tune it.

## Predeclared evaluation gates

Development must show:

- real activation in at least two independent positive episodes;
- association-output divergence;
- at least two beneficial identity outcomes;
- no new permanent or terminal swap;
- no broad control activation;
- no activation driven solely by missing evidence;
- causal framewise output with delay zero;
- prefix invariance and no future-frame access;
- identical detector evidence between paired arms;
- recursive run-root MP4 count zero.

The evaluation decision taxonomy separately records no eligible pairs, eligible
but never activated, activation without divergence, divergence without benefit,
beneficial effect, and harmful effect.

Runtime acceptance, validation, implementation, profile creation, evaluation,
and promotion all remain separately unauthorized.
