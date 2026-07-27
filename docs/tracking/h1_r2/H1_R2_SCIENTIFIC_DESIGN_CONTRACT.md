# H1-r2 Scientific Design Contract

Date: 2026-07-27

Status: design only. Implementation, evaluation, threshold search, profile
creation, and promotion are not authorized.

## Locked H1-r1 interpretation

- `H1_R1_DECISION=REJECTED_CONFIGURATION`
- `H1_R1_EXECUTION_VERIFIED=YES`
- `H1_R1_TELEMETRY_VERIFIED=YES`
- `H1_R1_CURRENT_POLICY_ACTIVATION=0`
- `H1_R1_QUALITY_EFFECT=NOT_MEASURED`
- `H1_R1_CURRENT_GATE_COMBINATION_FEASIBLE=NO`

The rejection applies to the heterogeneous raw-cost gate formulation, not to
the general concept of causal hidden-owner reservation. The six H1-r1 episodes
are development and audit evidence. They are ineligible for independent H1-r2
validation.

## Primary mathematical design

The primary quantity is `owner_preference_score`. It is bounded but is not
calibrated and must not be called a probability.

For each track candidate `t` and contested detection `d`, compute the same
feature vector and the same common-scale evidence score:

```text
Q(t,d) =
    0.250 * overlap_similarity
  + 0.200 * normalized_center_similarity
  + 0.100 * scale_similarity
  + 0.200 * appearance_similarity
  + 0.100 * motion_consistency
  + 0.100 * track_freshness
  + 0.025 * appearance_available
  + 0.025 * motion_available
```

All component qualities are in `[0,1]`; the nonnegative weights sum to one.
Missing appearance or motion receives neutral quality `0.5` plus availability
`0`, rather than a fabricated distance or positive signal.

For hidden owner `h` and visible competitor `v`:

```text
owner_preference_score =
    clip(0.5 + 0.5 * (Q(h,d) - Q(v,d)), 0, 1)
```

Identical evidence yields exactly `0.5`. Swapping hidden and visible arguments
yields the complement, apart from floating-point rounding. This construction
uses no sigmoid and makes no calibration claim.

Detection confidence is a shared candidate-eligibility gate because both
tracks contest the same detection; it cannot favor one track. Maximum stale
detection opportunities is also an eligibility rule applied identically.
Every exact feature formula, range, missingness rule, and cadence dependency is
frozen in `H1_R2_FEATURE_SEMANTICS_REGISTRY.csv`.

## Design alternatives

### Rule-based common-scale score (selected)

Benefits are boundedness, hand-verifiable behavior, symmetric definitions, no
fit on six reused episodes, and low runtime cost. Risks are domain-chosen
weights and lack of calibration. It needs no learned labels or probability
calibration. Missingness is explicit and symmetric. Overfitting risk is lower
than fitted alternatives, although a development-only activation threshold is
still required.

### Learned monotonic relative score

This would fit nonnegative coefficients for the same candidate features with a
pairwise logistic or ranking loss. It could adapt feature importance but needs
independent owner labels, grouped fitting, regularization, sign constraints,
fixed seeds, and untouched calibration data. Sparse contention labels create
high overfitting risk. Runtime is low after fitting. Missingness masks remain
explicit. It is deferred until adequate development labels exist.

### Pairwise preference model

This would consume `feature(hidden,d) - feature(visible,d)` for identically
defined features. Antisymmetry is natural and stronger interactions are
possible, but unconstrained nonlinear models can violate monotonicity and are
harder to interpret. It requires more pairwise labels, grouped validation,
regularization, and calibration before any probability terminology. Runtime
depends on model class. It is not selected for H1-r2.

## Monotonic and numerical requirements

- Greater hidden overlap cannot lower owner preference.
- Smaller hidden normalized center distance cannot lower owner preference.
- Stronger hidden appearance similarity cannot lower preference when valid.
- Longer hidden staleness cannot increase preference with other evidence fixed.
- Missing appearance is neutral quality with an availability loss, never
  strong positive evidence.
- Identical hidden and visible evidence produces neutral preference `0.5`.
- Stronger visible evidence reduces hidden-owner preference.
- The score remains finite and bounded in `[0,1]`.
- Uniform bbox scaling with unchanged normalized geometry preserves geometry
  features up to rounding.
- A feature unavailable to one side follows the same mask and fallback rule
  that would apply if it were unavailable to the other side.

## Coefficient and threshold policy

The eight coefficients above are fixed from domain reasoning before any H1-r2
implementation. They are not learned on the six H1-r1 episodes. The activation
threshold is not selected in this contract. Its selection rule is frozen:

1. annotate the predeclared development pool without viewing H1-r2 output;
2. choose one threshold using development data only;
3. require activation in at least two positive episodes from two video keys;
4. satisfy the frozen development control-activation limit;
5. lock the threshold, feature registry, coefficients, and code SHA;
6. inspect the validation output once, without retuning.

If coefficients are later learned, that is a different hypothesis revision.
It must predeclare owner labels, pairwise logistic loss, inverse-frequency
class balancing, L2 regularization, nonnegative evidence coefficients,
deterministic seeds, grouped development cross-validation by video/session,
and an untouched calibration split. Only independently calibrated output may
use probability terminology.

## Development and validation separation

Development contains the six H1-r1 audit episodes. All carry
`validation_eligible=false` and the exact reason
`used_for_h1_r1_design_and_cost_audit`.

Validation windows are predeclared on different video keys and on recording
sessions 2019-11-28 and 2019-11-30, while development is session 2019-11-29.
Their positive/control roles remain blinded and must be assigned from GT and
parent evidence before implementation. No H1-r2 output may be inspected during
that assignment. The frozen window rows may not be replaced after output is
seen.

## Golden-case authority

`H1_R2_GOLDEN_CASES.yaml` contains ten arithmetic examples computed from the
formula above without importing production association code. An eventual
implementation must reproduce them within `1e-9`, plus complement and uniform
scale-invariance properties.

## Evaluation contract

`H1_R2_EVALUATION_GATE.json` is frozen before implementation. It requires
paired RF_ACC23 baseline and candidate runs using identical verified detector
evidence, `causal_framewise`, output delay zero, prefix invariance, no future
frames, recursive run-root MP4 count zero, no permanent or terminal swaps, and
no worsening of the locked `000302` guardrail. Runtime overhead is bounded but
cannot be measured in this task.

No tracking experiment, GPU inference, complete video, Hard6, full-13, runtime
benchmark, or threshold search was run to create this contract.

## Remaining authorization blockers

- Validation positive/control roles require blinded GT/parent annotation.
- The development-only activation threshold has not been selected.
- No independent reviewer has signed the design packet.

Therefore `implementation_authorized=false`,
`evaluation_authorized=false`, and `promotion_authorized=false`.
