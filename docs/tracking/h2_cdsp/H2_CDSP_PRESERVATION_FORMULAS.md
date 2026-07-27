# H2-CDSP preservation formulas

All formulas are design-only and operate on current/past state. They do not
assign or reserve a detection.

## Constants

| Symbol | Value | Interpretation |
|---|---:|---|
| `H_c` | 6 frames | state-confidence half-life |
| `H_a` | 8 frames | appearance-reliability half-life |
| `H_m` | 4 frames | motion-reliability half-life |
| `g_0` | 0.05/frame | base uncertainty growth |
| `g_m` | 0.10/frame | weak/missing-motion growth |
| `b` | 0.15 | persistent boundary-crossing uncertainty penalty |
| `A_max` | 10 frames | finite maximum preservation age |
| `C_min` | 0.30 | minimum usable state confidence |
| `U_max` | 0.75 | maximum usable normalized uncertainty |
| `v_max` | 0.25 diagonals/frame | normalized center-velocity cap |
| `s_max` | `log(1.25)`/frame | absolute log-scale-rate cap |

Every numeric input must be finite. `clip(z,l,h)=min(max(z,l),h)`.

## 1. State confidence

For integer dropout age `a >= 0` and trusted initial confidence
`C_0 in [0,1]`:

```text
C(a) = clip(C_0 * 2^(-a / H_c), 0, 1)
```

A trusted match sets `C_0=1`. Without a new trusted match, `C(a+1) <= C(a)`.
Missing optional evidence does not enter this formula and cannot raise it.
Non-finite `C_0` invalidates the state.

## 2. Optional reliability

For availability indicator `I in {0,1}` and finite initial quality
`q_0 in [0,1]`:

```text
R_a(a) = clip(I_a * q_a0 * 2^(-a / H_a), 0, 1)
R_m(a) = clip(I_m * q_m0 * 2^(-a / H_m), 0, 1)
```

When appearance or motion is missing, its indicator is zero and reliability is
exactly zero. Missing evidence never receives a neutral positive constant.

Appearance is never refreshed from an unassigned detection. Motion reliability
is zero after LK failure, malformed velocity, a noncausal timestamp, or a
frame-continuity break.

## 3. Uncertainty

Initial normalized uncertainty is exactly `U_0=0.10` after a trusted match.
Let `B_seen(a)` equal `0.15` when any propagated box at ages `1..a` has crossed
an image boundary and `0` otherwise:

```text
U(a) = clip(
    U_0 + a * (g_0 + g_m * (1 - R_m(a))) + B_seen(a),
    0,
    1
)
```

`B_seen(a)` is persistent and non-decreasing. For fixed causal evidence,
`U(a+1) >= U(a)`. Motion reliability decays with age, so it cannot make later
uncertainty smaller. A new trusted match resets `U_0` to `0.10`; no other
transition may reduce uncertainty.

## 4. Causal geometry propagation

For trusted bbox center `c_0`, width `w_0`, height `h_0`, and diagonal
`d_0=sqrt(w_0^2+h_0^2)`:

```text
v_n = clip_norm((c_0 - c_previous) / d_previous, v_max)
r_w = clip(log(w_0 / w_previous), -s_max, s_max)
r_h = clip(log(h_0 / h_previous), -s_max, s_max)

c(a) = c_0 + a * v_n * d_0
w(a) = w_0 * exp(a * r_w)
h(a) = h_0 * exp(a * r_h)
```

`clip_norm` preserves direction and caps Euclidean magnitude. If motion is
unavailable, `v_n=r_w=r_h=0`; geometry is held and uncertainty grows under
`R_m(a)=0`.

Dividing displacement by bbox diagonal and using log scale ratios makes
equivalent small and large boxes produce identical normalized motion.

## 5. Re-entry usability

Preserved state is usable as bounded ordinary-association evidence exactly
when:

```text
usable(a) =
    state in {DROPOUT_GRACE, OCCLUSION_PRESERVED, STALE_PRESERVED}
    and core_state_valid
    and causal_frame_continuity
    and 0 <= a <= A_max
    and C(a) >= C_min
    and U(a) <= U_max
```

`usable=true` exposes only a `PreservedStateEvidence` diagnostic record with
the propagated reference, confidence, uncertainty, and optional reliabilities.
The only permitted future consumer is a one-for-one substitution for an
absent or baseline-degraded track-local bbox, motion, or appearance reference.
It uses the unchanged baseline candidate set, costs, weights, gates, and
solver. It adds no owner score, penalty, bonus, reservation, veto, candidate,
direct assignment, emission, creation, or refresh.

## 6. Invalidation

The state becomes `INVALIDATED` when any condition holds:

- age is greater than `A_max`;
- confidence is below `C_min`;
- uncertainty exceeds `U_max`;
- geometry, confidence, uncertainty, frame index, or required history is
  missing, malformed, NaN, or infinite;
- frame continuity is broken or frame index moves backward;
- video key changes;
- bbox width/height is non-positive;
- source state is already `INVALIDATED` or `TERMINATED`.

Removal from the active baseline track dictionary routes to `TERMINATED`, not
`INVALIDATED`. `TERMINATED` is absorbing. An invalidated but still-live
baseline track can return to `VISIBLE_CONFIRMED` only through a new ordinary
trusted match that does not use H2 evidence; this trusted-match exception is
evaluated before the otherwise fail-closed invalidated-source rule.

## 7. Feasibility proof

Take realistic non-perfect inputs `C_0=1`, `U_0=0.10`,
`I_m=1`, and `q_m0=0.8`.

At age 4:

```text
C(4) = 2^(-4/6) = 0.6299605
R_m(4) = 0.8 * 2^(-4/4) = 0.4
U(4) = 0.10 + 4 * (0.05 + 0.10 * 0.60) = 0.54
```

The state is usable without perfect evidence. With both optional channels
missing at age 3:

```text
C(3) = 0.7071068
U(3) = 0.10 + 3 * 0.15 = 0.55
```

It remains usable briefly. At age 5 with motion missing, `U(5)=0.85`, so it is
unusable even before the age cap. At age 11, every state is invalid regardless
of optional quality. Thus usable short-dropout and unusable long/uncertain
regions are both non-empty.

The proof does not require perfect appearance, perfect motion, or a perfect
bbox match. Missing appearance never strengthens state. Uniform bbox scaling
leaves `v_n`, `r_w`, and `r_h` unchanged.
