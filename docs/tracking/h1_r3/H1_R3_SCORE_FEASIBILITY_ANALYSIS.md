# H1-r3 Score Feasibility Analysis

This analysis is independent of production scoring code. It uses only the
frozen feature ranges and arithmetic in the H1-r3 design contract.

## Contribution bounds

| Contribution | Input difference | Weight | Support range |
|---|---:|---:|---:|
| overlap | `[-1,1]` | `0.60` | `[-0.60,0.60]` |
| freshness | `[-1,1]` | `0.15` | `[-0.15,0.15]` |
| appearance | `[-1,1]` | `0.15` | `[-0.15,0.15]` |
| motion | `[-1,1]` | `0.10` | `[-0.10,0.10]` |
| normalized center | `[-1,1]` | `0` | diagnostic only |
| scale | `[-1,1]` | `0` | diagnostic only |

Optional pair quality and masks lie in `[0,1]`; therefore they can shrink but
cannot expand the optional ranges.

Summing intervals gives:

```text
core support in [-0.75,0.75]
relative owner support in [-1,1]
owner preference bounds in [0,1]
```

## Missingness combinations

For appearance and motion independently, the four mask combinations are:

| Hidden mask | Visible mask | Contribution interval |
|---:|---:|---:|
| 0 | 0 | full channel range |
| 0 | 1 | full channel range |
| 1 | 0 | full channel range |
| 1 | 1 | exact quality-weighted difference |

Appearance uses `[-0.15,+0.15]`; motion uses `[-0.10,+0.10]`. Activation uses
the total lower bound. Replacing an observed contribution by its full interval
cannot raise that lower bound. The Cartesian product covers all 16 masks.

## Non-empty regions

### Hidden activation witness

Use detection box `[0,0,100,100]`, hidden box `[5,0,105,100]`,
and visible box `[120,0,220,100]`. Both ages are zero and both optional
channels are missing.

```text
IoU_hidden = 9500/10500 = 0.9047619048
IoU_visible = 0
core = 0.60 * 0.9047619048 = 0.5428571429
lower support = 0.5428571429 - 0.15 - 0.10 = 0.2928571429
owner preference lower bound = 0.6464285714
```

The hidden box is not a perfect match, yet its worst-case lower bound exceeds
`0.25` and its IoU advantage exceeds `0.10`.

### Visible-support witness

Swap the two candidate states. The support interval becomes
`[-upper,-lower]`; the visible-support region is non-empty.

### Ambiguous witness

Use identical finite boxes and ages with both optional channels observed and
identical. The support interval is `[0,0]`; the pair abstains.

## Boundary compatibility

The selected score threshold is exactly:

```text
0.5 + 0.5 * 0.25 = 0.625
```

There is no second independent margin. The lower-bound score and support
comparisons are equivalent.

For a same-size hidden box shifted by `100/11` pixels, IoU is exactly `5/6`.
Against a zero-IoU visible box with both optional channels missing:

```text
lower support = 0.60 * (5/6) - 0.25 = 0.25
```

It activates at equality. A `9.2`-pixel shift has lower support below `0.25`
and abstains. Both coordinates are jointly realizable.

## Feasible ranges

- Support margins in `[0,0.75]` permit some mathematical activation.
- With both optional channels missing, margins in `[0,0.50]` remain feasible.
- Equivalent lower-bound score thresholds are `[0.5,0.875]` generally and
  `[0.5,0.75]` with both optional channels missing.
- Relative IoU minima in `[0,1]` are mathematically attainable.

The frozen support margin `0.25`, score threshold `0.625`, and relative IoU
minimum `0.10` are interior values. They reserve half of the conservative
missing-optional operating span for abstention.

## Failure conditions checked

The deterministic checker fails if:

- the maximum score cannot exceed the threshold;
- score threshold and support margin cease to be equivalent;
- the relative IoU minimum exceeds its bound;
- hidden and visible availability rules differ;
- reference-box or LK provenance differs by side;
- realistic non-perfect activation witnesses disappear;
- masking observed evidence increases the hidden-owner lower bound;
- raw golden boxes do not reproduce the declared features;
- development and validation overlap;
- validation outputs appear;
- H1-r2 is reopened or H1-r3 implementation is authorized.
