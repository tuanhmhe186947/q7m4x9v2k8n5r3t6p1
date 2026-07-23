# Phase 3 worked golden cases

All expected numbers below are hand-derived independently of production
feature functions.

## Non-square image distance

For `W=1000`, `H=500`:

```text
horizontal 100 px:
  d_axis = sqrt((100/1000)^2 + 0^2) = 0.1
  d_diag = 100/sqrt(1000^2+500^2) = 0.08944271909999159

vertical 100 px:
  d_axis = sqrt(0^2 + (100/500)^2) = 0.2
  d_diag = 100/sqrt(1000^2+500^2) = 0.08944271909999159
```

Thus the axis metric differs by direction while the diagonal metric is
equal for equal pixel displacement. Neither value is physical distance.

## Equal-distance neighbor

At one frame, actor A is at normalized x `0.5`, B at `0.4`, and C at
`0.6`. Both axis distances are `0.1`. Stable keys are
`video-a|track=B` and `video-a|track=C`.

```text
minimum distance = 0.1
tie count = 2
lexicographically first canonical key = video-a|track=B
```

Orders `A,B,C` and `C,A,B` must therefore select the same B key.

## Partner continuity

For actor A:

```text
frame 0 partner = B
frame 1 partner = B
continuity valid = true
same partner = true
switch = false
```

If frame 1 partner is C, continuity remains valid, same partner is false
and switch is true. If frame 1 has no neighbor, continuity is false and
switch remains false.

## ROI denominator

Five observed frames; ROI is available on frames 1–3 and contact occurs
on frames 1–2:

```text
observed = 5
available numerator = 3
availability denominator = 5
availability ratio = 3/5 = 0.6
contact numerator = 2
contact denominator = 3
contact ratio = 2/3 = 0.6666666666666666
```

The rejected all-frame contact result is `2/5=0.4`.

## Zero ROI availability

For five observed frames with no available ROI:

```text
available_count = 0
contact_count = 0
target_roi_unit_available = false
contact_ratio placeholder = 0
```

The placeholder is unavailable evidence, not measured zero contact.

## Leakage fail-closed

Requests for `target_roi_contact`, `target_roi_distance`,
`target_roi_contact_ratio_unit` and
`label_selected_roi_class_indicator` must all raise before tensor
construction. `roi_feeder_contact` is the label-independent control and
remains allowed by the current contract.
