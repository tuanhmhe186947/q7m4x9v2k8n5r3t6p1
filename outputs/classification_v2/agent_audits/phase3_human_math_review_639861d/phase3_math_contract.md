# Phase 3 mathematical contract

Implementation authority: `639861d9d41112a5bfdddbb96c1ce15471e07acb`.
This bounded package is review evidence, not an official data rebuild.

## Distance

For image width `W`, height `H` and center displacement `(dx, dy)` in
pixels:

```text
d_axis = sqrt((dx/W)^2 + (dy/H)^2)
d_diag = sqrt(dx^2 + dy^2) / sqrt(W^2 + H^2)
```

`d_axis` is anisotropic in pixel space on non-square images. `d_diag` is
isotropic for equal pixel displacements. Both are dimensionless
image-coordinate metrics. Neither is physical or world-plane distance.
Distance is available only with valid actor and partner geometry, positive
finite image dimensions and a distinct canonical identity.

The existing social-near threshold `0.08` remains bound to
`image_axis_normalized_distance` version
`classification_v2.image_axis_normalized_distance.v1`. Phase 3 did not
transfer or recalibrate this threshold.

## Social identity and tie-break

Canonical identity hierarchy:

1. `object_track_key`;
2. source/dataset/video-scoped `track_id`;
3. source/dataset/video-scoped `object_id`.

`pig_id` is metadata only. Nearest selection orders candidates by axis
distance ascending and then canonical partner key ascending. Raw row
position is not a tie-break.

Partner continuity is valid only when current and previous social
observations both have a neighbor inside the same `temporal_unit_key` and
canonical actor trajectory. It compares `nearest_partner_key`. Missing
neighbor makes continuity unavailable and does not represent a switch.

## ROI aggregation

For one temporal unit:

```text
available_count = sum(observed AND ROI_available AND geometry_valid)
contact_count = sum(observed AND ROI_available AND geometry_valid AND contact)
availability_ratio = available_count / observed_count
contact_ratio = contact_count / available_count
```

When `available_count=0`, `target_roi_unit_available=false`. The numerical
contact-ratio placeholder is `0`, but the false availability mask forbids
interpreting it as measured no contact.

Behavior-selected `target_roi_*` and `roi_target_*` fields are review
evidence only. They fail closed if requested as model input. Explicit
all-class fields such as `roi_feeder_contact` remain eligible only under
the existing label-independent whitelist.

## Frozen boundaries

Phase 1 temporal-pair validity and Phase 2 motion schema hash
`ec0c511b5f5198240492be49c0492e543c9e38eb4a4ff446259b958c2a59963b`
remain unchanged. Phase 4 semantic invalidation and release authority are
not implemented here.
