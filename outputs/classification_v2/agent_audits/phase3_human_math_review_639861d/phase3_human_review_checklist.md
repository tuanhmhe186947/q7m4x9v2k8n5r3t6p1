# Phase 3 human sign-off checklist

Reviewer: ____________________

Review date: ____________________

Decision: `ACCEPT_PHASE_3` / `REJECT_PHASE_3`

- [ ] Recompute the non-square `1000 x 500` horizontal and vertical
  distance examples.
- [ ] Confirm the axis metric is anisotropic in pixel space and the
  diagonal metric is isotropic for equal pixel displacement.
- [ ] Confirm neither metric is described as physical/world distance.
- [ ] Confirm threshold `0.08` remains bound to the axis metric and was
  not silently transferred or recalibrated.
- [ ] Inspect both equal-distance row orders and confirm partner B is
  selected with tie count two.
- [ ] Confirm canonical hierarchy is `object_track_key`, scoped
  `track_id`, then scoped `object_id`; `pig_id` is metadata only.
- [ ] Confirm blank, duplicate and cross-video `pig_id` cases remain
  identity-safe.
- [ ] Confirm continuity uses `nearest_partner_key` and resets on
  temporal-unit or actor identity change.
- [ ] Confirm no-neighbor is unavailable continuity, not partner switch.
- [ ] Recompute the ROI `2/3` contact result and reject `2/5`.
- [ ] Confirm zero ROI availability has
  `target_roi_unit_available=false`.
- [ ] Confirm four label-selected ROI requests fail model export while
  `roi_feeder_contact` remains allowed.
- [ ] Run `phase3_independent_reference_verifier.py` and require exit
  code zero with `172` checks and no errors.
- [ ] Inspect the bounded 10 CVAT + 10 legacy trace and independent
  checker payload.
- [ ] Confirm Phase 1 pair validity and Phase 2 motion schema/hash remain
  unchanged.
- [ ] Confirm Phase 4 invalidation/release gaps remain open and no
  official artifact was regenerated.

Automated PASS does not constitute Phase 3 acceptance. `READY_FOR_PHASE4`
must remain `NO` until this checklist records `ACCEPT_PHASE_3`.
