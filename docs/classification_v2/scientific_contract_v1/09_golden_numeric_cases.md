# Golden numerical cases

Generated from the machine-readable golden cases in the primary
contract. Expected values are recomputed by the independent validator.

## case.stationary_actor

A valid stationary pair is measured zero, distinct from missing motion.

- Pair masks: `{"valid_motion_pair": [false, true]}`
- Expected numeric: `{"speed_n_per_second": [null, 0.0]}`
- Expected aggregate: `{"speed_mean_valid_pairs": 0.0}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.invalid_pairs_excluded_from_aggregates`

## case.constant_horizontal_velocity

Constant x velocity has zero vector and tangential acceleration.

- Pair masks: `{"valid_motion_pair": [false, true, true]}`
- Expected numeric: `{"vx_n_per_second": [null, 0.1, 0.1], "vy_n_per_second": [null, 0.0, 0.0], "tangential_acceleration_n_per_second2": [null, null, 0.0], "acceleration_vector_magnitude_n_per_second2": [null, null, 0.0]}`
- Expected aggregate: `{"speed_mean_valid_pairs": 0.1}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.invalid_pairs_excluded_from_aggregates`

## case.constant_diagonal_velocity

Diagonal image velocity uses the exact axis-normalized Euclidean norm.

- Pair masks: `{"valid_motion_pair": [false, true]}`
- Expected numeric: `{"vx_n_per_second": [null, 0.3], "vy_n_per_second": [null, 0.4], "speed_n_per_second": [null, 0.5]}`
- Expected aggregate: `{"speed_mean_valid_pairs": 0.5}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.invalid_pairs_excluded_from_aggregates`

## case.speed_change_same_direction

Tangential acceleration is nonzero while direction change is zero.

- Pair masks: `{"valid_motion_pair": [false, true, true]}`
- Expected numeric: `{"direction_change_rad": [null, null, 0.0], "tangential_acceleration_n_per_second2": [null, null, 0.1], "ax_n_per_second2": [null, null, 0.1], "ay_n_per_second2": [null, null, 0.0]}`
- Expected aggregate: `{}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.invalid_pairs_excluded_from_aggregates`

## case.direction_change_constant_speed

Constant scalar speed with a turn has zero tangential but nonzero vector acceleration.

- Pair masks: `{"valid_motion_pair": [false, true, true]}`
- Expected numeric: `{"direction_change_rad": [null, null, 1.5707963267948966], "tangential_acceleration_n_per_second2": [null, null, 0.0], "acceleration_vector_magnitude_n_per_second2": [null, null, 0.14142135623730953]}`
- Expected aggregate: `{}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.invalid_pairs_excluded_from_aggregates`

## case.first_frame_unit

First frame has no temporal pair and is excluded from pair aggregates.

- Pair masks: `{"valid_motion_pair": [false]}`
- Expected numeric: `{}`
- Expected aggregate: `{"speed_mean_valid_pairs": 0.0}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.pair_reset_at_temporal_unit_key`, `invariant.invalid_pairs_excluded_from_aggregates`

## case.cross_temporal_unit_boundary

A sorted adjacent row in another temporal unit cannot form a pair.

- Pair masks: `{"valid_motion_pair": [false, false]}`
- Expected numeric: `{"speed_n_per_second": [null, null]}`
- Expected aggregate: `{"speed_mean_valid_pairs": 0.0}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.no_cross_temporal_unit_pair`, `invariant.pair_reset_at_temporal_unit_key`

## case.missing_middle_frame

A frame gap is sparse velocity support, not a contiguous path pair.

- Pair masks: `{"valid_motion_pair": [false, true], "adjacent_motion_pair_valid": [false, false], "sparse_velocity_pair_valid": [false, true]}`
- Expected numeric: `{"vx_n_per_second": [null, 0.1]}`
- Expected aggregate: `{}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.invalid_pairs_excluded_from_aggregates`

## case.nonpositive_delta_t

Zero elapsed time invalidates every rate.

- Pair masks: `{"valid_motion_pair": [false, false]}`
- Expected numeric: `{"vx_n_per_second": [null, null]}`
- Expected aggregate: `{"speed_mean_valid_pairs": 0.0}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.monotonic_timestamp_within_actor_unit`

## case.invalid_previous_bbox

Invalid previous geometry invalidates the pair.

- Pair masks: `{"valid_motion_pair": [false, false]}`
- Expected numeric: `{}`
- Expected aggregate: `{"speed_mean_valid_pairs": 0.0}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.invalid_pairs_excluded_from_aggregates`

## case.invalid_current_bbox

Invalid current geometry invalidates the pair.

- Pair masks: `{"valid_motion_pair": [false, false]}`
- Expected numeric: `{}`
- Expected aggregate: `{"speed_mean_valid_pairs": 0.0}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.invalid_pairs_excluded_from_aggregates`

## case.all_pairs_invalid

All-invalid motion emits zero placeholders only with availability false.

- Pair masks: `{"valid_motion_pair": [false, false, false]}`
- Expected numeric: `{}`
- Expected aggregate: `{"speed_mean_valid_pairs": 0.0}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.invalid_pairs_excluded_from_aggregates`, `invariant.valid_pair_coverage_exported`

## case.exactly_one_valid_pair

One valid pair is the complete aggregate denominator.

- Pair masks: `{"valid_motion_pair": [false, false, true]}`
- Expected numeric: `{"speed_n_per_second": [null, null, 0.19999999999999998]}`
- Expected aggregate: `{"speed_mean_valid_pairs": 0.19999999999999998}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.invalid_pairs_excluded_from_aggregates`, `invariant.valid_pair_coverage_exported`

## case.equal_distance_neighbor_tie

Equal distances resolve by canonical stable identity.

- Pair masks: `{}`
- Expected numeric: `{"distance": 0.1}`
- Expected aggregate: `{}`
- Selected neighbor: `track|a`
- Tolerance: `1e-09`
- Invariants: `invariant.deterministic_social_tie_break`

## case.row_permutation_neighbor_tie

Permuting tied candidate rows does not change the selected neighbor.

- Pair masks: `{}`
- Expected numeric: `{"distance": 0.1}`
- Expected aggregate: `{}`
- Selected neighbor: `track|a`
- Tolerance: `1e-09`
- Invariants: `invariant.deterministic_social_tie_break`, `invariant.row_order_invariance`

## case.blank_pig_id_stable_track

Blank pig_id does not break partner continuity when stable track identity exists.

- Pair masks: `{"social_pair_available": [false, true]}`
- Expected numeric: `{"partner_continuity": 1.0}`
- Expected aggregate: `{"valid_social_pair_count": 1}`
- Selected neighbor: `track|a`
- Tolerance: `1e-09`
- Invariants: `invariant.stable_partner_identity`

## case.duplicate_pig_id_tracks

Duplicate pig_id values do not merge distinct stable tracks.

- Pair masks: `{"same_partner": false}`
- Expected numeric: `{"partner_continuity": 0.0}`
- Expected aggregate: `{}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.stable_partner_identity`

## case.roi_three_of_five_available

ROI contact denominator contains only three available frames.

- Pair masks: `{}`
- Expected numeric: `{"target_roi_availability_ratio_unit": 0.6}`
- Expected aggregate: `{"target_roi_contact_ratio_unit": 0.6666666666666666}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.roi_availability_denominator`

## case.roi_none_available

Zero ROI availability yields placeholder zero and availability false.

- Pair masks: `{}`
- Expected numeric: `{"target_roi_availability_ratio_unit": 0.0}`
- Expected aggregate: `{"target_roi_contact_ratio_unit": 0.0}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.roi_availability_denominator`

## case.non_square_image_distance

Axis-normalized and diagonal-normalized metrics differ on a non-square image.

- Pair masks: `{"distance_available": true}`
- Expected numeric: `{"axis_normalized": 1.118033988749895, "diagonal_normalized": 0.6324555320336759}`
- Expected aggregate: `{}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.distance_metric_version_binding`

## case.padding_vs_observed_zero

A padded zero and a measured stationary zero remain distinguishable by masks.

- Pair masks: `{"observed_mask": [1, 0], "length_mask": [1, 0]}`
- Expected numeric: `{"values": [0.0, 0.0]}`
- Expected aggregate: `{"observed_count": 1}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.invalid_pairs_excluded_from_aggregates`

## case.review_only_roi_excluded

Label-selected target ROI fields cannot enter model schema.

- Pair masks: `{}`
- Expected numeric: `{"selected_count": 0}`
- Expected aggregate: `{}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.review_only_not_in_model_tensor`, `invariant.target_roi_leakage_guard`

## case.missing_required_exporter_feature

Removing one required motion feature fails before tensor export.

- Pair masks: `{}`
- Expected numeric: `{"expected_dimension": 12, "actual_dimension": 11}`
- Expected aggregate: `{}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.producer_exporter_ordered_schema_equal`, `invariant.fixed_tensor_dimension`, `invariant.no_silent_column_shrinkage`

## case.reordered_exporter_feature

A reordered feature list fails even when dimension is unchanged.

- Pair masks: `{}`
- Expected numeric: `{"expected_dimension": 12, "actual_dimension": 12}`
- Expected aggregate: `{}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.producer_exporter_ordered_schema_equal`

## case.unexpected_extra_model_feature

An undeclared extra feature is a fail-closed schema violation.

- Pair masks: `{}`
- Expected numeric: `{"expected_dimension": 12, "actual_dimension": 13}`
- Expected aggregate: `{}`
- Selected neighbor: ``
- Tolerance: `1e-09`
- Invariants: `invariant.producer_exporter_ordered_schema_equal`
