# Independent adversarial review

Review pass: `review.adversarial_pass_20260723_v1`

This is a falsification-oriented review. Passing tests does not imply
scientific correctness or approval of the current implementation.

## Method

- Reviewed the initially rendered package without editing it.
- Falsified references using missing masks and unresolved schema names.
- Exercised first-pair, temporal-gap, invalid-geometry and zero-denominator cases.
- Compared equal-distance candidates under row permutation and identity collisions.
- Compared axis and diagonal metrics on a non-square image.
- Injected missing, reordered and unexpected tensor features.
- Checked source symbols and two-way code-contract mappings.
- Searched generated prose for unsupported physical-distance claims.

## Findings

### review.contract_omission_masks: CONTRACT_OMISSION

The initial schema referenced valid_acceleration_pair and motion_feature_available without feature entities; both entities and a validator reference check were added after the read-only pass.

- Severity: `HIGH`
- Evidence: Initial model schema cross-reference inspection.
- Related gaps: gap.pair_coverage_missing, gap.vector_acceleration_absent
- Disposition: Contract omission corrected; implementation remains blocked.

### review.implementation_motion_schema: IMPLEMENTATION_VIOLATION

The exporter still selects only available columns and current native output lacks the full versioned 12-feature schema.

- Severity: `CRITICAL`
- Evidence: spatial_sequence_export._available_feature_names and current motion producer.
- Related gaps: gap.silent_motion_schema_shrink, gap.native_evidence_semantic_versions_missing
- Disposition: Blocking implementation remediation required.

### review.implementation_missingness: IMPLEMENTATION_VIOLATION

Invalid temporal pairs are numerically zero-filled and support coverage is not exported completely.

- Severity: `CRITICAL`
- Evidence: spatiotemporal._add_temporal_deltas and absent coverage columns.
- Related gaps: gap.invalid_pair_zero_semantics, gap.pair_coverage_missing
- Disposition: Blocking mask-aware producer and aggregate patch required.

### review.implementation_acceleration: IMPLEMENTATION_VIOLATION

d(speed)/dt is named as general acceleration and vector components/magnitude are absent; the constant-speed turn falsifies the current interpretation.

- Severity: `HIGH`
- Evidence: Golden constant-speed direction-change case and _add_temporal_deltas.
- Related gaps: gap.acceleration_name_semantics, gap.vector_acceleration_absent
- Disposition: Rename and add vector acceleration before rebuild.

### review.implementation_social_roi: IMPLEMENTATION_VIOLATION

Temporal partner continuity uses pig_id and target ROI contact uses all frames as denominator.

- Severity: `HIGH`
- Evidence: _add_social_context_columns and _add_temporal_unit_aggregates.
- Related gaps: gap.unstable_partner_continuity, gap.roi_denominator_all_frames
- Disposition: Stable partner key and ROI-availability denominator are blocking.

### review.test_omission_ties: TEST_OMISSION

The static social producer has a stable tie-break, but no direct equal-distance row-permutation regression was found.

- Severity: `MEDIUM`
- Evidence: Social tests cover invalid bbox and frame grouping, not equal-distance permutations.
- Related gaps: gap.equal_tie_permutation_test_missing
- Disposition: Add focused deterministic permutation tests.

### review.scientific_distance_limit: SCIENTIFIC_LIMITATION

Axis-normalized social distance and diagonal-normalized ROI distance are distinct image metrics; neither supports physical-distance language.

- Severity: `HIGH`
- Evidence: Non-square 200 by 100 golden case and absence of homography.
- Related gaps: gap.distance_metrics_unversioned, gap.homography_not_available
- Disposition: Version metrics, bind thresholds and retain conservative claim boundary.

### review.unresolved_tracking_quality: UNRESOLVED_AMBIGUITY

No authoritative independent tracking-quality failure predicate was located for the complete Classification V2 path.

- Severity: `MEDIUM`
- Evidence: Frame-local quality masks combine geometry but do not define a separate end-to-end tracking-quality contract.
- Related gaps: gap.tracking_quality_policy_unknown
- Disposition: Requires design review; do not silently infer policy.

### review.lineage_invalidation: AUDIT_OMISSION

Downstream invalidation is documented but not enforced by one dependency-aware semantic hash gate.

- Severity: `MEDIUM`
- Evidence: Current component gates do not traverse every formula/mask/schema dependency.
- Related gaps: gap.invalidation_not_machine_enforced
- Disposition: Add machine-enforced invalidation before any downstream reuse.

### review.baseline_social_helper_regression: IMPLEMENTATION_VIOLATION

Three existing social-context regressions fail at the starting SHA because the helper requires temporal_unit_key from frame-local fixtures.

- Severity: `MEDIUM`
- Evidence: Relevant Classification V2 regression: 123 passed and 3 failed with KeyError temporal_unit_key in _add_social_context_columns.
- Related gaps: gap.social_helper_temporal_unit_regression
- Disposition: Preserve as a baseline implementation gap; repair only in the later contract-driven remediation task.

## Reviewer conclusion

The contract package is suitable to drive a remediation patch, but current code is not scientifically approved. Passing contract-tool tests proves consistency of the declared audit package, not correctness of the feature implementation.
