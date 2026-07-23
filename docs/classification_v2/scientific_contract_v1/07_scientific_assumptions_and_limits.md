# Scientific assumptions and limits

Generated from `00_pipeline_contract.yaml`.

## assumption.image_coordinate_not_physical_distance

Axis-normalized and diagonal-normalized distances are image-coordinate metrics only.

- Status: `IMPLEMENTATION_DIFFERS_FROM_CONTRACT`
- Defensibility: Exact formulas are reproducible for a fixed camera and image geometry.
- Limit: Perspective and depth make equal image changes unequal world-plane motion.
- Required evidence: Homography/calibration and validation are required for physical-distance claims.

## assumption.fixed_camera_roi_context

Static ROI annotations are camera-specific scene context.

- Status: `IMPLEMENTED_PARTIALLY_TESTED`
- Defensibility: Reasonable only when image registration and camera geometry are unchanged.
- Limit: Does not transfer automatically to another camera or pen.
- Required evidence: Per-camera registration/hash and recalibration.

## assumption.homography_future_modality

No calibrated homography/world-plane distance is active in this contract.

- Status: `DECLARED_NOT_IMPLEMENTED`
- Defensibility: Separating absent calibration prevents false physical claims.
- Limit: World distances, metres and metres/second are unavailable.
- Required evidence: Calibration targets, homography uncertainty and external distance validation.

## assumption.pig_id_annotation_local

pig_id is annotation-local metadata and never sole trajectory identity.

- Status: `IMPLEMENTATION_DIFFERS_FROM_CONTRACT`
- Defensibility: Stable source/video/object keys are available.
- Limit: No cross-video biological identity claim.
- Required evidence: Independent identity system for cross-recording animals.

## assumption.decoded_frame_clock

Active source motion clock is source_frame_index divided by verified 30 FPS.

- Status: `IMPLEMENTED_AND_TESTED`
- Defensibility: All active videos were audited as 30 FPS, 1800 frames and 60 seconds.
- Limit: Future sources require a new FPS/time authority rather than source-type inference.
- Required evidence: Container and decoded-frame timing audit.

## assumption.review_not_ground_truth_perfection

Human review decisions are evidence with coverage/provenance, not proof of biological truth.

- Status: `UNKNOWN_REQUIRES_REVIEW`
- Defensibility: Coverage and uncertainty can be audited.
- Limit: Single-reviewer disagreement and annotation ambiguity remain.
- Required evidence: Agreement study or explicit single-review limitation.

## assumption.threshold_metric_binding

Every threshold remains bound to the metric/version on which it was calibrated.

- Status: `IMPLEMENTATION_DIFFERS_FROM_CONTRACT`
- Defensibility: Prevents silent semantic changes.
- Limit: Changing metric requires recalibration, not numeric threshold reuse.
- Required evidence: Versioned calibration report.

## assumption.grouped_generalization_scope

Claims are limited to recording-date/video-safe internal evaluation.

- Status: `IMPLEMENTED_AND_TESTED`
- Defensibility: Grouped splits reduce within-recording leakage.
- Limit: No external farm/camera/cohort generalization claim.
- Required evidence: External validation dataset.
