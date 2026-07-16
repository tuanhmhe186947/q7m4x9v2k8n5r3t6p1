# Classification V2 Hidden Review Memory

## Active decision

Hidden review is required after enhanced frame features and before temporal
harmonization. Hidden is a frame/object visibility attribute, not a behavior
target and not a native-unit decision.

CVAT Hidden values are tracking-derived and untrusted until current human
review. Legacy values retain prior-review provenance unless a current reviewer
marks the item unclear.

## Review design

Use four disjoint cohorts:

- census every untrusted `Hidden=Yes` and stratified-audit trusted Yes;
- label-independent high-risk `Hidden=No` enrichment;
- stratified-random `Hidden=No` for false-negative estimation;
- low-risk clean negative controls.

Random rows must store stratum population, inclusion probability and inverse
sampling weight. Report the post-stratified random false-negative estimate
separately from high-risk correction yield.

## Operator contract

Use `review_hidden_quality_gui.py`. It shows full-frame bbox context and a
letterboxed actor crop, then writes decisions only. Do not use the old
direct-source GUI for a new lineage because it can write corrected XML/CSV.

Default coverage rejects missing, duplicate, pending and unclear decisions.
Apply writes `hidden_reviewed_frame_features.csv`, preserves row count and
changes only Hidden plus declared provenance fields.

Unselected CVAT No remains `untrusted_tracking_derived`. Do not silently promote
it to visible trusted metadata. High trusted-Hidden ratio is audited and does
not automatically exclude or down-weight a sequence window.

## Evidence and status

The existing technical reference is
`outputs/classification_v2/rebuilds/hidden_review_v6_full_20260714`. Its 5,131
items contain 4,122 Yes confirmations, 384 high-risk No, 601 random No, and 24
clean controls. Behavior is descriptive metadata only; target-derived risk and
sampling fields are absent.

The manifest SHA256 is
`3e4fec14c466a89370a1e20d913cb024bd1dda1fa8db9c1fabdf8a51fa31072e`.
The predeclared design has adequate planned random/high-risk clustered support.
Commits `32eaa2b` and `aaf8460` preserve 30 old payload rows through identifier
v2 and bind migration artifacts technically. The user confirms no review has
started, so those rows have unverified provenance and are not human evidence.

Commit `f2179e3` upgrades media evidence to schema v2 with exact manifest and
frame-context hashes. The 24-item dual-source smoke resolves 12 video and 12
crop items; the full gate resolves 4,613 video and 518 crop items. Both report
zero missing or unknown-source media.

Verified human coverage is 0/5,131. The clean authority must be rebuilt under
`outputs/classification_v2/human_review_runs/<RUN_ID>` with zero carried rows.
The scientific gate is `BLOCKED_INCOMPLETE_OR_INSUFFICIENT_REVIEW`; apply,
temporal rebuild, snapshot, and training remain blocked. Follow section 8A of
`docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`.
