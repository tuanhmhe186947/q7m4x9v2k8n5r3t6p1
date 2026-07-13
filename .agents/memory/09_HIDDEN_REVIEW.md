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

Current workload-policy implementation commit is `5212a59`. The complete
classification regression passed with 150 tests. The balanced 64-row/source
smoke selected 41 review items and passed the independent coverage checker.

The versioned v5 full manifest contains 5,171 unique review items: 4,121 Yes
confirmations, 211 high-risk No, 647 random No, and 192 clean controls. It has
4,649 CVAT and 522 legacy items, zero missing untrusted Yes, zero trusted quota
mismatches, and zero high-risk cap violations. The evidence is under
`outputs/classification_v2/rebuilds/hidden_review_v5_full_20260713`.

Full human review and decision apply are not complete. Follow section 8A of
`docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`.
