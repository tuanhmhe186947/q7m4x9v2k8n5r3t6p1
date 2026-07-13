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

- census every source-scope `Hidden=Yes`;
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

Implementation commit is `a242d5d`. Ruff and compileall passed. Eighteen
relevant tests passed. The representative short build used 64 rows per source,
selected 98 review items and resolved media for 98/98.

Full Hidden manifest generation and full human review are not complete. Follow
section 8A of
`docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`.
