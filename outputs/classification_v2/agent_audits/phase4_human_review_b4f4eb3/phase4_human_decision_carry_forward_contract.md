# Human-decision carry-forward contract

## Hidden decisions

Match only `review_key` and require identical source,
dataset, video, object-track, span, visual-media, crop/full-frame, review
schema, and decision schema authority.

## Behavior decisions

Match only `review_unit_key` and require identical actor,
temporal unit/span, original-label, visual-media, review-task, review schema,
and decision schema authority.

Forbidden matching: position, row_number, nearest_time, pig_id_only.

Possible classifications are:
- `CONFLICT`
- `EXACT_CARRY_FORWARD_CANDIDATE`
- `INVALID_DECISION_SCHEMA`
- `NEW_ONLY_REQUIRES_REVIEW`
- `OLD_ONLY_AUDIT_EVIDENCE`
- `REQUIRES_HUMAN_REVALIDATION`
