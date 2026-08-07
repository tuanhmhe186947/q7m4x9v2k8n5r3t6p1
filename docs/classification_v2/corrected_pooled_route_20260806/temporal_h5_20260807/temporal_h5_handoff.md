# Current H5 Authority Closure

The historical `COMMON_H5_T6_R` contract is retained and rebound to the current
reviewed authority. It is not a new temporal design.

- Legacy H5 is offsets `0..4`; the frozen T6 target is offsets `5..10`.
- Current matched cohort: 33,166 targets; CVAT 28,628/28,724 and legacy
  4,538/4,538 pass H5 validity.
- Causality audit: zero future-frame, split, video, and actor-scope violations.
- The current 46D H5 bundle is external managed output; its manifest is bound
  to the reviewed snapshot, split, cohort, temporal contract, and schema.
- Loader and local two-source forward/loss/backward smoke pass. This is only an
  engineering contract check, not H5 model evidence.

The future causal-history comparison is `T6` versus `T6+H5` on the identical
`COMMON_H5_MATCHED_COHORT`. H6, H12, and H24 remain deferred. E0 remains T6
only, on `FOLD_3`, with no history or posture input.
