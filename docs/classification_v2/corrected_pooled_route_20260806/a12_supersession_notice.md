# A12 supersession notice

Status: `SUPERSEDES_STALE_GATE_DESIGN`

Replacement: `permit_policy.json`, `experiment_funnel.json`, and
`readiness_decision.json` in this directory.

## Removed from the hard gate

The old source-balanced gain question is not a direct leakage test. The
following requirements are removed from A12 authorization:

- mandatory Macro-F1 gain `+0.02`;
- mandatory target-recall gain `+0.03`;
- both provenances in every held-out fold;
- the 75-unit common-support cohort as the sole gate;
- legacy-versus-additional-source winner selection; and
- equal gain across sources.

The old approximately `+0.038` result is evidence about the superseded gate
design. The current approximately `+0.005264` Macro-F1 result and
approximately `-0.2917` target-recall result are also diagnostic evidence
about that design. Neither is a behavior-model claim.

## A12-A — direct source-leakage safety (hard gate)

PASS requires rejection of source type/name, dataset ID, filename/path,
reviewer/decision metadata, partition fields, source-specific cache/padding,
stale schemas, target-derived review fields, future frames outside the
registered causal view, and neighboring units crossing split boundaries.
Legitimate recording-date, scene, occlusion, behavior-support, and difficulty
variation is not itself leakage.

Current evidence: the representation probe reports
`source_type_entered_representation_input=false`, the forbidden-field audit is
empty, and grouped/future-frame controls pass. Therefore A12-A is `PASS` for
the current authority, subject to the validator binding those artifacts.

## A12-B — overlap and grouping integrity (hard gate)

PASS requires current manifest/audit evidence for construction-time exclusion
of intended overlapping dates/videos, exact and near duplicates, frame
intervals, native units, tracks, videos, recording-date groups, and inherited
model-window roles. Current grouped split purity and duplicate/window audits
pass. A current machine-readable artifact explicitly proving the user-confirmed
construction rule that additional dates/videos were excluded from the legacy
collection was not located during the bounded authority search.

Therefore A12-B is `INCONCLUSIVE`, not failed. The finite correction is to
locate the construction manifest/audit and hash-bind it before S1. No data
rebuild is authorized merely to recreate a missing proof edge.

## A12-C — provenance-slice reporting (diagnostic)

Final OOF must report provenance slices when support exists. Unsupported
source/class/fold cells are `NOT_ESTIMABLE`, never silently zero. Single-source
folds remain valid grouped folds; they are described as support limitations.
