# Scientific Claim Registry

## Machine Authority

`14_CLAIM_REGISTRY.json` is the admission authority. The validator derives the
admissible status from field completeness. A claim requested as `SUPPORTED`
with missing lineage is rejected and classified `HOLD_INCOMPLETE_LINEAGE`.
Markdown summaries cannot promote a machine-held claim.

## Admission Contract

A claim may be `SUPPORTED` only when all fields below are complete:

- `claim_id`, exact claim text, scope, and status,
- code Git SHA plus dirty-worktree declaration,
- semantic config hash,
- data, split, and relevant artifact hashes,
- evaluator name and version or hash,
- evidence class,
- quantitative evidence,
- limitations and invalidation conditions,
- authority or reviewer.

Allowed statuses are `PROPOSED`, `HOLD_INCOMPLETE_LINEAGE`, `SUPPORTED`,
`CONTRADICTED`, `SUPERSEDED`, and `RETIRED`.

## Evidence Classes

- `CODE_VERIFIED`: source or contract verified, no performance claim.
- `ARTIFACT_VERIFIED`: immutable artifact and lineage verified.
- `RUN_VERIFIED`: declared run reproduced under registered inputs.
- `HUMAN_VERIFIED`: explicit reviewer decision with review lineage.
- `HISTORICAL_MEMORY`: preserved context, not current claim authority.
- `USER_SUPPLIED_UNVERIFIED`: user-provided evidence awaiting registration.

## Registry

| Claim ID | Status | Claim | Authority |
| --- | --- | --- | --- |
| `C2V2-REVIEW-001` | `SUPERSEDED` | Pre-review training block | snapshot V3 |
| `TRACK-263-WEIGHT-001` | `CONTRADICTED` | Weight alone explains `000263` | `AGENTS.md` |

These entries constrain action but are not promoted performance claims.
Future entries must use a versioned manifest or report containing every field.
