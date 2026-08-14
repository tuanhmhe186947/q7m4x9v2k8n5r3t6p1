# Governance execution-model design audit

| Current rule | Failure mode | Safety purpose | False-positive case | Target behavior | Code location |
| --- | --- | --- | --- | --- | --- |
| A permit requires live `head_sha` to equal the recorded worktree head. | A task-owned descendant commit blocks the next permit. | Reject unrelated branch movement. | An exclusive registered task commits B after A. | Preserve immutable base A and advance only an explicitly validated accepted task head to B. | `AgentGovernanceLedger.permit` |
| A permit requires the live fingerprint to equal the previous snapshot. | Authorized edits or generated evidence block the next permit. | Detect unexplained dirty changes. | A permitted artifact is created between two checkpoints. | Compare fresh observation to accepted progress; accept only a classified task-owned transition. | `_worktree_identity`, `permit`, `advance` |
| Any `active_permit` blocks a new permit. | An expired permit is unusable yet remains a zombie lock. | Prevent simultaneous mutation permits. | A valid task needs a fresh permit after expiry. | Expire and classify the old permit, then issue exactly one fresh permit only when the transition is authorized. | `permit`, `advance` |
| The worktree snapshot is overwritten at `advance`. | Base provenance is lost as the task progresses. | Record current state for CAS and closure. | A -> B -> C cannot later prove its start. | Keep base and accepted snapshots separately; record fresh actual observations in events. | `create`, `advance`, record validation |
| Cursor is inferred from mutable plan metadata. | Completed history can be presented as an earlier active step. | Enforce one active logical step. | Durable DONE history disagrees with a stale cursor. | Derive the next valid cursor from append-only `STEP_ADVANCED` history and reject regressions. | `_validate_record`, `advance` |

`TASK_OWNED_AUTHORIZED` requires the admitted worktree, a fresh CAS mutation,
the active permit's expected accepted state, permitted effect(s), declared task
scope, valid Git lineage, and no unexplained path. `EXTERNAL_OR_OWNER` and
`UNKNOWN_OR_MIXED` fail closed. Existing records remain readable; without an
explicit task scope, a changed state is not automatically accepted.
