# Governance execution-model repair: root-cause map

This map is the pre-fix audit for `GOVERNANCE-EXECUTION-MODEL-REPAIR-V2-20260814-01`.
The reproduction uses disposable synthetic state only; it does not open or mutate
the active S1 task, Temporal-v2 data, model code, or scientific authorities.

1. Descendant commit blocks the next permit.
   - Conflation: `AUTHORIZED_TASK_PROGRESS` becomes `UNRELATED_HEAD_CHANGE`.
   - Cause: `permit` compares HEAD with one mutable worktree snapshot.
   - Repair: keep `BASE_HEAD`; advance `ACCEPTED_TASK_HEAD` for scoped descendants.

2. Permitted artifact blocks the next permit.
   - Conflation: known task delta becomes `UNKNOWN_DELTA`.
   - Cause: no durable exact artifact binding exists in accepted lineage.
   - Repair: classify path/effect ownership before advancing the accepted fingerprint.

3. Expired permit blocks amendment or continuation.
   - Conflation: `EXPIRED_PERMIT_HISTORY` becomes `ACTIVE_VALID_PERMIT`.
   - Cause: guards test `active_permit` presence, not validity at transition time.
   - Repair: normalize expiry into append-only history before slot evaluation.

4. Accepted untracked evidence becomes unknown.
   - Conflation: governance lineage is replaced by raw Git status.
   - Cause: exact path/content is not bound to accepted task lineage.
   - Repair: Git `untracked` alone cannot override an accepted hash binding.

5. Predictable artifacts require repeated administrator amendments.
   - Conflation: execution scope becomes one-file transactions.
   - Cause: bootstrap lacks bounded roots and effect categories.
   - Repair: register bounded roots with traversal protection and effect limits.

6. New task is not execution-ready.
   - Conflation: predictable metadata is deferred as a user repair.
   - Cause: bootstrap stores no artifact roots or bounded execution envelope.
   - Repair: create G0 with worktree, scope, base/accepted state, and permit lifecycle.

7. Recovery requires governance repair.
   - Conflation: interruption becomes takeover or rebaseline.
   - Cause: cursor and accepted lineage are not first-class recovery state.
   - Repair: rotate ownership while preserving base, accepted, actual, cursor, and events.

The synthetic pre-fix fixture in `tests/test_governance_execution_model.py`
proves the historical sequence A-I and records the exact blocker classes before
the manager repair is applied. The full acceptance fixture in
`tests/test_governance_execution_model_acceptance.py` proves continuous
scientific-task progress and negative drift rejection without scientific data.
The implementation must preserve fail-closed rejection for owner-only,
external, mixed, out-of-scope, stale-CAS, unrelated HEAD, traversal,
merge-conflict, and scientific-data mutations.
