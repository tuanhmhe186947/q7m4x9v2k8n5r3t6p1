# Agent Governance Protocol V2

Status: implementation contract for `AGENT-REFORM-20260813-01`.

## Problem statement

The legacy controls protect task bytes with locks, leases, owner tokens, and
compare-and-swap checks. They do not prove that an agent retrieved current
authority, communicated a stable plan, used the selected skills, integrated an
accepted result, extracted evidence from a failure, or retired its worktree.
Passing the legacy validator therefore gives false assurance about behavior.

V2 makes the execution loop machine-enforced:

```text
retrieve -> confirm plan -> permit bounded effect -> verify -> deliver
         -> record outcome -> learn -> regress -> retire
```

## Compatibility boundary

- Preserve every managed V1 task block byte-for-byte unless its owner mutates it
  through `manage_short_memory.py`.
- Keep the V1 manager and reader available for existing tasks.
- Store V2 active records and append-only events below the canonical ignored
  `.agents/runtime/agent_governance_v2/` directory. Never create a worktree-local
  shadow ledger.
- Replace mandatory full reads of `01_PROJECT_MEMORY_SHORT.md` with a bounded
  bootstrap query. Retrieve one relevant V1 or V2 task on demand.
- Promote only validated, reusable corrections into tracked project memory.

## Task record invariants

A V2 task record must contain:

- a typed task class and risk class;
- authority receipts with scope, locator, status, read time, and exact hash;
- acceptance IDs, explicit risks, and explicit non-actions;
- structured skill selections with role, purpose, source, and pinned hash;
- a versioned plan whose digest covers every step and allowed effect;
- one owner, one live runtime, one lease, one worktree binding, revision, and
  record hash;
- an append-only, hash-chained event stream.

Unknown authority, an unconfirmed plan, stale hashes, a wrong skill route, or a
worktree collision fails closed before an action permit is issued.

## Plan and action protocol

1. `create` validates the packet but leaves the task `PLANNED`.
2. `confirm-plan` binds a visible confirmation reference to the plan digest.
   High-risk, destructive, paid, protected-authority, publication, or remote
   effects require an explicit user confirmation reference.
3. `admit-worktree` permits one active worktree per task. A second worktree needs
   a declared parent/child purpose and separate lifecycle record.
4. `permit` binds one active step, its allowed effects, current authority and
   skill digests, the worktree fingerprint, and an expiry.
5. `advance` consumes the permit, binds typed evidence to acceptance IDs, closes
   the old step, and opens the next step in one CAS mutation.
6. `amend-plan` revokes any live permit, increments the plan version, changes the
   digest, and requires confirmation again. A terminal step cannot be followed
   by a hidden new effect.

## Execution progression model

### Repair acceptance markers

The execution model is accepted only when the following markers are backed by
the synthetic and negative-control fixtures:

- `GOVERNANCE_EXECUTION_MODEL_REPAIR_COMPLETE=YES`
- `AUTHORIZED_TASK_PROGRESS_CONTINUOUS=YES`
- `EXPIRED_PERMIT_ZOMBIE_BLOCKING=0`
- `PER_FILE_ADMIN_SCOPE_CHURN_REQUIRED=NO`
- `ACCEPTED_RECONCILED_ARTIFACT_RECLASSIFICATION=PASS`
- `AUTHORIZED_FINGERPRINT_PROGRESSION=PASS`
- `TASK_CURSOR_MONOTONIC=YES`
- `SESSION_RECOVERY_STATE_PRESERVED=YES`
- `SYNTHETIC_CONTINUOUS_TASK_PROGRESS=PASS`
- `SYNTHETIC_EXTERNAL_DRIFT_REJECTION=PASS`
- `S1_GOVERNANCE_FAILURE_PATTERN_REGRESSION=PASS`

The hard boundaries remain fail-closed for unexpected external edits,
unrelated HEAD changes, mixed unknown state, stale CAS, scope escape,
path traversal, merge-owner conflict, owner-only capture, and scientific-data
mutation. These controls are evaluated at G0, while G1 records a release
candidate and G2 performs canonical integration and closure.

Governance validates meaningful trust boundaries, not every internal operation.
The conceptual boundaries are:

- `G0`: task authorization and the bounded execution envelope;
- continuous task-owned execution inside that envelope;
- `G1`: release-candidate and evidence readiness;
- `G2`: canonical integration and task closure.

Task-start provenance is immutable: `BASE_HEAD` and `BASE_FINGERPRINT` record
the admitted worktree. `ACCEPTED_TASK_HEAD` and `ACCEPTED_TASK_FINGERPRINT`
advance only after a governance-validated task transition. `ACTUAL_WORKTREE_HEAD`
and `ACTUAL_WORKTREE_FINGERPRINT` are fresh observations; they never overwrite
base provenance merely because a task continued.

`AUTHORIZED_TASK_PROGRESS != UNEXPECTED_EXTERNAL_DRIFT`. A transition is
`TASK_OWNED_AUTHORIZED` only when its registered worktree, CAS state, permit,
Git lineage, and declared task scope or artifact root all agree. Exact accepted
path and content bindings remain accepted even when Git reports them as
untracked. `EXTERNAL_OR_OWNER` and `UNKNOWN_OR_MIXED` changes fail closed.

`EXPIRED_PERMIT_HISTORY != ACTIVE_VALID_PERMIT`. Expiry is normalized into
append-only permit history before the next manager transition evaluates the
active slot. Historical expiry cannot block a fresh bounded permit or a safe
scope recovery, while a currently valid permit still blocks conflicting admin
transitions.

Good: a scoped task edits `docs/governance/`, records evidence, commits a
descendant, and requests a new permit after expiry. The manager preserves its
base while accepting the scoped, lineal transition.

Bad: the same task contains a scoped edit plus an unexplained `src/` change.
The mixed transition is rejected and requires owner reconciliation; it is never
accepted merely because the task has a lease.

Session recovery rotates ownership only. It retains the base and accepted
provenance, worktree, append-only history, and current logical checkpoint.
Release-candidate and canonical-integration validation remain separate gates,
including destination owner-work checks.

## Execution-model repair acceptance markers

The synthetic acceptance suite records these durable markers:

- `SYNTHETIC_CONTINUOUS_TASK_PROGRESS=PASS`;
- `SYNTHETIC_EXTERNAL_DRIFT_REJECTION=PASS`;
- `S1_GOVERNANCE_FAILURE_PATTERN_REGRESSION=PASS`;
- `EXPIRED_PERMIT_ZOMBIE_BLOCKING=0`;
- `ACCEPTED_RECONCILED_ARTIFACT_RECLASSIFICATION=PASS`;
- `AUTHORIZED_FINGERPRINT_PROGRESSION=PASS`.

The repaired model makes these former blockers impossible in a correctly
authorized bounded task: `MISSING_EXECUTION_ROUTE_SELF_BLOCK`,
`EXPIRED_ACTIVE_PERMIT_ZOMBIE`, `TASK_OWNED_FINGERPRINT_SELF_BLOCK`,
`ACCEPTED_ARTIFACT_UNKNOWN_RECLASSIFICATION`, and
`PER_FILE_SCOPE_AMENDMENT_CHURN`.

## Pre-permit task-file reconciliation

`reconcile-prepermit-task-file` is a narrow post-hoc adoption operation, never
retroactive write authorization. It accepts only one explicitly enumerated
`TASK_PLAN_METADATA` JSON file under `docs/`, whose name identifies it as a
plan, hash, in-scope path, prior-write event, expected accepted fingerprint,
and current actual fingerprint all match fresh observations.

The caller must provide explicit administrator authority, state that the
original write lacked proven authority, and prove zero owner-only,
external/unknown, and mixed delta. The one permitted dirty path must be the
exact supplied path; extra paths, source, scripts, data, manifests, model
artifacts, checkpoints, binaries, and blanket worktree adoption are rejected.
The event preserves the unproven-write classification and advances only the
accepted fingerprint. It does not manufacture a HEAD transition or alter normal
permit semantics: a later ordinary permit is still required for new effects.

## Progressive delivery

`main` is the delivery spine, not an end-of-task destination. In `shared_main`
mode, every verified bounded implementation milestone is committed directly to
local `main` before the next implementation milestone starts; the commit and
the following `advance` record are its integration evidence.

An exclusive worktree is permitted only for actual concurrent-owner work or a
concrete isolation risk. Its verified milestone must be integrated into `main`
and revalidated before the next implementation milestone. It must never become
a queue of completed changes awaiting a separate reconciliation task.

`review-outcome` and `close` record the delivered result; they do not defer or
authorize ordinary integration. Reconciliation is reserved for a real merge
conflict, mixed ownership, or failed integration. Validation remains bounded to
the changed milestone unless new evidence invalidates earlier accepted work.

## Evidence contract

Evidence is not free-form `done` text. Each item contains:

- `evidence_id` and `kind`;
- an existing path, immutable URI, command result, or decision reference;
- SHA-256 when the evidence is a file;
- the acceptance IDs it supports;
- status `PASS`, `FAIL`, `OBSERVED`, or `NOT_AVAILABLE`.

The manager verifies local paths and hashes when provided. A `DONE` step must
cover every acceptance ID assigned to that step with `PASS` or an explicitly
allowed observation status.

## Worktree lifecycle

Every admitted worktree follows:

```text
ADMITTED -> ACTIVE -> RESULT_CAPTURED -> OUTCOME_REVIEWED
```

The reviewed outcome is exactly one of `ACCEPTED`, `PARTIAL`, `REJECTED`,
`BLOCKED`, or `UNKNOWN`.

- `ACCEPTED` requires integration proof, revalidation on the target branch, and
  a retirement disposition.
- `PARTIAL` requires the accepted subset to be integrated, rejected material to
  be excluded, and unique evidence to be extracted.
- `REJECTED` requires failure evidence extraction before retirement.
- `BLOCKED` and `UNKNOWN` remain protected and cannot be retired.

Every dirty path has exactly one disposition: `INTEGRATE`, `EXTRACT_EVIDENCE`,
`PRESERVE_USER_OWNED`, `DISCARD_VERIFIED_SCRATCH`, or `UNKNOWN_HALT`.
Worktree removal and branch deletion remain separate decisions.

## Learning disposition

Closeout requires exactly one disposition:

- `VALIDATED_CORRECTION`: root cause, correction, validation evidence, reuse
  conditions, and non-reuse boundaries are complete.
- `UNVERIFIED_FAILURE`: observations, evidence, hypotheses, preserved location,
  and the next validation are complete. It is not promoted as learned truth.
- `NO_DURABLE_LESSON`: evidence-backed rationale explains why no reusable
  correction exists.

An apology is not a disposition. A user correction or validator failure emits a
skill-maintenance event. `MAINTENANCE_DUE` clears only after the affected skill
is updated or deliberately reviewed and validated.

## Skill authority

`.agents/skills/skill_inventory.json` is the canonical inventory. Registry,
portfolio, and README files are views and must match it. Selection validation
rejects:

- unknown or disabled skills;
- implicit selection of a future skill;
- missing dependencies;
- missing purpose or role;
- a task-class route without its required reasoning coverage;
- a claimed skill use whose pinned `SKILL.md` hash has drifted.

## Closeout gate

A task cannot become `CLOSED` until:

- plan steps have valid terminal evidence;
- the outcome and every dirty-path disposition are recorded;
- accepted code is integrated and revalidated on the target branch, or failure
  evidence is extracted and hash-bound;
- one learning disposition is complete;
- skill-maintenance impacts are resolved or explicitly left `MAINTENANCE_DUE`;
- every worktree has a protected or retirement disposition;
- the regression manifest distinguishes fixture judge tests from live-agent
  trials.

## Required negative controls

The implementation must reject all of the following:

1. a permit without authority receipts or plan confirmation;
2. a stale authority, skill, plan, worktree, revision, or record hash;
3. an unknown skill, wrong reasoning route, empty purpose, future implicit
   skill, or broken dependency;
4. a second active worktree without a declared relationship;
5. `DONE` evidence that is missing, non-existent, hash-mismatched, or unrelated
   to the step's acceptance IDs;
6. a new effect after a terminal step without `amend-plan` and reconfirmation;
7. accepted work that is not integrated and revalidated on the target;
8. failed or partial work with unique evidence not extracted;
9. an apology or incomplete correction tuple presented as learning;
10. retirement based only on age, cleanliness, lease expiry, ancestry, or patch
    equivalence;
11. retirement while dirty paths, active references, processes, or unknown
    ownership remain;
12. fixture-only regression results presented as live-agent reliability.
13. a task-owned descendant, accepted artifact, or expired permit is treated as
    external drift solely because it is a descendant, untracked, or historical;
14. artifact roots that are wildcarded, project-root-wide, traversing, or
    outside the declared effect envelope;
15. owner-only, mixed, unrelated-head, merge-owner, binary, or scientific-data
    changes entering accepted task lineage without exact authority.

## Migration order

1. Add the V2 manager, schema validation, canonical skill inventory, and tests.
2. Add the bounded bootstrap and make new material tasks V2 by default.
3. Register the current reform task as the first compatibility migration; do
   not rewrite unrelated V1 capsules.
4. Integrate and revalidate the reform on `main`.
5. Retire the reform worktree only after the closeout gate records eligibility.
6. Audit old worktrees into a deferred ledger; do not bulk-delete candidates.
