# Method And Decision State

## Current State Machine

Only these forward transitions are valid:

`PROPOSED -> DESIGNED -> IMPLEMENTED -> DEV_PASS -> VALIDATED -> FROZEN -> PROMOTED`

Terminal branches are `REJECTED`, `BLOCKED`, `SUPERSEDED`, and
`NOT_REPRODUCIBLE`. A terminal re-entry requires an explicit reopening record
and resumes at the earliest invalidated gate.

The machine authority is `13_METHOD_STATE.json`. Every new transition records
lineage, gate evidence, limitations, authority, and timestamp. A metric value
alone cannot satisfy a transition.

## Historical Legacy State Machine

Use only these forward transitions:

`DRAFT -> STATIC_CHECKED -> SYNTHETIC_VALIDATED -> SHORT_RUN_VALIDATED`

`SHORT_RUN_VALIDATED -> FULL_RUN_AUTHORIZED -> EXECUTED -> EVALUATED`

`EVALUATED -> ACCEPTED -> RETIRED`

Any state may move to `BLOCKED`, `CONTRADICTED`, or `SUPERSEDED`.
Re-entry requires explicit evidence and starts from the earliest invalidated gate.

## Transition Contract

Every transition records:

- method or decision ID,
- previous and next state,
- code Git SHA and dirty-worktree status,
- semantic config hash,
- input data or artifact hashes,
- evaluator and evidence class,
- gate results and limitations,
- authority granting the transition.

Missing fields block the transition. A later state never implies an earlier gate.

## Current Entries

| ID | State | Authority | Next allowed transition |
| --- | --- | --- | --- |
| `c2v2.reviewed_lineage` | `SUPERSEDED` | snapshot V3 | Use amendment V1 |
| `c2v2.reviewed_lineage.amendment_v1` | `FROZEN` | amendment snapshot V4 | Bounded S2 only; no S3 |
| `tracking.current_baseline` | `FROZEN` | tracking freeze | Bounded authorized audit |
| `agent.memory_lifetime_v1` | `VALIDATED` | validator | Maintain |
| `agent.governance_hardening_v1` | `VALIDATED` | registry + eval | Maintain |
| `agent.atomic_task_memory_v1` | `VALIDATED` | manager + eval | Authorized admin test |
| `agent.memory_maturity_v1` | `VALIDATED` | registry + manager + eval | Test a domain entry |
| `c2v2.social_topk_k3_v1` | `DEV_PASS` | S2 reports | Resolve gates; paired S0/S1/S2 only |
| `c2v2.motion_episode_boundary_v1` | `DEV_PASS` | exact-SHA report | Fresh holdout only |

Do not duplicate metric detail here. Link the authoritative registry or report.
