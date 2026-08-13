---
name: agent-governance-steward
description: >-
  Enforce project agent governance V2 for material tasks: bounded memory recall,
  authority receipts, visible plan confirmation, action permits, structured
  skill routing, typed evidence, correction learning, integration, and worktree
  retirement. Use before effects and at every phase or closeout boundary.
---

# Agent Governance Steward

Use this skill to run the machine-enforced project loop. Do not substitute a
prose checklist for a manager transition.

## Start

1. Run `manage_agent_governance.py bootstrap` from the registered worktree.
2. Retrieve only the active task and authority scopes named by bootstrap.
3. For a new material task, prepare a V2 JSON packet containing task class,
   current authority receipts and hashes, acceptance IDs, risks, non-actions,
   structured skill selections, and plan steps with allowed effects.
4. Run `create`, retain the private token only in the live session, and show the
   plan to the user.
5. Run `confirm-plan`. High-risk effects require a user confirmation reference.
6. Obtain `permit` before each effect.

The manager records actual skill use as hash-chained task events. When a user
correction, repeated failure, dependency drift, or validator failure implicates
a skill, add a `MAINTENANCE_DUE` record through the task closeout packet. Clear
it only after a reviewed skill update or an evidence-backed no-change review.

## Phase boundaries

- Use `advance` to consume the current permit, bind typed evidence, close the
  step, and open its successor atomically.
- Use `amend-plan` if new work is discovered. It revokes the permit and plan
  confirmation; communicate and reconfirm the new digest.
- Never add a hidden effect after the final step.

## Closeout

1. Run `review-outcome` with one typed outcome and a disposition for every dirty
   path.
2. For accepted work, integrate and revalidate on the target branch before
   closeout. For failed or partial work, hash-bind and extract unique evidence.
3. Run `close` with exactly one learning disposition:
   `VALIDATED_CORRECTION`, `UNVERIFIED_FAILURE`, or `NO_DURABLE_LESSON`.
4. Keep blocked, unknown, user-owned, actively referenced, or dirty-unknown
   worktrees protected. Worktree retirement and branch deletion are separate.
5. Run governance validators and live-agent regression when available. Fixture
   judge tests never establish live-agent reliability.

## Commands

The manager is:

```text
.agents/skills/project-state-steward/scripts/manage_agent_governance.py
```

Use `--help` on `create`, `confirm-plan`, `permit`, `advance`, `amend-plan`,
`review-outcome`, and `close`. Pass JSON packets for complex typed payloads.

## Stop conditions

Stop on unknown or conflicting authority, stale hashes or revisions, missing
reasoning coverage, a second active worktree, absent evidence, unintegrated
accepted work, unextracted failure evidence, incomplete learning fields, or an
unknown cleanup owner.
