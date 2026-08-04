---
name: project-state-steward
description: >-
  Track planned-prompt checklists and reconcile project memory, decisions,
  workflow, memory maturity, living dossier, and repository hygiene. Use before
  the first effect of a material
  multi-step task, at step or phase boundaries, after interrupted work, and
  automatically before final handoff when code, tests, configuration, data
  contracts, generated artifacts, or a durable correction changed. Also use
  for stale context, expired short memory, workflow drift, promotion, skill
  maintenance, or cleanup uncertainty.
---

# Project State Steward

## Overview

Keep the repository understandable to a fresh agent. Preserve only verified,
reusable knowledge and classify cleanup candidates without risking project
authority, user work, data, models, or scientific lineage.

## Required Outcome

For a material planned task and its final handoff:

1. Create each new material task through the atomic short-memory manager before
   the first effect.
2. Checkpoint each completed step through revision/hash CAS before the next
   step starts.
3. Reconcile the session delta with project memory and workflow.
4. Store validated corrective methods, not a diary of mistakes.
5. Audit files created or exposed by the session.
6. Remove only session-owned, proven-regenerable waste.
7. Record uncertain cleanup candidates with evidence and a disposition.
8. Check whether the selected skills and the skill portfolio need maintenance.

Do not update every memory file by habit. A task checklist is temporary
execution state, not durable knowledge; never promote it merely to show work.
Simple read-only Q&A does not create a checklist.

## Step 0: Track The Planned Prompt

Trigger when the prompt uses a multi-step plan or requires edits, execution,
deletion, publication, permission, or another external effect. Before the first
effect, create one `01_PROJECT_MEMORY_SHORT.md` task with:

- Stable task ID, concise prompt outcome, task status, and opened timestamp.
- Acceptance criteria and selected reasoning/execution skills.
- Stable step IDs using `TODO`, `IN_PROGRESS`, `BLOCKED`, `DONE`, `DEFERRED`,
  or `CANCELLED`.

Keep at most one `IN_PROGRESS` step per task. Update only when a step or phase
starts, completes, blocks, defers, or cancels. Do not write after every command.
Each `DONE` step requires a concrete artifact, test, diff, decision, or
inspection as evidence. Once acceptance is proven, checkpoint `DONE` before the
next step's first effect. Each unfinished step requires the smallest next
action. Derive task status from its steps; do not mark a task `DONE` while an
actionable step remains open.

After a restart, treat surviving `IN_PROGRESS` as interrupted `IN_PROGRESS`
work, not proof that the step failed or completed. Inspect the declared evidence
and actual side effects before execution. If acceptance is already proven,
record `DONE` and evidence. If work is partial, resume from the smallest safe,
idempotent checkpoint. If the effect cannot be repeated safely or ownership
changed, mark `BLOCKED` and halt. Never rerun the whole step blindly or infer
completion from chat history alone.

Use
`.agents/skills/project-state-steward/scripts/manage_short_memory.py` for every
new task and every managed task mutation:

```text
rtk python <manager> create ...
rtk python <manager> inspect --task-id <TASK_ID>
rtk python <manager> checkpoint ... --expected-revision <N> \
  --expected-block-sha256 <SHA256>
rtk python <manager> renew ...
rtk python <manager> reconcile ...
rtk python <manager> recover ...
rtk python <manager> admin-takeover ...
rtk python <manager> takeover ...
rtk python <manager> rollover
```

Retain the generated private owner token only in the owning live session; short
memory stores only its SHA-256. The manager enforces an OS lock, atomic replace,
worktree binding, owner-token proof, lease, revision/hash CAS, and byte identity
for every non-owned task. If a crash loses the token, `recover` may rotate it
during an active lease only when the recorded runtime owner matches the current
`CODEX_THREAD_ID`. A different or unbound runtime requires explicit user
authorization and `admin-takeover` with exact task ID, revision, hash, worktree,
confirmation phrase, reason, and authorization reference. Each ownership change
adds a hash-bound audit event. Never `apply_patch` a managed block. Halt on ID
collision, stale CAS, unaudited active-lease takeover, runtime mismatch, or
external task drift.

The tool plan and short checklist serve different scopes: the tool plan drives
the live interaction, while short memory survives a new chat. Reconcile them at
every phase boundary and before final handoff.

## Step 1: Establish the Session Delta

Read the mandatory project memory files and any task-specific authority. Then
inspect:

```text
rtk git status --short
rtk git diff --stat
rtk git diff -- <relevant-paths>
```

Separate these classes before writing or deleting anything:

- Changes that existed before the session or belong to the user.
- Changes made in the current session.
- Generated artifacts produced by current commands.
- Durable decisions, corrections, and workflow changes.
- Hypotheses or unresolved failures that remain unverified.

Never claim ownership of a dirty path merely because it appears in `git status`.

## Step 2: Admit Knowledge Only After Validation

Promote a correction into persistent memory only when all fields are known:

- `Root cause`: the mechanism that produced the failure.
- `Validated correction`: the code change or operational procedure that fixes it.
- `Evidence`: focused test, reproduction, audit, or accepted authority.
- `Reuse when`: the conditions under which the correction applies again.
- `Do not reuse when`: boundaries, contraindications, or invalidating evidence.

Keep an unresolved diagnosis in the current task report or current-decision file
with an explicit `UNVERIFIED` status. Do not turn guesses, raw errors, or failed
attempts into learned rules.

Use this compact entry shape:

```markdown
## YYYY-MM-DD concise topic

- Root cause:
- Validated correction:
- Evidence:
- Reuse when:
- Do not reuse when:
- Supersedes:
```

Omit `Supersedes` only when no earlier rule is replaced.

## Step 3: Route State to the Correct Authority

### Memory routing precedence

- `01_PROJECT_MEMORY_SHORT.md`: daily state, active managed resume capsules,
  and one-day closeout.
- `04_PROJECT_MEMORY_MEDIUM.md`: paused/dormant work, never a duplicate of an
  active short-memory task.
- `05_PROJECT_MEMORY_LONG.md`: stable accepted project facts only.
- `12_PROJECT_CHARTER.md`: stable project direction only.
- `13_METHOD_STATE.md`: method and decision transitions.
- `14_CLAIM_REGISTRY.md`: scientific claim status and lineage.
- `16_HALT_CONDITIONS.md`: permission and fail-closed gates.

When an older routing bullet conflicts with this lifecycle, this subsection wins.

- Update `01_PROJECT_MEMORY_SHORT.md` only with current task state, today's
  scientific handoff, and one compact previous-day closeout.
- Update `02_CURRENT_DECISION.md` for current accepted state, active blockers,
  exact authority, and the next permitted action.
- Update `08_WORKFLOW.md` for repeatable sequences, gates, and handoff procedure.
- Update `03_PROJECT_RULES.md` only for a durable project policy or explicit user
  rule, not an implementation detail.
- Update broader memory files only when their declared domain requires it.
- Update `09_HIDDEN_REVIEW.md` only for `classification_v2` Hidden-review state.
- Update `10_REPO_HYGIENE.md` for cleanup evidence, protected paths, and deferred
  disposal decisions.

Prefer editing or superseding an existing entry over appending a near-duplicate.
Preserve historical provenance when it still matters, but keep stale state out of
the current-decision and short-memory surfaces.

## Step 3A: Enforce Memory Lifecycle

1. Read `Opened` and `Expires` from short memory using `Asia/Saigon`.
2. If current, reconcile task and step states at each phase boundary.
3. If expired, run the manager's atomic `rollover` command. It refuses any open
   legacy task until its owner adopts that task.
4. Retain every nonterminal managed block byte-for-byte. Its `DONE` evidence,
   current step, next action, owner, revision, and hash remain the resume point.
5. Remove terminal task bodies and summarize their outcomes in
   `Previous-Day Closeout` for one day only.
6. Reset the daily lifecycle and closeout without changing retained task bytes.
   Move work to medium only when explicitly paused/dormant, after removing its
   active capsule so no dual current authority exists.
7. Run `manage_memory_maturity.py scan` at material closeout. Treat elapsed
   inactivity as a review reminder only; never treat it as evidence.
8. Register completed reusable knowledge, then require typed evidence, current
   authority, deliberate acceptance, limitations, source disposition, and
   revalidation triggers before `promote`.
9. Use `reopen` when a trigger fails and `synthesize` to repair the generated
   dossier after interruption. Never manually patch its generated section.

A paused task in medium requires its task ID, remaining step IDs, `status`,
`opened`, `next_action`, `evidence`, and an `exit` condition. A long-term item
requires accepted authority, scope, acceptance date, and an invalidation
condition.

Use the maturity manager for every registry mutation:

Start candidate packets from
`templates/memory_maturity_candidate.json`; bind real paths and hashes before
registration, and never leave placeholder values in a reviewed entry.

```text
rtk python <maturity-manager> scan
rtk python <maturity-manager> register --packet <candidate.json> ...
rtk python <maturity-manager> review --entry-id <ID> ...
rtk python <maturity-manager> promote --entry-id <ID> ...
rtk python <maturity-manager> reopen --entry-id <ID> ...
rtk python <maturity-manager> revise --entry-id <ID> --packet <candidate.json> ...
rtk python <maturity-manager> synthesize
```

If accepted knowledge is contradicted, mark its method or claim
`CONTRADICTED`, preserve provenance, and return to the earliest invalidated
gate. Never copy raw chat, command logs, or error diaries into active memory.

## Step 3B: Balance the Skill Portfolio

Treat skills as a portfolio, not a flat toolbox. Use the catalog description to
select candidates, then read the complete `SKILL.md` for each selected skill.

- Classify selected skills as reasoning, synthesis, execution, domain, or
  verification.
- For nontrivial work, select at least one reasoning or verification skill.
- For architecture, behavior, debugging, evaluation, synthesis, or data-contract
  work, require a reasoning skill; verification alone does not satisfy the gate.
- Select code or domain skills only when the task actually needs them.
- Record selected skills and their purpose in the working plan or run manifest.
- At handoff, update `11_SKILL_PORTFOLIO.md` with use, review, evidence, and next
  action; do not count mere availability as usage.

Mark a skill for maintenance after a user correction, repeated failure, changed
dependency or CLI contract, validator failure, or stale reference. Review active
code skills after 30 days without review and reasoning skills after 60 days
without review. Use `skill-creator` to change a skill, validate it, and keep the
change separate from algorithm edits.

Do not bulk-refresh every skill or force an unrelated reasoning skill into a
small task. When a task changes architecture, behavior, data contracts, system
design, debugging strategy, evaluation, or synthesis, a code skill alone is
insufficient.

## Step 4: Audit Repository Hygiene

Classify each relevant candidate before any cleanup action.

### Regenerable

Examples include bytecode caches, test caches, linter caches, and temporary files
created by the current session whose producer and rebuild command are known.

Remove a regenerable item automatically only when all conditions hold:

- The current session created it or ownership is otherwise certain.
- It is not tracked and contains no user change.
- No manifest, source, test, documentation, or active process references it.
- Recreating it does not require unavailable input or irreplaceable compute.
- Removal stays inside the repository and is allowed by current safety rules.

### Review Required

Treat old worktrees, project temp directories, duplicate outputs, large generated
artifacts, caches with unknown producers, and untracked feature directories as
review-required. Record:

- Exact path or bounded path pattern.
- Size and age when useful.
- Producer or likely origin.
- Reference and lineage checks performed.
- Rebuild command or reason it may be irreplaceable.
- Recommended action and required confirmation.

### Protected

Protect all user-owned or pre-existing dirty changes, tracked deletions, source,
tests, configs, datasets, models, checkpoints, manifests, review decisions,
accepted outputs, registered worktrees, and scientific lineage evidence.

Unknown means protected until evidence changes the classification.

## Step 5: Use the Cleanup Tool Safely

When broad storage inspection is needed and the project tool is available, use
the workflow documented in `docs/STORAGE_CLEANUP_TOOL.md`. Treat its scan as
discovery, not deletion authority.

- Keep selection empty by default.
- Prefer Windows Recycle Bin over permanent deletion.
- Require preview, fingerprint revalidation, and typed confirmation.
- Never bypass protected-item or lineage-review classifications.
- Never start broad cleanup merely to make `git status` look clean.

Do not launch a persistent dashboard at every handoff. Use the ledger for normal
session cleanup and the dashboard for explicit or material storage review.

## Step 6: Reconcile and Verify

After memory or workflow edits:

```text
rtk git diff -- <changed-memory-and-skill-files>
rtk git diff --check
rtk grep "^.{101,}$" <changed-markdown-files>
rtk python .agents/skills/project-state-steward/scripts/validate_memory_contract_v2.py
rtk pytest tests/test_project_governance_contract.py
```

Run focused tests and linters required by the implementation. Re-read the updated
top sections to ensure the current state does not contradict an older entry.

Before final response, report:

- Memory files read.
- Memory and workflow files updated, or why no update was warranted.
- Validation evidence for each learned correction.
- Cleanup performed.
- Protected or deferred cleanup candidates and their ledger location.

## Safety and Failure Rules

- In audit-only tasks, keep source code unchanged. Project memory may be updated
  only for durable, verified findings allowed by project instructions.
- If a memory patch fails twice, stop and re-read the exact target section.
- If correction evidence is incomplete, record no learned rule.
- If cleanup evidence is incomplete, delete nothing and record the candidate.
- If concurrent user changes overlap a target, preserve them and narrow the edit.
- If automatic observer hooks are inactive, use project-local memory as the
  cross-chat authority; do not assume conversation context will persist.
