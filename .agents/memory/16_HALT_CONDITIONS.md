# Halt And Permission Contract

## Machine Authority

`16_HALT_CONDITIONS.json` is the executable contract for halt triggers, action
permissions, observation envelopes, retry limits, and stop behavior.

For new material tasks, V2 is the active governance protocol: start with
`.agents/memory/00_AGENT_BOOTSTRAP.md`, resolve authority through
`.agents/skills/project-state-steward/scripts/manage_agent_governance.py`, and
validate selected skills against `.agents/skills/skill_inventory.json`. Existing
V1 capsules remain on `manage_short_memory.py` until an explicit migration;
never create a V2 record with the same task ID as a V1 capsule.

## Halt Before Effects

Stop before edits, execution, deletion, publication, or external effects when:

- material intent, scope, authority, or acceptance criteria remain ambiguous,
- required lineage, config, data, evaluator, or artifact hashes are missing,
- current evidence contradicts the proposed claim or state transition,
- review coverage, leakage, split, schema, or reproducibility gates fail,
- a long run lacks its declared short gate or permission,
- cleanup ownership or rebuildability is uncertain,
- rollover finds an open legacy task that has not been atomically adopted,
- task ownership, runtime thread, worktree, revision, block hash, or active
  lease conflicts outside a validated recovery path.

Read authoritative project sources first. Ask the user only for facts that cannot
be safely discovered locally and that materially change the result.

Additional halt triggers:

- Two authorities claim current status for the same scope.
- Claim promotion lacks its complete manifest.
- A method transition skips any required forward state.
- A managed task was changed outside `manage_short_memory.py`.
- A lost-token recovery lacks a matching recorded/current `CODEX_THREAD_ID`.
- An active or ambiguous owner takeover lacks exact user authorization, fresh
  task CAS/worktree confirmation, or a hash-bound audit event.
- Long-memory promotion relies on elapsed time, inactivity, or completion
  without an accepted maturity packet and satisfied revalidation triggers.
- A remote storage mutation lacks a fresh authorization that binds target path,
  source hash/bytes, operation, byte ceiling, replacement permission, expected
  final hash/size, and unique-purpose proof.
- A stopped Studio, CPU-only status, CLI/browser access, or reuse language is
  being treated as authorization to write remotely or broaden resource scope.
- `pig-gpu-l4-r2` is proposed for any access without a fresh user authorization
  that names R2 and specifies the preservation or disposal outcome.
- A V2 permit is requested with an invalid canonical inventory or generated
  skill view drift.
- A caller-supplied coordination root differs from the root resolved from the
  registered worktree.
- A V1 and V2 task share an ID, or a V2 record/event/plan/owner invariant is
  malformed or hash-invalid.
- Worktree retirement lacks independent removal, reference, process, and
  preserved-evidence proof. Branch deletion is a separate disposition.

## Permission Boundaries

- Project-local Markdown may be maintained under standing approval.
- Destructive cleanup, external publication, deployment, and permission changes
  require their specific authority.
- Existing long-run permissions apply only to the exact declared lineage and gates.
- A successful tool call never grants scientific or operational promotion.
- A successful remote read, authentication, or control-plane observation never
  grants permission to copy, extract, overwrite, delete, or otherwise mutate
  remote storage.

## Observation Envelope

Every harness action returns `status`, `summary`, `next_actions`, and
`artifacts`. Errors also return `root_cause_hint`, `safe_retry`, and
`stop_condition`. Missing fields make the observation invalid.

## Retry And Stop

- Retry only when the root-cause hint identifies a bounded, reversible action.
- Revalidate the failed precondition before retrying.
- Stop after the declared retry limit or when the stop condition is true.
- Never retry a destructive, external, or promotion action by weakening gates.

## Halt Record

Record:

- `status`: `BLOCKED`, `CONTRADICTED`, or `NEEDS_AUTHORITY`,
- exact failed gate,
- evidence inspected,
- actions intentionally not taken,
- smallest next action that could reopen progress,
- owner or authority needed.

Do not label work blocked merely because it is difficult. Halt only on a declared
gate, unresolved material ambiguity, missing authority, or unsafe external effect.
