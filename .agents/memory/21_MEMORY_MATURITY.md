# Memory Maturity And Living Dossier

## Authority

- Machine authority: `21_MEMORY_MATURITY.json`.
- Manager: `project-state-steward/scripts/manage_memory_maturity.py`.
- Reading surface: generated `Living Project Dossier` in file `05`.
- The registry is canonical; the dossier is derived and recoverable.

## Maturity Principle

Elapsed time and lack of edits are not evidence. They may justify inspection,
but never promotion. A completed task is also not automatically durable
knowledge. Promotion requires reusable value, typed evidence, current authority,
an explicit review event, accepted scope and limitations, source disposition,
and revalidation triggers.

## State Machine

`CANDIDATE -> EVIDENCE_BOUND -> REVIEWED -> ACCEPTED -> PROMOTED`

Branches are `BLOCKED`, `REJECTED`, `REVALIDATION_REQUIRED`, `CONTRADICTED`,
`SUPERSEDED`, and `ARCHIVED`. Re-entry requires new evidence and a new review;
history is append-only in the transition ledger.

## Typed Admission

- `project_fact`: evidence-bound accepted project fact.
- `project_contract`: accepted scope, invariant, or execution contract.
- `validated_method`: method registered as `FROZEN` or `PROMOTED`.
- `supported_claim`: claim registry status must be `SUPPORTED`.
- `validated_correction`: root cause, correction, evidence, reuse, and non-reuse.
- `limitation`: accepted boundary that prevents unsafe generalization.

Each entry binds source and evidence hashes, authority references, scope,
limitations, invalidation conditions, and at least one revalidation trigger.
Scientific methods and claims require an independent review event.

## Revalidation

Triggers are event-based: source or artifact hash change, authority replacement,
method-state change, claim-status change, or a declared manual condition. A
failed trigger removes the entry from the current dossier on synchronization
and requires explicit reopening. Calendar age alone never changes truth status.

## Operational Workflow

1. `scan` classifies candidates and promoted entries without mutation.
2. `register` admits a deduplicated candidate packet.
3. `review` records accept, hold, or reject after typed gates pass.
4. `promote` writes canonical registry state and regenerates the dossier.
5. `reopen` moves invalidated knowledge back to evidence review.
6. `revise` replaces its evidence packet while preserving acceptance and
   promotion history.
7. `archive` closes non-promoted material with provenance.
8. `synthesize` repairs a stale derived dossier after interruption.

Promotion must state that the medium source was removed from active state,
retained only as non-authoritative history, or was never a medium item. This
prevents one fact from having two current authorities.
