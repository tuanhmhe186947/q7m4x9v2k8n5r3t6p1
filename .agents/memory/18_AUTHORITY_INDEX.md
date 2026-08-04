# Authority Index

## Precedence

1. Current explicit user direction for this workspace and task.
2. Root `AGENTS.md` safety and execution contract.
3. Scope-specific current authority listed in `18_AUTHORITY_INDEX.json`.
4. Supporting authority listed for that same scope.
5. Historical archives, ledgers, reports, and raw observations.

Lower-precedence material cannot silently replace higher-precedence authority.
A contradiction at the same current level requires `NEEDS_AUTHORITY` and a halt
before edits, execution, promotion, deletion, or publication.

## Scope Contract

Each machine entry has one `current_authority`, zero or more supporting and
historical authorities, `valid_from`, `supersedes`, and conflict handling.
Two entries with the same scope are invalid. A current authority cannot point to
an archive. Missing or unreadable current authority is fail-closed.

The `memory.maturity` scope is machine-governed by
`21_MEMORY_MATURITY.json`. File `05` is its generated reading surface, not a
second promotion authority. A mismatch requires synthesis or revalidation.

## Retrieval Order

1. Read `01_PROJECT_MEMORY_SHORT.md` and check expiry.
2. Read this index and select the task scope.
3. Read only the scope's current and supporting authorities.
4. Retrieve historical authorities only for provenance or contradiction audit.
5. Record a halt when sources at the same precedence remain inconsistent.

## Archives

- `17_PROJECT_MEMORY_SHORT_ARCHIVE_2026-07-31.md`: pre-lifecycle short memory.
- `20_CURRENT_DECISION_ARCHIVE_2026-07-31.md`: pre-index decision history.

Archives preserve evidence but never become current merely because they contain
more detail or a newer-looking metric.
