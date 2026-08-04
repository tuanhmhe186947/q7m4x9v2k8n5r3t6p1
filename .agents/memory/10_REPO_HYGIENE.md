# Repository Hygiene Ledger

## 2026-07-31 governance harness cleanup

- `REMOVED`: two `__pycache__` directories created by this session's
  `compileall`; both are deterministic bytecode caches.
- `REMOVED`: the same two caches were recreated by final validator/tests and
  removed again after their resolved paths were verified inside the workspace.
- `REMOVED`: untracked `validate_memory_contract.py`, superseded by referenced
  `validate_memory_contract_v2.py` and absent from repository references.
- `REVIEW_REQUIRED`: `tools/pig_autoresearch/` remains untracked with unknown
  ownership; this classification is superseded by the explicit user upgrade.
- `DURABLE_SESSION_OUTPUT`: retain the autoresearch policy, candidate,
  immutable launchers, harness, docs, authorization example, and focused test.
- `REMOVED`: invalid tracking/classification templates were superseded by the
  JSON candidate surface and deleted through `apply_patch`.
- `REMOVED`: session-created autoresearch bytecode and two matching focused
  test bytecode files; all are reproducible from compile/test commands.
- `PROTECTED`: all other caches below `tests/` and `tests/forensic/`; ownership
  is uncertain, so no broad cache deletion ran.
- `PROTECTED`: all pre-existing dirty paths, tracked deletions, worktrees,
  scientific artifacts, data, models, and user-owned changes remain untouched.

## 2026-07-31 memory lifecycle migration

- `PROTECTED`: `17_PROJECT_MEMORY_SHORT_ARCHIVE_2026-07-31.md` preserves the
  pre-migration short-memory history and must not be deleted as cleanup.
- `PROTECTED`: `12` through `16` memory authorities, the steward skill,
  validator, and governance regression test are active feature work.
- No data, model, scientific artifact, or pre-existing dirty path was deleted.

Use this ledger for evidence-backed cleanup decisions. It is not deletion
authority. Unknown or pre-existing dirty paths remain protected.

## Labels

- `REGENERABLE`: producer and rebuild method are known; no durable state.
- `REVIEW_REQUIRED`: origin, references, lineage, or ownership needs review.
- `PROTECTED`: user work, tracked changes, source, data, model, manifest,
  accepted output, worktree, or scientific evidence.
- `REMOVED`: disposed through an approved recoverable workflow with evidence.

## 2026-07-31 baseline

- `PROTECTED`: tracked deletions under `.codex_tmp/legacy_smoke/` predate this
  skill session; do not restore or alter them without ownership review.
- `PROTECTED`: active modifications in `.serena/project.yml`, `pyproject.toml`,
  the classification review GUI, and its focused test.
- `PROTECTED`: active modification in
  `src/pig_behavior/classification_v2/review/mini_cvat_adjudication.py`.
- `PROTECTED`: untracked storage-cleanup source, documentation, tests, and
  `.serena/memories/` are active feature work until reviewed or committed.
- `REGENERABLE`: `src/pig_behavior/storage_cleanup/__pycache__/` is Python
  bytecode, but it was present before this session and was not removed.
- `REVIEW_REQUIRED`: `.codex_runs/` and `.codex_tmp/` contain historical
  comparisons, logs, archives, and run directories; inspect provenance before
  any disposal.
- `REGENERABLE, DEFERRED`: `.serena/cache/python/` appears to be a pre-existing
  Serena cache. It was not removed because ownership and active-use checks were
  not sufficient for automatic cleanup.
- No cleanup deletion was performed during the skill-creation session.

## Entry Shape

- Date and bounded path:
- Label:
- Producer or origin:
- Reference and lineage checks:
- Rebuild method:
- Decision and evidence:
- Approval or removal record:
