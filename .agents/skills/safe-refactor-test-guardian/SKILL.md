---
name: safe-refactor-test-guardian
description: >-
  Guard small classification_v2 code and schema changes with import review,
  compile checks, focused tests, row/schema audits, diffs, and rollback. Use
  before and after editing classifier modules, scripts, configs, or tests.
---

# Safe Refactor Test Guardian

## Purpose

Keep classifier changes isolated, reviewable, test-backed, and reversible without
touching raw data or concealing schema and row-count regressions.

## When to use

Invoke before editing any classification source, operator script, config,
contract, public API, or test and again before staging or committing the change.

## Project context

Stable modules, active modules, stubs, deprecated files, and numbered operator
scripts coexist. Preserve canonical schemas and the `00` through `09` workflow;
never infer that an empty or old-looking file is safe to delete.

## Required inputs

- exact user request and allowed file scope;
- target files, direct imports, callers, tests, and public schemas;
- clean starting SHA and existing unrelated worktree changes;
- expected row/key counts and canonical columns;
- bounded verification commands and rollback boundary.

## Scientific invariants

- Read targets and import graph before editing.
- Classify each touched module as stable, active, stub, or deprecated.
- Keep patches small and separate refactors from algorithm changes.
- Preserve public APIs or provide an explicit migration.
- Audit canonical columns, row counts, keys, and no-silent-drop behavior.
- Use synthetic or tiny data before any broader run.
- Preserve user changes and never modify raw data.

## Ordered procedure

1. Read project instructions, memory, target files, imports, callers, and tests.
2. Record starting status and isolate unrelated changes.
3. Write a small change plan with expected schemas, counts, and tests.
4. Patch only the selected behavior using `apply_patch`.
5. Inspect each file diff and scan changed files for long lines.
6. Run `py_compile`, package `compileall`, and import checks.
7. Run focused unit and tiny-data smoke tests.
8. Compare canonical schemas, row/key counts, and output audit errors.
9. Summarize changed behavior, intentionally unchanged behavior, and risks.
10. Provide a non-destructive rollback path before commit.

## Required outputs

Produce a change plan, changed-files report, compile report, test report,
schema diff, row-count diff, unresolved-risk list, and rollback instructions.

## Validation commands

Run changed-file `py_compile`, package `compileall`, focused pytest, import
smoke, [canonical-column audit](../checks/audit_canonical_columns.py),
`git diff --check`, and an overlong-line scan. Never substitute a long training
run for a focused test.

## Stop conditions

Stop on unexpected worktree overlap, ambiguous stable/stub status, public API
breakage, compile/import failure, unexplained row loss, canonical schema drift,
failed focused tests, or a patch broader than the approved plan.

## Forbidden actions

Do not edit `data/`, overwrite final outputs without a flag, delete stubs
automatically, combine broad refactor and algorithm change, alter locked splits,
use broad exception suppression, run long training, reset hard, or delete
untracked files without review.

## Completion report format

Use the shared [completion report](../templates/skill_completion_report.md) and
local [change report contract](templates/change_report.json). List every changed
file, behavior, check, count delta, unresolved risk, rollback, and PASS/FAIL.
