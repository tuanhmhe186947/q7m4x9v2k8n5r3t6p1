# Agent Instructions for PIG_Behavior_Project

Before working on this repository, always read these files first:

1. `.agents/memory/01_PROJECT_MEMORY_SHORT.md`
2. `.agents/memory/02_CURRENT_DECISION.md`
3. `.agents/memory/03_PROJECT_RULES.md`
4. `.agents/memory/08_WORKFLOW.md`

For broader tracking or architecture tasks, also read:

5. `.agents/memory/04_PROJECT_MEMORY_MEDIUM.md`
6. `.agents/memory/05_PROJECT_MEMORY_LONG.md`
7. `.agents/memory/06_BENCHMARK_NOTES.md`
8. `.agents/memory/07_LEGACY_DIFF_NOTES.md`

Critical settled facts:

- Do not blame detector weight for `Pigs291119_000263_30fps` IDSW regression.
- The user confirmed that both old and new weights produce IDSW ≈ 6 on current code for `000263`.
- Legacy 21/06 produced IDSW ≈ 2 for `000263`.
- Therefore focus on code/config/runtime behavior differences between legacy
  21/06 and current `hybrid_bytetrack`.
- `Pigs291119_000302_30fps` improved mainly because of the new detector weight.
- Current preferred baseline is `hybrid_bytetrack + iou0_area0_condarea0_merge0`.
- Do not enable `condarea` by default without explicit ablation.
- Primary suspect for `000263`: `association.py` raw_id owner/penalty/bypass
  and `all_detection_indices` matching for `hybrid_bytetrack`.
- Secondary suspect: forced post-processing in `runner.py` for `hybrid_bytetrack`.

Rules:

- When asked to audit, do not modify code.
- When asked to patch, keep the patch small and reversible.
- Do not run long tracking/evaluation/benchmark unless the user explicitly requests it.
- Always report which memory files were read before making changes.
- Keep code lines within the repository formatter/linter limit before commit.
  Wrap long conditions, strings, comprehensions, function calls, and argument
  lists proactively. Before every commit that changes code, run a changed-file
  overlong-line scan, for example `rg -n "^.{101,}$" <changed-files>`, and fix
  any matches before `git commit` so pre-commit does not fail on line length.

File edit safety:

- For manual file edits, use `apply_patch` with small, reviewable hunks. Avoid
  shell redirects, heredocs, here-strings, `cat`, or ad hoc scripts to write
  source/config/docs unless generating a mechanical artifact is unavoidable.
  After editing, inspect `git diff -- <file>` before staging or committing.
- For Markdown edits, first identify a stable heading or nearby anchor and patch
  only that section. Do not delete/recreate `.md` files, append with shell
  redirection, or overwrite from a temporary file. If a patch hunk fails, re-read
  the nearby lines, retry a smaller hunk, then run `git diff --check` and an
  overlong-line scan before staging.
- Markdown append/update failure prevention protocol:
  1. Re-read the exact target section immediately before editing.
  2. Patch under a stable heading or insert one dated section near the top.
  3. Keep each hunk scoped to one section and fewer than about 40 changed lines.
  4. Never append with `>>`, `Set-Content`, `Add-Content`, heredoc,
     here-string, `cat`, or a temporary overwrite file.
  5. If context matching fails, stop, re-read 20-40 nearby lines, and retry with
     a smaller hunk. Do not switch to shell-writing as a fallback.
  6. After the patch, run `git diff -- <file>`, `git diff --check`, and
     `rg -n "^.{101,}$" <file>` before staging.

Legacy preserved docs:

- `.agents/AGENTS.md`
- `.agents/PROJECT_MEMORY.md`
- `.agents/WORKFLOW.md`
