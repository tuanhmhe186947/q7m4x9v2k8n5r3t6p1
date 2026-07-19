---
name: andrej-karpathy-skills
description: >-
  Behavioral guidelines to reduce common LLM coding mistakes, tailored for
  Codex in PIG_Behavior_Project. Helps maintain simplicity, state assumptions,
  make surgical changes, and perform goal-driven execution.
---

# Karpathy Guidelines for Codex

## Purpose

Apply Andrej Karpathy's behavioral guidelines to minimize common LLM coding
mistakes, ensuring high-quality, simple, surgical, and goal-driven
implementation in the `PIG_Behavior_Project` codebase.

## When to use

Invoke this skill at the beginning of any coding task, before planning,
refactoring, or modifying existing code, files, configs, or scripts.

## Project context

This project has strict quality gates, scientific invariants, review
requirements (Hidden and behavior reviews), and execution rules. The Karpathy
guidelines provide a strong foundation to respect these rules by ensuring
Codex is cautious, surgical, and clear.

## Required inputs

- Exact user request, goal, or problem statement.
- Current project state (e.g., active worktree, branch, memory files).
- Target files, schemas, and dependencies.

## Scientific invariants (The 4 Core Principles)

### 1. Think Before Coding (Don't assume, surface tradeoffs)
- State assumptions explicitly. If uncertain, ask the user.
- If multiple interpretations exist, present them instead of picking silently.
- Push back if a simpler approach exists.
- Stop and clarify confusion immediately.

### 2. Simplicity First (Minimum viable code)
- Implement only requested features; avoid speculative abstractions.
- Avoid configuring or generalizing single-use code unnecessarily.
- Avoid writing overly defensive error handling for impossible scenarios.
- Keep implementation as concise as possible (e.g., 50 lines instead of 200).

### 3. Surgical Changes (Precise editing, style matching)
- Touch only what is necessary to fulfill the request.
- Do not make "drive-by" changes to adjacent formatting or comments.
- Match existing style exactly, even if you would do it differently.
- Identify dead code but do not delete it unless requested.
- Clean up orphans (unused imports/vars) created by your changes, but do
  not touch pre-existing dead code.

### 4. Goal-Driven Execution (Verifiable success criteria)
- Transform tasks into outcome-based, verifiable goals with verification
  steps.
- Set up tests, checks, or canaries to verify correctness before and after.
- Loop independently until verification passes.

## Ordered procedure

1. **Understand & Read**: Read root `AGENTS.md` and the four mandatory memory
   files (`01_PROJECT_MEMORY_SHORT.md`, `02_CURRENT_DECISION.md`,
   `03_PROJECT_RULES.md`, and `08_WORKFLOW.md`). Report which memory files
   were read.
2. **Clarify & Surface**: Analyze the task. Identify and explicitly state
   assumptions, ambiguities, or tradeoffs.
3. **Plan**: Formulate a minimal, surgical change plan with verifiable goals.
4. **Execute**: Modify the target files using `apply_patch`.
5. **Verify**: Run compilation checks, focused tests, and the project's
   linter/linter line-length scan (`rg -n "^.{101,}$"`).
6. **Report**: Summarize the changes, verify what was changed and what was
   intentionally not changed, and outline any risks.

## Required outputs

- Written confirmation of memory files read.
- Surgical patches applied to codebase.
- Clear verification/test results.
- Brief summary of changes, unchanged behavior, and risks.

## Validation commands

- Python compile check: `python -m py_compile <changed-files>`
- Overlong line check: `rg -n "^.{101,}$" <changed-files>`
- Pytest verification: `pytest -q` or focused test commands.

## Stop conditions

- Encountering code where the implementation path has multiple major
  interpretations (stop and ask the user).
- Overlap with unrelated worktree changes or uncommitted work.
- Compilation or unit test failure.
- Line length exceeding 100 characters in modified files.

## Forbidden actions

- Making drive-by improvements or style refactors to adjacent, unrelated
  files/functions.
- Deleting pre-existing dead code without permission.
- Running long tracking/evaluation/benchmark runs unless explicitly
  requested.
- Appending markdown or editing files using shell redirects (`>>`, `cat`).

## Completion report format

Format output directly to the user in a concise, structured markdown report,
listing changes, verification results, and highlighting key decisions.
