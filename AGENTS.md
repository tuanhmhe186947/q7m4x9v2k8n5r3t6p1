# Agent Instructions for PIG_Behavior_Project

## CLOUD EXECUTION CONTRACT (PERMANENT)

For `pig-project` / `training-pig-project-L4`:

1. Raw videos, loose crops, and bulk preprocessing stay LOCAL.
2. Train-ready bulk tensors stay in Teamspace Drive, never in
   `/teamspace/studios/this_studio`.
3. Studio contains runtime code, environment, configs, and small logs only.
4. Never upload `.git`, `.codex*`, tracking projects, notebooks, historical
   outputs, raw videos, loose crops, or packed datasets into `this_studio`.
5. Debug on CPU before switching the same Studio to GPU; fix small errors in
   place and do not stop/start for them.
6. No arbitrary startup timeout and no unconditional `finally: studio.stop()`.
7. Report progress only from live logs and optimizer-step evidence; never
   simulate a scientific result.
8. Optimize wall-clock time to a valid result.
9. Deleted `pig-gpu-l4-gcp` is never inspected, started, restored, or recreated.
10. The current R128 Drive checkpoint is already PASS at
    `/teamspace/uploads/classification_v2/cloud_r128_recovery_20260817_gcp/r128_cache`.
    If its recorded evidence remains valid, do not re-upload, re-hash, or
    rediscover it.
11. R64 three-seed results and full-T6 data are existing inputs; do not rerun
    R64, rebuild full-T6, or repeat a passed CPU preflight.
12. The active recovery runs only T6/R128 seeds `20260804`, `20260805`, and
    `20260806`, 4164 steps each, on the same authorized Studio. Inspect the
    active task/permit/worktree once, then resume from its checkpoint.

## MAIN-ONLY RESUME CONTRACT (2026-08-17)

- `C:\Users\ironh\Downloads\PIG_Behavior_Project` on local `main` is the
  sole source of truth for this recovery. Commit verified source, config, and
  memory corrections on `main` before any later cloud action.
- Do not create, select, or resume from a classification worktree for this
  recovery. Existing worktree task records are historical/protected state, not
  a reason to move execution away from `main`.
- A passed stage is a completed stage: do not repeat Drive upload/hash checks,
  CPU preflight, R64, full-T6 preparation, or the already completed
  `T6/R128/20260804` trial. First inspect its checkpoint and copy it to Drive
  if durable publication is still pending.
- Continue only `T6/R128` seeds `20260805` and `20260806`, sequentially, with
  the existing runtime, Drive cache, and 4164-step contract. Persist and verify
  each result before starting the next seed.

## V2 governance activation

New material tasks start with `.agents/memory/00_AGENT_BOOTSTRAP.md` and the
manager `.agents/skills/project-state-steward/scripts/manage_agent_governance.py`
and follow
`bootstrap -> create -> confirm-plan -> record-skill-read -> permit ->
advance/amend -> review-outcome -> close`. V1 remains the compatibility path
for existing V1 capsules, including this reform capsule until migration closes.
The canonical skill authority is `.agents/skills/skill_inventory.json`; the
worktree lifecycle ledger is `.agents/memory/22_WORKTREE_LIFECYCLE.json` and
generated views are checked by the governance validator. Simple read-only questions do
not create a task; fixture-only passes, apologies, clean worktrees, and
unmerged branches are not completion evidence.

## Progressive delivery and worktree use

- STRICT PROHIBITION ON WORKTREE CREATION WITHOUT EXPLICIT USER REQUEST:
  Under no circumstances may an agent create a new git worktree (`git worktree add`),
  isolated worktree directory (e.g. in `.codex_worktrees/`), or separate branch workspace
  on its own initiative.
- All development, edits, tests, and commits MUST strictly execute in the primary
  workspace root (`c:\Users\ironh\Downloads\PIG_Behavior_Project`) directly on `main`
  (`shared_main` mode).
- Creating a new worktree is PERMITTED ONLY when the USER EXPLICITLY COMMANDS it
  in their prompt (e.g. "tạo worktree cho ...", "create a worktree for ...").
- `main` is the continuous delivery branch. For all work, use `shared_main` mode
  and commit each verified bounded milestone to local `main` immediately; that commit
  is the integration for the milestone.
- do not accumulate a queue of completed worktree changes or unmerged branches.

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

For `classification_v2` data/review work, also read:

9. `.agents/memory/09_HIDDEN_REVIEW.md`

For skill selection or maintenance work, also read:

10. `.agents/memory/11_SKILL_PORTFOLIO.md`

For project direction, method promotion, scientific claims, agent governance,
or halt/permission decisions, also read:

11. `.agents/memory/12_PROJECT_CHARTER.md`
12. `.agents/memory/13_METHOD_STATE.md`
13. `.agents/memory/14_CLAIM_REGISTRY.md`
14. `.agents/memory/15_AGENT_REGRESSION.md`
15. `.agents/memory/16_HALT_CONDITIONS.md`
16. `.agents/memory/18_AUTHORITY_INDEX.md`
17. `.agents/memory/19_REASONING_ROUTING.md`
18. `.agents/memory/21_MEMORY_MATURITY.md`

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

- ABSOLUTE PROHIBITION ON FABRICATING OR SIMULATING EXPERIMENTAL OUTCOMES:
  Under no circumstances may an agent generate synthetic numbers, use random
  number generators (e.g. `np.random`), heuristics, or dummy tensor loops to
  mock or simulate experimental training/evaluation results and present them as
  measured metrics. If an experiment, trial, or metric has not been genuinely
  executed end-to-end on real data with real weights and saved checkpoints, the
  agent MUST report `NOT_YET_EXECUTED` or `NOT_MEASURED`. Presenting simulated
  values as real evidence is a catastrophic breach of scientific integrity.
- ABSOLUTE PROHIBITION ON SPECULATING OR REPORTING UNVERIFIED TRIAL STATUS:
  Under no circumstances may an agent report, claim, or speculate that a training
  trial or evaluation is 'running', 'in progress', or 'almost done' unless direct
  terminal log output explicitly demonstrating active device allocation and step
  advancement (e.g. `Step X/Y`, `Peak VRAM`, `CUDA Device: ...`) has been fetched
  and verified in the current turn. Reporting speculative, optimistic, or ungrounded
  status without direct log proof is strictly prohibited.
- ABSOLUTE PROHIBITION ON RUNNING UN-CACHED VIDEO DECODING ON PAID CLOUD GPU:
  Under no circumstances may an agent boot or run a paid cloud GPU instance
  to perform un-cached raw MP4 frame decoding or CPU-bound crop extraction
  loops. All video frame extraction, cropping, and dataset caching MUST be
  executed, validated, and packaged locally on the local machine / local GPU
  first. Paid cloud GPU compute is strictly reserved for high-throughput
  tensor model training and inference on pre-cached, pre-verified dataset
  artifacts. Whenever an un-cached run is identified or cloud execution ends,
  the remote Studio must be verified STOPPED immediately.
- When asked to audit, do not modify code.
- When asked to patch, keep the patch small and reversible.
- Before executing a user issue, prompt, or requested workflow, inspect the
  relevant local authority, source, config, data contract, and current state.
  Resolve discoverable technical facts with read-only checks first. If the
  intended outcome, scope, terminology, authority, source lineage, acceptance
  criteria, or a material design choice is still not fully clear, stop before
  edits, runs, or external effects and ask the user concise clarifying questions.
  State what is known, exactly what remains ambiguous, and why the answer changes
  the implementation or result. Do not silently choose a direction or continue
  until the user answers. Do not ask about facts that can be safely discovered
  from authoritative project sources.
- The user grants standing approval for project-local Markdown edits. Treat
  Markdown confirmation as "Yes, and don't ask again for these files"; do not
  request confirmation solely to create or modify a `.md` file in this
  workspace. This does not override sandbox boundaries or the edit-safety rules
  below.
- Before evaluation, benchmark, ablation, or nontrivial implementation, inspect
  the available skill catalog and record the selected skills in the working
  plan and, when applicable, the run manifest.
- Use `find-skills` only for a demonstrated catalog gap. Use `skill-creator` to
  create or upgrade a reusable project-local skill, validate it before relying
  on it, and commit skill changes separately from algorithm changes.
- After any implementation session that changes source, config, tests, data
  contracts, generated artifacts, or establishes a durable correction, invoke
  `project-state-steward` before the final response without waiting for the user
  to request it. Reconcile memory and workflow, audit cleanup candidates, and
  preserve unknown or user-owned files. Do not force a memory edit when the
  session produced no durable knowledge.
- Treat `01_PROJECT_MEMORY_SHORT.md` as daily state plus bounded resume
  capsules for active managed tasks. On first read after `Expires`, run
  `manage_short_memory.py rollover`: remove terminal task bodies, retain each
  nonterminal managed block byte-for-byte, and reset only daily state. Do not
  carry project history in short memory.
- When a prompt needs a multi-step plan, file edits, execution, or an external
  effect, use `manage_short_memory.py create` before the first effect. Retain
  the returned owner token only in that live session. Use `inspect` plus
  revision/hash CAS for `checkpoint`, `renew`, audited-drift `reconcile`, or
  expired-lease `takeover`. If the token was lost after a crash, `recover` may
  rotate it during an active lease only when the task is already bound to the
  current `CODEX_THREAD_ID`. Otherwise require user-authorized
  `admin-takeover` with the exact task ID, revision, hash, worktree,
  confirmation phrase, reason, and authorization reference.
  Never edit a managed task block manually. Update only at phase boundaries;
  checkpoint `DONE` before the next step's first effect and attach evidence.
- Checklist steps use `TODO`, `IN_PROGRESS`, `BLOCKED`, `DONE`, `DEFERRED`, or
  `CANCELLED`. Keep at most one `IN_PROGRESS` step per task. A `DONE` step
  requires concrete evidence; every unfinished step requires a next action.
  Simple read-only questions do not create checklist noise.
- On restart, treat a surviving `IN_PROGRESS` as interrupted `IN_PROGRESS`
  work. Inspect its declared evidence and side effects before changing state:
  mark it `DONE` if acceptance is already proven, otherwise resume from the
  smallest safe checkpoint or halt if repeating the effect is unsafe. Never
  rerun it blindly or infer completion from chat history alone.
- In concurrent sessions, create unique task IDs through the atomic manager.
  The OS lock, owner token, worktree binding, lease, revision, and block hash
  are mandatory. Halt on collisions, CAS drift, unaudited active-lease
  takeover, runtime-thread mismatch, or any non-owned block change; never
  bypass the manager with a Markdown patch. Every credential recovery or
  ownership transition must append a hash-bound audit event.
- At rollover, keep active managed tasks in `01` as resume capsules so `DONE`
  steps and the exact next action survive across days. Route only explicitly
  paused or dormant work to `04`; never keep two current copies of one task.
  Retain completed outcomes for one day and purge older closeout history.
- Admit only stable accepted project facts to `05`; transient blockers,
  hypotheses, and unfinished work belong in `04`.
- A learned correction requires root cause, validated correction, evidence,
  reuse conditions, and non-reuse boundaries. Remembering an error without its
  validated corrective method does not count as learning.
- Do not default to code skills alone. For nontrivial architecture, behavior,
  debugging, evaluation, synthesis, or data-contract work, select at least one
  reasoning skill; a verification skill alone does not satisfy this gate. Track
  the choice in
  `.agents/memory/11_SKILL_PORTFOLIO.md`.
- Resolve current authority through `18_AUTHORITY_INDEX.json`. If two sources
  claim current authority for one scope, halt before effects. Use
  `19_REASONING_ROUTING.md` for mandatory task-to-reasoning-skill mapping.
- Resolve medium-to-long promotion through `21_MEMORY_MATURITY.json` and
  `manage_memory_maturity.py`. Elapsed time, inactivity, and task completion
  are review triggers only, never promotion evidence. Run `scan` at material
  closeout; use deliberate `review`, `promote`, `reopen`, and `synthesize`
  transitions instead of manually editing the generated dossier in file `05`.
- For tracking experiments, use `tracking-experiment-guardian` and obey its
  lineage, guardrail, promotion, and no-MP4 gates.
- Do not run long tracking/evaluation/benchmark unless the user explicitly requests it.
- For the active `classification_v2` goal, the user grants standing permission
  for a necessary full data or model run after the exact semantic configuration
  passes static/synthetic checks, a short representative run, and all declared
  audits. Do not ask again solely because the run is long. Repeat the short gate
  after any semantic change; full OOF still requires its technical launch gate.
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

- Markdown failure-stop rule:
  1. Treat `.md` files as hand-edited project memory, not generated output.
  2. If two `apply_patch` attempts fail for the same Markdown target, stop and
     re-read the exact file section before trying again.
  3. Do not recover from a failed Markdown patch by using PowerShell writers,
     shell redirects, temporary files, or whole-file replacement.
  4. For append-like changes, insert under an existing heading or add one small
     dated heading near the top with `apply_patch`.
  5. If the target location is ambiguous after re-reading, ask the user or
     report the ambiguity instead of guessing with a broad rewrite.

Legacy preserved docs:

- `.agents/AGENTS.md`
- `.agents/PROJECT_MEMORY.md`
- `.agents/WORKFLOW.md`
