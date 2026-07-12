# Workflow

## Active classification_v2 Workflow Override

Use this section as the current workflow. Older tracking/RGB-D/FastAPI notes in
this file are historical unless the user explicitly switches workstreams.

Current state:

- `classification_v2` behavior recognition is the active project priority.
- The project is pre-full ready, not Q2 complete.
- `q2_progress_report.json` reports `PASS_PARTIAL_ROADMAP`.
- The latest pre-full gate summary is 44/44 gates passing.
- Full OOF remains fail-closed until explicit authorization is written.
- Q2 claim remains locked until full OOF and postrun completion gate pass.

Command conventions:

1. Work from `C:\Users\ironh\Downloads\PIG_Behavior_Project`.
2. Use CMD semantics for project commands:
   `cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project`
3. Set `PYTHONPATH` before classification commands:
   `set PYTHONPATH=%CD%\src`
4. Prefer:
   `C:\Users\ironh\anaconda3\envs\pig_project\python.exe`
5. Use packed letterboxed actor and visual-context caches for full experiments.
6. Do not repeat seek/crop/resize frame loops when packed caches already exist.

Edit and commit rules:

1. Before edits, read root `AGENTS.md` plus:
   `01_PROJECT_MEMORY_SHORT.md`, `02_CURRENT_DECISION.md`,
   `03_PROJECT_RULES.md`, and `08_WORKFLOW.md`.
2. Use `apply_patch` for source/config/docs edits.
3. Avoid redirects, heredocs, here-strings, `cat`, or ad hoc write scripts for
   manual edits.
4. For Markdown memory/workflow files, do not delete/recreate the file first.
   Read exact nearby text, patch a small hunk, and preserve historical content
   unless the user explicitly asks to remove it.
5. If an `apply_patch` hunk fails, re-read the nearby lines and retry with a
   smaller context-matched patch instead of shell-writing the file.
6. For every Markdown edit, choose a stable heading or nearby anchor before
   patching. If the anchor is missing, add a small dated section near the top;
   do not use redirects, temporary overwrite files, or delete-add rewrites.
7. After a Markdown edit, inspect `git diff -- <file>`, run `git diff --check`,
   and scan changed `.md` files for overlong lines before staging.
8. Wrap long Markdown command lines with CMD continuation `^`.

Pre-full refresh sequence after any commit:

```bat
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src
set PY=C:\Users\ironh\anaconda3\envs\pig_project\python.exe
%PY% scripts\dev_tools\preflight_classification_v2_full_multimodal_oof.py ^
  --snapshot-json outputs\classification_v2\training_snapshots\c2v2_27ed5c9963904c52.json ^
  --runtime-benchmark-audit-json ^
  outputs\classification_v2\model_benchmarks_visual_v3\summary_head\runtime_benchmark_audit.json
%PY% scripts\dev_tools\write_classification_v2_full_oof_authorization_template.py
%PY% scripts\dev_tools\write_classification_v2_full_oof_authorization_file.py
%PY% scripts\dev_tools\check_classification_v2_full_oof_authorization_template.py
%PY% scripts\dev_tools\check_classification_v2_full_oof_authorization_file.py
%PY% scripts\dev_tools\check_classification_v2_full_oof_authorization_writer.py
%PY% scripts\dev_tools\check_classification_v2_full_oof_preflight_freshness.py
%PY% scripts\dev_tools\write_classification_v2_full_oof_launch_packet.py
%PY% scripts\dev_tools\check_classification_v2_full_oof_launch_packet.py
%PY% scripts\dev_tools\check_classification_v2_full_oof_execution_gate.py
%PY% scripts\dev_tools\check_classification_v2_full_oof_completion_gate.py
%PY% scripts\dev_tools\write_classification_v2_full_oof_postrun_registration_packet.py
%PY% scripts\dev_tools\check_classification_v2_full_oof_postrun_registration_packet.py
%PY% scripts\behavior_review_tools\classification_v2_write_q2_progress_report.py
%PY% scripts\dev_tools\check_classification_v2_q2_progress_report.py
```

Full OOF authorization rule:

- Do not run full OOF until `full_oof_authorization.json` has
  `authorized=true`.
- Require `acknowledges_long_run=true`.
- Require `acknowledges_no_q2_claim_until_verified=true`.
- Require non-empty `reviewer`.
- Require matching preflight config SHA256 and git commit.
- Require `check_classification_v2_full_oof_execution_gate.py` to allow
  execution.

Use these generated files as the source of truth:

```text
outputs/classification_v2/model_design/full_oof_launch_packet.md
outputs/classification_v2/model_design/full_oof_launch_packet.json
outputs/classification_v2/model_design/full_oof_authorization.json
outputs/classification_v2/model_design/full_oof_postrun_registration_packet.md
outputs/classification_v2/model_design/full_oof_postrun_registration_packet.json
```

Post-full required order:

1. Cross-fit calibration.
2. Confusion-focus comparison.
3. Ablation report refresh.
4. Experiment registry registration.
5. Completion gate.
6. Q2 progress report refresh.

Only after the completion gate reports `q2_claim_allowed=true` may the result
be described as a Q2 internal improvement candidate.

Memory refresh after full OOF:

1. Update `01_PROJECT_MEMORY_SHORT.md` with final PASS/FAIL, key metrics,
   output paths, and claim boundary.
2. Update `02_CURRENT_DECISION.md` with accepted result decision and blockers.
3. Update `06_BENCHMARK_NOTES.md` with final OOF/control metrics and confusion
   findings.
4. Update `08_WORKFLOW.md` only if launch/postrun command sequence changed.
5. Commit the memory refresh separately.

## Preserved Historical Workflow

## Current classification_v2 Q2 workflow

Use this workflow when continuing the behavior-recognition roadmap.

1. Work from project root:
   `C:\Users\ironh\Downloads\PIG_Behavior_Project`
2. For CMD execution, set:
   `set PYTHONPATH=%CD%\src`
3. Prefer Python:
   `C:\Users\ironh\anaconda3\envs\pig_project\python.exe`
4. Before any code edit, read root `AGENTS.md` and memory files
   `01_PROJECT_MEMORY_SHORT.md`, `02_CURRENT_DECISION.md`,
   `03_PROJECT_RULES.md`, and `08_WORKFLOW.md`.
5. Use `apply_patch` for source/config/docs edits. Do not write source files
   with redirects, heredocs, here-strings, `cat`, or ad hoc scripts.
6. Before every code commit, run an overlong-line scan on changed code files,
   for example `rg -n "^.{101,}$" <changed-files>`, and run
   `git diff --check`.

### classification_v2 pre-full gates

Run these checks before any full OOF launch:

```bat
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src
set PY=C:\Users\ironh\anaconda3\envs\pig_project\python.exe
%PY% scripts\dev_tools\preflight_classification_v2_full_multimodal_oof.py ^
  --snapshot-json outputs\classification_v2\training_snapshots\c2v2_27ed5c9963904c52.json ^
  --runtime-benchmark-audit-json ^
  outputs\classification_v2\model_benchmarks_visual_v3\summary_head\runtime_benchmark_audit.json
%PY% scripts\dev_tools\write_classification_v2_full_oof_authorization_template.py
%PY% scripts\dev_tools\write_classification_v2_full_oof_authorization_file.py
%PY% scripts\dev_tools\check_classification_v2_full_oof_authorization_template.py
%PY% scripts\dev_tools\check_classification_v2_full_oof_authorization_file.py
%PY% scripts\dev_tools\check_classification_v2_full_oof_authorization_writer.py
%PY% scripts\dev_tools\check_classification_v2_full_oof_preflight_freshness.py
%PY% scripts\dev_tools\write_classification_v2_full_oof_launch_packet.py
%PY% scripts\dev_tools\check_classification_v2_full_oof_launch_packet.py
%PY% scripts\dev_tools\check_classification_v2_full_oof_execution_gate.py
%PY% scripts\dev_tools\check_classification_v2_full_oof_completion_gate.py
%PY% scripts\dev_tools\write_classification_v2_full_oof_postrun_registration_packet.py
%PY% scripts\dev_tools\check_classification_v2_full_oof_postrun_registration_packet.py
%PY% scripts\behavior_review_tools\classification_v2_write_q2_progress_report.py
%PY% scripts\dev_tools\check_classification_v2_q2_progress_report.py
```

The expected pre-full status is `PASS_PARTIAL_ROADMAP` with fail-closed full
OOF authorization. That is a ready-for-human-authorization state, not a
completed Q2 result.

### Full OOF authorization rule

Do not run the full OOF command until:

- `full_oof_authorization.json` has `authorized=true`.
- `acknowledges_long_run=true`.
- `acknowledges_no_q2_claim_until_verified=true`.
- reviewer is non-empty.
- preflight config hash and git commit match the current clean preflight.

After full OOF completes, run the postrun commands from
`outputs/classification_v2/model_design/full_oof_postrun_registration_packet.json`
in order: calibration, confusion focus comparison, ablation report refresh,
registry registration, and completion gate.

### Full OOF launch packet

Use the generated launch packet as the single source of truth:

```text
outputs/classification_v2/model_design/full_oof_launch_packet.md
outputs/classification_v2/model_design/full_oof_launch_packet.json
```

The current packet is ready for human authorization review and targets:

```text
outputs/classification_v2/model_full/full_multimodal_oof
```

The full run must use cached letterboxed actor images and packed visual context.
Do not run ad hoc full loops that repeatedly seek, crop, resize, and convert
frames when the packed caches are available.

### Post-full memory refresh

After a successful full OOF and postrun completion gate:

1. Update `01_PROJECT_MEMORY_SHORT.md` with the final PASS/FAIL state, key
   metrics, output paths, and claim boundary.
2. Update `02_CURRENT_DECISION.md` with the accepted result decision and any
   remaining blockers.
3. Update `06_BENCHMARK_NOTES.md` with final OOF/control metrics and confusion
   findings.
4. Update `08_WORKFLOW.md` if the launch or postrun command sequence changed.
5. Commit the memory refresh separately from code changes.

## Before every coding task

1. Read root `AGENTS.md`.
2. Read `.agents/memory/01_PROJECT_MEMORY_SHORT.md`.
3. Read `.agents/memory/02_CURRENT_DECISION.md`.
4. Read `.agents/memory/03_PROJECT_RULES.md`.
5. State which memory files were read.

## Audit task

If the user asks to audit/check/diff:

- Do not modify code.
- Do not run tracking/evaluation/inference.
- Use read-only commands only.
- Report findings with file/function/behavior/risk.

## Patch task

If the user asks to patch:

- Modify only requested scope.
- Prefer one small patch at a time.
- State exact files changed.
- State behavior changed.
- State behavior intentionally not changed.
- Do not run long benchmarks unless requested.
- Before every commit that changes code, scan changed files for overlong lines:
  `rg -n "^.{101,}$" <changed-files>`.
- Wrap long conditions, strings, comprehensions, function calls, and argument
  lists proactively so formatter/linter hooks do not fail the commit.

File-write safety:

- For source/config/docs edits, use `apply_patch` with small hunks.
- Do not use shell redirects, heredocs, here-strings, `cat`, or ad hoc write
  scripts unless the file is a generated artifact.
- Inspect `git diff -- <file>` after editing and fix overlong lines before
  staging.

## Verification task

When user allows verification, run in this order:

1. Static/syntax/import check.
2. Single video `Pigs291119_000263_30fps`.
3. Single video `Pigs291119_000302_30fps`.
4. 3-video common set.
5. 7-video full set.

## Preserved legacy workflow notes from previous `.agents/WORKFLOW.md`

- ROI definitions live in `data/annotations/roi/ROI_annotations.coco.json`
  with related scene background and mask assets.
- CVAT XML annotations are processed into classification datasets and feature tables.
- Detection and tracking previously centered around `pig-track-for-annotation`
  workflows with GPU fallback to CPU.
- RGB-D occlusion handling used depth calibration files such as
  `depth_scale.npy`, `inverse_intrinsic.npy`, and `rot.npy`.
- Behavior training, export, inference, and FastAPI dashboard flows remain
  documented in `.agents/WORKFLOW.md`.
- CI/quality gates previously documented there remain preserved as legacy workflow guidance.

## 2026-07-12 classification_v2 current override

- Treat `classification_v2` behavior recognition as the active priority unless
  the user explicitly switches back to tracking.
- Pre-full readiness currently means `PASS_PARTIAL_ROADMAP` with 44/44 gates
  passing in the latest refreshed pre-full state, not a completed Q2 result.
- Full OOF remains fail-closed until `full_oof_authorization.json` is explicitly
  authorized with reviewer, long-run acknowledgement, no-Q2-claim
  acknowledgement, matching preflight config hash, and matching git commit.
- Before any full run, refresh the preflight, authorization template/file
  checks, authorization writer check, preflight freshness check, launch packet
  check, execution gate, completion gate, postrun packet check, and Q2 progress
  report.
- After any commit, refresh the same sequence again before requesting human full
  OOF authorization so preflight freshness points at current HEAD.
- Do not launch full OOF unless
  `check_classification_v2_full_oof_execution_gate.py` allows execution.
- After full OOF, run calibration, confusion-focus comparison, ablation report
  refresh, experiment registry registration, and completion gate before making
  any Q2 result claim.
