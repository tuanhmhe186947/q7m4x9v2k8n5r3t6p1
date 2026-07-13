# Workflow

## Reviewed-data rebuild gate

For a new `classification_v2` data lineage, follow
`docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`. Full runs are
authorized only after the same semantic config passes static checks, a short
legacy+CVAT chain, and schema/count/hash/output/runtime audits.

Required frame-data order:

```text
enhanced frame features
  -> two-sided Hidden manifest and media gate
  -> human Hidden decisions and fail-closed coverage
  -> hidden_reviewed_frame_features.csv
  -> temporal harmonization and sequence windows
  -> behavior review units and behavior decision apply
```

CVAT Hidden is tracking-derived/untrusted before review. The Hidden GUI must
show full-frame context, write decision CSV only, and apply each decision to
its declared frame/object key. Do not use the legacy GUI that writes corrected
source copies for a new lineage.

Detailed settled Hidden policy and validation evidence are in
`.agents/memory/09_HIDDEN_REVIEW.md`.

The current versioned full Hidden template is
`outputs/classification_v2/rebuilds/hidden_review_v5_full_20260713`. It contains
5,171 unique items and passed independent coverage. Continue with GUI human
review and fail-closed decision coverage; do not treat the template alone as a
reviewed artifact.

The current identifier-v2 short chain is independently verified under
`outputs/classification_v2/rebuilds/scientific_smoke_identifier_v2_20260713`.
After changing source/features/temporal/image/train-ready ordering, rebuild a
new bounded root and run both lineage and consolidated gates:

```bat
set S9=scripts\classification_v2\09_final_release_audit
set BASE=outputs\classification_v2\rebuilds
set ROOT=%BASE%\scientific_smoke_identifier_v2_20260713
set REPEAT=%BASE%\scientific_smoke_identifier_v2_repeat_20260713
%PY% %S9%\check_classification_v2_identifier_v2_lineage.py ^
  --root %ROOT% ^
  --repeat-root %REPEAT% ^
  --overwrite
%PY% %S9%\check_classification_v2_technical_smoke_gate.py ^
  --root %ROOT% ^
  --repeat-root %REPEAT% ^
  --overwrite
```

Expected statuses are `PASS_IDENTIFIER_V2_TECHNICAL_HUMAN_REVIEW_BLOCKED` and
`PASS_TECHNICAL_SMOKE_HUMAN_REVIEW_BLOCKED`, never training authorization.
Use both sources and all 10 behaviors. Builders must exit nonzero on audit
errors. Existing derived outputs require explicit `--overwrite`; prefer a new
versioned directory for changed semantics.

The temporal-view code contract is `PASS IN CODE` at `bb225ff`. After reviewed
windows exist, build `fixed6_observed_time`, `fixed6_normalized_phase`, and
`native6_16` with the block `02` temporal-view builder, then run its structural
shortcut checker. The fixed-six view reuses existing harmonized six-frame
windows; never sample six quantiles across a legacy burst. Keep all original
windows in the selection ledger and keep source/native-length metadata outside
model tensors. An unmitigated source shortcut is a training hard stop.

Training-contract code now uses fold-local preprocessing, native-event mass
weighting, and immutable lineage. A requested `output_dir` is an output root,
not the artifact directory. The trainer owns this exact layout:

```text
output_root\fold_id\run_id
```

Downstream Python callers must use `training_run_dir(audit)`. Check a completed
packet with block `04` `check_classification_v2_run_lineage.py`. Independent
remote fold rows are merged only through block `06`
`classification_v2_merge_run_registry.py`; never concatenate or overwrite the
central registry manually. These contracts pass at `16cdb93`, but no real run
is allowed before the reviewed snapshot and smoke gates pass.

For a reviewed full-multimodal candidate, rerun the lineage checker with
`--require-interaction-lineage`. Snapshot v2 must show one ordered hash for
split, image-window, and interaction-window manifests; exporter audits must
match the same hash. Full preflight additionally requires an explicit
`--lineage-audit-json` and binds snapshot, lineage, ordered-window, config, and
code hashes. Bounded technical audits keep training authorization false and
must therefore be rejected by this preflight.

The current canonical reviewed artifact is not human-review complete. Complete
all mandatory review units, pass the fail-closed decision-coverage audit, then
rebuild reviewed windows with `--disable-fast-reuse`. Use recording-date or
validated session groups; never random-split frames or overlapping windows.

Do not launch model training from a new rebuild until its versioned
data/cache/fold hashes are frozen and all local model smoke gates pass.

## Active classification_v2 Workflow Override

Use this section as the current workflow. Older tracking/RGB-D/FastAPI notes in
this file are historical unless the user explicitly switches workstreams.

Current state:

- `classification_v2` behavior recognition is the active project priority.
- Status authority is `docs/CLASSIFICATION_V2_CURRENT_STATE.md`.
- The identifier-v2 technical chain passes at commit `a83d5a5`: 688 frame rows,
  63 native/review units, 438 ordered windows, exact X whitelist, and 8/8
  source-to-window repeatability.
- Temporal-view manifests and structural shortcut checks pass 22 synthetic
  tests at `bb225ff`; no active reviewed packet has been built from them yet.
- Fold-local preprocessing, native-event weighting, and immutable run lineage
  pass at `97f83c5`, `73b901d`, and `16cdb93`.
- The mask-safe factory at `318bf58` exposes ten exact model modes and four
  temporal encoders. Current classification regression is 319 passed and 181
  deselected; this is fixture evidence, not training authorization.
- Transformer training remains blocked until strict fixed-six loaders emit real
  `time_delta` tensors bound to ordered manifest hashes.
- The active lineage stops at block `01`: the Hidden v5 template passes, but
  human Hidden decisions are only 30/5,171 and apply is incomplete.
- Behavior review also fails closed with 3/4,670 decisions, 4,667 missing, and
  one pending.
- Do not rebuild train-ready exports, refresh model preflight, or launch model
  training until both review layers pass and versioned hashes are frozen.
- The 73,668-window/32,727-native-unit full OOF at commit `18d6692` had
  split-to-multimodal positional misalignment. Keep it only as historical
  compute/pipeline evidence, never as model-performance evidence.
- Use only the numbered script workflow under
  `scripts/classification_v2/00_*` through `09_*`.
- Q2 claim remains locked until a new reviewed-lineage full run and block `09`
  completion gate both pass.

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
9. For Markdown append/update work, follow this exact failure-prevention
   protocol:
   - Re-read the target section immediately before editing.
   - Patch under a stable heading or add one dated section near the top.
   - Keep each hunk scoped to one section and fewer than about 40 changed lines.
   - Never use `>>`, `Set-Content`, `Add-Content`, heredoc, here-string,
     `cat`, or temporary overwrite files for manual Markdown edits.
   - If a hunk fails, re-read 20-40 nearby lines and retry with a smaller hunk.
   - Verify with `git diff -- <file>`, `git diff --check`, and an overlong-line
     scan before staging.
10. Markdown failure-stop rule:
    - Treat `.md` files as hand-edited project memory, not generated output.
    - If two `apply_patch` attempts fail for the same Markdown target, stop and
      re-read the exact file section before trying again.
    - Do not recover from a failed Markdown patch by using PowerShell writers,
      shell redirects, temporary files, or whole-file replacement.
    - For append-like changes, insert under an existing heading or add one
      small dated heading near the top with `apply_patch`.
    - If the target location is ambiguous after re-reading, ask the user or
      report the ambiguity instead of guessing with a broad rewrite.

Future full-run refresh sequence after snapshot readiness:

Do not run this sequence now. It becomes active only after the reviewed data,
cache, whitelist, and fold hashes are frozen and all model smoke gates pass.

```bat
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src
set PY=C:\Users\ironh\anaconda3\envs\pig_project\python.exe
set S5=scripts\classification_v2\05_preflight_authorization
set S8=scripts\classification_v2\08_publication_reporting
set S9=scripts\classification_v2\09_final_release_audit
%PY% %S5%\preflight_classification_v2_full_multimodal_oof.py ^
  --snapshot-json outputs\classification_v2\training_snapshots\c2v2_27ed5c9963904c52.json ^
  --runtime-benchmark-audit-json ^
  outputs\classification_v2\model_benchmarks_visual_v3\summary_head\runtime_benchmark_audit.json
%PY% %S5%\write_classification_v2_full_oof_authorization_template.py
%PY% %S5%\write_classification_v2_full_oof_authorization_file.py
%PY% %S5%\check_classification_v2_full_oof_authorization_template.py
%PY% %S5%\check_classification_v2_full_oof_authorization_file.py
%PY% %S5%\check_classification_v2_full_oof_authorization_writer.py
%PY% %S5%\check_classification_v2_full_oof_preflight_freshness.py
%PY% %S5%\write_classification_v2_full_oof_launch_packet.py
%PY% %S5%\check_classification_v2_full_oof_launch_packet.py
%PY% %S5%\check_classification_v2_full_oof_execution_gate.py
%PY% %S5%\write_classification_v2_full_oof_postrun_registration_packet.py
%PY% %S5%\check_classification_v2_full_oof_postrun_registration_packet.py
%PY% %S8%\classification_v2_write_q2_progress_report.py
%PY% %S8%\check_classification_v2_q2_progress_report.py
%PY% %S9%\check_classification_v2_full_oof_completion_gate.py
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

## Historical classification_v2 pre-full workflow

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

### Historical classification_v2 pre-full gates

Run these checks before any full OOF launch:

```bat
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src
set PY=C:\Users\ironh\anaconda3\envs\pig_project\python.exe
set S5=scripts\classification_v2\05_preflight_authorization
set S8=scripts\classification_v2\08_publication_reporting
set S9=scripts\classification_v2\09_final_release_audit
%PY% %S5%\preflight_classification_v2_full_multimodal_oof.py ^
  --snapshot-json outputs\classification_v2\training_snapshots\c2v2_27ed5c9963904c52.json ^
  --runtime-benchmark-audit-json ^
  outputs\classification_v2\model_benchmarks_visual_v3\summary_head\runtime_benchmark_audit.json
%PY% %S5%\write_classification_v2_full_oof_authorization_template.py
%PY% %S5%\write_classification_v2_full_oof_authorization_file.py
%PY% %S5%\check_classification_v2_full_oof_authorization_template.py
%PY% %S5%\check_classification_v2_full_oof_authorization_file.py
%PY% %S5%\check_classification_v2_full_oof_authorization_writer.py
%PY% %S5%\check_classification_v2_full_oof_preflight_freshness.py
%PY% %S5%\write_classification_v2_full_oof_launch_packet.py
%PY% %S5%\check_classification_v2_full_oof_launch_packet.py
%PY% %S5%\check_classification_v2_full_oof_execution_gate.py
%PY% %S5%\write_classification_v2_full_oof_postrun_registration_packet.py
%PY% %S5%\check_classification_v2_full_oof_postrun_registration_packet.py
%PY% %S8%\classification_v2_write_q2_progress_report.py
%PY% %S8%\check_classification_v2_q2_progress_report.py
%PY% %S9%\check_classification_v2_full_oof_completion_gate.py
```

For that historical lineage, the expected pre-full status was
`PASS_PARTIAL_ROADMAP` with fail-closed full OOF authorization. It was not a
completed Q2 result and is not the expected status of the active rebuild.

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

The historical packet targeted:

```text
outputs/classification_v2/model_full/full_multimodal_oof
```

Do not reuse this path or packet for a future reviewed-lineage run.

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

## Historical 2026-07-12 classification_v2 override

This records the old pre-full contract and must not be executed for the active
reviewed-data rebuild.

- `classification_v2` behavior recognition became the active priority.
- That lineage reported `PASS_PARTIAL_ROADMAP` with 44/44 pre-full gates; this
  was not a completed Q2 result.
- Its full OOF remained fail-closed until `full_oof_authorization.json` was
  authorized with reviewer, acknowledgements, config hash, and git commit.
- Before that full run, the workflow refreshed the preflight, template/file
  checks, authorization writer check, preflight freshness check, launch packet
  check, execution gate, completion gate, postrun packet check, and Q2 progress
  report.
- After each commit, it refreshed the sequence before requesting authorization.
- It launched only when `check_classification_v2_full_oof_execution_gate.py`
  allowed execution.
- After full OOF, it required calibration, confusion comparison, ablation report
  refresh, experiment registry registration, and completion gate before making
  any Q2 result claim.
