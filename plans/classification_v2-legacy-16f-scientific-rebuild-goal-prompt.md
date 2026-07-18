# Classification V2 Legacy 16f Scientific Rebuild Goal Prompt

Version: 1.0

Status: standalone goal prompt for a new chat

## Goal Request

```text
Bạn đang làm việc trong project:

C:\Users\ironh\Downloads\PIG_Behavior_Project

Hãy tạo goal với objective:

Chạy và hoàn thiện toàn bộ scientific legacy 16-frame rebuild từ native CVAT
task_0..task_3 đến
legacy_frame_object_annotations.csv. Ở mỗi bước phải chạy audit/checker, sửa
ngay lỗi code hoặc derived-data logic, chạy lại short gate và chỉ chuyển bước
khi gate PASS. Không bắt đầu model training, không chạy OOF và không sửa raw
data dưới data/.

Đây là execution goal, không dừng ở việc lập kế hoạch. Full dense recovery
được phép sau khi static/synthetic checks, source audit, provenance audit,
CVAT input audit và exact one-complete-group recovery smoke đều PASS với cùng
semantic config và input hashes.
```

## Required Authority Order

1. `AGENTS.md`.
2. `.agents/memory/01_PROJECT_MEMORY_SHORT.md`.
3. `.agents/memory/02_CURRENT_DECISION.md`.
4. `.agents/memory/03_PROJECT_RULES.md`.
5. `.agents/memory/08_WORKFLOW.md`.
6. `docs/LEGACY_16F_REBUILD_FROM_SCRATCH_RUNBOOK.md`.
7. `src/legacy_burst_recovery/README_legacy_burst_recovery.md`.
8. Current source code, fresh input hashes and fresh audit evidence.
9. This prompt for execution scope and completion criteria.

Fresh code and audit evidence supersede historical counts or obsolete `k0`
notes in memory. Record every discrepancy and update active memory only after
the rebuilt lineage is verified.

## Current Source Facts

- Annotation authority is selected per task, never merged:
  `task_0..task_2 = annotations.xml`, `task_3 = annotations.json`.
- XML `image@id` and JSON `shape.frame` must resolve through that task's
  `data/manifest.jsonl`.
- Behavior authority is the lowest CVAT task frame inside each burst. Its
  suffix can be `k0..k5`; do not assume `k0` and do not majority-vote labels.
- For each actor present on the authority frame, map that behavior to all six
  anchors and all recovered dense frames.
- Each `k0..k5` bbox and Hidden value remains frame-level annotation evidence.
- Current fresh read-only audit loads 27,662 bbox rows and finds no duplicate
  anchor identities, but it reports 18 review issues representing five
  incomplete actor-burst keys. Recompute these counts at goal start.
- `data/` is immutable. A raw CVAT correction requires an exact task/frame
  report and user re-export; never patch XML/JSON automatically.

## Skills And Worktree

Use at least:

- `safe-refactor-test-guardian`;
- `dataset-contract-leakage-guard`;
- `experiment-lineage-reproducibility`.

Verify project root, branch and worktree list before edits. Use the current
main worktree unless the user explicitly assigns another worktree to this
session. Do not touch tracking files or tracking worktrees.

## Output Lineage

Create one fresh run:

```text
outputs/legacy_16f_rebuild/<RUN_ID>/
  00_behavior_source/
  01_provenance/
  02_video_policy/
  03_cvat_audit/
  04_cvat_inputs/
  05_short_smoke/
  06_full_recovery/
  07_export/
  08_audits/
```

Never overwrite a historical/canonical output. Bind code SHA, dirty-worktree
status, source hashes, config, command, runtime and every output hash.

## Ordered Execution

### P0 - Reconcile And Static Gate

- Read all authorities and inspect the exact current implementations/callers.
- Run compile, import, focused tests, Ruff and line-length checks.
- Verify the generator entry point and every command path in the runbook.
- Record selected skills, worktree, code SHA and dirty status.

### P1 - Native CVAT Source Audit

- Run `check_cvat_annotation_quality.py --print-issues`.
- Confirm selected XML/JSON sources, row counts, frame-manifest binding,
  behavior/Hidden presence, bbox validity, duplicate identity and six-slot
  coverage.
- For every issue, report task, zero-based frame ID, one-based frame position,
  total task frames, image name, group, pig ID, observed and missing slots.
- Fix code/checker bugs immediately with regression tests.
- If the issue is raw annotation, stop only that dependency and request a
  corrected re-export; do not drop, synthesize or relabel the row.

### P2 - Build Behavior Source

- Run `pig_behavior.data.classification_dataset` with `--dry-run`.
- Prove no row loss, no label vote, no Hidden defaulting and exact first-task-
  frame authority evidence.
- After PASS, rerun without `--dry-run` into `00_behavior_source`.
- Hash `behavior_clean_merged.csv`,
  `behavior_with_feats_rectROI.csv` and
  `classification_source_lineage.json`.

### P3 - Source Video Provenance

- Run `src/legacy_burst_recovery/truy_nguon_multi_bbox.py --dry-run`.
- Resolve `G:\My Drive` and all five provenance roots, including source 4 via
  `.shortcut-targets-by-id`.
- Preserve canonical `video_final` for group hashes and use
  `video_local_path` only for Windows runtime/existence.
- Require day/video agreement, manifest authority, candidate fallback,
  complete six-slot actor keys, unchanged row count and zero unresolved video.
- After PASS, run the exact same configuration without `--dry-run`.

### P4 - Duplicate-Video Policy

- Require an explicit reviewed `exclude_source_videos.csv`; never infer or
  create an empty policy merely to pass.
- Run duplicate preview/audit dry-run, then write the versioned preview.
- Build nodup center/all-bbox scaffold with complete row accounting.
- Treat this scaffold as provenance metadata only, never bbox/behavior
  authority.

### P5 - Native CVAT Recovery Inputs

- Run `classification_v2_rebuild_legacy_cvat_recovery_inputs.py --audit-only`.
- Require exact first-task-frame behavior, six independent anchors, frame span
  15, zero duplicate keys and explicit exclusion accounting.
- Do not accept `PASS_WITH_DECLARED_EXCLUSIONS` without inspecting every actor.
- After clean PASS, generate center, anchor, issue, audit and input-manifest
  artifacts under `04_cvat_inputs`.

### P6 - Exact Short Recovery Gate

- Select one whole `group_id` proven complete by the audit.
- Do not use leading-row truncation or `--max-rows`.
- Run `legacy_burst_recovery.main` with `full_legacy_burst`, six CVAT anchors,
  scene mask and crops into `05_short_smoke`.
- Run the post-recovery checker.
- Require exactly 16 dense frames per eligible actor, unchanged six GT anchor
  bboxes, behavior authority propagation, valid Hidden provenance, no
  duplicate frame/object key and no silent row loss.

### P7 - Full Dense Recovery

- Proceed automatically only if P0-P6 PASS and hashes/config are unchanged.
- Use a fresh `06_full_recovery` root and the exact short semantic config.
- A restart may use resume only with matching config and input hashes.
- Monitor progress, runtime, errors, OOM and partial artifacts.
- If code/logic fails, fix it, add a regression test and repeat the exact short
  gate before restarting full.

### P8 - Full Recovery Audit

- Run the full output checker on all dense rows.
- Verify 16-frame completeness, six anchor offsets `0,3,6,9,12,15`, bbox
  source, behavior, Hidden, timestamps, key uniqueness, source-video policy
  and row accounting.
- Produce class, day/video, task/source and exclusion support reports.

### P9 - Frame-Object Export

- Run `legacy_burst_recovery.export_legacy_annotations`.
- Use the dense map as source and independently reload native CVAT behavior
  authority.
- Require `legacy_frame_object_annotations.csv`, export audit, behavior
  authority audit and discrepancies CSV.
- Do not use `--training-only` for the canonical export.

### P10 - Final Lineage Gate

- Verify every eligible actor has 16 frame rows.
- Verify zero duplicate canonical frame/object keys and zero invalid bbox.
- Verify all six CVAT anchor bboxes and frame-level Hidden evidence are
  unchanged.
- Verify behavior belongs to the canonical ten classes and matches first-task-
  frame authority independently.
- Hash all source manifests, policies, dense/export artifacts and audits.
- Write `legacy_16f_rebuild_completion_audit.json`.
- Update runbook/README and active memory with observed counts and remaining
  claim boundaries.

## Error Handling Contract

1. Code defect: patch minimally, add regression test, rerun static and short.
2. Derived schema/count/hash defect: diagnose authority and regenerate only
   inside a fresh/versioned run root.
3. Raw CVAT defect: do not edit; report exact CVAT correction location and
   wait for user re-export.
4. Human policy ambiguity: do not invent a decision; request exact policy.
5. Full-run interruption: resume only with identical hashes/config.
6. Never bypass a failure with broad `try/except`, `drop_duplicates`, row
   deletion, default labels, copied bbox, `allow_unresolved`, or stale output.

## Completion Criteria

The goal is complete only when:

- all P0-P10 gates PASS;
- the final export exists under the fresh lineage;
- source, dense and export row/key accounting is complete;
- every retained actor has a valid 16-frame native unit;
- no raw data or tracking code changed;
- no model training or OOF ran;
- README and runbook commands match the verified executable flow;
- final report lists commands, exit codes, counts, hashes, fixes, exclusions,
  runtime, changed files, Git diff summary and unresolved risks.

Do not mark the goal complete merely because a full process exits zero.

## New-Chat Use

Paste the `Goal Request` together with this file path:

```text
C:\Users\ironh\Downloads\PIG_Behavior_Project\plans\
classification_v2-legacy-16f-scientific-rebuild-goal-prompt.md
```

Tell the new agent to read and execute the entire prompt, not only summarize
it. Return the final hash-bound handback to the parent classification chat.
