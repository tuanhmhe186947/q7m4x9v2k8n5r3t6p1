# Project Rules

## classification_v2 active rules

Full-run permission is standing but conditional. For each changed data/model
lineage, run static/synthetic checks, a short representative legacy+CVAT chain,
and schema/count/hash/output/runtime audits before full. Stop on any failed
gate; never use a full run as the first correctness test.

Current reviewed data is not human-review complete. No pending,
`review_later`, missing, duplicate, or unexpected mandatory `review_unit_id`
may enter final main training. Use a versioned rebuild root and never mix
canonical artifacts from a different lineage.

Hidden-specific rules:

- Hidden is a frame/object visibility attribute, never the 10-class target.
- Never trust CVAT Hidden solely because tracking emitted Yes or No.
- Audit both `Yes -> No` and false-negative `No -> Yes` corrections.
- Census untrusted Yes, stratified-audit trusted Yes, and use risk,
  stratified-random, and clean-control No cohorts.
- Do not propagate one Hidden decision across a 6/16-frame native unit unless
  an explicit reviewed span is stored.
- Do not edit raw XML/CSV; GUI writes decision CSV and apply writes a new
  derived frame-feature artifact.
- Unreviewed CVAT No remains untrusted. Do not silently coerce it to visible
  trusted metadata.
- High Hidden ratio is audited, not an automatic exclusion/down-weight rule.
- Report random weighted false-negative estimates separately from high-risk
  correction yield.

Current execution precedence: finish the versioned Hidden and behavior review
lineage before rebuilding trainer inputs or authorizing another full OOF. The
previous commit-`18d6692` full run is historical engineering evidence only.

1. Treat `classification_v2` behavior recognition as the active goal unless the
   user explicitly switches back to tracking.
2. Do not run full OOF training unless the authorization file is explicitly
   enabled and the execution gate allows it.
3. Do not make a Q2 result claim from pre-full, pilot, smoke, or shortcut
   artifacts. Q2 claim requires full OOF plus postrun completion gate.
4. Use letterbox image preprocessing for bbox actor crops. Do not square-stretch
   pig crops because it distorts body shape.
5. Reuse packed actor and visual-context image caches for training experiments.
   Do not repeatedly seek/crop/resize video frames in full loops when cache
   artifacts already exist.
6. `pig_id` is annotation-local and must not be used as cross-video identity.
7. Keep model inputs leakage-safe. Exclude manual/review/audit identifiers,
   path columns, label columns, and policy text from model X.
8. Use full-frame or partner visual context for interaction behaviors. Do not
   infer fight/social-nose only from isolated actor crops when partner context is
   required for the experiment.
9. Keep review decisions applied by `review_unit_id`; do not silently drop rows
   or alter original raw data under `data/`.
10. Before committing code changes, scan changed files for overlong lines and
    run `git diff --check`.

The tracking rules below are historical/preserved for tracking tasks. They do
not supersede the active classification_v2 rules above.

## 2026-07-03 IDSW guard rules

1. Preserve the split lost-track reacquire guard implementation that produced:
   `outputs/eval/hybrid_bytetrack/20260703_193439/smooth_det020_loose/`
   `iou0_area0_condarea0_merge0/tracking_metrics.csv`.
2. Current best tradeoff for `000231` + `000302` requires:
   - `lost_track_reacquire_guard=true`
   - `lost_track_reacquire_non_same_raw_distance_guard=false` as the default/base setting
   - `lost_track_reacquire_raw_owner_guard=true`
   - `lost_track_different_raw_hidden_owner_bypass=true`
   - `lost_track_different_raw_hidden_owner_min_missed=2`
   - `lost_track_different_raw_hidden_owner_min_center_gain=0.03`
3. Do not turn off `lost_track_reacquire_raw_owner_guard` globally; it fixes
   `000302` but damages `000231`.
4. Do not remove the conditional different-raw hidden-owner bypass without an
   ablation against `000231` and `000302`.
5. Do not assume appearance threshold tuning alone solves this tradeoff; tested
   `0.15` did not change the `000231=8`, `000302=0` result.
6. Do not reintroduce the need for
   `--profile-override lost_track_reacquire_non_same_raw_distance_guard=false`;
   this is now the base/default so tracking/eval/optimizer use it automatically.

## General rules

1. Always preserve the user's current experimental conclusion unless existing
   files clearly contradict it.
2. Do not repeatedly reopen settled hypotheses.
3. Do not blame weight for `000263` IDSW increase.
4. Prefer small, reversible patches.
5. Do not mix unrelated changes in one patch.
6. Do not run long benchmark/tracking unless the user explicitly requests.
7. When asked to audit, do not modify code.
8. When asked to patch, modify only the requested scope.
9. Always state which files were changed.
10. Always state which behavior changed and which behavior was intentionally not changed.
11. Always report which memory files were read before making changes.
12. Keep code lines within the repository formatter/linter limit before commit.
    Wrap long conditions, strings, comprehensions, function calls, and argument
    lists proactively. Before every commit that changes code, run a changed-file
    overlong-line scan, for example `rg -n "^.{101,}$" <changed-files>`, and
    fix any matches before `git commit` so pre-commit does not fail on line
    length.
13. For manual file edits, use `apply_patch` instead of shell write commands.
    Do not rewrite/delete-add an existing file when a small targeted patch is
    enough. If command output is compressed or lossy, re-read the exact file
    content before patching so a formatting fix does not corrupt the text.
14. Avoid repeating file-write failures: do not use shell redirects, heredocs,
    here-strings, `cat`, or ad hoc scripts to write source/config/docs unless a
    generated artifact truly requires it. After every source/config/docs edit,
    inspect `git diff -- <file>` and run the changed-file overlong-line scan
    before staging or committing.
15. For Markdown memory/workflow files, do not start by deleting and recreating
    the file. First read the exact current text, then use `apply_patch` with a
    small context-matched hunk. If a full rewrite seems necessary, prefer adding
    a new "active override" section at the top and preserve historical content
    below unless the user explicitly asks to remove it.
16. If an `apply_patch` hunk fails, do not immediately switch to shell-writing
    the file. Re-read the nearby lines with an exact reader, reduce the patch to
    a smaller hunk, and retry. After the retry, inspect `git diff -- <file>` to
    confirm no Markdown structure was corrupted.
17. Markdown append/edit protocol is mandatory for `.md` files:
    identify the exact heading or nearby anchor first, patch only that section,
    keep each hunk small enough to review, and avoid whole-file replacement.
    Never append by shell redirection, here-doc, here-string, `cat`, or a
    temporary generated overwrite. If the intended anchor is missing, add a new
    dated section near the top with `apply_patch` and preserve all existing
    historical content below it.
18. After editing any `.md` file, run `git diff --check` and a changed-file
    overlong-line scan before staging. For Markdown command examples, wrap long
    Windows CMD commands with `^` continuation instead of leaving one long line.
19. Markdown append/update failure prevention protocol is strict:
    - Re-read the exact target section immediately before editing.
    - Patch under a stable heading or insert one dated section near the top.
    - Keep each hunk scoped to one section and fewer than about 40 changed
      lines.
    - Never append with `>>`, `Set-Content`, `Add-Content`, heredoc,
      here-string, `cat`, or a temporary overwrite file.
    - If context matching fails, stop, re-read 20-40 nearby lines, and retry
      with a smaller hunk. Do not switch to shell-writing as a fallback.
    - After patching, run `git diff -- <file>`, `git diff --check`, and
      `rg -n "^.{101,}$" <file>` before staging.
20. Markdown failure-stop rule:
    - Treat `.md` files as hand-edited project memory, not generated output.
    - If two `apply_patch` attempts fail for the same Markdown target, stop and
      re-read the exact file section before trying again.
    - Do not recover from a failed Markdown patch by using PowerShell writers,
      shell redirects, temporary files, or whole-file replacement.
    - For append-like changes, insert under an existing heading or add one
      small dated heading near the top with `apply_patch`.
    - If the target location is ambiguous after re-reading, ask the user or
      report the ambiguity instead of guessing with a broad rewrite.

## Tracking-specific rules

1. For `evaluate_tracking.py` behavior and metric comparisons, treat commit
   `b697c4eba36db280cbf01f446873da17bcac509d` as the main historical reference
   unless the user explicitly asks for another snapshot.
2. Do not assume `hybrid_bytetrack` is already legacy-compatible.
3. Do not assume folder name `iou0_area0_condarea0_merge0` proves runtime flags
   were correct; inspect config/runtime path if needed.
4. Preserve the `runner.py` post-processing gates that improved IDSW:
   - identity guard: `cfg.enable_offline_smoothing and cfg.identity_swap_guard`
   - temporal refinement and overlap hidden island stabilization:
     `cfg.enable_offline_smoothing and (cfg.smooth_boxes or cfg.refine_boxes)`
   - `stabilize_overlap_hidden_islands(shapes, cfg)` must run after
     `refine_shapes_temporally(...)` in that second block.
4. Keep `hybrid_bytetrack` default rule flags OFF unless explicitly requested.
5. Do not enable `condarea` by default unless the user asks or ablation proves it.
6. Be careful with raw ByteTrack IDs; they may be unstable after occlusion.
7. For `000263` IDSW, inspect association logic before changing detector.
8. For `000302` improvement, remember it is attributed to weight, not necessarily tracking logic.
9. XML CVAT export is a support output, not the main objective.
10. Main objective is stable identity tracking for 8 pigs.

## Code-change rules

1. If changing `association.py`, isolate one behavior at a time:
   - raw_id logic
   - matching phase
   - lost/reid handling
2. If changing `runner.py`, do not silently force offline smoothing by mode.
3. If changing `detections.py`, document how it differs from legacy `tracking_engine.py`.
4. If changing `config.py`, document default mode/rule behavior clearly.
5. If changing evaluation path, ensure stale XML cannot be confused with fresh XML.
6. Before committing code, scan changed files for overlong lines with a command
   such as `rg -n "^.{101,}$" <changed-files>` and wrap matches proactively;
   do not rely on pre-commit failure to catch line-length issues.
7. When changing text or code files, prefer small context-matched patches over
   whole-file replacement. Verify the patch with `git diff` before staging.

## Verification rules

When user permits running checks, verify in this order:

1. Static/syntax/import check.
2. Single video `Pigs291119_000263_30fps`.
3. Single video `Pigs291119_000302_30fps`.
4. 3-video common set:
   - `Pigs281119_000085_30fps`
   - `Pigs291119_000263_30fps`
   - `Pigs291119_000302_30fps`
5. 7-video full set.

Metrics to watch:

- `remapped_idsw`
- `remapped_idf1_pct`
- `remapped_hota_pct`
- `remapped_fragments`
- `gap_tolerant_fragments`
- `fp`
- `fn`

## Preserved legacy agent rules

The previous `.agents/AGENTS.md` remains in place and contains broader
repository coding standards for PyTorch, OpenCV, GPU fallback, quality checks,
and command formatting. Root `AGENTS.md` is now the main Codex entrypoint; use
the preserved file as supplemental implementation guidance when relevant.
