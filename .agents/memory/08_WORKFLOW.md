# Workflow

V2 governance activation uses `.agents/memory/00_AGENT_BOOTSTRAP.md`,
`.agents/skills/project-state-steward/scripts/manage_agent_governance.py`,
`.agents/skills/skill_inventory.json`, and
`.agents/memory/22_WORKTREE_LIFECYCLE.json`.

## 2026-08-13 Lightning resume and execution gate

Before any Lightning, GPU, or execution command, resolve the existing storage
authority and require these exact values:

```text
TEAMSPACE=ironheart211224/pig-project
ONLY_AUTHORIZED_STUDIO=training-pig-project-l4
STUDIO_NAME_MUST_NOT_BE_INFERRED_FROM_TEAMSPACE=YES
STUDIO_CREATE_DURING_RESUME=FORBIDDEN
AGENT_BROWSER_UI=FORBIDDEN
```

If `training-pig-project-l4` is missing or mismatched, stop and report it (historical `pig-gpu-l4-gcp` is stale/deleted). Never create
another Studio automatically. Starting the existing `training-pig-project-l4` on CPU is
authorized for `/inputs` runtime materialization, host binding, resolver checks,
and the six CPU preflights. Do not switch to L4/GPU until all six CPU
preflights pass. If it is `STOPPED`/`SLEEPING`, start/wake this same Studio on
CPU, wait for SSH/runtime reachability, and resume the current gate. Manual
stop is a lifecycle pause, not task termination; do not replay completed work.
Use PowerShell, `uvx` Lightning CLI/SDK, SSH, or shared filesystem; humans
handle genuinely UI-only actions. Keep one objective per prompt/execution
scope and do not turn a small blocker into an infrastructure investigation.

On context compaction, recover from the task manager and machine-readable
checkpoint before repeating a phase. Check backend, process, and checkpoint
state independently; UI silence alone is not evidence of a stall. Preserve the
exact error before claiming quota, auth, or platform failure, and distinguish
observed facts from inference.

Do not reinterpret Teamspace Drive, Studio runtime, `/inputs`, or Data
Connections without the existing storage authority. Do not spend L4 credit on
debugging, hashing, binding, Git, packaging, waiting, or preflight. Never
create, delete, or migrate infrastructure for convenience, and never reset,
clean, stash, or restore unrelated owner work. Use short incremental prompts by
default; use a long campaign prompt only when explicitly requested. Persist a
confirmed correction before continuing.

## 2026-08-17 Main-only no-repeat resume gate

Before any continuation command, work from the primary repository on `main`
and read the current decision once. Do not move to an existing classification
worktree or create a new one. Treat every recorded PASS and completed trial as
an idempotent checkpoint: inspect its output and durable publication state,
then continue from the first incomplete step.

For the active R128 recovery, the first incomplete action is to publish and
verify the existing seed `20260804` result on Drive. Then run only seeds
`20260805` and `20260806`, one at a time, for 4164 optimizer steps each, using
the same Studio, runtime, and Drive cache. Never repeat upload, CPU preflight,
R64, full-T6 preparation, seed `20260814`, or Studio creation.

## 2026-08-11 Classification V2 canonical remote code deployment

This workflow is not permission to mutate remote storage during inventory or
cleanup. Before its transfer/extraction step, a current task must hold the
user-approved remote-mutation envelope in the Lightning remote-storage and
scope gate: exact target path, source hash/bytes, operation, byte ceiling,
replacement permission, expected final hash/size, and unique-purpose proof.
The retained Teamspace Drive archive is referenced by its hash; it is never
copied into a Studio volume for staging, backup, inspection, or convenience.
`pig-gpu-l4-r2` remains no-touch unless a fresh authorization names it.

Before any post-S1 scientific run, export one compact runtime bundle directly
from the final canonical Git SHA. Include the complete tracked runtime closure
(`src/pig_behavior`, `scripts/classification_v2`, root packaging files, and
the required corrected-route authority); exclude `.git`, datasets, outputs,
caches, virtual environments, temporary worktrees, and all working-tree dirt.

Record the canonical Git SHA, archive SHA256, size, file count, included paths,
and critical-entrypoint hashes in the bundle manifest. Extract and prove the
exact package locally under an isolated import path before starting Lightning.
The proof must cover `resolution_pipeline`, `stage1_temporal_screening`,
`remote_input_resolution`, host binding, the resolution executor, and its
non-training CLI path.

Transfer the archive and manifest once, verify its SHA256 remotely before an
atomic extraction under a canonical-SHA-specific directory, then repeat the
isolated import proof there. PIECEMEAL_REMOTE_MODULE_COPYING is forbidden. A
hash or import-source mismatch fails closed before input binding, media decode,
or optimizer execution. Reuse the already verified remote inputs; do not
retransmit them.

## 2026-08-11 CVAT host-side media realization

Materialize CVAT observations from the scientific `source_video_key` and its
exact registered `source_video_path`. Convert only a registered
`data/videos/*.mp4` path to a path relative to the verified input root, then
derive the host realization below that root. Never recursively search,
fuzzy-match a filename, or open an opaque CVAT context key as media. Fail
closed for zero or more than one registered path; preserve legacy crop and
frame/box/actor semantics.

## Historical 2026-08-10 Lightning active-resource preflight

This historical preflight is superseded for active execution by the exact
2026-08-13 Lightning resume and execution gate above. Retain its values only as
lineage evidence; do not use them to select an active resource.

The historical preflight validated these resource types and identities before
its GPU launch:

```text
LIGHTNING_RESOURCE_NAMING_CONTRACT_VERSION=20260810-v2
TEAMSPACE_NAME=pig-project
STUDIO_NAME=pig-gpu-l4
SSH_ALIAS=lightning-pig-gcp
OLD_STUDIO_NAME=pig_project
OLD_STUDIO_NAME_STATUS=DEPRECATED_DO_NOT_USE_FOR_ACTIVE_EXECUTION
TEAMSPACE_AND_STUDIO_MUST_NOT_BE_INFERRED_FROM_EACH_OTHER=YES
RESOURCE_TYPE_MUST_BE_EXPLICIT=YES
```

The preflight requires `teamspace == pig-project`, `studio == pig-gpu-l4`,
`ssh_alias == lightning-pig-gcp`, one GPU, and `NVIDIA L4`. It fails closed
for `studio == pig-project`, `studio == pig_project`, or
`teamspace == pig-gpu-l4` in a new active execution. Historical artifacts are
not rewritten; their old names remain historical evidence only.

## 2026-07-31 project autoresearch loop

1. Read scope authority, method state, halt conditions, and selected skills.
2. Edit only `tools/pig_autoresearch/candidate.json`; use a fresh run tag and
   exactly one parameter from one declared family.
3. Run `python tools/pig_autoresearch/prepare.py`; it validates policy hashes,
   control-plane hashes, source config hashes, candidate schema, and adapter
   imports/options without executing the workload.
4. Stop when `authorization_eligible=false`; never edit method state to bypass
   registration or frozen/blocked authority.
5. Have an independent authorized reviewer create one permit below
   `.agents/authorizations/autoresearch/` from the printed bindings.
6. Execute only canonical `train.py --execute --authorization <permit>`.
7. Read `run_manifest.json`, `run_result.json`, and gate details; record
   tracking as `keep`/`discard`/`crash`, classification as `diagnostic`/`crash`.
8. Treat a tracking `keep` as campaign evidence only; normal validation,
   repeatability, method transition, claim, and promotion gates still apply.

Tracking trials pin the full baseline parameter set, path-config hash, fixed
rule combo, Standard V2 evaluator, Hidden inclusion, zero-MP4 contract, target
videos, baseline metrics hash, and per-video/aggregate acceptance gates.

## 2026-07-31 autoresearch adapter correction

- Root cause: the original templates passed unsupported tracking and
  classification CLI options, mutated the wrong config branch, relied on
  implicit defaults, and had no atomic permit or authority binding.
- Validated correction: use immutable launchers, one JSON candidate surface,
  exact project CLIs, pinned effective baseline, source/control hashes,
  single-use permit claim, fixed budget, structured ledger, and fail-closed
  metric/artifact/worktree checks.
- Evidence: `17` focused regression tests plus Ruff, compile, JSON, adapter
  parser probe, dry-run, and missing-permit rejection passed.
- Reuse when: adapting bounded one-variable experiments to this repository.
- Do not reuse when: authority has not registered a campaign, evaluator/data
  contracts changed without repinning, or a task needs scientific promotion
  rather than diagnostic trial evidence.
- Supersedes: editable `train.py` and tracking/classification template flow.

## 2026-08-03 automatic project-state stewardship

- Run `project-state-steward` before final handoff after material implementation
  changes; do not wait for a separate user request.
- Promote a correction only after root cause, corrective method, evidence,
  reuse conditions, and non-reuse boundaries are established.
- Route daily state and active managed resume capsules to `01`, paused/dormant
  work to `04`, stable knowledge to `05` or `12`, and authority to `02`.
- Never duplicate one task in `01` and `04`; the active capsule is its sole
  execution-state authority until explicitly paused or completed.
- Route repeatable process to `08` and cleanup evidence to
  `10_REPO_HYGIENE.md`.
- Reconcile or supersede stale entries instead of appending duplicate notes.
- Remove only current-session, proven-regenerable waste. Protect unknown,
  pre-existing dirty, scientific, and lineage-bearing paths.
- For nontrivial work, select at least one reasoning or verification skill and
  record its purpose and evidence in `11_SKILL_PORTFOLIO.md`.
- For architecture, debugging, evaluation, synthesis, or data-contract work,
  require a reasoning skill; verification alone is insufficient.
- Queue skill maintenance after user correction, repeated failure, dependency or
  CLI change, validator failure, or the relevant review-age threshold.

### Planned-prompt task ledger (V2 default; V1 compatibility fallback)

For a new material task, run the bounded V2 bootstrap and create a typed
record, then communicate and confirm the plan digest, record hash-bound skill
reads, issue a permit before each effect, advance with typed evidence, and use
`amend-plan` when scope changes. Review the outcome with exact dirty-path
dispositions; accepted work needs target integration/revalidation, while
partial or failed work needs unique hash-bound evidence extraction. Close only
with a typed learning disposition and a lifecycle retirement decision. Use the
V1 short-memory manager only to resume existing V1 capsules, including the
governance reform migration capsule.

1. Trigger when a prompt uses a multi-step plan or requires edits, execution,
   deletion, publication, permission, or another external effect.
2. Before the first effect, run the project-local `manage_short_memory.py
   create` command with a unique ID, prompt outcome, acceptance criteria,
   selected skills, and stable step IDs. Retain its private token only in the
   owning live session. Skip this for simple read-only Q&A.
3. Use only `TODO`, `IN_PROGRESS`, `BLOCKED`, `DONE`, `DEFERRED`, and
   `CANCELLED`. Keep at most one `IN_PROGRESS` step for each task.
4. At a phase boundary, run `inspect`, then `checkpoint` using the returned
   revision/hash and the private owner token. Once acceptance is proven,
   checkpoint `DONE` before the next step's first effect. Do not update after
   every command or patch a managed block manually.
5. Concurrent sessions rely on the manager's OS lock, worktree binding, lease,
   owner token, revision/hash CAS, atomic replace, and non-owned byte check.
   Halt on ID collision, stale CAS, unaudited active-lease takeover,
   runtime-thread mismatch, or external drift.
   After auditing legitimate byte-only drift, only the owner may run
   `reconcile` with the recorded revision/hash and inspected current raw hash.
6. Attach concrete evidence to every `DONE` step and a next action to every
   unfinished step. Derive task status from the step states.
7. Before final handoff, reconcile the checklist against the working plan and
   actual artifacts. Never mark work complete solely because a command exited.
8. On restart, treat surviving `IN_PROGRESS` as interrupted `IN_PROGRESS` work.
   Inspect evidence and side effects first. Mark it done when acceptance is
   proven; otherwise resume the smallest safe checkpoint or block an unsafe
   repeat. Never rerun the whole step from chat history alone.
9. If a crash lost the private token, inspect the task. Run `recover` without
   waiting for lease expiry only when its recorded runtime owner matches the
   current `CODEX_THREAD_ID`, and use the inspected revision/hash/worktree.
10. If the runtime owner differs or is absent, do not infer death from PID,
    process lists, or inactivity. Obtain explicit user authorization, then run
    `admin-takeover` with the exact task ID repeated, fresh revision/hash,
    expected worktree, confirmation phrase, reason, and authorization reference.
    Preserve the resulting hash-bound audit event for the ownership transition.

### Current authority retrieval

1. Check `01` expiry before using its content.
2. Select task scope through `18_AUTHORITY_INDEX.json`.
3. Read the scope's current and supporting authorities only.
4. Use `19_REASONING_ROUTING.md` before selecting code or domain skills.
5. Halt if two same-precedence sources remain current or contradictory.

### Session closeout sequence

1. Read `01` and compare `Expires` with `Asia/Saigon`. If expired, run
   `manage_short_memory.py rollover` under the project coordination root.
2. Inventory the session delta and separate user work from agent-owned output.
3. Close each step with evidence or an explicit next action. Keep active
   managed tasks in short as resume capsules. Move only explicitly paused or
   dormant work to medium, after removing its active capsule.
4. Resolve `21_MEMORY_MATURITY.json`, then run
   `manage_memory_maturity.py scan`. Register only completed, reusable,
   evidence-bound candidates; elapsed inactivity never satisfies a gate.
5. Review and promote an eligible candidate through the maturity manager after
   closing or demoting its medium source. Reopen any failed trigger, then
   regenerate the living dossier with `synthesize`.
6. Record accepted method state and scientific claims in `13` and `14`.
7. Run the halt contract in `16` before any gated transition or external effect.
8. Audit generated paths using `10`; delete only proven session-owned waste.
9. Record skill use and maintenance signals in `11`.
10. Rollover removes terminal task bodies, retains each nonterminal managed block
   byte-for-byte, writes one compact previous-day closeout, and drops older
   closeout/superseded narration. It halts on any open unmanaged task.
11. Run the v2 governance validator and focused unit tests.
12. For governance changes, run the pinned agent suite at least three times.
13. Label fixture results self-test only; never claim live-agent reliability.

If no durable knowledge, workflow change, active carryover, or cleanup evidence
was produced, do not edit the corresponding authority merely to show activity.

## Classification V2 ROI path rule (2026-07-31)

- Use `data/annotations/roi/ROI_annotations.toy_adjusted.coco.json` for
  Behavior GUI, review exports, and future ROI-dependent feature rebuilds.
- Do not default new Classification V2 commands to
  `data/annotations/roi/ROI_annotations.coco.json`; that path is the old ROI
  file retained for provenance.
- When exporting or rebuilding any ROI-dependent features, record that the
  adjusted ROI was used and recompute derived values rather than reusing
  artifacts built against the old ROI file.

## Classification V2 post-review learning gate (2026-08-01)

- Compose completed review layers sequentially against the current effective
  label. A later `accept` preserves the preceding effective label; never apply
  raw row overwrite semantics.
- Keep the targeted-analysis composition separate from the independently
  sampled control, then create a final-label candidate containing both.
- Run a HIGH-only fixed-point temporal audit on each final-label candidate.
  Any newly selected HIGH unit blocks review close and receives a bounded
  ±90-frame presentation view; MEDIUM findings remain diagnostic unless a
  separately declared review gate promotes them.
- Repeat compose and HIGH-only audit until HIGH=0. Never relabel a temporal gap
  automatically.

- Before opening any targeted or control residual scope, run
  `build_final_behavior_review_view.py`. Use the resulting presentation view,
  never the target-only scope CSV, as `--review-units-csv`.
- Preserve the same output session when replacing a target-only presentation;
  exact `review_unit_id` keys retain prior decisions and resume position.
- CVAT uses the standard 90-frame context radius. Legacy remains target-only
  when trusted adjacent full-scene actor context is unavailable.

- Use `docs/CLASSIFICATION_V2_POST_REVIEW_LEARNING_PIPELINE.md` as the operator
  sequence for the 2,729 review, residual controls, mini-CVAT source chain,
  feature diagnostics, and final integration preflight.
- Before review close, run only deterministic control-scope predeclaration from
  an explicit parent population and primary scope containing no outcomes.
- Freeze authority only when primary count is exactly 2,729 and the control
  review contains at least 120 fully resolved items with complete quality rows.
- Treat weighted residual error, estimated selector recall, and feature effects
  as diagnostics, not automatic patches or model inputs.
- Require explicit resolution for mini-CVAT Behavior or Hidden conflicts. For
  sequential source edits, each next before-hash must match the prior after-hash.
- Bind adjusted ROI and rebuilt frame features, then full-recompute T6, T8, T12,
  and T16; never reuse an old window structure.
- Review outcomes, reasons, selection metadata, ranks, paths, and IDs remain
  forbidden model-X inputs.

- After review close, express every selector, spatiotemporal-data, and model
  optimization as executable code or an explicit formula with input hashes,
  seed, configuration, frozen split, metric, and single-family ablation. Keep
  fixed camera/pen/ROI bindings explicit so later transfer evaluation can rerun
  the same contract; do not claim transfer from the current single-pen data.

## Classification V2 behavior-consistency re-review (2026-08-01)

Validated v3 narrowing supersedes unconditional retention of every ranked
nearest-history candidate:

- Always retain the actor unit that generated a consistency finding.
- Retain a non-fight episode-partner candidate only with bidirectional support
  and either same-track fight within 24 frames or at least 48 supporting frames
  whose support span overlaps the target interval.
- Drop proximity-only context and already-fight partner rows unless they are
  independent actors for another finding.
- When narrowing an active scope, create a new authority/session. Reuse decisions
  by exact `review_unit_id`, require every prior correction to remain selected,
  and write source/destination hashes plus a row-level reuse lineage table.

1. When the primary scope reaches exactly 2,729 unique terminal decisions,
   record its byte hash before any consistency audit.
2. Keep the primary ledger immutable. Build findings from its terminal labels
   plus the review-independent frame authority into a separate output scope.
3. Flag, but never auto-correct, linked fight-partner disagreement,
   social/fight conflicts, corrected interaction partners, and isolated label
   islands within an actor trajectory.
4. Never infer the episode partner from nearest-at-target alone. Build a
   bidirectional nearest-history graph over the context window, rank candidates
   by unique supporting frames, and retain synchronized partner units.
5. Keep the actor followed by ranked partner candidates in the targeted GUI.
   Candidate status is review context, never an automatic label correction.
6. Complete the targeted consistency scope before freezing review-close
   authority, then review the independently randomized residual control of at
   least 120 units.
7. Do not use the targeted scope to estimate selector false-negative rate; its
   enrichment makes it a correction tool, not a population-representative
   control.

Temporal and review-unit key collections must be serialized as JSON arrays.
The keys themselves contain `|`, so pipe-delimited storage corrupts linked-key
boundaries. The JSON-array correction is covered by a regression test and is
reusable wherever Classification V2 composite temporal keys are persisted.

## Reviewed-data training handoff (2026-07-30)

This is the single post-review sequence. Do not run steps marked
`AFTER_REVIEW_CLOSE` while the GUI review session is active.

1. `DONE_NOW`: preserve the toy ROI candidate and its manifest.
2. `DONE_NOW`: keep `ROI_annotations.coco.json` unchanged until promotion.
3. `AFTER_REVIEW_CLOSE`: freeze review-session output, manifest, and hashes.
4. `AFTER_REVIEW_CLOSE`: run decision-coverage and apply-output gates.
5. `AFTER_REVIEW_CLOSE`: apply behavior decisions exactly once to a new output.
6. `AFTER_REVIEW_CLOSE`: promote toy ROI and rebuild ROI-dependent features.
7. `AFTER_REVIEW_CLOSE`: rebuild frame-local, temporal, and T6/T8/T12/T16
   artifacts from the same source and label authority.
8. `AFTER_REVIEW_CLOSE`: run duplicate, leakage, mask, schema, and row-count
   audits; fail closed on any mismatch.
9. `AFTER_REVIEW_CLOSE`: create a reviewed snapshot and bind the approved
   grouped train/validation split by exact hash.
10. `AFTER_REVIEW_CLOSE`: build image/context caches and run loader,
    forward/backward, checkpoint, and tiny-overfit smoke gates.
11. `AFTER_REVIEW_CLOSE`: run the bounded baseline; full training requires its
    separate authorization and launch gate.

The review-independent evidence already passed is recorded in
`02_CURRENT_DECISION.md`. The reviewed snapshot must be versioned and must not
overwrite provisional, failed, or prior train-ready artifacts.

## Tracking workflow after three-mode reconstruction (2026-07-29)

- Use the three independent authorities under
  `docs/tracking/three_mode_historical_reconstruction/` for historical/current
  B0, R0, and B1 claims. Do not infer one lineage from another.
- Treat RF_ACC23 and current R0 as the same scientific prediction artifact for
  the scoped full-13 comparison; do not claim a current-R0 improvement.
- State the B0 result as a whole-pipeline development-set improvement with the
  IDSW limitation, not as a pure ByteTrack algorithm improvement.
- Treat surviving H5b/H4 XMLs as the historical hybrid authority. Do not use
  current B1 or the failed `0.20/64` replay as its executable substitute.
- The only next tracking task is a docs-only superseding method-freeze
  decision. Do not access unseen data, run tracking, alter profiles, or promote
  a method during that decision.

## Tracking authority use after scientific grounding (2026-07-29)

- Resolve tracking names through
  `docs/tracking/scientific_grounding/TRACKING_METHOD_IDENTITY_REGISTRY_20260729.csv`.
- Before citing a parameter or metric, require its effective-config or metric
  provenance ledger row and evidence class.
- Scope current 2x2 conclusions to current frozen artifacts. Never relabel
  current B1 as historical H5b/H4 or R0 as strongest historical method.
- Keep the unseen freeze suspended. The next permitted tracking action is only
  the read-only B1 detector-cache path and tracker-boundary audit.

## Tracking profile selection after retirement (2026-07-28)

- Active presentation profiles are exactly `bytetrack_raw`, `realtime_fast`,
  and `hybrid_bytetrack`.
- Use `run_tracking_mode.py --mode realtime_fast` for the causal zero-delay
  realtime method. Do not use or recreate the retired `realtime` alias.
- Requests for `realtime_balanced` or `realtime_quality_delayed` are
  historical-only; `realtime_fast_h1_r2` remains a rejected experiment.
- Historical manifests may contain retired names and must remain readable
  without rewriting those values.
- Preserve shared cache/replay, telemetry, evaluation, and offline-repair
  infrastructure. A future `rf_hybrid_offline` task requires a new isolated
  worktree from the updated main branch.

## Classification V2 correction accepted workflow (2026-07-26)

- `SCIENTIFIC_ACCEPTED_SHA=a35e0b9aae8b55167b4562cfc7e26a45e2b4e312`
- `OPERATIONAL_FINAL_EXECUTION_SHA=PENDING_EXACT_SHA_AUDIT`
- Use `docs/CLASSIFICATION_V2_OPERATIONAL_RUNBOOK.md` and the centralized
  lineage config for one-stage-at-a-time execution.

- `PHASE4_IMPLEMENTATION_SHA=a35e0b9aae8b55167b4562cfc7e26a45e2b4e312`
- `ACCEPTED_IMPLEMENTATION_SHA=a35e0b9aae8b55167b4562cfc7e26a45e2b4e312`
- `OBJECT_TRACK_KEY_EXACT_SHA_REAUDIT=PASS`
- `PHASE1_4_INTEGRATED_ACCEPTANCE=PASS`
- `PHASE4_HUMAN_SIGNOFF=APPROVED`
- `REVIEWER=TuanHM`
- `REVIEW_DATE=2026-07-26`
- `REVIEWED_SHA=a35e0b9aae8b55167b4562cfc7e26a45e2b4e312`
- `MAIN_SYNC_STATUS=COMPLETE`
- `READY_FOR_LINEAGE_REBUILD_PLANNING=YES`
- `READY_TO_REBUILD_FRAME_LOCAL=NO`

The accepted object identity contract is
`schema.classification_v2.object_track_key`,
`classification_v2.object_track_key.v1`: source, dataset, and video scope,
then authoritative `track_id`, with `object_id` fallback, under RFC3986
UTF-8 escaping. `PIG_ID_AUTHORITATIVE=NO`, and production key values are
unchanged. The Group A checker, identity conformance, negative controls,
two-root 14-stage chain, current-authority links, and release preflight
passed after the exact-SHA fast-forward.

Readiness is limited to planning. Do not start lineage rebuilding, GUI
review, model execution, or training in this status update.

Source planning must use the canonical current-source section in README:
the completed `outputs/legacy_16f_rebuild/legacy_16f_rebuild_20260718_v2`
P0-P10 export and crops, exactly 12 behavior XMLs, ROI JSON, and `data/videos`.
Do not substitute the superseded `legacy_full_multigt_masked_nodup_16f`
paths.

## Classification V2 acceptance-reopened workflow (2026-07-25)

- `PHASE4_IMPLEMENTATION_SHA=76a0458e39769d3e7fac865dd16439a0ed3c3a04`
- `ACCEPTED_IMPLEMENTATION_SHA=76a0458e39769d3e7fac865dd16439a0ed3c3a04`
- `PHASE4_EXACT_SHA_AUDIT=PASS`
- `PHASE1_4_INTEGRATED_ACCEPTANCE=REOPENED`
- `PHASE4_HUMAN_SIGNOFF=APPROVED`
- `REVIEWER=TuanHM`
- `REVIEW_DATE=2026-07-24`
- `REVIEWED_SHA=76a0458e39769d3e7fac865dd16439a0ed3c3a04`
- `MAIN_SYNC_STATUS=CODE_INTEGRATED_BUT_ACCEPTANCE_REOPENED`

The differential gate classified the ten prior failures as follows:

- one `ACCEPTED_IMPLEMENTATION_DEFECT`: Group A production output and its
  independent checker disagree on `object_track_key`;
- one `PREEXISTING_MISSING_IGNORED_ARTIFACT`: the V6 contract node requires
  three Git-ignored human-review CSVs absent from fresh worktrees;
- eight `PREEXISTING_ENVIRONMENT_OR_PATH` failures: sandboxed Python cannot
  stat `G:\My Drive`; both SHAs pass the complete 17-node legacy file outside
  that isolation.

No documentation change caused a test failure. The preserved stash remains
unapplied and is not authority.

Decision:
`ACCEPT_PHASE_4_AND_PHASE_1_4_INTEGRATED_IMPLEMENTATION`
was the historical human sign-off. The controlling decision is now
`REOPEN_PHASE1_4_INTEGRATED_ACCEPTANCE_FOR_IMPLEMENTATION_CORRECTION`.

`MAIN_HEAD_AFTER_DOCUMENTATION_SYNC=THIS_DOCUMENTATION_COMMIT` denotes the
single status-only commit whose exact SHA is in Git history. It does not alter
or replace the accepted implementation SHA.

Audit source:
`.codex_tmp/worktrees/phase1_4_final_acceptance_76a0458/outputs/`
`classification_v2/agent_audits/phase1_4_final_acceptance_76a0458/`.
Stable archive:
`C:\Users\ironh\Downloads\PIG_Behavior_Project_AUDIT_ARCHIVE\`
`phase1_4_final_acceptance_76a0458\`.
The archive has 87 files and matches the source SHA-256 inventory. The exact
report and checklist are `final_phase1_4_acceptance_report.md` and
`final_phase1_4_acceptance_checklist.md`; verdict:
`PASS_PHASE1_4_INTEGRATED_ACCEPTANCE`.

### Integrated gate results

- Candidate manifests: 28 production-built, 28 valid, zero audit-assembled.
- Upstreams: 26 `CURRENT_AUTHORITATIVE`, zero noncurrent accepted.
- Negative controls: 14/14 integrated, 15/15 builder, 7/7 extended.
- Transactions: 9/9 no-prior and 9/9 prior-candidate injections passed.
- Failed transactions left zero new valid candidates and zero partial
  authoritative outputs.
- LF/CRLF canonical equality and cross-root determinism passed.
- Runtime dependency closure passed 17/17.
- Phase 1–4 invariants passed; Phase 2 schema hash remains
  `ec0c511b5f5198240492be49c0492e543c9e38eb4a4ff446259b958c2a59963b`.
- Release authority remains entirely false.

### Planning-only authority

- `READY_FOR_LINEAGE_REBUILD_PLANNING=NO`
- `READY_TO_REBUILD_FRAME_LOCAL=NO`
- `READY_FOR_SOURCE_REBUILD=NO`
- `READY_FOR_HIDDEN_REVIEW=NO`
- `READY_FOR_TEMPORAL_HARMONIZATION=NO`
- `READY_FOR_NATIVE_EVIDENCE=NO`
- `READY_FOR_PIG_STRENET=NO`
- `READY_FOR_BEHAVIOR_GUI=NO`
- `READY_FOR_TRAIN_READY_EXPORT=NO`
- `READY_FOR_TENSOR_EXPORT=NO`
- `READY_FOR_MODEL_EXECUTION=NO`
- `READY_FOR_TRAINING=NO`

Lineage rebuild planning is blocked. A separately authorized correction must
align the independent checker with the current `object_track_key` contract and
repeat the affected exact-SHA acceptance gate. The former planning workflow
below is historical and superseded.

The next workflow may only determine the authoritative rebuild start, check
whether source merge remains `CURRENT_AUTHORITATIVE`, design an isolated run
ID, lock inputs and manifests, define output roots, dry-run/preflight,
population reconciliation, decision carry-forward, rollback, and prepare a
separate frame-local execution authorization. Frame-local execution is not
authorized.

Pre-sync dirty work remains recoverable in a backup branch, external package,
and unapplied stash; it was not restored to `main`.

Historical candidates `c1df0cc4`, `f42f0d33`, and `99d63723` remain preserved
as superseded Phase 4 implementation/correction records.

## Active mixed source workflow (2026-07-20)

Build the mixed source from the locked legacy P0-P10 export and the 12 XML
behavior files in `data/annotations/classification`. Never substitute the
older `data/annotations/tracking` directory. Hash every source and the merged
CSV before Hidden review. Preserve `legacy_recovered` and
`cvat_tracking_xml` provenance through temporal harmonization, review units,
folds and the training snapshot.

## Human-review evidence and sampling workflow

Use this order for each exact source lineage:

1. Finish two-sided frame/object Hidden review and apply it.
2. Harmonize native temporal units and build unreviewed windows.
3. Build causal Pig-STRENet artifacts from the harmonized frame table.
4. Exact-join validity-masked review evidence and assign behavior cohorts.
5. Write the immutable behavior scientific design before any decision.
6. Run GUI smoke, then resume full review on the same decision roots.
7. Require exact coverage and PASS the behavior scientific gate before
   authorizing a reviewed training snapshot.

History-to-target features require complete history and target. Random behavior
audit estimates only the post-high-risk residual intervention rate using exact
sampling weights. High-risk yield and clean controls remain diagnostic;
unselected units are never renamed human-verified clean. Review and sampling
fields are prohibited from model-X.

## Authoritative two-pass classifier research flow

1. Establish a stable measurement base. Search credible sequence heads and
   visual backbones with grouped inner validation, progressive budgets, and
   modest tuning; stop when rank and calibration are stable enough to measure
   input effects rather than spending the final compute budget.
2. Freeze its data, folds, seeds, preprocessing, optimizer exposure, capacity
   envelope, and metrics. Run seven singles and all 21 pairs, then use a
   predeclared beam to add one modality per level. Stop on a frozen no-gain rule.
3. Confirm the selected set with leave-one-out and the all-seven endpoint. Each
   subset uses parameter-matched-zero, availability-only, and real controls.
   Freeze the subset before comparing fusion architectures separately.
4. On rented GPUs, jointly tune the selected visual backbone, temporal model,
   and fusion. Increase budgets progressively from correctness and short pilots
   to multi-seed development; never select from outer-fold predictions.
5. On the tuned strong finalist, repeat matched zero/availability/real
   ablations for every retained modality. Lock a candidate only after global
   and behavior-specific confirmatory gates pass.

Candidate families may include strong 2D image encoders with sequence heads
and end-to-end video backbones; the RTX 3050 does not prune this search space.
Use the local GPU for semantic/correctness gates and bind remote runs to the
same manifest and implementation SHA. Reuse valid caches, predictions,
checkpoints, and diagnostics; rerun only after semantic changes or failed
artifact audits. Existing legacy results remain screening evidence, not proof
that high-capacity joint tuning or confirmatory ablation has been executed.

The existing all-seven run is only the ladder endpoint/reference. A negative
subset result enters failure attribution before rejection: input/availability
audit, modality-only probe, actor-residual probe, within-stratum permutation,
learning curves and gradient health, then a stronger mask-aware fusion control.
Classify the outcome as `NO_SIGNAL`, `REDUNDANT_WITH_ACTOR`, `UNDERPOWERED`,
`OPTIMIZATION_FAILURE`, `FUSION_CAPACITY_FAILURE`, or `DATA_QUALITY_FAILURE`.

Run only synthetic and representative subset canaries on legacy 16f. Its rare
class support and unreviewed source make exhaustive ranking low-value. Execute
the complete 21-pair and beam ladder after the reviewed main snapshot is frozen.

## Active worktree routing rule

Use the current main worktree by default. When the user starts two concurrent
sessions and explicitly assigns a worktree/branch to one session, that
assignment is binding for that session only. Do not assume that
`PIG_task_model` or `PIG_task_tracking` is permanent. Verify the assigned
repository root and branch before every implementation session; if no separate
assignment was made, remain in `C:\Users\ironh\Downloads\PIG_Behavior_Project`.

Creating a worktree does not merge or copy uncommitted changes. Do not merge,
copy, stash, commit, or apply changes between worktrees unless the user
explicitly requests that operation. Tracking remains separate only when the
user assigns `C:\Users\ironh\Downloads\PIG_task_tracking` to a session.

## Legacy CVAT correction to recovered 16f

Use the canonical classification source lane, not a separate model pipeline:

```text
task_0..task_3 annotations + manifest
  -> versioned provenance scaffold + explicit source-video policy
  -> duplicate preview and nodup scaffold with row accounting
  -> fail-closed CVAT anchor audit
  -> CVAT-derived center/scaffold and six-anchor bbox tables
  -> one complete-group dense recovery smoke
  -> post-recovery behavior/bbox/frame/key checker
  -> versioned full dense recovery
  -> frame-object export + independent native-CVAT k0 authority audit
  -> existing classification_v2 merge/features/review flow
```

Resolve each shape through `manifest.jsonl`. For each actor, propagate only
the `k0` behavior to `k1..k5` and all 16 dense frames. Preserve six independent
CVAT bboxes; recover only the ten intervening frames. Hidden is separate from
behavior propagation. Repeat audit and smoke after any annotation hash change.
Never overwrite raw `data/`, the old dense reference, or canonical outputs.
`PASS_WITH_DECLARED_EXCLUSIONS` is not a clean pass: inspect and explicitly
approve every excluded actor key or complete its six anchors before recovery.

The executable clean-root command sequence is
`docs/LEGACY_16F_REBUILD_FROM_SCRATCH_RUNBOOK.md`. Root CSV paths are
historical and must not be used. `exclude_source_videos.csv` is a reviewed
policy input; duplicate preview/filter artifacts are derived outputs.

## C6 temporal-control matrix activation

The code-ready temporal matrix is fail-closed. While legacy data is being
cleaned, only static and synthetic commands are legal:

```bat
set PY=C:\Users\ironh\anaconda3\envs\pig_project\python.exe
set PYTHONPATH=%CD%\src
set C6TDIR=scripts\classification_v2\04_baselines_smokes
set C6T=%C6TDIR%\check_c2v2_c6_temporal_controls.py
set C6TCFG=configs\classification_v2\legacy_development_c6_temporal_controls_code_ready_v1.json
%PY% %C6T% --config %C6TCFG% --static-preflight
%PY% %C6T% --config %C6TCFG% --synthetic-preflight
```

After a clean lineage handoff, create a new versioned short config with the
handoff ID, exact clean input hashes, fresh output root, and
`data_run_authorized=true`. Run `--data-preflight`; do not run constant or
shuffled-delta modes when the identifiability gate says they are equivalent to
real timing. Run each authorized `--run-mode` twice using `repeat01` and
`repeat02`, then run `--audit-short-gate`.

Full development requires a separate config with scope
`full_development_confirmation`, explicit development authorization, and the
exact path/hash/status/config hash of the PASS short gate. It never authorizes
full OOF. Sequence shuffle must use one shared permutation across actor,
geometry, motion, ROI, social, pen, union/full-frame content and their aligned
availability/quality masks.

## C6 2026-07-19 rebuild screening record

After the clean technical rebuild handoff, the required order was completed:

1. C6 temporal controls: 9 modes x 2 fresh repeats, then A128 freeze.
2. C6 modality inputs/cache from the new rebuild lineage.
3. C6 modality matrix: 22 modes x 2 fresh processes, then paired evaluate.

The modality decision is `PASS` with 44 valid packets, 14 paired comparisons,
2,000 video-cluster bootstrap draws per comparison, and zero errors. It is
still `legacy-only-unreviewed-development`; Hidden and behavior review remain
double-check pending. The actor/context pipeline uses hash-bound `.npy`
features and records zero source-media reads. No full development, Q2 or
main-branch authorization follows from this screening.

Interpret this matrix at two levels. The global gate may defer a branch from
legacy full-development, but it must not erase its per-class evidence. Preserve
all ten behavior rows and classify every branch as `retained`, `deferred for
reviewed-lineage retest`, or `unsupported because of insufficient class
support`; never rewrite `deferred` as generally useless.

For the future main reviewed lineage, run behavior-conditional modality work in
this order:

1. Freeze the reviewed source manifest, snapshot, native units, folds, actor
   base, temporal view, feature whitelist, seeds, and metric contract.
2. Run the seven-single, 21-pair, beam and leave-one-out subset ladder with all
   three controls; report all ten classes and behavior groups.
3. Add paired intervals, calibration/NLL, source/availability strata, harm
   bounds and the failure-attribution probes before rejecting a branch.
4. Freeze the selected subset, then compare fusion families from predeclared
   class-modality hypotheses or a declared factorial design.
5. Repeat static, synthetic, tiny-overfit, resume, and representative short
   gates after every semantic change. Run bounded full-development only for
   candidates that pass both behavior-specific and global safety gates.
6. Lock finalists before requesting the separate full-OOF authorization. Do
   not use outer-fold predictions to choose class-modality weights.

## C6 modality matrix activation

The matrix has one direct actor-only arm and three controls for each optional
branch: parameter-matched zero, availability-only, and real values. The
branches are geometry, motion, ROI, numeric social, pen context, union context,
and full-frame context. Union and full-frame are always separate experiments.

While the bound legacy lineage is dirty, only these commands are legal:

```bat
set C6DIR=scripts\classification_v2\04_baselines_smokes
set C6=%C6DIR%\check_classification_v2_legacy_c6_modality_matrix.py
set C6CFG=configs\classification_v2\legacy_development_c6_modality_matrix_code_ready_v1.json
python %C6% --config %C6CFG% --action static-preflight
python %C6% --action synthetic-preflight
```

The first project-data action remains fail-closed until the user supplies a
clean lineage. After handoff, create a new config with updated hashes, a
nonblank handoff ID, a new output root, and explicit authorization. Then run
`build-cache`, two separate `run-repeat` commands, and `evaluate` in that order.
Do not start full development until this short matrix passes. Never infer a
full-data model decision from the current dirty lineage.

## Legacy 16-frame native unit versus model input

For bounded legacy development, preserve the complete 16-frame burst as the
grouping, split, support, and evaluation unit. A model does not need to consume
all 16 contiguous frames. The current one-sequence contract is:

```text
C6 contiguous centered: offsets 5,6,7,8,9,10
C8 contiguous centered: offsets 4,5,6,7,8,9,10,11
S6 uniform span-16:     offsets 0,3,6,9,12,15
```

Use one sequence per native unit. Compare S6 with C6 to isolate temporal span
at fixed six-frame input, then compare C8 with C6 to isolate sequence length.
Keep the cache, native-unit set, fold, model, loss, seed, epoch count, and
optimizer exposure identical. Preserve real elapsed deltas: each S6 step spans
three original frame intervals.

The 2026-07-17 paired decision retains C6 for this one-sequence profile. S6 and
C8 remain registered ablations, not promoted defaults. The older sliding-T6
candidate has four windows per native unit and different optimizer exposure;
report it only as historical context, never as a causal paired comparison.
This legacy choice does not alter the main branch's source manifest or primary
temporal view and does not authorize Q2 or full OOF.

## Pen-boundary context is an isolated model candidate

The enhanced frame-feature step now derives label-independent pen context from
`data/annotations/scene/mask.png`. Canonical runs must bind mask SHA-256
`b59b998ef49335b730c5f117e7161f24ccd277d3b5130c0e640dab7bbb980658`,
threshold at 127, and use nearest-neighbor only when frame-size resizing is
required. Mask paths, hashes, availability, quality, inward normals,
`pen_center_inside` and binary `pen_near_boundary` stay outside model-X.

Spatial export may emit `pen_boundary_context`, but the current trainer
whitelist and full model do not enable it. The first promotion experiment
changes one family only: paired `actor_geometry_motion` versus
`actor_geometry_motion_pen` on the same native units, folds, temporal view,
seed, backbone, loss and sampler. Both modes receive `motion_delta`, so gain
cannot be attributed merely to adding generic movement. Run synthetic and short
real feature gates before any bounded model pilot. No full OOF or
external-camera claim is authorized by feature availability alone.

## Canonical Hidden evidence tiers

Hidden remains a frame/object review decision. After apply, sequence windows
must be rebuilt from frame rows; do not fast-reuse a pre-review window manifest
while the default Hidden quality policy is enabled.

The canonical policy is:

```text
main_train:
  hidden_burden_ratio <= 0.25
  hidden_longest_run_ratio <= 0.20

robust_train_only:
  hidden_burden_ratio <= 0.50
  hidden_longest_run_ratio <= 0.40

exclude:
  either robust limit is exceeded
  window_sample_weight = 0.0
```

Apply this contract independently to T6, T8, T12 and T16. Hidden burden uses
the current frame-level `hidden` values after review apply, including untrusted
Hidden=Yes conservatively. Hidden ratios, run lengths, trust and policy tiers
are audit/mask metadata and must never enter model-X. The generated
`--no-exclude-high-hidden-from-main` CLI option is ablation-only.
## 2026-07-20 runtime gate closure

1. Keep `bytetrack_raw` immutable; a common-harness runtime probe must not
   regenerate its quality predictions.
2. The current GPU harness uses one video, detector, GT, mask and environment
   with primary/reverse Fast and Balanced orders, `include_hidden=true`,
   `iou0_area0_condarea0_merge0`, and zero generated MP4.
3. Both causal profiles are prediction-repeatable but below 30 FPS. Balanced
   is closer to native with lower backlog/output age; Fast keeps the stronger
   full-13 identity result. Neither is a native operational winner.
4. Keep `realtime_quality_delayed` in the Pareto/report evidence as a
   post-video upper bound, but do not call it a realtime winner. Reopen its
   realtime contract only with a new causal/fixed-delay hypothesis and the
   staged funnel.
5. Authority: `docs/TRACKING_REALTIME_RUNTIME_PROBE_DECISION_20260720.json`.

## 2026-07-20 tracking Pareto update

1. Hybrid remains complete and `bytetrack_raw` remains a fixed quality/runtime
   authority; never rerun raw solely to compare a new tracking candidate.
2. The Fast far-right guard is promoted at `74cad2b` after full-13/repeat
   PASS (`IDSW 59`, HOTA `95.63%`, IDF1 `95.37%`, `000302=0`, MP4 `0`).
3. Fast is a causal quality reference, not a speed winner: native-throughput
   and common-harness runtime gates remain open. Do not claim it is faster.
4. Balanced remains a non-dominated quality reference but no new family opens
   without a predeclared window gate; Quality remains retained delayed evidence
   while its finite-delay candidates remain rejected by the declared gates.
5. Use `docs/TRACKING_REALTIME_PARETO_SELECTION_DECISION_20260720.json` as the
   current Pareto authority and preserve no-MP4/lineage rules.
6. The first Balanced far-right screen tied on both frozen windows and is
   rejected; do not advance it to full video or full-13.

## 2026-07-19 tracking paper critical-path override

1. Keep dependency order separate from selection rank. Hybrid is complete;
   then compare every valid Fast, Balanced, and Quality authority. Fast is the
   operational control, while Quality is a mandatory challenger with full
   right to become the paper realtime winner.
2. RB3 Balanced has repeatable quality at `IDSW 111`, but repeat p95
   `60.06 ms` fails the frozen `45.29 ms` gate. Keep it opt-in and do not open
   another Balanced family without new predeclared evidence.
3. RQ1 rolling lags `12/15/30` are rejected at frozen QW01 because IDF1 and
   HOTA regress despite lower IDSW. Do not run RQ1 on later windows or videos.
4. RQ2 and RQ3 are closed without promotion. RQ4 retains the output-equivalent
   clone optimization, but QW01 fails effective FPS, p95, and repeat runtime
   ratio. Do not run later Quality windows, video, hard set, or full-13 from
   this family.
5. Compare valid Fast, Balanced, and Quality authorities before locking the
   realtime winner. Quality is mandatory: a valid finite-delay Quality
   implementation becomes the paper method when Pareto-best. Current Quality
   finite-delay candidates fail runtime, so retain them as negative evidence;
   delay `-1` remains a post-video upper bound. The decision is multi-objective:
   identity, HOTA/IDF1, throughput, p50/p95, delay, memory, lineage and
   application cost all matter; accuracy alone cannot select the winner.
6. The same-contract include-Hidden `bytetrack_raw` authority now passes
   (IDSW `145`, HOTA `88.91%`, IDF1 `88.47%`, loop-FPS `22.65/27.03`, zero
   MP4). Fast is the current causal reference because it has the lowest
   causal IDSW, but its `000302` guard is open (`6` versus ceiling `2`) and its
   speed claim remains pending a common-harness runtime audit. Run the targeted
   Fast follow-up before building the paper comparison as raw -> selected
   realtime -> hybrid.
7. Do not spend time completing three realtime modes unless each earns a
   distinct scientific claim. A weak or semantically incorrect profile may be
   rebuilt; retain its old artifacts as labeled evidence and validate the
   replacement through the frozen staged funnel with zero generated MP4.

## 2026-07-18 H4-to-H5 execution update

1. Treat H4 as a proven component, not a promoted profile or completed lane.
2. Do not run H4-only full-13: its hard set improves only one difficult video,
   below the frozen minimum of two.
3. Open H5 on parent-derived `000233` frames `1104-1119`.
4. Screen H5 by hard window, full `000233`, then the same four-video hard set.
5. Evaluate H4 and H5 together at the hard-set stage. Full-13 opens only if
   at least two difficult videos improve and aggregate guardrails pass.
6. Keep `include_hidden=true`, fresh roots, input rehashing, and zero MP4.
7. Keep every realtime profile closed until hybrid has a separate stop-gate
   and lane-completion decision.

## 2026-07-18 active tracking-only workflow override

This section overrides the classification workflow below until the user
explicitly switches workstreams.

1. Work only in `PIG_task_tracking` on `task/update-tracking`.
2. Keep classification code, data, model runs, and review artifacts untouched.
3. Repeat the hybrid funnel per isolated residual family: hard window, full
   video, hard set, full-13, repeat, clean authority audit, and separate
   profile promotion. One promoted candidate does not complete the lane.
4. Use `include_hidden=true`, `iou0_area0_condarea0_merge0`, fresh roots, input
   rehashing, payload-integrity checks, and zero generated MP4.
5. Hybrid near-wall geometry is promoted at `4876217`; authority SHA256 is
   `6b6899109ddbca43042645b503896e41c985bd838eb64513eeb72a9210015665`.
6. Recompute residual hybrid events after every promotion. Open a realtime
   transfer study only after a separate hybrid lane-completion decision passes.
   Then use fast as the operational reference; balanced must pass a
   predeclared identity-stability and latency gate relative to fast.
## Isolated reviewed-Q2 execution roots

Operator commands for source rebuild, Hidden review, behavior review and apply
write only below
`human_review_workspace/classification_v2/<RUN_ID>`. Agent commands never write
that root and never reuse current canonical output folders. Every agent audit
or downstream artifact uses one unique
`outputs/classification_v2/agent_audits/<AUDIT_RUN_ID>` root.

The operator sends `RUN_ID`, `REVIEW_STAGE`, reviewer and review-code SHA at
handoff. Until `REVIEW_STAGE=behavior_complete`, the agent stops before
post-review rebuild, snapshot and project-data model smoke. After handoff, the
artifact map, generated contract, model-input manifest, snapshot and P0 audit
must all stay under the same agent root. Follow runbook section 17.2.1.

## One engine, isolated lineage profiles

Do not maintain an independent 16-frame feature implementation. Shared modules
may be reused, but source manifests, data hashes, reviews, folds, snapshots,
goals, authorizations, metrics and claims remain isolated by lineage:

```text
canonical classification_v2 engine
  + legacy-only-unreviewed-development profile
      -> legacy source only
      -> historical prompt/goal and configuration-screening lane
      -> T6/T8/T12/T16 inside each native 16-frame burst
      -> isolated development artifacts and claims
      -> no automatic activation or PASS transfer to the main goal
  + main classification profile
      -> source set bound only by its own versioned manifest
      -> legacy 16f currently excluded; no implicit legacy merge
      -> Hidden and behavior review required for its own selected data
      -> fixed6_observed_time primary view
      -> reviewed final artifacts after all gates pass
```

Legacy 16f remains unreviewed even though P0-P10 is technically clean. It needs
its own Hidden and behavior review before reviewed/train-ready use. While it is
outside the main source manifest, those decisions do not replace or block the
main profile's review coverage.

Review is a profile-level scientific policy, not an unavoidable code-path
dependency. A user-authorized exploratory profile may bypass human review only
when its manifests and outputs explicitly remain unreviewed. The current
mixed-source Q2 lineage is review-required and stays blocked until both review
layers pass; this does not block the separate legacy development profile.

Run the legacy L0-L8 scoped goal in a new chat. On completion, write and verify
its immutable handback audit, return to the original chat, and resume the
parent mixed-source P0-P8 goal. Never treat legacy completion as parent-goal
completion.

## Reviewed-data rebuild gate

For a new `classification_v2` data lineage, follow
`docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`. Full runs are
authorized only after the same semantic config passes static checks, a short
legacy+CVAT chain, and schema/count/hash/output/runtime audits.

This is standing user authorization: once those gates pass, proceed with a
necessary full run without asking again only because it is full or long. If a
gate fails or any semantic input changes, stop and repeat the short chain. Full
OOF must also pass its immutable launch packet and execution gate.

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

The old v6 root is a technical template/media reference only. Its 30 carried
payload rows are unverified because the user confirms no review has started;
do not continue from or carry that CSV. Build a clean root under
`human_review_workspace/classification_v2/<RUN_ID>` and start at zero.
Decision outputs live only below that root, while agent audits use
`outputs/classification_v2/agent_audits/<AUDIT_RUN_ID>`. During review, agents
may read but must not write the selected human root or launch either GUI. The
same read-only rule remains after handoff; agent evidence stays in its audit
root.

Separate roots prevent artifact collisions, but not code-version races. The
operator starts only after `READY_FOR_HUMAN_REVIEW` handoff with an exact Git
SHA and a short-gate-passing semantic configuration.

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

For Pig-STRENet artifact work, build the causal pair manifest before any model
integration. Legacy windows use explicit relative starts but export actual
frame boundaries; XML uses its native six-frame target and preceding same-track
history. Reject mixed-coordinate completeness calculations.

Run the control matrix in this order:

```text
T0 -> T1 -> H0 -> HA -> HS -> HR -> HRev -> PM
```

All derived views from one native event must have weights summing to `1.0`.
Pack ROI dynamics and fixed top-K social edges with deterministic row indexes
and masks. Partner routing is geometry-only. Keep availability/provenance out
of default model X and expose it only to HA. A bounded artifact canary must bind
input, code, config, environment and artifact hashes before trainer work.

The corrected media-bridge canary is
`pig_strenet_media_bridge_legacy_20260719_canary11`. It is an exporter/audit
PASS only: actor-crop difference maps are materialized from legacy crop files,
and full-scene ROI patches are decoded from the bound source video. The XML
follow-up `pig_strenet_media_bridge_xml_20260719_canary02` also passes both
pixel branches. The resolver rejects static `background.png`/`Image #1`, binds
video path plus frame index, records source hashes and writes per-pixel
provenance. Repeat the short gate after any semantic change.

The earlier XML follow-up canary at
`pig_strenet_xml_real_20260719_canary01/07_pig_strenet_attempt2` is retained as
pre-bridge evidence. Its scene-pixel block was an exporter-resolution issue,
not a source-data absence. The corrected XML run remains
`xml-only-unreviewed-technical-canary`: it cannot be treated as reviewed data
or used to claim accuracy, promotion or training readiness. `max-native-events`
must select target keys without truncating the full frame table, so causal
history and scene/social context remain available.

### Reviewed spatial tensor low-memory gate

For a reviewed spatial NPZ, materialize immutable NPY shards before any
trainer or smoke loads the full population:

```text
set "C2V2_EXPORT_DIR=scripts\classification_v2\02_train_ready_exports"
python "%C2V2_EXPORT_DIR%\classification_v2_materialize_spatial_memmap.py" ^
  --npz <tensor-export>\X_spatial_sequences.npz ^
  --audit-json <tensor-export>\spatial_sequence_audit.json ^
  --output-dir <train-ready>\spatial_memmap_bundle
```

The output directory is immutable and must not already exist. The materializer
streams ZIP members to NPY files, validates the canonical schema and tensor
content hash, then publishes atomically. Training and SpatialTCN smokes prefer
`spatial_memmap_bundle` and copy only requested rows. A compressed NPZ whose
uncompressed payload exceeds 256 MiB must fail before full-memory loading when
the memmap bundle is absent. Never raise that limit to bypass materialization.

Verify every shard hash on the first load. `--skip-memmap-file-hashes` is only
for subsequent strata in the same bounded smoke after one verified load. Keep
padding invalidity, motion/social feature invalidity, and numeric zero values
as distinct model-input states. Commit `a034440e` is the implementation
authority for this gate.

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

Commit `abae856` adds the model-selection layer on top of this lineage. Each
epoch writes window and native-unit validation predictions, but only grouped
inner-validation native-unit supported macro-F1 may select a checkpoint; NLL
is the deterministic tie-breaker. The selected packet contains:

```text
best_validation_predictions.csv
best_validation_native_unit_predictions.csv
best_validation_aggregation_audit.json
best_validation.pt
oof_test_predictions.csv
oof_test_native_unit_predictions.csv
oof_test_aggregation_audit.json
```

Outer-test artifacts are evaluation-only. Native prediction rows retain source
and split-group metadata for later grouped reports, while those fields remain
outside model X. Resume must match the native-selection policy in checkpoint
v6 and run identity v3; policy drift is a hard error.

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

## Legacy-only unreviewed development lane

The user separately authorizes bounded development on the legacy 16-frame
source without waiting for current human review. Use a new versioned root under
`outputs/classification_v2/legacy_only_unreviewed_development`; never write into
the reviewed rebuild, canonical train-ready, or historical full-OOF folders.

The lane must preserve one complete 16-frame burst as the native unit, group
splits by recording date or video, keep all overlapping windows from a burst in
one role, and bind source, feature whitelist, cache, fold, and config hashes.
Every artifact and metric must carry the exact scope label
`legacy-only-unreviewed-development` and `human_review_complete=false`.

Build model-input tiers for window lengths `6`, `8`, `12`, and `16` only after
temporal harmonization. All tiers inherit the same burst-level split. Compare
both all-sliding windows with per-burst event-mass normalization and a
deterministic one-window-per-burst matched view. Keep model settings fixed and
change only temporal length; aggregate evaluation to the 16-frame native unit.

Commit `21b34fd` implements the exact model-input boundary for this ladder.
The builder emits one full-universe selection ledger and one observed-time slot
manifest for each of the eight tier/view combinations. Training config must
bind the matching view, selection column, manifest filename, and exact input
length. Actor/context inputs are never truncated; padded spatial capacity is
sliced only after post-tier masks are proven false.

Run the ladder in this order: read-only source audit, complete-unit short chain,
full legacy data rebuild, leakage-safe snapshot freeze, loader sample,
one-batch forward/backward, tiny overfit, resume, then one short development
fold. A full or long run is permitted only after the exact short configuration
passes and receives the existing explicit authorization. Results from this lane
cannot support a reviewed main-branch Q2 claim.

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
  temporal encoders.
- The visual-backbone contract at `07ed768` supports audited ResNet18 160/224
  and ResNet34 224 controls. Unit tests use random init and do not download
  pretrained weights; active-data pilots remain blocked by the snapshot.
- The visual schedule at `2bd2fda` applies frozen, `layer4_only`, and optional
  full stages to actor and union-context ResNets. Backbone/head optimizer groups
  are stable across resume and bind checkpoint v5, run identity v2, run
  manifest v2, and registry v4. Its V0/V1/V2 audit has zero optimizer steps.
- Native-unit checkpoint selection at `abae856` supersedes those lineage schema
  versions for new runs with checkpoint v6, identity v3, manifest v3,
  prediction manifest v2, registry v5, and run audit v3.
- Native source/missingness probes at `9b04209` require the exact ordered trainer
  whitelist and train-ready window SHA256, aggregate to `temporal_unit_key`,
  fit grouped training roles only, and emit each eligible outer-test unit once.
  The availability probe permits only label-independent registered masks.
- The synthetic-only visual gate at `3be22f8` passes deterministic ResNet18-160
  gradient, ten-class tiny-overfit, eval, and in-memory resume checks. It never
  authorizes an active-data run.
- The strict loader at `111f152` aligns real fixed-six `time_delta` tensors to
  the complete window universe. Its checkpoint v4/registry v3 contract is
  superseded for new runs by the native-selection v6/v5 schemas above.
- Current classification regression is 429 passed and 181 deselected. This is
  fixture evidence, not training authorization.
- Transformer timing plumbing now passes in code, but every model run remains
  blocked until the reviewed snapshot and its exact hashes are frozen.
- The active lineage stops at block `01`: the reference Hidden design passes,
  but verified human coverage is 0/5,131 and apply is incomplete.
- Behavior review also starts at 0/4,670 verified decisions in the clean root.
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
