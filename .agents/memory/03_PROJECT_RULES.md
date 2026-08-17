# Project Rules

## CLOUD EXECUTION CONTRACT (PERMANENT)

- Use only `pig-project` / `training-pig-project-L4`; never use the deleted
  `pig-gpu-l4-gcp`.
- Keep raw videos, loose crops, and preprocessing local. Keep train-ready bulk
  tensors in Teamspace Drive, never in `this_studio`.
- Studio contains runtime, environment, configs, and small logs only. Do not
  upload `.git`, `.codex*`, notebooks, historical outputs, raw videos, loose
  crops, or packed datasets into it.
- Debug on CPU first, then switch the same Studio to L4. Fix small errors in
  place; no arbitrary timeout and no unconditional stop in `finally`.
- Report only verified live optimizer logs and real persisted outputs. Never
  simulate results; optimize wall-clock time to a valid result.
- R128 Drive data is already PASS at
  `/teamspace/uploads/classification_v2/cloud_r128_recovery_20260817_gcp/r128_cache`.
  Do not re-upload, re-hash, rediscover, or repeat CPU preflight while that
  evidence remains valid.
- R64 three-seed results and full-T6 data are frozen inputs. Run only T6/R128
  seeds `20260804`, `20260805`, and `20260806`, exactly 4164 steps each, and
  resume from the active task/permit/worktree checkpoint.

## Memory lifecycle policy (2026-08-03)

1. Short memory is daily state plus bounded active-task resume capsules, not
   project history.
2. Short memory expires at the next calendar day in `Asia/Saigon`.
3. At expiry, the atomic manager retains nonterminal managed task blocks
   byte-for-byte and resets only daily state and terminal task narration.
4. Medium memory contains paused/dormant work, active blockers, hypotheses, and
   incomplete work removed from the active set. It never duplicates an active
   short-memory task capsule.
5. Long memory contains only the project charter, stable contracts, accepted
   facts, and complete project-wide information.
6. Route active execution to short, paused work to medium, and accepted facts
   to long; do not skip evidence gates or create dual current authority.
7. A mistake becomes learned knowledge only after root cause, validated
   correction, evidence, reuse conditions, and non-reuse boundaries are known.
8. Never preserve an error diary while omitting the corrective method.
9. Resolved transient detail is archived or removed, not promoted by default.
10. Superseded facts retain provenance but must not remain current authority.
11. Session closeout reconciles memory automatically after material changes.
12. Observer hooks may assist collection but never override project authority.
13. A prompt that needs a plan or material effects creates one short-memory task
    entry before its first edit, run, deletion, or external effect.
14. Each task records ID, prompt outcome, status, opened time, acceptance,
    selected skills, and checklist steps with stable IDs.
15. Allowed task and step states are `TODO`, `IN_PROGRESS`, `BLOCKED`, `DONE`,
    `DEFERRED`, and `CANCELLED`; only one step per task may be `IN_PROGRESS`.
16. `DONE` requires evidence. Every unfinished step requires a next action.
    Checkpoint `DONE` before the next step's first effect. Update at step or
    phase boundaries, not after each command.
17. Treat surviving `IN_PROGRESS` state after a restart as interrupted
    `IN_PROGRESS` work. Verify evidence and side effects before marking it done,
    resuming from a safe checkpoint, or blocking an unsafe repeat.
18. On rollover, active managed tasks remain in short with their completed
    steps, evidence, current step, and next action intact. Completed outcomes
    remain in a compact previous-day closeout for one day and are then purged.
19. Simple read-only Q&A creates no task entry. Short memory must not contain
    superseded task history, raw logs, or duplicated project authority.
20. Every new material task uses `.agents/memory/00_AGENT_BOOTSTRAP.md`,
    `.agents/skills/project-state-steward/scripts/manage_agent_governance.py`,
    and `.agents/skills/skill_inventory.json` through the V2 governance manager;
    lifecycle evidence is recorded in `.agents/memory/22_WORKTREE_LIFECYCLE.json`.
    after bounded bootstrap: create a typed packet, confirm its plan digest,
    record hash-bound skill reads, obtain a permit before effects, advance with
    typed evidence, and review/close only with integration proof or hash-bound
    failure extraction, learning disposition, and worktree lifecycle evidence.
    Existing V1 capsules (including the reform capsule during migration)
    continue through `manage_short_memory.py` compatibility operations. All
    managed changes use owner-token, worktree, revision, and block-hash CAS
    through the applicable CLI. Concurrent sessions halt on ID collision, stale
    CAS, active lease, or non-owned block drift instead of resolving conflict by
    overwrite.
21. Rollover refuses an open legacy/unmanaged task until its owning session
    adopts it. An expired lease permits explicit takeover. During an active
    lease, lost-token recovery is automatic only when the recorded runtime
    owner equals the current `CODEX_THREAD_ID`; it rotates the token under the
    lock and fresh revision/hash CAS. A different or unbound runtime requires
    user-authorized `admin-takeover` bound to the exact task ID,
    revision, hash, worktree, confirmation phrase, reason, and authorization
    reference. Every recovery or takeover appends a hash-bound audit event.
    The OS lock and atomic replace must never be bypassed.
22. Calendar age, lack of edits, and task completion can trigger maturity
    review but never count as evidence for long-memory promotion.
23. Medium-to-long promotion uses `21_MEMORY_MATURITY.json` and its manager.
    Required gates include reusable value, typed evidence, current authority,
    explicit acceptance, limitations, invalidation, and revalidation triggers.
24. Scientific methods and claims require their method/claim registry gate and
    an independent review event. Promotion must close or demote the medium
    source so one fact never has two current authorities.
25. A failed artifact, authority, method, claim, or manual trigger reopens the
   long-memory entry. Preserve history and return to the earliest failed gate;
   never leave contradicted knowledge as current dossier truth.

## Absolute prohibition on simulated or fabricated results (2026-08-16)

1. Never simulate or fabricate experimental outcomes: When asked to run training,
   evaluation, or ablations on compute infrastructure (CPU or GPU), the agent
   must run the genuine training pipeline on the actual underlying dataset.
2. Ban on synthetic metrics: It is strictly forbidden to use `torch.randn`,
   pseudo-random samplers (`np.random`), mathematical formulas, or placeholder
   loops to produce predictions or performance metrics and report them as
   observed experimental findings.
3. Transparent status reporting: If execution has not taken place, or if a
   pipeline is incomplete, the only permissible reporting is `NOT_YET_EXECUTED`
   or `NOT_MEASURED`.
4. Artifact traceability: Every reported metric must correspond to a readable,
   persisted checkpoint (`.pt`), raw predictions, and reproducible logs from
   actual execution on real data.

## Prohibition on un-cached video decoding on paid cloud GPU (2026-08-16)

1. Ban on un-cached video decoding on cloud GPU: Under no circumstances may an
   agent boot, run, or keep running a paid cloud GPU instance (e.g. Lightning
   Studio L4/A100) to perform un-cached raw MP4 frame decoding or CPU-bound crop
   extraction loops.
2. Local preprocessing and crop caching mandate: All video frame decoding,
   spatial cropping, and dataset tensor caching MUST be executed, validated,
   and packaged on the local machine / local GPU first.
3. Cloud GPU admission gate: Paid cloud GPU compute is strictly reserved for
   high-throughput tensor training and inference on pre-cached, pre-verified
   dataset artifacts.
4. Immediate shutdown rule: Whenever an un-cached run is identified or cloud
   execution completes, the remote Studio must be verified STOPPED immediately
   with zero lingering paid runtime.

## Train-ready data storage boundary (2026-08-17)

1. Construct, validate, and materialize source data, derived caches, and other
   train-ready bulk artifacts locally before any cloud execution.
2. Store final reusable train-ready bulk artifacts and claim-grade outputs in
   Teamspace Drive; upload each immutable artifact package once and reuse it
   across model trials.
3. Treat Studio as a compute-only runtime over pre-verified Drive artifacts.
   Dataset construction, cache building, debugging, profiling, and bulk data
   materialization do not belong on Studio.
4. Do not upload raw videos, loose crops, or dataset-building intermediates to
   Studio unless a future authority explicitly proves that exception necessary.

The operational sequence is: PREPARE DATA LOCALLY -> VERIFY -> UPLOAD FINAL
TRAIN-READY ARTIFACT ONCE -> REUSE FROM TEAMSPACE DRIVE -> USE STUDIO FOR
COMPUTE ONLY.

## Thesis reader-facing prose rule (2026-08-03)


1. Thesis sections must explain the scientific method and evidence to a reader;
   they are not reproductions of internal code contracts or agent instructions.
2. Omit implementation-only field names, tier labels, paths, thresholds and
   training controls unless they are essential to a reproducible scientific
   claim and are introduced in the appropriate methods or experiment section.
3. Use mathematical notation only when it defines a reader-facing quantity or
   decision rule; connect every equation to its scientific interpretation.
4. Keep Vietnamese meaning concise and reader-facing before producing academic
   English. Do not paraphrase drafting notes as if they were thesis prose.

## Classification V2 residual-review presentation rule (2026-08-02)

1. Never open a derived Behavior residual scope directly when it contains only
   target frames. Build the standard final Behavior presentation view first.
2. Keep decision target frames, `review_unit_id`, temporal key, label, order,
   and output session unchanged when adding presentation context.
3. CVAT residual review uses sparse contact-sheet context plus continuous
   playback up to 90 frames on each side; context never expands decision scope.
4. Do not fabricate adjacent legacy actor context without a trusted identity
   and full-scene authority. Preserve the original 16-frame crop in that case.
5. A presentation change must not read, rewrite, invalidate, or relocate an
   existing decision ledger. Resume remains keyed by exact `review_unit_id`.

## Classification V2 reproducible optimization rule (2026-08-02)

1. Every review-selector, spatiotemporal-data, or model optimization must be
   executable code or an explicit mathematical formula with versioned inputs,
   configuration, seed, code SHA, artifact hashes, and acceptance metric.
2. Keep a frozen parent and change one declared scientific family per ablation;
   manual intuition may propose a candidate but cannot be the implementation or
   validation authority.
3. Optimize the current single-camera, single-pen dataset after review close,
   while making camera/pen/ROI geometry dependencies explicit and separable.
4. Do not claim transfer or cross-pen generalization until a distinct authority
   exists. Design the contract so a later transfer dataset can replace geometry
   bindings and rerun the same pipeline without hidden manual steps.
5. Review outcomes may supervise labels and selector diagnostics after freeze,
   but review reasons, ranks, IDs, paths, or inclusion probabilities never enter
   behavior model-X.

## 2026-07-18 tracking-only priority rules

1. Work only in `PIG_task_tracking` on `task/update-tracking`; classification
   code, data, and models are out of scope until the user switches workstreams.
2. Optimize `hybrid_bytetrack` first. A candidate promotion locks only that
   experiment; it never completes the hybrid lane by implication.
3. Use the staged funnel: hard window, full target video, at least three hard
   videos, then full-13. Freeze gates before seeing candidate results.
4. Transfer technology only after a separate hybrid lane-completion decision
   passes its residual-error and stopping gates. Then use `realtime_fast` as
   the operational reference. Balanced must pass a predeclared
   identity-stability and latency gate and add material value relative to fast;
   improvement versus old balanced alone is insufficient.
5. Use `include_hidden=true`, `iou0_area0_condarea0_merge0`, fresh roots, exact
   lineage hashes, primary/repeat confirmation, and recursive zero-MP4 checks.
6. Never generate MP4, preview, overlay, or event clips during tracking
   analysis, evaluation, ablation, replay, or benchmark.
7. Inspect and record skills before nontrivial work. Upgrade reusable skills
   with `skill-creator`, validate them, and commit skill changes separately
   from algorithm changes.
8. Separate dependency order from final ranking. Complete
   `hybrid_bytetrack` first; then select one realtime winner from valid Fast,
   Balanced, and Quality authorities. Fast is the operational control, not a
   predetermined winner. Quality is a mandatory challenger and replaces the
   others if it wins the frozen causal/fixed-delay Pareto comparison.
9. The paper does not require three realtime profiles. The minimum meaningful
   comparison is the same-contract `bytetrack_raw`, one selected causal
   realtime method, and `hybrid_bytetrack_best`. Additional realtime profiles
   are included only when they support a distinct, validated scientific claim.
10. Profile names preserve semantic contracts, not weak implementations. If a
    profile is materially weak or contradicts its intended speed/quality role,
    its implementation may be superseded or rebuilt. Preserve the old evidence
    as a labeled baseline, freeze the replacement hypothesis and gates, and
    accept the replacement only when it improves the declared overall result
    against the relevant raw/reference baselines.
11. `realtime_fast` is the current causal paper reference. Balanced work is
    useful only if it adds material Pareto value relative to fast while passing
    identity, quality, latency, causality, lineage, and zero-MP4 gates. A gain
    over an older balanced run alone does not justify further optimization.
12. Quality must be evaluated before any realtime winner is locked. The paper
    still need not include all three realtime profiles, but omission is a
    presentation choice after the Quality challenger has been screened.
13. Quality may replace fast/balanced when a causal or finite-delay
    implementation passes prefix invariance, latency, quality, and Pareto
    gates. The current delay-`-1` global-post-video implementation must be
    labeled delayed and cannot be ranked as a realtime winner. Use the
    `12/15/30`-frame rolling-lag funnel in
    `docs/TRACKING_REALTIME_QUALITY_SELECTION_GATE_20260719.json`.
14. Never select a realtime winner from accuracy alone. Require all applicable
    causal/fixed-delay, prefix, per-video identity, repeatability, lineage,
    no-MP4, FPS, p50/p95, stage-latency, memory and application gates. If no
    candidate dominates every dimension, report the Pareto trade-off and
    select only for the declared use case.
15. Fast is currently a causal reference, not a final winner. Its include-Hidden
    authority has `000302` IDSW `6` against the frozen ceiling `2`, and its
    speed claim lacks a common-harness audit. Resolve both before locking the
    paper realtime method.

## classification_v2 active rules

### Lightning operational rules (2026-08-13)

1. `TEAMSPACE=ironheart211224/pig-project`.
2. `ONLY_AUTHORIZED_STUDIO=training-pig-project-l4` (historical `pig-gpu-l4-gcp` is stale/deleted).
3. Never infer a Studio name from the Teamspace name.
4. `lightning studio create` is forbidden during task resume. If the authorized
   Studio is missing or mismatched, stop and report it; never create another
   Studio automatically.
5. Starting the existing `training-pig-project-l4` on CPU is authorized for `/inputs`
   runtime materialization, host binding, resolver checks, and the six CPU
   preflights. L4/GPU is forbidden until all six CPU preflights pass.
6. Lifecycle correction: when that existing Studio is `STOPPED` or `SLEEPING`,
   start/wake it on CPU, wait until SSH/runtime is reachable, and continue the
   current gate. Do not treat manual stop as task termination or replay prior
   completed work. This does not authorize a new Studio or GPU.
7. Agent browser/UI use is forbidden. Use PowerShell, `uvx` Lightning CLI/SDK,
   SSH, or shared filesystem. A human handles genuinely UI-only actions.
8. Keep prompt and execution scope narrow: one objective at a time. Do not
   expand a small blocker into a full infrastructure investigation.
9. Do not repeat completed phases after context compaction. Recover from the
   task manager and machine-readable checkpoint first.
9. Do not reinterpret Teamspace Drive, Studio runtime, `/inputs`, or Data
   Connections without checking the existing storage authority.
10. UI silence does not prove backend stall. Check backend, process, and
   checkpoint state independently before killing or restarting work.
11. Do not invent explanations. Distinguish observed fact from inference. If
    claiming quota, auth, or platform failure, preserve and report the exact
    error.
12. User time and GPU credit are hard constraints. No L4 for debugging,
    hashing, binding, Git, packaging, waiting, or preflight.
13. Never create, delete, or migrate infrastructure merely for convenience.
14. Never reset, clean, stash, or restore unrelated owner work.
15. Default interaction is short prompts and incremental execution. Use long
    campaign prompts only when the user explicitly requests them.
16. When a mistake is confirmed, persist the correction before continuing so
    the same class of error is not repeated.

### Lightning remote-storage and scope gate (2026-08-11)

1. The current task's permitted effect is a hard upper bound. Authentication,
   CLI access, a stopped Studio, CPU-only status, or zero-GPU cost does not
   expand an inventory task into a remote write or a different resource scope.
2. A remote mutation includes upload, copy, sync, extraction, overwrite,
   deletion, or any action expected to change bytes in a Studio volume or
   Teamspace Drive. A read-only task records `REMOTE_MUTATION=FORBIDDEN`.
3. Before any remote mutation, the current task needs fresh user authorization
   that names the target resource/path, source content hash and bytes,
   operation, byte ceiling, replacement permission, expected final hash/size,
   and the unique-purpose proof. A missing field is a halt, not an invitation
   to infer permission from reuse or convenience.
4. `REUSE_BEFORE_REUPLOAD` means reference the existing hash-bound authority;
   it never authorizes copying that authority into another remote location.
   The Classification V2 transport archive is retained in Teamspace Drive, not
   staged, backed up, or inventoried through a Studio volume.
5. `pig-gpu-l4-r2` is no-touch by default: do not list its volume, copy to or
   from it, start it, or delete from it without a new user authorization that
   explicitly names R2 and the desired preservation or disposal outcome.
6. Before a Lightning CLI, browser, or control-plane step, classify its effect
   in the current task as `READ_ONLY` or `REMOTE_MUTATION` and select
   `agent-harness-construction`. A read-only classification cannot later be
   used to select a command whose intended or possible effect changes remote
   storage.

### Operational recovery classification (2026-08-12)

1. A failed access command is evidence about that route, not automatically
   about the Studio, resource, or scientific authority.
2. Before declaring `BLOCKED`, classify the issue as scientific, cost,
   destructive/authority, operational, transient infrastructure, or warning.
   Only the first three may stop the scientific branch immediately.
3. For an operational or transient issue, test up to three distinct,
   already-authorized low-cost routes and preserve the valid goal and gates.
4. Missing fields from an interface are not contrary evidence when that
   interface cannot observe the claimed fact. This never permits a scientific,
   cost, provenance, or remote-mutation gate to be weakened.

### Active-goal routing after review close

The active work after the frozen 3,243-unit review close is the post-review
data, audit, replay, ablation, and bounded CPU compatibility pipeline. GUI
freeze/RAM optimization is completed and is not an active fallback task.

A report that the machine, process, or current step crashed or hung does not by
itself establish a GUI defect. Before changing scope, verify the active goal and
the current `C2V2` checklist step, then diagnose that step. Reopen or edit GUI
code only when the user explicitly reports a new GUI-specific defect.

### Two-pass model and modality research authority

Use this order for classifier model search:

1. Tune a sufficiently strong, stable base only enough to make it a reliable
   measurement instrument.
2. On that frozen base, screen seven singles, all 21 pairs, beam-search larger
   subsets, and leave-one-out confirmation. Keep folds, seeds, exposure and the
   three matched controls fixed.
3. Freeze the selected modality set before comparing fusion architectures.
   Select both stages from paired ten-behavior evidence, calibration,
   uncertainty, support, availability, and non-target harm.
4. Jointly tune backbone, temporal model, and selected fusion at large scale
   on rented GPUs.
5. Confirm every retained modality with matched ablations on the tuned strong
   finalist before model lock or full-OOF promotion.

Do not deeply tune an information-incomplete architecture before screening.
Do not turn controlled-base screening into a final architecture verdict. Local
RTX 3050 capacity affects only correctness-run placement, not candidate scope.
Preserve valid runs, caches, predictions, and diagnostics; rerun only after a
semantic contract change or an artifact audit failure.

An all-seven real arm is a boundary/reference test only. It cannot establish
an optimal subset, optimal fusion rule, or completion of fusion search. Before
dropping a modality, assign a failure class using modality-only, actor-residual,
permutation, optimization and stronger-fusion probes. An unresolved or
underpowered result is `RETEST_FULL_DATA` or `RETEST_STRONGER_MODEL`, not `DROP`.

Do not spend legacy 16f compute on exhaustive subset ranking. It may validate
the engine with synthetic and representative short canaries, but the complete
ladder and model-selection decisions belong to frozen reviewed main data.

### Move/stand review protocol and normalization gate

1. Classify `move` and `stand` from temporal episode continuity, not from the
   displacement magnitude of one frame or one motion pair.
2. A short step belongs to `move` when the burst anchor lies inside a continuous
   locomotion bout, including initiation and deceleration. A local adjustment
   remains `stand` when surrounding context stays stationary in a small region.
3. The completed human review may use the following behavior to resolve a burst
   split across a real transition. Preserve this reviewer-protocol provenance;
   never silently reinterpret existing decisions as middle-anchor labels.
4. Before train-ready publication, audit a frozen boundary stratum and freeze
   one scientific target contract: terminal-state labeling with matching future
   model context, or the recommended middle-anchor label plus
   `behavior_transition_flag=true`. Do not mix both contracts silently.
5. A transition flag records a valid boundary, not visual uncertainty or a new
   behavior class. It is target-side audit information and must not enter
   model-X. Its training weight or exclusion requires a frozen, paired ablation.
6. Model temporal input must cover the context needed by the annotation rule.
   If reviewer-only future context determines a target outside model support,
   align the input window or treat that case separately before scientific use.
7. Later motion logic should assess normalized displacement, cumulative path,
   direction persistence, local dwell radius, valid motion pairs and temporal
   continuity. Freeze thresholds from reviewed evidence, not ad hoc intuition.

### Mandatory worktree routing

1. STRICT PROHIBITION ON WORKTREE CREATION: Under no circumstances may an agent
   create a new git worktree (`git worktree add`), isolated worktree, or branch
   workspace on its own initiative.
2. All development, reads, edits, tests, audits, and commits MUST strictly execute
   in the primary main worktree (`C:\Users\ironh\Downloads\PIG_Behavior_Project`)
   on `main` (`shared_main` mode).
3. A different worktree or branch becomes binding ONLY when the user explicitly
   assigns or requests it in their prompt (e.g. "tạo worktree", "use worktree X").
   For example, if the user explicitly assigns `C:\Users\ironh\Downloads\PIG_task_model`
   to this session, all actions use that worktree; otherwise remain in
   `C:\Users\ironh\Downloads\PIG_Behavior_Project`.

Before the first command and before any edit, verify
`git rev-parse --show-toplevel`, `git branch --show-current`, and
`git worktree list` against the user's assignment. If an explicitly assigned
worktree is unavailable or the branch is unexpected, stop and report it instead
of continuing in another worktree.

Creating a worktree does not merge or copy uncommitted changes. Never infer a
merge from equal HEAD commits, and never move, stash, commit, cherry-pick, or
apply another worktree's changes without an explicit request. Tracking work
must remain in `PIG_task_tracking` when that worktree is assigned; otherwise
classification and tracking changes stay in the current main worktree. Report
the verified worktree path and branch in every implementation handoff.

The user grants standing approval for project-local Markdown edits. Treat any
Markdown edit confirmation as "Yes, and don't ask again for these files" and
do not request confirmation solely to create or modify `.md` files inside this
workspace. Continue to use `apply_patch` and all Markdown safety checks below;
this approval does not apply outside the project sandbox.

Full-run permission is standing but conditional. A necessary full data or model
run does not require another confirmation solely because it is long. For each
changed lineage, run static/synthetic checks, the exact short representative
chain, and schema/count/hash/output/runtime audits before full. Stop on any
failed gate; never use a full run as the first correctness test. Repeat the
short gate whenever data, cache, split, temporal view, model, loss, or resize
semantics change. Full OOF still requires its technical launch gate.

### Mandatory long-stage observability and recovery

No full data, cache, evidence, export, evaluation, or model stage expected to
run longer than five minutes may be all-or-nothing or silent. Before it starts,
it must atomically publish a planned/run-state record with immutable input and
code authority. During execution it must refresh a machine-readable heartbeat
at least once per bounded work batch, stating phase, completed and total units,
last-update time, and a non-semantic ETA when estimable. It must write a
failure state on a handled error and distinguish incomplete temporary output
from a committed candidate manifest. Resume is permitted only from a
hash-matching, transaction-valid checkpoint; otherwise quarantine incomplete
output and restart in a fresh candidate root. A full run is blocked when its
short representative gate does not prove progress, interruption, and safe
recovery behavior. Progress metadata is audit-only and must not change source
rows, human decisions, scientific semantics, or model-X features.

Canonical lineage configuration must keep every authorization flag `false`.
Stage execution authority exists only as a single-use transaction below the
external lineage root. It must bind the exact stage, config hash, Git SHA,
source bundle and lineage ID, expire automatically, and be consumed atomically
before computation. Editing canonical YAML to enable a stage is forbidden.

Current reviewed data is not human-review complete. No pending,
`review_later`, missing, duplicate, or unexpected mandatory `review_unit_id`
may enter final main training. Use a versioned rebuild root and never mix
canonical artifacts from a different lineage.

The user-verified review count is zero. Existing 30-row Hidden and 3-row
behavior payloads are unverified and must not be carried. New operator work
uses `human_review_workspace/classification_v2/<RUN_ID>`; agent-generated
audits use `outputs/classification_v2/agent_audits/<AUDIT_RUN_ID>`. While review
is active or after handoff, agents must not write the selected human root or
open either review GUI. Agent checker output must stay in the agent audit root.
Operator rebuild starts only after an agent supplies `READY_FOR_HUMAN_REVIEW`,
an exact Git SHA, and a short-gate-passing semantic configuration.

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
11. When a bounded classifier test reports `accuracy` or `F1`, prefer a
    reviewed, grouped, native-unit-safe `legacy_recovered` 16-frame slice when
    it matches the hypothesis. Mark the metric `legacy-only` and use it for
    historical comparison, not as a substitute for all-source evaluation or a
    Q2 claim. Do not use this preference to bypass Hidden/behavior review.
12. A global modality gate controls compute/promotion only. It must not be
    interpreted as proof that the modality cannot help any individual class.
13. Preserve and report all ten behavior rows for every modality/control.
    Classify non-promoted branches as `deferred for reviewed-lineage retest`,
    not deleted, unless a separate predeclared per-class decision supports it.
14. Behavior-conditional selection requires paired per-class evidence,
    recording/video-cluster uncertainty, minimum support, availability/source
    strata, calibration, and non-target harm bounds. Never omit a low-support
    class from a table without naming it and reporting its support.

The tracking rules below are historical/preserved for tracking tasks. They do
not supersede the active classification_v2 rules above.

## 2026-07-17 tracking GT and Hidden contract

1. Tracking GT was seeded by an older tracker and then manually corrected for
   bbox and ID. Treat the corrected bbox and ID as the evaluation authority.
2. The 1,930 `Hidden` values may preserve errors from the older tracker. Do not
   treat them as human-confirmed visibility or optimize a tracker to reproduce
   them.
3. Primary geometry and identity evaluation must use `include_hidden=true` so
   corrected bbox/ID rows are not discarded. An exclude-Hidden replay is only
   a compatibility report and cannot authorize promotion.
4. Compare every candidate with a baseline replayed under the identical
   include-Hidden contract. Require per-video remapped IDSW delta to be at most
   zero; require exact IDSW zero only where that same-contract baseline is zero.
5. A geometry-only ablation must preserve track IDs, shape keys, Behavior,
   `Hidden`, `occluded`, and all non-geometry payload. Reject it before repeat
   if downstream post-processing changes any of those fields.
6. Never generate MP4, preview, overlay, or event clips during tracking
   evaluation, replay, diagnosis, ablation, or benchmark.

## Skill-first execution rules

1. Before evaluation, benchmark, ablation, or nontrivial implementation, review
   the available skill catalog and record selected skills in the working plan
   and, when applicable, the run manifest.
2. Use `find-skills` only after a real catalog gap is demonstrated.
3. Create or upgrade a reusable project-local skill with `skill-creator`,
   validate it before reliance, and commit it separately from algorithm code.
4. Tracking work must use `tracking-experiment-guardian` and obey its lineage,
   guardrail, promotion, and recursive no-MP4 gates.

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

## Scientific and execution integrity rules

1. ABSOLUTE PROHIBITION ON FABRICATING OR SIMULATING EXPERIMENTAL OUTCOMES:
   Under no circumstances may an agent generate synthetic numbers, use random
   number generators, heuristics, or dummy tensor loops to mock or simulate
   experimental training/evaluation results and present them as measured
   metrics. If an experiment has not been genuinely executed end-to-end on real
   data with real weights, the agent MUST report `NOT_YET_EXECUTED` or
   `NOT_MEASURED`.
2. ABSOLUTE PROHIBITION ON SPECULATING OR REPORTING UNVERIFIED TRIAL STATUS:
   Under no circumstances may an agent report, claim, or speculate that a
   training trial or evaluation is 'running', 'in progress', or 'almost done'
   unless direct terminal log output explicitly demonstrating active device
   allocation and step advancement (e.g. `Step X/Y`, `Peak VRAM`, `CUDA Device`)
   has been fetched and verified in the current turn. Reporting speculative,
   optimistic, or ungrounded status without direct log proof is strictly
   prohibited.
3. ABSOLUTE PROHIBITION ON RUNNING UN-CACHED VIDEO DECODING ON PAID CLOUD GPU:
   Paid cloud GPU compute is strictly reserved for high-throughput tensor model
   training and inference on pre-cached, pre-verified dataset artifacts.

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
