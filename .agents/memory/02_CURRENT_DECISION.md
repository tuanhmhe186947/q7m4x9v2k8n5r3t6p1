# Current Decision

## 2026-08-17 Main-only cloud execution map

- `EXECUTION_AUTHORITY=main`. Do not use a classification worktree as the
  source of truth for this recovery, and do not leave a completed change only
  in a worktree.
- `TEAMSPACE_SHORT=pig-project`; the verified namespace is
  `ironheart211224/pig-project`.
- `STUDIO=training-pig-project-L4` (the CLI may display the normalized
  lowercase form `training-pig-project-l4`). The deleted
  `pig-gpu-l4-gcp` is historical only and must never be inspected, started,
  restored, recreated, or used.
- The persistent Studio URI is
  `lit://ironheart211224/pig-project/studios/training-pig-project-L4/`.
- SSH connection is `lightning studio ssh --name training-pig-project-L4
  --option RemoteCommand=bash`. The local installed executable used for this
  connection is
  `C:\Users\ironh\AppData\Local\uv\cache\archive-v0\ORHJMKmEugGFY9IW\Scripts\lightning.exe`.
- The first live probe reported `NOT_RUNNING`; the existing Studio was then
  started and switched in place. The latest control-plane probe reports
  `status=Running` and `machine=L4` for `training-pig-project-l4`. SSH runtime
  reachability is still unverified after the switch; no trial process or
  scientific progress is claimed.
- Teamspace Drive mount is `/teamspace/uploads`. Join these adjacent fragments
  without a separator for the verified R128 cache namespace:
  `/teamspace/uploads/classification_v2/`
  `cloud_r128_recovery_20260817_gcp/r128_cache`.
- Packed tensor:
  join `/teamspace/uploads/classification_v2/cloud_r128_recovery_20260817_gcp/`
  `r128_cache/packed_rgb_128_letterbox.npy`,
  `12075663488` bytes, SHA256
  `c352a74cade4587e9dcbb8c3eead0c095c992306549b53da6d8b2a361691f5ee`.
- Index:
  join `/teamspace/uploads/classification_v2/cloud_r128_recovery_20260817_gcp/`
  `r128_cache/packed_image_cache_index.csv`,
  `47781243` bytes, SHA256
  `9ccef8607973cfb8c8377474665af5d62874b5beea39ad716872b187f8d29d68`.
- Runtime root is `/teamspace/studios/this_studio/runtime`; Python is
  `/teamspace/studios/this_studio/runtime/.venv/bin/python`; source root is
  `/teamspace/studios/this_studio/runtime/src`.
- The production launcher is
  join `/teamspace/studios/this_studio/runtime/scripts/classification_v2/`
  `04_baselines_smokes/classification_v2_run_post_s1_resolution_screen.py`.
  Runtime bundle data is under
  `/teamspace/studios/this_studio/runtime/r128_recovery_20260817`; bulk data
  and claim-grade outputs must not be stored in `this_studio`.
- CPU preflight evidence is
  `docs/classification_v2/s1_04_remote_cpu_preflight_20260817.json` and is
  already PASS: counts `14706/12421/2285`, real batch
  `[16,6,3,128,128]`, forward/loss/backward/throwaway step PASS, cache bound,
  no raw-video or loose-crop fallback, and no cache build on Studio.
- Do not repeat upload, hash, CPU preflight, R64, or full-T6 preparation.
  Run only `T6/R128` seeds `20260804`, `20260805`, and `20260806`, exactly
  `4164` optimizer steps per seed, using the existing runtime and Drive cache.
  The three-seed R64 results and full-T6 data are already inputs for the next
  stage, not work to rerun.
- Before launch, bind each trial's durable result directory on Teamspace Drive
  explicitly; checkpoints, predictions, logs, descriptors, and metrics must
  never accumulate under `this_studio`.

## 2026-08-16 Cloud GPU compute waste prohibition & mandatory local preprocessing

- Running un-cached raw video decoding loops on paid cloud GPUs (e.g. L4) is
  strictly forbidden.
- All video frame decoding, spatial bounding box cropping, and dataset caching
  (such as 128x128 letterbox RGB crops for T6) MUST be extracted, validated,
  and packaged locally on the local machine / local GPU first.
- Paid cloud GPU instances are reserved strictly for high-throughput tensor
  training and evaluation over pre-cached, verified artifacts.
- Remote Studio `training-pig-project-l4` must remain `STOPPED` until all dataset caches
  are built and verified locally.

## 2026-08-13 Lightning operational authority and resume rules

- `TEAMSPACE=ironheart211224/pig-project` exactly.
- `ONLY_AUTHORIZED_STUDIO=training-pig-project-l4` exactly (historical `pig-gpu-l4-gcp` is stale/deleted). Never infer a Studio name
  from the Teamspace name.
- `lightning studio create` is forbidden during task resume. If the authorized
  Studio is missing or mismatched, stop and report it; never create another
  Studio automatically.
- Starting the existing `training-pig-project-l4` on CPU is authorized for `/inputs`
  runtime materialization, host binding, resolver checks, and the six CPU
  preflights. L4/GPU is forbidden until all six CPU preflights pass.
- Lifecycle correction: `STOPPED`/`SLEEPING` is a resumable Studio state, not a
  scientific blocker. Start/wake this same authorized Studio on CPU, wait for
  SSH/runtime reachability, then resume the current gate without replaying work.
- This correction was validated after a repeated manual-stop resume failure on
  2026-08-13; it does not authorize a new Studio, browser/UI, or L4/GPU.
- Agent browser/UI use is forbidden. Use PowerShell, `uvx` Lightning CLI/SDK,
  SSH, or shared filesystem; a human handles genuinely UI-only actions.
- Keep execution scope narrow: one objective at a time. Do not expand a small
  blocker into a full infrastructure investigation.
- On context compaction, recover from the task manager and machine-readable
  checkpoint before repeating work; do not repeat completed phases.
- Do not reinterpret Teamspace Drive, Studio runtime, `/inputs`, or Data
  Connections without checking the existing storage authority.
- UI silence does not prove a backend stall. Check backend, process, and
  checkpoint state independently before killing or restarting work.
- Report observed facts separately from inference. For quota, auth, or platform
  failure claims, preserve and report the exact error.
- User time and GPU credit are hard constraints. No L4 for debugging, hashing,
  binding, Git, packaging, waiting, or preflight.
- Never create, delete, or migrate infrastructure for convenience.
- Never reset, clean, stash, or restore unrelated owner work.
- Default interaction is short prompts and incremental execution. Use long
  campaign prompts only when the user explicitly requests them.
- When a mistake is confirmed, persist the correction before continuing so the
  same class of error is not repeated.
- This decision is reachable through the canonical authority index and is
  supported by `03_PROJECT_RULES.md` and `08_WORKFLOW.md`.

## 2026-08-13 Overnight continuation execution anchor

- Active task: `C2V2-CONT-20260813-01`; immutable parent:
  `C2V2-20260812-02@revision-75`.
- Inherited completed facts: base post-S1 materialization `PASS` with
  `9151758436` bytes; official resolver `PASS`; host binding `PASS`; total
  observations `201792` (`CVAT=143550`, `Legacy=58242`).
- Canonical CVAT registration SHA256:
  `891a7bbe28ca33fc6fb1f264d9ea3bc90476376d8d7f4735b9eeedb5a7752526`.
- CVAT runtime media root:
  `/teamspace/studios/this_studio/pig_e0_r3/inputs/data/videos`; materialization
  `PASS`; `12` files; `3428817239` bytes; SHA256 `12/12`.
- Scientific scope is inner-development-only, T6, resolution sequence
  `R64 -> R128 -> R160`; outer feedback and R224 remain forbidden. Frozen
  controls are B1, seed `20260804`, `4164` steps, AdamW, lr `0.003`, wd `0`,
  batch `16`, FP32, no scheduler, no early stopping.
- Previous CVAT R64 failure was absent runtime media; post-materialization R64
  is not complete. Exact next gate: CVAT R64 CPU preflight, then remaining
  five CPU preflights only if it passes.

## 2026-08-13 Overnight continuation CVAT R64 stop

- One bounded CVAT R64 CPU preflight ran on `pig-gpu-l4-gcp`; no duplicate
  process was observed and no other preflight ran.
- Result: `BLOCKED`. The first sampled window was CVAT video
  `Pigs291119_000216_30fps`, frames `0-5`; the official loader returned
  `image_load_failed@0..5` and `observed_frames=0`. Sixteen sampled CVAT
  windows showed the same error.
- The registered logical path was
  `data/videos/Pigs291119_000216_30fps.mp4`, resolved under
  `/teamspace/studios/this_studio/pig_e0_r3/inputs`; the file existed at
  `255698503` bytes. No deeper cause is asserted.
- The required failure checkpoint could not be recorded because the managed
  continuation had only its anchor step, already checkpointed `DONE`; the
  exact manager response was `step_missing_or_duplicate` for the missing
  post-anchor step. No second continuation task is authorized by the current
  execution authority.
- `SIX_CPU_PREFLIGHTS` is not satisfied; R64 reproduction, R128, R160, and
  L4/GPU remain forbidden. Exact next gate: resolve the supported checkpoint
  continuation without rerunning CVAT R64.

## 2026-08-11 Classification V2 post-S1 canonical remote package rule

- Status: `IN_PROGRESS`; this is an engineering deployment repair, not a new
  scientific experiment or a performance claim.
- `REMOTE_SCIENTIFIC_CODE_DEPLOYMENT_MODE=SINGLE_CANONICAL_BUNDLE`.
- The bundle is derived from exact canonical Git objects, excludes all working
  tree dirt, records its canonical Git SHA and SHA256, and replaces all
  piecemeal remote module copying.
- Local and remote import closure must prove that every critical post-S1 module
  resolves from the bundle before input binding, real-media preflight, or an
  optimizer step. Any import or hash mismatch fails closed.
- The only permitted scientific comparison remains the sequential T6/B1
  resolution screen: R64 first, then R128 and R160 only after R64 control
  acceptance. Inputs, labels, folds, seed, event weights, and 4,164 steps are
  unchanged; no outer access, H5, posture, Stage 2, or backbone experiment.

## 2026-08-11 CVAT scientific-media and host-runtime path contract

- `SCIENTIFIC_MEDIA_ID_IS_RUNTIME_PATH=NO` and
  `RUNTIME_MEDIA_PATH_REQUIRES_AUTHORITY_RESOLUTION=YES`.
- A CVAT context key remains scientific identity only and must never be passed
  to OpenCV as a file.
- The exact `source_video_key` maps through its registered `source_video_path`
  to an authority-relative `data/videos/*.mp4` path. Fuzzy matching is
  forbidden; zero or ambiguous registrations fail closed.
- Host paths are derived only under the verified input root. Legacy crop
  handling remains unchanged.

## Historical 2026-08-10 Lightning resource naming contract

This historical contract is superseded for active execution by the exact
2026-08-13 Lightning operational authority above. Its names remain only as
lineage evidence.

- `LIGHTNING_RESOURCE_NAMING_CONTRACT_VERSION=20260810-v2`.
- `TEAMSPACE_NAME=pig-project`.
- `STUDIO_NAME=pig-gpu-l4`.
- `SSH_ALIAS=lightning-pig-gcp`.
- `OLD_STUDIO_NAME=pig_project` and
  `OLD_STUDIO_NAME_STATUS=DEPRECATED_DO_NOT_USE_FOR_ACTIVE_EXECUTION`.
- `TEAMSPACE_AND_STUDIO_MUST_NOT_BE_INFERRED_FROM_EACH_OTHER=YES`.
- `RESOURCE_TYPE_MUST_BE_EXPLICIT=YES`.
- Historical records may retain `pig_project`; active execution must reject
  that name and must not treat teamspace and studio names as interchangeable.

## Classification V2 pre-GPU E0 authority and next permitted action (2026-08-08)

- Status: `PRE_GPU_MAIN_AUTHORITY_READY`; resolve the immutable release with
  `classification-v2-pre-gpu-authority-20260808^{commit}` after final tag
  creation. Main, not a temporary worktree, is the execution authority.
- E0: `B3_ACTOR_T6_PLUS_GEOMETRY_MOTION`, T6, FOLD_3, seed 20260804, actor
  RGB + geometry 6D + motion 12D only. The 16-step/AdamW/LR/checkpoint details
  are engineering freeze settings, not S1/final-model selections.
- Environment: staged CRLF `e0_environment/uv.lock` SHA256 `6b783d…103ca`
  is copied verbatim as package-root `uv.lock` and realized with `--extra pt`.
  Root `uv.lock` is development-only.
- Gates: local executable preflight and bounded smoke pass; outer test is
  BLOCKED for training, validation, checkpoint selection, metrics, and exports.
- Closed authorities: reviewed snapshot, grouped split/A12, H5, posture, and
  feature contracts remain frozen; H5 and posture are OFF in E0. H6/H12/H24,
  outer OOF, and S1 remain deferred.
- Next permitted action: resume the existing Lightning CPU/SSH setup from the
  pre-GPU tag. Do not create another scientific worktree or allocate a GPU.

## Contract

- Scope: current workstreams, blockers, next permitted actions, and authority links.
- Historical decisions are read-only in
  `20_CURRENT_DECISION_ARCHIVE_2026-07-31.md`.
- Authority precedence and conflict handling are defined in
  `18_AUTHORITY_INDEX.md` and `18_AUTHORITY_INDEX.json`.
- If two sources claim current authority for the same scope, halt before effects.

## Short-Memory Task Ledger

- Status: `VALIDATED` and revalidated on `2026-08-04`.
- Authority: `AGENTS.md`, `03_PROJECT_RULES.md`, `08_WORKFLOW.md`, and
  `project-state-steward` local v5 and its atomic task manager.
- Behavior: every new material task uses OS locking, a private owner token,
  worktree binding, lease, revision/hash CAS, atomic replace, and non-owned byte
  preservation. Rollover keeps nonterminal task blocks as byte-identical resume
  capsules; medium memory receives only explicitly paused/dormant work.
- Recovery: interrupted steps require evidence inspection. A lost token may be
  rotated during an active lease only when the recorded runtime owner matches
  the current `CODEX_THREAD_ID`; different or unbound threads require exact
  user-authorized `admin-takeover`. Legitimate byte drift still requires
  owner-authenticated `reconcile` with recorded and current raw hashes.
- Evidence: manager suite `14/14`; final governance suite `39/39`; Ruff,
  compile, JSON, skill, and governance validators passed. The pinned 25-task
  fixture passed three runs with
  `pass^3=1.0`; its negative control produced `pass^3=0.0`. Live task
  `HARNESS-20260804-01` recovered revision `5 -> 6` under the same runtime,
  added one audit event, retained its active lease, and rejected the old token.
- Forbidden: task entries for simple read-only Q&A, per-command memory writes,
  evidence-free completion, parallel `IN_PROGRESS` steps, cross-session task
  overwrite, manual managed-block patches, PID/inactivity-based owner-death
  inference, unaudited active-lease takeover, duplicate short/medium authority,
  blind reruns of `DONE` work, or superseded history.

## Autoresearch Harness

- Status: `DEV_PASS`; control-plane and regression contracts are validated,
  but no scientific experiment has executed.
- Authority: `tools/pig_autoresearch/policy.json`, tracking freeze authority,
  classification current state, method state, and halt conditions.
- Evidence: `17` focused tests, Ruff, compile, JSON parse, adapter `--help`,
  dry-run, missing-authorization fail-closed check, and diff hygiene passed.
- Next permitted tracking action: register a separate campaign method through
  authority, freeze baseline/acceptance, then issue one bound permit.
- Next permitted classification action: authorized synthetic diagnostic only.
- Forbidden: optimize the frozen method, self-authorize, train classification,
  reuse a permit, or promote a harness `keep` directly to a scientific claim.

## Governance Hardening

- Status: `VALIDATED`.
- Goal: enforce memory lifecycle, authority, method, claim, skill, halt, and eval
  contracts with deterministic validators.
- Current authority: files `12` through `19`, the project-state-steward skill,
  and focused governance tests.
- Evidence: v2 validator `PASS`, focused tests `8 passed`, pinned 14-task
  fixture suite `pass^3=1.0`, and negative control `pass^3=0.0`.
- Next permitted action: maintain contracts or collect a separate live-agent
  baseline without changing the pinned task/judge pair.
- Forbidden: cite fixture self-tests as live Codex capability evidence.

## Evidence-Based Memory Maturity

- Status: `VALIDATED`; the governance contract passed one promotion,
  invalidation, revision, re-review, and re-promotion cycle. No historical
  medium item was automatically promoted.
- Authority: `21_MEMORY_MATURITY.json`, its atomic manager, and the generated
  living dossier section in file `05`.
- Decision: elapsed time, inactivity, and task completion trigger review only.
  Promotion requires typed evidence, current authority, explicit acceptance,
  limitations, source disposition, and event-based revalidation triggers.
- Recovery: registry state is canonical. A stale or interrupted dossier is
  regenerated with `synthesize`; changed evidence, method, claim, or authority
  requires `reopen` at the failed gate.
- Evidence: nine manager tests, a two-process stale-CAS race, focused governance
  tests, Ruff, compile, AR-021 through AR-023 fixtures, and the event-triggered
  governance revalidation cycle pass. The negative control remains
  `pass^3=0.0`.
- Next permitted action: monitor registered triggers and admit a non-governance
  project fact only when its own evidence and acceptance gates pass. Do not
  bulk-promote file `04`.
- Forbidden: promotion by age, manual edits inside the generated dossier,
  unsupported claims, or active duplication between medium and long memory.

## Classification V2

### S2 social Top-K K=3 implementation (2026-08-04)

- Status: `DEV_PASS` for implementation and bounded CPU compatibility only.
- Commits: implementation `a8f727a5`, real checker `5bfc7180`, immutable
  evidence binding `6c2f2049`.
- Bundle manifest SHA256:
  `694948f570dfde2c6771efc2ad46f8a79bd0ba20ccda5768d18af631b9139119`;
  compatibility report SHA256:
  `cd2eb003eb40fc2b0dae3eff0c7ce0ba2953b0fab6f9be868f8796d118f2cc37`.
- Contract: `[T,3,10]` geometry-only partner tokens, separate availability,
  quality, and neighbor-count tensors, valid-count permutation-invariant
  pooling, and no source, identity, review, or behavior fields in model-X.
- Evidence: eight real T/source strata and `62` final focused tests pass;
  permutation, invalid-partner, invalid-motion, gradients, tiny-overfit,
  checkpoint-reload, and optimizer-resume deltas satisfy their gates.
- Boundary: no paired behavior result, no GPU/full training, no paper metric,
  and no authorization for S3. Resolve the remaining model-readiness blockers
  before paired S0/S1/S2 screening; S3 remains blocked.

### Post-review playwithtoy boundary amendment V1 (2026-08-03)

- Human authority: f180-f186 is `playwithtoy`; f187-f191 is `stand` for
  `Pigs281119_000114_30fps`, track `2`; last nose contact is f186.
- Native unit f186-f191 is a resolved transition and is excluded with weight
  zero. Eight windows are affected; T6 f186-f191, T8 f180-f187, and T12
  f180-f191 change from trainable to invalid.
- Frozen amendment snapshot:
  `outputs/classification_v2/agent_audits/`
  `reviewed_engineering_snapshot_amendment_v4_1decfe4_20260803_231000`.
- Snapshot JSON SHA256:
  `ab86e2e04267cfdc8248f9bdb8774615479d67a3589f7a25844bb1a4c93a639e`.
- Candidate manifest SHA256:
  `c9a277e2ab1088d2a43833a86a0dcc031f32870367ec9e378f4dcb8032632f03`.
- Effective reviewed-frame authority SHA256:
  `4400f36c473954784ae3d8d520eb5e1b5e79a23792d21dd0475bfb419d061a4f`.
- Materialized population: `245,680` reviewed frames; `165,305` unified
  windows; `159,410` trainable; `5,895` excluded. All eight affected windows
  have train mask false and effective weight zero.
- Independent audit SHA256:
  `97673d3c9e7bf9df78ee409a9bcb8b7575396f18f1b8d3191d7e50ee675a1ede`;
  status `PASS`.
- Snapshot V3 remains immutable but is `SUPERSEDED` for future training.
  `c2v2.reviewed_lineage.amendment_v1` is `FROZEN` for local smoke and bounded
  pilot/debug only. Screening, claim-grade training, and paper metrics remain
  blocked.
- Accepted social ladder: S0 no-social, S1 current social 10D, S2 masked
  permutation-invariant Top-K K=3, then S3 GAT only if S2 passes its gate.
  Next permitted implementation is S2; S3 remains blocked.

### Model-readiness audit blocks training (2026-08-03)

- Status: `CONTRACT_SMOKE_PASS_TRAINING_BLOCKED`; no baseline, screening,
  claim-grade, or final-promotion run is authorized.
- Code authority is clean at
  `e666d85342f794752605efdb7ce767564290c321`. The audit package is
  `outputs/classification_v2/model_readiness_audit/`
  `model_readiness_e666d85_20260803_185056`.
- Frozen reviewed engineering authority remains snapshot V3, with `33,355`
  native units, `165,305` windows, `159,413` trainable windows, and `5,892`
  excluded windows. Behavior review/apply and reviewed-frame authority pass.
- Motion 12D, spatial 46D, pair validity, date-grouped split, forbidden-X,
  future-frame, and trainable cross-label gates pass. Eight real T/source
  forward-backward strata, tiny overfit, checkpoint resume, and `121` focused
  tests pass.
- Hidden review is `PASS`. The current reviewed rebuild proves complete original
  Hidden GUI coverage (`5,233/5,233`, zero missing, duplicate, or unresolved
  items), row-preserving apply (`245,680` input/output rows), and temporal
  harmonization consuming that apply. Behavior-review Hidden notes are a later,
  complementary exclusion layer: all `252/252` note-bearing rows are excluded,
  `84` related rows are propagated, and zero note-bearing rows remain trainable.
- Training mass fails: raw native-unit mass ratio is `11.0`, no fold-local
  event-weight artifact is bound, and three trainable `playwithtoy` windows
  retain target/ROI-derived sub-one weights without accepted loss policy.
- B0-B3 constructors and synthetic CUDA checks pass, but no reviewed RGB
  adapter binds them to the snapshot. BALANCED remains fail-closed scaffold;
  social and complete A12 authorities remain unresolved.
- Next permitted action: resolve sample-weight semantics, then implement
  fold-local event weights plus a reviewed RGB adapter and rerun the one-batch
  gates. Production code changed by this audit: `NO`.

### Hidden visibility weighting remains inconclusive (2026-08-03)

- Status: `DEV_PASS_INCONCLUSIVE`; no gamma, model, or paper claim is promoted.
- Valid run code is `c01a3849a6b11aa863037ef5710924cb8fe37cd2`.
  The hash-bound evaluator is
  `3c712c9bc4ba7a4ec1fd816d35a9f261b438883e`; the reporting branch is clean
  at `9b4242cc54b9cd3a81d16815b90b68fd2ca0f9b4`.
- Summary authority is
  `outputs/classification_v2/paired_ablation/hvw_s3e5_v2/summary` in the
  behavior/posture paired-ablation worktree. Its `10/10` inventoried artifacts
  hash-verify and all three source-run inventories pass.
- Gamma 0.5 minus gamma 0 native macro-F1 delta is `+0.00107456`, with t-based
  CI95 `[-0.00150936, +0.00365849]`; two of three seeds are positive. Native
  balanced-accuracy CI also crosses zero, and no per-class harm guardrail fires.
- The comparison keeps `134,412` main rows, adds only `98` robust-only train
  rows, preserves the `25,001`-row validation set and 2,630 optimizer steps,
  and has zero Hidden/review fields in model inputs.
- The v1 seed is invalid batch-order-confounded evidence and remains excluded.
  The pose result remains independently `INCONCLUSIVE` and unchanged.
- Current decision: retain main-only training. A later Hidden-weighting test
  requires new predeclared evidence; sensitivity arms do not select gamma.

### Motion episode and boundary diagnostic not promoted (2026-08-03)

- Status: `DEV_PASS_POSSIBLE_NOT_ESTABLISHED`; no selector or model-X
  promotion is authorized.
- Code authority: `074b3559847b3077a427d779e90246e3478997c3`.
- Exact run: `C:\pig_runs\classification_v2_reviewed_rebuild_20260802_v1\`
  `diagnostics\motion_episode_boundary_full_074b355_v1`.
- The 46D reference reproduced exactly. Causal predecessor evidence had
  `delta AP=-0.000106` versus 46D and `+0.003537` versus its matched
  availability-only control; grouped CI95% `[-0.002990, 0.007589]` crosses
  zero. Current-only also remained inconclusive; symmetric future context was
  unsupported and remains offline-only.
- Sidecar contract reads no behavior/review field, recomputes motion within
  native units, forbids legacy cross-burst identity, and preserves missing
  context as NaN plus validity. Twelve-decimal canonicalization is required
  before fit and serialization for exact replay.
- Next permitted action: a separately frozen source-aware fresh holdout or a
  separately predeclared behavior-model ablation. Do not consume the reserved
  control 120 for further tuning and do not promote subgroup move/stand point
  estimates.

### Reviewed rebuild and spatial engineering gate pass (2026-08-03)

- Review close contains `3,243` reviewed units; the two fixed-point `explore`
  decisions remain unchanged by explicit user decision.
- Reviewed rebuild authority:
  `C:\pig_runs\classification_v2_reviewed_rebuild_20260802_v1`.
- Population: `165,305` windows, `159,413` trainable, `5,892` excluded;
  frozen split `134,412` train and `25,001` validation with zero declared
  native-unit, actor, source-video, or group overlap.
- Low-memory code authority:
  `a034440e93726973a3062282ede4d6b8ad0a41cc`. The 920,418,240-byte tensor
  payload is consumed through 24 immutable memmap shards; full-memory NPZ
  fallback is blocked above 256 MiB.
- Evidence: all eight T/source forward strata, finite backward gradients, tiny
  overfit, checkpoint reload, deterministic optimizer resume, Ruff, compile,
  and `128` focused tests pass. No GPU or full training ran.
- Final reviewed engineering contract code authority:
  `e666d85342f794752605efdb7ce767564290c321`. Immutable snapshot authority is
  `C:\pig_runs\classification_v2_reviewed_engineering_snapshot_20260803_v1\snapshot_v3`,
  ID `reviewed_engineering_4c430dfae2d193dc`.
- Final package audit binds full code SHA, `24/24` shard hashes, tensor-content
  hash, eight T/source smoke hashes, finite gradients, tiny overfit, exact
  checkpoint reload and optimizer resume. The final focused suite is
  `127 passed`; independent audit and snapshot inventory checks pass.
- Next permitted action requires a separate execution decision: materialize the
  original-label-only replay sidecar, freeze a posture authority for its paired
  ablation, or authorize bounded reviewed-data model training. Do not access the
  final test or promote selector/model metrics from this engineering snapshot.
- GUI freeze/RAM optimization is already resolved. Do not route a later
  post-review crash or hang back to GUI work unless the user explicitly reports
  a new GUI defect.
- The fixed-point and residual-review subsections below are superseded state
  retained for review-lineage provenance.

### Fixed-point review closure blocked by two HIGH units (2026-08-02)

- Completed targeted 39 and control 120 compose without semantic conflict into
  a `3,241`-unit final-label candidate with `493` source-label corrections.
- Candidate authority:
  `composite_behavior_review_3241_faee589_20260802_073943_v1/`;
  decision SHA256 `950e803e6c1e9ee15abbd6b809b73f47519e812ece34a7cb20d39d1593ae658a`.
- Fixed-point audit found two unreviewed HIGH units in one fight-bounded run:
  `Pigs291119_000225_30fps.mp4`, track 7, anchors 204 and 210.
- Current permitted action: review only the two-unit ±90-frame context view
  `post_review_fixed_point_high2_context_3241_950e803_20260802_075000_v1/`,
  then recompose and rerun the HIGH-only fixed-point audit.
- Blocked until HIGH=0: final review-close freeze, selector optimization,
  corrected-source/data rebuild, posture integration, train-ready export, and
  model training.
- Optimization authority requires code/formula, hashes, seed, fixed split and
  ablation evidence. Current single-pen optimization is allowed after review
  close; transfer remains untested and geometry dependencies must stay explicit.

### Composite review and residual scopes ready (2026-08-02)

- Residual presentation correction is active for both targeted 39 and control
  120. Use the `post_review_residual_*_context_*_v1` views and exact commands.
- CVAT targets retain exact six-frame decision scope while presentation adds
  sparse contact-sheet context and continuous playback up to ±90 frames.
- Review keys, temporal keys, labels, order, output sessions, and existing
  decisions remain unchanged, so resume stays valid by `review_unit_id`.
- Eighteen legacy controls remain original 16-frame actor crops because no
  trusted adjacent full-scene actor authority exists; do not fabricate it.

- Primary 2,729, consistency V3 697, and micro-review 4 compose sequentially
  into 3,082 unique keys with zero input or quality-semantic conflicts.
- Final composite authority is under
  `outputs/classification_v2/review_authority/`:
  `composite_behavior_review_3082_faee589_20260802_055557_v2/`.
- Final net source corrections are 447; 471 keys changed at least once and 24
  returned to source after later review.
- Reverse audit selected 39 unreviewed correction-halo targets: 12 HIGH units
  across six fight-bounded runs and 27 MEDIUM units.
- A disjoint 120-unit residual control is frozen with seed `20260801`; sampling
  used no review outcomes and must not be repeated.
- Next action is fixed: review 39 targeted units, then the frozen 120 controls,
  then run one fixed-point audit for newly created HIGH fight gaps only.
- Train-ready export and training remain blocked by these reviews and the
  corrected-source evidence/feature rebuild.

### Superseded post-v3 audit (2026-08-02)

- The 697-unit targeted v3 consistency rereview is complete: `578 accept`,
  `119 corrected`, zero missing, duplicate, pending, or excluded units.
- Decision and strength ledgers are byte-identical at SHA256
  `e9294ed939dc6cb60dbc95468d9617bb13a290d219e00a21852654a72f310d77`.
- Four newly created HIGH temporal islands remain in
  `Pigs291119_000216_30fps`, tracks 6/7, anchors 378/390. Review only these four
  before the independent residual-control cohort of at least 120.
- V3 contains 352 reviewed partner units outside the primary 2,729 ledger.
  Final composition must union 3,081 unique units and preserve prior notes;
  applying v3 as a simple overwrite of the primary ledger is forbidden.
- Reviewed-data freeze, posture derivation, train-ready export, and training
  remain blocked until micro-review, residual control, corrected-source feature
  rebuild, composite-decision audit, and immutable hashes all pass.

- Current consistency authority is targeted v3: 697 units, with 108/155 v2
  decisions reused, all 13 corrections retained, and 589 reviews remaining.
- v3 supersedes the proximity-broad 1,184-unit v2 scope; v1 and v2 remain
  immutable lineage evidence.

- Status: `PRIMARY_BEHAVIOR_REVIEW_COMPLETE_CONSISTENCY_REREVIEW_REQUIRED`;
  final reviewed-data authority remains blocked.
- Authority: `docs/CLASSIFICATION_V2_CURRENT_STATE.md`, `09_HIDDEN_REVIEW.md`,
  `docs/CLASSIFICATION_V2_POST_REVIEW_LEARNING_PIPELINE.md`, and the approved
  design plan `docs/CLASSIFICATION_V2_BEHAVIOR_POSTURE_BURST_PLAN.md`.
- Validated infrastructure: deterministic residual control sampling with a
  minimum of 120, review-close hash authority, sequential mini-CVAT source
  authority, changed/unchanged feature diagnostics, and integration preflight.
- Accepted behavior/posture direction: one stable behavior target and one
  stable posture target per native burst; preserve direct ten-class behavior;
  posture classes are `upright/sitting/lying`; unresolved posture is masked,
  not forced to a learned class. `social-nose` and `fight` do not imply posture.
- Bounded safe derivation: under the current fixed feeder geometry, a frozen
  reviewed `eat` burst implies `posture=upright`. This does not authorize the
  same derivation for other ROI behaviors or a changed scene authority.
- Preparatory posture contracts are `SYNTHETIC_PASS`: independent masked target,
  frozen/synthetic authority builder, strict proposal calibration, deterministic
  selective-review scope, and guarded CLI are implemented. Real execution stays
  fail-closed until Behavior review is frozen and policy explicitly authorizes it.
- Move/stand review is episode-level: short motion inside continuous locomotion
  is `move`, while local adjustment surrounded by stationary context is `stand`.
  Existing boundary decisions may use the following behavior. Preserve that
  provenance and audit a frozen transition stratum before choosing terminal-state
  or middle-anchor-plus-transition-flag authority for train-ready data.
- Primary evidence: `2,729/2,729` terminal decisions, SHA256
  `3982c7e606f54a5c8d87b795e40c2c775d20fd668213b291ea932c4fecbcc9e3`.
  The ledger was unchanged by the read-only consistency audit.
- The 704-unit v1 scope is superseded because nearest-at-target pairing omitted
  persistent fight partners that had already separated.
- Next permitted action: complete the corrected 1,184-unit temporal encounter
 scope at `behavior_consistency_rereview_3982c7e_20260801_213521_v3`, then
  complete the independent residual control of at least 120 before freeze.
- Required ROI for future recomputation:
  `data/annotations/roi/ROI_annotations.toy_adjusted.coco.json`.
- Forbidden: active ledger reads by the post-review tools, automatic threshold
  changes, review fields in model-X, window reuse, train-ready promotion, model
  training, full OOF, or Q2 claim before all post-review gates pass.

## Tracking

- Status: `FROZEN`.
- Authority:
  `docs/tracking/reconciliation/FOUR_METHOD_TRACKING_FREEZE_AUTHORITY_20260729.json`.
- Current methods: `bytetrack_raw`, `hybrid_bytetrack`, `realtime_fast`,
  `rf_hybrid`.
- Next permitted action: read-only lineage or tracker-boundary audit.
- Forbidden: unseen evaluation, promotion, or detector-weight blame for the
  `000263` regression.

## Repository Hygiene

- Status: `PROTECTED_DIRTY_WORKTREE`.
- Authority: `10_REPO_HYGIENE.md`.
- Preserve unknown, pre-existing, scientific, lineage-bearing, and user-owned
  paths.
- Delete only session-owned, proven-regenerable artifacts after reference
  checks.
