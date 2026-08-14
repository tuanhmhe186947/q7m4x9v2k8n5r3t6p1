# Skill Portfolio Ledger

## 2026-08-12 task manager compaction identity correction

- Root cause: compacted continuations dropped the date/sequence portion of a
  task ID, so distinct `C2V2` tasks both emitted `C2V2-99`.
- `project-state-steward`: recovered and compact-repaired the owned task only;
  immutable archive content and the scientific blocker remained unchanged.
- `safe-refactor-test-guardian` and `skill-creator`: made the continuation ID
  task-scoped, documented the invariant, and added a two-task regression test.
- Evidence: commit `826416c`, `20 passed` manager tests, compile pass, and
  `quick_validate.py` PASS.
- Reuse when: compacting any task whose family prefix may recur on one day.
- Do not reuse when: mutating an unowned or conflicted task; use its manager
  recovery/takeover path first.

## 2026-08-11 Lightning remote-storage scope correction

- `agent-introspection-debugging`: identified a policy failure, not a platform
  failure: a stopped Studio and available CLI access were treated as permission
  to broaden a read-only task, producing an unapproved duplicate archive.
- `project-state-steward`: bound the correction to `GOV-20260811-01` and added
  hard scope, halt, workflow, and regression rules without any Lightning action.
- `agent-self-evaluation`: rejected the earlier narrow diagnosis that mentioned
  only R2 inspection; the material failure was an unauthorized remote copy.
- `agent-harness-construction`: marked for maintenance. Its next review must
  require `READ_ONLY` versus `REMOTE_MUTATION` classification and the complete
  authorization envelope before a remote-capable command is chosen.
- Evidence: `.agents/memory/03_PROJECT_RULES.md`, `08_WORKFLOW.md`,
  `16_HALT_CONDITIONS.md`, `AR-026`, clean Markdown diff/line checks, and the
  unchanged fixture governance suite (`pass^3=1.0`).
- Reuse when: any task can reach Drive, a Studio, or a remote volume.
- Do not reuse when: the task is solely local and has no remote-capable action.

## 2026-08-09 Frozen-XML tracking presentation demo

- `tracking-experiment-guardian`: separated presentation rendering from tracking
  evaluation; used only the frozen four-method manifest and retained each
  method's causal/offline claim boundary.
- `experiment-lineage-reproducibility`: bound the seven MP4s to source video,
  XML manifest, scene ledger, and SHA-256 artifact index without a rerun.
- `computer-vision-opencv`: verified OpenCV writer/reader behavior and validated
  that every emitted MP4 opens at 30 FPS with its expected frame count.
- `video-editing`: structured the hardest-video extension as four full-mode
  exports plus four synchronized 10-second comparison scenes.
- `agent-harness-construction`: recovered from the system NumPy DLL failure by
  retrying in a bounded `uv` environment and preserved the original output root.
- `project-state-steward`: resumed the expired demo task atomically and
  preserved unrelated posture and Classification V2 dirty paths.
- Evidence: `outputs/tracking_demo_20260809/`, fifteen readable MP4s,
  `run_manifest_extended.json`, `artifact_manifest.json`, and
  `TRACKING_DEMO_SCRIPT_VI.md`.
- Reuse when: exporting presentation-only overlays from a frozen XML/video
  authority with no detector or tracker execution.
- Do not reuse when: evaluating, tuning, or claiming new tracking performance.

## 2026-08-08 Classification V2 pre-GPU canonicalization

- `experiment-lineage-reproducibility`: reconciled the intended staged E0 lock,
  hash-bound the transfer inventory/package, and retained the 16-step pilot as
  engineering-only evidence.
- `project-state-steward`: preserved the owned ledger, classified 57 worktrees,
  retired 55 only after orphan checks, and kept the distinct active tracking
  task protected.
- `agent-self-evaluation`: audited the final handoff for authority clarity,
  validation coverage, transfer actionability, and the no-GPU boundary.
- Evidence: staged-lock SHA256 `6b783d…103ca`, transfer descriptor SHA256
  `7899771d…d6476`, focused E0 suite `10 passed`, and
  `pre_gpu_worktree_inventory_20260808.json`.
- Reuse when: preparing a hash-bound, inner-only E0 transfer from canonical main.
- Do not reuse when: scientific selection, outer OOF, or paid execution is in scope.

## 2026-08-07 Classification V2 historical H5 closure

- `scientific-ablation-controller`: preserved a single future matched family:
  T6 versus T6+H5 on identical current target units; no history result was
  promoted and E0 remained unchanged.
- `dataset-contract-leakage-guard`: verified strict pre-target ordering,
  source/video continuity, frozen split binding, zero future access, and no
  history behavior labels in model input.
- `experiment-lineage-reproducibility`: bound the external H5 bundle to the
  reviewed snapshot, split, cohort, temporal contract, and 46D schema.
- `safe-refactor-test-guardian`: covered central T6 legacy offsets, H5 window
  identity, feature export, JSON contracts, compile, focused tests, and a
  two-source loader-forward-backward smoke.
- `project-state-steward`: recovered the same-session H5 task credential and
  reconciled current authority, canonical route, readiness, and protected files.
- Evidence: `temporal_h5_20260807` closure/contract/readiness artifacts, H5
  bundle SHA256 `4449cc67c76b4e0d123c65b0c4c71b77704fce4a47dcf4bd47019d4568149`,
  and 25 focused tests.
- Reuse when: rebinding the existing `COMMON_H5_T6_R` contract to unchanged
  target semantics and current reviewed/split authority.
- Do not reuse when: the T6 anchor, reviewed snapshot, split, or feature
  semantics changes, or when a claim would require H5 model evidence.

## 2026-08-05 Classification V2 pre-GPU plan continuation

- `scientific-ablation-controller`: kept A12, social S0/S1/S2, native OOF and
  permit decisions paired and fail-closed; no practical gain was promoted.
- `experiment-lineage-reproducibility`: bound the current b4536eb run, paired
  outputs, evidence hashes and the isolated continuation handoff.
- `grouped-cv-evaluation`: verified native-unit pairing, four-fold coverage,
  calibration scope and the bounded-only metric boundary.
- `project-state-steward`: recovered and checkpointed `C2V2-20260805-15`,
  preserved the dirty main worktree and recorded protected-authority hashes.
- Evidence: `outputs/classification_v2/model_readiness_audit/`
  `c2v2_plan_continuation_b4536eb_20260805_01`.
- Reuse when: continuing the immutable pre-GPU plan after a bounded gate run.
- Do not reuse when: treating bounded evidence as paper metrics or a GPU permit.

## 2026-08-05 Section 2.8 evidence-guided human review

- `thesis-evidence-writing`: drafted a Vietnamese-first and original English
  methodology subsection that separates candidate generation, human
  adjudication, corrected-source rebuilding and model-window scope.
- `iterative-retrieval`: progressively narrowed the outline, current review
  authority, evidence builders, decision contract and consistency audit before
  writing; no historical result was promoted.
- `project-state-steward`: recorded task `THESIS-20260805-28` as DONE and
  preserved the existing dirty worktree and unrelated untracked candidates.
- Evidence: `docs/thesis_drafts/CHAPTER_2_8_EVIDENCE_GUIDED_HUMAN_REVIEW_VI_DRAFT.md`,
  review-selection/evidence/contract sources, the reproduction contract, current
  state authority, and heading/line/semantic scans.
- Reuse when: drafting a review-method subsection whose candidate rules and
  corrected-source lineage are implementation-bound but paper metrics remain
  separate.
- Do not reuse when: claiming review completion, model performance or a final
  visual without the matching frozen artifact authority.

## 2026-08-05 Classification V2 strict A12 RG-04 closeout

- `scientific-ablation-controller`: kept the user-selected strict four-fold
  A12 scope and recorded S0/S1/S2 as one bounded social family; no promotion
  from engineering-only evidence.
- `experiment-lineage-reproducibility`: verified the refreshed handoff's
  ten-artifact hash manifest and bound RG-04 to commit `e212632c` and its
  fold-run code SHA.
- `grouped-cv-evaluation`: checked paired native-unit coverage, four-fold
  identity, missing-class reporting, and the `paper_facing_ready=false`
  boundary.
- `safe-refactor-test-guardian`: reran `31` focused tests, compileall, Ruff,
  diff-check, and the changed-file line scan on the isolated candidate.
- `project-state-steward`: recovered and checkpointed task
  `C2V2-20260805-08` at `A12-03 DONE`; preserved main, ledgers, labels,
  splits, and untracked cleanup candidates.
- Evidence: refreshed handoff
  `rg03_readiness_handoff_e212632_20260805_095700`, independent check PASS,
  ten of ten artifact hashes PASS, and RG-04 report SHA256
  `3e022bb7bcf9a7eb6ee2981791e8df4d97e7f419f9f7867b65c49e576e41e2a0`.
- Reuse when: a bounded paired social diagnostic must be reconciled without
  changing the strict A12 source-support scope.
- Do not reuse when: claiming paper metrics, authorizing paid GPU, or treating
  single-source folds as source-balanced evidence.

## 2026-08-04 crash-aware managed-task recovery

- `agent-architecture-audit`: separated provable same-thread identity from
  unreliable PID, process-list, and inactivity heuristics.
- `agent-introspection-debugging`: traced the blocking behavior to a lost private
  token with no authenticated active-lease recovery path.
- `agent-harness-construction`: added narrow `recover` and `admin-takeover`
  actions with deterministic observations, stop conditions, CAS, and audit.
- `plan-orchestrate`: kept implementation, governance, regression, and live
  revalidation as separate checkpoints.
- `safe-refactor-test-guardian`: required process races, token invalidation,
  confirmation/CAS failures, compile, JSON, lint, and focused regressions.
- `project-state-steward`: upgraded to local v5 and recovered the live task from
  revision 5 to 6 under the same `CODEX_THREAD_ID` without lease expiry.
- Root cause: the manager could not authenticate a resumed crashed session after
  its private token was lost, so every active lease remained fail-closed.
- Validated correction: bind tasks to `CODEX_THREAD_ID`; allow same-thread token
  rotation under lock and fresh CAS; require exact user-authorized administrative
  takeover for every different or unbound thread; hash-bind each ownership event.
- Evidence: `14` manager and `39` final governance tests, `25` governance tasks
  over three fixture runs, negative control `pass^3=0.0`, one live audit event,
  and old-token rejection.
- Reuse when: a Codex process crashes and the task ledger survives with intact
  task ID, worktree, revision, hash, and runtime binding.
- Do not reuse when: runtime identity differs or is absent, task bytes drifted,
  worktree changed, or user authorization does not bind the exact inspected state.

## 2026-08-04 Classification V2 social Top-K K=3

- `scientific-ablation-controller`: kept S2 as the only changed social
  representation family and preserved the paired S0/S1/S2 gate before S3.
- `multimodal-sequence-model-builder`: enforced `[T,3,10]` partner tensors,
  per-partner encoding, masked valid-count pooling, and separate masks.
- `safe-refactor-test-guardian`: required hash-bound real artifacts, tamper
  rejection, `62` focused tests, Ruff, compile, JSON, and diff checks.
- `project-state-steward`: resumed the owned S2 task through the audited
  one-time takeover and preserved the dirty main worktree and frozen lineage.
- Evidence: commits `a8f727a5`, `5bfc7180`, and `6c2f2049`; manifest SHA256
  `694948f570dfde2c6771efc2ad46f8a79bd0ba20ccda5768d18af631b9139119`;
  compatibility SHA256
  `cd2eb003eb40fc2b0dae3eff0c7ce0ba2953b0fab6f9be868f8796d118f2cc37`.
- Reuse when: adding one optional modality with explicit masks and a paired
  ablation gate on the same reviewed lineage.
- Do not reuse when: claiming S2 performance, authorizing S3/GAT, or changing
  the reviewed lineage, social schema, split, or training-mass contract.

## 2026-08-03 post-review frame-label amendment

- `agent-architecture-audit`: separated the human frame boundary from the
  six-frame native-unit training contract and avoided forcing a mixed unit.
- `dataset-contract-leakage-guard`: required unit exclusion, zero weight, and
  invalidation of every dependent cross-label window.
- `experiment-lineage-reproducibility`: bound the amendment to reviewed-frame,
  window-manifest, snapshot, and visual-evidence hashes without overwriting.
- `scientific-ablation-controller`: froze S0/S1/S2 as one social-representation
  family and kept S3 blocked until the predeclared S2 gate passes.
- `safe-refactor-test-guardian`: required independent real-data audit, focused
  tests, CPU loader/backward checks, and unchanged 46D/shard contracts.
- `project-state-steward`: checkpointed the task, preserved prior authorities,
  superseded snapshot V3 for future training, and froze amendment V1.
- Evidence: exact f180-f186 `playwithtoy`, f187-f191 `stand`; eight affected
  windows, three newly invalidated, zero affected nonzero-weight windows,
  independent audit SHA256
  `97673d3c9e7bf9df78ee409a9bcb8b7575396f18f1b8d3191d7e50ee675a1ede`,
  and snapshot SHA256
  `ab86e2e04267cfdc8248f9bdb8774615479d67a3589f7a25844bb1a4c93a639e`.
- Reuse when: a verified post-review frame boundary splits one native unit.
- Do not reuse when: the native unit remains behavior-stable or the correction
  can be represented safely by the normal review-unit ledger.

## 2026-08-03 playwithtoy boundary review evidence

- `agent-harness-construction`: bounded the action to one immutable 12-frame
  case and returned deterministic paths, hashes, and verification fields.
- `computer-vision-opencv`: decoded exact zero-based frames and rendered
  full-frame and consistent-crop sheets with bbox and adjusted toy ROI overlays.
- `project-state-steward`: tracked authority, generation, verification, and
  cleanup without changing reviewed labels, ROI authority, or annotations.
- Evidence: four readable 4x3 PNG sheets and one hash-bound JSON manifest;
  all output hashes matched independently and f191 was the sole no-contact frame.
- Reuse when: one bounded reviewed-frame decision needs visual confirmation.
- Do not reuse when: interactive editing or a new decision ledger is required.

## 2026-08-03 Classification V2 model-readiness audit

- `dataset-contract-leakage-guard`: verified reviewed spatial lineage, 12D/46D
  order, pair/mask semantics, zero forbidden-X fields, zero future dependence,
  zero trainable cross-label windows, and pure grouped splits. The follow-up
  correction traced the complete Hidden coverage/apply lineage and the later
  Behavior-note exclusion audit instead of requiring direct snapshot binding.
- `agent-introspection-debugging`: identified the category error: absence of a
  direct snapshot path was incorrectly treated as absence of proven review,
  despite complete applied-lineage evidence.
- `scientific-ablation-controller`: separated spatial contract evidence from
  real RGB B0-B3 readiness, kept BALANCED fail-closed, and blocked all training
  campaigns while prerequisite gates fail.
- `experiment-lineage-reproducibility`: used one unique audit directory, clean
  code authority, reviewed snapshot hashes, memmap shard hashes, bounded run
  artifacts, checkpoint parity, and a 60-artifact hash manifest.
- `project-state-steward`: preserved the dirty main worktree, tracked task
  `C2V2-20260803-14` plus correction task `C2V2-20260803-15` through atomic
  checkpoints, and recorded no production-code or review-decision change.
- Evidence: eight reviewed T/source forward-backward strata, reviewed spatial
  tiny overfit and resume parity, synthetic CUDA B0-B3 runtime, a bounded
  date-held-out source probe, and `121` focused tests pass.
- Reuse when: deciding whether a reviewed Classification V2 snapshot may enter
  baseline or ablation training.
- Do not reuse when: Hidden manifest/decision/apply hashes, RGB cache, weight
  policy, schema, split, or code SHA changes; rerun the live audit instead.

## 2026-08-03 Hidden visibility weighting confirmation

- `scientific-ablation-controller`: kept one changed family, gamma 0.5 as the
  sole candidate, fixed seeds/split/exposure, and no automatic promotion.
- `dataset-contract-leakage-guard`: verified unchanged grouped validation,
  zero Hidden/review fields in model-X, and zero label or canonical-tensor edits.
- `experiment-lineage-reproducibility`: bound clean run/evaluator SHAs, design
  and input hashes, three run inventories, checkpoints, and summary inventory.
- `safe-refactor-test-guardian`: isolated the evaluator, added inventory and
  dependency regressions, and passed 183 focused tests, Ruff, compile, line,
  diff, and completion audits.
- `project-state-steward`: retained the invalid v1 run as failure evidence,
  recorded the no-promotion decision, and preserved the active review authority.
- Root cause: inserting robust rows into one NumPy permutation changed every
  main-row batch order; the first evaluator also assumed the wrong inventory
  key and an undeclared optional `tabulate` dependency.
- Validated correction: use a separate deterministic robust stream with a
  bit-exact main projection, verify the current `artifacts` inventory schema,
  render Markdown internally, and bind evaluator SHA plus clean worktree.
- Evidence: three valid five-epoch seeds PASS; gamma-zero is 8/8 for each seed;
  summary is `INCONCLUSIVE` with `10/10` artifact hashes and no promotion.
- Reuse when: adding a very small weighted cohort to a paired training arm or
  aggregating current Classification V2 run inventories.
- Do not reuse when: changing optimizer exposure, class weights, grouped split,
  label authority, or using Hidden as model input or behavior uncertainty.

## 2026-08-03 posture GUI target-context validation

- `agentic-engineering`: captured a failing validator baseline before the
  minimal fix and compared the exact real-item result afterward.
- `safe-refactor-test-guardian`: limited changes to the posture GUI validator
  and focused tests; protected the scope, ledger schema, and resume contract.
- `project-state-steward`: preserved the 85 saved decisions and kept this GUI
  correction separate from scientific posture or behavior authority.
- Root cause: optional context diagnostics were treated as incomplete TARGET
  media, while the actual `missing_video` TARGET diagnostic was not recognized.
- Validated correction: classify row and media failures by TARGET frame; keep
  context failures as warnings and fail closed for missing TARGET evidence.
- Evidence: 39 GUI tests, Ruff, compile, line and diff checks passed; real item
  0086 has zero TARGET blockers and the posture ledger hash is unchanged.
- Reuse when: a review view adds optional context beyond immutable target frames.
- Do not reuse when: any TARGET frame row or TARGET media is genuinely missing.

## 2026-08-03 motion episode and boundary diagnostic

- `scientific-ablation-controller`: held folds, seed, 46D reference and one
  motion-representation family fixed across seven paired variants.
- `dataset-contract-leakage-guard`: enforced explicit review-independent
  sidecar columns, no legacy pig-ID linking, NaN plus validity, no control 120,
  and matched source-availability controls.
- `experiment-lineage-reproducibility`: bound exact inputs and sidecar hashes,
  detected CSV precision instability, and required byte-identical smoke and
  exact-SHA OOF replay.
- `safe-refactor-test-guardian`: scoped isolated configs/module/CLI/tests;
  `15/15` focused tests, Ruff, compile, line and diff checks passed.
- `project-state-steward`: retained phase evidence and prevented a possible
  development signal from becoming a promoted selector or paper claim.
- Reuse when: evaluating review-independent temporal feature families against
  the frozen post-review selector with matched missingness controls.
- Do not reuse when: training the final behavior model, consuming the reserved
  control for tuning, or treating source-context availability as behavior.

## 2026-08-03 evidence-based memory maturity

- `agent-architecture-audit` and `plan-orchestrate`: separated operational
  completion, evidence maturity, acceptance, publication, and later reopening.
- `iterative-retrieval` and `knowledge-ops`: kept one canonical fact home and
  made the long surface a generated, progressively retrieved dossier.
- `agent-harness-construction`: implemented typed actions, lock/CAS, atomic
  writes, structured observations, safe retry, and crash-repair synthesis.
- `agent-eval` and `eval-harness`: added AR-021 through AR-023 and three-run
  positive/negative controls for age, revalidation, and dual authority.
- `skill-creator`: upgraded and validated `project-state-steward` local v4.
- `safe-refactor-test-guardian`: scoped deterministic tests, Ruff, compile,
  JSON, line, and diff gates without touching scientific execution.
- `project-state-steward`: owns candidate registration, review, promotion,
  reopening, archive, and dossier synthesis through the maturity manager.
- Evidence: nine maturity tests, two-process race, 23 focused governance tests,
  23-task fixture `pass^3=1.0`, required negative control `pass^3=0.0`, Ruff,
  compile, skill validation, and maturity audit.
- Reuse when: completed project knowledge may be useful across future sessions.
- Do not reuse when: the item is unfinished, transient, unsupported, or still
  active in medium memory.

## 2026-08-03 atomic multi-session resume harness

- `agent-architecture-audit`: identified shared-Markdown lost-update, stale
  ownership, and cross-day working-set loss as wrapper/persistence failures.
- `agent-harness-construction`: specified lock, private owner token, worktree,
  lease, revision/hash CAS, atomic replace, and structured error observations.
- `iterative-retrieval` and `knowledge-ops`: retained active task capsules in
  short memory while keeping daily history TTL-bound and medium nonduplicative.
- `agent-eval` and `eval-harness`: added process races, crash-release, takeover,
  hash-drift, and multi-day resume regression coverage plus `AR-019/AR-020`.
- `project-state-steward`: upgraded to local v3 with the atomic task manager.
- Reuse when: two chats work concurrently or a material task spans days.
- Do not reuse when: simple read-only Q&A creates no durable task state.

## 2026-08-03 short-memory task checklist

- User correction: short memory stored narrative and superseded history but did
  not expose which planned prompt steps were done, open, blocked, or evidenced.
- `iterative-retrieval` and `knowledge-ops`: resolved the lifecycle authority
  progressively and separated current task state, one-day closeout, medium
  carryover, and durable knowledge.
- `agent-harness-construction`: defined task/step states, crash-consistent
  checkpoints, evidence recovery, rollover, and the bounded line budget.
- `skill-creator`: upgraded `project-state-steward` to local v2 and validated
  its metadata and bundle.
- `agent-self-evaluation`: completed the final requirement and residual-risk
  check; fixture determinism is not claimed as live-agent reliability.
- Evidence: short-memory live checklist, governance validator tests, AR-015
  through AR-018 three-run fixture cases, skill validation, and bundle hashes.
- Reuse when: a prompt needs a multi-step plan or material effects that must
  survive a crash, new chat, or calendar-day rollover.
- Do not reuse when: the prompt is simple read-only Q&A or command-level logging
  would add noise without changing task state.

## 2026-08-02 Tracking as a separate scientific component

- User direction: identity tracking must have its own scientific question
  (RQ2), methodology, evaluation, runtime analysis, failure analysis and
  downstream error-propagation boundary.
- `thesis-evidence-writing`: expanded the consolidated outline so tracking is
  not reduced to a standard-tracker implementation note and so causal/offline
  semantics remain distinct.
- `academic-paper`: preserved the five-question research structure and linked
  RQ2 to Sections 2.4 and 3.5 without inventing tracker results.
- Evidence: attachment `5A. REQUIRED IDENTITY-TRACKING CONTENT`, tracking
  freeze authority, revised outline, and governance validation.
- Reuse when: identity continuity is a prerequisite for actor-conditioned
  temporal analysis or long-term profiles.
- Do not reuse when: a study has no identity-bearing temporal task and tracking
  is genuinely outside its scientific scope.

## 2026-08-02 Consolidated thesis outline with detection and legacy cohort

- User correction: “supplementary” refers to temporal/day/video diversity, not a
  dataset excluded from training. Root cause was confusing a source's role in
  diversity analysis with its membership in the merged training dataset.
- Validated correction: describe historical legacy bursts and the newer source
  as merged behavior data for training, while reporting their provenance and
  diversity contribution separately; retain mapping, quality and grouped-split
  checks before promoting a merged snapshot.
- Reuse when: a historical and current behavior source are intentionally
  combined before training and the thesis must explain why both are present.
- Do not reuse when: a source is explicitly evaluation-only or a separate
  robustness cohort by the registered experiment contract.
- `thesis-evidence-writing`: organized a Vietnamese-first, evidence-aware
  outline that separates detection data construction, historical behavior-burst
  provenance, primary review authority, supplementary legacy diversity, and
  final results. It also bound each planned visual to a source or `PENDING`
  status.
- `academic-paper`: applied the outline/plan conventions and academic-writing
  anti-pattern checks without translating prose or inventing results.
- `agent-architecture-audit`: checked the outline for authority-layer
  contamination, especially the risk of treating notebook history, review
  metadata, or detector output as downstream scientific evidence.
- `plan-orchestrate`: used for step decomposition and dependency ordering only;
  no orchestration command was executed.
- `agent-self-evaluation`: completed the final handoff scorecard; accuracy and
  completeness were checked against the attachment requirements, while long
  table rows remain a readability trade-off of the requested outline format.
- `project-state-steward`: reconciled the new outline and skill records, kept
  the protected dirty worktree intact, and deferred all unrelated cleanup.
- Evidence: `docs/thesis_drafts/PIG_BEHAVIOR_COMPLETE_THESIS_OUTLINE_WITH_DETECTION_V1.md`,
  notebook inspection summary, `git diff --check`, line scan, and memory
  contract validator PASS.
- Reuse when: methodology structure must incorporate detection provenance and
  historical behavior data while keeping final claims artifact-bound.
- Do not reuse when: final manifests and evaluator artifacts have already been
  reconciled; then write the corresponding manuscript sections and results.

## 2026-08-02 Section 2.3 annotation and review protocol draft

- `thesis-evidence-writing`: drafted a Vietnamese methodology subsection that
  presents human review as annotation quality control, separates review layers
  from model inputs, and keeps quantitative counts out of the prose until the
  final dataset authority is reconciled.
- `project-state-steward`: preserved the provisional status of the review
  authority and did not promote internal ledger counts to thesis results.
- Evidence: `docs/thesis_drafts/CHAPTER_2_3_BEHAVIOR_ANNOTATION_REVIEW_VI_DRAFT.md`,
  the post-review learning pipeline, GUI guide, review-close authority, and
  fixed-point audit.
- Reuse when: a thesis section must describe annotation review without turning
  reviewer metadata or selector diagnostics into model evidence.
- Do not reuse when: a final reconciled dataset snapshot and evaluator have
  already been bound for Chapter 3 reporting.

## 2026-08-02 Section 2.2 representation-scope correction

- `thesis-evidence-writing`: removed CVAT/legacy burst-history from the main
  data subsection and retained only the time-ordered actor-track linkage needed
  to explain model inputs and reproducibility.
- Root cause: an internal annotation-transfer detail had been mistaken for a
  reader-facing data representation or model design choice.
- Evidence: user review of `docs/thesis_drafts/CHAPTER_2_2_DATA_SOURCES_NATIVE_UNITS_VI_DRAFT.md`,
  revised heading, and diff/line checks.
- Reuse when: implementation history does not change the scientific input,
  target, or evaluation semantics.
- Do not reuse when: annotation-unit semantics directly determine labels,
  sampling, leakage control, or an evaluated experimental factor.

## 2026-08-02 Section 2.2 timing and visibility refinement

- `thesis-evidence-writing`: integrated the user-confirmed distinction between
  source timing and MP4 playback timing, timestamp evidence, variable pig
  visibility, and the non-guaranteed identity support of back markings.
- `project-state-steward`: preserved the existing dirty worktree and recorded
  the correction only in the thesis draft and skill ledger.
- Evidence: `docs/thesis_drafts/CHAPTER_2_2_DATA_SOURCES_NATIVE_UNITS_VI_DRAFT.md`,
  `Help_Pigs291119_000226_30fps/times.txt`, and Markdown diff/line checks.
- Reuse when: describing processed video whose frame count, playback FPS and
  biological observation time differ, or when group size is an upper bound.
- Do not reuse when: timestamps or source-rate semantics are not verified for
  the specific processed clip family.

## 2026-08-02 Section 2.2 data-source draft

- `thesis-evidence-writing`: drafted the Vietnamese data-source subsection
  around study context, source-time conversion, native units and model-window
  semantics, with explicit evidence statuses and Figure 3/Figure 5 anchors.
- `project-state-steward`: preserved the distinction between user-confirmed
  acquisition facts and pending final manifests.
- Evidence: `docs/thesis_drafts/CHAPTER_2_2_DATA_SOURCES_NATIVE_UNITS_VI_DRAFT.md`,
  the thesis blueprint, figure plan, and Markdown line/diff checks.
- Reuse when: a methodology subsection must describe data and timing without
  turning pending manifests or results into claims.
- Do not reuse when: final train-ready counts or evaluator outputs are not bound
  to immutable artifacts.

## 2026-08-02 RGB/depth thesis visual boundary

- `thesis-evidence-writing`: refined Section 2.1 from project-explanation prose
  to a methods-facing overview centered on model inputs, temporal windows, and
  evidence boundaries; retained the RGB/depth input caveat.
- `figure-designer`: audited Figure 2 and Figure 3 roles so acquisition media
  are not mistaken for current model features.
- `computer-vision-opencv`: inspected real RGB scene and actor-crop media and
  confirmed they are suitable candidate study visuals.
- `project-state-steward`: reconciled the correction in the thesis memory and
  ran the memory-contract validator.
- Evidence: `docs/thesis_drafts/CHAPTER_2_1_OVERVIEW_FRAMEWORK_VI_DRAFT.md`,
  `docs/thesis_drafts/THESIS_FIGURE_AND_TABLE_PLAN.md`,
  `docs/CLASSIFICATION_V2_THESIS_BLUEPRINT_EVIDENCE_MAP.md`, two inspected
  media files, and validator PASS with governance tests `6/6`.
- Reuse when: a captured sensing modality is available but its contribution
  has not been demonstrated in the current paper-facing model.
- Do not reuse when: a registered ablation or model-input whitelist proves the
  modality is part of the evaluated feature set.

## 2026-08-02 thesis blueprint and evidence map

- `academic-paper`: reasoning and synthesis; mapped the retained thesis
  template to an English thesis outline and separated protocol, evidence, and
  unsupported result claims.
- `documents`: document-template execution; inspected the retained DOCX
  structure, section layout, headings, tables, and figures. Render QA was
  attempted but LibreOffice was unavailable in the environment.
- `project-state-steward`: governance; recorded the newer review artifacts as
  provisional evidence because older current-state summaries still require
  reconciliation.
- Evidence: `docs/CLASSIFICATION_V2_THESIS_BLUEPRINT_EVIDENCE_MAP.md`, memory
  contract validator PASS, and project governance tests `6/6`.
- Reuse when: a thesis or paper must be drafted from a retained document
  template while data and result authorities are still evolving.
- Do not reuse when: the final model, evaluation, or claim registry is being
  asserted without its bound artifact hashes.

## 2026-08-02 external academic-skill comparison

- `skill-creator`: created and validated the project-local
  `thesis-evidence-writing` skill after identifying a real gap between generic
  paper skills and this thesis's Vietnamese-first, evidence/visual-gated
  workflow.
- `academic-paper`, `figure-designer`, and `pre-submission-reviewer`: used as
  overlap checks against the two external repositories; no external skill text
  or bundled code was copied.
- Evidence: repository snapshots at Supervisor-Skills `aff5de9` and
  Academic Research Skills `32823c3`, license review, and validator PASS.
- Reuse when: drafting or converting a thesis subsection while data authority,
  visual anchors, and user confirmation must remain explicit.
- Do not reuse when: editing source data, running experiments, or promoting a
  scientific result without its bound authority.

## 2026-08-02 review-close fixed-point gate

- `agent-architecture-audit`: separated targeted-analysis, independent-control,
  final-label, and fixed-point authorities instead of treating all reviewed rows
  as one interchangeable ledger.
- `dataset-contract-leakage-guard`: preserved review metadata outside model-X
  and blocked train-ready rebuilding while HIGH temporal inconsistency remains.
- `experiment-lineage-reproducibility`: bound both completed ledgers, the 3,121
  and 3,241 compositions, adjusted ROI, source evidence, and context view by
  exact hashes and paths.
- `scientific-ablation-controller`: translated the user requirement into a
  frozen-parent, one-family-at-a-time optimization contract; transfer remains a
  later distinct evaluation authority.
- `safe-refactor-test-guardian`: added a reversible HIGH-only residual selector
  flag without hiding the full findings table; compile, help, and `5/5` focused
  tests passed.
- `project-state-steward`: recorded the two-unit fixed-point blocker and
  reproducible optimization boundary without promoting the 3,241 candidate.
- Reuse when: completed review layers must close under a bounded temporal
  fixed-point before corrected-source rebuild or selector learning.
- Do not reuse when: a temporal gap is to be relabeled automatically, or review
  metadata is proposed as a predictive model input.

## 2026-08-02 residual-review context and resume correction

- `intent-driven-development`: fixed acceptance conditions around longer
  presentation context, immutable six-frame target scope, and resumable keys.
- `safe-refactor-test-guardian`: compared all IDs, temporal keys, target frames,
  labels, and order before allowing the replacement presentation views.
- `project-state-steward`: registered the presentation correction without
  modifying completed decisions, source annotations, or frozen membership.
- Root cause: the residual pipeline opened target-only scope CSVs directly and
  bypassed the standard final Behavior presentation-view builder.
- Validated correction: build presentation-only views with sparse context and
  continuous CVAT playback up to ±90 frames while reusing the exact session.
- Evidence: targeted `39/39` and control `120/120` preserve their keys, targets,
  labels, and order; 39 targeted and 102 control CVAT rows gained context;
  `31/31` focused GUI tests and two frame-cache builds passed.
- Reuse when: any derived Behavior scope is about to be opened in the final
  review GUI.
- Do not reuse when: adjacent legacy context lacks trusted actor/full-scene
  authority or presentation context is proposed to expand decision scope.
- Maintenance trigger: final-view context radius, resume key, decision target,
  legacy identity authority, or residual-scope builder changes.

## 2026-08-02 composite authority and residual discovery

- `agent-architecture-audit`: separated sequential review-layer semantics,
  review-informed suspicion discovery, independent controls, and final label
  authority so a later `accept` cannot erase an earlier correction.
- `dataset-contract-leakage-guard`: kept selection reasons, review outcomes,
  ranks, paths, and review identifiers outside model-X; no residual finding
  changed a label automatically.
- `project-state-steward`: registered the composite, targeted scope, control
  scope, superseded trials, and exact next gates without cleaning the dirty
  worktree or modifying completed ledgers.
- Root cause: raw row overwrite is not a valid composition rule when later
  review layers contain `accept`, and distant corrections can over-activate a
  long temporal gap if adjacency is not required.
- Validated correction: apply review layers sequentially relative to the
  current effective label and activate a residual gap only when its adjacent
  boundary unit or an in-gap unit was corrected.
- Evidence: `3,082` unique reviewed keys, `447` final source corrections,
  `39` disjoint targeted units, a disjoint frozen `120`-unit control, and
  `54/54` focused tests at commit `faee589`.
- Reuse when: multiple review layers overlap or reviewed corrections are used
  to discover unreviewed temporal inconsistencies.
- Do not reuse when: a proxy finding is being treated as label truth, review
  metadata is proposed for model-X, or a new control is sampled after outcomes.
- Maintenance trigger: review decision semantics, temporal adjacency, label
  ontology, control estimand, or composite application order changes.

## 2026-08-02 completed-v3 post-review audit

- `agent-architecture-audit`: separated structural ledger completion from
  temporal anomaly closure, stale GUI evidence, and composite-authority state.
- `dataset-contract-leakage-guard`: proved that v3 adds 352 reviewed partner
  units outside the primary ledger and blocked a lossy row-for-row overwrite.
- `project-state-steward`: rolled expired short memory forward and recorded the
  verified four-unit blocker without promoting reviewed labels to train-ready.
- Evidence: 697/697 v3 decisions and quality rows, no duplicate or missing keys;
  post-review audit over 245,680 frames and 33,355 units; governance tests 6/6.
- Reuse when: a derived rereview scope can add keys or alter neighboring labels.
- Do not reuse when: source-relative composite mapping and all residual controls
  have already been frozen under a newer authority.

## 2026-08-01 Behavior consistency v3 refinement

- `agent-architecture-audit`: challenged whether temporal nearest-history was a
  valid proxy for direct fight participation instead of accepting v2 ranking.
- `dataset-contract-leakage-guard`: preserved primary/v2 ledgers, reused decisions
  only by exact review-unit identity, and kept review selection outside model-X.
- `project-state-steward`: routed the validated selector correction and protected
  prior authorities while recording the new v3 handoff.
- Root cause: v2 admitted up to three pigs that were nearest anywhere within
  +/-90 frames, so nearby bystanders entered synchronized target bursts without
  evidence that they participated in the target interaction.
- Validated correction: retain every actor; retain only non-fight partners with
  bidirectional support plus either nearby same-track fight or strong target-span
  support. Seed a new session with exact retained decisions.
- Evidence: 155 v2 decisions contained 142 accepts and 13 corrections to fight;
  v3 retained all 13, reduced 1,184 to 697, reused 108, and passed 39 focused
  tests, Ruff, compile, contract, hash, and resume-position checks.
- Reuse when: a broad encounter graph is used to locate missing fight labels in
  crowded single-camera bursts and prior decisions must survive scope refinement.
- Do not reuse when: partner support is treated as a behavior label, for a new
  camera/pen authority, or before thresholds are re-audited on that population.
- Supersedes: unfiltered retention of all ranked v2 temporal candidates.

## 2026-08-01 Behavior interaction-consistency re-review

- `agent-architecture-audit`: separated immutable primary decisions, derived
  anomaly findings, targeted re-review decisions, and later reviewed-data
  integration so no derived flag silently becomes label authority.
- `dataset-contract-leakage-guard`: kept review reasons, linked keys, ranks,
  paths, and consistency findings outside model-X and train-ready authority.
- `project-state-steward`: superseded the flawed 704-unit authority with the
  current 1,184-unit v2 scope while preserving v1 and its possible decisions.
- Root cause: nearest-at-target changed from the persistent partner to a
  bystander after the interacting pig separated, so the paired GUI could omit
  the actual episode participant.
- Validated correction: rank bidirectional nearest-history candidates over the
  context window and append their synchronized units before current-nearest or
  boundary context; candidate status never changes labels automatically.
- Evidence: the real f1794 case ranks ID_2 with 93 supporting frames before
  ID_8 with 16; focused GUI and audit suite passed `38/38`; the primary ledger
  remained byte-identical at SHA256
  `3982c7e606f54a5c8d87b795e40c2c775d20fd668213b291ea932c4fecbcc9e3`.
- Reuse when: interaction participants can separate before the reviewed burst.
- Do not reuse when: candidate rank is being treated as behavioral truth.
- Maintenance trigger: interaction ontology, primary ledger schema, temporal
  composite-key format, GUI ordering, or residual-control contract changes.

## 2026-08-01 Mini-CVAT canonical video filename resolution

- Root cause: the finite fallback list omitted the canonical CVAT video suffix
  `_30fps` and canonical `Pigs` casing when `video_key` omitted both.
- Validated correction: extend only the bounded flat-file guesses while keeping
  direct source paths, nested `color.mp4`, and prior flat filenames unchanged.
- Evidence: real `pigs301119_000327` resolved to
  `Pigs301119_000327_30fps.mp4`; focused tests `5/5`, compile and Ruff passed.
- Reuse when: a CVAT `video_key` maps to the project's canonical flat video
  filename convention.
- Do not reuse when: multiple source videos exist or the source uses a different
  naming contract; those cases must remain fail-closed.
- Skills: `safe-refactor-test-guardian` for bounded compatibility coverage and
  `project-state-steward` for durable correction routing.

## Machine Authority

`11_SKILL_PORTFOLIO.json` is the versioned portfolio and routing authority.
The Markdown file remains a concise human ledger; it cannot replace file hash,
dependency, review-age, proof-task, or stale-signal checks.

## 2026-08-01 behavior-posture burst architecture plan

- `agent-architecture-audit`: traced the existing auxiliary posture head from
  configuration through loader, model, and loss, and separated implementation
  presence from correct semantic authority.
- `plan-orchestrate`: decomposed the accepted behavior/posture design into
  authority, proposal, selective review, export, model, and validation gates.
- `dataset-contract-leakage-guard`: kept posture labels and review metadata out
  of model-X and preserved native-unit/grouped-split boundaries.
- `multimodal-sequence-model-builder`: preserved direct ten-class supervision
  while defining an independent masked three-class posture head.
- `safe-refactor-test-guardian`: defined version, migration, cache-reuse, and
  focused-test boundaries for the later implementation.
- `project-state-steward`: routed the approved plan into current handoff state.
- Implementation evidence: the same skill set produced the masked posture
  contract, safe derivation and proposal gates, deterministic selective-review
  scope, guarded CLI, and `20` passing focused tests without active-ledger access.
- Evidence: accepted user semantics and
  `docs/CLASSIFICATION_V2_BEHAVIOR_POSTURE_BURST_PLAN.md`; no source or active
  review authority was read or changed.
- Maintenance trigger: posture ontology, native-burst/window semantics,
  auxiliary target schema, balanced loss, or review-close authority changes.

## 2026-08-01 Behavior GUI CVAT prefetch latency

- Root cause: the idle callback decoded sparse CVAT contact-sheet frames with
  repeated codec seeks on the Tk thread; Windows reported Not Responding.
- Validated correction: use one latest-only daemon prefetch worker, isolated
  captures, a thread-safe two-sheet cache, and bounded sequential frame decode.
- Evidence: the actual 2,100-row three-CSV write took `0.038 s`; one affected
  23-frame sheet decoded exactly in `3.466 s` off-thread; `79` tests passed.
- Reuse when: derived review media can be prepared independently of Tk widgets
  and all mutable decoder/cache state has an explicit concurrency boundary.
- Do not reuse when: rendering touches Tk objects, decisions, active ledgers,
  shared `VideoCapture` instances, or unbounded queues/caches.
- Skills: `latency-critical-systems` for hot-path measurement,
  `safe-refactor-test-guardian` for bounded regression coverage, and
  `project-state-steward` for durable correction routing.

## 2026-08-03 Classification V2 spatial low-memory gate

- `agent-architecture-audit`: reasoning; separated repository tensor loading
  from unrelated Codex service memory and traced NPZ, hashing, batch, and model
  boundaries. `plan-orchestrate` was read only for mandatory architecture
  routing coverage and was not represented as an implementation executor.
- `latency-critical-systems`: split startup integrity hashing, peak RAM,
  batch-copy size, and fallback behavior instead of treating speed as one
  metric.
- `safe-refactor-test-guardian`: preserved 46D order, mask semantics, row
  counts, hashes, and full-OOF caller compatibility through focused tests.
- `project-state-steward`: recorded the validated correction without staging
  unrelated dirty root changes or deleting application-owned processes.
- Evidence: commit `a034440e`; `128` focused tests, Ruff, compile, real
  165,305-row memmap forward, finite backward, tiny overfit, checkpoint reload,
  and optimizer-resume parity pass.
- Maintenance trigger: spatial NPZ/memmap schema, array names, hash policy,
  full-memory threshold, trainer batch indexing, or mask inputs change.

## 2026-08-01 Classification V2 GUI operator entrypoint

- `intent-driven-development`: reasoning; fixed observable launcher safety,
  resume, session isolation, dry-run, and explicit source-apply boundaries.
- `safe-refactor-test-guardian`: verified V1 callers before deletion and kept
  V1 sidecar migration compatibility while retaining V2 coverage.
- `project-state-steward`: reconciled the durable operator guide and preserved
  unrelated dirty memory, authority, review, and source-data paths.
- Evidence: focused V2 GUI/sidecar/editor/apply tests, Ruff, compile, command
  dry-runs, input status, legacy-caller search, line scan, and diff check pass.
- User correction: one shared launcher was mistaken for one available GUI;
  `list-guis` now separates active, newly blocked, closed, and internal types.
- Maintenance trigger: GUI CLI arguments, current review/source authority,
  adjusted ROI authority, or mini-CVAT sidecar/apply contract changes.

## 2026-08-01 Classification V2 post-review learning pipeline

- `agent-architecture-audit`: reasoning; separated active review, frozen review,
  corrected-source, diagnostic-learning, rebuild, and training authorities.
- `plan-orchestrate`: reasoning; ordered dependencies so no outcome is read
  before review close and no rebuilt window is reused.
- `dataset-contract-leakage-guard`: guardrail; excluded decision, quality,
  selector, path, ID, source, and label fields from model-X.
- `experiment-lineage-reproducibility`: lineage; bound scopes, decisions,
  quality, mini-CVAT chains, ROI, and rebuilt features by exact hashes.
- `scientific-ablation-controller`: kept weighted selector and feature contrasts
  diagnostic and prohibited automatic logic changes.
- `safe-refactor-test-guardian`: covered deterministic sampling, coverage,
  conflicts, source chains, leakage rejection, and active-ledger paths.
- `project-state-steward`: reconciled current state and workflow while
  preserving pre-existing dirty and user-owned paths.
- Evidence: `41` focused tests, Ruff, compile, CLI parser probes, line scan, and
  fail-closed synthetic contracts passed; no production review data was read.
- Maintenance trigger: review CSV schema, identity apply manifest, ROI authority,
  feature schema, or final reviewed-Q2 contract changes.

## 2026-07-31 project autoresearch harness

- `andrej-karpathy-skills`: reasoning; mapped the pinned upstream experiment
  loop to a project-owned candidate/policy/ledger boundary.
- `agent-harness-construction`: reasoning; defined action space, observation,
  authorization, timeout, recovery, and stop contracts.
- `tracking-experiment-guardian`: domain guardrail; enforced frozen authority,
  lineage, Standard V2, Hidden, baseline, per-video, and zero-MP4 gates.
- `safe-refactor-test-guardian`: verification; scoped launcher/config changes
  and the `17`-test regression suite.
- `project-state-steward`: closeout; reconciled workflow, current state, skill
  evidence, and cleanup ownership.
- `agent-self-evaluation`: final synthesis; checked requested scope, evidence
  boundaries, authority blockers, verification, and handoff actionability.
- Evidence: focused tests, Ruff, compile, JSON, adapter probe, dry-run,
  missing-permit rejection, line scan, and diff check passed.
- Maintenance trigger: upstream commit, project CLI/evaluator, authority,
  baseline, config hash, or control-plane dependency changes.

## 2026-07-31 governance hardening completion

- `agent-architecture-audit`: identified current/history contamination and
  distributed authority as primary wrapper risks.
- `agent-harness-construction`: defined typed action, observation, retry, and
  stop contracts.
- `agent-eval` and `eval-harness`: produced pinned 14-task, three-run metrics
  with deterministic pass and fail controls.
- `iterative-retrieval` and `knowledge-ops`: drove scope-first retrieval,
  archive separation, and deduplicated authority routing.
- `plan-orchestrate`: reviewed for architecture routing only; it did not execute
  implementation, recorded as no real use in the JSON portfolio.
- `agent-self-evaluation`: reserved for final requirement-by-requirement audit.
- Evidence: validator `PASS`, focused tests `8/8`, fixture `pass^3=1.0`, negative
  control `pass^3=0.0`, project-local whole-bundle hash pinned.

## 2026-07-31 memory lifecycle and agent harness maintenance

- `agent-harness-construction`: reasoning; selected for memory routing,
  action/observation/recovery contracts, and halt behavior.
- `project-state-steward`: governance/execution; upgraded and used for the
  session-close lifecycle and repository hygiene boundary.
- `continuous-learning-v2`: learning boundary; consulted to keep global
  observer hooks inactive and project scope explicit.
- `ai-regression-testing`: verification; used to define regression cases from
  the observed memory and skill-selection failures.
- `agent-self-evaluation`: verification; required for final multi-step review.
- Evidence: lifecycle documents, validator, focused governance test, skill
  quick-validation, and an independent forward test all passed.
- Maintenance queue: re-run the validator after every lifecycle change and
  review reasoning skills after 60 days or a user correction.

Track skill coverage, freshness, evidence, and maintenance. Availability is not
usage. Read a selected skill completely before relying on it.

## Selection Contract

- Nontrivial work requires at least one reasoning or verification skill.
- Architecture, behavior, debugging, evaluation, synthesis, and data-contract
  work require a reasoning skill; verification alone is insufficient.
- Add domain or code skills only when the task needs them.
- Record selected skill, purpose, and evidence in the plan or run manifest.
- Do not bulk-load or bulk-refresh unrelated skills.

## Reasoning And Synthesis Attention Queue

- `agent-architecture-audit`: inspect agent boundaries and failure modes.
- `agent-harness-construction`: design tools, guardrails, state, and evaluation.
- `agent-introspection-debugging`: diagnose agent behavior and tool-use drift.
- `agent-self-evaluation`: evaluate completion quality and residual risk.
- `plan-orchestrate`: structure multi-step work and dependency order.
- `iterative-retrieval`: retrieve context progressively without flooding memory.
- `knowledge-ops`: organize durable knowledge and retrieval paths.
- `eval-harness`: design repeatable agent evaluation and regression checks.

## Forward-Test Evidence (2026-07-31)

- A synthetic cross-module API-contract closeout selected
  `agent-self-evaluation` as the reasoning skill.
- The agent paired it with `project-state-steward` for governance and hygiene,
  and did not edit files or delete artifacts.
- This project-state session also used `agent-self-evaluation` after validation;
  evidence is the passing skill validator, clean diff check, and forward-test.
- This proves selection behavior, not production usage; future real sessions
  must record their own evidence.

## Initial Baseline (2026-07-31)

- Reasoning and synthesis skills have no project-local production usage evidence
  yet. Mark them `ATTENTION_REQUIRED` until selected for a relevant task.
- Code and experiment skills are active but require maintenance after a user
  correction, repeated failure, dependency/CLI change, validator failure, or
  30 days without review.
- Reasoning skills require review after a correction, regression, or 60 days
  without review.

## Entry Shape

- Skill and category:
- Selected for and date:
- Last reviewed:
- Evidence of use:
- Staleness or failure signal:
- Next maintenance action:

## 2026-08-02 Thesis master-outline consolidation

- Skill category: reasoning, thesis synthesis and governance
- Selected for date: `academic-paper`, `thesis-evidence-writing`,
  `project-state-steward`
- Last reviewed: 2026-08-02
- Evidence of use: created the evidence-bounded master outline V2, separated
  protocol from result claims, added task-specific evaluation populations and
  a PENDING multi-day/multi-video results section, bound figures to prose and
  preserved current review blockers
- Staleness failure signal: outline or figure plan diverges from current
  authority, or a pending artifact is written as a result
- Next maintenance action: review after the next authority reconciliation or
  chapter-level meaning checkpoint

## 2026-08-02 Chapter 2.1--2.3 thesis revision

- Skill category: reasoning, evidence-bounded academic drafting
- Selected date: `academic-paper`, `thesis-evidence-writing`,
  `project-state-steward`
- Evidence use: revised the Vietnamese thesis passes for framework and data
  source-time representation, added accepted English academic drafts for 2.1
  and 2.2, and replaced the 2.3 scope with notebook-grounded detection-data
  construction and detector contracts while retaining behavior-review material
  as superseded provenance; added equations for the notebook's activity,
  duplicate and detector-output filtering without promoting exact run settings
- Staleness failure signal: revised prose contradicts current authority or
  treats review metadata, depth or anomaly screening as validated model input
- Next maintenance action: perform the Vietnamese meaning checkpoint before
  English conversion and update the figure captions with bound artifacts

## 2026-08-03 Classification V2 goal-routing correction

- Skill category: reasoning and project-state governance
- Selected date: `agent-introspection-debugging`, `project-state-steward`
- Evidence use: verified the active 3,243-unit post-review goal and encoded a
  fail-closed route guard after an agent mistook a later crash for the already
  resolved GUI freeze/RAM problem
- Staleness or failure signal: any agent resumes GUI work from a generic crash
  report without a newly reported GUI defect
- Next maintenance action: retrieve the active goal and current `C2V2` step
  before changing workstream after an interruption

## 2026-08-03 Reviewed engineering snapshot completion

- Skill category: data-contract reasoning, lineage, verification, governance
- Selected date: `agent-architecture-audit`,
  `dataset-contract-leakage-guard`, `experiment-lineage-reproducibility`,
  `safe-refactor-test-guardian`, `project-state-steward`
- Evidence use: bound eight forward-smoke hashes and the tiny-overfit checkpoint
  to immutable snapshot V3, rejected short-SHA lineage, passed 127 focused
  tests, and independently verified both final inventories
- Staleness or failure signal: any replay or training run changes X, split,
  masks, code SHA, or checkpoint lineage without a new frozen snapshot
- Next maintenance action: reuse these gates when a separate reviewed-data,
  original-label replay, or posture-ablation execution is authorized

## 2026-08-03 Thesis Sections 2.3--2.4 handoff

- Skill category: reasoning, evidence-bounded academic drafting and governance
- Selected date: `academic-paper`, `thesis-evidence-writing`,
  `project-state-steward`
- Evidence use: added original academic English for the accepted Section 2.3
  and drafted the Vietnamese meaning pass for Section 2.4, including local-ID
  scope, association evidence, causal/offline boundaries and figure references
- Staleness or failure signal: prose implies identity continuity across the
  six-week study or reports tracking quality without a bound evaluation set
- Next maintenance action: convert Section 2.4 to English only after the user
  accepts its Vietnamese technical meaning

## 2026-08-03 Academic prose and visual restraint

- Skill category: reasoning, evidence-bounded academic editing and governance
- Selected date: `academic-paper`, `thesis-evidence-writing`,
  `project-state-steward`
- Evidence use: revised Section 2.2 into cohesive academic English, removed a
  redundant source-time diagram, and removed unbound tracking figures from 2.4
- Staleness or failure signal: each subsection receives a figure by default or
  prose refers to a visual that provides no additional scientific evidence
- Next maintenance action: require a specific explanatory or evidential role
  before adding any further thesis figure

## 2026-08-03 Reader-facing terminology audit

- Skill category: reasoning, evidence-bounded academic editing and governance
- Selected date: `academic-paper`, `thesis-evidence-writing`,
  `project-state-steward`
- Evidence use: audited Sections 2.1--2.4 and replaced repository shorthand
  such as actor, legacy, native unit, sidecar and authority with descriptions
  of individuals, sampling procedures, manual review and evaluation scope
- Staleness or failure signal: a reader must know an internal file label or
  project role to understand a methodological sentence
- Next maintenance action: repeat the terminology audit when Sections 2.5+
  are drafted and remove working-note headings before manuscript export

## 2026-08-03 Equal annotation-process provenance

- Skill category: reasoning, evidence-bounded academic editing and governance
- Selected date: `thesis-evidence-writing`, `project-state-steward`
- Evidence use: corrected Section 2.2 so Procedure 1 and Procedure 2 are both
  described as manual CVAT annotation; only their sampling coverage differs
- Staleness or failure signal: wording such as current/earlier, primary/legacy
  or supplementary makes one peer annotation source appear subordinate
- Next maintenance action: preserve equal-source wording in later data,
  annotation and training sections

## 2026-08-03 Master-outline and visual-plan authority

- Skill category: reasoning, thesis structure and visual-evidence governance
- Selected date: `thesis-evidence-writing`, `academic-paper`,
  `project-state-steward`
- Evidence use: kept `PIG_BEHAVIOR_THESIS_MASTER_OUTLINE_V2.md` as the chapter
  structure, treated section drafts and data-label outline as supporting
  material, and reduced the figure plan to retained or conditional visuals
- Staleness or failure signal: a section is assigned a figure by default, or a
  supporting outline creates a competing chapter hierarchy
- Next maintenance action: assign final contiguous figure numbers only after
  the retained visual set is selected

## 2026-08-03 Section 2.3 filtering and negative-sample correction

- Skill category: reasoning, evidence-bounded academic editing and governance
- Selected date: `thesis-evidence-writing`, `project-state-steward`
- Evidence use: aligned the Vietnamese and English Section 2.3 drafts with the
  notebook implementation; corrected masked-image notation and denominator,
  described relaxed selection passes without overclaiming strict filtering,
  and separated background references from detector negative samples
- Staleness or failure signal: manuscript prose uses identical symbols for raw
  and masked images, treats all selected frames as satisfying initial filters,
  or calls an empty-pen background image a negative detector sample without a
  split record
- Next maintenance action: apply the same evidence and reader-facing audit to
  later dataset and evaluation sections

## 2026-08-03 Section 2.4 mode-specific association correction

- Skill category: tracking-method reasoning, evidence-bounded academic editing
  and governance
- Selected date: `tracking-experiment-guardian`, `thesis-evidence-writing`,
  `project-state-steward`
- Evidence use: removed the universal additive association-cost equation from
  Section 2.4 and replaced it with a mode-dependent abstraction; documented
  the distinct semantics of ByteTrack, causal association, RF transfer and
  post-video hybrid repair
- Staleness or failure signal: a manuscript equation implies that all tracking
  modes share IoU, position, appearance, area and identity-penalty weights
- Next maintenance action: keep Chapter 3 tracking comparisons at complete
  method level and bind detector, cadence, video population and evaluator
  before reporting mode metrics
