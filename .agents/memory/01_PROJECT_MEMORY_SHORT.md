# Project Memory Short

## 2026-07-15 legacy L6 motion valid negative decision

- Commits `cc01582`, `96e25b4`, and `6447c94` implement the crash-bounded
  cached-motion trainer, three-control short matrix, and paired evaluator.
- Six fresh-process GPU runs passed with 30 steps each, deterministic repeats,
  73,400,320 peak reserved bytes, no OOM/retry, and zero cleanup residue.
- Across 245 native units and 33 video clusters, motion minus parameter-matched
  zero macro-F1 is `-0.0018138438`, with interval
  `[-0.0260250944, 0.0233049368]`.
- The valid decision is
  `DO_NOT_EXPAND_MOTION_FROM_CURRENT_SHORT_EVIDENCE`; no full motion run is
  authorized. Continue L6 with all-class ROI relations from the zero control.
- This applies only to unreviewed `legacy_16f`. Reassess motion on frozen
  merged-reviewed data, whose rare-class support is materially larger.

## 2026-07-14 one canonical engine and explicit review profiles

- Legacy-only development and mixed-source reviewed data use one canonical
  data engine. Context, geometry, ROI, motion, social, posture, temporal
  harmonization, cache, fold, and training modules must not be forked or
  copied for the 16-frame lane.
- `legacy-only-unreviewed-development` is a profile over that engine: select
  only `legacy_recovered`, record an explicit human-review waiver, build
  `T6/T8/T12/T16` views inside each 16-frame burst, and isolate outputs,
  hashes, metrics, and claims.
- `mixed-reviewed` selects legacy plus CVAT, requires both review layers under
  the currently frozen Q2 protocol, and uses `fixed6_observed_time` as the
  primary model view. Native units remain legacy 16-frame and CVAT 6-frame.
- Human review is not a universal technical prerequisite for every exploratory
  run. It is mandatory for artifacts called reviewed/final and for the active
  mixed-source Q2 lineage. A waived profile must remain visibly unreviewed and
  cannot inherit reviewed or Q2 authorization.
- Complete the scoped legacy L0-L8 goal in a new chat, emit the immutable
  handback audit, then return to the original chat and resume the blocked
  parent P0-P8 goal. Legacy completion never completes the parent implicitly.

## 2026-07-14 exact legacy temporal model inputs

- Commit `21b34fd` binds eight explicit legacy model views: `T6`, `T8`,
  `T12`, and `T16`, each with all-sliding event-balanced and centered-matched
  sampling.
- The common evaluation unit remains the complete 16-frame burst. The loader
  binds view name, selection column, slot manifest, exact observed timing, and
  `temporal_input_frames`; it rejects malformed lengths and timing.
- Actor and union-context tensors must already be exact `T`. Spatial tensors
  may have capacity 16 only when every post-`T` length-mask slot is false.
- Four synthetic tier forwards produce finite `[2, 10]` logits with the same
  25,115 parameters. Evidence is 438 classification tests passing with 181
  deselected, Ruff, compileall, and zero optimizer steps or weight downloads.
- This is `PASS IN CODE`. A versioned short real-data chain must pass before a
  full legacy rebuild, and neither action authorizes training or full OOF.

## 2026-07-14 legacy-only unreviewed development authorization

- The user authorizes bounded classifier development from the 72,864-row
  `legacy_recovered` reference without waiting for current Hidden or behavior
  review.
- Every artifact and metric must be labeled
  `legacy-only-unreviewed-development`; this lineage is for loader, model,
  runtime, controlled ablation, and historical-comparison work only.
- Recording/video-safe groups, one 16-frame native burst per unit, exact
  feature whitelists, immutable hashes, and short-before-full gates remain
  mandatory.
- Model inputs must expose controlled `T6`, `T8`, `T12`, and `T16` tiers built
  inside each burst after harmonization. Evaluate every tier at the common
  16-frame native-unit grain.
- This branch cannot be called reviewed/final train-ready data, replace the
  all-source reviewed evaluation, authorize a Q2 claim, or weaken the existing
  human-review and full-OOF gates.
- New chats must create the scoped goal from
  `plans/classification_v2-legacy-16f-development-goal-prompt.md` and track
  L0-L8 in its dedicated execution ledger.
- After L0-L8 genuinely complete, return the hash-bound handback to the parent
  Q2 chat. The parent goal remains incomplete until separately resumed.

## 2026-07-14 native source and missingness probes

- Commit `9b04209` replaces the positional, all-numeric window source probe
  with exact trainer-whitelist and ordered-window SHA256 binding.
- Repeated windows are averaged once per `temporal_unit_key`; preprocessing and
  the linear probe fit only grouped training roles, and every eligible native
  unit must appear in outer test exactly once.
- A separate availability-only behavior diagnostic accepts only
  `window_image_context_complete`, `scene_context_ready`, and
  `scene_partner_context_ready`. Label-gated `interaction_context_ready` fails.
- Evidence is 14 focused tests and 429 classification tests with 181 deselected,
  Ruff, compileall, import smoke, and zero overlong changed lines.
- This is fixture-only engineering evidence. It does not authorize active-data
  training while Hidden and behavior review remain incomplete.

## 2026-07-14 native-unit checkpoint selection

- Commit `abae856` replaces window-level early stopping with grouped inner
  validation at the native temporal-unit grain.
- The primary score is supported-class macro-F1; native-unit NLL breaks ties.
  Outer-test predictions are never eligible for model selection.
- Mean class probabilities collapse windows by `temporal_unit_key`. Blank or
  duplicate keys, target/fold/source/group conflicts, malformed probabilities,
  nonfinite losses, and native-unit row loss fail closed.
- New runs emit best-validation and outer-test evidence at both window and
  native-unit levels. Native output preserves `source_type` and
  `split_group_key` as audit-only metadata outside model X.
- Checkpoint v6, run identity v3, run manifest v3, prediction manifest v2,
  registry v5, and run audit v3 bind the exact selection policy.
- Evidence is 415 classification tests passing with 181 deselected, Ruff,
  compileall, zero overlong changed Python lines, and a passing synthetic
  checkpoint/resume audit. No project-data training or full OOF ran.
- Human Hidden and behavior review remain the active blockers. A bounded metric
  test should prefer a reviewed, grouped, native-safe legacy 16f slice and must
  be labeled `legacy-only`; it cannot support the all-source Q2 claim alone.

## 2026-07-14 audited visual freeze schedule

- Commit `2bd2fda` adds one versioned schedule for actor and union-context
  backbones: frozen warm-up, ResNet `layer4` only, then optional full fine-tune.
- All parameters enter stable optimizer groups before training. The backbone
  uses a lower LR, frozen BatchNorm stays in eval mode, and heads stay trainable.
- A no-download, zero-step V0/V1/V2 audit proves identical pretrained status,
  normalization, freeze schedule, and head signature across controlled changes.
- Checkpoint v5, run identity v2, run manifest v2, and registry v4 bind stage,
  optimizer groups, schedule, trainable counts, config, fold, and artifact hashes.
- Evidence is 39 focused tests, 391 classification tests with 181 deselected,
  compileall, Ruff, checkpoint resume, and zero project-data rows read.
- This is engineering PASS only. Hidden and behavior review still block the
  active snapshot, pretrained pilots, model selection, and full OOF.

## 2026-07-14 deterministic visual tiny-overfit gate

- Commit `3be22f8` adds a data-free ResNet18 actor-temporal correctness gate for
  one-batch gradients, 20-event ten-class overfit, and optimizer-state resume.
- Two deterministic CUDA runs share the same semantic SHA256. The persisted
  audit reports accuracy 1.0, loss ratio 0.076818, zero resume-logit drift,
  778,576,384 peak VRAM bytes, and finite nonzero backbone/head gradients.
- The gate caught train/eval BatchNorm drift during development. It now scores
  only after post-fit running-stat recalibration; that smoke-only policy does
  not select the final training BatchNorm policy.
- Evidence is 35 focused tests and 385 classification tests with 181 deselected.
  The audit explicitly denies snapshot and full-OOF authorization.

## 2026-07-14 audited ResNet backbone interface

- Commit `07ed768` adds versioned `smoke_cnn`, ResNet18, and ResNet34 frame
  encoders to the common mask-safe model factory.
- Exact ImageNet enums and RGB mean/std are part of the contract. Tests resolve
  pretrained metadata without downloading weights; actual forwards use
  `NONE_RANDOM_INIT` only.
- Controlled forwards pass for ResNet18 at 160 and 224 px and ResNet34 at
  224 px. Parameter counts are 11,185,658, 11,185,658, and 21,293,818.
- Evidence is 31 focused model tests, the dry-run backbone audit, and 381
  classification tests with 181 deselected. No optimizer or training ran.
- This closes the independent production-backbone interface only. Human Hidden
  and behavior review still block the active snapshot and every model pilot.

## 2026-07-14 target-independent Hidden v6 authority

- Active Hidden authority is
  `outputs/classification_v2/rebuilds/hidden_review_v6_full_20260714`.
- Its 5,131 unique items comprise 4,122 Yes confirmations, 384 high-risk No,
  601 stratified-random No, and 24 clean controls.
- Selection excludes behavior and target-derived fields. The manifest SHA256 is
  `3e4fec14c466a89370a1e20d913cb024bd1dda1fa8db9c1fabdf8a51fa31072e`.
- Commit `f2179e3` binds media audit v2 to manifest/frame-context hashes. The
  24-item smoke and 5,131-item full resolver both have `media_missing=0`.
- Commits `32eaa2b` and `aaf8460` migrate v5 onto identifier v2, then carry all
  30 decisions into v6 with zero payload/context drift and bound SHA256 hashes.
- Current coverage is 30/5,131 with 5,101 missing. Random and high-risk reviewed
  support are both zero, so the scientific gate is explicitly blocked.
- Evidence is 31 focused tests and 377 classification tests with 181 deselected.
  Hidden apply, reviewed snapshot, model training, and full OOF remain forbidden.

## 2026-07-14 historical baseline controls

- Commit `e5d6417` registers the commit-`18d6692` full OOF only as
  `HISTORICAL_ONLY` and the legacy sequence checkpoint only as
  `HISTORICAL_ARCHITECTURE_ONLY`.
- The audit reproduces 151,440/160,740 split-to-context positional mismatches;
  image and interaction manifests agree with each other at all positions.
- Registration-time hashes cover 34 artifacts and 527,948,648 bytes. They do
  not prove origin-time inputs because the old run did not bind those hashes.
- `pig_behavior_sequence.pt` is a ten-output ResNet34 architecture reference,
  not a performance baseline without dataset, grouped-split, config, and seed
  lineage.
- Evidence is 5 focused tests and 356 classification tests with 181 deselected.
  Training, model promotion, paired comparison, and Q2 claims remain forbidden.
- The next independent task is Hidden clustered uncertainty and
  target-independent prevalence gates. Human review remains the P0 blocker.

## 2026-07-14 native-unit paired evaluation contract

- Commit `1b6ba3d` preserves the complete fold-assignment universe while
  collapsing strict ten-class window probabilities to one prediction per
  eligible native temporal unit.
- Prediction labels and folds must match the authority manifest; malformed
  keys, probability vectors, conflicting targets, missing units, and split
  drift fail closed.
- Primary pooled macro-F1 uses the fixed global ten-class order. Supported-fold,
  behavior-group, source/video/recording, calibration, and class-fold evidence
  remain explicit.
- Paired uncertainty resamples recording clusters with identical unit, target,
  cluster, and fold mappings. Percentile bootstrap reports no pseudo p-value.
- Evidence is 31 focused tests, 351 classification tests with 181 deselected,
  and a passing four-unit synthetic checker with no full-data read or training.
- The next independent task is historical-baseline reconciliation as an
  engineering control. Human Hidden and behavior review remain blockers.

## 2026-07-14 strict fixed-six timing loader

- Commit `111f152` loads ordered real `time_delta` values for the primary
  fixed-six view while preserving the complete training-window universe.
- Missing, reordered, duplicate, negative, or contradictory slot records fail
  closed; unselected windows remain explicit NaN/mask rows instead of dropping.
- Actor, spatial, and union branches receive the same timing tensor while
  retaining branch-specific observation masks.
- Checkpoint schema v4 and registry v3 bind the separate temporal-manifest hash.
- Evidence is 334 classification tests, four passing synthetic checkers, no
  pretrained download, no full-data read, and no model training.
- The next independent task is native-unit collapse and paired evaluation on
  synthetic predictions. Human Hidden and behavior review remain blockers.

## 2026-07-14 mask-safe configurable model factory

- Commit `318bf58` adds ten exact model modes and four temporal encoders behind
  one validated factory; direct 10-class supervision remains mandatory.
- Availability and quality masks gate optional data without entering X as
  behavior evidence. Masked NaN values cannot change logits.
- Checkpoint schema v3 and registry v2 now bind `model_mode`; old configs are
  read compatibly and reserialize with an explicit mode.
- Evidence is 78 focused tests, 319 classification tests, model-factory and
  multitask dry-runs, zero weight downloads, and zero optimizer steps.
- ResNet backbones and strict Transformer training remain blocked. The next
  independent task is real fixed-six `time_delta` loading and hash lineage.

## 2026-07-14 fold-local training lineage contracts

- Commits `97f83c5` and `73b901d` add training-fold-only preprocessing and
  native-event mass weighting without reading validation/test statistics.
- Commit `16cdb93` adds immutable run identity, isolated `fold_id/run_id`
  artifacts, checkpoint schema v2, append-only registry rows, remote merge,
  resume/hash audits, and lineage-aware caller paths.
- Evidence is 33 focused tests, 292 classification tests with 181 deselected,
  zero overlong Python lines, and a passing checkpoint/resume smoke.
- No model training or OOF ran. Hidden remains 30/5,171 and behavior remains
  3/4,670 with one pending, so the reviewed snapshot is still blocked.
- The next independent task is the configurable model factory with explicit
  modality availability/quality masks and forward-shape contracts.

## 2026-07-13 fixed-six temporal-view contract

- Commit `bb225ff` adds keyed manifests for `fixed6_observed_time`,
  `fixed6_normalized_phase`, and the `native6_16` ablation.
- The primary view reuses existing post-harmonization six-frame windows for
  both sources. It does not sample six quantiles across a legacy 16-frame burst.
- Every original window remains in a selection ledger; every native 6/16 unit
  remains in the native view, and missing observations remain explicit masks.
- Persisted manifests bind row counts and ordered stable-key hashes. Structural
  source/length/padding/timing/quality/availability shortcuts fail closed unless
  a separate valid mitigation artifact is supplied.
- Fixture evidence is 22/22 tests; the full classification regression is
  243 passed and 181 deselected. No real reviewed artifact, training, or OOF was
  run, so human Hidden and behavior review remain the active blockers.

## 2026-07-13 snapshot and launch-lineage hardening

- Commit `7cb4637` makes snapshot v2 fail closed on blank/duplicate keys,
  contract drift, invalid freeze, and ordered split/image/interaction mismatch.
- Commit `dd0e6ff` binds preflight, execution, and human authorization to the
  snapshot hash, lineage-audit hash, ordered `window_id` hash, config, and code.
- Interaction-context export now emits the same ordered-key audit and requires
  explicit `--overwrite` before replacing derived outputs.
- Regression evidence is 221 classification tests passing. No model training or
  full OOF was run.
- The bounded identifier-v2 packet remains technical-only because its Hidden
  and behavior-review authorizations are false.

## 2026-07-13 identifier-v2 positional lineage correction

- Commit `bfdf913` fixed a critical row-order defect: image and interaction
  windows had been sorted independently from split/target/spatial rows.
- In the historical full artifact, split-to-image and split-to-interaction
  positional mismatches were `151,440/160,740`; image and interaction happened
  to agree with each other. Historical OOF metrics are therefore runtime/debug
  evidence only, not classifier-quality evidence.
- Commit `a83d5a5` centralizes ordered `window_id` validation and adds fail-closed
  source-to-window identifier-v2 auditing plus overwrite guards.
- Current bounded authority is
  `outputs/classification_v2/rebuilds/scientific_smoke_identifier_v2_20260713`.
  It passes with 688 frame rows, 63 native units, 438 windows, exact 110-field X,
  zero trainable missing spatial slots, and 8/8 byte-identical repeat artifacts.
- Ordered sequence/image/train-ready/spatial hash is
  `05656ad78ecb65fac2341bc865f741039fe0c1e6b28211c65f2fc2c7973d7996`.
- Status remains `PASS_IDENTIFIER_V2_TECHNICAL_HUMAN_REVIEW_BLOCKED`; all
  training, full-OOF, reviewed-dataset, and Q2-claim authorizations are false.

## 2026-07-13 classification_v2 technical assurance gate

- Technical smoke authority is
  `outputs/classification_v2/rebuilds/scientific_smoke_v1/audits/technical_smoke_gate.json`.
- Commit `1679aca` reports `PASS_TECHNICAL_SMOKE_HUMAN_REVIEW_BLOCKED` with
  688 frame rows, 63 native/review units, 438 windows, both sources, and all
  10 behaviors.
- Tabular X matches the exact 110-column trainer whitelist; all 73 temporal
  evidence fields are present, while Hidden/review/target fields stay outside X.
- All 342 trainable smoke windows have complete spatial slots. The 544 missing
  slots belong only to 96 retained mask-false windows.
- Five repeated enhanced/harmonized/interval/window CSV pairs are byte-identical.
- Builders now exit nonzero on audit errors and require explicit `--overwrite`
  before replacing derived artifacts.
- This is code/data-generation assurance only. Hidden coverage is 30/5,171 and
  behavior coverage is 3/4,670 with one pending, so reviewed training remains
  blocked.

## 2026-07-13 authoritative classification_v2 state

- Use `docs/CLASSIFICATION_V2_CURRENT_STATE.md` as the status authority.
- The active path is the reviewed-data rebuild, not postrun promotion of the
  previous full OOF artifact.
- Hidden v5 has a valid 5,171-item template but no complete human decisions.
- Behavior coverage is 3/4,670 units, with 4,667 missing and one pending.
- Therefore Hidden apply, behavior apply, the reviewed train-ready snapshot,
  model smoke on that snapshot, and a new full OOF are all blocked.
- The old commit-`18d6692` full OOF is historical engineering evidence only.

## 2026-07-13 classification_v2 Hidden review workload

- Current workload-policy implementation commit is `5212a59`.
- Hidden review now census-selects every untrusted CVAT Hidden=Yes, samples
  trusted legacy Yes by recording-date/behavior stratum, and caps initial
  high-risk Hidden=No review at one item per stratum.
- Versioned full evidence is
  `outputs/classification_v2/rebuilds/hidden_review_v5_full_20260713`.
- V5 has 5,171 unique items: 4,649 CVAT and 522 legacy. It has zero missing
  untrusted Yes, trusted-stratum quota mismatches, or high-risk cap violations.
- This only makes the review workload auditable. Human Hidden decisions and
  behavior decisions remain incomplete, so the dataset is not train-ready.

## Historical 2026-07-13 full OOF and workflow migration

- Full multimodal OOF training completed in
  `outputs/classification_v2/model_full/full_multimodal_oof/`.
- Verified full outputs contain 73,668 window predictions and 32,727 native
  temporal predictions; accuracy is `0.5216793473` and supported macro-F1 is
  `0.4156053847`.
- These metrics are engineering evidence from the previous data lineage. They
  cannot become the final Q2 result by postrun processing alone because current
  Hidden and behavior human-review gates are incomplete.
- All classification operator scripts now live only under
  `scripts/classification_v2/00_*` through `09_*`. The former split namespaces
  and compatibility wrappers were removed.
- Workflow migration commits are `d7d22a8` and `1491d78`. The structural audit
  is block `09` script `check_classification_v2_workflow_layout.py`.

## Historical 2026-07-13 pre-full hardening refresh

- The previous lineage recorded its verified HEAD in
  `outputs/classification_v2/model_design/q2_progress_report_audit.json` key
  `current_git_commit`.
- `q2_progress_report_audit.json` is valid with `PASS_PARTIAL_ROADMAP`,
  44/44 gates passing, clean git, `full_oof_execution_allowed=false`,
  `authorization_authorized=false`, and `q2_claim_allowed=false`.
- Full OOF execution gate is now hardened with 4 rejection cases, including a
  near-authorized authorization file that has all boolean approvals true but
  missing `reviewer` and `reviewed_at`.
- Preflight runtime benchmark drift now allows audit/auth-only changes without
  rebenchmarking, while keeping runtime/model/training changes fail-closed.
- This pre-full state was later followed by the historical full run. It does not
  describe the active reviewed-data rebuild and must not authorize another run.

## Architecture contract retained from 2026-07-12

- Active priority is `classification_v2` behavior recognition, not tracking
  ablation, unless the user explicitly switches back to tracking.
- Current target claim remains Q2-strong only: improved pig behavior recognition
  under recording-date/video-safe validation. Do not claim external farm,
  camera, or cohort generalization until external validation exists.
- Pipeline is built around multimodal spatio-temporal inputs:
  bbox/letterboxed actor image sequence, ROI relation features, motion,
  social/partner context, interaction visual context, event-balanced weights,
  native temporal OOF folds, and strict feature whitelist leakage guards.
- `pig_id` is annotation-local and must not be treated as the same animal across
  videos or sessions.
- Canonical actor image cache is letterboxed, not square-stretched:
  `outputs/classification_v2/image_cache_v2_letterbox/`.
- Every future full OOF launch still requires preflight plus explicit
  authorization bound to the active data/cache/config/code hashes.
- The previous 44/44 `PASS_PARTIAL_ROADMAP` report is stale for the current
  rebuild and must not be refreshed until human review and snapshot gates pass.
- Local RTX 3050 limits development batch size, not the research architecture;
  remote/rented GPU execution remains allowed after the same lineage gates.
- Postrun calibration, confusion analysis, ablation, registry, and completion
  checks remain required after a future reviewed-lineage full run.

## 2026-07-08 realtime full runtime chunk validation

- Runtime 13-video validation completed in two chunks: `outputs/eval/realtime/runtime_check_quality_delayed_simple_7video/iou0_area0_condarea0_merge0` plus `outputs/eval/realtime/runtime_check_quality_delayed_simple_remaining6/iou0_area0_condarea0_merge0`.
- Compared with `outputs/eval/realtime/realtime_balanced_13video/iou0_area0_condarea0_merge0`, per-video runtime total `remapped_idsw 75 -> 21`, `fp/fn` stayed `2320/1055`, no per-video IDSW regression, and `000302=0`.
- Remaining runtime IDSW: `000114=2`, `000231=6`, `000233=9`, `000263=2`, `000327=2`; all other 8 videos are `0`.

## 2026-07-08 realtime simple low-gain component pass

- Improved `realtime_quality_delayed` artifact candidate further by adding an opt-in second pass for simple motion components only: `realtime_motion_pair_simple_min_gain=0.005`, `realtime_motion_pair_simple_max_component_size=2`.
- Evidence artifact: `outputs/eval/realtime/probe_motion_pair_simple005_comp2_13video/iou0_area0_condarea0_merge0`.
- Compared with gain-gate candidate `outputs/eval/realtime/probe_motion_pair_gainmin004_edges2_13video/iou0_area0_condarea0_merge0`: `ALL remapped_idsw 27 -> 21`, `remapped_hota_pct 95.89 -> 96.60`, `remapped_idf1_pct 96.41 -> 97.02`, `fp/fn unchanged 2320/1055`.
- Per-video improvements with no IDSW regression on 13-video artifact probe: `000085 2 -> 0`, `000327 4 -> 2`, `000330 2 -> 0`; `000233` stayed `9`, `000302` stayed `0`.
- Runtime smoke check: `outputs/eval/realtime/runtime_check_quality_delayed_simple_233_302/iou0_area0_condarea0_merge0` using actual `realtime_quality_delayed` code path produced `000233 remapped_idsw=9` and `000302 remapped_idsw=0`, matching the artifact expectation for target/guardrail.
- Rejected probes: global `min_allowed_edge_gain=0.02` regressed `000114/000327`; global `max_jump=0.08/0.12` regressed `000233`; `memory_frames=20` regressed `000231`; `memory_frames=40` unchanged; global `min_gain=0.005` regressed `000233/000327`.
- Additional runtime smoke checks: `outputs/eval/realtime/runtime_check_quality_delayed_simple_263/iou0_area0_condarea0_merge0` produced `000263 remapped_idsw=2`; `outputs/eval/realtime/runtime_check_quality_delayed_simple_085/iou0_area0_condarea0_merge0` produced `000085 remapped_idsw=0`. Both match artifact expectations.

## 2026-07-08 realtime dense fallback gain gate

- Improved current `realtime_quality_delayed` motion-pair candidate by tightening dense-component fallback: `realtime_motion_pair_dense_fallback_max_edges=2`, `realtime_motion_pair_dense_fallback_min_median_gain=0.05`, `realtime_motion_pair_dense_fallback_min_edge_gain=0.04` while keeping `max_component_size=4`, `max_component_edges=3`, `max_support_ratio=0.35`.
- Evidence artifact: `outputs/eval/realtime/probe_motion_pair_gainmin004_edges2_13video/iou0_area0_condarea0_merge0`.
- Compared with previous dense candidate `outputs/eval/realtime/probe_motion_pair_comp4_edges3_dense_13video/iou0_area0_condarea0_merge0`: `ALL remapped_idsw 31 -> 27`, `000233 13 -> 9`, no per-video IDSW regression, `000302=0`, `fp/fn unchanged 2320/1055`; HOTA/IDF1 effectively unchanged at `95.89/96.41`.
- Runtime check on `000233` planned graph confirms allowed dense fallback edges become `{ID_1-ID_3, ID_1-ID_8}`, excluding weak low-min-gain `ID_2-ID_8` that caused extra switches.

## 2026-07-08 realtime motion-pair quality-delayed candidate

- Added opt-in `realtime_motion_pair_stabilizer` for `mode=realtime` only. It relabels short-memory motion-consistent ID attributes, then filters proposed relabel graph to small/sparse components. The current 13-video candidate uses `realtime_motion_pair_max_component_size=4`, `realtime_motion_pair_max_component_edges=3`, and dense-component rare-edge fallback (`max_edges=3`, `max_support_ratio=0.35`); this admits sparse four-ID episodes like `000327` and a limited rare-edge subset in dense `000233` while still blocking the dominant long cascade edge.
- Important implementation fix: the planning pass must use `deepcopy`; shallow `shape.copy()` mutates nested `attributes` and accidentally applies broad relabel before component filtering.
- Enabled the stabilizer in `realtime_quality_delayed`, not in `realtime_balanced`. Treat this as a quality-delayed candidate, not the pure causal realtime baseline.
- Validated runtime 5-video result: `outputs/eval/realtime/codex_motion_pair_quality_5video_fix/iou0_area0_condarea0_merge0`. Compared with `outputs/eval/realtime/realtime_balanced_5video/iou0_area0_condarea0_merge0`: `ALL remapped_idsw 43 -> 25`, `remapped_hota_pct 90.12 -> 92.20`, `remapped_idf1_pct 90.18 -> 92.50`, `fp/fn unchanged 849/661`.
- Per-video remapped IDSW in this candidate: `000231=8` (from `12`), `000233=15` (unchanged, no regression), `000263=2` (from `12`), `000328=0` (from `4`), `000302=0` (guardrail preserved).
- 13-video artifact probe with component size `4`, edge cap `3`, and rare-edge fallback: `outputs/eval/realtime/probe_motion_pair_comp4_edges3_dense_13video/iou0_area0_condarea0_merge0`. Compared with `outputs/eval/realtime/realtime_balanced_13video/iou0_area0_condarea0_merge0`: `ALL remapped_idsw 75 -> 31`, `remapped_hota_pct 92.77 -> 95.89`, `remapped_idf1_pct 93.12 -> 96.41`, `fp/fn unchanged 2320/1055`. No per-video remapped IDSW regression in this 13-video set; `000302` stayed `0`. `000327` improved `8 -> 4`; `000233` improved `15 -> 13` but remains the weakest realtime video.
- Do not promote this as realtime causal base without broader regression and explicit discussion that it is delayed/post-tracking stabilization. Next step should validate more videos and then design an online-buffer equivalent if true realtime latency is required.

## 2026-07-08 realtime profile cleanup and failed probes

- `realtime_balanced` was changed to inherit from a realtime-only eval base:
  `enable_offline_smoothing=false`, `identity_swap_guard=false`,
  `smooth_boxes=false`, and `refine_boxes=false`. Single `000233` with these
  offline flags forced off matched the prior metrics, so the current realtime
  failures are from online association/tracking behavior rather than offline
  smoothing.
- Added named realtime eval profiles:
  - `realtime_fast`: speed-oriented probe, `detect_every_n_frames=2`,
    `det_conf=0.25`, `max_raw_detections=32`, no offline smoothing.
  - `realtime_balanced`: current causal probe stack.
  - `realtime_quality_delayed`: finite-window local repair probe only; no
    suffix/long future repair.
- Rejected/neutral probes from this continuation:
  - `overlap_small_box_suppression=true` on realtime `000233`: no metric change
    (`remapped_idsw` stayed `15`).
  - hybrid causal guard stack on realtime `000233`: no improvement; FP slightly
    increased.
  - `tracker_type=botsort` on realtime `000233`: no metric change.
  - looser `realtime_visible_better_competitor_min_cost=0.28`,
    `min_gain=0.025` on `000263`: regressed `remapped_idsw 12 -> 16`, so do
    not promote.
  - `local_pair_swap_repair=true` with a 12-frame window on realtime `000263`:
    no metric change.
- Current conclusion: remaining realtime IDSW is not solved by porting existing
  hybrid causal guards or existing finite-window repair as-is. Next useful
  implementation should be a new online/short-buffer identity stabilizer, not a
  broad reject/hold guard and not offline suffix repair.

## 2026-07-08 realtime balanced profile

- Added `realtime_balanced` to `scripts/evaluate_tracking.py` as the current
  named realtime probe profile. It packages the useful opt-in realtime stack:
  `smooth_det020_loose` recovery settings, `occlusion_aware_matching=false`,
  `realtime_visible_close_competitor_guard=true`,
  `realtime_visible_better_competitor_reject=true`,
  `realtime_visible_better_competitor_prefer=true`, and
  `realtime_low_conf_recovery_guard=true`.
- `realtime_balanced` is still a probe profile, not a finished realtime
  baseline. It preserves the `000302` guardrail in the single-video check:
  `outputs/eval/realtime/realtime_balanced_302_guardrail/iou0_area0_condarea0_merge0`
  produced `remapped_idsw=0`, `remapped_hota_pct=99.38`,
  `remapped_idf1_pct=99.69`.
- 5-video validation with the named profile:
  `outputs/eval/realtime/realtime_balanced_5video/iou0_area0_condarea0_merge0`.
  This matches the prior long override candidate: `ALL remapped_idsw=43`,
  `fn=661`, `fp=849`, `remapped_hota_pct=90.12`,
  `remapped_idf1_pct=90.18`; per-video remapped IDSW remains
  `000231=12`, `000233=15`, `000263=12`, `000328=4`, `000302=0`.
- Rejected new probe: `realtime_reid_shadow_visible_hold`. Broad version on
  `000263` reduced remapped IDSW `12 -> 8` but badly damaged idmap coverage and
  HOTA/IDF1 (`remapped_hota_pct` about `81.70`). Narrowing to
  `max_missed=5` kept `000263` at `12` IDSW and did not improve HOTA. The guard
  was removed from code. Do not re-add a hold/consume duplicate-shadow reid
  guard without a fundamentally better discriminator.

## 2026-07-08 realtime failed guard probes

- Tried a narrow opt-in `realtime_occluded_reid_duplicate_guard` idea for
  `000263` reid switches. Default `min_iou=0.55` did not trigger. Lowering to
  `min_iou=0.45` reduced `000263` remapped IDSW `12 -> 8`, but badly damaged
  IDF1/HOTA/idmap coverage (`remapped_hota_pct` about `81.70` versus `90.62`),
  so this is not a promotion candidate. The underlying evidence is still useful:
  wrong `000263` reid detections around `792/846/865` overlap visible tracks by
  only about `0.46-0.49` IoU, so simple duplicate rejection is too blunt.
- Tried an opt-in visible row-regret reject for `000231` frame `1368`
  (`selected_cost≈0.668`, `track_best≈0.226`). It triggered exactly once but did
  not reduce remapped IDSW and slightly worsened FP/FN, so it was removed.
- Current realtime candidate remains the missed3 low-conf stack:
  `outputs/eval/realtime/probe_realtime_low_conf_recovery_missed3_5video/iou0_area0_condarea0_merge0`.
  Continue from diagnostics rather than re-adding the rejected duplicate or
  row-regret guards.

## 2026-07-08 realtime missed3 candidate

- Realtime baseline at `outputs/eval/realtime/baseline_current/iou0_area0_condarea0_merge0` remains the main realtime comparison point.
- Current useful realtime candidate keeps all new realtime guards opt-in:
  `occlusion_aware_matching=false`,
  `realtime_visible_close_competitor_guard=true`,
  `realtime_visible_better_competitor_reject=true`,
  `realtime_visible_better_competitor_prefer=true`,
  `realtime_low_conf_recovery_guard=true`.
- Tuned `realtime_low_conf_recovery_min_missed` default for the opt-in guard to `3`.
  Single `000233` improved versus the broad low-conf guard: `IDSW=15`,
  `FN=388`, `remapped_hota_pct=84.21` at
  `outputs/eval/realtime/probe_realtime_low_conf_recovery_233_missed3/iou0_area0_condarea0_merge0`.
- 5-video candidate
  `outputs/eval/realtime/probe_realtime_low_conf_recovery_missed3_5video/iou0_area0_condarea0_merge0`:
  `ALL remapped_idsw=43`, `fn=661`, `fp=849`, `remapped_hota_pct=90.12`,
  `remapped_idf1_pct=90.18`. Per-video remapped IDSW:
  `000231=12`, `000233=15`, `000263=12`, `000328=4`, `000302=0`.
  This is not final, but it is cleaner than the earlier broad low-conf guard
  because it preserves most IDSW gain while recovering FN.
- Rejected probe: `realtime_late_reid_guard` for stale occlusion reid on `000263`
  reduced `000263` IDSW `12 -> 10` but badly damaged IDF1/HOTA and idmap coverage
  (`remapped_hota_pct=81.66`), so it was removed from code.
- Diagnostics from `outputs/pred/realtime/probe_realtime_missed3_263_debug/.../association_debug_events.csv`:
  remaining `000263` switches are mostly `reid` from `OCCLUDED/occlusion_hold`
  despite low selected costs, e.g. frames `792` missed `3`, `846` missed `33`,
  `865` missed `5`. A simple late-missed gate is not safe; next direction should
  inspect whether these reid detections are duplicates/extra boxes or need a
  causal short-window identity stabilizer rather than a broad reject.

## 2026-07-08 realtime coverage candidate

- Baseline realtime at
  `outputs/eval/realtime/baseline_current/iou0_area0_condarea0_merge0` has a
  major coverage/FN problem: 13-video `ALL` `fn=72669`, `recall_pct=60.58`,
  `remapped_idsw=115`, `remapped_hota_pct=57.55`.
- Strongest realtime lever so far is `occlusion_aware_matching=false`; on the
  5-video guard set it reduces `fn` from about `32425` to `601`, but exposes
  visible close-competitor swaps.
- Added opt-in `realtime_visible_close_competitor_guard=true` for realtime only
  when `occlusion_aware_matching=false`. It resolves near-tie high-confidence
  visible assignments toward an otherwise unserved competitor track.
- The useful discriminator came from debug:
  - `000302` good trigger: frame `555`, selected `track 8` vs preferred
    `track 4`, costs `0.194358` vs `0.204259`, margin about `0.0099`.
  - `000263` false trigger with wider margin: frame `421`, costs `0.261192`
    vs `0.276337`, margin about `0.0151`.
  - Default `realtime_visible_close_competitor_margin=0.012` keeps the `000302`
    fix while blocking the `000263` false trigger.
- Current 5-video realtime candidate:
  `outputs/eval/realtime/probe_close_competitor_margin012_5video/iou0_area0_condarea0_merge0`.
  Compared with realtime baseline subset, it is a large coverage/HOTA
  improvement but not an IDSW win: `fn=601`, `fp=942`, `remapped_idsw=63`,
  `recall_pct=99.15`, `remapped_hota_pct=88.07`. Per-video remapped IDSW:
  `000231=27`, `000233=20`, `000263=12`, `000328=4`, `000302=0`.
- Do not promote this as the default realtime config yet. Next realtime work
  should reduce the remaining visible-swap IDSW on `000231/000233/000263/000328`
  without reintroducing hidden/occluded coverage loss.

## 2026-07-07 visible-start suffix gate full success

- New best full 12-video candidate:
  `outputs/eval/hybrid_bytetrack/codex_visible_suffix_gate_full/iou0_area0_condarea0_merge0`.
- Versus `outputs/eval/hybrid_bytetrack/Best_tracking/iou0_area0_condarea0_merge0`:
  `ALL` remapped IDSW improved `11 -> 0`; every per-video remapped IDSW is
  `0`.
- Guardrails stayed clean: `000085=0`, `000225=0`, `000231=0`, `000302=0`,
  `000328=0`.
- Targets fixed: `000233=0`, `000263=0`.
- Key code change: `suffix_pair_swap_repair=true` is now narrowed by requiring
  both shapes at `swap_start` to have `Hidden=No`. This blocks the false
  hidden-start suffix swaps previously seen on `000085` frame 17 and `000225`
  frame 264, while still allowing the visible-start `000263` suffix repair
  around frame 193.

- Do not hardcode `000263`/`000302` as optimizer target videos anymore.
- For optimizer ranking defaults, derive weak target videos from:
  `outputs/eval/hybrid_bytetrack/Tracking mới tắt smooth/yolov8/iou0_area0_condarea0_merge0/tracking_metrics.csv`
- Current weakest set from that file is:
  - `Pigs291119_000263_30fps`
  - `Pigs291119_000226_30fps`
  - `Pigs301119_000327_30fps`
  - `Pigs301119_000328_30fps`
- For `evaluate_tracking.py` metric comparisons, use commit `b697c4eba36db280cbf01f446873da17bcac509d` as the relevant historical reference instead of legacy 21/06.
- Critical IDSW-preserving tracking flow in `src/pig_behavior/tracking/runner.py`:
  - `apply_identity_swap_guard(...)` runs only when `cfg.enable_offline_smoothing and cfg.identity_swap_guard`.
  - `refine_shapes_temporally(...)` and then `stabilize_overlap_hidden_islands(...)` run only when `cfg.enable_offline_smoothing and (cfg.smooth_boxes or cfg.refine_boxes)`.
  - Do not change this back to `cfg.enable_offline_smoothing or cfg.mode == "hybrid_bytetrack"`; that drift was identified as a likely cause of worse IDSW versus commit `b697c4eba36db280cbf01f446873da17bcac509d`.
- Current tracking CLI flow:
  - Use `scripts/track_videos.py` for batch/single-video tracking.
  - `track_videos.py` calls `python -m pig_behavior.tracking.cli`; `src/pig_behavior/tracking/cli.py` must keep its `__main__` entrypoint.
  - `track_videos.py --eval-config <name>` reuses `evaluate_tracking.py` presets and passes them to tracking CLI as `--profile-override KEY=VALUE`.
  - `track_videos.py` must pass `src` via `PYTHONPATH` to the subprocess so module execution works without editable install.
  - `--no-emit-hidden-tracks` keeps tracker-maintained/interpolated boxes in the output but writes their `Hidden` attribute as `No` for CVAT relabeling; it does not disable internal tracking/association/occlusion state.
- Current runtime variants to compare are:
  - C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\eval\hybrid_bytetrack\Tracking moi tat smooth\yolov8
  - C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\eval\hybrid_bytetrack\Tracking moi bat smooth
- In reports, no smooth is currently worse than smooth; do not assume the unsmoothed runtime is the better baseline.
- Optimizer default scopes should stay tracking-focused.
- Detector-only presets were moved to explicit `--scope detector_probe` because `overnight_iou0` showed detector-only metrics were identical to `base`.
- Continue focusing on code and runtime behavior in association.py and runner.py, not detector weight.
- 2026-07-05 practical hard-set direction:
  - Use `hidden_owner_guard=true` + `hidden_owner_guard_hold_assignment=true` as the current practical opt-in base for hard-set work; it solves the `000231` frame-401 hidden-owner issue while preserving `000302=0` in later checks.
  - Do not keep tuning `reentry_ambiguous_hold` or `reentry_unowned_raw_mismatch_reject`/quarantine thresholds as the main path; those branches either damaged `000231`/`000302` or missed `000328`.
  - Next `000328` work should use a separate episode-level repeated unowned raw-ID mismatch detector, not per-assignment hold/reject rules.
# 2026-07-03 Best tradeoff found: lost-track reacquire split guards

- New strong 2-video tradeoff result:
  `outputs/eval/hybrid_bytetrack/20260703_193439/smooth_det020_loose/iou0_area0_condarea0_merge0/tracking_metrics.csv`
- Config was `smooth_det020_loose` plus `lost_track_reacquire_non_same_raw_distance_guard=false`.
- Key metrics:
  - `Pigs291119_000231_30fps`: IDSW `2`, HOTA `0.9705892717094201`, IDF1 `0.9847241970177549`.
  - `Pigs291119_000302_30fps`: IDSW `0`, HOTA `0.9930104703678451`, IDF1 `0.9964355605255801`.
  - `ALL`: IDSW `2`, HOTA `0.9820366705826231`, IDF1 `0.9907038986528682`.
- Preserve the current split `lost_track_reacquire_guard` design in `association.py` / `config.py`:
  - keep `lost_track_reacquire_guard=true`;
  - `lost_track_reacquire_non_same_raw_distance_guard=false` is now the default/base setting after the strong 9-video `20260703_194929` run;
  - do not disable raw-owner guard globally because it fixes `000302` but badly hurts `000231`;
  - preserve the conditional `lost_track_different_raw_hidden_owner_bypass` with `min_missed=2` and `min_center_gain=0.03`.
- `outputs/eval/hybrid_bytetrack/20260703_194929/smooth_det020_loose/iou0_area0_condarea0_merge0/` validated this as a good 9-video base; future tracking/eval/optimizer runs should not require a long override for this guard.

- 2026-07-05 successful candidate `20260705_220622`: hard-set remapped IDSW `000231=0`, `000263=2`, `000328=0`, `000302=0`, `ALL=2`. Candidate config: hidden-owner hold plus `reentry_unowned_raw_mismatch_episode_reject=true`, `reentry_unowned_raw_mismatch_episode_action=hold`, `episode_min_missed=1`, `episode_max_missed=20`, `episode_max_events=8`, `episode_max_cost=0.36`. Remaining `000263` frames `193/195` are track 3/4 fight/occlusion with raw IDs still consistent; user suspects possible GT ambiguity, so do not add broad guard before visual/GT confirmation.
- 2026-07-07 new 5-video candidate `outputs/eval/hybrid_bytetrack/20260707_082640/smooth_det020_loose/iou0_area0_condarea0_merge0`: improves weak `000233` without breaking hard guardrails. Remapped IDSW: `000231=0`, `000233=6`, `000263=2`, `000328=0`, `000302=0`, `ALL=8` versus `Best_tracking` `000233=9`, `ALL=11`. Add-on opt-in guard: `occlusion_reid_prefer_gap_over_bad_match=true`, `occlusion_reid_bad_match_action=reject`, raw mismatch + unowned raw + occlusion_hold only, `min_missed=7`, `max_missed=12`, `min_cost=0.55`, `max_cost=0.70`. This reject action intentionally does not consume the detection; the max-cost upper bound prevents `000231` frame-906 style regression.
- 2026-07-07 suffix repair candidate `outputs/eval/hybrid_bytetrack/codex_suffix_5video_min1500/iou0_area0_condarea0_merge0`: best current 5-video opt-in, remapped IDSW `000231=0`, `000233=6`, `000263=0`, `000328=0`, `000302=0`, `ALL=6`. Adds `suffix_pair_swap_repair=true` on top of protected practical config and 000233 guarded config. Key fix is suffix identity crossing for `000263` `Pig_3/Pig_4` after heavy overlap around frames `193/195`; default `suffix_pair_swap_min_suffix_frames=1500` avoids false suffix swaps seen with broad `60` frame setting on `000231/000233/000328/000302`. Keep opt-in until broader regression passes.
- 2026-07-07 new best 5-video opt-in candidate `outputs/eval/hybrid_bytetrack/codex_overlap_suppress_5video/iou0_area0_condarea0_merge0`: remapped IDSW `000231=0`, `000233=2`, `000263=0`, `000328=0`, `000302=0`, `ALL=2`. Adds `overlap_small_box_suppression=true` on top of the suffix candidate. This suppresses small low-confidence boxes in high-overlap frames (`min_iou=0.40`, `max_area_ratio=0.65`, `max_score=0.75`) and fixes the `000233` short box-crossing switches without breaking hard guardrails. Keep opt-in pending broader regression.
- 2026-07-07 current best 5-video opt-in candidate `outputs/eval/hybrid_bytetrack/codex_hidden_suffix_id_swap_5video/iou0_area0_condarea0_merge0`: remapped IDSW `000231=0`, `000233=0`, `000263=0`, `000328=0`, `000302=0`, `ALL=0`. Adds `hidden_suffix_id_swap_repair=true` on top of the overlap-suppress candidate. It catches the `000233` `ID_8/ID_1` low-confidence hidden suffix crossing around `1107-1118` using hidden-run length, max overlap, low median hidden score, and long suffix gates. Keep opt-in pending broader regression before base promotion.
- 2026-07-07 broader 12-video regression `outputs/eval/hybrid_bytetrack/20260707_174142/smooth_det020_loose/iou0_area0_condarea0_merge0` proved the full 5-video stack is not a safe common baseline. It improved `000233=0` and `000263=0`, but regressed previously clean videos: `000085: 0 -> 2` and `000225: 0 -> 2` remapped IDSW versus `Best_tracking`.
- 2026-07-07 ablation on `000085/000225/000233/000263`:
  - `ablate_control_assoc_occlusion_4video`: `000085=0`, `000225=0`, `000233=6`, `000263=2`.
  - `ablate_suffix_only_4video`: `000085=2`, `000225=2`, `000233=6`, `000263=0`; therefore current `suffix_pair_swap_repair=true` is unsafe and must not be promoted.
  - `ablate_overlap_only_4video`: `000085=0`, `000225=0`, `000233=2`, `000263=2`; `overlap_small_box_suppression=true` appears safe on this 4-video ablation.
  - `ablate_overlap_hidden_no_suffix_4video`: `000085=0`, `000225=0`, `000233=0`, `000263=2`; current safest common candidate is protected association/occlusion base plus `overlap_small_box_suppression=true` and `hidden_suffix_id_swap_repair=true`, explicitly with `suffix_pair_swap_repair=false`.
- 2026-07-07 full 12-video no-suffix common candidate `outputs/eval/hybrid_bytetrack/no_suffix_common_candidate_full/iou0_area0_condarea0_merge0`: remapped IDSW `ALL 11 -> 2` versus `Best_tracking`, with no per-video IDSW regression. Key per-video: `000085=0`, `000225=0`, `000231=0`, `000233 9 -> 0`, `000263=2`, `000302=0`, `000328=0`. This is now the safest broader candidate; remaining target is `000263=2` and must not be fixed with current `suffix_pair_swap_repair`.
