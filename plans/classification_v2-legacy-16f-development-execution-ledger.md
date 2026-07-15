# Classification V2 `legacy_16f` Development Execution Ledger

Version: 1.1

Opened: 2026-07-14

Scope: `legacy-only-unreviewed-development`

Goal authority:
`classification_v2-legacy-16f-development-goal-prompt.md`

Parent goal: canonical reviewed all-source P0-P8, currently blocked and not
replaced by this ledger.

## Status Vocabulary

- `PASS`: current-lineage evidence satisfies every declared gate.
- `PASS IN CODE`: fixtures prove implementation, but project-data evidence is
  still missing.
- `IN_PROGRESS`: safe implementation or verification remains.
- `BLOCKED`: an external dependency prevents further meaningful work.
- `NOT_STARTED`: prerequisite work has not passed.

## Milestone Ledger

| Milestone | Status | Current evidence or next gate |
|---|---|---|
| L0 state reconciliation | PASS | Authority, lane, counts, and claim boundary locked |
| L1 short packet | PASS | Exact cache, slot, fold, and repeat gate at `00dc2e0` |
| L2 full legacy lineage | PASS | Full repeat-bound lineage at `59647e2` |
| L3 immutable inputs | PASS | Committed-SHA gate at `0414adc` |
| L4 model correctness | PASS | Real-cache correctness gate at `3ef4235` |
| L5 core baselines | PASS | T6 sliding retained as bounded legacy_16f baseline |
| L6 modality loop | IN_PROGRESS | Full ROI rejected; next gate is numeric social relations |
| L7 imbalance policy | NOT_STARTED | Requires retained L6 candidate |
| L8 candidate/handback | NOT_STARTED | Requires controlled L0-L7 evidence |

## Settled Contracts

- Native and evaluation unit: one complete 16-frame burst.
- Input tiers: T6, T8, T12, and T16.
- Sampling: stride-3 event-balanced sliding plus one centered matched window.
- Primary metrics aggregate predictions to the native 16-frame burst.
- Grouping is recording/video safe; `pig_id` is annotation-local.
- Every artifact and metric carries the legacy-only unreviewed claim flag.
- Exact short evidence is mandatory before any full expansion.
- Local 4 GiB VRAM controls correctness placement, not architecture scope.
- Rented GPU pilots are permitted after the exact target-environment short gate.
- Legacy rare-class support cannot estimate merged-data support or its ceiling.

## Canonical Legacy Source Identity

- Canonical short name: `legacy_16f`.
- This is the only allowed short name; no alternate alias may be introduced.
- Raw authority is
  `data/raw/legacy_full_multigt_masked_nodup_16f/legacy_dense_tracklet_map.csv`.
- Raw SHA256 is
  `ff73c158ef879eb8177b0c18783fc751945fe1d6af97a4b8235cd71681fabbcb`.
- The raw table has 72,864 rows; the derived legacy export has 4,554 complete
  native bursts with 16 frames each and SHA256
  `adbdb572b976e9f63cff5f9b29ced649f37fa80dd382336b3678f71ac50ff636`.
- Internal identity is `source_type=legacy_recovered` and
  `dataset_id=legacy_recovered_16f`.
- Stage `00_scope` reads a mixed staging table only as an upstream container;
  it selects 72,864 legacy rows and zero CVAT rows for this lane. Therefore
  this lineage is not the merged dataset.

## 2026-07-15 Goal Clarification

- The user confirms that rare behaviors are substantially more numerous in the
  merged dataset than in this legacy 16-frame lane.
- Zero or weak rare-class metrics here remain bounded legacy diagnostics; they
  cannot reject an architecture for the future merged-reviewed lineage.
- The local RTX 3050 remains the safe correctness host. Scientifically justified
  heavy pilots may run on rented GPUs with identical immutable lineage fields.
- The formal T1-versus-V1 paired gate rejected T1 only for this legacy T16
  search. V1 masked mean is retained for the two-protocol T6/T8/T12/T16 ladder.
- T1 remains bounded negative evidence, and a small Transformer is deferred
  because no temporal-capacity benefit survived the gate.
- Reassess TCN, Transformer, and retained controls on frozen merged-reviewed
  data; local VRAM is not an architecture rejection criterion.

## Current Short Evidence

Root:

```text
outputs/classification_v2/legacy_only_unreviewed_development/
short_temporal_tiers_v3_20260714
```

Observed evidence:

- 496 frame rows;
- 31 complete native bursts;
- 310 sliding windows;
- 124 centered matched windows;
- eight temporal model views;
- all 10 canonical labels;
- zero duplicate native keys;
- zero outside-native windows;
- zero missing timing slots;
- zero dropped or relabeled rows;
- event-mass maximum error `0.0`;
- all 13 repeated CSV outputs byte-identical;
- every stage 00-05 CSV and audit carries the exact unreviewed claim pair;
- strict real-tier loader audit valid for all eight views.

Latest completed achievement:

```text
aae63c3 feat: preserve legacy lineage claims end to end
```

Verification at that boundary:

- focused claim, tier, cache, and L1 tests: 47 passed;
- classification regression: 463 passed, 181 deselected;
- Ruff, `py_compile`, `compileall`, diff check, and long-line scan PASS;
- optimizer steps: zero;
- full dataset reads: zero;
- bounded video decode count: 64 for each independent cache build.

## L0 Reconciliation Refresh

- Reconciled HEAD: `fc6c594`; the implementation boundary remains `c41f1ed`.
- The three pre-existing worktree changes match the prompt's preserve list.
- Bounded packet counts, claim flags, loader status, and recorded hashes match.
- `CLASSIFICATION_V2_CURRENT_STATE.md` and the core ledger still report the
  earlier 429-test boundary; this lane's later verified boundary is 442 passed
  with 181 deselected.
- This documentation drift does not authorize training or change L1 gates.

## L1 PASS Evidence

- 496 exact `legacy_video_bbox` contexts and 310 window rows;
- 2,728 image slots with zero native-unit or frame-order mismatch;
- packed tensor shape `[496, 160, 160, 3]`, dtype `uint8`;
- zero cache misses, source-media reads, and packed pixel mismatches;
- 31 native bursts, four videos, three recording-date folds;
- zero recording-group, video, or window-fold leakage;
- exact class/source support-table reconciliation;
- all ten primary/repeat artifacts byte-identical;
- packed tensor SHA256
  `982724aa0c99852dd53805c2e1a7557daf0db9677f007e6fa32dd765eda4b105`;
- cache manifest SHA256
  `7cc08b8f2d7796a17d5294ad706cd82f6c7f6e27a3e55269d593664632732637`.
- refreshed L1 audit is bound to `aae63c37f5f16f032503ec4ef5bae966397e7396`;
- its dirty-worktree record contains only the three preserved user changes.

This PASS authorizes only the equivalent full legacy L2 data build. Model
training, canonical full OOF, reviewed/final naming, and Q2 claims remain false.
The cache/fold boundary remains `00dc2e0`; claim hardening rollback is
`git revert aae63c3`.

## Achievement Log

| Date | Achievement | Result | Commit |
|---|---|---|---|
| 2026-07-14 | Legacy development lane authorized | PASS | `a2323a7` |
| 2026-07-14 | Temporal tier controls documented | PASS | `1e9b393` |
| 2026-07-14 | Tier manifests and audits | PASS IN CODE | `ef0b3bd` |
| 2026-07-14 | Exact temporal model input binding | PASS IN CODE | `21b34fd` |
| 2026-07-14 | Absolute burst frame indices | PASS | `2049a2d` |
| 2026-07-14 | Strict short-packet loader audit | PASS | `c41f1ed` |
| 2026-07-14 | Legacy L1 cache and fold gate | PASS | `00dc2e0` |
| 2026-07-14 | End-to-end legacy lineage claims | PASS | `aae63c3` |
| 2026-07-14 | Deterministic full legacy L2 lineage | PASS | `59647e2` |
| 2026-07-14 | Immutable legacy L3 input gate | PASS | `0414adc` |
| 2026-07-14 | Legacy L4 model-correctness ladder | PASS | `3ef4235` |
| 2026-07-15 | Memory-safe L5 224px cache gate | PASS | `15a5368` |
| 2026-07-15 | Pretrained 4 GiB VRAM gate | PASS | `93449ae` |
| 2026-07-15 | Crash-bounded L5 cached-feature consumer | PASS | `b425c86` |
| 2026-07-15 | Deterministic L5 cached short training gate | PASS | `b6f74f7` |
| 2026-07-15 | Schema-v2 exact repeat gate | PASS | `4b52991` |
| 2026-07-15 | Full V0/T16 centered baseline | PASS | `22875cb` |
| 2026-07-15 | V1 resolution-only repeat gate | PASS | `4db26bf` |
| 2026-07-15 | Full V1/T16 resolution control | PASS | `3eb5a49` |
| 2026-07-15 | Hardware/data interpretation clarification | PASS | `65286f5` |
| 2026-07-15 | V2 backbone-only repeat gate | PASS | `df0df10` |
| 2026-07-15 | Full V2/T16 backbone control | PASS | `ae9cc43` |
| 2026-07-15 | T1 masked-TCN exact repeat gate | PASS | `5bddebc` |
| 2026-07-15 | Full T1/T16 temporal control | PASS | `14c6e4b` |
| 2026-07-15 | Legacy L5 strategy revision | PASS | `48f332d` |
| 2026-07-15 | T1-versus-V1 paired evaluator | PASS | `1989cdf` |
| 2026-07-15 | Immutable T1-versus-V1 decision config | PASS | `ff4a3d8` |
| 2026-07-15 | Temporal ladder engine and repeat gates | PASS | `4f3fc58` |
| 2026-07-15 | Immutable temporal short matrix | PASS | `cc38244` |
| 2026-07-15 | Deterministic CUDA workspace binding | PASS | `076501c` |
| 2026-07-15 | Immutable temporal full matrix | PASS | `fda8f43` |
| 2026-07-15 | Temporal ladder decision evaluator | PASS | `bec5560` |
| 2026-07-15 | Immutable temporal decision config | PASS | `a929e02` |
| 2026-07-15 | ROI relation cache core and audits | PASS | `8445ae8` |
| 2026-07-15 | ROI cache config and repeat gate | PASS | `3e59197` |
| 2026-07-15 | Crash-bounded ROI short trainer | PASS | `c32d3fa` |
| 2026-07-15 | Immutable ROI short matrix config | PASS | `6753ee7` |
| 2026-07-15 | Paired ROI promotion decision | PASS | `821e931` |
| 2026-07-15 | Full ROI authorization and config | PASS | `984a6c9` |
| 2026-07-15 | Full ROI decision evaluator | PASS | `e82d6c5` |
| 2026-07-15 | Full ROI confirmation decision | PASS | `29cfdd0` |

## L2 PASS Evidence

Roots:

```text
outputs/classification_v2/legacy_only_unreviewed_development/
full_legacy_lineage_v2_20260714
full_legacy_lineage_v2_repeat_20260714
```

- 72,864 exact frame/object rows and 4,554 native 16-frame bursts;
- T6/T8/T12/T16 sliding rows are 18,216/13,662/9,108/4,554;
- 45,540 all-sliding rows and 18,216 centered-matched rows;
- all ten labels, zero row loss, label drift, cross-burst windows, or bad timing;
- all eight strict loader views pass shape, timing, observation, and mask checks;
- all 24 primary/repeat lineage CSV pairs are byte-identical and claim-safe;
- 4,545 development-valid and 9 policy-invalid bursts are retained explicitly;
- consolidated audit has zero errors and is bound to `59647e2`;
- focused gate: 51 passed; classification regression: 467 passed, 181 deselected.

The audit is `08_l2_audit/legacy_development_l2_audit.json` under the primary
root. L2 authorizes only the L3 immutable-input freeze; model training remains
unauthorized.

## L3 PASS Evidence

- 72,864 RGB actor crops use exact 160x160 aspect-preserving letterbox;
- packed tensor shape is `[72864, 160, 160, 3]`, dtype `uint8`;
- all 72,864 packed rows passed fresh pixel and loader equivalence;
- zero cache misses, loader failures, source-media reads, or pixel mismatches;
- 45,540 windows inherit 12 recording-date folds without leakage;
- all 40 frozen artifacts verified by size, hash, rows, shape, and dtype;
- primary/repeat image, fold, class, and source manifests are byte-identical;
- `quality_mask` is control-only and forbidden metadata is excluded from X;
- length-to-label uplift and maximum T6/T8/T12/T16 class drift are `0.0`;
- the committed-SHA audit is bound to `0414adc` with zero errors;
- focused L3 tests: 8 passed; classification regression: 475 passed,
  181 deselected; Ruff, compile, diff, and long-line checks pass.

The audit is `13_l3_audit/legacy_development_l3_audit.json` under the primary
root. L3 authorizes only bounded L4 model-correctness work. Accuracy/F1
comparison, canonical full OOF, reviewed/final naming, and Q2 claims remain
unauthorized.

## L4 PASS Evidence

- exact T16 centered input and `native_oof_005` are config-hash frozen;
- all 4,554 native units remain in lineage and nine policy-invalid units are
  excluded from optimizer input;
- masked-value invariance, temporal order sensitivity, and invalid-mask
  rejection pass on real packed crops;
- visual, temporal, and behavior-head gradients are finite and nonzero;
- two deterministic steps match exactly, including model and optimizer state;
- checkpoint load and the next optimizer step are exactly equivalent;
- 20 unique real native bursts reach 1.0 memorization with loss ratio
  `0.0002209130`;
- one epoch covers all 3,897 eligible train bursts and 62,352 packed frames;
- cache misses, source reads, video decodes, and train/test group overlap are 0;
- peak VRAM is 464,460,288 bytes; held-out predictions and metrics are absent;
- focused L4 tests: 4 passed; classification regression: 479 passed,
  181 deselected; Ruff, compile, diff, and long-line checks pass.

The audits are under `14_l4_model_correctness` in the primary root. L4
authorizes only controlled L5 development baselines. Canonical full OOF,
reviewed/final naming, and Q2 claims remain unauthorized.

## L5 224px Cache Gate Evidence

- frozen config SHA256 is
  `24bfca98440b2cc5983451eb639868610d724eadf89ea36db907540cb8f86e4c`;
- local GPU is declared as 4 GiB with a 70% peak-VRAM ceiling;
- ResNet18/ResNet34 frame batches are capped at 16/8 and OOM retry is forbidden;
- all 72,864 direct-source 224px letterboxed crops exist with zero failures;
- packed tensor shape is `[72864, 224, 224, 3]`, dtype `uint8`, with zero
  missing files, shape mismatches, dtype mismatches, or pixel mismatches;
- the packer uses `flush_close_reopen_each_checkpoint_v1` and records 72
  mapping reopens; the all-row audit reopens its read-only mapping 36 times;
- strict packed loading records 60/60 packed hits, zero cache misses, zero
  source-media loads, and zero video decodes or seeks;
- focused tests: 24 passed; classification regression: 490 passed,
  181 deselected; Ruff, compile, diff, and long-line checks pass;
- the full audit is
  `15_l5_core_baselines/legacy_development_l5_224_cache_full_audit.json` and is
  bound to commit `15a5368` with status `PASS_LEGACY_DEVELOPMENT_L5_CACHE_FULL`.

This gate authorized only the exact pretrained-weight and short VRAM work
recorded below. It did not authorize a model-quality claim.

## L5 Pretrained Weight And VRAM Gate Evidence

- commit `93449ae` enforces a 210-character Windows weight-path limit after the
  first long-path attempt stopped before download or CUDA initialization;
- ResNet18 V1 weight SHA256 is
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`;
- ResNet34 V1 weight SHA256 is
  `b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f`;
- CPU-only weight preparation records CUDA uninitialized before and after;
- the RTX 3050 Laptop GPU reports 4,294,443,008 total bytes;
- the allocator is hard-capped at 3,006,110,105 bytes with no OOM retry;
- V0 ResNet18-160 batch 16 peaks at 125,829,120 reserved bytes;
- V1 ResNet18-224 batch 16 peaks at 171,966,464 reserved bytes;
- V2 ResNet34-224 batch 8 peaks at 169,869,312 reserved bytes;
- all controls run two exact passes over 64 spread rows with identical feature
  hashes, 512 float32 features, zero nonfinite values, and zero OOM;
- allocated and reserved CUDA bytes return to zero after every control;
- focused visual tests: 9 passed; classification regression: 499 passed,
  181 deselected; Ruff, compile, diff, and long-line checks pass.

The weight and probe audits are under `15_l5_core_baselines`. This gate permits
the exact cached-feature short gate and, only if that passes unchanged, its full
expansion. Baseline metrics, outer-holdout predictions, reviewed/final naming,
canonical full OOF, and Q2 claims remain unauthorized.

## L5 Pretrained Feature Cache Evidence

- commit `e34521d` adds isolated, resumable FP32 feature caches with immutable
  manifests, 2,048-row mmap checkpoints, a 70% allocator ceiling, and no OOM
  retry;
- CPU-only preflight resolves all six short/full V0/V1/V2 sources and leaves
  CUDA uninitialized; 8 focused tests and 507 classification tests pass with
  181 deselected, plus Ruff, compile, diff, and long-line gates;
- six 496-row short runs execute in separate non-overlapping processes; every
  primary/repeat feature tensor and index is byte-identical, with zero
  nonfinite values, source reads, OOM, retry, optimizer step, or retained VRAM;
- the short gate SHA256 is
  `529dc610066fdddb9804130a31b9da100a6ffa059f8807d8e7c94ad84d89df7a`
  and explicitly authorizes only the exact full feature-cache expansion;
- V0 full has shape `[72864, 512]`, tensor SHA256
  `a27f9f3ba589bc104016a132d562220b92defb6db756db19307ac0cee977f91b`,
  index SHA256
  `c62b950f8d0d2936ad737a0863e722a0b1858281c9e9d132a49b7b3769355d40`;
- V1 full tensor/index SHA256 values are
  `d5394e82343250904991d53b36f4113a07b8c6774c4b651a31de3da267f7b44b`
  and `34ede8120b49591a65099940e5ad1b6246f3381278ac3bccc16dd1f80223c5dc`;
- V2 full tensor/index SHA256 values are
  `c1ca185a919c2b8ee5b925ddc0ea5ba4f96d6655e458ce6d0593a4ba101ce992`
  and `3b984d70615ed2f351e7db709396ff5b11dfa3e5b5e68bf642a997882b72af32`;
- every full control completes 72,864 rows with 36 input mapping opens, matches
  its 64-row VRAM-probe feature hash, and returns CUDA allocated/reserved bytes
  plus process-level `nvidia-smi` usage to zero before the next control.

The three full packets contain the required run, environment, artifact,
checkpoint, prediction, and registry lineage. They authorize the controlled L5
baseline inputs only; no classifier metric or held-out prediction exists yet.

## L5 Cached-Feature Consumer Evidence

- commit `b425c86` adds the exact fold-to-window-to-slot-to-feature join,
  explicit 512-value model-X whitelist, canonical temporal head, and a
  CPU-only consumer audit CLI;
- the committed V0/T16 centered packet is
  `15_l5_core_baselines/cfd_v0_t16_b425c86`; run-manifest SHA256 is
  `199a6e5e3502f3c728910461ed1ddec18e976cba326fd34f61c6bf0b2269850b`;
- 3,652 train and 245 validation native units produce 3,897 model windows and
  62,352 exact slots; 648 outer units remain routing-only and create zero
  feature slots, predictions, or metrics;
- `fold_manifest.csv` contains 4,545 eligible units with zero recording,
  video, or native-unit role overlap; `native_routing_manifest.csv` preserves
  all 4,554 units, including nine explicit policy-invalid exclusions;
- the independent leakage checker accepts exactly 512 features with zero
  forbidden fields, and the native checker reports zero duplicate keys or
  incorrect 16-frame lengths;
- the local hardware contract is fixed at 4 GiB, 4,294,443,008 validated
  bytes, and a 3,006,110,105-byte allocator cap; FP32 and no AMP/retry remain
  mandatory;
- the bounded audit loads two 64-window batches per train/validation role,
  peaks at 2,103,552 loaded bytes per batch, closes the mmap after every batch,
  and records CUDA uninitialized before and after with peak VRAM zero;
- eight focused tests and 515 classification tests pass with 181 deselected,
  plus Ruff, compile, diff, and changed-file line-length gates.

This PASS authorizes implementation and execution of the exact cached-feature
short training gate only. L5 remains `IN_PROGRESS`; baseline metrics,
outer-holdout predictions, reviewed/final naming, canonical full OOF, and Q2
claims remain unauthorized.

## L5 Cached Short Training Gate Evidence

- commits `9a878e7` and `b6f74f7` add the immutable V0/T16 centered short
  config, batchwise mmap training, fresh-process CUDA runner, complete lineage,
  repeat comparator, and explicit release of persistent cuBLAS workspaces;
- config SHA256 is
  `921868d721d6e584d0880f1bd5bdd6f8d7386b6131fe95cac69310cfa4e21d15`;
- both runs use the same 80 class-balanced training native units, all 245
  validation native units, 68,234 model parameters, three epochs, and nine
  optimizer steps; outer-holdout rows, predictions, and metrics remain zero;
- primary `ct_v0_t16_p_b6f74f7` and repeat `ct_v0_t16_r_b6f74f7` run in
  distinct PIDs 11,460 and 19,104 with non-overlapping intervals;
- selected epoch is 3; bounded development macro-F1 is `0.2273314111`,
  accuracy is `0.2897959184`, and native-unit NLL is `1.9787436113`;
- parameter, prediction, and epoch-metric SHA256 values match exactly at
  `90f8b125e90f59d2681d370c7fcfef50dc3c448525b9d05a6b475d321d092ff7`,
  `fea4fbb59f33c59e083fc74d3c45ba9445802b8d4fe2d1e92c9a8768fa4fe89a`,
  and `c201c65237d5dd3b38d1ae4808796801b71d6988d0d7ed01f1d94af3c2183ff6`;
- maximum loaded batch is 2,103,552 bytes and peak reserved VRAM is 94,371,840
  bytes in each run; post-cleanup allocated and reserved bytes are both zero;
- the first committed attempt `ct_v0_t16_p_9a878e7` is preserved as FAIL
  because a 64 MiB cuBLAS workspace remained inside the live process; it did
  not OOM, retry, overlap another process, or leave GPU memory after exit;
- all 13 required artifacts in each passing packet independently match their
  manifest hashes; focused tests are 11 passed and classification regression is
  526 passed with 181 deselected, plus Ruff, compile, diff, and line gates;
- short-gate SHA256 is
  `f63fe685fd7fb7285c2a9393f15111742a8601e6bf27937b5f1ea762a9e2fc1f`.

This PASS authorizes only the exact full V0/T16 centered cached-feature
expansion. Other visual or temporal controls remain unauthorized, L5 stays
`IN_PROGRESS`, and all reviewed/final, canonical full-OOF, and Q2 claims stay
false. Rollback is `git revert b6f74f7` followed by `git revert 9a878e7`.

## L5 Schema-V2 And Full V0/T16 Evidence

- commit `4b52991` adds the crash-bounded schema-v2 trainer, audit command,
  focused tests, and immutable short-v2 config;
- short config SHA256 is
  `e0de0686e4e18085f649ba701dbd7d46d785bf6092474a304f0ad45c15682045`;
  implementation SHA256 is
  `b6e4a8f2b6376c58c40595656a37e7a9c3e042a860127a3825471ab04278dca7`;
- runs `ct_v0_t16_p_4b52991_v2` and `ct_v0_t16_r_4b52991_v2` use distinct
  PIDs 10,332 and 16,584, 80 balanced train units, all 245 validation units,
  and nine optimizer steps without overlap;
- their parameter, prediction, and epoch-metric SHA256 values match exactly at
  `90f8b125e90f59d2681d370c7fcfef50dc3c448525b9d05a6b475d321d092ff7`,
  `05643dc8e21448c3c831e0b75be76848dedad0803df1a50c59ba33e19562fb7a`,
  and `7ccd3a3309ced7f87254efe81c5e71c5802de6a94fc4db9eb4d65d5f95327cee`;
- the 14-field short gate passes with SHA256
  `84250309d0edb7b56d6989edc9040d668ccc203b8dc9673073409e9c61045dbb`;
- commit `22875cb` binds the exact full V0/T16 centered config with SHA256
  `77b4aa21a46335d93e5cacdc4f29daa3c9983eccf9a2bfc3ffa1c179f8927092`;
- run `ct_v0_t16_full_22875cb_v2` passes on 3,652 train and 245 validation
  native units for three epochs and 345 optimizer steps in 4.4476819 seconds;
- selected epoch 3 has bounded development macro-F1 `0.3486204147`, accuracy
  `0.6000000000`, and native-unit NLL `1.2490196145`;
- result, run-manifest, and checkpoint SHA256 values are
  `1becf8ea2bcc471a2dcb776fb0348cbbe05dca3530593c151d437baf0bbdb223`,
  `a077b96622c9ba86cf945f585234b1f19280c368a62fc2878c91563047272538`,
  and `1f46e2fc16ff5da0fe833f2667dc212217897162f35976ee0b02e8c6e0fb24c6`;
- peak reserved VRAM is 94,371,840 bytes and cleanup returns allocated and
  reserved bytes to zero in every run; no OOM, retry, AMP, source read, outer
  row, outer prediction, or concurrent GPU process occurred;
- CLI packet audit and the independent checker pass all 13 required artifacts.

That gate authorized only full V0/T16. The separately gated V1 resolution
evidence follows below. Canonical full OOF and Q2 claims remain false.

## L5 V1 Resolution-Only Evidence

- commit `4db26bf` adds schema-v3 V1 gating and binds the only scientific
  change as ResNet18/160 to ResNet18/224 at fixed weights, T16 view, temporal
  head, optimizer, split, subset, and claim boundary;
- the V1 consumer `cfd_v1_t16_c921671` and all 25 declared parent artifacts
  pass CPU mmap, leakage, outer-access, and independent hash audits;
- short config and implementation SHA256 values are
  `d868138c3cc9ab9fa6a2b37c1068d8771a97bd20bc615e23680152f60309c8e3`
  and `9a1525382df94935ad780179a3d264aaeb975bf236402bb5c00e789c5d9a219b`;
- `ct_v1_t16_p_4db26bf_v3_cuda` and `ct_v1_t16_r_4db26bf_v3_cuda` use
  distinct PIDs 6,668 and 7,764, 80 balanced train units, all 245 validation
  units, nine optimizer steps, and non-overlapping intervals;
- all 14 equality fields pass; parameter, prediction, and epoch-metric hashes
  are `d1e4a953a2b3ed4b7e3b0267e3f826ca1f819669c3c37a07d4a506d893301424`,
  `caf7ca082aee1d52d7b3fb5ab06ecfaf41792015377207a23e67833b9cf3f2ce`,
  and `d400517b47ce14ff46c211f3ec765dc12718bf2b8c9d383ec40f2461119b4e67`;
- the V1-only gate SHA256 is
  `51f1cb805e0fde63f1a3f40b34863f94ef65aaf78c0830322431e68b5315aa44`;
- the CPU-only `.venv` attempt `ct_v1_t16_p_4db26bf_v3` is preserved as FAIL:
  it stopped before model creation with zero steps and zero peak VRAM, and is
  excluded from the gate; the CUDA 12.1 environment was then matched to V0;
- commit `3eb5a49` binds full config SHA256
  `5665a8316112951562b178b7940b2e2124e72515b3007701017aff6d4e05cf15`;
- `ct_v1_t16_full_3eb5a49_v3` passes 3,652 train and 245 validation units,
  three epochs, and 345 optimizer steps in 4.5933299 seconds;
- selected epoch 1 has bounded development macro-F1 `0.3528183193`, accuracy
  `0.6571428571`, and native-unit NLL `1.0717006652`;
- result, run-manifest, and checkpoint SHA256 values are
  `af283e487730ab9769946b7430601173abe8fafc289d74bf1c05b01cc89a6ad7`,
  `f13b75f2caa15323684448bb670c087a9711d0e6fc6286d007db23b81a304958`,
  and `9d9cfce1642eeb1b1be12110fe609a9831960ab6d37fda66fd33719aad2f8df6`;
- all passing runs peak at 94,371,840 reserved bytes, clean up to zero, and
  have no OOM, retry, AMP, source read, outer prediction, or process overlap.

## L5 V2 Backbone-Only Evidence

- commit `df0df10` adds schema-v4 V2 gating. The only visual-family change is
  ResNet18/224 to ResNet34/224; ImageNet-1K V1 family, T16 view, masked-mean
  head, optimizer, seed, split, subset, and claim boundary remain fixed;
- short config and implementation SHA256 values are
  `525798bf2de9ec4917993539597384dd997e9cbe2562528b85895aca89bfb9ff`
  and `512e85666e790e0c66c8bbace50b65265e37142e41ddcf4ae8acf7aa0eb7eb07`;
- runs `ct_v2_t16_p_df0df10_v4` and `ct_v2_t16_r_df0df10_v4` use distinct
  PIDs 13,512 and 2,732 with non-overlapping intervals and nine steps each;
- all 14 equality fields pass. Parameter, prediction, and epoch-metric hashes
  are `d021ff43e49da67ebb2f4b37335524a33414ff0e01144e7db661135ca0234328`,
  `a44e6e46fe6362b01fc97a3b85ad0307c3b824fb1aa3d89e672f8366d4a45173`,
  and `b443f0da0f7e1c0a95d27a05c8895ed0f39aff42ad0681ef1365dc69cd25d27b`;
- the V2-only short gate SHA256 is
  `317e2385ef32359f97024c2a81e945889fd5d0a5ace1407c60d277e8139905cb`;
- commit `ae9cc43` binds full config SHA256
  `b0fbcba16774f7efe2ad0981f2d4e544187e83dbcd6be332118e0a89f5eec9a6`;
- `ct_v2_t16_full_ae9cc43_v4` passes 3,652 train and 245 validation units,
  three epochs, 345 steps, and selects epoch 3 in 4.5280627 seconds;
- bounded legacy metrics are macro-F1 `0.4245712948`, accuracy `0.6163265306`,
  and native-unit NLL `1.0568459007`;
- result, run-manifest, and checkpoint SHA256 values are
  `ca7d070a343afcc2bad57e40c3e40d2c6c8311235888e080e67a1192af14a6e2`,
  `87591f58b56ad9d6a768d3e4a9527ac32763267f9b929e440df0691ccfa280d2`,
  and `7553d38fca70be3f3ac2878058314278b9c877c8bb3bb712d907d37bc5f65b8d`;
- all 13 artifacts pass independent hash audits. Peak reserved VRAM is
  94,371,840 bytes with cleanup `0/0`, no OOM/retry/source read/outer access;
- 52 L5 tests and 533 classification tests pass with 181 deselected.

Against V1, paired macro-F1 changes by `+0.071753`, but the exploratory
33-video cluster-bootstrap interval is `[-0.038441, 0.135424]`; accuracy changes
by `-0.040816`. This does not prove promotion on one legacy validation date.
V1 ResNet18/224 remains the efficient temporal-search control and V2 remains
the capacity reference. Neither result estimates merged-data rare support.

## L5 T1 Temporal-Control Decision Evidence

- commit `5bddebc` adds the schema-v5 masked-TCN gate; two short runs match on
  parameter, prediction, and epoch hashes across all 14 equality fields;
- commit `14c6e4b` binds the full T1 config SHA256
  `747ae1da9b6558be8a12ca1d085d70c52cccad828770160b4bfac1ee557f72a4`;
- `ct_t1_t16_full_14c6e4b_v5` trains 3,652 units for three epochs and selects
  epoch 1 with macro-F1 `0.3353958340`, accuracy `0.6448979592`, and NLL
  `1.0569776080` in `33.8660293` seconds;
- all 13 T1 artifacts independently pass their hashes; peak reserved VRAM is
  98,566,144 bytes with cleanup `0/0` and no OOM, retry, or outer access;
- commit `1989cdf` adds the isolated evaluator and three focused tests; commit
  `ff4a3d8` binds its exact config without changing the historical trainer;
- 20 focused evaluation tests and 538 classification tests pass, with 181
  deselected; Ruff, compile, diff, and changed-code line gates pass;
- the paired artifact covers the exact same 245 units and 33 videos with 2,000
  deterministic video-cluster bootstrap iterations;
- T1 minus V1 is `-0.0174224853` macro-F1, `-0.0122448980` accuracy, and
  `-0.0147230573` NLL; the macro-F1 interval is
  `[-0.0412497075, 0.0048069076]`;
- the four-class legacy rare group has only 20 units and recall `0.0` for both
  models. This is no positive rare-class evidence and says nothing about the
  substantially larger rare support in merged-reviewed data;
- T1 uses 167,435 parameters versus 68,234 and takes `7.3728711x` the runtime;
  its peak reserved VRAM is only `1.0444444x`, so VRAM did not reject it;
- decision: `RETAIN_V1_REJECT_T1_FOR_LEGACY_T16_SEARCH`; T1 fails the primary,
  positive-CI, simpler-model margin, and runtime criteria;
- Transformer is deferred for this legacy search. TCN and Transformer remain
  eligible for fresh gated evaluation on frozen merged-reviewed data or rented
  GPU hardware when scientifically justified.

The artifact is `legacy_l5_t1_v1_paired_decision_v1.json` under the L5 output
root. File SHA256 is
`f6b9b7418272786cf9d5a0dc247ed912699b25f25d1d2fbbb8b9260c8aa17266`;
payload SHA256 is
`a1fff465de092f32c694a8bdd317e8ff0311b98c192db2f1511c869a1eae8a8f`.

That decision authorized the frozen V1 masked-mean T6/T8/T12/T16 ladder under
both temporal sampling protocols. The completed ladder evidence follows.

## L5 Temporal-Ladder Decision Evidence

- all eight full views pass 14/14 artifact audits after the exact short matrix
  passed 8/8 deterministic primary/repeat checks without OOM or retry;
- the short-matrix gate SHA256 is
  `b36e03d85e7f8090b1e8542948fb9cbe0a6db1273339b7c3000a5930f0a471f9`;
- the full config SHA256 is
  `559d24ba29d6192291eb5a8d7b6896616b1bed782450a9242f1c4afa3bd85559`;
- T6 sliding has macro-F1 `0.5343181014`, accuracy `0.6857142857`, and NLL
  `1.1206917637` on the same 245 native units and 33 video clusters;
- T6 sliding minus T6 centered macro-F1 is `+0.1713935431`, with paired
  video-cluster interval `[0.0218242521, 0.2271418980]`;
- T6 sliding minus T16 centered is `+0.1599775548`, with interval
  `[0.0005092838, 0.2220031926]`;
- intervals cross zero against T8 centered, T12 centered, and T8 sliding, so
  universal pairwise superiority is not established;
- the rare group has only 20 units; T6 sliding recall is `0.25` with no recall
  drop against another view, but this support is not representative of merged;
- T16 centered and sliding are exactly equivalent in parameters,
  probabilities, and predicted labels as required by their shared one window;
- the official evaluator and Git guard pass at committed code SHA `d68232c`;
  this reissue changes only the ambiguous dataset alias in decision metadata,
  while all eight bound model runs and numerical results remain unchanged.

Decision: retain `t6_sliding` only as the bounded L6 working baseline. Sliding
also changes window count and optimizer exposure, so no causal claim about T6
length alone is allowed. Three epochs are bounded baseline evidence, not final
convergence evidence. The architecture family is not finalized, and all
retained or rejected controls must be reassessed on frozen merged-reviewed
data, whose rare-behavior support is materially larger than `legacy_16f`.

The official artifact is
`legacy_l5_temporal_ladder_decision_v1.json` under the L5 output root. File
SHA256 is
`f3316568c797967e66a0a6ea3cac8e4804a35e12439b3d31527e10ef93abd4e1`;
payload SHA256 is
`9f727cf248401eea718ccab2c9f367f9cc2c3ea88e4a214d0e9831badf4c7919`.

L5 is `PASS`. This authorizes L6 geometry-first work from the exact T6 sliding
baseline only; reviewed/final naming, canonical full OOF, and Q2 claims remain
false.

## L6 Geometry Cache Evidence

The frozen T6 geometry cache is the first L6 input-family artifact. It binds
the raw `legacy_16f` authority, the L5 decision, image-context manifest,
T6 slot manifest, and frame-geometry reference.

- primary and independent-repeat caches each contain 15,588 model-visible
  windows, 93,528 slots, and an exact `[15588, 6, 8]` float32 tensor;
- the explicit fields are `cx_n`, `cy_n`, `bw_n`, `bh_n`, `area_n`,
  `aspect_ratio`, `box_diag_n`, and `box_compactness`;
- all four derived artifacts are byte-identical across the two output roots;
- all 93,528 slots are available in this lane, so availability-only remains a
  constant-channel diagnostic and is not behavior evidence;
- recomputed geometry matches `frame_geometry.csv` within `1e-12`, with zero
  media reads and zero outer-holdout slots materialized;
- the source probe correctly reports `NOT_ESTIMABLE_SINGLE_LEGACY_SOURCE` for
  `source_type=legacy_recovered` and `dataset_id=legacy_recovered_16f`.

The primary manifest SHA256 is
`0e0b79a5423f820193717995a89748817ab9e3e62e078694d1b838521c37b5a7`.
The repeat manifest SHA256 is
`f9df0e5e0361f655b6dd51b1c8e9a8b81e0232509c199fbb94d1a996d3584b60`.
The PASS repeat-gate file SHA256 is
`d84c1a88b1067b4c3bfeaca8bfb5cd03f860fb85bff829fe80ecf145139decd1`
at evaluator commit `1cb0798`.

The geometry training implementation is now `PASS IN CODE` before GPU launch:

- all three controls use the same 521-wide model and 69,404 parameters;
- normalization fits unique train `frame_uid` rows only and records zero
  validation or outer-holdout reads;
- missing-modality inference zeros geometry and availability together;
- the runtime permits one run per fresh process, caps the allocator at 70% of
  detected VRAM, forbids OOM retry, and requires zero post-run VRAM residue.

The first config-v1 canary was preserved as a failure packet. Training reached
the cleanup gate without OOM, but CUDA retained a 64 MiB cuBLAS workspace after
`empty_cache`; the process then exited and `nvidia-smi` returned to zero usage.
No automatic retry occurred. Config v1 SHA256 is
`35351189226fc2dab00c6bb2c8e34646ea9bf32591b0d321937230113eeb8046`.

Runtime v2 explicitly clears cuBLAS workspaces before testing allocated and
reserved bytes, matching the already proven L5 cleanup policy. Its fresh output
root is separate, and config v2 SHA256 is
`68e68b2db4f36c6abfdd727dee414197d3541e5f38f0b5c32d4244fb1847c377`.

The config-v2 canary passed CUDA training and cleanup, then exposed a Windows
artifact-path failure: the first blocked filename was 274 characters long.
The partial planned packet is preserved; GPU usage again returned to zero, and
no retry occurred. Runtime v3 now checks every artifact path before preflight or
CUDA and converts a finalization exception into an auditable failed packet.

Config v3 uses a separate shorter output root. Its longest declared canary path
is 205 characters against a conservative 240-character gate. Config v3 SHA256
is `51462300634a10244217cc1290c7af8f8a9e6597a16a47859cec41ac7e3e8ab6`.

Config v3 passed all three committed CPU preflights and all six fresh-process
short runs. Every run used 30 optimizer steps, peaked at 73,400,320 reserved
VRAM bytes, reported no OOM or retry, and cleaned allocated/reserved bytes to
zero. Each mode is byte-deterministic across its two runs.

Native validation point metrics are:

- parameter-matched zero: macro-F1 `0.2675196654`, accuracy `0.3959183673`;
- availability-only: macro-F1 `0.2642331606`, accuracy `0.3918367347`;
- geometry: macro-F1 `0.2982000716`, accuracy `0.4408163265`.

The six-process short matrix is `PASS`; SHA256 is
`aeeb3b14101726f68cabf63d52f9bad0f5327ebf88bb5dc6b28ee8a88f5dbc3c`.
The predeclared paired native/video-cluster decision is also `PASS`:

- geometry minus parameter-matched zero macro-F1 is `+0.0306804062`, with
  33-video cluster interval `[0.0058898605, 0.0597986455]`;
- geometry minus availability-only macro-F1 is `+0.0339669109`, with interval
  `[0.0052591705, 0.0667705673]`;
- geometry improves NLL against zero by `0.0313320951` and accuracy by
  `0.0448979592`;
- rare-group macro-F1 changes by `-0.0078361982` against zero on only 20 units;
- availability-only minus zero is `-0.0032865047`, and its interval crosses
  zero, as required for the constant-channel diagnostic.

The decision is `RETAIN_GEOMETRY_FOR_FULL_LEGACY_DEVELOPMENT`. Its artifact
SHA256 is
`180f91ff038c6c661717330365bd77df08c70883da878a287f1471fe631586f8`.
The full authorization gate SHA256 is
`e4637f4d6640d85a046f51c5f76d9f13a44dbff07e99280479c82c0f984aae18`,
and the full config SHA256 is
`20e7ba457f27abd1e8ed93341ac77b348985f8e62f4eab39311d90d38b945198`.
L6 remains `IN_PROGRESS`: full geometry evidence must pass before geometry can
be retained, and this legacy-only result cannot be generalized to merged data.

### Full Geometry Confirmation

All three full packets passed independent audit at config commit `ae3303b`:

- each used 3,652 train native units, 14,608 windows, three epochs, and 1,371
  optimizer steps with the same selection and normalization hashes;
- every mode used 69,404 parameters and peaked at 73,400,320 reserved bytes;
- cleanup was 0/0 allocated/reserved, with no OOM, retry, media read, or outer
  prediction.

The paired evaluator at commit `3f02382` closed 245 validation native units and
33 video clusters. Full point metrics are:

- parameter-matched zero: macro-F1 `0.4852128908`, accuracy `0.6489795918`;
- availability-only: macro-F1 `0.4413578715`, accuracy `0.6653061224`;
- geometry: macro-F1 `0.4470778216`, accuracy `0.6693877551`.

Geometry minus zero macro-F1 is `-0.0381350692`, with video-cluster interval
`[-0.0872930139, 0.0770815601]`. Geometry improves NLL by `0.0694050386` and
accuracy by `0.0204081633`, but its rare-group macro-F1 changes by
`-0.1245920746`. Geometry minus availability-only is only `+0.0057199501`, with
interval `[-0.0493362817, 0.0686686375]`. Availability-only minus zero is
`-0.0438550192`, so the constant-channel diagnostic is not bounded at full
exposure.

The valid negative decision is
`REJECT_GEOMETRY_AS_UNSUPPORTED_IN_FULL_LEGACY_DEVELOPMENT`. Continue L6 motion
from the parameter-matched zero control. Reassess geometry on merged-reviewed
data; this rejection does not transfer to that lineage. The decision config
SHA256 is
`70fb72b3819d76db8572fe715df04eb2b9f8679a030ef14de30fa24cbc235ab2`;
the decision artifact SHA256 is
`25f4e9919c3579b16ea393678027d95fc706c975b9c57be99b981cca654a2e04`.

## L6 Motion Cache Evidence

Commits `f6090ea`, `61061a3`, and `adf2cb6` freeze the canonical T6 motion
cache and its independent repeat gate. Motion starts from the parameter-matched
zero control; rejected geometry values are used only as ordering authority and
never enter the motion tensor.

- primary and repeat each contain 15,588 windows, 93,528 slots, and exact
  `[15588, 6, 10]` float32 motion tensors;
- 77,940 within-window frame pairs are available, while exactly 15,588 first
  slots are zero and unavailable and no later slot is unavailable;
- 13,277 windows start after frame zero, and 11,691 of their raw first rows have
  nonzero burst-level motion that is reset to zero before cache materialization;
- no unit aggregate, geometry value, media read, or outer-holdout slot enters
  the cache;
- motion, availability, window-index, and slot-index artifacts are byte-equal
  across separate primary and repeat builds.

The primary manifest SHA256 is
`dabc971259e437d1374cea0f4f0c850415b02c2dbf2990a8d763d0843a6bf825`;
the repeat manifest SHA256 is
`5736e89cc1bed977a3e3a077af034e51f9d15509ce3e207c88a560ea2b961914`.
The PASS repeat gate SHA256 is
`6cdc1c796cc48430045161fc880fa8eb0031aec1ef0ce9e89900a1c344d04a17`.

The first v1 build exposed a CSV empty-string/NaN audit mismatch after writing
its artifacts. It was not retried or promoted; the preserved failure packet
SHA256 is
`13177898608040aec8886d338c365c41c38f78b7136ff3db5d916722557bfd85`.
The v2 build fixed only that serialization audit and reran both caches from
fresh output roots.

### Motion Short Matrix And Decision

Commits `cc01582`, `96e25b4`, and `6447c94` close the crash-bounded short
motion matrix and paired decision. All three CPU preflights and six separate
GPU processes passed. Each run used 30 optimizer steps, peaked at 73,400,320
reserved bytes, had no OOM or retry, and cleaned CUDA allocation/reservation to
zero. All per-mode repeats are byte-deterministic.

Native validation point metrics are:

- parameter-matched zero: macro-F1 `0.2620738697`, accuracy `0.4204081633`;
- availability-only: macro-F1 `0.2572327329`, accuracy `0.4122448980`;
- motion: macro-F1 `0.2602600258`, accuracy `0.4244897959`.

Motion minus zero macro-F1 is `-0.0018138438`, with 33-video cluster interval
`[-0.0260250944, 0.0233049368]`. Motion minus availability-only is
`+0.0030272930`, with interval `[-0.0162803684, 0.0235850302]`. Motion improves
NLL against zero by `0.0216145856`, but point macro-F1, both promotion margins,
and both positive-lower-bound criteria fail. The rare-group macro-F1 change
against zero is `-0.0031650335` on only 20 units.

The valid negative decision is
`DO_NOT_EXPAND_MOTION_FROM_CURRENT_SHORT_EVIDENCE`. Do not run full motion and
do not put motion values into the next candidate. Continue L6 with all-class
ROI relations from the parameter-matched T6 base. Reassess motion on frozen
merged-reviewed data; this rejection does not transfer to that lineage.

The short config SHA256 is
`712d80ca6e6d5fd0761d3215f811edab9a360c84f37f3d9bcebb2bbee9702007`;
the short matrix SHA256 is
`fb7245efbe445bb4fb83ecb28a915742c5a5a93d09ffc70c3641864b06b2610a`;
the decision config SHA256 is
`084818121bea41da149dde955db940aedc4780c9f8f8474f7e24d7e4d573aa63`;
the PASS decision artifact SHA256 is
`9be7fd93854771b93b58bbcd9f8fa1ed96a8ddbda8e65fa8e68a4baf30a4d978`.

L6 remains `IN_PROGRESS`; the next one-family gate is all-class ROI relations.

## L6 ROI Relation Cache And Short Decision

Commits `8445ae8`, `92f7925`, `9491899`, and `3e59197` freeze the canonical
all-class ROI relation cache and its independent repeat gate.

- The cache has 15,588 windows, 93,528 T6 slots, and 18 explicit feeder,
  drinker, and toy relation features; every slot is available.
- Geometry supplies row order only. No geometry value, target-selected field,
  aggregate, label, identifier, path, fold, or review field enters the tensor.
- All four data artifacts are byte-identical across independent builds.
- Primary and repeat manifest SHA256 values are
  `6d62a2ab025619808a605313678da94efd9a6a585d9595866186d4116dd08138`
  and `61d3c3de23fc19971f7833da0834ccd87c68cdd2bfa1772c215f50b5d5f7c753`.
- The PASS repeat-gate SHA256 is
  `5b0f7bb0e31c13af3da0d4dab2f48d544f1fa6b10c4e9b18347fcf2cede05918`.

All three CPU preflights and six separate GPU processes passed. Every run used
30 optimizer steps and 70,704 parameters, peaked at 73,400,320 reserved bytes,
had no OOM or retry, and cleaned CUDA allocation/reservation to zero. The zero,
availability-only, and ROI macro-F1 values are `0.2420943922`, `0.2405788407`,
and `0.2886109023`; all repeats are deterministic.

ROI minus zero macro-F1 is `+0.0465165101`, with 33-video cluster interval
`[0.0134524177, 0.0768469401]`. ROI minus availability-only is `+0.0480320616`,
with interval `[0.0178011460, 0.0771756533]`. Availability-only minus zero is
`-0.0015155515`, and its interval crosses zero.

The valid short decision is `RETAIN_ROI_RELATION_FOR_FULL_LEGACY_DEVELOPMENT`.
The short matrix SHA256 is
`29e9a2c7fe41979c92083b84b7bc8f354a7db671ddef263e61269b318a9bdda6`;
the decision artifact SHA256 is
`a4ec60c2850efd84d3d52cf295ace0c22a180a15c93be8d21a6617ec73bf186d`.
The short result authorized one exact full confirmation; it did not promote
ROI into the next candidate.

## L6 ROI Full Confirmation Decision

The hash-bound full confirmation is the file
`l6r_full_decision_v1.json` under
`outputs/classification_v2/legacy_only_unreviewed_development/l6r_full_v1/`.
Its artifact SHA256 is
`5a9a2b4b61b7ddeef0b5155ec69b678d73f0acd53917db98d1d6271cab5f1af3`.
The full training config SHA256 is
`6ea481082e69f632395ef1483f3986214488a639e3d7fc8857a2121f96bf1103`;
the authorization-gate SHA256 is
`9764b6a518a0153fe66f27151321fdf005b1f2d6419519bcd19b719e211a2f90`.

- Full controls use the same T6 native-unit universe and 70,704 parameters.
- Zero, availability-only, and ROI macro-F1 are `0.4966025667`,
  `0.4727197983`, and `0.5082292933`.
- ROI minus zero is `+0.0116267266`, with 33-video cluster interval
  `[-0.0398806556, 0.0906766805]`; ROI minus availability-only is
  `+0.0355094951`, with interval `[-0.0248897889, 0.0986581204]`.
- Availability-only minus zero is `-0.0238827684`, with interval
  `[-0.0629523019, 0.0339059054]`.
- All three fresh GPU processes used 1,371 steps, peaked at 73,400,320
  reserved bytes, had no OOM/retry, and cleaned allocation/reservation to 0/0.

Decision: `DO_NOT_EXPAND_ROI_RELATION_FROM_CURRENT_SHORT_EVIDENCE`.
The full ROI gain misses the required margin and positive interval-low gate;
the availability diagnostic also fails its bounded-difference check. Do not
carry ROI values into the next candidate. Continue L6 numeric social relations
from the parameter-matched T6 zero control. This rejection is limited to
unreviewed `legacy_16f`; reassess ROI on merged-reviewed data, whose rare-class
support is materially larger. L6 remains `IN_PROGRESS`; canonical full OOF and
merged-data claims remain unauthorized.

## L6 Numeric Social Relation Cache Evidence

Commits `6372dc6` through `7a42ce6` freeze the numeric-social cache and its
independent repeat gate. The cache exposes ten explicit distance, overlap,
density, contact, relative-motion, and aggression-proxy values. Partner IDs are
audit metadata only; top-K partner, geometry, motion, and ROI values do not
enter model X.

- Primary and repeat tensors have shape `[15588, 6, 10]`; 92,664 of 93,528
  slots are available and 864 are explicitly unavailable.
- All 15,588 windows are rebased locally, with 74,669 valid consecutive
  same-partner pairs, zero media reads, and zero outer-holdout slots.
- Social tensor, availability, window index, and slot index are byte-identical
  across the two output roots.
- The source probe is `NOT_ESTIMABLE_SINGLE_LEGACY_SOURCE`, as required for
  this single-source lane.

The primary and repeat manifest SHA256 values are
`5a0f66842e4fd0d8af363d3da1ebb762edd118586d62dd8a3bea4f4e6399a192` and
`c5adf7bb08ed86a5f3b4cdfc0d8aade88328fe354fb251cadabf10f1c23af111`.
The PASS repeat-gate SHA256 is
`3d4206c6679bc8f0cebe77c6da764ce8edb29deb2417f0cfacf82b6311d28d9f`.

This closes cache materialization only. The next gate is the parameter-matched
three-mode short trainer: zero, availability-only, and numeric-social.

## Full-Run Boundary

The full legacy L2 data build is complete. Any semantic change invalidates its
dependent evidence. Canonical full OOF remains outside this goal.

## Handback Boundary

Do not mark this goal complete until L0-L8 PASS and
`legacy_16f_goal_completion_audit.json` exists with immutable hashes. After
completion, update project memory and return to the parent chat. The parent
goal must re-audit reviewed all-source P0-P8 rather than inheriting a false PASS.
