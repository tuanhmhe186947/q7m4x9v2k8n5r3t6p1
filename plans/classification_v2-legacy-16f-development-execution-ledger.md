# Classification V2 Legacy 16F Development Execution Ledger

Version: 1.0

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
| L5 core baselines | IN_PROGRESS | Short train PASS at `b6f74f7`; full V0/T16 centered next |
| L6 modality loop | NOT_STARTED | Requires retained L5 baseline |
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

## Full-Run Boundary

The full legacy L2 data build is complete. Any semantic change invalidates its
dependent evidence. Canonical full OOF remains outside this goal.

## Handback Boundary

Do not mark this goal complete until L0-L8 PASS and
`legacy_16f_goal_completion_audit.json` exists with immutable hashes. After
completion, update project memory and return to the parent chat. The parent
goal must re-audit reviewed all-source P0-P8 rather than inheriting a false PASS.
