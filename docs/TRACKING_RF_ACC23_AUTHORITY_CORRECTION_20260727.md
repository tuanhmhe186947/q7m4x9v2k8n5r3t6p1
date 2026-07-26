# RF_ACC23 authority correction and lineage lock — 2026-07-27

Verification-first remediation of the Causal Realtime Tracking implementation
represented by `RF_ACC23`. **No tracking algorithm was changed.** The H1
candidate was not implemented, because the gate that authorizes candidate work
(a verified RF_ACC23 baseline) cannot be closed in the current environment.

---

## 0. Blocking finding — the named audit authority does not exist

The task designates `BAO_CAO_AUDIT_RF_ACC23_20260726.md` as the primary audit
authority. **That file does not exist and has never existed in this
repository.**

Evidence:

| Search | Result |
|---|---|
| Working tree (`find` on all non-ignored paths) | not found |
| Git index (`git ls-files`) | not found |
| Every commit on every ref (`git ls-tree` per commit) | not found |
| Case-insensitive patterns `BAO`, `AUDIT`, `RF_ACC`, `ACC23` | no match |

Consequently every claim attributed to that report is treated as
`LINEAGE_REPORTED_UNVERIFIED` unless independently confirmed below. Nothing was
reconstructed or inferred to fill the gap.

The string `RF_ACC23` appears in exactly **one** tracked file in the entire
repository: `scripts/README.md`, line 79, which describes `realtime_fast` as
"cau hinh causal tot nhat hien tai, co frame skipping va RF_ACC23". It appears
**nowhere** in `.agents/memory/`.

---

## 1. Authority files actually read

| File | Lines | Status |
|---|---:|---|
| `AGENTS.md` | 112 | read |
| `.agents/AGENTS.md` | 188 | read |
| `.agents/memory/00_README.md` | 33 | read |
| `.agents/memory/01_PROJECT_MEMORY_SHORT.md` | 1459 | read (targeted) |
| `.agents/memory/02_CURRENT_DECISION.md` | 1667 | read (targeted) |
| `.agents/memory/03_PROJECT_RULES.md` | 416 | read (targeted) |
| `.agents/memory/04_PROJECT_MEMORY_MEDIUM.md` | 40 | read |
| `.agents/memory/05_PROJECT_MEMORY_LONG.md` | 188 | read |
| `.agents/memory/06_BENCHMARK_NOTES.md` | 497 | read (targeted) |
| `.agents/memory/07_LEGACY_DIFF_NOTES.md` | 61 | read |
| `.agents/memory/08_WORKFLOW.md` | 1120 | read (targeted) |
| `Kế Hoạch Tương Lai.md` | 776 | read (targeted) |
| `BAO_CAO_AUDIT_RF_ACC23_20260726.md` | — | **MISSING** |

Skills present and consulted: `tracking-experiment-guardian`,
`experiment-lineage-reproducibility`, `scientific-ablation-controller`,
`safe-refactor-test-guardian`, `computer-vision-opencv` (all under
`.agents/skills/`; no version field is declared in their `SKILL.md` front
matter, so no version can be reported).

---

## 2. Phase A — repository state (all CODE_VERIFIED)

```
CURRENT_HEAD=5fa23de88455cae34c41d938c497d74233a7bc43
CURRENT_BRANCH=main
TRACKING_LAST_CHANGE_SHA=d925c9004e7aff5a3c8469b158d2428432c6031a
RF_ACC23_PROMOTION_SHA=d925c9004e7aff5a3c8469b158d2428432c6031a
TRACKING_WORKTREE_CLEAN=YES
```

The report's historical HEAD `5fa23de` **is** still current, but note that two
classification-v2 commits (`0ba2bfe`, `5fa23de`) landed after the balanced-model
merge; neither touches tracking.

### 2.1 Semantic equivalence — PASS, by tree identity

The last commit to touch `src/pig_behavior/tracking/` **is the RF_ACC23
promotion commit itself**. Nothing in tracking has changed since.

```
git rev-parse d925c90:src/pig_behavior/tracking = 8ba3d50d8322ab1c72b8f50b6ce4c9b1013f799a
git rev-parse HEAD:src/pig_behavior/tracking    = 8ba3d50d8322ab1c72b8f50b6ce4c9b1013f799a
git diff --stat d925c90 HEAD -- src/pig_behavior/tracking  ->  (empty)
```

`RF_ACC23_SEMANTIC_EQUIVALENCE=PASS`. This is the strongest available form of
the proof: identical Git tree objects, not merely equivalent behaviour.

### 2.2 What RF_ACC23 actually is

`d925c90` made exactly two semantic changes:

1. Added ten `realtime_core_*` keys to `REALTIME_FAST_CONFIG` — an
   **association tiebreak family** (unassigned tiebreak + pairwise tiebreak).
2. Repointed the `realtime` presentation profile from
   `realtime_quality_delayed` to `realtime_fast`.

"RF_ACC23" is therefore the `realtime_fast` profile *including* the core
tiebreak family. It is a pure association-ordering change; it adds no detector,
smoothing, or repair stage.

### 2.3 Exported effective semantic configuration

Resolved from code, not from documentation.
`SEMANTIC_CONFIG_SHA256=9bf4ce6d07423ab517b4705c716e3eb012349b756b7c0591cc3458eac207808d`

| Field | Value | Status |
|---|---|---|
| `det_conf` | `0.25` | CODE_VERIFIED |
| `detect_every_n_frames` | `2` | CODE_VERIFIED |
| `max_raw_detections` | `32` | CODE_VERIFIED |
| `realtime_lk_point_batching` | `True` | CODE_VERIFIED |
| `occlusion_aware_matching` | `False` | CODE_VERIFIED |
| `enable_offline_smoothing` | `False` | CODE_VERIFIED |
| `identity_swap_guard` / `smooth_boxes` / `refine_boxes` | `False` | CODE_VERIFIED |
| `USE_IOU_FALLBACK` / `USE_AREA_OCCLUSION_FREEZE` / `USE_CONDITIONAL_AREA_OCCLUSION_FREEZE` / `USE_MERGED_BOX_SPLIT` | `False` | CODE_VERIFIED |
| visible-competitor guards | `prefer=True`, `close_guard=True`, `margin=0.08`, `max_cost=0.40`, `min_center_x_ratio=0.67` | CODE_VERIFIED |
| unassigned tiebreak | `True`, `require_score_nondecrease=True`, `max_cost_delta=0.01`, `min_appearance_gain=0.01`, `min_detection_iou=0.30`, `max_selected_cost=0.40` | CODE_VERIFIED |
| pairwise tiebreak | `True`, `max_total_cost_increase=0.05`, `min_total_appearance_gain=0.10`, `min_detection_iou=0.30` | CODE_VERIFIED |
| **hidden reservation flags** | **absent from `realtime_fast`** | CODE_VERIFIED |
| motion-pair stabilizer / local pair-swap repair | absent from `realtime_fast` | CODE_VERIFIED |
| output timing contract | `causal_framewise`, delay `0` | CODE_VERIFIED |

The absence of `causal_hidden_detection_reservation` in `realtime_fast`
confirms the premise of H1: the reservation family exists only in
`REALTIME_BALANCED_CONFIG`, tuned for `det_conf=0.20`
(`min_iom=0.96`, `min_gain=0.17`, `max_alternative_cost=0.25`,
`allow_visible_hold=True`, `hold_min_gain=0.17`).

---

## 3. Phase B — verified-information matrix

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | RF_ACC23 metrics came from `d925c90` | **MISSING** | No metrics artifact for `d925c90` exists anywhere in the repo |
| 2 | Metrics used `detect_every_n_frames=2` | **CODE_VERIFIED** | `realtime.py:19` |
| 3 | `include_hidden=true` | **MISSING** | No run manifest exists to confirm the evaluation flag |
| 4 | Eval contract `iou0_area0_condarea0_merge0` | **CODE_VERIFIED (config)** / **MISSING (run)** | All four flags are `False` in `REALTIME_BASE_CONFIG`, which matches that contract name; no run manifest confirms it was the evaluation contract used |
| 5 | Hard6 = 000114, 000231, 000233, 000263, 000327, 000302 | **LINEAGE_REPORTED_UNVERIFIED** | All six resolve to real videos in the locked manifest, but no repository artifact defines this membership |
| 6 | IDSW=53, HOTA≈97.044%, IDF1≈97.077%, FP/FN=486/610, fragments=107, wrong-ID 8579→5219, Hard6 55→49 | **CONTRADICTED (partially)** | See §3.1 |
| 7 | 000302 had zero ID switches under RF_ACC23 | **MEMORY_VERIFIED_HISTORICAL for the parent, MISSING for RF_ACC23** | `06_BENCHMARK_NOTES.md:82` records `000302=0` for the 2026-07-20 far-right-guard build, which is RF_ACC23's parent, not RF_ACC23 |
| 8 | GT for 000216 is current and authoritative | **CONTRADICTED** | Memory flags an open `000216` margin-08 mechanism decision; GT file present and hashed but authority not established |
| 9 | Old runtime values belong to pre-RF_ACC23 code | **MEMORY_VERIFIED_HISTORICAL** | All recorded runtime figures (loop-FPS `19.08/25.48`, ratio `0.9588`) are dated 2026-07-20, three days before `d925c90` |
| 10 | A common GPU benchmark harness already exists | **CONTRADICTED** | `scripts/benchmarks/` contains no common harness; memory (`06:105`) states the speed claim "is not authorized until raw/Fast/Balanced use one comparable runtime harness" — i.e. it did not exist as of 2026-07-19 |

### 3.1 The metrics claim, examined precisely

The last `realtime_fast` full-13 evidence recorded anywhere in this repository
is `.agents/memory/06_BENCHMARK_NOTES.md:80-82` (2026-07-20):

> IDSW `59`, HOTA `95.63%`, IDF1 `95.37%`, FP/FN `486/610`, fragments `107`,
> gap-tolerant fragments `8`; no per-video IDSW regression and `000302=0`.

Against the claimed RF_ACC23 figures:

| Metric | Memory (2026-07-20 parent) | Claimed RF_ACC23 | Assessment |
|---|---|---|---|
| FP/FN | `486/610` | `486/610` | **identical** — coherent, since a pure association-tiebreak change cannot move detection counts |
| fragments | `107` | `107` | **identical** — same reasoning |
| IDSW | `59` | `53` | plausible direction, **unverified** |
| HOTA | `95.63%` | `97.044%` | **+1.41 pp — unverified and large** |
| IDF1 | `95.37%` | `97.077%` | **+1.71 pp — unverified and large** |

The FP/FN and fragment agreement is a point *in favour* of the claim's internal
consistency. The HOTA/IDF1 jump is the part that needs an artifact: for
calibration, the previous step in the same lane moved IDSW `69→59` (−10) for
HOTA `94.35%→95.63%` (+1.28 pp). The claim asks for a *larger* HOTA gain from a
*smaller* IDSW gain (−6). That is not impossible — HOTA's association term is
not linear in IDSW — but it is exactly the kind of value that must not be
promoted to fact without the run.

`wrong-ID matched-animal frames 8579→5219` and `Hard6 IDSW 55→49` appear
nowhere in the repository in any form.

### 3.2 RF_ACC23 was promoted without an evidence package

Every comparable tracking change in this project shipped with a decision JSON in
`docs/`. There are 90 such documents. The newest tracking decision document is
dated **20260720**. `d925c90` is dated **20260723** and has **no**
corresponding decision document, no memory entry, and no run root under
`outputs/`.

`d925c90` changed 4 files: `scripts/README.md`, `realtime.py`,
`tests/test_tracking_improvements.py`, `tests/test_tracking_profiles.py`. It
contains no evidence artifact.

---

## 4. Phase C — lineage lock (COMPLETED)

An immutable input manifest is locked. It asserts **no quality metric**.

```
Artifact : outputs/tracking/rf_acc23_lineage/rf_acc23_lineage_manifest_20260727_v1.json
MANIFEST_SHA256        = 0cfb26acc7766e05c497d9efdfafa40dc92f2d5c527e0338b89602eef0838dfc
SEMANTIC_CONFIG_SHA256 = 9bf4ce6d07423ab517b4705c716e3eb012349b756b7c0591cc3458eac207808d
```

Contents: 13 videos (path, size, SHA-256, container frame count, FPS,
resolution), 13 GT XMLs (path, SHA-256), detector weight SHA-256, resolved
semantic config, resolved timing contract, Git state, environment snapshot.

Verified facts from the lock:

- 13 videos, 13 ground-truth files, **zero** missing GT.
- Total container frame count **23,400** (13 × 1,800; 60 s at 30 FPS each).
- All six reported Hard6 members resolve to real videos.
- `Pigs291119_000263_30fps` uses the GT filename variant
  `Tracking_annotation_Pigs291119_000263_30fps.xml` — a real inconsistency that
  any reproduction script must handle; the locker resolves it explicitly.
- `Pigs291119_000216_30fps` is flagged `GT_AUTHORITY_FLAGGED_IN_MEMORY`.
- Population is recorded as `DEVELOPMENT_AND_EVALUATION_SET` with
  `is_final_unbiased_test_set: false`.

The tool refuses to overwrite an existing manifest.

---

## 5. Phase D — correctness gates (existing suite is GREEN)

`180 passed` across the seven existing tracking gate suites:

```
tests/test_tracking_profiles.py        tests/test_tracking_prefix_invariance.py
tests/test_tracking_repeatability.py   tests/test_tracking_telemetry.py
tests/test_tracking_no_mp4.py          tests/test_tracking_baseline_lock.py
tests/test_tracking_improvements.py
```

The timing contract was additionally resolved through real code
(`telemetry.resolve_output_timing_contract`) against the actual
`realtime_fast` config: `causal_framewise`, delay `0`, `is_causal_delay_zero:
true`.

Correctness-gate infrastructure for causality, prefix invariance,
repeatability, and no-MP4 **already exists** and passes. It did not need to be
rebuilt.

---

## 6. Phases E–G — BLOCKED, with measured cause

### 6.1 Environment regression: CUDA is unavailable

| Item | Value |
|---|---|
| GPU present | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, driver 610.62 |
| Installed torch | `2.12.1+cpu` |
| `torch.version.cuda` | `None` (CPU-only build) |
| `torch.cuda.is_available()` | `False` |

The hardware exists; the installed PyTorch cannot use it.

**Phase G is impossible as specified.** It requires CUDA Events, peak CUDA
memory, GPU utilisation, temperature, power, clocks, and P-state. None can be
measured without a CUDA-enabled build.

### 6.2 Measured CPU reproduction cost

Bounded probe on real data (YOLOv8 weight, `Pigs291119_000302_30fps`, warm-up
excluded, n=10):

```
CPU_DETECT_MEDIAN_S = 0.9650 s/frame
detections required for 13 videos at detect_every_n_frames=2 = 11,700
ESTIMATED_DETECT_ONLY = 3.14 hours
```

That is detector inference **only**. It excludes Lucas-Kanade optical flow on
all 23,400 frames, association, evaluation, and the mandatory exact-repeat run.
A single baseline plus repeat is realistically an order of magnitude beyond a
bounded session.

More importantly, a CPU-derived baseline would establish a **new lineage that is
not comparable** to any historical GPU-derived figure, so it could not settle
the §3.1 contradiction — which is the actual question.

### 6.3 Why H1 was not implemented

The promotion-quality gates require comparing the candidate against verified
RF_ACC23 values for IDSW, wrong-ID duration, FP/FN, HOTA/IDF1, and the
`000302` guardrail. **None of those baseline values is verified.** Implementing
H1 now would produce a candidate that cannot be accepted or rejected against any
gate, and the brief is explicit: *"Do not merge or promote a candidate merely
because tests pass."*

Per the stated failure behaviour, work stopped at the failing gate rather than
proceeding to gather more numbers.

---

## 7. Corrections to be applied to project memory

Only after the blockers below are cleared. Nothing in this document should be
promoted into memory as a metric.

1. Record that `d925c90` is both the RF_ACC23 promotion SHA **and** the last
   tracking change; tracking has been frozen since 2026-07-23.
2. Record `SEMANTIC_CONFIG_SHA256` and the lineage `MANIFEST_SHA256`.
3. Record that RF_ACC23 shipped **without** a decision artifact, unlike every
   prior tracking promotion.
4. Mark the claimed RF_ACC23 metrics as `UNVERIFIED_HISTORICAL_CLAIM`, not as
   superseding the 2026-07-20 `realtime_fast` entry.
5. Record the CUDA-unavailable environment regression as a live blocker on all
   tracking runtime work.
6. Record the `000263` GT filename variant and the `000216` GT authority flag.
7. Restate that the 13-video set is a development/evaluation set.

---

## 8. Smallest reversible next actions

In priority order:

1. **Restore a CUDA-enabled PyTorch** matching the RTX 3050 (4 GB) and driver
   610.62. This single action unblocks Phase C reproduction and all of Phase G.
2. **Produce or locate the RF_ACC23 evidence package.** If the author of
   `d925c90` holds the run root or `BAO_CAO_AUDIT_RF_ACC23_20260726.md`, import
   it and verify its hashes against the locked manifest. That is far cheaper
   than reproduction and is the only route that settles §3.1 on the original
   lineage.
3. Only if (2) fails: run a fresh GPU baseline under the locked manifest,
   labelled `FRESH_REPRODUCTION`, and explicitly supersede — never overwrite —
   the unverified claim.
4. Implement H1 only after a verified baseline exists.
