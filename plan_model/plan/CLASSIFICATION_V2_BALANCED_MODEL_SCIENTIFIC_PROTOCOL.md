# Classification V2 — BALANCED Causal Main Model: Scientific Study Protocol

**Rigorous, evidence-driven study on the current PROVISIONAL trusted labels. Not paper-grade.**

| Field | Value |
|---|---|
| GIT_SHA | `4fc94c2d3c9ebba5d6dc2ba6ad0aa5e7149e9c98` (`main`) |
| LABEL_AUTHORITY_STATUS | `PROVISIONAL_TRUSTED_PRE_BEHAVIOR_REVIEW` |
| PAPER_GRADE_USE | **NO** |
| CURRENT_CANONICAL_MANIFEST_DIRECTLY_VERIFIED | **NO** (run root `C:/pig_runs/…` unreachable this session) |
| FIRST_IMPLEMENTATION_TARGET | `BALANCED_CAUSAL_MAIN_MODEL` |
| Primary view | `T6_CONTIGUOUS` (provisional cross-source primary) |

> **Label discipline.** The current labels are carefully prepared and **provisionally trusted**, but they are **pre-Behavior-review** and are **not** final ground truth. They are allowed for architecture development, loader/tensor verification, baseline development, loss pilots, small controlled training, ablation screening, runtime/VRAM profiling, debugging, and hypothesis generation. They are **forbidden** for final paper headline results, final checkpoint promotion, definitive superiority claims, final calibration/deployment thresholds, final rare-class conclusions, and external comparison. Every provisional finding is classified for Behavior-review sensitivity (§11) and re-derived post-review (§12).

> **Authority separation.** Because the canonical manifest is not directly verifiable here, data status is reported in three separate registers — **user-reported current state**, **historical computed-output corroboration** (agent-audit, *not* promoted to canonical), and **provisional artifact evidence** — recorded in `provisional_label_snapshot_audit.json`. Historical `agent_audits/` outputs are never silently promoted to current canonical authority.

---

## 1. Scientific objectives

The study answers ten questions, not merely "which model is most accurate":

1. Which modality contributes useful **non-shortcut** evidence?
2. Which **custom** architectural modification contributes measurable gain?
3. Which **classes** benefit from each modification?
4. Are gains stable **across recording dates and sources**?
5. Are gains **preserved after Behavior review**?
6. Which **loss** handles imbalance best without sacrificing calibration or majority-class performance?
7. Does the model remain **causal, efficient and deployable**?
8. Are improvements larger than **seed and recording-date variation**?
9. Is model **complexity justified by effect size**?
10. Do the modules provide evidence **beyond a pretrained backbone used unchanged**?

---

## 2. Primary implementation target — BALANCED_CAUSAL_MAIN_MODEL

Implement **only** the balanced model first (not all candidates simultaneously). Components: shared **pretrained** visual backbone (standard, not claimed novel); grouped geometry–motion encoder; continuous ROI relation branch; ROI-conditioned modulation; actor–partner relation branch; short current-window temporal encoder; longer causal-history encoder; quality-aware multimodal gated fusion; availability/missing-modality masks; final 10-class behavior head; masked auxiliary heads where scientifically justified. **Scientific contribution comes from the customized architecture and its verified effects — not from claiming the backbone is novel.**

### 2.1 Exact exported schema dimensions (per-frame spatial groups)

From the historical spatial-sequence build (`model_upgrade_blueprint` P1 `array_shapes`; **to be re-verified on the reviewed export via the loader contract audit, §Phase 1**). These are the model-input group widths `D` (T padded to 16; primary view T6):

| Group | D | Contents (feature_semantics_v2 / motion_schema) |
|---|---:|---|
| `bbox_xywh_n` | 4 | normalized centre + size `cx_n, cy_n, bw_n, bh_n` |
| `bbox_shape_n` | 2 | area / aspect (shape) |
| `motion_delta` | 10 | window-local motion (schema names 12: vx,vy,bw_rate,bh_rate,area_rate,aspect_rate,speed,dir_change,tang_acc,ax,ay,acc_mag; NPZ export width 10) |
| `roi_class_relation` | 18 | 3 ROI classes (feeder/drinker/toy) × 6 relations (signed distance, overlap, IoU, near, contact, inside) |
| `social_relation` | 10 | nearest-partner geometry, density, contact |
| `quality_mask` | 6 | validity gate (mask, **not** a predictive feature) |
| **spatial predictive total** | **44** | (excludes quality_mask) |
| tabular whitelist | 102 | window-level families (used only where scientifically justified; not the primary numeric path) |

Actor image `[B,T,3,H,W]` (64 px cached; 128/160 target). `FORBIDDEN_FEATURES_PRESENT_IN_X = 0` by contract (§7).

### 2.2 Required custom modules (specification + hypotheses)

**1. Grouped Geometry–Motion Encoder.** Do **not** feed all numeric features into one undifferentiated MLP. Separate ≥5 groups: posture geometry (`bbox_xywh_n`, `bbox_shape_n`), motion dynamics (`motion_delta`), ROI relations (`roi_class_relation`), social relations (`social_relation`), availability/quality controls (`quality_mask` + availability masks). Per-group `Linear→LayerNorm→GELU` then project. Hypotheses: geometry improves **lying vs sitting**; motion improves **stand vs move vs explore**; motion history improves **fight vs ordinary movement**.

**2. ROI-Conditioned Modulation.** Use continuous ROI features (signed distance, overlap, inside ratio, approach/retreat velocity, ROI identity embedding, availability mask). Compare `ROI_NONE` / `ROI_CONCAT` / `ROI_FILM_OR_CONDITIONAL_MODULATION`. **No target-derived ROI features** (`target_roi_*`/`roi_target_*` forbidden). **No source-dependent binary `near_boundary` shortcut** as the main signal. Target classes: eat, drink, playwithtoy, stand/explore-near-ROI.

**3. Actor–Partner Relation Modeling.** Compare `SOCIAL_NONE` / `NEAREST_PARTNER_VECTOR` / `TOP_K_PARTNER_TOKENS` / `SMALL_GAT_OR_GCN`. Balanced default = **TOP_K_PARTNER_TOKENS** (K=3) with a small attention/relation encoder. Respect semantics: **fight labels all directly-involved pigs** (audit: 9,488 frames), **social-nose is actor-only** (4,050 frames). Edge features: relative geometry, relative velocity, overlap/contact, approach/retreat, partner confidence, partner availability mask. Add a full graph **only if** it beats the pairwise encoder by a meaningful margin under group-safe evaluation.

**4. Two-Timescale Causal Temporal Modeling.** `CURRENT_TARGET_WINDOW` + `CAUSAL_HISTORY_PRECEDING_THE_TARGET`, **no future frames**. Compare `CURRENT_WINDOW_ONLY` / `CURRENT_PLUS_CAUSAL_HISTORY` / `SINGLE_TEMPORAL_ENCODER` / `TWO_TIMESCALE_GATED_FUSION`. Hypotheses: current window sufficient for clear posture; history improves move/explore, eat/drink persistence, fight onset/approach. **Audit every tensor offset; require `FUTURE_FRAME_DEPENDENCE = 0`.**

**5. Quality-Aware Multimodal Fusion.** Compare `NAIVE_CONCATENATION` / `STATIC_LATE_FUSION` / `AVAILABILITY_GATED_FUSION` / `QUALITY_AWARE_GATED_FUSION`. The gate uses **only legitimate control signals**: modality availability, valid-frame ratio, motion validity, ROI availability, partner availability, visibility/occlusion quality, temporal-history availability. **Never** feed reviewer identity, review cohort/batch, source identity, behavior labels, or target-derived fields to the gate. Report per-modality gate statistics stratified by class, source, Hidden/occlusion, ROI availability, interaction-context availability.

---

## 3. Loss and class-imbalance study

Full spec in `imbalance_loss_experiments.yaml`. Imbalance is analysed at **separate units** (source-box / frame-object / temporal-interval / train-ready native-unit / model-window) and **never conflated**. The current temporal-interval imbalance is provisional **≈82:1** — **do not tune the loss on the 353:1 source-box ratio**. Class priors come from **train-fold native temporal units only**; never from validation/test; recompute per outer fold once train-ready data exists. Candidates **L0–L7**: CE reference; weighted CE (bounded strategies); effective-number CB (β∈{0.99,0.999,0.9999}, selected on inner val); **balanced-softmax / logit-adjustment (primary; τ∈{0.5,1.0,tuned-inner})**; focal (controlled, γ∈{0.5,1,2}); LDAM-DRW; decoupled cRT/τ-norm; controlled sampling. The winning loss is chosen by the **multi-criteria rule** (macro-F1 + balanced accuracy + rare recall + no common-class collapse + date-stability + calibration + gradient stability + reproducibility) — **a loss that wins from one unstable rare-class fold is disqualified.** One primary + one fallback selected. The mandatory per-loss ablation table (23 columns) is defined in the loss YAML.

---

## 4. Scientific metrics

Full contract in `scientific_metrics_contract.yaml`. **Primary = `MACRO_F1_SUPPORTED_AT_NATIVE_TEMPORAL_UNIT`** (overlapping windows collapsed to one prediction per native unit before scoring; window-level inflation is never the primary result). Secondary discrimination (macro recall, balanced accuracy, weighted F1, per-class P/R/F1, confusion matrix, one-vs-rest PR-AUC emphasised for rare classes, AUROC only with adequate support, always with support). Calibration/uncertainty (NLL, Brier, ECE, reliability, class-wise calibration, confidence-coverage, selective risk, abstention) — a model with good recall but unusable confidence is flagged. **Confusion-pair metrics** for the 8 specified pairs. **Efficiency metrics** measured after implementation (never projected-as-measured).

---

## 5. Split and leakage protocol

Full audit in `split_authority_audit.json`. **Primary outer split authority = `OUTER_DATE_GROUP_ID`** (calendar recording date; date-safe across sources). Evidence-derived structure:

- **13 calendar-date outer groups** (14 date-tokens; `101219a`/`101219b` share calendar date Dec-10 → bound together unless proven independent).
- **Cross-source dates `281119` & `291119`** carry both CVAT and legacy units → all units of that date stay in **one** outer fold (never split by source).
- **Severe date skew:** 3 dates hold **89%** of units (291119 44%, 301119 29%, 281119 16%); 11 legacy tokens hold 11%. Naive leave-one-date-out is unbalanced. **Recommended:** claim-grade leave-one-date-out over the 3 large dates + one pooled small-legacy-dates fold; per-date reporting with support and CI. **Effective outer generalization units ≈ 4 → low statistical power (stated as a limitation).**
- Inner validation by recording session / video / burst-group (678 groups) within outer-train.
- Emit machine-readable columns `outer_date_group_id`, `session_group_id`, `inner_recording_group_id`; validate split purity before training. No random-row/window split; no neighbouring intervals across train/test; no duplicate crop/window leakage; `object_track_key` purity; no source/path/id/review/target-ROI features in X; no future-frame dependence.

**Source-shortcut control (mandatory for every major result):** source-stratified + source-balanced results; train-one-source/test-other (281119/291119 enable within-date); feature-only and gate-weight source-decode probes; direct-source-identifier removal; date-safe split. **Any gain that disappears under source-balanced date-safe evaluation is not a valid contribution (hard gate A12).**

---

## 6. Experimental phases

- **Phase 0 — Authority & snapshot:** `provisional_label_snapshot_audit.json` (paths, hashes, Git SHA, label-status, distributions by unit, decision/temporal-unit counts, recording-date groups, unresolved state). `LABEL_AUTHORITY_STATUS=PROVISIONAL_TRUSTED_PRE_BEHAVIOR_REVIEW`, `PAPER_GRADE_USE=NO`.
- **Phase 1 — Loader & tensor-contract:** verify exact dims, mask semantics, feature ordering, forbidden-column exclusion, **causal temporal offsets**, current/history separation, missing-modality representation, stable batching, determinism; one-batch fwd/bwd locally.
- **Phase 2 — Baselines B0–B3** (standard CE first; no complex fusion before the ladder is stable).
- **Phase 3 — Imbalance-loss study on fixed B2/B3** (hold split/seed/backbone/aug/optimizer/schedule/batch/early-stop constant to isolate the loss). Select primary + fallback.
- **Phase 4 — ROI & temporal customization** (ROI concat vs FiLM; current-window vs two-timescale causal history).
- **Phase 5 — Social branch** (none / nearest / top-K / small GAT; interaction-ready subset analysis).
- **Phase 6 — Quality-aware fusion** (naive concat vs gated; require improvement/preservation on overall macro-F1, low-quality/occluded subset, missing-partner subset, source-balanced result, calibration).
- **Phase 7 — Full BALANCED model** (combine only modules that passed their gates; no module kept for conceptual appeal). Plus **Phase 7b cross-length study** (§10).
- **Phase 8 — Post-Behavior-review reproduction** (`behavior_review_reproduction_contract.yaml`): rebuild, compute label-change rates, re-run retained set, quantify drift; withdraw provisional conclusions if ranking changes materially.

---

## 7. Statistical reliability & practical significance

Full plan in `CLASSIFICATION_V2_STATISTICAL_ANALYSIS_PLAN.md`. Seeds: pilot 1 / screening ≥3 / claim-grade ≥5. Paired comparisons on the same outer folds. **Statistical unit = recording date (outer) / native temporal unit (within)** — never overlapping windows. Inference: recording-date cluster bootstrap; paired bootstrap of metric differences; date-level permutation where valid. Multiple testing: preregistered primary comparisons; Holm/FDR for secondary. Predefined minimum effects: `PRIMARY_MACRO_F1_MIN_GAIN=0.02`, `TARGET_CLASS_RECALL_MIN_GAIN=0.03`, `LIGHTWEIGHT_MAX_MACRO_F1_DROP=0.03` (adjustable **before** final test, never after, to favour a model). **A custom module is retained only when** its target hypothesis is supported, the gain is date-stable, its 95% CI is compatible with a practically meaningful benefit, source-balanced results do not collapse, runtime cost is justified, calibration is not unacceptably worsened, it remains causal, and its benefit survives post-review reproduction.

---

## 8. Architecture-claim requirements

**STANDARD_COMPONENTS_REUSED:** pretrained ResNet/MobileNet backbone; standard optimizer/normalization; standard causal-convolution primitives; standard masked cross-entropy / logit-adjustment. **PROJECT_SPECIFIC_ARCHITECTURAL_CHANGES:** grouped geometry–motion encoding; ROI-conditioned modulation; actor–partner relation modeling; two-timescale causal temporal reasoning; quality-aware missing-modality gated fusion; hierarchical auxiliary supervision where retained. Each change is reported with **hypothesis → measured effect → ablation evidence → limitations** (see `model_claim_ablation_map.csv`). Wording discipline: **"project-specific architectural adaptation" / "data-adapted multimodal design" / "novel integration, subject to literature verification" / "to the best of the systematic literature search"** — never "novel" merely because newly implemented in this repository. Every novelty statement traces to the bibliography (`literature_bibliography.csv`), with the ROI-context branch flagged as an explicit prior-art gap.

---

## 9. Compute policy

Local RTX 3050 4 GB: loader tests, tensor checks, one-batch fwd/bwd, tiny-overfit, reduced-resolution pilots, lightweight profiling. Rented 16–24 GB: loss study, baseline ladder, BALANCED ablations, initial multi-seed. Rented 40–80 GB: optional FULL causal model / offline teacher / final expensive multi-seed only when justified. **The scientific architecture is not constrained to 4 GB.** No large checkpoints or duplicate datasets are created in this task.

---

## 10. Temporal-view authority and cross-length validation

**Native source grains are distinct and not identical model windows:** `NATIVE_SOURCE_GRAINS = CVAT_6F_INTERVAL | LEGACY_16F_BURST`. Canonical view families (use these exact names in configs/manifests/tables/metrics; **the ambiguous name `6c` is not used**):

- **`T6_CONTIGUOUS`** — 6 consecutive source frames; **provisional primary cross-source view**.
- **`T8_CONTIGUOUS`**, **`T12_CONTIGUOUS`**, **`T16_CONTIGUOUS`** — 8/12/16 consecutive source frames.
- **`S6_AT_16_SPARSE`** — legacy-only sparse view; exact offsets `[0,3,6,9,12,15]` inside one native legacy 16-frame burst; five pair deltas `[3,3,3,3,3]`; **must use real elapsed seconds**; **must never be called contiguous T6** and **must never be the primary cross-source view**.
- **`HISTORICAL_C6_SCREEN`** — historical legacy-only development evidence; **not** equivalent to `T6_CONTIGUOUS` or `S6_AT_16_SPARSE`; its prior metrics **do not transfer** to the mixed reviewed lineage (`HISTORICAL_C6_METRICS_TRANSFERRED = NO`).

**FINAL-VIEW COMPUTATION RULE.** Views are built **only from reviewed frame-local primitives after Behavior decision apply.** For each view independently: (1) select exact frame indices; (2) record selected offsets + timestamps; (3) recompute motion pairs inside the view; (4) recompute ROI transitions inside the view; (5) recompute partner/social transitions inside the view; (6) recompute temporal aggregates; (7) recompute availability/quality masks; (8) assign one deterministic window ID; (9) validate label consistency across all constituent native units. **Do not** import pair/aggregate features from native-unit review evidence or another view, truncate a T16 aggregate to make T6/T8/T12, allow a pair whose first endpoint is outside the current view, or mix sparse and contiguous views in one primary training population. `PAIR_FEATURES_RECOMPUTED_PER_VIEW = YES`, `AGGREGATES_RECOMPUTED_PER_VIEW = YES`.

**CVAT multi-interval windows.** A CVAT T8/T12/T16 view may span >1 native 6-frame interval. Every constituent interval must be Behavior-reviewed, train-eligible, have a resolved final label, **share the same final behavior label**, belong to the same actor/object authority, and stay in the same split group. Reject the window on any mismatch → `CROSS_LABEL_WINDOWS = 0`.

**Cross-length study (Phase 7b).** Run **only after** the provisional winning loss is selected on a fixed `T6_CONTIGUOUS` baseline. Compare `T6/T8/T12/T16_CONTIGUOUS` holding constant the snapshot, folds, backbone, modality set, fusion, loss, optimizer, schedule, resolution, seeds, metric, and collapse rule. For each view report: eligible window count, native-unit coverage, class distribution, source distribution, physical-span distribution (timestamps present: `timestamp_start_sec`/`timestamp_end_sec`), effective observation rate, adjacent-pair coverage, missingness, macro-F1-supported, balanced accuracy, per-class F1/recall, calibration, confusion-pair metrics, latency, peak VRAM, parameter count, and **source-decode accuracy from view metadata**. **Do not** conclude a longer view is better from window-level accuracy — longer views may reduce eligible support, cross more label boundaries, and increase imbalance/latency/memory.

**Causal deployment.** All contiguous views are **trailing causal** views ending at the prediction time; no view contains future frames relative to its endpoint (`FUTURE_FRAME_DEPENDENCE = 0`). `S6_AT_16_SPARSE` is a legacy-only sparse ablation, not the causal cross-source deployment view unless its prediction-time semantics are separately proven.

**Decision rule.** `PRIMARY_CROSS_SOURCE_VIEW = T6_CONTIGUOUS` (provisional). Promote T8/T12/T16 only when macro-F1-supported improves by a practically meaningful margin, gains are date-stable, target temporal classes improve, source-balanced results remain valid, calibration is not materially degraded, reduced eligible support does not explain the gain, latency/VRAM remain acceptable, and no future-frame/cross-label leakage exists. Keep `S6_AT_16_SPARSE` as a legacy-only ablation.

---

## 11. Behavior-review sensitivity

Classify every provisional finding as `ROBUST_PRE_REVIEW_FINDING` / `LIKELY_STABLE_BUT_REQUIRES_CONFIRMATION` / `LABEL_SENSITIVE_FINDING` / `NOT_SUPPORTED`. Expected label-sensitive areas: stand/move/explore; fight/social-nose; ROI behavior vs stand/explore; playwithtoy; low-visibility interaction. The labels are trusted, not suspect — Behavior review is a final quality gate for residual ambiguity (`model_claim_ablation_map.csv` carries the per-claim class).

---

## Required final report

```
GIT_SHA=4fc94c2d3c9ebba5d6dc2ba6ad0aa5e7149e9c98
CURRENT_CANONICAL_MANIFEST_DIRECTLY_VERIFIED=NO
LABEL_AUTHORITY_STATUS=PROVISIONAL_TRUSTED_PRE_BEHAVIOR_REVIEW
PAPER_GRADE_USE=NO
PROVISIONAL_TEMPORAL_UNIT_COUNT=33355
FINAL_TRAIN_READY_UNIT_COUNT=NOT_READY
UNIQUE_OUTER_DATE_GROUPS=13 (calendar-date-safe; 14 date-tokens with 101219a/b merged; cross-source dates 281119/291119 bound to one fold)
SPLIT_PURITY=SPEC_READY (build-time audit required before training: forbidden-features=0, future-frame=0, no date/track/unit spanning folds)
FORBIDDEN_FEATURES_PRESENT_IN_X=0 (by contract; audited at build)
FUTURE_FRAME_DEPENDENCE=0 (required and audited)
BASELINE_MODELS_SPECIFIED=4 (B0,B1,B2,B3)
CUSTOM_MODULES_SPECIFIED=5 (grouped geometry-motion; ROI-conditioned modulation; actor-partner relation; two-timescale causal temporal; quality-aware gated fusion) [+ availability/missing-modality masks and masked auxiliary heads as supporting]
LOSS_FUNCTIONS_SPECIFIED=8 (L0-L7)
PRIMARY_LOSS_CANDIDATE=balanced_softmax / logit_adjustment (L3)
PRIMARY_LOSS_SELECTION_CRITERIA=multi-criteria (macro-F1-supported + balanced accuracy + meaningful rare recall + no common-class collapse + date-stable + acceptable calibration + gradient stability + reproducible; NOT one unstable tail fold)
FALLBACK_LOSS_CANDIDATE=effective_number_cb (L2, beta tuned inner) [secondary fallback: LDAM-DRW]
PRIMARY_METRIC=MACRO_F1_SUPPORTED_AT_NATIVE_TEMPORAL_UNIT
SECONDARY_METRICS=macro_recall, balanced_accuracy, weighted_f1, per_class_precision/recall/f1, confusion_matrix, one_vs_rest_PR_AUC(rare-emphasis), AUROC(adequate-support), confusion-pair metrics
CALIBRATION_METRICS=NLL, Brier, ECE, reliability_diagram, class_wise_calibration, confidence_coverage_curve, selective_risk, abstention
RUNTIME_METRICS=params, trainable_params, MACs/FLOPs, peak_train_VRAM, peak_infer_VRAM, preprocessing, feature-prep, backbone_latency, temporal/fusion_latency, model-only, end-to-end, per-pig-window, 8-pig throughput, isolated + competing-load (measured after implementation)
STATISTICAL_UNIT=recording date (outer) / native temporal unit (within); NOT overlapping windows
CONFIDENCE_INTERVAL_METHOD=recording-date cluster bootstrap + paired bootstrap of metric differences (date-level permutation where valid)
MULTIPLE_TESTING_POLICY=preregistered primary comparisons; Holm/FDR for secondary tests
MINIMUM_PRACTICAL_EFFECT=macro-F1 +0.02; target-class recall +0.03; lightweight max drop 0.03
FIRST_IMPLEMENTATION_TARGET=BALANCED_CAUSAL_MAIN_MODEL
PRIMARY_CUSTOM_MODEL_CHANGES=grouped geometry-motion encoder; ROI-conditioned modulation; actor-partner relation branch; two-timescale causal temporal fusion; quality-aware gated fusion (+ availability masks, masked aux heads)
QUALITY_AWARE_FUSION_ABLATION_READY=YES
TWO_TIMESCALE_CAUSAL_ABLATION_READY=YES
ROI_CONDITIONING_ABLATION_READY=YES
SOCIAL_RELATION_ABLATION_READY=YES
LOSS_ABLATION_READY=YES
PRE_POST_BEHAVIOR_REVIEW_REPRODUCTION_DEFINED=YES
TRAINING_ALLOWED_FOR_EXPLORATORY_PILOT=YES
TRAINING_ALLOWED_FOR_FINAL_PAPER=NO
NATIVE_SOURCE_GRAINS=CVAT_6F_INTERVAL|LEGACY_16F_BURST
TEMPORAL_VIEWS_BUILT=T6_CONTIGUOUS,T8_CONTIGUOUS,T12_CONTIGUOUS,T16_CONTIGUOUS,S6_AT_16_SPARSE (SPECIFIED; built only post-Behavior-review per FINAL-VIEW COMPUTATION RULE)
AMBIGUOUS_6C_NAME_USED=NO
PAIR_FEATURES_RECOMPUTED_PER_VIEW=YES
AGGREGATES_RECOMPUTED_PER_VIEW=YES
CROSS_LABEL_WINDOWS=0 (rejected by CVAT multi-interval single-label rule; audited at build)
PRIMARY_CROSS_SOURCE_VIEW=T6_CONTIGUOUS
LEGACY_ONLY_SPARSE_ABLATION=S6_AT_16_SPARSE
HISTORICAL_C6_METRICS_TRANSFERRED=NO
FINAL_VERDICT=SCIENTIFIC_EXPLORATORY_PROTOCOL_READY
```

## Deliverables in this package

`provisional_label_snapshot_audit.json`, `imbalance_loss_experiments.yaml`, `scientific_metrics_contract.yaml`, `split_authority_audit.json`, `balanced_model_scientific_experiment_matrix.csv`, `CLASSIFICATION_V2_STATISTICAL_ANALYSIS_PLAN.md`, `behavior_review_reproduction_contract.yaml`, `model_claim_ablation_map.csv`, `balanced_model_implementation_order.csv` — alongside the canonical research package (`CLASSIFICATION_V2_DEEP_MODEL_RESEARCH.md`, `pig_behavior_model_candidates.yaml`, `data_modality_audit.json`, `experiment_matrix.csv`, `lightweight_pareto_plan.csv`, `implementation_task_breakdown.csv`, `literature_bibliography.csv`).

*No production data, human decisions, contracts, source files, or official artifacts were modified. No large checkpoints or duplicate datasets were created. Parameter/MAC/VRAM/latency figures are analytical projections until measured on the implementation commit.*
