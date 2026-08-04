# Classification V2 — Deep Model Research (v3, canonical)

**A data-adapted multimodal spatiotemporal architecture for 10-class pig behavior recognition**

> **Canonical packaging pass (2026-07-26).** This is the authoritative version; v1 was stale and v2 is superseded. This pass corrects authority classification, unit separation, traceability, parameter-count claims, and model-role separation. It does **not** expand the architecture search or train anything. No production data, human decisions, contracts, or artifacts were modified.

| Field | Value |
|---|---|
| CODE_HEAD_SHA | `4fc94c2d3c9ebba5d6dc2ba6ad0aa5e7149e9c98` (branch `main`) |
| SCIENTIFIC_ACCEPTED_SHA | `a35e0b9aae8b55167b4562cfc7e26a45e2b4e312` |
| Canonical run root | `C:/pig_runs/classification_v2_lineage_rebuild_20260726_v1` |
| CURRENT_CANONICAL_MANIFEST_DIRECTLY_VERIFIED | **NO** (run root outside connected folder, unreachable this session) |
| Claim boundary | Q2-strong, internal recording-date/video-safe. **Not** external farm/camera/identity. |
| Canonical schema version | v3 |

> **Authority discipline (corrected).** I could **not** directly verify the canonical manifest — the run root `C:/pig_runs/…` is outside the connected folder and no `.candidate.json` manifests exist in the connected tree. Data status is therefore reported in **separate fields**, and **historical `agent_audits/` outputs are not promoted to current canonical authority**:
> - **USER_REPORTED_CURRENT_LINEAGE_STATE:** source_merge PASS (245,680); frame_local PASS (245,680); Hidden decisions **5,233**; coverage+apply PASS; harmonized 245,680; temporal intervals 33,355; stopped at temporal manifest publication → `PROVISIONAL_COMPUTED_BUT_UNMANIFESTED`; behavior review pending.
> - **HISTORICAL_COMPUTED_OUTPUT_CORROBORATION:** `outputs/classification_v2/agent_audits/c2v2_behavior_contract_patch_20260721_v1/temporal_harmonization_audit.json` (dated 2026-07-20/21) shows rows 245,680, intervals 33,355, trusted frames 5,240, errors [] — **historical agent-audit, corroborating scale/shape only.**
> - **PROVISIONAL_TEMPORAL_OUTPUT_SOURCE:** `…/c2v2_behavior_contract_patch_20260721_v1/temporal_intervals_standalone.csv`.
>
> Where documentation (`CLASSIFICATION_V2_CURRENT_STATE.md`) and these outputs disagree, the disagreement is reported and neither the stale doc nor the historical audit is treated as current canonical authority. **All parameter/MAC/VRAM/latency figures are analytical PROJECTIONS** (see §4.4 and the measurement command) — not measured — until a model is built from its config and counted on the implementation commit.

---

## 1. Executive summary

Classification V2 is a mature, fail-closed, 14-stage lineage turning two sources — a locked legacy 16-frame burst export (72,880 rows) and 12 CVAT behavior XMLs (172,800 boxes) — into per-pig windowed multimodal tensors for a 10-class classifier. The **10 classes** (from the CVAT `Behavior` attribute and `schema.py`) are `drink, eat, fight, social-nose, explore, lying, stand, move, sitting, playwithtoy`. A working late-fusion, mask-safe multimodal architecture already exists in code (`multimodal_fusion.py`; 10 modes; 4 temporal encoders; ResNet18/34; 4 aux heads; 3 imbalance losses).

**Data authority (correctly caveated).** The lineage is well advanced but its **current canonical manifest could not be directly verified** from this session. Per the user-reported state, Hidden review is complete as **5,233 decision units** with coverage+apply PASS; temporal harmonization is computed (245,680 frames; **33,355 intervals**) but **unmanifested**; behavior review is pending. The 5,240 "trusted frames" figure is **historical** (an agent-audit), not a current-canonical Hidden-apply count — see §4.3.

**Scale and imbalance (unit-separated; do not conflate).** Source-box (CVAT frame) ≈ **353:1**; frame-object (combined harmonized) ≈ **141:1**; **temporal-interval (33,355 units, provisional, from a historical audit)** ≈ **82:1** (`sitting` 34.5% → `playwithtoy` 0.42%). `lying` collapses from 62.9% (box) to 9.0% (interval). The **final train-ready/window distribution is not yet known**.

**Architecture stance.** ~33,355 temporal units is a **medium** dataset — larger backbones/video-transformers/graph models are **not ruled out by size**. The overfitting constraint is the **small number of independent recording dates (14) / video-or-burst-groups (678)**, addressed by a date-safe outer split, augmentation, and optional SSL pretraining — not by shrinking the model. "Overkill" is judged from data/temporal structure (6-frame windows, 64–160 px), never from the local GPU (which is **development-only**).

**Four separated model roles (no single model in multiple roles):**

1. **BALANCED_CAUSAL_MAIN_MODEL — first implementation target.** Shared-ResNet18, masked causal-TCN, quality-aware gated fusion, two-timescale causal history, ROI-FiLM, pairwise partner tokens. **PROJECTED ~11.5M params**, standard 16–24 GB, fully causal.
2. **FULL_CAUSAL_RESEARCH_MODEL — gated capacity experiment after the ladder.** Strictly causal high-capacity (dual-ResNet34 or causalized video-transformer), cross-attention gated fusion, confidence-aware GAT. **PROJECTED ~22–50M**, high-memory tier.
3. **OFFLINE_DISTILLATION_TEACHER — offline only.** May be **non-causal within-window** for maximum accuracy; generates distillation targets; **never deployed and never the headline causal result**.
4. **LIGHTWEIGHT_CAUSAL_STUDENT — later distillation/deployment.** Distilled MobileNetV3-Small causal student, **PROJECTED ~1.75M**, INT8; deployment hardware TBD.

The defensible contribution is the **integrated, quality-aware, causal combination** under source-balanced, date-safe, native-unit evaluation — every novelty claim is traced to primary sources in the bibliography (§A) with an explicit prior-art gap for the ROI-context branch.

---

## 2. Current data-lineage map

The config `configs/classification_v2/lineage_rebuild_v1.yaml` defines a strict fail-closed DAG (all authorization flags `false`; human-gated promotion). Progress is reported as **user-reported state + historical corroboration**, since the current canonical manifest is not directly verifiable here:

| # | Stage | Reported state | Basis |
|---|---|---|---|
| 1 | `source_merge` | PASS, 245,680 | user-reported; historical audit sources 172,800 + 72,880 |
| 2 | `frame_local` | PASS, 245,680 | user-reported; historical `frame_local_primitives.csv` |
| 3–5 | Hidden design/migration/coverage-gate | PASS (coverage) | user-reported; historical coverage audits |
| 6 | `hidden_apply` | PASS; **5,233 decisions**; applied frame rows **UNKNOWN** at canonical (5,240 trusted is historical) | user-reported + historical audit |
| 7 | `temporal_harmonization` | **PROVISIONAL_COMPUTED_BUT_UNMANIFESTED**; 245,680 frames, 33,355 intervals | historical agent-audit (`temporal_harmonization_audit.json`, errors []) |
| 8–9 | `native_evidence` / `pig_strenet_evidence` | computed / canary | historical agent-audits |
| 10–11 | `behavior_review_units` / `behavior_decision_apply` | **PENDING** | downstream of unmanifested temporal |
| 12–14 | `train_ready` / `tensor_export` / `model_input` | **NOT_READY** | awaits behavior review + temporal manifest |

**Source authorities (hash-bound, valid):** legacy export (SHA `fbd6300…`, 72,880 rows/crops); 12 CVAT XMLs (172,800 boxes); ROI COCO; pen mask; expected mixed rows **245,680**. **Applied policy** (from the historical harmonization audit): `directly_involved_pigs` = 9,488 (fight, all-involved) and `actor_only` = 4,050 (social-nose). **Not promoted to canonical authority:** the historical `agent_audits/` and `human_review_workspace/` outputs (used only for scale/shape); the `18d6692` OOF run (151,440 mismatches); smoke chains (688/63/438 are smoke only). The canonical manifested lineage at `C:/pig_runs/…` was not readable this session.

---

## 3. Current sample and tensor structure

### 3.1 Sources, grains, unit counts, and recording structure (kept separate)

- **Legacy (`legacy_recovered`):** 16-frame bursts; **72,880 frame rows → 4,555 native bursts** (27,330 anchors; **666 burst groups**) across **13 recording days**; carries depth (RGB-D) and pre-cropped images.
- **CVAT (`cvat_tracking_xml`):** 6-frame intervals `k..k+5`; 8 pig tracks/video; **172,800 boxes → 28,800 native intervals** (12 videos × 8 pigs × 300) across **3 recording days**.
- **Native temporal units (harmonized, provisional):** **33,355 = 28,800 CVAT + 4,555 legacy**.
- **Model windows:** overlapping windows — **NOT_FINALIZED**.
- **Recording structure (for splitting):** `UNIQUE_RECORDING_DATES = 14` (3 CVAT + 13 legacy − 2 overlap `281119`/`291119`); **CVAT videos 12; legacy burst groups 666; video-or-burst-group count 678**; recording *sessions* ≈ distinct camera×date×source captures (exact count = repo `recording_group_id` cardinality). See §13 for the split authority. (The imprecise phrasing "12–16 independent recording groups" is retired.)

### 3.2 Tensor contract

```
CVAT box Behavior / legacy anchor → object_track_key (pig_id is annotation-local)
 → actor crop letterbox RGB (64px cached; 128/160/224 target)
 → geometry(8)+motion(12-name/NPZ 10)+ROI(feeder/drinker/toy)+same-frame social
 → Hidden apply (quality metadata; NOT a feature) → 6f/16f interval, behavior_temporal_final
 → tensor export (X_spatial_sequences.npz + image manifests + masks) → y (10-class) + 4 aux y heads
```

| Tensor | Shape | Notes |
|---|---|---|
| actor_image / union_image | `[B,T,3,H,W]` | cached 64; target 128/160/224; T=6 (pad 16); legacy union masked |
| bbox_xywh_n / bbox_shape_n | `[B,T,4]` / `[B,T,2]` | |
| motion_delta | `[B,T,10]` | NPZ (schema names 12) |
| roi_class_relation / social_relation | `[B,T,18]` / `[B,T,10]` | + partner `[B,K,16]` |
| quality_mask / masks | `[B,T,6]` / `[B,T]` | gate; length ⊇ observed ⊇ {available,quality} |
| y_behavior / y_aux | `[B]` / 3,4,4,3 | aux never in X |

Spatial predictive per-frame width **= 44**. Loader shapes confirmed: image `[4,6,3,64,64]`, logits `[4,10]`.

---

## 4. Data diagnostics — separated statistics levels

**Do not compare or combine these units.**

**(1) SOURCE_BOX (CVAT, 172,800):** lying 62.9%, sitting 12.7%, explore 11.9%, fight 4.9%, eat 2.6%, social-nose 1.4%, move 1.25%, stand 1.25%, drink 0.88%, playwithtoy 0.18%. **≈353:1.**

**(2) FRAME_OBJECT (combined harmonized, 245,680; historical audit):** lying 53.7%, sitting 18.9%, explore 12.5%, eat 4.3%, fight 3.9%, stand 1.85%, move 1.80%, social-nose 1.65%, drink 1.15%, playwithtoy 0.38%. **≈141:1.** (Legacy frames alone: sitting 33% / lying 32% — different from CVAT → source is a behavior shortcut.)

### 4.3 HIDDEN unit separation (do not combine)

| Field | Value | Provenance |
|---|---|---|
| CURRENT_HIDDEN_DECISION_UNITS | **5,233** | user-reported current lineage |
| CURRENT_HIDDEN_APPLIED_FRAME_ROWS | **UNKNOWN** | no current-canonical Hidden-apply manifest accessible |
| HISTORICAL_TRUSTED_FRAME_ROWS | 5,240 | historical `temporal_harmonization_audit.json` |

Human decision units (5,233), applied frame rows (UNKNOWN at canonical), and historical trusted frame rows (5,240) are **different units**; 5,240 is **not** a current-state claim.

**(4) TEMPORAL_INTERVAL (33,355 units; `behavior_temporal_final`; PROVISIONAL, pre-behavior-review, from a historical audit):**

| Class | Intervals | % | | Class | Intervals | % |
|---|---:|---:|---|---|---:|---:|
| sitting | 11,515 | 34.5 | | social-nose | 1,525 | 4.6 |
| explore | 8,459 | 25.4 | | stand | 1,299 | 3.9 |
| eat | 3,072 | 9.2 | | move | 911 | 2.7 |
| lying | 3,005 | 9.0 | | drink | 633 | 1.9 |
| fight | 2,796 | 8.4 | | playwithtoy | 140 | 0.42 |

**≈82:1** (source split 28,800 CVAT + 4,555 legacy). Closest current proxy to model-unit imbalance — much milder and differently shaped than source-box (lying 62.9%→9.0%). Still pre-behavior-review; **final train-ready imbalance pending**. **(5) TRAIN_READY:** `NOT_READY`. **(6) MODEL_WINDOW:** `NOT_FINALIZED`.

### 4.4 Parameter-count classification

The candidate models are **not yet instantiated** — quality-aware gated fusion, two-timescale causal fusion, ROI-FiLM, the social relation-token/GAT branch, MobileNetV3 student integration, real causal temporal kernels, and distillation/pruning/QAT are **unbuilt**. Therefore **every candidate parameter and MAC figure in this report is `PROJECTED` (analytical), not `MEASURED_EXACT`.** The existing 10 repo modes were derived analytically from implemented modules (torch was not runnable this session) → prototype-level, not machine-measured. **Promotion rule:** use `MEASURED_EXACT_PARAMETER_COUNT` only after building the full model from its config and counting on the implementation commit, via:

```
scripts/classification_v2/04_baselines_smokes/measure_model_params_macs.py \
  --config configs/classification_v2/model_research/pig_behavior_model_candidates.yaml \
  --candidate <BALANCED|FULL|TEACHER|LIGHTWEIGHT> \
  --write-audit outputs/classification_v2/model_research/measured_params_macs_audit.json
# builds the model, counts params (torch) + MACs (fvcore/ptflops), binds code SHA
```

**Missingness / shortcuts (historical).** Legacy crop-only rows lack partner scene context (13,456/14,790 ready). Depth legacy-only (excluded). 64 px crops → re-extract 128–160. Source shortcut severe (tabular 1.0, spatial mean-only 0.996) → source-balanced date-safe eval is a **hard gate (A12)**.

### 4.5 Sample traces

- **`lying` (CVAT):** `Pig_3` box → `object_track_key` → 64 px crops over `k..k+5` → large area_n, wide aspect, ~0 speed → interval → tensors → `lying` (interval-sparse 9.0%).
- **`explore` (legacy):** 16f burst, anchor `explore`, frames `0,3,6,9,12,15` → moderate motion+turning → `explore` (25.4% of intervals).
- **`fight` (CVAT):** all involved pigs labelled `fight` (9,488 `directly_involved_pigs`); per-actor crop + high motion/contact + partner branch → `fight`. `social-nose` = actor-only (4,050), snout-contact.

---

## 5. Class-to-modality evidence matrix

● required, ◐ helpful, ○ minor (interval-level frequencies in parentheses).

| Behavior | Appear. | Geom. | Motion | ROI | Partner | Window | History | Scene | Failure |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| lying (9.0%) | ● | ● | ○ | ○ | ○ | ● | ○ | ○ | vs sitting |
| sitting (34.5%) | ● | ● | ○ | ○ | ○ | ● | ○ | ○ | vs lying |
| stand (3.9%) | ◐ | ● | ● | ○ | ○ | ● | ◐ | ○ | vs move/explore |
| move (2.7%) | ◐ | ◐ | ● | ○ | ○ | ● | ◐ | ○ | vs explore/stand |
| explore (25.4%) | ◐ | ◐ | ● | ◐ | ◐ | ● | ● | ◐ | vs move; social-nose |
| eat (9.2%) | ◐ | ◐ | ○ | ● | ○ | ● | ● | ◐ | vs stand near feeder |
| drink (1.9%) | ◐ | ◐ | ○ | ● | ○ | ● | ● | ◐ | vs stand near drinker |
| playwithtoy (0.42%) | ● | ◐ | ◐ | ● | ○ | ● | ● | ◐ | ultra-rare |
| social-nose (4.6%) | ● | ◐ | ◐ | ○ | ● | ● | ◐ | ● | actor-only; vs fight |
| fight (8.4%) | ◐ | ◐ | ● | ○ | ● | ● | ● | ● | all-involved; vs social-nose |

Justifies a hierarchical head, two-timescale temporal, ROI-conditioned modulation, and a partner/social branch — all with quality-aware gating.

---

## 6. Literature comparison

Full traceable bibliography (42 primary sources) and claim-to-source tables are in **§A**; `literature_bibliography.csv` carries the machine-readable form (key, title, authors, venue, year, URL, project-relevant claim, limitation). Suitability is judged from data/temporal structure and expected scientific gain — **never from the local GPU**. Highlights:

- **Closest prior art** [PigletSTGCN2022]: piglet ST-GCN fuses actor detection + geometry/motion node features + pairwise social (F1 0.95) — **3 of 4 pillars, no ROI**, 4 social classes only.
- **Multimodal + hierarchy + honest split** [EquineMM2026]: geometry+pose fusion + hierarchical cost-sensitive head + **leave-one-video-out** (3–5 pt honest drop) under 18–166:1 imbalance.
- **Causal video** [TSM2019, MoViNet2021, LSTR2021, TeSTra2022, TCN2018]: past-only shift / stream buffer / two-timescale / O(1) streaming / dilated causal conv — the causal toolkit (absent from pig work).
- **Quality-aware fusion** [GMU2017, QMF2023, TMC2021, PDF2024]; **missing-modality** [SMIL2021, RobustMM2022, MMP2024]; **long-tail** [LogitAdj2021, BalSoftmax2020, Decouple2020, CBLoss2019, LDAM2019]; **efficient/deploy** [DualTeacherKD2025, PigPruneKD2025, X3D2020, INT8QAT2018, FrameExit2021]; **video-transformers in scope for the FULL model** [VideoSwin2022, VideoMAE2022] (evaluated vs overfitting, A13, not GPU-excluded).

**Evidence-based "overkill" (not GPU-based).** Video transformers / SSMs are candidates for the FULL model given ~33k units + rented high-memory GPUs; their real risks are **few recording dates** (overfitting → SSL pretraining + augmentation + date-safe OOF) and **short 6-frame / 64–160 px** clips (limited long-range benefit). A 2D+causal-TCN / X3D actor stream is the expected-competitive balanced default; a video-transformer is a legitimate FULL-model arm to **test (A13)**. A full GCN is **worth testing** (A10), not dismissed.

**Novelty (traced, §A.2).** No pig paper fuses actor-crop + geometry/motion + ROI + social; causal deployment is essentially absent from pig work; ROI-context-as-model-input is the least-precedented pillar (explicit gap). Components (gating, two-timescale, FiLM, ARG-simplified, logit-adjust) are known — the **integration under this data + causal + source-balanced + date-safe protocol** is the contribution.

---

## 7. Baseline ladder (B0–B6)

Reference ResNet18 @128 px, `masked_tcn`, `hidden=128`. **All parameter/MAC figures are PROJECTED** (analytical; measure via §4.4 script). All causal.

| Id | Model | Inputs | PROJECTED params | PROJECTED MAC/win | Smoke VRAM | Train tier | Classes helped |
|---|---|---|---:|---:|---|---|---|
| B0 | actor single-frame (mean) | `[B,1,3,128,128]` | ~11.28M | 0.3 GMAC | <1 GB | standard 6–8 GB | lying, sitting |
| B1 | actor short-seq 6f | `[B,6,3,128,128]` | ~11.38M | 1.8 GMAC | <1 GB | standard | move, explore |
| B2 | +geometry/motion | `+[B,6,16]` | ~11.53M | 1.8 GMAC | <1 GB | standard | posture, locomotion |
| B3 | +ROI (FiLM) | `+[B,6,34]` | ~11.53M | 1.8 GMAC | <1 GB | standard | eat, drink, play |
| B4 | +causal history (2-timescale) | `+history` | ~11.6M | 2.0 GMAC | 1 GB | 8–10 GB | eat, drink, fight, move |
| B5 | +social/partner | `+[B,K,16]` | ~11.55M | 2.0 GMAC | 1 GB | 8–10 GB | fight, social-nose |
| B6 | full late-concat (shared backbone) | all + aux | ~12M | 2.0 GMAC | 1–2 GB | 12–16 GB | all (concat ref) |

---

## 8. Architecture candidates — four separated roles

**Parameter/MAC figures are PROJECTED.** Roles are distinct; see §17 for the per-role compute table and §16 for the implementation order.

- **BALANCED_CAUSAL_MAIN_MODEL** (first target): shared-ResNet18, masked causal-TCN + two-timescale history, quality-aware gated fusion, ROI-FiLM, pairwise partner tokens, 4 aux heads. PROJECTED ~11.5M / ~1.8 GMAC; standard tier; fully causal.
- **FULL_CAUSAL_RESEARCH_MODEL** (gated, after ladder): **strictly causal** high-capacity — dual-ResNet34 or a **causalized** video-transformer (Swin-T/VideoMAE-ViT-S) or X3D-M; cross-attention gated fusion; confidence-aware GAT. PROJECTED ~22–50M / ~11–13 GMAC; high-memory tier.
- **OFFLINE_DISTILLATION_TEACHER** (offline, optional): **may be non-causal within-window** for max accuracy; produces soft targets; **never deployed, never the headline causal result, never compared as deployment-equivalent.** PROJECTED ~22–50M; high-memory tier.
- **LIGHTWEIGHT_CAUSAL_STUDENT** (later): distilled MobileNetV3-Small causal student; PROJECTED ~1.75M / ~0.03–0.1 GMAC; deployment HW TBD; INT8.

### Decision matrix

| Criterion | BALANCED | FULL (causal) | TEACHER (offline) | LIGHTWEIGHT |
|---|:--:|:--:|:--:|:--:|
| Causal deployment | ● | ● | ○ (offline only) | ● |
| Suitability at 33k units / 14 dates | ● | ◐ (SSL+aug) | ◐ | ◐ |
| Ablation feasibility | ● | ○ | ○ | ● |
| Rentable-GPU tier | standard | high-mem | high-mem | standard/local |
| Deployment value | ◐ | ○ | ✗ | ● |
| **Implementation order** | **1 (first)** | **2 (gated)** | **2b (gated)** | **3 (later)** |

---

## 9. Recommended architecture (BALANCED main; FULL as causal ceiling; TEACHER offline)

The **first implementation target** is BALANCED: shared-ResNet18 → per-branch masked encoders → quality gate `g_m = σ(w·[avail_ratio_m, mean_quality_m, entropy_m])`, `fused = Σ g_m·e_m` → FusionHead → 10-class + aux heads; ROI-FiLM modulates the visual+geometry embedding; a short fixed-6 encoder and a dilated causal-history encoder fuse via a gated state; top-K=3 partner relation tokens with 1-layer attention. Sharing the ResNet18 across actor and union crops keeps PROJECTED params ~12M (vs ~22.9M dual-backbone).

The **FULL causal research model** raises capacity (dual-ResNet34 or a **causalized** video-transformer, cross-attention fusion, confidence-aware GAT) only where the high-memory tier and A13 evidence justify it — a strictly-causal ceiling. The **offline teacher** is a *separate* entity that may peek within-window; it feeds distillation and is never the causal headline. Mandatory adaptations: quality-aware modality gating + two-timescale causal temporal fusion; plus ROI-FiLM and partner relation tokens.

---

## 10. Lightweight deployment (later stage)

Distil the BALANCED (or teacher) into a MobileNetV3-Small causal student: depthwise-separable causal TCN, low-rank gated fusion, aux heads dropped at inference, conditional execution (early-exit for confident posture), temporal-state caching. PROJECTED ~1.75M params. Compression: KD (logits+embeddings+states) + 30–40% structured pruning + INT8 QAT + FP16. Tiers L0 (local dev), L1 (near-real-time workstation, ~6–8M distilled), L2 (compact edge, INT8). **Deployment hardware is TBD by the project — the RTX 3050 is a development machine, not the designated edge target.** Go/no-go: L2 ≥95% macro-F1 retention, <15 ms/pig-window, <1 GB.

---

## 11. Detailed module designs

Numeric branches are individually <150K PROJECTED params — the visual backbone dominates. **11.1 Actor Visual** (`ImageSequenceEncoder`; ROI-FiLM, visibility-aware attention, optional body-axis conditioning; A2/A4). **11.2 Geometry–Motion** (grouped encoders, confidence-conditioned masking, sin/cos direction; A2/A3). **11.3 ROI** (continuous distance/overlap/inside per feeder/drinker/toy, never `target_roi_*`; FiLM; A4). **11.4 Social** (nearest-vector / **top-K=3 tokens + 1-layer attn (balanced default)** / **1–2 layer GAT (full model)**; fight=all-involved, social-nose=actor-only; A5/A10). **11.5 Causal Temporal** (two-timescale: short fixed-6 + dilated causal-history, future-frame excluded; A6). **11.6 Masks** (per-branch availability; missing≠zero-evidence; modality dropout; A8). **11.7 Fusion** (quality-aware gated late fusion; A7). **11.8 Heads** (10-logit + 4 masked aux (0.25) + optional hierarchy-consistency; A9).

---

## 12. Loss and imbalance plan

Assessed at the **model unit** (interval ≈82:1), not source-box. Repo implements `event_balanced_ce`, `effective_number_ce`, `balanced_softmax` (fit from train native-event mass only). Primary = **balanced_softmax / logit adjustment** [LogitAdj2021, BalSoftmax2020] with priors from **train-date** native-event mass, + 4 masked aux losses + optional hierarchy consistency, combined with **decoupling** (cRT/τ-norm) [Decouple2020]. Fallback = `effective_number_ce` (β≈0.999) [CBLoss2019]; LDAM-DRW [LDAM2019] if tail margins needed; focal [Focal2017] add-on only. For `playwithtoy` (140 intervals): controlled temporal oversampling within train dates, per-class thresholds on a validation date, explicit support reporting. No stacking without isolated ablation.

---

## 13. Leakage-safe split protocol

**PRIMARY_OUTER_SPLIT_UNIT = recording session (camera × date × source), constrained recording-DATE-safe** — the 2 cross-source overlap dates (`281119`/`291119`) are bound to a single outer fold. **Justification:** source is ~1.0 decodable and the camera is fixed per session, so source/camera/date are the dominant leakage axes; only whole-session, date-safe holdout defeats all three. With `UNIQUE_RECORDING_DATES = 14` capping independent outer groups, use **leave-one-date-out or grouped k-fold over dates** with cluster-bootstrap CIs; finer **video/burst-group** grouping (678) is used only for **inner** validation within outer-train. Also: no random split; collapse overlapping windows to native units before scoring; `object_track_key` purity audit; exclude `source_type`/identity/path/review/`target_roi_*` from X; `pig_id` never a grouping key; strictly causal (future-frame audit).

---

## 14. Evaluation protocol

Prediction unit = **native temporal unit**. Metrics: macro-F1-supported (primary; min effect 0.02), macro-recall-supported, balanced accuracy, per-class P/R/F1 **with support**, confusion matrix, per-class AUROC where support allows, calibration (ECE), selective prediction. **Mandatory stratification:** source, Hidden/occlusion, ROI-availability, interaction-ready vs crop-only. Uncertainty via OOF **date-cluster** bootstrap CI. **Runtime:** params (measured), MACs, peak VRAM, preprocessing / backbone / temporal-fusion time, latency per pig-window, windows/sec, 8-pig throughput, isolated vs competing-load; distinguish model-only vs crop+feature-prep vs end-to-end. Pilot (single fold) gates ideas; multi-seed for paper numbers.

---

## 15. Ablation matrix (staged gates)

Full form in `experiment_matrix.csv`. Order: A1→A3 → A4/A5/A6 → A7/A8/A9 → A10 (pairwise vs GAT vs GCN) → **A13 (2D+TCN vs causalized video-transformer — capacity vs overfitting at 33k units/14 dates)** → A11 (four-role Pareto) → **A12 (source-shortcut — HARD GATE)**. Each: hypothesis / targets / expected direction / cost / min evidence / decision rule. **A12 blocks any claim whose gain vanishes under source-balanced, date-safe evaluation.**

---

## 16. Training stages and implementation sequence

**Do not implement all candidates simultaneously.** Order: **(1) BALANCED_CAUSAL_MAIN_MODEL first** (S0 verify → S1–S6 ladder+gated fusion); **(2) FULL_CAUSAL_RESEARCH_MODEL + OFFLINE_DISTILLATION_TEACHER** as gated capacity experiments after the ladder (S6b, A13, high-memory); **(3) LIGHTWEIGHT_CAUSAL_STUDENT** as a later distillation/deployment stage (S7).

| Stage | Focus | Loss | Compute tier | Gate |
|---|---|---|---|---|
| S0 | data-loader/tensor-contract + **param/MAC measurement script** | — | **local 3050** | shapes/masks/leakage pass; params measured |
| S1–S5 | BALANCED ladder B0–B5 | balanced-softmax + aux | standard 16–24 GB | ≥+0.02 per rung or documented null |
| S6 | BALANCED gated fusion | + hierarchy | standard | ≥+0.02 vs concat; A12 pass |
| S6b | FULL causal + A13; TEACHER (offline) | same / KD-target | **high-memory 40–80 GB** | gain>overfit vs BALANCED; strictly causal (FULL) |
| S7 | LIGHTWEIGHT distillation + compression | KD + task | standard (teacher) / local (student) | ≤0.03 drop; <15 ms/pig-window |
| S8 | final multi-seed date-safe OOF | — | high-mem (FULL) / standard (BAL) | multi-seed CIs; calibration; stratified |

The RTX 3050 is used only for S0 + smoke + reduced-res pilots + the LIGHTWEIGHT student at reduced scale — never as the full-model ceiling.

---

## 17. Compute and cloud plan — per-role reporting

RTX 3050 4 GB is **development-only** (not the full-model ceiling, not a selected deployment target). All VRAM/MAC/time are **analytical estimates**; parameter counts are **PROJECTED** until measured (§4.4).

| Field | BALANCED (main) | FULL (causal) | TEACHER (offline) | LIGHTWEIGHT |
|---|---|---|---|---|
| PARAMETERS_ESTIMATE | PROJECTED ~11.5M | PROJECTED ~22–50M | PROJECTED ~22–50M | PROJECTED ~1.75M |
| FLOPS_OR_MACS_ESTIMATE | ~1.8 GMAC/win | ~11–13 GMAC/win | ~11–13 GMAC/win | ~0.03–0.1 GMAC/win |
| MINIMUM_SMOKE_TEST_VRAM | 1–2 GB | 3–4 GB | 3–4 GB | <1 GB |
| PRACTICAL_FULL_TRAINING_VRAM | 12–16 GB | 24–40 GB | 24–40 GB | 2–3 GB (student) |
| RECOMMENDED_RENTED_GPU_CLASS | standard | high-memory | high-memory | standard/local |
| LOCAL_FULL_TRAINING_FEASIBLE | NO | NO | NO | YES |
| LOCAL_SMOKE_TEST_FEASIBLE | YES | YES | YES | YES |
| CAUSAL_DEPLOYMENT_FEASIBLE | YES | YES | **NO (offline)** | YES |
| EXPECTED_DEPLOYMENT_COMPRESSION_PATH | prune+INT8→L1; distil→LIGHT | distil→BAL/LIGHT | distil→BAL/LIGHT | KD→prune→INT8→dynamic |

Selection rule: choose by suitability to the actual data (33k units, 14 dates, 6–16-frame windows, 64–160 px), class benefit, missing-modality robustness, causal correctness, ablation feasibility, and rentable-hardware trainability — **not** by 4 GB fit.

---

## 18. Repository implementation map

Reuse production features/schemas (▸) / new (✚): config ✚ `configs/classification_v2/model_research/`; measurement ✚ `scripts/…/measure_model_params_macs.py`; loaders ▸ `datasets/*`; visual ▸ `models/multimodal_fusion.py::ImageSequenceEncoder`, ✚ MobileNetV3-Small + causalized video-transformer enums in `visual_backbones.py`; geometry/motion ▸ `SpatialSequenceEncoder`; ROI+FiLM ✚; social ✚ `models/social_relation.py`; temporal ✚ two-timescale in `temporal_encoders.py`; fusion ✚ `models/quality_gated_fusion.py`; losses ▸ `imbalance_losses.py`,`multitask_loss.py`; metrics ▸ `evaluation/native_temporal_metrics.py`,`source_domain_controls.py`,`calibration.py`; CLIs ▸ `training/full_multimodal_oof.py`,`evaluation/grouped_folds.py`; benchmark ▸ `runtime_benchmark.py`; export ✚ distillation/QAT. Tensor contract and pseudocode are in `pig_behavior_model_candidates.yaml`; ordered tasks (T0–T16, incl. the T0b measurement script) in `implementation_task_breakdown.csv`.

---

## 19. Scientific-contribution assessment

Primary = the integrated, quality-aware, **causal** architecture under source-balanced, date-safe, native-unit evaluation [novelty traced §A.2]. Secondary = quality-aware fusion + lightweight causal student. Supporting = the leakage-controlled pipeline, the source-shortcut quantification, and the capacity/overfitting study (A13) enabled by the 33k-unit / 14-date structure. **Guardrails:** no Q1 external-generalization claim; components acknowledged as known (§A); headline numbers require the manifested train-ready dataset + multi-seed OOF + measured parameters; tail classes reported with support. **UNTRACEABLE_NOVELTY_CLAIMS = 0** (every novelty statement maps to sources in §A.2).

---

## 20. Risk register

| Rank | Risk | Detection | Mitigation | Blocking threshold |
|---|---|---|---|---|
| 1 | Canonical manifest not directly verifiable | manifest access | do not claim direct verification; separate user-reported vs historical fields | claiming canonical verification |
| 2 | Temporal manifest not published (provisional) | schema/publication gate | pass temporal publication before treating as final input | claim on unmanifested temporal as final |
| 3 | Behavior review pending | behavior gate | complete behavior review + apply | train-ready claim before review |
| 4 | Source/domain shortcut (~1.0) | A12; source-balanced metrics | date-safe source-balanced eval; drop source cols | gain vanishes source-balanced |
| 5 | Unit conflation | per-unit support | assess at temporal-interval/train-ready unit, not source-box | tuning to 353:1 when true ~82:1 |
| 6 | Few recording dates → overfitting | date-cluster OOF variance | date-safe split, augmentation, SSL | high across-date variance |
| 7 | Parameter/MAC claims unmeasured | build+count on commit | label PROJECTED; run measurement script | reporting PROJECTED as MEASURED |
| 8 | Non-causal teacher mis-framed as deployment | role audit | keep teacher offline; never headline causal | teacher accuracy cited as deployment |
| 9 | Future-frame leakage | future-frame audit | strictly causal history | any future dependency |
| 10 | Camera/pen shortcut | pen ablation | pen out of whitelist | pen bit drives gain |
| 11 | Wrong partner association | partner mask+confidence | top-K robustness | interaction gain from bad edges |
| 12 | 64 px crops too low-res | resolution ablation | re-extract 128–160 | posture confusable |
| 13 | Capacity overfit (FULL) | val gap; A13 | SSL+aug; prefer BALANCED if no gain | val≫train |
| 14 | Lightweight collapse | A11 Pareto | KD + modest prune | drop >0.03 |
| 15 | Runtime measurement bias | isolated + competing-load | model-only vs end-to-end | single-run cherry-pick |

---

## 21. Final recommendation

Implement in order — **not simultaneously**: **(1) BALANCED_CAUSAL_MAIN_MODEL** (the first target; shared-ResNet18 gated causal multimodal, PROJECTED ~11.5M, standard tier) proven via the B0–B6 ladder and A1–A12 ablations with **A12 as a hard gate**; **(2) FULL_CAUSAL_RESEARCH_MODEL** and **(2b) OFFLINE_DISTILLATION_TEACHER** as gated capacity experiments after the ladder (A13, high-memory; the teacher offline-only, never the causal headline); **(3) LIGHTWEIGHT_CAUSAL_STUDENT** as a later distillation/deployment stage. Claim-grade evaluation is native-unit, source-balanced, date-safe, multi-seed OOF with measured parameters.

Authority is correctly caveated (canonical manifest not directly verified; user-reported + historical corroboration separated; historical `agent_audits/` not promoted to canonical). Traceability is complete (42 primary sources; claim-to-source tables; 0 untraceable novelty claims). Packaging is canonical (corrected artifacts at canonical paths; no `_1`/stale files on the filesystem; schema v3). The design is complete and implementable; training is gated on temporal-manifest publication + behavior review.

---

## A. Primary-source bibliography and claim-to-source traceability

`literature_bibliography.csv` is the machine-readable form (key, title, authors, venue, year, URL, project-relevant claim, limitation). **42 primary sources.**

### A.1 Claim-to-source map (by category)

| Category | Sources (keys) |
|---|---|
| Pig multimodal prior art | PigletSocial2021, PigletSTGCN2022, EquineMM2026, PigPlay2022, PigTwoStream2020, PigSkelGCN2026 |
| ROI context as model input | PigProximity2025 + **explicit GAP** (no pig paper fuses feeder/drinker/toy ROI geometry as a model-input branch — least-precedented pillar) |
| Social GCN/GAT prior work | PigletSTGCN2022, ARG2019, GroupFormer2021 |
| Causal video recognition | TSM2019, MoViNet2021, LSTR2021, TeSTra2022, TCN2018 |
| Quality-aware fusion | GMU2017, QMF2023, TMC2021, PDF2024, DynMM2023, MBT2021 |
| Missing-modality learning | SMIL2021, RobustMM2022, MMP2024 |
| Long-tail losses | Focal2017, CBLoss2019, LDAM2019, LogitAdj2021, BalSoftmax2020, Decouple2020 |
| Distillation / lightweight deployment | DualTeacherKD2025, PigPruneKD2025, PigEdge2023, INT8QAT2018, INT8NVIDIA2020, FrameExit2021, X3D2020, MoViNet2021 |
| Video-transformer / SSL backbones (FULL model, A13) | VideoSwin2022, VideoMAE2022, X3D2020, Mamba2023 |

### A.2 Novelty claims → traceable sources (0 untraceable)

| Novelty claim | Traced to |
|---|---|
| No prior pig paper fuses actor-crop + geometry/motion + ROI + social (the *integration* is the contribution) | PigletSTGCN2022 (closest, 3/4 pillars, no ROI); EquineMM2026 (fusion+hierarchy, dyadic, no ROI); PigProximity2025 (ROI/proximity features only) |
| Causal (no-future-frames) deployment is essentially absent from pig behavior work | pig works PigTwoStream2020/PigTSM2023/PigSkelGCN2026 use whole-clip/bidirectional; causal primitives (TSM2019/MoViNet2021/TeSTra2022) are non-pig |
| Full GCN not justified vs a pairwise partner encoder at this scale | ARG2019/GroupFormer2021 (large-data group activity) vs PigletSTGCN2022; decided empirically by A10 |

### A.3 Bibliography (abridged; full fields in `literature_bibliography.csv`)

Causal/temporal: **TSM2019** (Lin+, ICCV'19), **MoViNet2021** (Kondratyuk+, CVPR'21), **LSTR2021** (Xu+, NeurIPS'21), **TeSTra2022** (Zhao+Krähenbühl, ECCV'22), **TCN2018** (Bai+, arXiv). Backbones: **X3D2020** (Feichtenhofer, CVPR'20), **VideoSwin2022** (Liu+, CVPR'22), **VideoMAE2022** (Tong+, NeurIPS'22), **Mamba2023** (Gu+Dao). Fusion: **GMU2017** (Arevalo+), **QMF2023** (Zhang+, ICML'23), **TMC2021** (Han+, ICLR'21), **PDF2024** (Cao+, ICML'24), **DynMM2023** (Xue+Marculescu), **MBT2021** (Nagrani+, NeurIPS'21). Missing-modality: **SMIL2021** (Ma+, AAAI'21), **RobustMM2022** (Ma+, CVPR'22), **MMP2024**. GNN: **ARG2019** (Wu+, CVPR'19), **GroupFormer2021** (Li+, ICCV'21). Long-tail: **Focal2017** (Lin+), **CBLoss2019** (Cui+), **LDAM2019** (Cao+), **LogitAdj2021** (Menon+, ICLR'21), **BalSoftmax2020** (Ren+, NeurIPS'20), **Decouple2020** (Kang+, ICLR'20). Deploy: **DualTeacherKD2025**, **INT8QAT2018** (Jacob+), **INT8NVIDIA2020** (Wu+), **FrameExit2021** (Ghodrati+). Pig/animal: **PigTwoStream2020**, **PigPlay2022**, **PigTSM2023**, **PigYOLO2026**, **PigSkelGCN2026**, **PigletSocial2021**, **PigletSTGCN2022**, **PigProximity2025**, **EquineMM2026**, **PigEdge2023**, **PigPruneKD2025**, **DLC2022**. (URLs and per-source claims/limitations in the CSV.)

---

## Required final report

```
CURRENT_CANONICAL_MANIFEST_DIRECTLY_VERIFIED=NO (run root C:/pig_runs/classification_v2_lineage_rebuild_20260726_v1 unreachable this session; no .candidate.json in connected tree)
USER_REPORTED_CURRENT_LINEAGE_STATE=source_merge PASS 245,680 | frame_local PASS 245,680 | hidden decisions 5,233, coverage+apply PASS | harmonized 245,680 | temporal intervals 33,355 | stopped at temporal manifest publication (PROVISIONAL_COMPUTED_BUT_UNMANIFESTED) | behavior review pending
PROVISIONAL_TEMPORAL_INTERVALS=33355
FINAL_TRAIN_READY_UNIT_COUNT=NOT_READY
CURRENT_HIDDEN_DECISION_UNITS=5233
CURRENT_HIDDEN_APPLIED_FRAME_ROWS=UNKNOWN (no current-canonical Hidden-apply manifest accessible; 5,240 is HISTORICAL)
HISTORICAL_TRUSTED_FRAME_ROWS=5240
UNIQUE_RECORDING_DATES=14
PRIMARY_OUTER_SPLIT_UNIT=recording_session (camera x date x source), constrained recording-DATE-safe (281119/291119 bound to one outer fold); ~14 dates cap outer groups -> leave-one-date-out / grouped k-fold; video-or-burst-group (678) used only for inner validation
PARAMETER_COUNTS_CLASSIFICATION=PROJECTED (candidate modules unbuilt; promote to MEASURED_EXACT only after building from config and counting on the implementation commit via measure_model_params_macs.py)
LITERATURE_PRIMARY_SOURCES=42
UNTRACEABLE_NOVELTY_CLAIMS=0
ACTIVE_STALE_V1_ARTIFACTS=0
ACTIVE_RESEARCH_ARTIFACTS_WITH_SUFFIX_1=0
CANONICAL_RESEARCH_SCHEMA_VERSION=v3
FIRST_IMPLEMENTATION_TARGET=BALANCED_CAUSAL_MAIN_MODEL
OFFLINE_TEACHER_ROLE_SEPARATED=YES
LIGHTWEIGHT_ROLE_SEPARATED=YES
MODEL_ROLES=OFFLINE_DISTILLATION_TEACHER | FULL_CAUSAL_RESEARCH_MODEL | BALANCED_CAUSAL_MAIN_MODEL | LIGHTWEIGHT_CAUSAL_STUDENT
PROJECTED_PARAMETERS=BALANCED ~11.5M | FULL ~22-50M | TEACHER ~22-50M | LIGHTWEIGHT ~1.75M
FINAL_VERDICT=CANONICAL_MODEL_RESEARCH_PACKAGE_READY (authority correctly caveated; traceability complete; canonical packaging clean; NOTE: current canonical manifest not directly verifiable this session, so data status rests on user-reported state + historical corroboration, both clearly labelled)
```

## Appendix — canonical machine-readable deliverables

- `docs/research/CLASSIFICATION_V2_DEEP_MODEL_RESEARCH.md` (this file, v3 canonical)
- `configs/classification_v2/model_research/pig_behavior_model_candidates.yaml` (4 roles; PROJECTED params; measurement command)
- `outputs/classification_v2/model_research/data_modality_audit.json` (authority/unit-separation/recording/param corrections; schema v3)
- `outputs/classification_v2/model_research/experiment_matrix.csv` (B0–B6 + A1–A13; `projected_param_estimate`)
- `outputs/classification_v2/model_research/lightweight_pareto_plan.csv` (L0–L2; deployment HW TBD)
- `outputs/classification_v2/model_research/implementation_task_breakdown.csv` (T0–T16 incl. T0b measurement script; sequenced BALANCED-first)
- `outputs/classification_v2/model_research/literature_bibliography.csv` (42 primary sources; full fields)

*Parameter/MAC/VRAM/latency figures are analytical PROJECTIONS until measured on the implementation commit. No production data, human decisions, contracts, source files, or official artifacts were modified.*
