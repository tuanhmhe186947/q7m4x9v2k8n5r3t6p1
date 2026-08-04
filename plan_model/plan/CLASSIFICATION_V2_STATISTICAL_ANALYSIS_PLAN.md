# Classification V2 — Statistical Analysis Plan (BALANCED model study)

**Pre-registered analysis plan. Labels are PROVISIONAL_TRUSTED_PRE_BEHAVIOR_REVIEW; PAPER_GRADE_USE=NO.**

Companion to `CLASSIFICATION_V2_BALANCED_MODEL_SCIENTIFIC_PROTOCOL.md`. This plan is fixed **before** looking at final test results; thresholds may be adjusted before the final test but never afterward to favour a model.

## 1. Inferential units (the central correctness point)

Thousands of overlapping model windows are **not** independent observations. The effective inferential unit depends on the question:

- **Generalization across conditions → recording date.** Outer folds are whole calendar dates (`OUTER_DATE_GROUP_ID`, 13 groups). Reported uncertainty is over dates.
- **Per-sample discrimination → native temporal unit.** Overlapping windows are collapsed to one prediction per native unit before scoring.
- **Never** treat a window as an independent observation for a confidence interval.

**Power caveat (explicit).** The date structure is severely skewed: 3 dates hold 89% of units (291119 44%, 301119 29%, 281119 16%); the other 11 legacy date-tokens hold 11%. The **effective number of independent outer generalization units is ~4** (three large dates + one pooled small-legacy fold). Date-level confidence intervals will therefore be **wide**, and date-level tests have **low power**. This is a real limitation of the current data and is reported alongside every date-clustered result; it is not hidden by switching to window-level counts.

## 2. Seeds and repetition

- Pilot: 1 seed, 1 inner split (feasibility only).
- Screening: ≥3 seeds.
- Claim-grade (exploratory): ≥5 seeds where compute allows.

Report mean, standard deviation, median, 95% CI, per-date fold results, worst-date result, and the number of failed runs. Models share the same outer folds → **paired** comparisons throughout.

## 3. Estimation and inference

- **Primary uncertainty:** recording-date **cluster bootstrap** of `MACRO_F1_SUPPORTED_AT_NATIVE_TEMPORAL_UNIT` (resample dates with replacement; recompute native-unit-collapsed metric).
- **Model comparison:** **paired bootstrap of the metric difference** on the same dates/folds/seeds; report the difference distribution and its 95% CI, not just two separate CIs.
- **Optional permutation:** date-level permutation test of the paired difference where exchangeability holds.
- **Seed variance** is reported separately from date variance; a gain must exceed **both** seed and date variation (objective 8).

## 4. Multiple testing

- **Preregistered primary comparisons** (one per custom module and the loss selection) are tested first, uncorrected, at the predefined effect thresholds.
- **Secondary / exploratory** comparisons (per-class, per-confusion-pair, per-strata, cross-length family) use **Holm correction or FDR control** within each family.
- The ablation families (loss L0–L7; cross-length T6–T16; social variants) are declared families in advance; no uncorrected fishing across them.

## 5. Effect sizes and decision gates

Predefined minimum practical effects (from the metrics contract): macro-F1 **+0.02**; target-class recall **+0.03**; lightweight max drop **0.03**. A tiny statistically significant difference is **not** a victory. A custom module is retained only if (1) its target hypothesis is supported, (2) the gain is date-stable, (3) the 95% CI is compatible with a practically meaningful benefit, (4) source-balanced results do not collapse (hard gate A12), (5) runtime cost is justified, (6) calibration is not unacceptably worsened, (7) it remains causal, and (8) the benefit survives post-Behavior-review reproduction. The **loss** winner is chosen by the multi-criteria rule and is disqualified if its macro-F1 comes from one unstable rare-class fold.

## 6. Source-shortcut adjudication

Because source is historically ~perfectly decodable, every headline comparison is repeated **source-balanced and date-safe**, with feature-only and gate-weight source-decode probes and (where 281119/291119 permit) train-one-source/test-other. **Any gain that vanishes under source-balanced date-safe evaluation is reported as not a valid scientific contribution.**

## 7. Calibration analysis

Report NLL, Brier, ECE, reliability diagrams, class-wise calibration (where support allows), confidence-coverage, selective risk, and abstention performance. A model improving rare-class recall while producing unusable confidence scores is flagged explicitly and not recommended for any thresholded use.

## 8. Behavior-review sensitivity and reproduction

All conclusions are provisional. Each is tagged `ROBUST_PRE_REVIEW_FINDING` / `LIKELY_STABLE_BUT_REQUIRES_CONFIRMATION` / `LABEL_SENSITIVE_FINDING` / `NOT_SUPPORTED`. After Behavior review, the reproduction contract re-runs the retained set on the same folds/seeds and quantifies drift; **if model ranking changes materially, provisional conclusions are withdrawn.** Nothing here is promoted to a paper-grade claim pre-review.

## 9. Reporting template (per comparison)

| Field | Content |
|---|---|
| comparison | model A vs B (paired, same folds/seeds) |
| primary effect | Δmacro-F1-supported (native unit), mean + 95% paired-bootstrap CI |
| per-date | Δmacro-F1 per outer date + worst-date |
| target classes | Δper-class recall/F1 + support |
| source control | source-balanced Δ + source-decode probe result |
| calibration | ΔNLL/Brier/ECE |
| stability | seed SD, failed runs, gradient notes |
| runtime | measured params/MACs/latency/VRAM |
| effect vs threshold | pass/fail vs predefined minimum effect |
| review-sensitivity | one of the four classes |
| decision | retain / drop / inconclusive |

*Statistical unit = recording date (outer) / native temporal unit (within). Labels are provisional; results are exploratory; no paper-grade claim is made pre-Behavior-review.*
