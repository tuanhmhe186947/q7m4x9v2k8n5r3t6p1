# H1-r2 Development Missingness Audit — 2026-07-27

## Locked conclusion

- `H1_R2_FROZEN_DEVELOPMENT_DECISION=FAIL_NO_ACTIVATION`
- `H1_R2_OWNER_PREFERENCE_APPLIED=0`
- `H1_R2_ASSOCIATION_OUTPUT_DIVERGENCES=0`
- `H1_R2_QUALITY_IMPROVEMENT_PROVEN=NO`
- `H1_R2_VALIDATION_AUTHORIZED=NO`

Equal baseline and candidate metrics mean no effect. Zero new permanent and
terminal swaps were observed on the no-op execution path. Safety under real
activation is `NOT_MEASURED`.

This audit does not reopen the frozen development decision. It did not run
tracking, detector inference, GPU inference, validation, or runtime evaluation.
It did not change the score, threshold, coefficients, production code, or
profiles.

## Evidence integrity

The source is the immutable candidate-pair export:

`outputs/tracking/h1_r2_development_quality_20260727_121643/`
`H1_R2_DEVELOPMENT_ACTIVATION_TRACE.csv`

The row-level integrity audit passed:

| Check | Result |
|---|---:|
| Source trace SHA-256 | `2538fe509e01e3bb50727cee2d61c2eaf239d18e2d221b66678e45d074cde49e` |
| Locked evidence inventory SHA-256 | `d47a29518bacc107907b868dae55478f9b9d405b0be7836515355dcfb1b591cd` |
| Total rows | 770 |
| `missing_evidence` abstentions | 737 |
| Valid score pairs | 33 |
| Applied | 0 |
| Non-development rows | 0 |

The six allowed episode IDs were checked against the frozen development
population. No validation row was loaded or copied.

## Main finding: the features were not missing

All eight normalized feature values were valid on both sides for all 770 rows.
Both appearance and motion masks were also `1` on both sides for all rows.
There is one observed feature-validity combination:

`hidden=11111111, visible=11111111`

The dominant counter label is therefore misleading. Exactly 737 rows had a
present, finite hidden `overlap_similarity` below the hidden-only eligibility
floor of `0.50`. Those same 737 rows are the complete `missing_evidence`
population:

| Condition | Count |
|---|---:|
| Hidden overlap below `0.50` | 737 |
| Hidden overlap at least `0.50` | 33 |
| Missing appearance on either side | 0 |
| Missing motion on either side | 0 |
| Detection confidence below `0.25` | 0 |
| Multiple observed blockers among the frozen pre-score gates | 0 |

The code intentionally returns `reason="missing_evidence"` for low detection
confidence, stale state, low hidden overlap, or absent optional evidence. The
telemetry reason does not distinguish unavailable data from a present value
that fails a semantic eligibility floor.

## Missingness decomposition

| Feature | Hidden valid | Visible valid | Both valid | Alone blocks |
|---|---:|---:|---:|---:|
| overlap similarity | 770 | 770 | 770 | 737 |
| normalized center similarity | 770 | 770 | 770 | 0 |
| scale similarity | 770 | 770 | 770 | 0 |
| appearance similarity | 770 | 770 | 770 | 0 |
| motion consistency | 770 | 770 | 770 | 0 |
| track freshness | 770 | 770 | 770 | 0 |
| appearance available | 770 | 770 | 770 | 0 |
| motion available | 770 | 770 | 770 | 0 |

For every feature, hidden-only, visible-only, and both-side missing counts are
zero. The detailed CSV also reports one-of-multiple-blocker counts, all zero.

Episode totals:

| Episode | Role | Rows | Low overlap | Valid score pairs |
|---|---|---:|---:|---:|
| E01 | positive | 203 | 203 | 0 |
| E02 | positive | 367 | 344 | 23 |
| E03 | positive | 7 | 7 | 0 |
| E04 | positive | 91 | 81 | 10 |
| E05 | control | 0 | 0 | 0 |
| E06 | control | 102 | 102 | 0 |

All 770 pairs were constructed on detector frames. Skipped detector frames
produced zero pairs by design. With `detect_every_n_frames=2`, skipped frames
causally update track state using LK when available and predicted-box fallback
otherwise. The export records the H1-r2 motion-validity mask, not direct LK
success versus fallback; exact LK provenance is therefore unresolved. Both
motion masks were nevertheless valid for all 770 pairs.

Hidden age is reconstructed exactly in detector opportunities as
`round((1-track_freshness)*5)`. The raw-frame occlusion duration is only a
two-frame cadence proxy and is labeled as such in the aggregate CSV.

## Feature provenance

### Overlap similarity

`FixedTrack` state and `Detection.box` flow through
`association_reference_box`, `_h1_r2_features_for_track`,
`build_owner_preference_features`, and `overlap_similarity`. The result is
clipped IoU in `[0,1]`. Hidden and visible use the same formula, but only the
hidden side receives the absolute `0.50` eligibility floor.

Classification: `CONTRACT_TOO_RESTRICTIVE`.

### Normalized center similarity

The center distance is divided by the sum of reference and detection box
diagonals, then converted to similarity in `[0,1]`. Both sides use the same
formula, box validity rule, and causal state.

Classification: `CONTRACT_CORRECT_MISSING` with zero missing rows.

### Scale similarity

The symmetric absolute log-area ratio is divided by `log(4)` and converted to
a clipped similarity. Both sides use identical state and normalization.

Classification: `CONTRACT_CORRECT_MISSING` with zero missing rows.

### Appearance similarity

`FixedTrack.mean_hist()` supplies the retained track descriptor and
`Detection.hist` supplies the current descriptor. Both are normalized as
nonnegative histograms. Similarity is one minus Hellinger distance. Invalid
descriptors yield neutral value `0.5` with mask `0`; valid descriptors yield
mask `1`.

Both masks were `1` in all rows. Appearance is computed and threaded correctly.

Classification: `CONTRACT_CORRECT_MISSING`.

### Motion consistency

The caller supplies `track.predicted_box` when the track was detected before
and has at least two hits. Center residual is normalized by mean object
diagonal. Invalid motion yields neutral value `0.5` with mask `0`.

Both masks were `1` in all rows. The export cannot distinguish successful LK
from predicted-box fallback on the preceding skipped frame.

Classification: `UNRESOLVED` only for direct LK provenance, not score plumbing.

### Track freshness

`track.missed` is measured in detector opportunities and mapped to
`1-clip(missed/5,0,1)`. Hidden tracks are prefiltered to missed values `1..5`.
The same formula is applied to visible tracks.

Classification: `CONTRACT_CORRECT_MISSING` with zero missing rows.

### Availability indicators

`appearance_available` and `motion_available` are explicit binary score
features and validity masks. They are computed for hidden and visible tracks
with the same definitions and weights. All four side-specific values are `1`
for all rows.

Classification: `CONTRACT_CORRECT_MISSING`.

## Plumbing verdict

No evidence supports a computed-but-unthreaded feature, representation/name
mismatch, incorrectly false mask, asymmetric availability rule, or premature
descriptor deletion. The source and row export agree:

- appearance banks are retained and averaged;
- both track sides call the same feature builder;
- both optional masks reach the decision function;
- skipped frames update causal track geometry without incrementing detector
  opportunities;
- H1-r2 candidate construction occurs only on detector frames.

`FEATURE_PLUMBING_DEFECT_FOUND=NO`.

## The 33 valid score pairs

The score is uncalibrated and is not a probability.

| Statistic | `owner_preference_score` |
|---|---:|
| count | 33 |
| minimum | 0.382969 |
| p10 | 0.396972 |
| p25 | 0.408824 |
| median | 0.421524 |
| p75 | 0.448526 |
| p90 | 0.524389 |
| p95 | 0.551106 |
| maximum | 0.560375 |

Development GT interpretation matches the visible assignment box to
development GT at IoU at least `0.50`. It is diagnostic and does not replace a
tracking replay:

- likely beneficial: 2;
- likely harmful: 10;
- neutral: 0;
- ambiguous: 21.

The row-level export contains all eight hidden features, all eight visible
features, masks, score, relative score, threshold margin, matched development
GT, and the diagnostic interpretation.

## Diagnostic threshold screening

The first count below changes only the displayed score cutoff. The second
keeps the frozen minimum quality margin `0.20`; it remains zero at every
cutoff.

| Cutoff | Score-only pass | Positive episodes | Controls | Beneficial | Harmful | Ambiguous | Full frozen-margin pass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 5 | 2 | 0 | 1 | 1 | 3 | 0 |
| 0.52 | 4 | 2 | 0 | 1 | 1 | 2 | 0 |
| 0.54 | 3 | 2 | 0 | 0 | 1 | 2 | 0 |
| 0.56 | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| 0.58 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

No score-only cutoff separates likely beneficial evidence from harmful and
ambiguous evidence. The absence of control rows among the 33 valid pairs does
not establish safety: controls were excluded earlier by the overlap floor, and
no real reservation was applied.

This screening does not authorize a threshold change or a tracking run.

## Feasibility answers

1. **Does the current policy score enough real contention?** No. Only 33 of
   770 logged pairs reach scoring, and none reaches the frozen operating gate.
2. **Are some features auxiliary?** Yes. Appearance and motion already have
   explicit masks and neutral missing values. They are auxiliary evidence.
3. **Can the score handle missing raw features?** Mathematically yes. The
   explicit mask features distinguish neutral imputation from observed data.
4. **Can availability be relaxed symmetrically and causally?** Yes in
   principle, but availability is not the observed blocker here. A redesign
   must instead separate data availability from semantic pair eligibility.
5. **Would that be principled?** A predeclared symmetric relative-geometry
   contract and explicit reason taxonomy are principled. Lowering the score
   threshold from these six episodes is not.
6. **What dominates?** A restrictive hidden-only overlap eligibility floor,
   followed by an infeasible score/margin operating range among the remaining
   pairs. Feature plumbing is not the dominant issue.

`ROOT_DIAGNOSIS=MULTIPLE_DESIGN_FAILURES`.

## Next hypothesis decision

`H1_R3_PRINCIPLED_DESIGN_AVAILABLE=YES`.

A design-only H1-r3 may be prepared with these preimplementation constraints:

- preserve symmetric common-scale features;
- distinguish missing data from ineligible geometry and stale state;
- replace the hidden-only absolute-IoU availability gate with a predeclared
  symmetric relative-geometry contention contract;
- retain explicit appearance/motion masks and neutral missing values;
- predeclare a causal candidate-set rule that excludes unrelated hidden tracks;
- freeze new development and untouched validation populations before coding;
- do not choose a replacement threshold from this audit.

This authorizes design work only. H1-r3 implementation and all tracking,
validation, runtime, and promotion work remain unauthorized.

`NEXT_ACTION=DESIGN_H1_R3`.

## Validation blindness

Before and after hashes:

- validation manifest:
  `531f20a692d593c9ea4be0534e334af4262e8c013e43d828f0f0f894ad4761b6`;
- role assignment:
  `ae867355ff5ee04693451a52121e31606364d96def69dc7c0a03a585dfac3f0f`.

No validation output directory exists. No validation window was opened,
rendered, executed, scored, or summarized.

## Audit artifacts

Audit root:

`outputs/tracking/h1_r2_development_missingness_audit_20260727`

Final artifact manifest SHA-256:

`e46fe841ae7c0f05486945ce0eb4d7b8303618647d74da5b22044ea20af0b480`

The manifest covers the row integrity report, feature and combination tables,
episode/role/age/availability aggregates, all 33 valid pairs, score
distribution, diagnostic screening, provenance audit, and next-hypothesis
decision.
