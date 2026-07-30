# Tracking development evidence and reviewer defense report

## 1. Executive summary

### Development evidence

- Strongly established: on the frozen 13-video development population,
  `hybrid_bytetrack` has the highest HOTA (0.900291)
  and strongest identity diagnostics; `rf_hybrid` is a mixed negative transfer
  result rather than a quality upgrade.
- Descriptively observed: per-video paired effects, leave-one-video-out
  influence, and video-cluster bootstrap ranges are reported without treating
  frames as independent.
- Not established: byte-exact historical hybrid reproducibility, empirical
  real-time performance, a pure association-core effect, deployment readiness,
  or sufficiency for a tracking-method-primary paper.

### Generalization evidence not yet available

No claim is made about unseen recordings or sessions. The planned separate
12-video evaluation is required for every generalization statement.

## 2. Study scope

This package analyzes existing predictions only. Detector runs, tracker runs,
parameter tuning, per-video rules, MP4 generation, unseen access, and prediction
or GT modification are all zero. The 23,400 frames are measurement rows, not
independent inferential samples.

## 3. Four canonical methods

The active IDs are exactly `bytetrack_raw`, `hybrid_bytetrack`,
`realtime_fast`, and `rf_hybrid`. `rf_hybrid v2` is a rejected candidate,
standardized B1 is forensic-only, and no symmetric 2x2 claim is active.

## 4. Prediction and execution authorities

`bytetrack_raw` and `realtime_fast` have current executable prediction
authorities. Historical XMLs remain primary for `hybrid_bytetrack`; its full
accepted lineage is recovered but exact historical runtime is not. `rf_hybrid`
v1 is a frozen development transfer ablation.

## 5. Development population and dependence structure

All methods use the same 13 videos, frames 0-1799, GT hash, Standard V2
contract, 19 HOTA thresholds, IoU 0.50 eligibility, IDSW_STANDARD, and primary
`include_hidden=true` policy. No documented camera/pen/session metadata proves
cross-video independence. Video is therefore the descriptive cluster unit;
session aggregation is not justified and frame-level inference is forbidden.

## 6. Canonical aggregate results

| Method | HOTA | IDF1 | IDSW | FP/FN | Fragments | Wrong-ID | Terminal |
|---|---:|---:|---:|---:|---:|---:|---:|
| bytetrack_raw | 0.849511 | 0.920646 | 84 | 480/480 | 141 | 31846 | 27 |
| hybrid_bytetrack | 0.900291 | 0.991501 | 0 | 1579/1579 | 425 | 24 | 0 |
| realtime_fast | 0.888187 | 0.971892 | 29 | 473/597 | 107 | 11893 | 12 |
| rf_hybrid | 0.878281 | 0.957881 | 18 | 412/536 | 87 | 14515 | 14 |


## 7. Per-video/session consistency

All 52 method-video rows are supplied. Raw counts reproduce canonical global
totals. HOTA and IDF1 are not arithmetic means: HOTA combines per-alpha
sufficient statistics, while IDF1 recomputes from summed identity counts.
Per-session results are not produced because session grouping is unproven.

## 8. Paired comparisons

- C1 realtime_fast minus bytetrack_raw mean per-video HOTA difference:
  0.040905; W/T/L =
  9/0/4.
- C2 hybrid_bytetrack minus realtime_fast mean per-video HOTA difference:
  0.012761; W/T/L =
  6/0/7.
- C3 rf_hybrid minus realtime_fast mean per-video HOTA difference:
  -0.010729; W/T/L =
  11/0/2.

Aggregate advantage is never labeled consistent unless its video directions
support that description.

## 9. Uncertainty and influence analysis

Paired video bootstrap uses seed 20260730 and 10000
resamples. Percentile ranges are descriptive only. Formal CI status is
`INSUFFICIENT_INDEPENDENT_CLUSTERS_FOR_RELIABLE_CI` because recording-session
independence is unresolved. Leave-one-video-out results identify influential
clusters and replace frame-level significance claims.

## 10. Detection–association trade-offs

Hybrid has stronger AssA/IDF1 and identity severity but lower DetA and larger
FP/FN/fragments than the executable methods. It is therefore not best in every
dimension. rf_hybrid slightly improves detection counts while degrading broad
association quality and identity exposure.

## 11. IDSW=0 and fragmentation analysis

Conclusion: `IDSW_ZERO_SUPPORTED_BY_BROAD_IDENTITY_CONTINUITY`. Zero IDSW is supported jointly by
IDF1=0.991501, AssA=0.911904, only 24 wrong-ID animal-frames, eight recovered
short episodes, zero terminal episodes, and zero persistent swaps. The 425
Standard V2 fragments remain a real detection/localization continuity cost.
The same historical prediction has only six gaps exceeding the supplementary
15-frame gap-tolerant rule, indicating most strict fragments are short gaps,
not persistent owner loss. This does not substitute for human trajectory audit.

## 12. Hidden/visible sensitivity

Conclusion: `RANKING_ROBUST_TO_HIDDEN_POLICY`. Visible-only results are secondary and do
not replace the primary include-Hidden authority. Exclusion tests sensitivity;
it does not validate or correct tracker-derived Hidden labels. The hybrid-minus-
realtime HOTA difference is 0.012103 under the primary
policy and 0.011013 visible-only, so the development HOTA
advantage is not concentrated in Hidden observations. Realtime wrong-ID exposure
changes from 11,893 to 9,996 animal-
frames (16.0%
lower), but this cannot be interpreted as a causal occlusion-error fraction
because exclusion changes the matching population. Largest absolute per-video
HOTA changes are:
- rf_hybrid:Pigs291119_000216_30fps (-0.0667)
- bytetrack_raw:Pigs291119_000216_30fps (-0.0664)
- realtime_fast:Pigs291119_000216_30fps (-0.0660)
- hybrid_bytetrack:Pigs291119_000216_30fps (-0.0644).

## 13. Identity-error and GT audit status

The non-mutating audit pack includes all 24 hybrid wrong-ID animal-frames, all
hybrid episodes, all realtime terminal episodes, every persistent pairwise
swap, long raw-baseline episodes, recovered samples, Hidden-policy references,
and influential videos. `HUMAN_GT_AUDIT_COMPLETED=NO`; every item is explicitly
`NOT_REVIEWED`.

## 14. Hybrid historical-versus-rerun reproducibility

Historical HOTA is 0.900291 and the latest full rerun is
0.900240; historical IDF1 is 0.991501 and rerun IDF1 is
0.991437. Both have IDSW=0 and 24 wrong-ID frames. This establishes
metric-level near parity, not byte-exact prediction reproduction. Historical
XMLs remain primary and runtime provenance remains incomplete.

## 15. Complete-method fairness

C1 is a complete-method comparison; C2 is a complete-method comparison with a
historical-runtime limitation; C3 is a transfer ablation. No result supports a
pure association-core, single-stage causal, detector-controlled, or identical
topology claim.

## 16. Causal-versus-realtime status

`realtime_fast` has zero-delay causal semantics, but the frozen runtime
benchmark protocol has not been executed. The defensible wording is "causal
realtime-oriented method," not "real-time system."

## 17. Baseline adequacy

ByteTrack is an appropriate conventional baseline for a behavior-pipeline-
primary paper. Internal variants are not external baselines. An additional MOT
baseline is recommended—and required for a strong tracking-method-primary
claim—but is not run in this task.

## 18. Negative rf_hybrid transfer result

rf_hybrid reduces IDSW 29 to 18, FP 473 to 412, FN 597 to 536, and fragments
107 to 87. It also decreases HOTA 0.888187 to 0.878281 and IDF1 0.971892 to
0.957881, while increasing wrong-ID frames 11,893 to 14,515 and terminal
episodes 12 to 14. This is a scientifically valid mixed negative result.

## 19. Claims supported on development

The claim matrix contains 2 strongly supported development claims and
3 claims supported with major limitations. The strongest statements are
limited explicitly to the frozen development population.

## 20. Claims requiring unseen evaluation

1 matrix claim(s) are explicitly classified `REQUIRES_UNSEEN`; several
other development claims also require the 12-video set before generalization.

## 21. Likely reviewer objections and responses

Thirteen neutral objection responses are supplied. The most serious risks are
development overfitting/no unseen generalization, unresolved session
independence, and historical hybrid runtime incompleteness. None is concealed.

## 22. Limitations

- Development and method iteration share the same 13-video population.
- Recording-session independence cannot be proven from existing metadata.
- Hidden labels are not fully human-validated.
- Exact historical hybrid prediction/runtime reproduction is unavailable.
- The standardized runtime protocol is not executed.
- One conventional external tracker baseline is available.
- Mostly-tracked/partially-tracked/mostly-lost breakdown is unavailable.

## 23. Paper-ready wording

"On the frozen 13-video development set, the historical offline hybrid had the
highest HOTA and strongest identity-continuity diagnostics. The causal
realtime-oriented method improved broadly over the current executable
ByteTrack baseline as a complete pipeline, not as a detector-controlled
association-core ablation. Transfer of selected hybrid mechanisms into the RF
tracklets reduced IDSW but worsened HOTA, IDF1, and wrong-identity exposure."

## 24. Remaining work before final submission

1. Freeze and run the separate 12-video evaluation without method changes.
2. Complete human review of the generated GT/error audit pack.
3. Execute the frozen paired runtime protocol before any real-time claim.
4. Add an external MOT baseline only if tracking becomes a primary paper claim.
5. Report session-cluster inference only if authoritative session metadata is
   recovered.

This document is development evidence and reviewer preparation, not final
external validation. Unsupported claims in the matrix: 4.
