# 6. Limitations and Future Work

While this study provides empirical insights into multi-object tracking, downstream profile
distortion, and multimodal pig behavior recognition, several technical and methodological
limitations must be acknowledged:

## 6.1 Camera Geometry, Environmental Constraints, and Stocking Density

All video data in this study were acquired from a single commercial swine research facility
utilizing
centrally mounted overhead cameras (3.2 m height) providing a top-down planar field of view over
pens
measuring 4.8 m x 2.4 m housing static groups of eight pigs. Commercial swine production,
however,
encompasses diverse pen configurations:
- **Group Sizes and Stocking Densities**: Commercial growing-finishing facilities frequently
  house 15
  to 35 pigs per pen, or use large-group auto-sorting systems with several hundred animals.
  Higher
  stocking densities exponentially increase occlusion frequency and duration, which may
  challenge
  bipartite Hungarian matching and increase identity switch rates.
- **Flooring and Pen Enrichment**: The evaluated pens featured concrete slatted floors with
  minimal
  bedding. Facilities utilizing deep straw bedding or solid floors with liquid slurry will
  introduce
  visual camouflage, partial body submersion, and manure accumulation, which may degrade visual
  crop
  clarity.
- **Camera Angles and Optical Occlusion**: Top-down overhead mounting minimizes mutual body
  occlusion relative to oblique or side-view camera angles. If cameras are mounted at lower
  ceiling
  heights or at oblique angles, body overlap will increase substantially, requiring 3D spatial
  reasoning or multi-camera fusion.

## 6.2 Whole-Pipeline Confounding in Tracking Baseline Comparisons

When comparing our proposed tracking systems against the frozen Raw ByteTrack baseline, certain
upstream detection parameters differed:
- **Detector Cadence**: RealTime-Fast operated at a 15 Hz detector cadence (evaluating YOLOv8
  every 2
  frames) to optimize computational efficiency, whereas Raw ByteTrack and Hybrid-ByteTrack
  evaluated
  detections at the full 30 Hz video rate.
- **Proposal Topologies**: Upstream detector topologies featured minor differences in confidence
  filtering thresholds (0.25 confidence threshold and 32 proposals in the current pipeline
  versus
  historical 0.20 threshold and 20 proposals in historical baselines).

Consequently, comparisons involving Raw ByteTrack must be interpreted conservatively as
whole-pipeline
system comparisons rather than isolated association algorithm ablations. Future work will
benchmark
these association algorithms across strictly harmonized upstream detector proposals.

## 6.3 Behavior Evaluation Scope and Absence of Materialized Outer Test Cohort

A critical methodological boundary relates to the behavior dataset partitions:
- The 33,287 canonical 6-frame units (spanning 678 video clips across 12 source videos)
  constitute
  100% of the 5-fold video-group balanced cross-validation development dataset.
- While a nested 4-fold outer protocol was defined in `outer_oof_contract.json` (`status:
  FROZEN_PROTOCOL_NOT_AUTHORIZED`), no distinct outer physical dataset was materialized on disk,
  resulting in zero outer test evaluations ($OUTER\_TEST\_EVALUATIONS = 0$).
- Although 5-fold video-group cross-validation rigorously guarantees that no video or pig is
  shared
  between training and validation sets, true out-of-distribution generalization across different
  farms,
  different pig genetics, or different age cohorts remains unmeasured and represents an
  important
  priority for future multi-center validation campaigns.

## 6.4 Severe Class Imbalance in Rare Ethogram Behaviors

In commercial swine ethograms, behavioral actions exhibit extreme biological skew. In our
dataset,
maintenance behaviors such as resting (`lying`, `stand`) accounted for more than 75% of
canonical
units. In contrast, critical animal welfare events such as `fight`, `playwithtoy`, and `sitting`
each
accounted for less than 3% of the dataset.

Although video-group balanced cross-validation, class-aware gating, and weighted loss functions
mitigated this imbalance, per-class F1 scores for rare, short-duration behaviors remain lower
than
those for stationary postures. Future work should investigate few-shot learning, event-based
temporal
sampling, and multimodal contrastive pre-training to improve detection of infrequent,
high-impact
welfare events.
