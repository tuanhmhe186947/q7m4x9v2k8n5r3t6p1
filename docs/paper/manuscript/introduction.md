# 1. Introduction

## 1.1 Precision Livestock Farming and the Need for Individual Monitoring

In modern precision livestock farming (PLF), automated computer vision offers substantial
potential for non-invasive, continuous monitoring of animal health and welfare in commercial
group-housing facilities. For swine production, continuous video tracking enables early
detection
of lameness, monitoring of individual feed and water consumption, quantification of activity
levels,
and timely identification of aggressive interactions. Because pigs are social animals housed in
groups, aggregate group-level metrics frequently conceal early clinical signs or welfare issues
exhibited by specific individuals. Establishing valid longitudinal behavioral time budgets—the
cumulative proportion of time each individual allocates to feeding, drinking, locomotion,
resting,
and social interactions—is therefore the foundational scientific requirement for automated
phenotyping and individualized veterinary intervention.

## 1.2 The Identity Preservation Bottleneck

Despite substantial progress in deep learning for computer vision, automated individual tracking
in
commercial swine housing remains exceptionally challenging. Group-housed pigs exhibit
near-identical
morphology and visual texture, lack distinct natural pigmentation patterns, and frequently
engage in
close physical contact, mounting, and crowding around resources such as feeders and drinkers.
Under
these conditions, standard Multi-Object Tracking (MOT) algorithms frequently experience identity
switches (IDSW), where the assigned tracking identifiers of two or more animals are erroneously
swapped or fragmented across occlusions.

In conventional computer vision research, tracking algorithms are primarily benchmarked using
detection and association metrics such as Higher Order Tracking Accuracy (HOTA), Multiple Object
Tracking Accuracy (MOTA), and Identification F1 (IDF1). While these metrics quantify
bounding-box
overlap and association errors across frames, they do not measure the scientific consequence of
tracking failures on downstream biological phenotypes. In livestock behavioral science, the
primary
analytical unit is not the per-frame bounding box, but the individual longitudinal time budget.
When an identity switch occurs, the behavioral actions of one animal are erroneously attributed
to
another. If an identity switch resolves quickly, the misattribution may remain negligible;
conversely, if an identity switch persists over several minutes, the cumulative behavioral
profile of
both animals is severely corrupted. A critical unanswered question in automated livestock
monitoring
is how upstream tracking errors propagate mechanistically into downstream behavioral profiles,
and
whether conventional discrete metrics like IDSW sufficiently predict downstream profile
distortion.

## 1.3 Overview of the Proposed Framework

To address these challenges, this paper presents an end-to-end framework integrating
multi-object
tracking and multimodal spatio-temporal behavior recognition, evaluated across rigorous,
preregistered experimental protocols.

First, recognizing that farm operations impose divergent computational and latency requirements,
we
formulate a dual-mode tracking framework:
1. **RealTime-Fast (Causal Online Mode)**: Designed for low-latency edge deployment and
   streaming
   alert systems (e.g., immediate intervention during tail-biting or severe aggression). This
   mode
   operates with a 15 Hz detector cadence, utilizing Kalman filtering with velocity gating and
   spatial proximity associations without future temporal buffers.
2. **Hybrid-ByteTrack (Retrospective Offline Mode)**: Designed for retrospective behavioral
   profiling, longitudinal phenotyping, and epidemiological research where complete video
   recordings
   are accessible. This mode incorporates backward trajectory smoothing, graph-based
   spatial-temporal
   refinement, and trajectory stitching to maximize identity continuity.

Second, we bridge the methodological divide between computer vision tracking metrics and
livestock
behavioral science by formulating the **Total Variation (TV) Profile Distortion Metric**. By
holding
dense, human-reviewed behavioral annotations fixed across 96 pig-video sessions (28,800
canonical
6-frame units), we isolate the exact mathematical propagation of tracking identity swaps into
individual behavioral time budgets. We explicitly compare the predictive capacity of discrete
identity switch counts versus cumulative wrong-identity exposure duration.

Third, for fine-grained behavior classification across 10 ethogram categories, we develop
multimodal
spatio-temporal sequence models that integrate actor visual crops, 6D bounding-box spatial
geometry,
12D relative motion trajectories, and partner/social latent contexts. We investigate Model J, an
architecture featuring local spatial attention and learned class-aware gating, through a 2x2
factorial ablation to isolate architectural mechanisms. We further explore knowledge
distillation
from an ensemble teacher ($J_{PBNEW}$) and locked 50/50 ensemble systems ($E_{J\_FIXED50}$).

Finally, we assess upstream tracking generalization on 12 independent held-out videos under
Standard
V2 bipartite matching ($CONFIRMATORY_B$), and downstream behavior models across 33,287 canonical
units across 678 videos under a 5-fold video-group balanced cross-validation scheme.

## 1.4 Summary of Contributions

The principal scientific contributions of this study are as follows:

1. **Dual-Mode Tracking Formulation and Held-Out Validation (Claims C1, C2, C3)**: We formulate
   and
   empirically evaluate causal online (RealTime-Fast) and retrospective offline
   (Hybrid-ByteTrack)
   tracking algorithms. On 12 independent held-out evaluation videos ($CONFIRMATORY_B$),
   Hybrid-ByteTrack achieved 93.55% HOTA, 98.41% IDF1, and 8 IDSW (an 87.5% reduction in IDSW
   vs.
   frozen Raw ByteTrack with 64 IDSW; delta HOTA +4.36% [1.73%, 7.31%], bootstrap positive
   fraction
   0.9999). RealTime-Fast achieved 90.33% HOTA, 95.03% IDF1, and 39 IDSW, halving upstream
   detector
   evaluations while reducing IDSW by 39.1%.
2. **Mechanistic Analysis of Downstream Profile Distortion (Claims C6, C7, C8)**: We formulate
   the
   Total Variation profile distortion metric and demonstrate on 12 development videos under
   fixed
   human ground truth that tracking swaps cause substantial profile corruption (mean A1 TV of
   0.0953
   for Raw ByteTrack, 0.0309 for RealTime-Fast, and 0.0001 for Hybrid-ByteTrack; LOOV sign
   consistency
   12/12 positive). Furthermore, we prove that cumulative wrong-identity exposure frames provide
   a
   stronger rank association with profile corruption than discrete switch counts for Raw
   ByteTrack
   (delta abs rho = +0.0668, 95% CI: [0.0116, 0.1501]).
3. **Multimodal Behavior Recognition and Factorial Ablation (Claims C4, C5)**: We develop
   multimodal
   sequence architectures for 10-class pig behavior recognition. A 2x2 factorial ablation across
   15
   checkpoints proves that class-aware gating is the primary standalone driver of performance
   (+0.005109 Macro-F1 over scaffold), whereas un-gated local spatial attention yields zero gain
   alone
   (+0.000000). A locked 50/50 ensemble ($E_{J\_FIXED50}$) achieves the highest validated 5-fold
   Macro-F1 of 0.693905 +/- 0.037650.
4. **Rigorous Preregistration and Data Governance (Claim C9)**: All evaluations adhere to strict
   reproducibility governance. We clarify that the 33,287 units represent 100% of the 5-fold CV
   development dataset, with zero outer test leakage ($OUTER\_TEST\_EVALUATIONS = 0$), and
   document
   the provenance and confidence intervals for all reported quantities.
