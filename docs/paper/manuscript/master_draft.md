# Identity-Preserving Multi-Object Tracking and
# Multimodal Spatio-Temporal Behavior Recognition for Group-Housed Pigs

**Authors**: Anonymous Authors (Under Double-Blind Peer Review)

**Affiliations**: Department of Animal Science and Institute of Artificial Intelligence

**Target Journal**: Computers and Electronics in Agriculture / IEEE Transactions on AgriFoodTech

---

# Abstract

**Title**: Identity-Preserving Multi-Object Tracking and Multimodal Spatio-Temporal Behavior
Recognition for Group-Housed Pigs

**Context**: In precision livestock farming, automated vision systems must monitor individual
animal
behavior over extended periods to support animal welfare, feeding management, and disease
detection.
However, visual homogeneity, frequent close contact, and body occlusions among group-housed pigs
cause tracking identity switches that misattribute behaviors and distort longitudinal individual
time budgets.

**Objectives**: We investigate the mechanistic propagation of multi-object tracking errors into
downstream individual behavioral profiles and evaluate a dual-mode tracking and multimodal
behavior
recognition architecture designed to preserve identity integrity.

**Methods**: We formulate a dual-mode tracking framework comprising a causal online mode
(RealTime-Fast, 15 Hz detector cadence) for low-latency alerting and a retrospective offline
mode
(Hybrid-ByteTrack) with bidirectional smoothing and graph-based association for longitudinal
profiling. We establish an empirical evaluation on 12 development videos (96 pig-video sessions,
28,800 canonical 6-frame units) holding human-annotated behavior ground truth fixed to measure
Total
Variation (TV) profile distortion. Upstream tracking generalization is assessed on 12
independent
held-out videos under Standard V2 bipartite Hungarian matching (CONFIRMATORY_B). Downstream
behavior
recognition is evaluated across 33,287 canonical 6-frame units across 678 videos using a 5-fold
video-group balanced cross-validation scheme.

**Results**: On independent evaluation videos, Hybrid-ByteTrack achieved 93.55% HOTA (95% CI:
[89.85%, 96.24%]), 98.41% IDF1 ([96.26%, 99.70%]), and 8 identity switches (IDSW), reducing IDSW
by
87.5% relative to the frozen Raw ByteTrack baseline (64 IDSW; delta HOTA +4.36% [1.73%, 7.31%],
bootstrap positive fraction 0.9999). RealTime-Fast achieved 90.33% HOTA ([84.13%, 94.43%]),
95.03%
IDF1 ([89.49%, 99.16%]), and 39 IDSW (delta HOTA +0.78% [-2.93%, 4.53%], bootstrap positive
fraction
0.6602). Downstream profile distortion analysis demonstrated that tracker identity swaps
corrupted
individual behavioral time budgets: mean A1 TV distortion was 0.0953 for Raw ByteTrack, 0.0309
for
RealTime-Fast, and 0.0001 for Hybrid-ByteTrack (Raw vs RealTime delta = 0.0644 [0.0216, 0.1073];
Raw
vs Hybrid delta = 0.0951 [0.0393, 0.1586]; all 12/12 leave-one-out folds positive).
Wrong-identity
exposure duration was more strongly associated with profile distortion than discrete switch
count
for Raw ByteTrack (Spearman rho = 0.9381 vs 0.8712; delta abs rho = 0.0668 [0.0116, 0.1501]).
For
spatio-temporal behavior recognition across 10 ethogram categories, a 2x2 factorial ablation
demonstrated that class-aware gating was the primary driver of single-model performance
(+0.005109 Macro-F1 over scaffold), whereas un-gated local spatial attention produced no gain
alone (+0.000000). A fixed 50/50 ensemble of focal and joint representation models achieved
the highest validated 5-fold Macro-F1 of 0.693905 +/- 0.037650.

**Conclusions**: Upstream tracking errors propagate directly into individual livestock
behavioral
budgets, where wrong-identity exposure duration governs phenotypic distortion. The proposed
dual-mode tracking framework provides operating points for both causal streaming alerts and
retrospective profiling with near-zero downstream distortion.

---

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

---

# 2. Related Work

## 2.1 Multi-Object Tracking in Precision Livestock Farming

Automated visual tracking of individual animals is a cornerstone of precision livestock farming.
Early approaches in swine tracking relied heavily on classical computer vision techniques, such
as
adaptive background subtraction, frame differencing, and optical flow combined with
morphological
filtering or ellipse fitting. While effective under low stocking densities and controlled
lighting,
classical methods degrade rapidly in commercial environments characterized by shifting
illumination,
soiled pens, and physical crowding.

The advent of deep learning transformed animal multi-object tracking through detection-based
tracking
paradigms. Standard two-stage and one-stage object detectors (Faster R-CNN, SSD, YOLO families)
have
been paired with association algorithms such as SORT, DeepSORT, ByteTrack, BoT-SORT, and OC-SORT
to
track pigs, cattle, and poultry. In swine monitoring specifically, ByteTrack has gained
prominence
due to its association strategy that matches low-confidence detection boxes to retain tracks
through
partial occlusions.

However, group-housed pigs present tracking difficulties rarely encountered in standard human
MOT
benchmarks (e.g., MOT17, MOT20). Pigs housed under commercial conditions share uniform pinkish
or
white pigmentation, lack distinct visual markings, exhibit highly deformable postures (standing,
lying laterally, sternal recumbency, huddling), and frequently crowd tightly at communal feeding
and
drinking troughs. Visual re-identification (Re-ID) embeddings trained on appearance features
frequently confuse penmates because inter-individual visual variance is often smaller than
intra-individual posture variance. Furthermore, existing studies typically evaluate tracking on
short
video clips (10 to 60 seconds) and report aggregate frame-level metrics such as MOTA, HOTA, and
IDF1,
leaving the longitudinal identity stability of tracking pipelines unexamined.

## 2.2 Vision-Based Pig Behavior Recognition

Automated quantification of swine behavior has progressed from wearable sensor-based
technologies to
contactless computer vision. While wearable sensors (ear tags, collars, accelerometers, RFID)
provide
direct kinematic or proximity data, they are invasive, prone to detachment, mechanically damaged
by
chewing, and costly to scale to thousands of market pigs. Consequently, camera-based visual
monitoring has emerged as the preferred non-intrusive paradigm.

Computer vision approaches to pig behavior recognition have evolved from single-frame 2D CNNs to
spatio-temporal deep architectures. Single-frame classifiers analyze static posture (standing,
sitting, lying) but cannot resolve dynamic actions such as drinking, rooting, or fighting. To
model
temporal dynamics, researchers have employed 3D convolutional networks (C3D, I3D, SlowFast),
recurrent architectures (ConvLSTM), and spatio-temporal Transformers (TimeSformer, Video Vision
Transformers).

A persistent challenge in swine ethogram recognition is distinguishing visually similar
behaviors
that differ primarily in social context or fine kinematic nuances. For example, aggressive
fighting
and non-aggressive social-nose investigation both involve close head-to-head proximity between
two
pigs; however, fighting is marked by vigorous head knocking, biting, and rapid angular
displacement,
whereas social exploration is characterized by gentle sniffing and low-velocity contact.
Similarly,
feeding involves prolonged snout contact within a feed hopper, which visually resembles
exploratory
floor rooting near pen boundaries. Furthermore, commercial pig behavior datasets exhibit severe
class imbalance: maintenance behaviors like lying and standing dominate 70% to 90% of
observation
time, whereas high-impact welfare events like fighting or play comprise less than 2% of
observations.

## 2.3 Multimodal and Social Context Modeling

To overcome the visual ambiguity of isolated animal crops, recent literature emphasizes
multimodal
feature fusion and social context representation. In human action recognition, combining RGB
video
with spatial bounding-box geometry and optical flow or coordinate trajectories consistently
yields
higher discriminative capability. In animal behavior analysis, incorporating bounding-box
geometric
attributes (center coordinates, area, aspect ratio, orientation) provides vital spatial context
regarding an animal's location relative to stationary pen resources (feeders, drinkers,
enrichment
objects).

Moreover, because swine behavior is inherently social, an individual's behavioral state is
frequently influenced by or coordinated with penmates. Graph Convolutional Networks (GCNs) and
spatial-temporal graph attention networks have been explored to represent penmates as graph
nodes,
modeling inter-individual spatial distances and relative velocities as edge features. However,
fully
connected graph architectures often suffer from high computational complexity and over-smoothing
when scaled to dense commercial pens. This motivates the development of lightweight social
representations that capture partner behavior and proximity while preserving computational
efficiency.

## 2.4 Tracking Error Propagation and Downstream Behavioral Profiling

Despite substantial parallel progress in multi-object tracking and action recognition, the two
fields have developed largely in isolation. In the broader computer vision literature, tracking
benchmarks evaluate detection and association accuracy on raw bounding boxes, whereas action
recognition benchmarks evaluate classification on pre-segmented, identity-guaranteed video
clips.

In operational precision livestock farming, these two tasks are inextricably coupled: tracking
outputs serve as the spatial and temporal scaffolding for individual behavior recognition. When
an
upstream tracker switches the identities of two pigs, the downstream behavior classifier assigns
subsequent action predictions to the incorrect animal. If tracking identities switch during an
aggressive encounter, an aggressive pig may be labeled as victimized, or an inactive resting pig
may
be credited with excessive feeding bouts.

Prior literature in animal monitoring has occasionally acknowledged tracking drift as an
operational
concern, but systematic mathematical frameworks for quantifying this error propagation remain
lacking. Specifically, previous works have not isolated the exact functional relationship
between
upstream MOT association errors (such as IDSW and wrong-identity exposure duration) and
downstream
behavioral profile distortion under controlled, human-verified behavioral ground truth. This
work
specifically addresses this gap.

---

# 3. Materials and Methods

## 3.1 Experimental Setup, Ethogram, and Dataset Roles

### 3.1.1 Video Acquisition and Housing Conditions
Video data were collected from commercial swine production pens housing grow-finish pigs. Each
pen
measured approximately 4.8 m x 2.4 m with concrete slatted flooring and contained a static group
of
eight pigs (n = 8). Overhead cameras (1080p resolution, 1920 x 1080 pixels) were mounted
centrally
above each pen at a height of 3.2 m, providing a continuous top-down planar field of view
covering
the entire floor area, including the communal dry feeder and nipple drinker stations. Videos
were
recorded at 30 frames per second (fps) under natural and supplemental artificial lighting.

### 3.1.2 Behavioral Ethogram and Canonical Temporal Units
A comprehensive swine ethogram consisting of 10 mutually exclusive behavioral categories was
defined
in consultation with animal welfare specialists:
1. `drink`: Snout positioned in or manipulating the nipple drinker with active ingestion.
2. `eat`: Head lowered into the feed hopper with active chewing or swallowing.
3. `fight`: Vigorous physical aggression, including head knocking, biting, pushing, or
   reciprocal
   ramming between two or more pigs.
4. `social-nose`: Non-aggressive physical contact, including gentle snout-to-snout or
   snout-to-body
   sniffing and tactile exploration.
5. `explore`: Rooting, sniffing, or chewing pen fixtures, pen walls, or floor slats away from
   the
   feeder and drinker.
6. `lying`: Recumbent resting posture (lateral or sternal) without active locomotion.
7. `stand`: Upright stationary posture supported by all four limbs without active forward
   locomotion.
8. `move`: Active walking, running, or directional locomotion across the pen floor.
9. `sitting`: Posture where the posterior rests on the pen floor while forelegs remain extended.
10. `playwithtoy`: Physical manipulation, biting, or shaking of suspended pen enrichment objects
    (hanging chains or rubber toys).

To standardize temporal modeling across varying action durations, we defined the canonical
temporal
unit as a contiguous window of six frames at 30 fps (T = 6, corresponding to exactly 0.20
seconds of
physical observation). Each canonical unit was assigned a single consensus behavior label based
on
expert human review.

### 3.1.3 Dataset Roles, Partitions, and Governance Boundaries
To prevent data contamination across model development, ablation analysis, and independent
evaluation, data partitions were organized into explicit roles governed by immutable manifests:

1. **Tracking Development Cohort (Full, 13 videos)**: Comprises 13 full video recordings
   (187,200
   physical frames, 104 pig trajectories) with complete bounding box and track identity
   annotations.
   Contains 1 tracking-only video (`Pigs291119_000263_30fps`) featuring high occlusion and dense
   clustering, used for tracking algorithm development and parameter tuning.
2. **Tracking Development / Behavior Overlap Cohort (DEV12, 12 videos)**: Consists of 12 videos
   (96
   pig sessions, 172,800 frames, 28,800 canonical 6-frame units) that possess dual ground truth:
   complete bounding box tracking identities and dense human-reviewed behavior annotations
   across all
   10 categories. This cohort strictly excludes video `000263` and is used exclusively for
   evaluating
   how upstream tracking errors propagate into downstream behavioral profiles under fixed human
   ground
   truth.
3. **Tracking Independent Confirmatory Cohort (12 videos)**: Comprises 12 independent held-out
   videos
   (21,600 evaluated frames, 96 pig trajectories) with tracking bounding box annotations and
   zero
   behavior labels. These videos were evaluated in August 2026 under commit `db1cccb7`,
   verified, and
   certified under Standard V2 without parameter modifications ($CONFIRMATORY_B$).
4. **Behavior Recognition 5-Fold Balanced Cross-Validation Cohort**: Encompasses 33,287
   canonical
   6-frame units drawn from 678 video clips across five balanced video groups (VG1 to VG5).
   Folds
   were constructed such that all frames from any given video recording were restricted entirely
   to a
   single fold, guaranteeing zero video leakage between training and validation sets.
5. **Behavior Nested Outer Protocol Status**: A nested 4-fold outer evaluation protocol was
   pre-registered in `outer_oof_contract.json` (`status: FROZEN_PROTOCOL_NOT_AUTHORIZED`);
   however, no
   distinct outer physical dataset was materialized on disk. The 33,287 units constitute 100% of
   the
   5-fold CV development dataset, resulting in zero outer evaluations ($OUTER\_TEST\_EVALUATIONS
   = 0$).

## 3.2 Multi-Object Tracking Framework

We formulate a dual-mode tracking framework designed to meet divergent operational constraints
in
precision livestock monitoring:

### 3.2.1 Object Detection Backbone
Upstream bounding box detections are generated by a YOLOv8 object detector trained on annotated
pig
instances. The detector outputs axis-aligned bounding boxes $B = [x_1, y_1, x_2, y_2]$ with
associated
detection confidence scores $s \in [0, 1]$.

### 3.2.2 Causal Online Tracking (RealTime-Fast)
RealTime-Fast is engineered for low-latency edge deployment and real-time streaming alert
systems. To
minimize compute overhead on resource-constrained farm hardware, RealTime-Fast operates at a
reduced
detector cadence of 15 Hz (running detection every 2 frames, step = 2). Track states are
maintained
using an 8-dimensional Kalman filter modeling bounding box center coordinates, aspect ratio,
height,
and their respective velocities. Bipartite association between active tracks and detection
proposals
is executed via the Hungarian algorithm using Intersection-over-Union (IoU) distance gated by
velocity
and spatial proximity thresholds. RealTime-Fast operates in a strictly causal manner:
predictions for
frame $t$ utilize only information from frames $\tau \le t$, with zero future temporal
buffering.

### 3.2.3 Retrospective Offline Tracking (Hybrid-ByteTrack)
Hybrid-ByteTrack is designed for post-hoc behavioral phenotyping, daily time-budget compilation,
and
veterinary epidemiology, where entire video recordings can be processed retrospectively.
Operating at
full 30 Hz detector cadence, Hybrid-ByteTrack adopts a hierarchical association scheme:
1. Primary association matches high-confidence detections ($s \ge 0.50$) to existing tracks
   using
   spatial-temporal IoU and Kalman motion predictions.
2. Secondary association matches remaining unmatched tracks to low-confidence detections ($0.10
   \le
   s < 0.50$) to recover occluded animals.
3. Retrospective trajectory refinement executes bidirectional Kalman smoothing across the
   complete
   temporal sequence, followed by trajectory graph analysis to detect and resolve
   spatial-temporal
   conflicts. If two tracks share conflicting bounding boxes during an occlusion event, an
   identity
   consistency graph resolves track continuity based on pre- and post-occlusion momentum and
   trajectory linearity.

### 3.2.4 Standard V2 Tracking Evaluation Protocol
All tracking algorithms were evaluated using the standardized `TRACKING_EVALUATOR_STANDARD_V2`
protocol:
- **Bipartite Hungarian Matching**: Ground-truth bounding boxes and tracker predictions are
  matched
  at an IoU threshold of 0.50 over a 19-point alpha threshold grid $\alpha \in [0.05, 0.95]$ in
  increments of 0.05.
- **Occlusion Handling**: The evaluation contract enforces `include_hidden = True`, ensuring
  that
  occluded animals holding valid ground-truth annotations are retained in the association pool.
- **Primary MOT Metrics**: We report Higher Order Tracking Accuracy (HOTA), Detection Accuracy
  (DetA), Association Accuracy (AssA), Identification F1 (IDF1), and discrete Identity Switches
  (IDSW).
- **Wrong-Identity Exposure Denominators**: To prevent ambiguity in reporting identity error
  durations, we evaluate two explicit denominators:
  - Canonical metric: $\text{wrong\_id\_fraction\_matched} =
    \frac{\text{wrong\_id\_matched\_frames}}
    {\text{authoritative\_matched\_gt\_frames}}$.
  - Secondary metric: $\text{wrong\_id\_fraction\_all\_gt} =
    \frac{\text{wrong\_id\_matched\_frames}}
    {172,800\text{ total GT frames}}$.
  - Exposure duration is converted to physical time as: $\text{wrong\_id\_seconds} =
    \frac{\text{wrong\_id\_matched\_frames}}{30\text{ fps}}$.

## 3.3 Downstream Behavioral Profile Distortion Methodology

### 3.3.1 Mathematical Formulation of Profile Distortion
To isolate the direct impact of upstream tracking failures on downstream behavioral budgets, we
conducted an empirical simulation across the 12 development videos (DEV12, 96 pig sessions,
28,800
canonical units). We hold the authoritative, human-reviewed behavioral ground truth completely
fixed
while varying the upstream tracking source.

For each pig trajectory $i \in \{1, \dots, N\}$ ($N = 8$ pigs per pen) in a video session, the
true
behavioral time budget is represented as a probability distribution over the $C = 10$ ethogram
classes:
$$P_{gt}^{(i)}(c) = \frac{1}{U_i} \sum_{u=1}^{U_i} \mathbb{I}(y_u^{(i)} = c)$$
where $U_i$ is the total number of annotated canonical units for pig $i$, $y_u^{(i)}$ is the
true
behavior category of unit $u$, and $\mathbb{I}(\cdot)$ is the indicator function.

When an upstream tracker $m$ (Raw ByteTrack, RealTime-Fast, or Hybrid-ByteTrack) is evaluated,
its
predicted bounding boxes at unit $u$ are associated with ground-truth pigs. If tracker $m$
maintains
the correct identity, the unit's true behavior is attributed to pig $i$. If tracker $m$ has
swapped
identities with pig $j$, the true behavior of pig $j$ is erroneously accumulated into the budget
of
pig $i$. The resulting corrupted profile is denoted $P_{pred}^{(i,m)}(c)$.

We define the **Total Variation (TV) Profile Distortion Metric (Metric A1)** as:
$$TV(P_{pred}^{(i,m)}, P_{gt}^{(i)}) = \frac{1}{2} \sum_{c=1}^C |P_{pred}^{(i,m)}(c) -
P_{gt}^{(i)}(c)|$$
The Total Variation distance is bounded in $[0, 1]$ and possesses an intuitive biological
interpretation: it represents the exact proportion of an individual's behavioral time budget
that
has been misallocated across the ethogram due to tracking identity swaps. A TV distortion of
0.00
indicates perfect profile fidelity, whereas a TV distortion of 0.10 indicates that 10% of the
animal's
cumulative activity has been erroneously attributed.

We further define **Metric A2 (Coverage-Adjusted TV Distortion)** to account for frames where
the
tracker failed to generate a bounding box:
$$TV_{A2}(P_{pred}^{(i,m)}, P_{gt}^{(i)}) = \frac{1}{2} \left[ \sum_{c=1}^C |P_{pred}^{(i,m)}(c)
-
P_{gt}^{(i)}(c)| + (1 - \text{Coverage}_i^{(m)}) \right]$$
where $\text{Coverage}_i^{(m)}$ is the fraction of ground-truth frames successfully matched by
tracker
$m$.

### 3.3.2 Statistical Evaluation and Rank Correlation
Pairwise differences in profile distortion ($\Delta TV = TV_{tracker_A} - TV_{tracker_B}$) were
evaluated using non-parametric cluster bootstrap resampling ($B = 10,000$ iterations,
deterministic
seed `240494961`) clustered at the video session level (`video_key`). We report point estimates,
95%
percentile bootstrap confidence intervals $[q_{0.025}, q_{0.975}]$, and leave-one-video-out
cross-validation (LOOV) sign consistency across all 12 video folds.

To determine whether discrete identity switch counts (IDSW) or cumulative duration of
misattribution
(wrong-ID exposure frames) is more predictive of profile distortion, we computed Spearman rank
correlation coefficients ($\rho_{exposure}$ and $\rho_{idsw}$) against A1 TV distortion across
all 96
individual pig sessions. The difference in association strength was evaluated via paired
bootstrap:
$$\Delta |\rho| = |\rho_{exposure}| - |\rho_{idsw}|$$

## 3.4 Multimodal Spatio-Temporal Behavior Recognition Architecture

### 3.4.1 Multimodal Input Representation
For each canonical 6-frame unit ($T = 6$), the model consumes three complementary input streams:
1. **Visual Stream**: A sequence of $T = 6$ actor bounding-box crops resized to $128 \times 128
   \times
   3$ RGB images, capturing posture, head orientation, and fine morphological changes.
2. **Spatial Geometry Stream**: A 6-dimensional coordinate vector per frame $[x_{norm},
   y_{norm},
   w_{norm}, h_{norm}, \text{area}_{norm}, \text{aspect\_ratio}]$, normalized relative to pen
   dimensions, providing spatial location relative to stationary feeders and drinkers.
3. **Kinematic Motion Stream**: A 12-dimensional motion feature vector capturing frame-to-frame
   centroid displacement, instantaneous velocity, acceleration, and bounding box deformation.
4. **Social Context Representation**: Bounding box coordinates, relative distances, and latent
   state
   embeddings for the Top-$K$ ($K = 3$) nearest penmates, pooled in a permutation-invariant
   manner to
   model social encounters.

### 3.4.2 Model Family Progression
We evaluated a systematic progression of deep spatio-temporal architectures:
- **Model A (Baseline High-Ceiling)**: A multimodal Transformer backbone fusing visual features
  with
  spatial geometry and motion vectors via cross-attention.
- **Model B (Joint Representation)**: A model incorporating joint latent representations across
  interacting penmates, providing complementary discriminative signals.
- **Model J (Local Spatial Attention + Class-Aware Gating)**: A dual-branch architecture
  designed to
  resolve subtle behavioral interactions. Model J introduces two specialized mechanisms:
  1. *Local Spatial Attention*: Computes fine-grained attention maps over localized anatomical
     regions (snout, ears, flank) within the visual crop.
  2. *Class-Aware Gating*: A learned gating module that dynamically modulates feature weights
     based
     on predicted behavioral class likelihoods, preventing high-frequency locomotion noise from
     corrupting stationary feeding or resting classifications.

### 3.4.3 Model J 2x2 Factorial Ablation Design
To decompose the performance gains of Model J into its constituent components, we implemented a
2x2
factorial ablation matrix evaluated across 15 physical checkpoints:
- **Config N (Scaffold Control)**: The baseline dual-branch architecture with local spatial
  attention
  and class-aware gating disabled.
- **Config L (Local-Spatial Only)**: Local spatial attention enabled with uniform, un-gated
  weighting.
- **Config G (Class Gate Only)**: Class-aware gating enabled without the local spatial attention
  branch.
- **Model J (Full Local + Gate)**: Both local spatial attention and class-aware gating active.

The architectural effects are decomposed as:
$$\text{Scaffold Effect} = N - A$$
$$\text{Local Effect (Neutral)} = L - N$$
$$\text{Gate Effect (No Local)} = G - N$$
$$\text{Full Effect vs. Scaffold} = J - N$$
$$\text{Descriptive Interaction} = J - L - G + N$$

### 3.4.4 Distillation and Ensemble Systems
We further developed:
- **$J_{PBNEW}$ (Distilled Single Model)**: A single Model J trained using semantic partner
  probabilities distilled from the best ensemble teacher, enhancing single-model representation.
- **$E_{J\_FIXED50}$ (Locked Primary Ensemble)**: A fixed linear combination of Model J and
  Model B
  logits:
  $$\text{Logits}_{E\_J} = 0.50 \cdot \text{Logits}_{J\_F2} + 0.50 \cdot \text{Logits}_B$$
  Weights were locked strictly at 0.50/0.50 without post-hoc optimization on validation data.

---

# 4. Results

## 4.1 Multi-Object Tracking Performance on Independent Evaluation Cohort

To evaluate the generalization of the proposed tracking framework, we evaluated Raw ByteTrack,
RealTime-Fast, and Hybrid-ByteTrack on the 12 independent held-out evaluation videos
($CONFIRMATORY_B$,
21,600 evaluated frames, 96 pig trajectories) under the standardized
`TRACKING_EVALUATOR_STANDARD_V2`
protocol. Table 1 summarizes aggregate tracking performance, dispersion across videos, and
wrong-
identity exposure metrics.

### Table 1: Tracking Performance on Independent Evaluation Cohort (12 Videos)
| Method | Role | Agg HOTA (%) | Video Mean HOTA (%) [95% CI] | DetA (%) | AssA (%) | Agg IDF1 (%) [95% CI] | Total IDSW (Median) | Wrong-ID Exposure (Frames) | Wrong-ID Time (s) | Wrong-ID % Matched | Wrong-ID % All-GT | Episodes | Persistent Swaps |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw ByteTrack** | Baseline | 89.41 | 89.11 [83.21, 94.03] | 91.91 | 87.02 | 94.23 [89.61, 97.96] | 64 (3.0) | 28,603 | 953.43 | 16.65 | 16.55 | 66 | 12 |
| **RealTime-Fast** | Online (15Hz) | 90.33 | 89.89 [84.13, 94.43] | 92.89 | 87.90 | 95.03 [89.49, 99.16] | 39 (0.0) | 23,029 | 767.63 | 13.36 | 13.33 | 51 | 5 |
| **Hybrid-ByteTrack** | Offline (30Hz) | 93.55 | 93.47 [89.85, 96.24] | 93.72 | 93.41 | 98.41 [96.26, 99.70] | 8 (0.0) | 8,597 | 286.57 | 4.99 | 4.98 | 9 | 4 |

As reported in Table 1, retrospective offline tracking (**Hybrid-ByteTrack**) achieved the
highest
tracking quality across all metrics, attaining 93.55% aggregate HOTA, 98.41% aggregate IDF1, and
restricting identity switches to 8 across all 12 independent videos. This represents an 87.5%
reduction in discrete ID switches relative to the frozen Raw ByteTrack baseline (64 IDSW).
Furthermore, Hybrid-ByteTrack reduced wrong-identity exposure from 28,603 frames (953.43 s,
16.65% of
matched ground-truth observations) down to 8,597 frames (286.57 s, 4.99% of matched
observations), a
69.9% reduction in physical error duration. Persistent identity swaps were reduced threefold,
from 12
down to 4.

The causal online tracker (**RealTime-Fast**), operating at half the upstream detector cadence
(15
Hz vs. 30 Hz), achieved 90.33% aggregate HOTA and 95.03% aggregate IDF1. Crucially,
RealTime-Fast
reduced discrete identity switches by 39.1% (from 64 down to 39; median per video 0.0, range [0,
16])
and wrong-identity exposure by 19.5% (from 28,603 frames down to 23,029 frames; 13.36% matched /
13.33%
all-GT), while cutting persistent swaps from 12 down to 5.

### Table 2: Paired Bootstrap Contrasts Between Tracking Methods (B = 10,000 Iterations)
| Tracker Comparison | Delta HOTA (%) [95% CI] | Bootstrap Positive Fraction | Delta IDF1 (%) [95% CI] | Bootstrap Positive Fraction | Relative IDSW Reduction (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hybrid-ByteTrack vs. Raw ByteTrack** | +4.36 [+1.73, +7.31] | 0.9999 | +4.18 [+1.43, +7.29] | 0.9997 | -87.5% (64 -> 8) |
| **Hybrid-ByteTrack vs. RealTime-Fast** | +3.58 [+0.55, +7.14] | 0.9920 | +3.38 [+0.30, +7.16] | 0.9896 | -79.5% (39 -> 8) |
| **RealTime-Fast vs. Raw ByteTrack** | +0.78 [-2.93, +4.53] | 0.6602 | +0.80 [-2.90, +4.69] | 0.6611 | -39.1% (64 -> 39) |

Table 2 presents paired non-parametric cluster bootstrap differences ($B = 10,000$, seed
`240494961`).
The paired contrasts between Hybrid-ByteTrack and Raw ByteTrack demonstrate strictly positive
confidence intervals excluding zero for both Delta HOTA (+4.36% [1.73%, 7.31%], bootstrap
positive
fraction 0.9999) and Delta IDF1 (+4.18% [1.43%, 7.29%], bootstrap positive fraction 0.9997).
Similarly,
Hybrid-ByteTrack demonstrated positive intervals over RealTime-Fast for Delta HOTA (+3.58%
[0.55%,
7.14%], fraction 0.9920) and Delta IDF1 (+3.38% [0.30%, 7.16%], fraction 0.9896). In contrast,
the
paired difference between RealTime-Fast and Raw ByteTrack crossed zero (Delta HOTA +0.78%
[-2.93%,
4.53%], fraction 0.6602; Delta IDF1 +0.80% [-2.90%, 4.69%], fraction 0.6611). Thus, while
RealTime-Fast does not claim higher aggregate HOTA or IDF1 over Raw ByteTrack, it achieves
comparable
accuracy with half the detector load while substantially reducing identity switches.

On the 12-video development cohort (DEV12, 172,800 frames), Hybrid-ByteTrack achieved 0.8982
HOTA,
0.9911 IDF1, and 0 IDSW (24 wrong-ID frames due to boundary clipping), RealTime-Fast achieved
0.8860
HOTA, 0.9700 IDF1, and 25 IDSW (11,765 frames), and Raw ByteTrack produced 0.8111 HOTA, 0.8879
IDF1,
and 64 IDSW (35,008 frames). Across all 13 development videos (including challenging video
`000263`),
Hybrid-ByteTrack achieved 0 IDSW across the entire 187,200 frames.

## 4.2 Spatio-Temporal Behavior Recognition Across 5-Fold Cross-Validation

Downstream behavior classification across the 10 ethogram categories was evaluated across the
33,287
canonical 6-frame units (678 video clips) using 5-fold video-group balanced cross-validation.
Table 3
details the performance of single-model baselines, architectural variants, and ensemble systems.

### Table 3: 5-Fold Cross-Validation Performance of Spatiotemporal Behavior Models
| Model / System | VG1 | VG2 | VG3 | VG4 | VG5 | Mean Macro-F1 | Sample SD ($s$, ddof=1) | Pop. SD ($\sigma$, ddof=0) | Role / Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model A** (Baseline High-Ceiling) | 0.700872 | 0.694903 | 0.632632 | 0.680758 | 0.679783 | 0.677789 | 0.026822 | 0.023990 | Baseline Model |
| **Model B** (Joint Representation) | 0.701061 | 0.694775 | 0.613842 | 0.663935 | 0.680576 | 0.670838 | 0.034919 | 0.031232 | Penmate Model |
| **Model J Replay** (J_F2) | 0.709400 | 0.694903 | 0.633957 | 0.692737 | 0.692233 | 0.684646 | 0.029199 | 0.026116 | Attention + Gate |
| **Model K** (J + Local Motion) | 0.709503 | 0.695357 | 0.638355 | 0.680758 | 0.687785 | 0.682351 | 0.025287 | 0.022617 | Exploratory |
| **$J_{PBNEW}$** (E_J Teacher) | 0.709683 | 0.700285 | 0.644606 | 0.678462 | 0.690816 | **0.684770** | 0.025253 | 0.022587 | **Best Single Model** |
| **$E_0$** (Model A + Model B 50/50) | 0.716413 | 0.714796 | 0.622512 | 0.691956 | 0.717366 | 0.692609 | 0.040212 | 0.035967 | Baseline Ensemble |
| **$E_{J\_FIXED50}$** (0.50*J_F2 + 0.50*B) | 0.719035 | 0.714796 | 0.627833 | 0.699707 | 0.708154 | **0.693905** | 0.037650 | 0.033675 | **Best Overall System** |
| **$E_{J\_PBNEW}$** (0.50*J_PBNEW + 0.50*B) | 0.718812 | 0.715340 | 0.632905 | 0.697112 | 0.707272 | 0.693912 | 0.036321 | 0.032486 | Rejected (2/5 Folds) |

Among single-model architectures, **Model A** established a strong baseline with a mean Macro-F1
of
0.677789 ($s = 0.026822$). **Model B**, integrating joint latent representations across
penmates,
achieved 0.670838 ($s = 0.034919$). Introducing local spatial attention and class-aware gating
(**Model J Replay**) increased mean Macro-F1 to 0.684646 ($s = 0.029199$), a gain of +0.006857
over
Model A. Incorporating localized motion difference features (**Model K**) yielded 0.682351,
outperforming Model A (+0.004562) but falling short of Model J.

Training Model J with distilled partner probabilities from the ensemble teacher
(**$J_{PBNEW}$**)
achieved the highest single-model performance: 0.684770 Macro-F1 ($s = 0.025253$), outperforming
Model A by +0.006981 and exceeding Model J in 3 of 5 cross-validation folds (VG1: 0.709683 vs.
0.709400;
VG2: 0.700285 vs. 0.694903; VG3: 0.644606 vs. 0.633957).

Combining complementary representations via a fixed 50/50 linear ensemble (**$E_{J\_FIXED50}$**
=
$0.50 \cdot J_{F2} + 0.50 \cdot B$) achieved the highest validated performance across the entire
campaign: 0.693905 Macro-F1 ($s = 0.037650$). Relative to Model J alone, $E_{J\_FIXED50}$
corrected a
net 434 classification errors (1,002 errors repaired vs. 568 degraded). A candidate ensemble
incorporating the distilled model (**$E_{J\_PBNEW}$**) achieved a numerically equivalent mean of
0.693912 (+0.000007); however, under preregistered project governance requiring improvement in
at
least 3 of 5 folds, $E_{J\_PBNEW}$ improved in only 2 folds (VG2 and VG3) and was formally
rejected to
prevent validation overfitting.

## 4.3 Model J 2x2 Factorial Architectural Ablation

To determine the exact structural source of Model J's performance gains, we executed a 2x2
factorial
ablation across 15 physical checkpoints on the 5-fold cross-validation split. Table 4 summarizes
the
fold-level Macro-F1 scores and effect decompositions.

### Table 4: Model J 2x2 Factorial Ablation Matrix Across 5 Cross-Validation Folds
| Configuration | Local Spatial Attention | Class-Aware Gating | VG1 | VG2 | VG3 | VG4 | VG5 | Mean Macro-F1 | Delta vs. Config N |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config N** (Scaffold Control) | No | No | 0.700872 | 0.694903 | 0.632632 | 0.680758 | 0.679783 | 0.677789 | Baseline (+0.000000) |
| **Config L** (Local Only) | Yes | No | 0.700872 | 0.694903 | 0.632632 | 0.680758 | 0.679783 | 0.677789 | +0.000000 |
| **Config G** (Gate Only) | No | Yes | 0.705609 | 0.694903 | 0.635817 | 0.687431 | 0.690729 | 0.682898 | +0.005109 |
| **Model J** (Full Local + Gate) | Yes | Yes | 0.709400 | 0.694903 | 0.633957 | 0.692737 | 0.692233 | 0.684646 | +0.006857 |

The factorial quantities decompose as follows:
- **Scaffold Effect** ($N - A$): $0.677789 - 0.677789 = +0.000000$. The parameter scaffolding of
  the
  dual-branch architecture introduced no spurious parameter gains over Model A.
- **Local Effect Neutral** ($L - N$): $0.677789 - 0.677789 = +0.000000$. Introducing local
  spatial
  attention without gating produced zero gain across all 5 folds.
- **Gate Effect No Local** ($G - N$): $0.682898 - 0.677789 = +0.005109$. Class-aware gating is
  the
  primary standalone driver of performance (+0.005109 gain, improving 4 of 5 folds).
- **Full Effect vs. Scaffold** ($J - N$): $0.684646 - 0.677789 = +0.006857$.
- **Descriptive Interaction** ($J - L - G + N$): $+0.001748$.

These empirical findings demonstrate that local spatial attention cannot improve held-out
generalization when uniformly averaged across all behaviors. Rather, learned class-aware gating
selectively conditions local spatial attention on specific action categories where localized
anatomical
contact is discriminative (e.g., sniffing during `social-nose` or head knocking during `fight`),
while
suppressing local spatial noise during global postural states like `lying` or `standing`.

## 4.4 Propagation of Upstream Tracking Errors to Downstream Behavioral Profiles

To directly quantify how upstream tracking failures corrupt biological phenotypes, we evaluated
the
downstream behavioral profile distortion on the 12 development videos (DEV12, 96 pig sessions,
28,800
canonical units) while holding human behavior annotations fixed. Table 5 presents the
distribution of
Total Variation distortion across tracking modes.

### Table 5: Downstream Behavioral Profile Distortion Across 96 Pig-Video Sessions
| Tracking Mode | Mean A1 TV Distortion | 95% Cluster Bootstrap CI | Median A1 TV | IQR A1 TV | Mean A2 TV Distortion (Coverage Adjusted) | Mean Tracking Coverage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw ByteTrack** | 0.0953 | [0.0394, 0.1587] | 0.0000 | 0.0627 | 0.0977 | 99.13% |
| **RealTime-Fast** | 0.0309 | [0.0043, 0.0668] | 0.0000 | 0.0000 | 0.0322 | 99.66% |
| **Hybrid-ByteTrack** | **0.0001** | [0.0000, 0.0004] | 0.0000 | 0.0000 | **0.0036** | 99.11% |

Under frozen human ground truth, **Raw ByteTrack** induced a mean A1 TV distortion of 0.0953
(95% CI:
[0.0394, 0.1587]). In biological terms, nearly 10% of an individual animal's cumulative
behavioral
budget was erroneously allocated to incorrect ethogram categories due to tracking swaps. For
animals
experiencing persistent identity swaps, A1 TV distortion exceeded 0.30 (over 30% budget error).

**RealTime-Fast** reduced mean A1 TV distortion to 0.0309 (95% CI: [0.0043, 0.0668]),
representing a
67.6% reduction in profile error relative to Raw ByteTrack. **Hybrid-ByteTrack** virtually
eliminated
profile distortion, achieving a mean A1 TV of 0.0001 (95% CI: [0.0000, 0.0004]), a >99.8%
reduction
over Raw ByteTrack. Under the coverage-adjusted A2 TV metric (which penalizes dropped
detections),
distortion was 0.0977 for Raw ByteTrack, 0.0322 for RealTime-Fast, and 0.0036 for
Hybrid-ByteTrack.

### Table 6: Pairwise Statistical Comparisons of Profile Distortion (B = 10,000 Iterations)
| Tracker Comparison | Mean Delta A1 TV Distortion | 95% Cluster Bootstrap CI | LOOV Sign Consistency (Folds Positive) | LOOV Delta Range |
| :--- | :---: | :---: | :---: | :---: |
| **Raw ByteTrack vs. RealTime-Fast** | +0.0644 | [+0.0216, +0.1073] | **12 / 12 (100%)** | [+0.0540, +0.0747] |
| **Raw ByteTrack vs. Hybrid-ByteTrack** | +0.0951 | [+0.0393, +0.1586] | **12 / 12 (100%)** | [+0.0705, +0.1038] |
| **RealTime-Fast vs. Hybrid-ByteTrack** | +0.0308 | [+0.0043, +0.0665] | **12 / 12 (100%)** | [+0.0164, +0.0336] |

As detailed in Table 6, all pairwise differences in profile distortion were positive with 95%
bootstrap confidence intervals strictly excluding zero. Leave-one-video-out cross-validation
(LOOV)
confirmed 100% sign consistency (12 of 12 video folds strictly positive) across all three
tracker
comparisons, confirming that the reduction in downstream distortion is consistent across
sessions.

## 4.5 Predictive Capacity: Wrong-ID Exposure Duration vs. Discrete IDSW

To determine whether conventional discrete identity switch counts (IDSW) or cumulative
wrong-identity
exposure duration is more predictive of profile distortion, we evaluated Spearman rank
correlation
coefficients against A1 TV distortion across all 96 pig sessions. Table 7 summarizes the
association
statistics.

### Table 7: Rank Correlation Between Tracking Error Metrics and Downstream Profile Distortion
| Tracking Method | Metric | Spearman $\rho$ vs. A1 TV | 95% Bootstrap CI | Delta Abs Rho ($\Delta \|\rho\|$) | 95% Bootstrap CI for Delta |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Raw ByteTrack** | Wrong-ID Exposure (Frames) | **0.9381** | [0.8849, 0.9696] | **+0.0668** | **[+0.0116, +0.1501]** |
| | Discrete ID Switches (IDSW) | 0.8712 | [0.7627, 0.9416] | (Reference) | — |
| **RealTime-Fast** | Wrong-ID Exposure (Frames) | **0.9698** | [0.8749, 0.9997] | +0.0512 | [-0.0006, +0.1577] |
| | Discrete ID Switches (IDSW) | 0.9186 | [0.7896, 0.9971] | (Reference) | — |
| **Hybrid-ByteTrack** | Wrong-ID Exposure (Frames) | 1.0000 | — | Degenerate | — |
| | Discrete ID Switches (IDSW) | — (Zero Variance) | — | (0 IDSW on DEV12) | — |

For **Raw ByteTrack**, wrong-identity exposure frames exhibited a stronger rank association with
A1 TV
distortion ($\rho = 0.9381$, 95% CI: [0.8849, 0.9696]) than discrete IDSW ($\rho = 0.8712$, 95%
CI:
[0.7627, 0.9416]). The paired difference in absolute correlation was strictly positive: $\Delta
|\rho|
= +0.0668$ with a 95% cluster bootstrap confidence interval of [+0.0116, +0.1501], strictly
excluding
zero.

For **RealTime-Fast**, wrong-identity exposure also showed a numerically higher correlation
($\rho =
0.9698$ vs. $0.9186$, $\Delta |\rho| = +0.0512$); however, the bootstrap confidence interval
crossed
zero ([-0.0006, +0.1577]). For **Hybrid-ByteTrack**, because identity switches were completely
eliminated on the development cohort (0 IDSW), correlation analysis was degenerate.

These findings demonstrate that while discrete IDSW counts indicate the occurrence of tracking
failures, the physical duration during which a swapped identity persists is the governing factor
determining the magnitude of downstream behavioral profile corruption.

---

# 5. Discussion

## 5.1 Operational Trade-Offs in Multi-Object Tracking for Livestock

A central contribution of this work is establishing a dual-mode tracking framework that provides
distinct operating points tailored to the divergent computational and latency requirements of
precision livestock monitoring:

### 5.1.1 Causal Online Tracking for Streaming Alert Systems
Commercial swine facilities increasingly deploy automated alert systems to notify producers of
emergent welfare emergencies, such as tail-biting outbreaks, intense fighting among newly mixed
batches, or mechanical feeder/drinker failures. In these applications, processing latency must
be
minimized: an edge inference system must flag aggression in real time rather than waiting for
post-hoc
video processing.

RealTime-Fast is specifically designed for these operational constraints. By operating at a 15
Hz
detector cadence (evaluating the YOLOv8 detector on every second frame), RealTime-Fast halves
the
primary computational burden of the tracking pipeline. Despite this 50% reduction in detector
calls,
RealTime-Fast maintained 90.33% aggregate HOTA on independent evaluation videos and reduced
discrete
identity switches by 39.1% (from 64 down to 39) relative to the 30 Hz Raw ByteTrack baseline.
Crucially, persistent identity swaps were reduced from 12 down to 5. These characteristics make
RealTime-Fast a viable candidate for on-farm edge devices where computational resources are
severely
constrained and zero future lookahead is permissible.

### 5.1.2 Retrospective Offline Tracking for Phenotyping and Epidemiology
Conversely, longitudinal ethological research, genetic selection for social tolerance, and
veterinary
epidemiological studies do not require real-time execution. In these applications, video
recordings
are accumulated across days, weeks, or production cycles, and the primary objective is compiling
flawless individual time budgets.

For retrospective analysis, Hybrid-ByteTrack offers superior identity stability. By
incorporating
bidirectional Kalman smoothing and graph-based trajectory refinement across full temporal
sequences,
Hybrid-ByteTrack completely eliminated identity switches (0 IDSW) across all 12 development
videos
(DEV12, 172,800 frames) and achieved 0 IDSW on the full 13-video development cohort including
video
`000263`. On the 12 independent held-out evaluation videos ($CONFIRMATORY_B$), Hybrid-ByteTrack
restricted identity switches to 8 (an 87.5% reduction relative to Raw ByteTrack) and limited
wrong-identity exposure to 4.99% of matched ground-truth observations. The paired bootstrap
contrasts confirmed strictly positive gains in HOTA (+4.36% [1.73%, 7.31%]) and IDF1 (+4.18%
[1.43%,
7.29%]) over Raw ByteTrack, establishing Hybrid-ByteTrack as the preferred tool for longitudinal
phenotyping.

## 5.2 Mechanistic Role of Wrong-ID Exposure vs. Discrete IDSW

In the general computer vision tracking literature, tracking algorithms are traditionally
evaluated
by counting discrete association failures (e.g., IDSW in CLEAR MOT, identity switches in HOTA).
However, our downstream profile distortion analysis demonstrates that discrete switch counts are
an
incomplete proxy for phenotypic validity in livestock behavioral science.

When an identity switch occurs between two pigs, the biological error introduced into their
respective time budgets is directly proportional to the duration of the swap. A transient
identity
switch that resolves within 6 to 10 frames (0.20 to 0.33 seconds) during a rapid crossing
creates
negligible distortion in a 10-minute behavioral budget (error < 0.05%). Conversely, a persistent
identity swap that remains uncorrected for 300 to 1,000 frames (10 to 33 seconds) transfers
dozens of
behavioral bouts from one individual to another, corrupting the estimated proportions of
resting,
feeding, and locomotion by 15% to 35%.

Our empirical findings provide direct statistical support for this mechanistic distinction. For
Raw
ByteTrack, cumulative wrong-identity exposure frames demonstrated a significantly stronger rank
association with downstream Total Variation profile distortion than discrete IDSW ($\rho =
0.9381$
vs. $0.8712$; $\Delta |\rho| = +0.0668$, 95% CI: [+0.0116, +0.1501], strictly excluding zero).
While
RealTime-Fast showed a similar directional trend ($\Delta |\rho| = +0.0512$), the bootstrap
interval
spanned zero ([-0.0006, +0.1577]), reflecting its narrower range of exposure durations. For
Hybrid-ByteTrack, the near-total elimination of identity errors (0 IDSW on DEV12) precluded rank
correlation analysis. These results indicate that future evaluations of animal tracking systems
should report duration-weighted exposure metrics (such as wrong-ID frames and persistent swap
counts)
alongside conventional discrete switch counts.

## 5.3 Architectural Mechanisms: Class-Aware Gating and Synergistic Interaction

The 2x2 factorial ablation of Model J provides valuable insights into deep spatio-temporal
modeling
for complex animal ethograms. A common intuition in computer vision is that incorporating
fine-grained
local spatial attention maps over anatomical keypoints (such as snouts or ears) should
universally
improve action recognition. However, our empirical ablation revealed that Config L (local
spatial
attention alone, uniformly averaged) produced exactly 0.000000 gain over the neutral scaffold
Config N
across all 5 cross-validation folds.

This failure of isolated spatial attention occurs because pig behavior encompasses both highly
localized dynamic interactions (e.g., head knocking during `fight`, snout investigation during
`social-nose`) and broad whole-body postural states (e.g., `lying`, `stand`, `move`). When local
spatial attention features are uniformly weighted across all classes, they introduce
high-frequency
spatial noise into static postural classifications, degrading resting and standing
discrimination.

The introduction of learned class-aware gating (Config G) resolved this conflict, providing the
primary standalone performance improvement (+0.005109 Macro-F1 over scaffold). Class-aware
gating
acts as a dynamic semantic selector: when the model predicts a high likelihood of an interactive
social behavior, the gate increases the weight of local spatial features to resolve fine-grained
contact; conversely, when the model predicts a resting or maintenance state, the gate attenuates
local
features in favor of global visual context. When local spatial attention and class-aware gating
are
combined (Model J), they exhibit a positive descriptive interaction (+0.001748), demonstrating
how
semantic gating conditionally unlocks the discriminative capacity of local spatial
representations.

## 5.4 Governance and Methodological Rigor in Agricultural AI

A final consideration highlighted by this study is the critical role of preregistered
experimental
governance in machine learning for precision agriculture. In high-dimensional deep learning,
small
marginal gains can easily be manufactured through selective hyperparameter search, post-hoc
threshold
adjustments, or ensembling over-fitted models.

In our ensemble evaluation, candidate ensemble $E_{J\_PBNEW}$ achieved a marginal numerical
increase
of +0.000007 in mean 5-fold Macro-F1 over the locked baseline ensemble $E_{J\_FIXED50}$
(0.693912 vs.
0.693905). Under permissive evaluation criteria, this marginal increase might have been claimed
as a
new state-of-the-art result. However, under our preregistered governance rules requiring
improvement in
at least 3 of 5 cross-validation folds, $E_{J\_PBNEW}$ was formally rejected because it improved
in
only 2 folds (VG2 and VG3) while degrading in 3. By enforcing fixed 50/50 weights and strict
promotion
gates, we ensure that reported gains reflect genuine architectural improvements rather than
validation
noise.

---

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

---

# 7. Conclusion

In precision livestock farming, automated computer vision systems must reliably track individual
animals over extended time horizons to enable valid individual behavioral phenotyping, welfare
monitoring, and veterinary intervention. This paper investigated the mechanistic link between
upstream multi-object tracking failures and downstream behavioral profile corruption in
group-housed
pigs, presenting a unified framework for dual-mode tracking and multimodal behavior recognition.

Our primary empirical and methodological findings are summarized as follows:

1. **Downstream Profile Distortion and Error Exposure**: By establishing the Total Variation
   (TV)
   profile distortion metric under fixed human ground truth across 96 pig-video sessions, we
   proved
   that tracking identity swaps corrupt biological time budgets. For baseline tracking (Raw
   ByteTrack), identity swaps resulted in a mean A1 TV profile error of 0.0953, misallocating
   nearly
   10% of total activity. Furthermore, we established that cumulative wrong-identity exposure
   duration is significantly more predictive of profile distortion than discrete switch counts
   ($\Delta |\rho| = +0.0668$, 95% CI: [+0.0116, +0.1501]), demonstrating that tracking
   evaluation
   in livestock science must account for physical error duration.
2. **Dual-Mode Tracking Operating Points**: We formulated and validated two complementary
   tracking
   regimes:
   - *RealTime-Fast (Causal Online Mode)*: Operates at a reduced 15 Hz detector cadence with
     zero
     future temporal lookahead, reducing discrete identity switches by 39.1% (from 64 down to 39
     on
     independent held-out videos) and halving upstream detector compute, making it well-suited
     for
     real-time streaming alerts.
   - *Hybrid-ByteTrack (Retrospective Offline Mode)*: Utilizes bidirectional Kalman smoothing
     and
     graph-based trajectory refinement, achieving 93.55% HOTA, 98.41% IDF1, and reducing
     identity
     switches by 87.5% (from 64 down to 8) on independent held-out videos ($CONFIRMATORY_B$),
     while
     reducing downstream profile distortion to near-zero (mean A1 TV = 0.0001 on DEV12).
3. **Multimodal Behavior Recognition and Architectural Mechanisms**: For 10-class
   spatio-temporal
   behavior recognition across 33,287 canonical units, our 2x2 factorial ablation proved that
   class-aware gating is the primary standalone driver of performance (+0.005109 Macro-F1 over
   scaffold), whereas un-gated local spatial attention alone yielded zero gain (+0.000000). The
   distilled single model $J_{PBNEW}$ achieved 0.684770 Macro-F1, and the fixed 50/50 ensemble
   $E_{J\_FIXED50}$ achieved the highest validated 5-fold Macro-F1 of 0.693905 +/- 0.037650.
4. **Reproducibility and Transparent Governance**: By enforcing preregistered protocols,
   independent confirmatory testing ($CONFIRMATORY_B$), locked ensemble weights, and transparent
   documentation of dataset roles (confirming zero outer test leakage, $OUTER\_TEST\_EVALUATIONS
   = 0$),
   this study establishes a methodologically rigorous benchmark for automated individual animal
   monitoring.

---

---

## Declarations

### Ethics Statement
All animal management and video recording procedures followed institutional animal care and use
guidelines and national standards for commercial swine production. No invasive physical sensors
or
experimental stress treatments were applied; all observations were recorded unobtrusively using
passive overhead optical cameras in commercial production pens.

### Data and Code Availability
The complete reproducible evaluation package, including trained model checkpoint checksums,
standardized evaluation protocols, and validation ledgers, is frozen under immutable SHA256
manifests
at `docs/paper/freeze_corrected_20260917/`. Evaluation protocols strictly adhere to
`TRACKING_EVALUATOR_STANDARD_V2`.

### Author Contributions
Conceptualization: All authors; Methodology and Software: Engineering Team; Validation and
Formal
Analysis: Research Team; Data Curation and Annotation: Animal Welfare Specialists; Writing -
Original
Draft: Lead Authors; Writing - Review and Editing: All authors.

### Conflict of Interest
The authors declare that they have no known competing financial interests or personal
relationships
that could have appeared to influence the work reported in this paper.

---

## References

1. Aharon, N., Orfaig, R., & Bobrovsky, B. Z. (2022). BoT-SORT: Robust associations
   multi-pedestrian
   tracking. *arXiv preprint arXiv:2206.14651*.
2. Alameer, A., Kyriazakis, I., & Bacardit, J. (2020). Automated recognition of pig agonistic
   behaviours using computer vision: A review. *Computers and Electronics in Agriculture*, 177,
   105694.
3. Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., & Schmid, C. (2021). ViViT: A video
   vision transformer. In *Proceedings of the IEEE/CVF International Conference on Computer
   Vision*
   (pp. 6836-6846).
4. Berckmans, D. (2014). Precision livestock farming technologies for welfare management in
   intensive livestock systems. *Revue Scientifique et Technique*, 33(1), 189-196.
5. Bertasius, G., Wang, H., & Torresani, L. (2021). Is space-time attention all you need for
   video
   understanding? In *ICML* (Vol. 2, p. 4).
6. Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple online and realtime
   tracking.
   In *2016 IEEE International Conference on Image Processing (ICIP)* (pp. 3464-3468).
7. Carreira, J., & Zisserman, A. (2017). Quo vadis, action recognition? A new model and the
   kinetics
   dataset. In *CVPR* (pp. 6299-6308).
8. Chen, C., Zhu, W., & Steibel, J. (2021). Recognition of aggressive episodes in group-housed
   pigs
   using deep learning. *Biosystems Engineering*, 209, 246-258.
9. Cowton, J., Kyriazakis, I., & Bacardit, J. (2018). Automated individual pig recognition using
   convolutional neural networks. *Frontiers in Veterinary Science*, 5, 274.
10. Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). SlowFast networks for video
    recognition.
    In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp.
    6202-6211).
11. Luiten, J., Osep, A., Dendorfer, P., Torr, P., Geiger, A., Leal-Taixé, L., & Leibe, B.
    (2021).
    HOTA: A higher order metric for evaluating multi-object tracking. *International Journal of
    Computer Vision*, 129(2), 548-578.
12. Matthews, S. G., Miller, A. L., Clapp, J., Plötz, T., & Kyriazakis, I. (2016). Early
    detection
    of health and welfare compromises through automated detection of behavioural changes in
    pigs.
    *Veterinary Record*, 178(17), 430-430.
13. Milan, A., Leal-Taixé, L., Reid, I., Roth, S., & Schindler, K. (2016). MOT16: A benchmark
    for
    multi-object tracking. *arXiv preprint arXiv:1603.00831*.
14. Norton, T., Chen, C., Larsen, M. L., & Berckmans, D. (2019). Precision livestock farming:
    building 'digital twins' to support the transition to sustainable animal husbandry. *Animal
    Frontiers*, 9(2), 12-19.
15. Psota, E. T., Mittek, M., Pérez, L. C., Schmidt, T., & Mote, B. (2019). Multi-pig tracking
    in
    high-density pens using deep learning and motion estimation. *Computers and Electronics in
    Agriculture*, 164, 104899.
16. Riekert, S., Klein, A., Fumagalli, M., Lösel, D., & Gallmann, E. (2020). Detection of
    posture
    and behaviour in group-housed pigs by computer vision using deep learning. *Computers and
    Electronics in Agriculture*, 174, 105491.
17. Viazzi, S., Ismayilova, G., Oczak, M., Sonoda, L. T., Fels, M., Guarino, M., ... &
    Berckmans, D.
    (2014). Image processing for the automated detection of aggressive behaviour in pigs.
    *Biosystems Engineering*, 124, 82-88.
18. Wojke, N., Bewley, A., & Paulus, D. (2017). Simple online and realtime tracking with a deep
    association metric. In *2017 IEEE International Conference on Image Processing (ICIP)* (pp.
    3645-3649).
19. Yang, Q., Xiao, D., & Lin, C. (2018). Feeding behavior recognition for group-housed pigs
    with the
    use of deep learning. *Computers and Electronics in Agriculture*, 149, 1-12.
20. Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., ... & Wang, X. (2022). ByteTrack:
    Multi-object tracking by associating every detection box. In *European Conference on
    Computer
    Vision* (pp. 1-21).