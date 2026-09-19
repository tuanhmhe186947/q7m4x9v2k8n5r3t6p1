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
