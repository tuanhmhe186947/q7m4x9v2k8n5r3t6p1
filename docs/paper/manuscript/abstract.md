# Abstract

**Title**: Identity-Preserving Multi-Object Tracking and Multimodal Spatio-Temporal Behavior
Recognition for Group-Housed Pigs

**Context**: In precision livestock farming, automated vision systems must monitor individual animal
behavior over extended periods to support animal welfare, feeding management, and disease detection.
However, visual homogeneity, frequent close contact, and body occlusions among group-housed pigs
cause tracking identity switches that misattribute behaviors and distort longitudinal individual
time budgets.

**Objectives**: We investigate the mechanistic propagation of multi-object tracking errors into
downstream individual behavioral profiles and evaluate a dual-mode tracking and multimodal behavior
recognition architecture designed to preserve identity integrity.

**Methods**: We formulate a dual-mode tracking framework comprising a causal online mode
(RealTime-Fast, 15 Hz detector cadence) for low-latency alerting and a retrospective offline mode
(Hybrid-ByteTrack) with bidirectional smoothing and graph-based association for longitudinal
profiling. We establish an empirical evaluation on 12 development videos (96 pig-video sessions,
28,800 canonical 6-frame units) holding human-annotated behavior ground truth fixed to measure Total
Variation (TV) profile distortion. Upstream tracking generalization is assessed on 12 independent
held-out videos under Standard V2 bipartite Hungarian matching (CONFIRMATORY_B). Downstream behavior
recognition is evaluated across 33,287 canonical 6-frame units across 678 videos using a 5-fold
video-group balanced cross-validation scheme.

**Results**: On independent evaluation videos, Hybrid-ByteTrack achieved 93.55% HOTA (95% CI:
[89.85%, 96.24%]), 98.41% IDF1 ([96.26%, 99.70%]), and 8 identity switches (IDSW), reducing IDSW by
87.5% relative to the frozen Raw ByteTrack baseline (64 IDSW; delta HOTA +4.36% [1.73%, 7.31%],
bootstrap positive fraction 0.9999). RealTime-Fast achieved 90.33% HOTA ([84.13%, 94.43%]), 95.03%
IDF1 ([89.49%, 99.16%]), and 39 IDSW (delta HOTA +0.78% [-2.93%, 4.53%], bootstrap positive fraction
0.6602). Downstream profile distortion analysis demonstrated that tracker identity swaps corrupted
individual behavioral time budgets: mean A1 TV distortion was 0.0953 for Raw ByteTrack, 0.0309 for
RealTime-Fast, and 0.0001 for Hybrid-ByteTrack (Raw vs RealTime delta = 0.0644 [0.0216, 0.1073]; Raw
vs Hybrid delta = 0.0951 [0.0393, 0.1586]; all 12/12 leave-one-out folds positive). Wrong-identity
exposure duration was more strongly associated with profile distortion than discrete switch count
for Raw ByteTrack (Spearman rho = 0.9381 vs 0.8712; delta abs rho = 0.0668 [0.0116, 0.1501]). For
spatio-temporal behavior recognition across 10 ethogram categories, a 2x2 factorial ablation
demonstrated that class-aware gating was the primary driver of single-model performance
(+0.005109 Macro-F1 over scaffold), whereas un-gated local spatial attention produced no gain
alone (+0.000000). A fixed 50/50 ensemble of focal and joint representation models achieved
the highest validated 5-fold Macro-F1 of 0.693905 +/- 0.037650.

**Conclusions**: Upstream tracking errors propagate directly into individual livestock behavioral
budgets, where wrong-identity exposure duration governs phenotypic distortion. The proposed
dual-mode tracking framework provides operating points for both causal streaming alerts and
retrospective profiling with near-zero downstream distortion.
