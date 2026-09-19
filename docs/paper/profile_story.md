# Downstream Profile Narrative: Propagation of Identity Errors into Behavioral Time Budgets

While computer vision tracking benchmarks conventionally evaluate bounding-box overlap and
identity switches, the primary scientific objective in livestock phenotyping is
constructing reliable, longitudinal behavior profiles for individual animals (e.g. daily
time budgets allocated to feeding, drinking, exploration, and aggression). A critical
unanswered question is how upstream tracking identity errors propagate into downstream
behavioral profiles.

To isolate this propagation cleanly, we conducted an empirical experiment across 12
tracking-development videos (96 pig sessions, 28,800 canonical 6-frame units) by
evaluating tracker performance while holding human-reviewed behavioral ground truth
completely fixed:

1. **Total Variation Profile Distortion (Primary Metric A1)**: Under frozen human behavior
   annotations, tracker identity swaps cause misattribution of behavioral states among
   pen-mates. Raw ByteTrack produces a substantial mean Total Variation (TV) distortion of
   0.0953 (95% cluster bootstrap CI: [0.0394, 0.1587], median 0.0000, IQR 0.0627).
   RealTime-Fast reduces this distortion to 0.0309 (95% CI: [0.0043, 0.0668], median
   0.0000, IQR 0.0000), representing a 67.6% reduction in profile error. Hybrid-ByteTrack
   achieves near-zero distortion with a mean A1 TV of 0.0001 (95% CI: [0.0000, 0.0004],
   median 0.0000, IQR 0.0000), representing a >99.8% reduction. Under a realistic protocol
   accounting for temporal tracking coverage (Metric A2, coverage >= 99.1%), A2 TV
   distortion is 0.0977 for Raw ByteTrack, 0.0322 for RealTime-Fast, and 0.0036 for
   Hybrid-ByteTrack.

2. **Pairwise Significance and Leave-One-Out Robustness**: All pairwise tracker
   comparisons demonstrate statistically significant reductions in profile distortion:
   - Raw vs. RealTime: Mean delta = +0.0644 (95% CI: [+0.0216, +0.1073], LOOV 12/12
positive).
   - Raw vs. Hybrid: Mean delta = +0.0951 (95% CI: [+0.0393, +0.1586], LOOV 12/12
positive).
   - RealTime vs. Hybrid: Mean delta = +0.0308 (95% CI: [+0.0043, +0.0665], LOOV 12/12
positive).
   In all cases, leave-one-video-out cross-validation confirms 100% sign consistency.

3. **Wrong-ID Exposure vs. Discrete IDSW (Claim B Sanitized)**: Wrong-ID exposure showed a
   stronger association with downstream profile distortion than IDSW for Raw ByteTrack
   (Spearman rho = 0.9381 vs 0.8712; Delta abs rho = +0.0668, 95% CI: [+0.0116, +0.1501],
   strictly excluding zero). RealTime-Fast showed the same directional trend (0.9698 vs
   0.9186; Delta abs rho = +0.0512), although uncertainty in the difference included zero
   (95% CI: [-0.0006, +0.1577]). For Hybrid-ByteTrack, the complete absence of ID switches
   (0 IDSW) renders correlation analysis degenerate. We therefore state that exposure
   frames provide superior predictive value for Raw ByteTrack, with consistent directional
   trends in RealTime-Fast where uncertainty spans zero.
