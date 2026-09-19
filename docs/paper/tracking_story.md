# Tracking Narrative: Dual-Mode Multi-Object Tracking for Group-Housed Livestock

Multi-object tracking of group-housed pigs presents severe challenges due to visual
homogeneity, frequent occlusions, and sudden physical interactions. In this work, we
formulate and evaluate a dual-mode tracking framework tailored to the operational
requirements of precision livestock farming:

1. **Causal Online Tracking (RealTime-Fast)**: Designed for low-latency streaming
   environments (e.g. real-time aggression detection or automated sorting gates), this
   mode processes video frame-by-frame without future temporal buffers. On the 12-video
   development cohort (96 pig trajectories), RealTime-Fast achieves an IDF1 of 0.9700 and
   a HOTA score of 0.8860, reducing identity switches from 64 (in the frozen Raw ByteTrack
   baseline) down to 25 (a 60.9% reduction) and wrong-identity exposure from 35,008 frames
   down to 11,765 frames (a 66.4% reduction). Across the full 13-video development cohort,
   it maintains an IDF1 of 0.9719 and HOTA of 0.8882.

2. **Retrospective Offline Tracking (Hybrid-ByteTrack)**: Designed for longitudinal
   behavioral profiling and epidemiological research where complete historical context is
   available, this mode incorporates backward trajectory smoothing, spatial-temporal graph
   refinement, and identity trajectory post-processing. On the 12-video development
   cohort, Hybrid-ByteTrack achieves an IDF1 of 0.9911 and HOTA of 0.8982, completely
   eliminating identity switches (0 IDSW) and reducing wrong-identity exposure to a
   residual 24 frames resulting from an edge boundary occlusion. Across the full 13-video
   development cohort, it achieves an IDF1 of 0.9915, HOTA of 0.9003, and 0 IDSW.

3. **Baseline Comparison**: Compared against a frozen Raw ByteTrack baseline (HOTA 0.8111,
   IDF1 0.8879, 64 IDSW on DEV12), both proposed tracking modes demonstrate substantial
   improvements in identity consistency and track fragmentation. We note that upstream
   detector configurations differ slightly between systems (0.25 confidence threshold and
   32 proposals vs historical 0.20/20 live topology), and we report these differences
   conservatively as whole-pipeline system comparisons. Finally, tracking generalization
   to unseen environments is preserved by strictly holding closed an independent 12-video
   confirmatory cohort under preflight integrity contracts.
