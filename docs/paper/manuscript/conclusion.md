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
