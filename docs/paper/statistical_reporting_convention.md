# Statistical Reporting Conventions for Manuscript

## 1. Central Tendency and Dispersion
- **Mean Across Folds/Videos**:
  $$\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$$
- **Sample Standard Deviation (Primary Journal Convention, ddof=1)**:
  $$s = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2}$$
  Reported by default for cross-validation fold variability and per-video distributions.
- **Population Standard Deviation (ddof=0)**:
  $$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^2}$$
  Documented in technical ledgers where the set of folds is treated as a closed evaluation
cohort.
- **Reporting Notation**:
  Metrics are presented as $\text{Mean} \pm s$ with sample standard deviation ($N=5$ for
behavior CV folds; $N=12$ or $13$ for tracking videos).

## 2. Cluster / Non-Parametric Bootstrap Uncertainty
- **Cluster Level**: Resampling is conducted strictly at the independent session/video
  level (`video_key`). Individual prediction windows or physical frames are never
  bootstrapped independently.
- **Replications**: $B = 10,000$ bootstrap iterations.
- **Deterministic Seed**: `240494961`.
- **Confidence Intervals**: 95% percentile confidence intervals $[q_{0.025}, q_{0.975}]$.
- **Difference Testing**: For pairwise comparisons $\Delta = A - B$, report the point
  estimate on the original dataset, the bootstrap mean difference, the 95% bootstrap CI,
  and the empirical probability $P(\Delta > 0)$. If the 95% interval contains zero, the
  finding is explicitly reported as including zero without claiming statistical
  significance.
