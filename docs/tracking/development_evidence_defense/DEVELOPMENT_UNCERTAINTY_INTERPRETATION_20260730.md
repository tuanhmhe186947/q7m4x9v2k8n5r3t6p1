# Development uncertainty interpretation

The resampling unit is the video, never the frame. The 10,000-resample paired
bootstrap uses seed `20260730` and reports descriptive percentile ranges.

These ranges are not claimed as robust confidence intervals because available
authorities do not prove that the 13 videos are independent recording-session
clusters. Leave-one-video-out results are therefore reported alongside every
comparison, and no `p < 0.05` decision is made.

Status: `INSUFFICIENT_INDEPENDENT_CLUSTERS_FOR_RELIABLE_CI`.
