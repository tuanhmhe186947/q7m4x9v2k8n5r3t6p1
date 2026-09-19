# Group-Aware Behavior Model Uncertainty Analysis

This document reports video-cluster non-parametric bootstrap uncertainty ($B=10,000$, seed
$240494961$) across spatiotemporal behavior models.

## 1. Summary of Pairwise Model Differences

| Comparison | Original $\Delta$ | Bootstrap Mean | 95% Bootstrap CI | $P(\Delta > 0)$ | Zero Included? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A vs. Model J** | +0.006856 | +0.007062 | [-0.003694, +0.018904] | 91.1% | **YES** |
| **Model A vs. E_J Fixed 50/50** | +0.016115 | +0.017965 | [+0.002627, +0.036671] | 98.8% | **NO (Excludes 0)** |
| **Model J vs. E_J Fixed 50/50** | +0.009259 | +0.010903 | [-0.002694, +0.027286] | 94.0% | **YES** |
| **Model J vs. J_PBNEW** | +0.000124 | +0.001086 | [-0.008222, +0.010703] | 55.2% | **YES** |

## 2. Scientific Interpretation
1. **Model A vs. E_J Fixed 50/50**: The ensemble system achieves a robust, statistically
   verifiable improvement over baseline Model A. The 95% bootstrap confidence interval
   ($[+0.0026, +0.0367]$) strictly excludes zero, with 98.8% of bootstrap iterations
   favoring E_J.
2. **Model A vs. Model J**: Model J exhibits a positive point estimate (+0.0069) and 91.1%
   of bootstrap replicates show improvement. However, under video-level clustering, the
   95% confidence interval spans zero ($[-0.0037, +0.0189]$). We therefore state
   transparently that while directionally favorable, single-model spatial gating gains
   include zero uncertainty across video clusters.
3. **Model J vs. E_J Fixed 50/50**: Ensembling Model J with Model B produces an additional
   +0.0093 point improvement with 94.0% positive replicates, though the interval
   ($[-0.0027, +0.0273]$) marginally includes zero.
4. **Model J vs. J_PBNEW**: Substituting semantic teacher probabilities yields negligible
   isolated gain (+0.0001 point delta, $P>0 = 55.2\%$, CI spans zero), confirming our
   governance decision to maintain Model J as primary ablated architecture.
