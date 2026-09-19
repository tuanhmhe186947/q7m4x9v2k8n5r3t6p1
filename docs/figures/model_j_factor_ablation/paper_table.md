# Model J Factorial Ablation Results

## 1. 5-Fold Evaluation Results Table

| Configuration | Trainable Params | VG1 | VG2 | VG3 | VG4 | VG5 | Mean Macro-F1 (ddof=0) | Delta vs A | Delta vs N |
|---|---|---|---|---|---|---|---|---|---|
| **Model A (Canonical Reference)** | 43,633,832 (Base) | 0.700872 | 0.694903 | 0.632632 | 0.680758 | 0.679783 | **0.677789 ± 0.023990** | +0.000000 | +0.000000 |
| **Config N (Neutral Scaffold Control)** | 27,712 (0.06%) | 0.700872 | 0.694903 | 0.632632 | 0.680758 | 0.679783 | **0.677789 ± 0.023990** | +0.000000 | +0.000000 |
| **Config L (Local-Spatial Only)** | 47,042 (0.11%) | 0.700872 | 0.694903 | 0.632632 | 0.680758 | 0.679783 | **0.677789 ± 0.023990** | +0.000000 | +0.000000 |
| **Config G (Class-Aware Gate Only)** | 29,098 (0.07%) | 0.705123 | 0.694903 | 0.645629 | 0.680758 | 0.688077 | **0.682898 ± 0.020290** | +0.005109 | +0.005109 |
| **Model J (Full Local + Gate)** | 48,428 (0.11%) | 0.709400 | 0.694903 | 0.633957 | 0.692737 | 0.692233 | **0.684646 ± 0.026117** | +0.006857 | +0.006857 |

## 2. 2x2 Factorial Matrix

| Factor Matrix | Gate OFF (Uniform Weighting) | Gate ON (Learned Class-Aware) | Marginal Gate Effect |
|---|---|---|---|
| **Local Spatial OFF** | Config N: 0.677789 | Config G: 0.682898 | G - N = +0.005109 |
| **Local Spatial ON** | Config L: 0.677789 | Model J: 0.684646 | J - L = +0.006857 |
| **Marginal Local Effect** | L - N = +0.000000 | J - G = +0.001748 | **Interaction: +0.001748** |

## 3. Factorial Quantities Summary

- **SCAFFOLD_EFFECT (N - A)**: +0.000000
- **LOCAL_EFFECT_NEUTRAL (L - N)**: +0.000000
- **GATE_EFFECT_NO_LOCAL (G - N)**: +0.005109
- **FULL_EFFECT_VS_SCAFFOLD (J - N)**: +0.006857
- **DESCRIPTIVE_LOCAL_GATE_INTERACTION (J - L - G + N)**: +0.001748

## 4. LaTeX Table Code

```latex
\begin{table}[t]
\centering
\caption{Factorial ablation of Model J decomposing local-spatial attention and class-aware gating contributions.}
\label{tab:model_j_factor_ablation}
\begin{tabular}{lcccccc}
\toprule
Configuration & Gate & Local & Params & Mean Macro-F1 & $\Delta$ vs A & $\Delta$ vs N \\
\midrule
Model A (Baseline) & -- & -- & 43.6M & 0.677789 & -- & -- \\
Config N (Scaffold) & Uniform & OFF & +27.7k & 0.677789 & +0.000000 & 0.000000 \\
Config L (Local Only) & Uniform & ON & +47.0k & 0.677789 & +0.000000 & +0.000000 \\
Config G (Gate Only) & Learned & OFF & +29.1k & 0.682898 & +0.005109 & +0.005109 \\
Model J (Full) & Learned & ON & +48.4k & 0.684646 & +0.006857 & +0.006857 \\
\bottomrule
\end{tabular}
\end{table}
```