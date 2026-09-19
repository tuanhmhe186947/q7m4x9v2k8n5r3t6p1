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
