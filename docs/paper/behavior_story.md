# Behavior Recognition Narrative: Architectural Factor Ablation and Ensemble Systems

Automated pig behavior recognition requires capturing both individual postural dynamics
and multi-animal social interactions. In this work, we developed and evaluated
spatiotemporal sequence models across a strict 5-fold video-group balanced cross-
validation protocol (12 videos, 28,800 canonical units) ensuring zero video leakage across
training and validation folds:

1. **Model Progression**:
   - **Model A (Baseline High-Ceiling)**: Achieves a 5-fold Macro-F1 of 0.677789 (sample
standard deviation $s=0.0268$, population $\sigma=0.0240$), providing a strong single-
model baseline integrating visual crops and trajectory features.
   - **Model B (Joint Representation)**: Explores joint latent representations across pen-
mates, achieving a 5-fold Macro-F1 of 0.670838 ($s=0.0349$, $\sigma=0.0312$). While
slightly lower in isolated accuracy, Model B offers complementary representations.
   - **Model J Replay (J_F2)**: Introduces local spatial attention and class-aware gating,
improving 5-fold Macro-F1 to 0.684646 ($s=0.0298$, $\sigma=0.0267$), a gain of +0.006857
over Model A.
   - **Model K (Local Motion Branch)**: Augments Model J with local motion features
(Macro-F1 0.682351). While outperforming Model A (+0.004562), it does not exceed Model J.
   - **J_PBNEW (Distilled Single Model)**: Replaces partner probabilities with semantic
probabilities from the best ensemble teacher, achieving 0.684770 Macro-F1 ($s=0.0253$,
$\sigma=0.0226$) and improving in 3 of 5 folds over J_F2, standing as our best single
behavior recognition model.

2. **Model J 2x2 Factorial Ablation (Ablation Language Sanitized)**:
   To understand the exact architectural mechanism of Model J, we conducted a 2x2
factorial ablation across 15 physical checkpoints, comparing:
   - Config N (Scaffold Control): 0.677789 (+0.000000 vs Model A). Demonstrates that the
dual-branch scaffold does not introduce spurious parameter gains.
   - Config L (Local-Spatial Only): 0.677789 (+0.000000 vs Config N). Reveals that local
spatial attention alone, when uniformly weighted, produces zero held-out improvement.
   - Config G (Class Gate Only): 0.682898 (+0.005109 vs Config N). Establishes learned
class-aware gating as the primary standalone driver of performance.
   - Model J (Full L + G): 0.684646 (+0.006857 vs Config N). Combining local spatial
attention with class-aware gating unlocks an additional +0.001748 descriptive interaction
gain.
   The class-aware gate provided the dominant standalone improvement. Local spatial
branches did not improve the neutral scaffold alone, while adding them to the gated
configuration yielded a small positive mean interaction that varied substantially across
folds.

3. **Ensemble Systems and Governance Integrity**:
   Ensembling diverse architectures produces substantial gains:
   - Baseline ensemble E_0 (Model A + Model B 50/50) achieves 0.692609 Macro-F1.
   - Fixed 50/50 ensemble E_J (0.50*J_F2 + 0.50*B) achieves 0.693905 Macro-F1 ($s=0.0376$,
$\sigma=0.0337$), correcting a net 434 errors over Model J alone and standing as our best
validated behavior recognition system.
   - Candidate ensemble E_J_PBNEW (0.50*J_PBNEW + 0.50*B) achieved a mean Macro-F1 of
0.693912 (+0.000007 vs E_J). However, under our project governance rules, promotion
requires improvement in >= 3/5 folds; because E_J_PBNEW improved in only 2 of 5 folds (VG2
and VG3), promotion was formally rejected and E_J was retained.
   - All ensemble weights were strictly locked at fixed 50/50 without validation weight
tuning, and the outer behavior test set remained 100% untouched (`OUTER_TEST_EVALUATIONS =
0`).
