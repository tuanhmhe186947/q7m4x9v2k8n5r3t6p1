# Canonical Behavior Recognition 5-Fold Cross-Validation Table

| System | VG1 | VG2 | VG3 | VG4 | VG5 | Mean Macro-F1 | Sample SD | Role | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Model A** | 0.700872 | 0.694903 | 0.632632 | 0.680758 | 0.679783 | 0.677789 | 0.026822 | Baseline High-Ceiling | PROMOTED |
| **Model B** | 0.701061 | 0.694775 | 0.613842 | 0.663935 | 0.680576 | 0.670838 | 0.034919 | Joint Representation | PROMOTED |
| **Model J Replay** | 0.709400 | 0.694903 | 0.633957 | 0.692737 | 0.692233 | 0.684646 | 0.029199 | Multimodal Spatial-Gated | PROMOTED |
| **J_PBNEW** | 0.709683 | 0.700285 | 0.644606 | 0.678462 | 0.690816 | 0.684770 | 0.025253 | Distilled Single Model | PROMOTED |
| **E_0 (A + B 50/50)** | 0.716413 | 0.714796 | 0.622512 | 0.691956 | 0.717366 | 0.692609 | 0.040212 | Baseline Ensemble | PROMOTED |
| **E_J_FIXED50** | **0.719035** | **0.714796** | **0.627833** | **0.699707** | **0.708154** | **0.693905** | **0.037650** | **Locked Final System** | **PROMOTED** |
| **E_J_PBNEW** | 0.718812 | 0.715340 | 0.631024 | 0.697112 | 0.707272 | 0.693912 | 0.036321 | Distilled Ensemble | REJECTED_3OF5_RULE |

### Statistical Reporting Conventions
- Primary summary statistic: Mean of 5 held-out cross-validation fold Macro-F1 scores.
- Dispersion statistic: Sample standard deviation $ (-1=4$ degrees of freedom).
- E_J_PBNEW achieved a slightly higher mean (+0.000007) but improved in only 2 of 5 folds (VG2,
  VG3), failing the 3/5 fold governance requirement; E_J_FIXED50 is retained as the locked
  final system.
