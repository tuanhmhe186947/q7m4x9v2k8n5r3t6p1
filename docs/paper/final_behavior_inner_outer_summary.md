# Final Behavior Evaluation Summary: Cross-Validation Authority

## Primary Estimand Authority
The primary scientific statistic is the mean of the 5 held-out cross-validation
fold Macro-F1 scores across the 5 video groups (VG1–VG5), with sample standard
deviation $ (-1=4$).

| Evaluation Scope | System | Samples | Videos | Groups | Mean Macro-F1 | Sample SD | Role & Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **5-Fold Group CV** | Model A | 33,287 | 678 | 5 | 0.677789 | 0.026822 | Baseline High-Ceiling |
| **5-Fold Group CV** | Model B | 33,287 | 678 | 5 | 0.670838 | 0.034919 | Joint Representation |
| **5-Fold Group CV** | Model J | 33,287 | 678 | 5 | 0.684646 | 0.029199 | Multimodal Spatial-Gated |
| **5-Fold Group CV** | J_PBNEW | 33,287 | 678 | 5 | 0.684770 | 0.025253 | Distilled Single Model |
| **5-Fold Group CV** | **E_J_FIXED50** | 33,287 | 678 | 5 | **0.693905** | **0.037650** | **Locked Final System** |
| **Pooled OOF Audit** | Synthetic 80-83% | 33,287 | 678 | 5 | *INVALID* | *INVALID* | INVALID_SUPERSEDED |
| **Outer Held-Out** | E_J_FIXED50 | 0 | 0 | 0 | NOT_MATERIALIZED | NOT_MATERIALIZED | Defined but un-materialized |

### Audit of the 80–83% Figures
The figures ~80.59% (Model B), ~82.68% (Model J), and ~83.47% (E_J) that
appeared in previous drafted summary tables were an ungrounded artifact
synthesized without protocol authority. They are formally classified as
INVALID_SUPERSEDED. The sole authoritative numbers remain the physical
5-fold mean Macro-F1 values: Model B = 0.670838, Model J = 0.684646,
E_J_FIXED50 = 0.693905.
