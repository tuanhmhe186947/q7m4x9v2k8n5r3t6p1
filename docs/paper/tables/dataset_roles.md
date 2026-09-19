# Experimental Dataset Roles and Cohort Definitions

| Dataset Cohort | Primary Role | Videos | Subjects | Annotated Units | Physical Frames | Annotation Types | Isolation & Purpose |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Tracking Development (Full)** | Tracking development & parameter tuning | 13 | 104 | NA | 187,200 | Bounding boxes, track IDs, occlusions | Contains 1 tracking-only video (`Pigs291119_000263`). Used for tracking pipeline development. |
| **Development Behavior-Overlap (DEV12)** | Identity-to-profile distortion analysis | 12 | 96 | 28,800 canonical units | 172,800 | Dual GT: Tracking IDs + 10-class behavior | Used strictly to isolate downstream profile distortion under fixed human labels. Excludes video 000263. |
| **Tracking Confirmatory Cohort** | Held-out tracking generalization | 12 | 96 | 0 | 172,800 | Tracking bounding boxes & IDs only | 12 held-out independent videos evaluated without post-hoc parameter or threshold tuning (`CONFIRMATORY_B`). |
| **Behavior 5-Fold Balanced CV** | Multimodal behavior model training & ablation | 12 | 96 | 28,800 canonical units | 172,800 | 10 mutually exclusive behavior categories | Group-aware 5-fold CV (VG1–VG5) ensuring zero video leakage across folds. Used for Model A, B, J, and ensembles. |
| **Behavior Nested Outer Protocol** | Protocol defined, not materialized | Held-out | Independent | Independent | Independent | Human-reviewed behavior ground truth | Defined in protocol contracts (`outer_test_evaluations = 0`); no distinct physical outer cohort was materialized. |
