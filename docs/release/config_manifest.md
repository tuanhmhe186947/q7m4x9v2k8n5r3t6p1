# Public Configuration Manifest

This document indexes configuration files retained in `configs/classification_v2/` for training contracts, cross-validation runs, ablation benchmarks, and paper provenance.

---

## 1. Summary by Family

| Configuration Family | Count | Description & Scientific Purpose |
| :--- | :---: | :--- |
| **Model Architectures & Contracts** | 12 | Core contracts (`model_architecture_contract_v1.json`, `trainer_contract_v1.json`, `trainer_contract_v2.json`, `lineage_rebuild_v1.yaml`). |
| **M0 Multimodal Full-T6 Baseline** | 8 | Initial multimodal spatio-temporal contracts and outer-fold baselines (`m0_full_multimodal_r34_t6_concat.json`, `m0_full_t6_scientific_training_contract_v1.json`). |
| **M1 Relational & Adversarial** | 3 | Group relational partner token and date-adversarial contracts (`m1_rp1_relational_partner_tokens_v1.json`, `m1_dg1_date_adversarial_v1.json`). |
| **M2 Visual Fine-Tuning (VFT)** | 12 | Visual backbone fine-tuning across validation groups VG1–VG5 (`m2_vft_*`). |
| **M3 Focal Gating (FG)** | 5 | Focal loss class gating across validation groups VG1–VG5 (`m3_fg_*`). |
| **M4 Balanced Adaptive Sampling (BAS)** | 5 | Balanced adaptive sampling across validation groups VG1–VG5 (`m4_bas_*`). |
| **M5 Staged Fine-Tuning (SFT)** | 5 | Staged fine-tuning across validation groups VG1–VG5 (`m5_sft_*`). |
| **Q2 Finalist Calibration & Layouts** | 18 | Finalist evaluation layouts, feature whitelists, and OOF metric contracts (`q2_*`, `reviewed_q2_*`). |
| **Model Research Contracts** | 4 | High-level research and candidate contracts under `model_research/`. |
| **Runtime & Operator Profiles** | 26 | Operator profiles, posture auto-validation, and execution recipes (`gui_operator_profile_v1.json`, `source_domain_control_v1.json`, etc.). |
| **Total** | **98** | Full scientific reproducibility preserved. |

---

## 2. Retention Policy Compliance
All 98 configuration files are retained to satisfy:
1. Exact parameter replay for Models A, B, J, and the factor ablation study.
2. Cross-validation fold definitions ensuring zero video leakage across VG1–VG5.
3. Strict lineage verification against `docs/paper/master_evidence_ledger.csv`.
