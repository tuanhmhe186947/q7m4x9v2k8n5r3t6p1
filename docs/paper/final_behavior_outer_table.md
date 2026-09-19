# Behavior Outer Evaluation Status Table

## Status: NESTED_OUTER_DEFINED_BUT_NOT_MATERIALIZED
An outer out-of-fold evaluation protocol was formally specified in
docs/classification_v2/corrected_pooled_route_20260806/outer_oof_contract.json
with status FROZEN_PROTOCOL_NOT_AUTHORIZED. No distinct physical outer
dataset was materialized separate from the 33,287 canonical units.

| System | Role | Protocol Status | Outer Samples | Outer Videos | Outer Evaluations | Outer Macro-F1 | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **E_J_FIXED50** | Locked Final System | NOT_AUTHORIZED | 0 | 0 | 0 | NOT_MEASURED | Protocol frozen; outer test cohort was not materialized on disk. |
| **Model J_F2** | Component System | NOT_AUTHORIZED | 0 | 0 | 0 | NOT_MEASURED | Component model; outer evaluations = 0. |
| **Model B** | Component System | NOT_AUTHORIZED | 0 | 0 | 0 | NOT_MEASURED | Component model; outer evaluations = 0. |
