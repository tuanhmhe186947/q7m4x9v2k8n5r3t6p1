# Master Evaluation Protocols (Corrected Freeze)

## Tracking Evaluation Protocol
- **Contract**: TRACKING_EVALUATOR_STANDARD_V2
- **Matching**: Bipartite Hungarian matching at IoU >= 0.50 (19-point alpha set [0.05, 0.95]).
- **Occlusion Handling**: include_hidden = True (hidden bounding boxes included in matching).
- **Primary Metrics**: HOTA, DetA, AssA, IDF1, IDSW, Wrong-ID Matched Frames (Exposure).
- **Wrong-ID Exposure Denominators**:
  - Canonical: wrong_id_fraction_matched (relative to authoritative matched ground-truth
    frames).
  - Secondary: wrong_id_fraction_all_gt (relative to total 172,800 ground truth boxes).
- **Provenance Classification**: CONFIRMATORY_B (Existing August 19-20, 2026 physical
  evaluation artifacts certified under Standard V2 without parameter tuning).
- **Bootstrap Parameters**: Seed 240494961, 10,000 video-level bootstrap replications.
- **Bootstrap Inference**: Reported as paired effect delta + 95% nonparametric bootstrap CI and
  Bootstrap Positive Fraction (never mislabeled as p-value).

## Behavior Recognition Protocol
- **Contract**: 10-class Macro-F1.
- **Canonical Label Order**: drink, eat, fight, social-nose, explore, lying, stand, move,
  sitting, playwithtoy.
- **Primary Estimand**: Mean of the 5 held-out cross-validation fold Macro-F1 scores across
  VG1–VG5.
- **Dispersion Statistic**: Sample standard deviation $ (-1=4$ degrees of freedom).
- **Dataset Population**: Video-Group Balanced 5-Fold CV (=33,287$ canonical 6-frame units
  across 678 unique videos).
- **Outer Test Status**: Defined as a 4-fold outer protocol in outer_oof_contract.json
  (FROZEN_PROTOCOL_NOT_AUTHORIZED), but not materialized as a distinct independent dataset.
