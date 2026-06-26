# Tracking Evaluation Report

## Run Config

- IoU threshold: `0.5`
- Include hidden boxes: `False`
- Gap tolerance frames: `15`
- Run missing tracker: `True`
- Force track: `True`
- Ground-truth directory: `C:\Users\ironh\Downloads\PIG_Behavior_Project\data\annotations\tracking`
- Prediction root: `C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\id_tracking\tracking_rule_benchmark\20260621_221927\iou0_area0_merge0`

## Summary

- Ground-truth videos found: `3`
- Videos evaluated: `3`
- Videos missing predictions: `0`

## Aggregate Metrics For Paper

| video_stem | remapped_mota_pct | remapped_idf1_pct | remapped_hota_pct | precision_pct | recall_pct | remapped_idsw | remapped_fragments | remapped_tracklets | remapped_avg_tracklet_length_frames | remapped_gap_tolerant_fragments | remapped_gap_tolerant_tracklets | remapped_gap_tolerant_avg_tracklet_length_frames | remapped_gap_tolerant_suppressed_fragments | idmap_coverage_pct | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 98.69 | 95.43 | 95.91 | 99.20 | 99.50 | 4 | 127 | 151 | 281.56 | 9 | 33 | 1288.36 | 118 | 96.05 | 342 | 212 |

## Per-Video Metrics For Paper

| video_stem | remapped_mota_pct | remapped_idf1_pct | remapped_hota_pct | precision_pct | recall_pct | remapped_idsw | remapped_fragments | remapped_tracklets | remapped_avg_tracklet_length_frames | remapped_gap_tolerant_fragments | remapped_gap_tolerant_tracklets | remapped_gap_tolerant_avg_tracklet_length_frames | remapped_gap_tolerant_suppressed_fragments | idmap_coverage_pct | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | 98.78 | 99.39 | 98.79 | 99.50 | 99.28 | 0 | 32 | 40 | 357.30 | 1 | 9 | 1588.00 | 31 | 100.00 | 72 | 104 |
| Pigs291119_000263_30fps | 97.85 | 98.92 | 97.91 | 98.36 | 99.53 | 2 | 76 | 84 | 165.95 | 6 | 14 | 995.71 | 70 | 99.99 | 233 | 66 |
| Pigs291119_000302_30fps | 99.43 | 88.02 | 90.85 | 99.74 | 99.71 | 2 | 19 | 27 | 529.04 | 2 | 10 | 1428.40 | 17 | 88.26 | 37 | 42 |

## Raw Absolute-ID Metrics For Audit

These strict metrics compare literal ID names before global remapping. Use them to audit CVAT/tracker naming issues, not as the main paper conclusion when initial ID numbering is arbitrary.

| video_stem | mota_pct | idf1_pct | hota_pct | precision_pct | recall_pct | idsw | fragments | tracklets | avg_tracklet_length_frames | gap_tolerant_fragments | gap_tolerant_tracklets | gap_tolerant_avg_tracklet_length_frames | gap_tolerant_suppressed_fragments | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | 98.78 | 99.39 | 98.79 | 99.50 | 99.28 | 0 | 32 | 40 | 357.30 | 1 | 9 | 1588.00 | 31 | 72 | 104 |
| Pigs291119_000263_30fps | 97.85 | 98.92 | 97.91 | 98.36 | 99.53 | 2 | 76 | 84 | 165.95 | 6 | 14 | 995.71 | 70 | 233 | 66 |
| Pigs291119_000302_30fps | 99.43 | 88.02 | 90.85 | 99.74 | 99.71 | 2 | 19 | 27 | 529.04 | 2 | 10 | 1428.40 | 17 | 37 | 42 |
| ALL | 98.69 | 95.43 | 95.91 | 99.20 | 99.50 | 4 | 127 | 151 | 281.56 | 9 | 33 | 1288.36 | 118 | 342 | 212 |

## Asset Coverage

| video_stem | has_prediction | gt_task_size | video_frame_count | video_fps | video_width | video_height |
| --- | --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | True | 1800 | 1800 | 30.00 | 1280 | 720 |
| Pigs291119_000263_30fps | True | 1800 | 1800 | 30.00 | 1280 | 720 |
| Pigs291119_000302_30fps | True | 1800 | 1800 | 30.00 | 1280 | 720 |

## Fixed ID Mapping

`tracking_id_mapping.csv` stores the fixed prediction ID_N -> GT ID_N mapping selected from whole-video matched overlap.

| video_stem | pred_id | mapped_gt_id | matched_frames | total_matched_frames | mapping_coverage |
| --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | ID_1 | ID_3 | 1771 | 14292 | 1.00 |
| Pigs281119_000085_30fps | ID_2 | ID_4 | 1797 | 14292 | 1.00 |
| Pigs281119_000085_30fps | ID_3 | ID_1 | 1766 | 14292 | 1.00 |
| Pigs281119_000085_30fps | ID_4 | ID_2 | 1800 | 14292 | 1.00 |
| Pigs281119_000085_30fps | ID_5 | ID_7 | 1799 | 14292 | 1.00 |
| Pigs281119_000085_30fps | ID_6 | ID_6 | 1778 | 14292 | 1.00 |
| Pigs281119_000085_30fps | ID_7 | ID_8 | 1800 | 14292 | 1.00 |
| Pigs281119_000085_30fps | ID_8 | ID_5 | 1781 | 14292 | 1.00 |
| Pigs291119_000263_30fps | ID_1 | ID_1 | 1788 | 13940 | 0.9999 |
| Pigs291119_000263_30fps | ID_2 | ID_2 | 1794 | 13940 | 0.9999 |
| Pigs291119_000263_30fps | ID_3 | ID_3 | 1751 | 13940 | 0.9999 |
| Pigs291119_000263_30fps | ID_4 | ID_4 | 1785 | 13940 | 0.9999 |
| Pigs291119_000263_30fps | ID_5 | ID_5 | 1800 | 13940 | 0.9999 |
| Pigs291119_000263_30fps | ID_6 | ID_6 | 1776 | 13940 | 0.9999 |
| Pigs291119_000263_30fps | ID_7 | ID_8 | 1763 | 13940 | 0.9999 |
| Pigs291119_000263_30fps | ID_8 | ID_7 | 1481 | 13940 | 0.9999 |
| Pigs291119_000302_30fps | ID_1 | ID_3 | 1800 | 14284 | 0.8826 |
| Pigs291119_000302_30fps | ID_2 | ID_1 | 1800 | 14284 | 0.8826 |
| Pigs291119_000302_30fps | ID_3 | ID_2 | 1800 | 14284 | 0.8826 |
| Pigs291119_000302_30fps | ID_4 | ID_4 | 1800 | 14284 | 0.8826 |

## Remapped Identity Diagnostics

- `tracking_remapped_identity_events.csv`: identity events after fixed ID remapping; these are continuity errors that remain after removing arbitrary initial ID numbering.
- Remapped identity event rows: `1680`
- Remapped ID switch rows: `4`

## Continuity Gap Diagnostics

- `tracking_continuity_gaps.csv`: matched-track gaps after fixed ID remapping. Gaps shorter than or equal to the configured tolerance are not counted as gap-tolerant fragments.
- Total matched-track gaps: `127`
- Tolerated gaps: `118`
- Remaining fragment gaps: `9`

| video_stem | gt_id | previous_matched_frame | next_matched_frame | gap_frames | tolerated | previous_pred_id | next_pred_id | id_changed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs291119_000263_30fps | ID_7 | 1538 | 1626 | 87 | False | ID_7 | ID_7 | False |
| Pigs291119_000263_30fps | ID_7 | 1468 | 1538 | 69 | False | ID_7 | ID_7 | False |
| Pigs291119_000263_30fps | ID_7 | 1124 | 1192 | 67 | False | ID_7 | ID_7 | False |
| Pigs291119_000263_30fps | ID_7 | 1636 | 1672 | 35 | False | ID_7 | ID_7 | False |
| Pigs291119_000302_30fps | ID_5 | 941 | 977 | 35 | False | ID_5 | ID_7 | True |
| Pigs281119_000085_30fps | ID_3 | 23 | 46 | 22 | False | ID_3 | ID_3 | False |
| Pigs291119_000263_30fps | ID_7 | 1098 | 1121 | 22 | False | ID_7 | ID_7 | False |
| Pigs291119_000302_30fps | ID_8 | 557 | 575 | 17 | False | ID_8 | ID_8 | False |
| Pigs291119_000263_30fps | ID_8 | 0 | 17 | 16 | False | ID_8 | ID_8 | False |
| Pigs291119_000302_30fps | ID_8 | 525 | 541 | 15 | True | ID_8 | ID_8 | False |
| Pigs291119_000302_30fps | ID_8 | 967 | 981 | 13 | True | ID_8 | ID_8 | False |
| Pigs291119_000263_30fps | ID_6 | 1024 | 1037 | 12 | True | ID_6 | ID_6 | False |
| Pigs291119_000263_30fps | ID_3 | 1466 | 1478 | 11 | True | ID_3 | ID_3 | False |
| Pigs281119_000085_30fps | ID_1 | 939 | 949 | 9 | True | ID_1 | ID_1 | False |
| Pigs281119_000085_30fps | ID_5 | 1536 | 1546 | 9 | True | ID_5 | ID_5 | False |
| Pigs291119_000263_30fps | ID_1 | 1066 | 1075 | 8 | True | ID_1 | ID_1 | False |
| Pigs291119_000263_30fps | ID_3 | 1491 | 1500 | 8 | True | ID_3 | ID_3 | False |
| Pigs291119_000302_30fps | ID_8 | 489 | 498 | 8 | True | ID_8 | ID_8 | False |
| Pigs281119_000085_30fps | ID_1 | 691 | 699 | 7 | True | ID_1 | ID_1 | False |
| Pigs281119_000085_30fps | ID_6 | 964 | 972 | 7 | True | ID_6 | ID_6 | False |

## Metric Guide

- `MOTA`: overall tracking accuracy. Penalizes missed objects, false positives, and ID switches.
- `IDF1`: identity consistency score. Higher means the same pig ID is preserved better.
- `Remapped *`: metrics after one-to-one global ID mapping. Use these for paper reporting when initial tracker ID numbering is arbitrary.
- `HOTA`: combined detection and association score.
- `MOTP IoU`: average box overlap quality for matched objects.
- `FP`: predicted boxes that did not match ground truth.
- `FN`: ground-truth boxes missed by prediction.
- `IDSW`: ID switches.
- `Fragments`: tracks that were interrupted and later recovered.
- `Tracklets`: continuous matched identity segments. `Avg. tracklet length` is their mean length in frames.
- `Gap-tolerant *`: continuity metrics after merging matched segments separated by short gaps up to `gap_tolerance_frames`.
