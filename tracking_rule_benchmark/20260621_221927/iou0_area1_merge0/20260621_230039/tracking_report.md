# Tracking Evaluation Report

## Run Config

- IoU threshold: `0.5`
- Include hidden boxes: `False`
- Gap tolerance frames: `15`
- Run missing tracker: `True`
- Force track: `True`
- Ground-truth directory: `C:\Users\ironh\Downloads\PIG_Behavior_Project\data\annotations\tracking`
- Prediction root: `C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\id_tracking\tracking_rule_benchmark\20260621_221927\iou0_area1_merge0`

## Summary

- Ground-truth videos found: `3`
- Videos evaluated: `3`
- Videos missing predictions: `0`

## Aggregate Metrics For Paper

| video_stem | remapped_mota_pct | remapped_idf1_pct | remapped_hota_pct | precision_pct | recall_pct | remapped_idsw | remapped_fragments | remapped_tracklets | remapped_avg_tracklet_length_frames | remapped_gap_tolerant_fragments | remapped_gap_tolerant_tracklets | remapped_gap_tolerant_avg_tracklet_length_frames | remapped_gap_tolerant_suppressed_fragments | idmap_coverage_pct | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 89.33 | 80.59 | 79.09 | 98.80 | 90.76 | 139 | 344 | 368 | 105.38 | 33 | 57 | 680.32 | 311 | 85.18 | 472 | 3950 |

## Per-Video Metrics For Paper

| video_stem | remapped_mota_pct | remapped_idf1_pct | remapped_hota_pct | precision_pct | recall_pct | remapped_idsw | remapped_fragments | remapped_tracklets | remapped_avg_tracklet_length_frames | remapped_gap_tolerant_fragments | remapped_gap_tolerant_tracklets | remapped_gap_tolerant_avg_tracklet_length_frames | remapped_gap_tolerant_suppressed_fragments | idmap_coverage_pct | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | 80.99 | 72.59 | 67.28 | 98.38 | 82.72 | 52 | 190 | 198 | 60.14 | 19 | 27 | 441.04 | 171 | 80.77 | 196 | 2488 |
| Pigs291119_000263_30fps | 87.89 | 68.65 | 67.26 | 98.14 | 90.20 | 83 | 128 | 136 | 92.89 | 12 | 20 | 631.65 | 116 | 73.03 | 240 | 1373 |
| Pigs291119_000302_30fps | 99.10 | 99.21 | 98.63 | 99.75 | 99.38 | 4 | 26 | 34 | 418.74 | 2 | 10 | 1423.70 | 24 | 99.65 | 36 | 89 |

## Raw Absolute-ID Metrics For Audit

These strict metrics compare literal ID names before global remapping. Use them to audit CVAT/tracker naming issues, not as the main paper conclusion when initial ID numbering is arbitrary.

| video_stem | mota_pct | idf1_pct | hota_pct | precision_pct | recall_pct | idsw | fragments | tracklets | avg_tracklet_length_frames | gap_tolerant_fragments | gap_tolerant_tracklets | gap_tolerant_avg_tracklet_length_frames | gap_tolerant_suppressed_fragments | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | 80.99 | 72.59 | 67.28 | 98.38 | 82.72 | 52 | 190 | 198 | 60.14 | 19 | 27 | 441.04 | 171 | 196 | 2488 |
| Pigs291119_000263_30fps | 87.89 | 68.65 | 67.26 | 98.14 | 90.20 | 83 | 128 | 136 | 92.89 | 12 | 20 | 631.65 | 116 | 240 | 1373 |
| Pigs291119_000302_30fps | 99.10 | 99.21 | 98.63 | 99.75 | 99.38 | 4 | 26 | 34 | 418.74 | 2 | 10 | 1423.70 | 24 | 36 | 89 |
| ALL | 89.33 | 80.59 | 79.09 | 98.80 | 90.76 | 139 | 344 | 368 | 105.38 | 33 | 57 | 680.32 | 311 | 472 | 3950 |

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
| Pigs281119_000085_30fps | ID_1 | ID_4 | 1432 | 11908 | 0.8077 |
| Pigs281119_000085_30fps | ID_2 | ID_3 | 695 | 11908 | 0.8077 |
| Pigs281119_000085_30fps | ID_3 | ID_1 | 908 | 11908 | 0.8077 |
| Pigs281119_000085_30fps | ID_4 | ID_2 | 1127 | 11908 | 0.8077 |
| Pigs281119_000085_30fps | ID_5 | ID_7 | 1412 | 11908 | 0.8077 |
| Pigs281119_000085_30fps | ID_6 | ID_6 | 857 | 11908 | 0.8077 |
| Pigs281119_000085_30fps | ID_7 | ID_8 | 1800 | 11908 | 0.8077 |
| Pigs281119_000085_30fps | ID_8 | ID_5 | 1387 | 11908 | 0.8077 |
| Pigs291119_000263_30fps | ID_1 | ID_3 | 625 | 12633 | 0.7303 |
| Pigs291119_000263_30fps | ID_2 | ID_2 | 1344 | 12633 | 0.7303 |
| Pigs291119_000263_30fps | ID_3 | ID_1 | 509 | 12633 | 0.7303 |
| Pigs291119_000263_30fps | ID_4 | ID_6 | 1638 | 12633 | 0.7303 |
| Pigs291119_000263_30fps | ID_5 | ID_5 | 1380 | 12633 | 0.7303 |
| Pigs291119_000263_30fps | ID_6 | ID_4 | 897 | 12633 | 0.7303 |
| Pigs291119_000263_30fps | ID_7 | ID_8 | 1762 | 12633 | 0.7303 |
| Pigs291119_000263_30fps | ID_8 | ID_7 | 1071 | 12633 | 0.7303 |
| Pigs291119_000302_30fps | ID_1 | ID_3 | 1800 | 14237 | 0.9965 |
| Pigs291119_000302_30fps | ID_2 | ID_1 | 1800 | 14237 | 0.9965 |
| Pigs291119_000302_30fps | ID_3 | ID_2 | 1800 | 14237 | 0.9965 |
| Pigs291119_000302_30fps | ID_4 | ID_4 | 1778 | 14237 | 0.9965 |

## Remapped Identity Diagnostics

- `tracking_remapped_identity_events.csv`: identity events after fixed ID remapping; these are continuity errors that remain after removing arbitrary initial ID numbering.
- Remapped identity event rows: `5798`
- Remapped ID switch rows: `139`

## Continuity Gap Diagnostics

- `tracking_continuity_gaps.csv`: matched-track gaps after fixed ID remapping. Gaps shorter than or equal to the configured tolerance are not counted as gap-tolerant fragments.
- Total matched-track gaps: `344`
- Tolerated gaps: `311`
- Remaining fragment gaps: `33`

| video_stem | gt_id | previous_matched_frame | next_matched_frame | gap_frames | tolerated | previous_pred_id | next_pred_id | id_changed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs291119_000263_30fps | ID_4 | 547 | 1138 | 590 | False | ID_1 | ID_4 | True |
| Pigs281119_000085_30fps | ID_6 | 867 | 1372 | 504 | False | ID_6 | ID_7 | True |
| Pigs291119_000263_30fps | ID_7 | 1468 | 1681 | 212 | False | ID_3 | ID_3 | False |
| Pigs291119_000263_30fps | ID_1 | 1246 | 1416 | 169 | False | ID_7 | ID_5 | True |
| Pigs291119_000263_30fps | ID_7 | 1087 | 1230 | 142 | False | ID_7 | ID_3 | True |
| Pigs291119_000263_30fps | ID_3 | 305 | 387 | 81 | False | ID_1 | ID_4 | True |
| Pigs281119_000085_30fps | ID_5 | 1657 | 1720 | 62 | False | ID_1 | ID_5 | True |
| Pigs281119_000085_30fps | ID_5 | 1518 | 1577 | 58 | False | ID_5 | ID_5 | False |
| Pigs291119_000263_30fps | ID_4 | 385 | 434 | 48 | False | ID_4 | ID_4 | False |
| Pigs281119_000085_30fps | ID_3 | 106 | 151 | 44 | False | ID_3 | ID_3 | False |
| Pigs281119_000085_30fps | ID_5 | 1736 | 1779 | 42 | False | ID_5 | ID_7 | True |
| Pigs281119_000085_30fps | ID_3 | 805 | 845 | 39 | False | ID_3 | ID_2 | True |
| Pigs291119_000302_30fps | ID_5 | 941 | 977 | 35 | False | ID_5 | ID_7 | True |
| Pigs291119_000263_30fps | ID_1 | 1059 | 1091 | 31 | False | ID_1 | ID_7 | True |
| Pigs291119_000263_30fps | ID_7 | 1723 | 1755 | 31 | False | ID_3 | ID_3 | False |
| Pigs291119_000263_30fps | ID_3 | 1219 | 1249 | 29 | False | ID_3 | ID_3 | False |
| Pigs281119_000085_30fps | ID_1 | 1101 | 1128 | 26 | False | ID_1 | ID_6 | True |
| Pigs281119_000085_30fps | ID_1 | 1149 | 1176 | 26 | False | ID_1 | ID_6 | True |
| Pigs281119_000085_30fps | ID_4 | 1414 | 1441 | 26 | False | ID_4 | ID_4 | False |
| Pigs281119_000085_30fps | ID_1 | 1727 | 1753 | 25 | False | ID_1 | ID_5 | True |

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
