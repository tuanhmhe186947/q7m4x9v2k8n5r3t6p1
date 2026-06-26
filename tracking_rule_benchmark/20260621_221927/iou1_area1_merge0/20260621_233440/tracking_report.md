# Tracking Evaluation Report

## Run Config

- IoU threshold: `0.5`
- Include hidden boxes: `False`
- Gap tolerance frames: `15`
- Run missing tracker: `True`
- Force track: `True`
- Ground-truth directory: `C:\Users\ironh\Downloads\PIG_Behavior_Project\data\annotations\tracking`
- Prediction root: `C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\id_tracking\tracking_rule_benchmark\20260621_221927\iou1_area1_merge0`

## Summary

- Ground-truth videos found: `3`
- Videos evaluated: `3`
- Videos missing predictions: `0`

## Aggregate Metrics For Paper

| video_stem | remapped_mota_pct | remapped_idf1_pct | remapped_hota_pct | precision_pct | recall_pct | remapped_idsw | remapped_fragments | remapped_tracklets | remapped_avg_tracklet_length_frames | remapped_gap_tolerant_fragments | remapped_gap_tolerant_tracklets | remapped_gap_tolerant_avg_tracklet_length_frames | remapped_gap_tolerant_suppressed_fragments | idmap_coverage_pct | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 89.92 | 80.86 | 79.74 | 98.82 | 91.35 | 144 | 334 | 358 | 109.03 | 35 | 59 | 661.54 | 299 | 85.18 | 467 | 3697 |

## Per-Video Metrics For Paper

| video_stem | remapped_mota_pct | remapped_idf1_pct | remapped_hota_pct | precision_pct | recall_pct | remapped_idsw | remapped_fragments | remapped_tracklets | remapped_avg_tracklet_length_frames | remapped_gap_tolerant_fragments | remapped_gap_tolerant_tracklets | remapped_gap_tolerant_avg_tracklet_length_frames | remapped_gap_tolerant_suppressed_fragments | idmap_coverage_pct | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | 82.84 | 73.65 | 69.59 | 98.55 | 84.42 | 49 | 187 | 195 | 62.32 | 22 | 30 | 405.10 | 165 | 80.98 | 179 | 2243 |
| Pigs291119_000263_30fps | 87.81 | 68.51 | 67.17 | 98.11 | 90.20 | 91 | 121 | 129 | 97.93 | 12 | 20 | 631.65 | 109 | 72.90 | 244 | 1373 |
| Pigs291119_000302_30fps | 99.10 | 99.21 | 98.63 | 99.69 | 99.43 | 4 | 26 | 34 | 418.97 | 1 | 9 | 1582.78 | 25 | 99.65 | 44 | 81 |

## Raw Absolute-ID Metrics For Audit

These strict metrics compare literal ID names before global remapping. Use them to audit CVAT/tracker naming issues, not as the main paper conclusion when initial ID numbering is arbitrary.

| video_stem | mota_pct | idf1_pct | hota_pct | precision_pct | recall_pct | idsw | fragments | tracklets | avg_tracklet_length_frames | gap_tolerant_fragments | gap_tolerant_tracklets | gap_tolerant_avg_tracklet_length_frames | gap_tolerant_suppressed_fragments | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | 82.84 | 73.65 | 69.59 | 98.55 | 84.42 | 49 | 187 | 195 | 62.32 | 22 | 30 | 405.10 | 165 | 179 | 2243 |
| Pigs291119_000263_30fps | 87.81 | 68.51 | 67.17 | 98.11 | 90.20 | 91 | 121 | 129 | 97.93 | 12 | 20 | 631.65 | 109 | 244 | 1373 |
| Pigs291119_000302_30fps | 99.10 | 99.21 | 98.63 | 99.69 | 99.43 | 4 | 26 | 34 | 418.97 | 1 | 9 | 1582.78 | 25 | 44 | 81 |
| ALL | 89.92 | 80.86 | 79.74 | 98.82 | 91.35 | 144 | 334 | 358 | 109.03 | 35 | 59 | 661.54 | 299 | 467 | 3697 |

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
| Pigs281119_000085_30fps | ID_1 | ID_4 | 1438 | 12153 | 0.8098 |
| Pigs281119_000085_30fps | ID_2 | ID_3 | 697 | 12153 | 0.8098 |
| Pigs281119_000085_30fps | ID_3 | ID_1 | 904 | 12153 | 0.8098 |
| Pigs281119_000085_30fps | ID_4 | ID_2 | 1127 | 12153 | 0.8098 |
| Pigs281119_000085_30fps | ID_5 | ID_7 | 1435 | 12153 | 0.8098 |
| Pigs281119_000085_30fps | ID_6 | ID_6 | 857 | 12153 | 0.8098 |
| Pigs281119_000085_30fps | ID_7 | ID_8 | 1800 | 12153 | 0.8098 |
| Pigs281119_000085_30fps | ID_8 | ID_5 | 1584 | 12153 | 0.8098 |
| Pigs291119_000263_30fps | ID_1 | ID_3 | 625 | 12633 | 0.7290 |
| Pigs291119_000263_30fps | ID_2 | ID_2 | 1344 | 12633 | 0.7290 |
| Pigs291119_000263_30fps | ID_3 | ID_1 | 509 | 12633 | 0.7290 |
| Pigs291119_000263_30fps | ID_4 | ID_6 | 1638 | 12633 | 0.7290 |
| Pigs291119_000263_30fps | ID_5 | ID_5 | 1380 | 12633 | 0.7290 |
| Pigs291119_000263_30fps | ID_6 | ID_4 | 881 | 12633 | 0.7290 |
| Pigs291119_000263_30fps | ID_7 | ID_8 | 1761 | 12633 | 0.7290 |
| Pigs291119_000263_30fps | ID_8 | ID_7 | 1071 | 12633 | 0.7290 |
| Pigs291119_000302_30fps | ID_1 | ID_3 | 1800 | 14245 | 0.9965 |
| Pigs291119_000302_30fps | ID_2 | ID_1 | 1800 | 14245 | 0.9965 |
| Pigs291119_000302_30fps | ID_3 | ID_2 | 1800 | 14245 | 0.9965 |
| Pigs291119_000302_30fps | ID_4 | ID_4 | 1778 | 14245 | 0.9965 |

## Remapped Identity Diagnostics

- `tracking_remapped_identity_events.csv`: identity events after fixed ID remapping; these are continuity errors that remain after removing arbitrary initial ID numbering.
- Remapped identity event rows: `5837`
- Remapped ID switch rows: `144`

## Continuity Gap Diagnostics

- `tracking_continuity_gaps.csv`: matched-track gaps after fixed ID remapping. Gaps shorter than or equal to the configured tolerance are not counted as gap-tolerant fragments.
- Total matched-track gaps: `334`
- Tolerated gaps: `299`
- Remaining fragment gaps: `35`

| video_stem | gt_id | previous_matched_frame | next_matched_frame | gap_frames | tolerated | previous_pred_id | next_pred_id | id_changed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs291119_000263_30fps | ID_4 | 547 | 1138 | 590 | False | ID_1 | ID_4 | True |
| Pigs281119_000085_30fps | ID_6 | 867 | 1372 | 504 | False | ID_6 | ID_7 | True |
| Pigs291119_000263_30fps | ID_7 | 1468 | 1681 | 212 | False | ID_3 | ID_3 | False |
| Pigs291119_000263_30fps | ID_1 | 1246 | 1416 | 169 | False | ID_7 | ID_5 | True |
| Pigs291119_000263_30fps | ID_7 | 1087 | 1230 | 142 | False | ID_7 | ID_3 | True |
| Pigs291119_000263_30fps | ID_3 | 305 | 387 | 81 | False | ID_1 | ID_4 | True |
| Pigs291119_000263_30fps | ID_4 | 385 | 434 | 48 | False | ID_4 | ID_4 | False |
| Pigs281119_000085_30fps | ID_3 | 106 | 149 | 42 | False | ID_3 | ID_3 | False |
| Pigs281119_000085_30fps | ID_3 | 805 | 845 | 39 | False | ID_3 | ID_2 | True |
| Pigs291119_000302_30fps | ID_5 | 941 | 977 | 35 | False | ID_5 | ID_7 | True |
| Pigs291119_000263_30fps | ID_1 | 1059 | 1091 | 31 | False | ID_1 | ID_7 | True |
| Pigs291119_000263_30fps | ID_7 | 1723 | 1755 | 31 | False | ID_3 | ID_3 | False |
| Pigs281119_000085_30fps | ID_1 | 1645 | 1676 | 30 | False | ID_7 | ID_7 | False |
| Pigs291119_000263_30fps | ID_3 | 1219 | 1249 | 29 | False | ID_3 | ID_3 | False |
| Pigs281119_000085_30fps | ID_1 | 1101 | 1128 | 26 | False | ID_1 | ID_6 | True |
| Pigs281119_000085_30fps | ID_1 | 1149 | 1176 | 26 | False | ID_1 | ID_6 | True |
| Pigs281119_000085_30fps | ID_4 | 1414 | 1441 | 26 | False | ID_4 | ID_4 | False |
| Pigs281119_000085_30fps | ID_1 | 1727 | 1753 | 25 | False | ID_6 | ID_7 | True |
| Pigs281119_000085_30fps | ID_6 | 1394 | 1419 | 24 | False | ID_7 | ID_4 | True |
| Pigs281119_000085_30fps | ID_6 | 1437 | 1459 | 21 | False | ID_4 | ID_7 | True |

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
