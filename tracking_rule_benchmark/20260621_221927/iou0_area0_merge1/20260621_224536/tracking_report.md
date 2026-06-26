# Tracking Evaluation Report

## Run Config

- IoU threshold: `0.5`
- Include hidden boxes: `False`
- Gap tolerance frames: `15`
- Run missing tracker: `True`
- Force track: `True`
- Ground-truth directory: `C:\Users\ironh\Downloads\PIG_Behavior_Project\data\annotations\tracking`
- Prediction root: `C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\id_tracking\tracking_rule_benchmark\20260621_221927\iou0_area0_merge1`

## Summary

- Ground-truth videos found: `3`
- Videos evaluated: `3`
- Videos missing predictions: `0`

## Aggregate Metrics For Paper

| video_stem | remapped_mota_pct | remapped_idf1_pct | remapped_hota_pct | precision_pct | recall_pct | remapped_idsw | remapped_fragments | remapped_tracklets | remapped_avg_tracklet_length_frames | remapped_gap_tolerant_fragments | remapped_gap_tolerant_tracklets | remapped_gap_tolerant_avg_tracklet_length_frames | remapped_gap_tolerant_suppressed_fragments | idmap_coverage_pct | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 52.98 | 60.59 | 54.07 | 98.94 | 53.65 | 38 | 917 | 941 | 24.36 | 218 | 242 | 94.72 | 699 | 87.10 | 246 | 19805 |

## Per-Video Metrics For Paper

| video_stem | remapped_mota_pct | remapped_idf1_pct | remapped_hota_pct | precision_pct | recall_pct | remapped_idsw | remapped_fragments | remapped_tracklets | remapped_avg_tracklet_length_frames | remapped_gap_tolerant_fragments | remapped_gap_tolerant_tracklets | remapped_gap_tolerant_avg_tracklet_length_frames | remapped_gap_tolerant_suppressed_fragments | idmap_coverage_pct | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | 50.99 | 62.12 | 53.19 | 99.05 | 51.55 | 10 | 282 | 290 | 25.59 | 78 | 86 | 86.29 | 204 | 91.60 | 71 | 6975 |
| Pigs291119_000263_30fps | 40.63 | 43.62 | 38.36 | 97.74 | 41.72 | 18 | 400 | 408 | 14.32 | 85 | 93 | 62.83 | 315 | 74.60 | 135 | 8163 |
| Pigs291119_000302_30fps | 67.07 | 73.32 | 66.69 | 99.59 | 67.42 | 10 | 235 | 243 | 39.75 | 55 | 63 | 153.32 | 180 | 91.19 | 40 | 4667 |

## Raw Absolute-ID Metrics For Audit

These strict metrics compare literal ID names before global remapping. Use them to audit CVAT/tracker naming issues, not as the main paper conclusion when initial ID numbering is arbitrary.

| video_stem | mota_pct | idf1_pct | hota_pct | precision_pct | recall_pct | idsw | fragments | tracklets | avg_tracklet_length_frames | gap_tolerant_fragments | gap_tolerant_tracklets | gap_tolerant_avg_tracklet_length_frames | gap_tolerant_suppressed_fragments | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | 50.99 | 62.12 | 53.19 | 99.05 | 51.55 | 10 | 282 | 290 | 25.59 | 78 | 86 | 86.29 | 204 | 71 | 6975 |
| Pigs291119_000263_30fps | 40.63 | 43.62 | 38.36 | 97.74 | 41.72 | 18 | 400 | 408 | 14.32 | 85 | 93 | 62.83 | 315 | 135 | 8163 |
| Pigs291119_000302_30fps | 67.07 | 73.32 | 66.69 | 99.59 | 67.42 | 10 | 235 | 243 | 39.75 | 55 | 63 | 153.32 | 180 | 40 | 4667 |
| ALL | 52.98 | 60.59 | 54.07 | 98.94 | 53.65 | 38 | 917 | 941 | 24.36 | 218 | 242 | 94.72 | 699 | 246 | 19805 |

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
| Pigs281119_000085_30fps | ID_1 | ID_4 | 368 | 7421 | 0.9160 |
| Pigs281119_000085_30fps | ID_2 | ID_3 | 325 | 7421 | 0.9160 |
| Pigs281119_000085_30fps | ID_3 | ID_1 | 942 | 7421 | 0.9160 |
| Pigs281119_000085_30fps | ID_4 | ID_2 | 1467 | 7421 | 0.9160 |
| Pigs281119_000085_30fps | ID_5 | ID_7 | 846 | 7421 | 0.9160 |
| Pigs281119_000085_30fps | ID_6 | ID_6 | 1421 | 7421 | 0.9160 |
| Pigs281119_000085_30fps | ID_7 | ID_8 | 376 | 7421 | 0.9160 |
| Pigs281119_000085_30fps | ID_8 | ID_5 | 1053 | 7421 | 0.9160 |
| Pigs291119_000263_30fps | ID_1 | ID_1 | 505 | 5843 | 0.7460 |
| Pigs291119_000263_30fps | ID_2 | ID_7 | 745 | 5843 | 0.7460 |
| Pigs291119_000263_30fps | ID_3 | ID_3 | 73 | 5843 | 0.7460 |
| Pigs291119_000263_30fps | ID_4 | ID_4 | 606 | 5843 | 0.7460 |
| Pigs291119_000263_30fps | ID_5 | ID_5 | 1428 | 5843 | 0.7460 |
| Pigs291119_000263_30fps | ID_6 | ID_6 | 258 | 5843 | 0.7460 |
| Pigs291119_000263_30fps | ID_7 | ID_8 | 331 | 5843 | 0.7460 |
| Pigs291119_000263_30fps | ID_8 | ID_2 | 413 | 5843 | 0.7460 |
| Pigs291119_000302_30fps | ID_1 | ID_3 | 1290 | 9659 | 0.9119 |
| Pigs291119_000302_30fps | ID_2 | ID_1 | 1788 | 9659 | 0.9119 |
| Pigs291119_000302_30fps | ID_3 | ID_2 | 1680 | 9659 | 0.9119 |
| Pigs291119_000302_30fps | ID_4 | ID_5 | 706 | 9659 | 0.9119 |

## Remapped Identity Diagnostics

- `tracking_remapped_identity_events.csv`: identity events after fixed ID remapping; these are continuity errors that remain after removing arbitrary initial ID numbering.
- Remapped identity event rows: `2970`
- Remapped ID switch rows: `38`

## Continuity Gap Diagnostics

- `tracking_continuity_gaps.csv`: matched-track gaps after fixed ID remapping. Gaps shorter than or equal to the configured tolerance are not counted as gap-tolerant fragments.
- Total matched-track gaps: `917`
- Tolerated gaps: `699`
- Remaining fragment gaps: `218`

| video_stem | gt_id | previous_matched_frame | next_matched_frame | gap_frames | tolerated | previous_pred_id | next_pred_id | id_changed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | ID_8 | 1 | 762 | 760 | False | ID_8 | ID_8 | False |
| Pigs291119_000263_30fps | ID_6 | 290 | 1049 | 758 | False | ID_6 | ID_6 | False |
| Pigs281119_000085_30fps | ID_3 | 1 | 614 | 612 | False | ID_4 | ID_3 | True |
| Pigs281119_000085_30fps | ID_5 | 26 | 538 | 511 | False | ID_5 | ID_5 | False |
| Pigs291119_000263_30fps | ID_8 | 773 | 1276 | 502 | False | ID_8 | ID_8 | False |
| Pigs291119_000302_30fps | ID_6 | 44 | 371 | 326 | False | ID_6 | ID_6 | False |
| Pigs291119_000302_30fps | ID_7 | 44 | 371 | 326 | False | ID_7 | ID_7 | False |
| Pigs291119_000263_30fps | ID_3 | 608 | 933 | 324 | False | ID_1 | ID_2 | True |
| Pigs291119_000302_30fps | ID_4 | 591 | 908 | 316 | False | ID_8 | ID_8 | False |
| Pigs291119_000263_30fps | ID_1 | 934 | 1208 | 273 | False | ID_1 | ID_1 | False |
| Pigs291119_000263_30fps | ID_2 | 903 | 1151 | 247 | False | ID_2 | ID_4 | True |
| Pigs291119_000263_30fps | ID_3 | 1287 | 1511 | 223 | False | ID_2 | ID_7 | True |
| Pigs291119_000263_30fps | ID_8 | 1354 | 1574 | 219 | False | ID_8 | ID_2 | True |
| Pigs281119_000085_30fps | ID_4 | 1 | 213 | 211 | False | ID_3 | ID_1 | True |
| Pigs291119_000302_30fps | ID_4 | 1361 | 1564 | 202 | False | ID_3 | ID_4 | True |
| Pigs281119_000085_30fps | ID_7 | 924 | 1121 | 196 | False | ID_7 | ID_7 | False |
| Pigs281119_000085_30fps | ID_8 | 924 | 1121 | 196 | False | ID_8 | ID_8 | False |
| Pigs291119_000263_30fps | ID_7 | 1087 | 1267 | 179 | False | ID_7 | ID_7 | False |
| Pigs291119_000263_30fps | ID_2 | 1201 | 1380 | 178 | False | ID_4 | ID_8 | True |
| Pigs291119_000263_30fps | ID_3 | 1090 | 1267 | 176 | False | ID_7 | ID_2 | True |

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
