# Tracking Evaluation Report

## Run Config

- IoU threshold: `0.5`
- Include hidden boxes: `False`
- Gap tolerance frames: `15`
- Run missing tracker: `True`
- Force track: `True`
- Ground-truth directory: `C:\Users\ironh\Downloads\PIG_Behavior_Project\data\annotations\tracking`
- Prediction root: `C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\id_tracking\tracking_rule_benchmark\20260621_221927\iou1_area0_merge1`

## Summary

- Ground-truth videos found: `3`
- Videos evaluated: `3`
- Videos missing predictions: `0`

## Aggregate Metrics For Paper

| video_stem | remapped_mota_pct | remapped_idf1_pct | remapped_hota_pct | precision_pct | recall_pct | remapped_idsw | remapped_fragments | remapped_tracklets | remapped_avg_tracklet_length_frames | remapped_gap_tolerant_fragments | remapped_gap_tolerant_tracklets | remapped_gap_tolerant_avg_tracklet_length_frames | remapped_gap_tolerant_suppressed_fragments | idmap_coverage_pct | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 51.21 | 58.10 | 52.48 | 98.90 | 51.88 | 38 | 989 | 1013 | 21.88 | 238 | 262 | 84.61 | 751 | 85.37 | 247 | 20561 |

## Per-Video Metrics For Paper

| video_stem | remapped_mota_pct | remapped_idf1_pct | remapped_hota_pct | precision_pct | recall_pct | remapped_idsw | remapped_fragments | remapped_tracklets | remapped_avg_tracklet_length_frames | remapped_gap_tolerant_fragments | remapped_gap_tolerant_tracklets | remapped_gap_tolerant_avg_tracklet_length_frames | remapped_gap_tolerant_suppressed_fragments | idmap_coverage_pct | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | 51.15 | 62.14 | 53.22 | 99.06 | 51.72 | 11 | 289 | 297 | 25.07 | 79 | 87 | 85.57 | 210 | 91.44 | 71 | 6951 |
| Pigs291119_000263_30fps | 40.72 | 43.73 | 38.44 | 97.75 | 41.81 | 18 | 400 | 408 | 14.35 | 85 | 93 | 62.97 | 315 | 74.66 | 135 | 8150 |
| Pigs291119_000302_30fps | 61.54 | 66.67 | 62.70 | 99.54 | 61.89 | 9 | 300 | 308 | 28.79 | 74 | 82 | 108.12 | 226 | 87.36 | 41 | 5460 |

## Raw Absolute-ID Metrics For Audit

These strict metrics compare literal ID names before global remapping. Use them to audit CVAT/tracker naming issues, not as the main paper conclusion when initial ID numbering is arbitrary.

| video_stem | mota_pct | idf1_pct | hota_pct | precision_pct | recall_pct | idsw | fragments | tracklets | avg_tracklet_length_frames | gap_tolerant_fragments | gap_tolerant_tracklets | gap_tolerant_avg_tracklet_length_frames | gap_tolerant_suppressed_fragments | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | 51.15 | 62.14 | 53.22 | 99.06 | 51.72 | 11 | 289 | 297 | 25.07 | 79 | 87 | 85.57 | 210 | 71 | 6951 |
| Pigs291119_000263_30fps | 40.72 | 43.73 | 38.44 | 97.75 | 41.81 | 18 | 400 | 408 | 14.35 | 85 | 93 | 62.97 | 315 | 135 | 8150 |
| Pigs291119_000302_30fps | 61.54 | 66.67 | 62.70 | 99.54 | 61.89 | 9 | 300 | 308 | 28.79 | 74 | 82 | 108.12 | 226 | 41 | 5460 |
| ALL | 51.21 | 58.10 | 52.48 | 98.90 | 51.88 | 38 | 989 | 1013 | 21.88 | 238 | 262 | 84.61 | 751 | 247 | 20561 |

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
| Pigs281119_000085_30fps | ID_1 | ID_4 | 368 | 7445 | 0.9144 |
| Pigs281119_000085_30fps | ID_2 | ID_3 | 335 | 7445 | 0.9144 |
| Pigs281119_000085_30fps | ID_3 | ID_1 | 942 | 7445 | 0.9144 |
| Pigs281119_000085_30fps | ID_4 | ID_2 | 1467 | 7445 | 0.9144 |
| Pigs281119_000085_30fps | ID_5 | ID_7 | 846 | 7445 | 0.9144 |
| Pigs281119_000085_30fps | ID_6 | ID_6 | 1421 | 7445 | 0.9144 |
| Pigs281119_000085_30fps | ID_7 | ID_8 | 376 | 7445 | 0.9144 |
| Pigs281119_000085_30fps | ID_8 | ID_5 | 1053 | 7445 | 0.9144 |
| Pigs291119_000263_30fps | ID_1 | ID_1 | 505 | 5856 | 0.7466 |
| Pigs291119_000263_30fps | ID_2 | ID_7 | 745 | 5856 | 0.7466 |
| Pigs291119_000263_30fps | ID_3 | ID_3 | 73 | 5856 | 0.7466 |
| Pigs291119_000263_30fps | ID_4 | ID_4 | 619 | 5856 | 0.7466 |
| Pigs291119_000263_30fps | ID_5 | ID_5 | 1428 | 5856 | 0.7466 |
| Pigs291119_000263_30fps | ID_6 | ID_6 | 258 | 5856 | 0.7466 |
| Pigs291119_000263_30fps | ID_7 | ID_8 | 331 | 5856 | 0.7466 |
| Pigs291119_000263_30fps | ID_8 | ID_2 | 413 | 5856 | 0.7466 |
| Pigs291119_000302_30fps | ID_1 | ID_3 | 1289 | 8866 | 0.8736 |
| Pigs291119_000302_30fps | ID_2 | ID_1 | 1788 | 8866 | 0.8736 |
| Pigs291119_000302_30fps | ID_3 | ID_2 | 1625 | 8866 | 0.8736 |
| Pigs291119_000302_30fps | ID_4 | ID_5 | 272 | 8866 | 0.8736 |

## Remapped Identity Diagnostics

- `tracking_remapped_identity_events.csv`: identity events after fixed ID remapping; these are continuity errors that remain after removing arbitrary initial ID numbering.
- Remapped identity event rows: `3252`
- Remapped ID switch rows: `38`

## Continuity Gap Diagnostics

- `tracking_continuity_gaps.csv`: matched-track gaps after fixed ID remapping. Gaps shorter than or equal to the configured tolerance are not counted as gap-tolerant fragments.
- Total matched-track gaps: `989`
- Tolerated gaps: `751`
- Remaining fragment gaps: `238`

| video_stem | gt_id | previous_matched_frame | next_matched_frame | gap_frames | tolerated | previous_pred_id | next_pred_id | id_changed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | ID_8 | 1 | 762 | 760 | False | ID_8 | ID_8 | False |
| Pigs291119_000263_30fps | ID_6 | 290 | 1049 | 758 | False | ID_6 | ID_6 | False |
| Pigs281119_000085_30fps | ID_5 | 26 | 538 | 511 | False | ID_5 | ID_5 | False |
| Pigs291119_000263_30fps | ID_8 | 773 | 1276 | 502 | False | ID_8 | ID_8 | False |
| Pigs291119_000302_30fps | ID_6 | 44 | 371 | 326 | False | ID_6 | ID_6 | False |
| Pigs291119_000302_30fps | ID_7 | 44 | 371 | 326 | False | ID_7 | ID_7 | False |
| Pigs291119_000263_30fps | ID_3 | 608 | 933 | 324 | False | ID_1 | ID_2 | True |
| Pigs291119_000302_30fps | ID_4 | 591 | 908 | 316 | False | ID_4 | ID_4 | False |
| Pigs281119_000085_30fps | ID_3 | 1 | 315 | 313 | False | ID_4 | ID_3 | True |
| Pigs291119_000263_30fps | ID_1 | 934 | 1208 | 273 | False | ID_1 | ID_1 | False |
| Pigs281119_000085_30fps | ID_3 | 346 | 614 | 267 | False | ID_3 | ID_3 | False |
| Pigs291119_000263_30fps | ID_2 | 903 | 1151 | 247 | False | ID_2 | ID_4 | True |
| Pigs291119_000263_30fps | ID_3 | 1287 | 1511 | 223 | False | ID_2 | ID_7 | True |
| Pigs291119_000263_30fps | ID_8 | 1354 | 1574 | 219 | False | ID_8 | ID_2 | True |
| Pigs281119_000085_30fps | ID_4 | 1 | 213 | 211 | False | ID_3 | ID_1 | True |
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
