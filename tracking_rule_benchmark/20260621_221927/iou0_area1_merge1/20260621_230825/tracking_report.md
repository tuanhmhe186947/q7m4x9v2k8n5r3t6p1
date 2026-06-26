# Tracking Evaluation Report

## Run Config

- IoU threshold: `0.5`
- Include hidden boxes: `False`
- Gap tolerance frames: `15`
- Run missing tracker: `True`
- Force track: `True`
- Ground-truth directory: `C:\Users\ironh\Downloads\PIG_Behavior_Project\data\annotations\tracking`
- Prediction root: `C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\id_tracking\tracking_rule_benchmark\20260621_221927\iou0_area1_merge1`

## Summary

- Ground-truth videos found: `3`
- Videos evaluated: `3`
- Videos missing predictions: `0`

## Aggregate Metrics For Paper

| video_stem | remapped_mota_pct | remapped_idf1_pct | remapped_hota_pct | precision_pct | recall_pct | remapped_idsw | remapped_fragments | remapped_tracklets | remapped_avg_tracklet_length_frames | remapped_gap_tolerant_fragments | remapped_gap_tolerant_tracklets | remapped_gap_tolerant_avg_tracklet_length_frames | remapped_gap_tolerant_suppressed_fragments | idmap_coverage_pct | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 51.43 | 57.88 | 51.61 | 98.65 | 52.38 | 98 | 1246 | 1270 | 17.62 | 246 | 270 | 82.89 | 1000 | 84.58 | 307 | 20347 |

## Per-Video Metrics For Paper

| video_stem | remapped_mota_pct | remapped_idf1_pct | remapped_hota_pct | precision_pct | recall_pct | remapped_idsw | remapped_fragments | remapped_tracklets | remapped_avg_tracklet_length_frames | remapped_gap_tolerant_fragments | remapped_gap_tolerant_tracklets | remapped_gap_tolerant_avg_tracklet_length_frames | remapped_gap_tolerant_suppressed_fragments | idmap_coverage_pct | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | 45.98 | 48.77 | 44.99 | 98.06 | 47.05 | 20 | 299 | 307 | 22.06 | 71 | 79 | 85.73 | 228 | 76.70 | 134 | 7623 |
| Pigs291119_000263_30fps | 51.95 | 57.40 | 47.87 | 97.92 | 53.52 | 61 | 667 | 675 | 11.11 | 101 | 109 | 68.77 | 566 | 82.94 | 159 | 6510 |
| Pigs291119_000302_30fps | 56.41 | 66.97 | 60.66 | 99.83 | 56.62 | 17 | 280 | 288 | 28.17 | 74 | 82 | 98.93 | 206 | 92.68 | 14 | 6214 |

## Raw Absolute-ID Metrics For Audit

These strict metrics compare literal ID names before global remapping. Use them to audit CVAT/tracker naming issues, not as the main paper conclusion when initial ID numbering is arbitrary.

| video_stem | mota_pct | idf1_pct | hota_pct | precision_pct | recall_pct | idsw | fragments | tracklets | avg_tracklet_length_frames | gap_tolerant_fragments | gap_tolerant_tracklets | gap_tolerant_avg_tracklet_length_frames | gap_tolerant_suppressed_fragments | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | 45.98 | 48.77 | 44.99 | 98.06 | 47.05 | 20 | 299 | 307 | 22.06 | 71 | 79 | 85.73 | 228 | 134 | 7623 |
| Pigs291119_000263_30fps | 51.95 | 57.40 | 47.87 | 97.92 | 53.52 | 61 | 667 | 675 | 11.11 | 101 | 109 | 68.77 | 566 | 159 | 6510 |
| Pigs291119_000302_30fps | 56.41 | 66.97 | 60.66 | 99.83 | 56.62 | 17 | 280 | 288 | 28.17 | 74 | 82 | 98.93 | 206 | 14 | 6214 |
| ALL | 51.43 | 57.88 | 51.61 | 98.65 | 52.38 | 98 | 1246 | 1270 | 17.62 | 246 | 270 | 82.89 | 1000 | 307 | 20347 |

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
| Pigs281119_000085_30fps | ID_1 | ID_3 | 2 | 6773 | 0.7670 |
| Pigs281119_000085_30fps | ID_2 | ID_4 | 179 | 6773 | 0.7670 |
| Pigs281119_000085_30fps | ID_3 | ID_1 | 486 | 6773 | 0.7670 |
| Pigs281119_000085_30fps | ID_4 | ID_2 | 1643 | 6773 | 0.7670 |
| Pigs281119_000085_30fps | ID_5 | ID_7 | 903 | 6773 | 0.7670 |
| Pigs281119_000085_30fps | ID_6 | ID_6 | 747 | 6773 | 0.7670 |
| Pigs281119_000085_30fps | ID_7 | ID_8 | 355 | 6773 | 0.7670 |
| Pigs281119_000085_30fps | ID_8 | ID_5 | 880 | 6773 | 0.7670 |
| Pigs291119_000263_30fps | ID_1 | ID_8 | 657 | 7496 | 0.8294 |
| Pigs291119_000263_30fps | ID_2 | ID_2 | 1182 | 7496 | 0.8294 |
| Pigs291119_000263_30fps | ID_3 | ID_7 | 820 | 7496 | 0.8294 |
| Pigs291119_000263_30fps | ID_4 | ID_4 | 915 | 7496 | 0.8294 |
| Pigs291119_000263_30fps | ID_5 | ID_5 | 995 | 7496 | 0.8294 |
| Pigs291119_000263_30fps | ID_6 | ID_6 | 1282 | 7496 | 0.8294 |
| Pigs291119_000263_30fps | ID_7 | ID_3 | 110 | 7496 | 0.8294 |
| Pigs291119_000263_30fps | ID_8 | ID_1 | 256 | 7496 | 0.8294 |
| Pigs291119_000302_30fps | ID_1 | ID_3 | 1097 | 8112 | 0.9268 |
| Pigs291119_000302_30fps | ID_2 | ID_1 | 1788 | 8112 | 0.9268 |
| Pigs291119_000302_30fps | ID_3 | ID_2 | 1595 | 8112 | 0.9268 |
| Pigs291119_000302_30fps | ID_4 | ID_8 | 159 | 8112 | 0.9268 |

## Remapped Identity Diagnostics

- `tracking_remapped_identity_events.csv`: identity events after fixed ID remapping; these are continuity errors that remain after removing arbitrary initial ID numbering.
- Remapped identity event rows: `3484`
- Remapped ID switch rows: `98`

## Continuity Gap Diagnostics

- `tracking_continuity_gaps.csv`: matched-track gaps after fixed ID remapping. Gaps shorter than or equal to the configured tolerance are not counted as gap-tolerant fragments.
- Total matched-track gaps: `1246`
- Tolerated gaps: `1000`
- Remaining fragment gaps: `246`

| video_stem | gt_id | previous_matched_frame | next_matched_frame | gap_frames | tolerated | previous_pred_id | next_pred_id | id_changed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pigs281119_000085_30fps | ID_8 | 1 | 762 | 760 | False | ID_8 | ID_8 | False |
| Pigs281119_000085_30fps | ID_3 | 1 | 755 | 753 | False | ID_3 | ID_3 | False |
| Pigs291119_000302_30fps | ID_5 | 934 | 1488 | 553 | False | ID_5 | ID_5 | False |
| Pigs291119_000302_30fps | ID_8 | 964 | 1488 | 523 | False | ID_7 | ID_8 | True |
| Pigs281119_000085_30fps | ID_5 | 26 | 538 | 511 | False | ID_5 | ID_5 | False |
| Pigs291119_000302_30fps | ID_8 | 499 | 931 | 431 | False | ID_4 | ID_8 | True |
| Pigs281119_000085_30fps | ID_3 | 1070 | 1459 | 388 | False | ID_6 | ID_1 | True |
| Pigs291119_000263_30fps | ID_1 | 1078 | 1467 | 388 | False | ID_7 | ID_7 | False |
| Pigs281119_000085_30fps | ID_6 | 1077 | 1459 | 381 | False | ID_3 | ID_3 | False |
| Pigs281119_000085_30fps | ID_4 | 1241 | 1615 | 373 | False | ID_1 | ID_4 | True |
| Pigs291119_000302_30fps | ID_6 | 44 | 371 | 326 | False | ID_6 | ID_6 | False |
| Pigs291119_000302_30fps | ID_7 | 44 | 371 | 326 | False | ID_7 | ID_7 | False |
| Pigs291119_000263_30fps | ID_1 | 616 | 885 | 268 | False | ID_3 | ID_2 | True |
| Pigs291119_000263_30fps | ID_7 | 1468 | 1697 | 228 | False | ID_7 | ID_4 | True |
| Pigs281119_000085_30fps | ID_1 | 683 | 910 | 226 | False | ID_1 | ID_6 | True |
| Pigs291119_000263_30fps | ID_1 | 1498 | 1721 | 222 | False | ID_7 | ID_4 | True |
| Pigs281119_000085_30fps | ID_4 | 1 | 213 | 211 | False | ID_4 | ID_1 | True |
| Pigs281119_000085_30fps | ID_4 | 760 | 947 | 186 | False | ID_3 | ID_6 | True |
| Pigs291119_000302_30fps | ID_5 | 29 | 197 | 167 | False | ID_5 | ID_5 | False |
| Pigs291119_000302_30fps | ID_8 | 29 | 197 | 167 | False | ID_4 | ID_4 | False |

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
