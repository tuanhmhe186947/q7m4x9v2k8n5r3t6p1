# Tracking Rule Flag Benchmark

- Output folder: `C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\evaluation\tracking_metrics\tracking_rule_benchmark\20260621_221927`
- Flag combinations: `8`
- Detailed metric rows: `32`

## Aggregate Metrics

| combo | USE_IOU_FALLBACK | USE_AREA_OCCLUSION_FREEZE | USE_MERGED_BOX_SPLIT | elapsed_sec | fps_evaluated_frames | remapped_mota_pct | remapped_idf1_pct | remapped_hota_pct | remapped_idsw | remapped_fragments | remapped_gap_tolerant_fragments | fp | fn | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iou0_area0_merge0 | False | False | False | 719.38 | 7.51 | 98.69 | 95.43 | 95.91 | 4 | 127 | 9 | 342 | 212 | C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\evaluation\tracking_metrics\tracking_rule_benchmark\20260621_221927\iou0_area0_merge0\20260621_223125 |
| iou0_area0_merge1 | False | False | True | 851.80 | 6.34 | 52.98 | 60.59 | 54.07 | 38 | 917 | 218 | 246 | 19805 | C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\evaluation\tracking_metrics\tracking_rule_benchmark\20260621_221927\iou0_area0_merge1\20260621_224536 |
| iou0_area1_merge0 | False | True | False | 901.51 | 5.99 | 89.33 | 80.59 | 79.09 | 139 | 344 | 33 | 472 | 3950 | C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\evaluation\tracking_metrics\tracking_rule_benchmark\20260621_221927\iou0_area1_merge0\20260621_230039 |
| iou0_area1_merge1 | False | True | True | 465.77 | 11.59 | 51.43 | 57.88 | 51.61 | 98 | 1246 | 246 | 307 | 20347 | C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\evaluation\tracking_metrics\tracking_rule_benchmark\20260621_221927\iou0_area1_merge1\20260621_230825 |
| iou1_area0_merge0 | True | False | False | 622.79 | 8.67 | 98.70 | 95.43 | 95.92 | 4 | 126 | 8 | 352 | 200 | C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\evaluation\tracking_metrics\tracking_rule_benchmark\20260621_221927\iou1_area0_merge0\20260621_231848 |
| iou1_area0_merge1 | True | False | True | 507.81 | 10.63 | 51.21 | 58.10 | 52.48 | 38 | 989 | 238 | 247 | 20561 | C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\evaluation\tracking_metrics\tracking_rule_benchmark\20260621_221927\iou1_area0_merge1\20260621_232716 |
| iou1_area1_merge0 | True | True | False | 444.13 | 12.16 | 89.92 | 80.86 | 79.74 | 144 | 334 | 35 | 467 | 3697 | C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\evaluation\tracking_metrics\tracking_rule_benchmark\20260621_221927\iou1_area1_merge0\20260621_233440 |
| iou1_area1_merge1 | True | True | True | 419.63 | 12.87 | 51.25 | 56.69 | 50.84 | 98 | 1251 | 255 | 307 | 20425 | C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\evaluation\tracking_metrics\tracking_rule_benchmark\20260621_221927\iou1_area1_merge1\20260621_234139 |

## Files

- `tracking_rule_benchmark_summary.csv`: one aggregate row per combo.
- `tracking_rule_benchmark_detailed_metrics.csv`: all per-video and ALL rows.
- Each combo folder contains the normal `tracking_report.md` and diagnostics.
