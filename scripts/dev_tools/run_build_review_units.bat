@echo off
setlocal
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src

python scripts\behavior_review_tools\classification_v2_build_review_units.py ^
  --intervals-csv outputs\classification_v2\sequence_features\temporal_label_intervals.csv ^
  --sequence-window-manifest-csv outputs\classification_v2\sequence_features\sequence_window_manifest.csv ^
  --window-review-manifest-csv outputs\classification_v2\review_templates\full_review_manifest.csv ^
  --output-dir outputs\classification_v2\review_units ^
  --max-units-per-template 5000

endlocal
