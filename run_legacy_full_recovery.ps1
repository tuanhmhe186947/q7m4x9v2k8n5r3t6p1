$env:PYTHONPATH = 'src'
uv run --python 3.11 --no-project --with ultralytics python -m legacy_burst_recovery.main `
  --input-csv 'outputs/legacy_16f_rebuild/legacy_16f_rebuild_20260718_v2/04_cvat_inputs/legacy_center_keyframes_from_cvat.csv' `
  --legacy-burst-bbox-csv 'outputs/legacy_16f_rebuild/legacy_16f_rebuild_20260718_v2/04_cvat_inputs/legacy_six_anchor_bboxes_from_cvat.csv' `
  --drive-root 'G:\My Drive' `
  --output-root 'outputs/legacy_16f_rebuild/legacy_16f_rebuild_20260718_v2/06_full_recovery' `
  --detector-weights 'models/detector/pig_detector_yolov8.pt' `
  --scene-mask 'data/annotations/scene/mask.png' `
  --mask-filter-detections `
  --mask-min-bbox-coverage 0.50 `
  --mask-require-center-inside `
  --track-end-mode full_legacy_burst `
  --extract-crops `
  --sequence-views legacy_old_pattern_6 `
  --progress `
  --flush-every 500 `
  --log-file 'outputs/legacy_16f_rebuild/legacy_16f_rebuild_20260718_v2/06_full_recovery/recovery.log'
exit $LASTEXITCODE
