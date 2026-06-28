# Project Memory Short

- Do not hardcode `000263`/`000302` as optimizer target videos anymore.
- For optimizer ranking defaults, derive weak target videos from:
  `outputs/eval/hybrid_bytetrack/Tracking mới tắt smooth/yolov8/iou0_area0_condarea0_merge0/tracking_metrics.csv`
- Current weakest set from that file is:
  - `Pigs291119_000263_30fps`
  - `Pigs291119_000226_30fps`
  - `Pigs301119_000327_30fps`
  - `Pigs301119_000328_30fps`
- Stop using legacy 21/06 as a decision reference.
- Current runtime variants to compare are:
  - C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\eval\hybrid_bytetrack\Tracking moi tat smooth\yolov8
  - C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\eval\hybrid_bytetrack\Tracking moi bat smooth
- In reports, no smooth is currently worse than smooth; do not assume the unsmoothed runtime is the better baseline.
- Optimizer default scopes should stay tracking-focused.
- Detector-only presets were moved to explicit `--scope detector_probe` because `overnight_iou0` showed detector-only metrics were identical to `base`.
- Continue focusing on code and runtime behavior in association.py and runner.py, not detector weight.
