# Scripts Layout

Canonical entrypoints in this folder:

- `track_videos.py`: run tracking on one or more videos
- `evaluate_tracking.py`: run tracking + GT evaluation
- `evaluate_best3_roboflow.py`: fixed 3-video benchmark
- `benchmark_tracking_weights.py`: multi-weight GT benchmark
- `benchmark_tracking_modes.py`: compare tracking modes side by side
- `eval_hard_scenes.py`: hard-scene identity diagnostics
- `detect_single_frame.py`: detector/frame debugging
- `run_roboflow_tracking.py`: Roboflow detection tracking flow
- `run_stable_tracking.py`: stable annotation tracking flow
- `connect_gdrive.py`: local dataset utility

Compatibility wrappers kept for old commands:

- `run_tracking.py`
- `eval_pipeline.py`
- `run_best3_yolov8_roboflow.py`
- `run_weight_tracking_gt_benchmark.py`

Grouped non-canonical scripts:

- `_internal/`: agent-only repository utilities
- `_legacy/`: old ad-hoc or preview scripts not part of the main workflow
- `_shortcuts/`: Windows `.bat` convenience launchers

Recommended commands:

```cmd
python scripts\track_videos.py -v Pigs291119_000263_30fps --mode hybrid_bytetrack
python scripts\evaluate_tracking.py -v Pigs291119_000263_30fps --mode hybrid_bytetrack
python scripts\evaluate_best3_roboflow.py --tag best3-roboflow
python scripts\benchmark_tracking_weights.py --mode hybrid_bytetrack
python scripts\benchmark_tracking_modes.py --weights models\detector\pig_detector_yolov8_roboflow.pt --video data\videos\Pigs291119_000263_30fps.mp4
```
