# Pig Behavior Project

AI research code for pig detection, tracking, and behavior recognition from
video. The maintained runtime code lives under `src/pig_behavior`; notebooks are
kept as archived experiment history.

## Classification V2 status

Classification V2 Phases 1–4 are accepted at implementation SHA
`76a0458e39769d3e7fac865dd16439a0ed3c3a04`. The exact-SHA Phase 1–4
integrated acceptance audit passed, and TuanHM approved the human sign-off on
2026-07-24.

Lineage rebuild planning is authorized. No rebuild, review GUI, evidence
generation, export, model execution, or training is authorized by that
acceptance. Canonical operational details remain in
`.agents/memory/02_CURRENT_DECISION.md` and `.agents/memory/08_WORKFLOW.md`.

## Repository Layout

```text
.
|-- artifacts/              # Artifact manifest with checksums and URL slots
|-- data/
|   |-- annotations/        # Source annotations grouped by workflow
|   |-- processed/          # Generated datasets grouped by workflow/run
|   |-- raw/                # Local extracted images, ignored by Git
|   `-- videos/             # Local videos, ignored by Git
|-- docs/                   # Model card, dataset card, reproducibility notes
|-- models/
|   |-- behavior/           # Behavior sequence classifier weights, ignored
|   `-- detector/           # YOLO detector weights, ignored
|-- notebooks/              # Archived research notebooks
|-- src/pig_behavior/       # Installable package
|   |-- api/                # FastAPI app, routes, schemas, dashboard HTML
|   |-- data_preparation/   # CVAT cleaning and tracking annotation workflows
|   |-- evaluation/         # Tracking metrics and report generation
|   |-- models/             # Model architectures and checkpoint loaders
|   `-- services/           # Inference, detection, and video tracking services
|-- tests/
`-- tools/
```

Large `.pt` and `.mp4` files are not committed. Publish them through an
external registry, then update `artifacts/manifest.yaml`.

## Runtime Artifacts

Place local artifacts at these paths:

```text
models/behavior/pig_behavior_sequence.pt
models/detector/pig_detector_yolo.pt
data/videos/pigs101219_full.mp4
```

Roles:

- `pig_behavior_sequence.pt` is the behavior sequence classifier.
- `pig_detector_yolo.pt` is the detector/tracker model for bounding boxes and
  track IDs.
- `pigs101219_full.mp4` is the demo video for the dashboard.

Verify files against `artifacts/manifest.yaml` before running experiments.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .[pt]
```

For notebooks and development checks:

```bash
pip install -r requirements-dev.txt
```

## Tracking Scripts

Detailed tracking, evaluation, optimizer, benchmark, and debug commands are kept in `scripts/README.md`.
Use that file as the source of truth for command order and current opt-in tracking candidates.

Current command split:

- `scripts\track_videos.py`: tracking-only runs and CVAT prediction/XML export.
- `scripts\evaluate_tracking.py`: tracking plus GT evaluation metrics.
- `scripts\optimize_tracking_metrics.py`: automated tracking config search.

The current best hard 5-video candidate is still opt-in. It is documented in `scripts/README.md` with the full `--profile-override` stack and has not been promoted into the base/default config yet.
## Dashboard

Start the API:

```bash
set PIG_BEHAVIOR_MODEL_BACKEND=pt
set PIG_BEHAVIOR_PT_MODEL_PATH=models\behavior\pig_behavior_sequence.pt
set PIG_BEHAVIOR_DETECT_MODEL_PATH=models\detector\pig_detector_yolo.pt
set PIG_BEHAVIOR_VIDEO_PATH=data\videos\pigs101219_full.mp4
set PIG_BEHAVIOR_BEHAVIOR_STRIDE_FRAMES=3
pig-behavior-api
```

Open:

```text
http://127.0.0.1:8000/dashboard
```

The dashboard pipeline is:

```text
video frame
  -> YOLO detector/tracker creates pig boxes and track IDs
  -> collect each tracked pig's temporal crop sequence
  -> behavior classifier receives 6 cropped frames plus tabular features
  -> dashboard aggregates behavior counts over time
```

The behavior classifier follows the notebook training contract:

```text
sequence_length = 6
offsets = [-3, -2, -1, 0, 1, 2] * behavior_stride_frames
default behavior_stride_frames = 3
```

Default frame window around the center frame:

```text
center-9, center-6, center-3, center, center+3, center+6
```

Behavior labels appear with a small delay because the window includes future
frames.

## API

Run locally:

```bash
pig-behavior-api
```

The service exposes:

- `GET /` and `GET /metadata`
- `GET /health`
- `GET /ready`
- `POST /predict`
- `GET /dashboard`
- `POST /tracking/start`
- `POST /tracking/stop`
- `GET /tracking/status`
- `GET /tracking/stream`

`uvicorn pig_behavior.api:app` remains supported.

## CLI

Build classification training data from CVAT native exports:

```bash
pig-build-classification-data
```

Generate tracking predictions for annotation/evaluation:

```bash
pig-track-for-annotation --video data\videos\Pigs281119_000085_30fps.mp4
```

Evaluate tracking predictions against CVAT XML ground truth:

```bash
pig-tracking-eval --run-missing-tracker
```

Training smoke test:

```bash
pig-behavior --mode train --dry-run
```

Export TFLite models:

```bash
pig-behavior --mode export
```

Behavior classifier inference from one crop uses padded sequence mode:

```bash
pig-behavior --mode infer ^
  --backend pt ^
  --pt-model models\behavior\pig_behavior_sequence.pt ^
  --image data\raw\images_clean\example.jpg
```

## Roboflow Workflow Integration

You can integrate Roboflow's serverless workflows to run detections side-by-side or as an alternative to the local YOLOv8 pipeline. The project integrates the "Detect, Count, and Visualize 3" workflow.

### Setup API Key

To run the Roboflow workflow, you must provide your Roboflow API key. You can pass it via command-line arguments or define the `ROBOFLOW_API_KEY` environment variable:

```cmd
set ROBOFLOW_API_KEY=your_api_key_here
```

### Running Detection on a Single Frame

To detect pigs on a single frame (e.g., frame 979) with a workspace mask applied:

```cmd
python scripts\detect_pig_frame.py ^
  --roboflow ^
  --roboflow-api-key your_api_key_here ^
  --start-frame 979
```

This will call the Roboflow Workflow API and save the annotated visualization image to:
`outputs\detections\detect_frame_979_roboflow.png`

### Running Detection on a Frame Range

To process a range of frames and generate a comparison video:

```cmd
python scripts\detect_pig_frame.py ^
  --roboflow ^
  --start-frame 800 ^
  --end-frame 1000 ^
  --save-images
```

The output video will be saved to:
`outputs\detections\detect_range_800_1000_roboflow.mp4`

### Running Integration Tests

To run the smoke tests for Roboflow integration, make sure your API key is set in the environment:

```cmd
set ROBOFLOW_API_KEY=your_api_key_here
pytest tests\test_roboflow_integration.py
```

## Docker

```bash
docker compose up --build
```

Compose mounts:

```text
./models/behavior/pig_behavior_sequence.pt
./models/detector/pig_detector_yolo.pt
./data/videos/pigs101219_full.mp4
./outputs
```

## Data Contract

Default processed CSV resolver:

```text
data/processed/classification/<YYYYMMDD_HHMMSS>/behavior_with_feats_rectROI.csv
```

Training images:

```text
data/raw/images_clean/
```

Required tabular sequence features:

```text
cx_n, cy_n, bw_n, bh_n, speed_feat,
min_dist_other, num_close_other, in_feeder, in_drinker, in_toy
```

Annotation folders:

```text
data/annotations/roi/             # static feeder/drinker/toy ROI COCO
data/annotations/scene/           # background.png and mask.png
data/annotations/tracking/        # CVAT video XML ground truth
data/annotations/classification/  # classification label assets
data/annotations/schemas/         # CVAT label schemas
```

## Quality Checks

```bash
python -m compileall src main.py
ruff check src main.py tools tests
pytest -q
python tools/clean_notebooks.py --check notebooks
```

## Release Notes

Code is MIT licensed. Model and video artifacts are marked research-use until
redistribution rights are confirmed. See `docs/MODEL_CARD.md`,
`docs/DATASET_CARD.md`, and `docs/reproducibility.md`.
