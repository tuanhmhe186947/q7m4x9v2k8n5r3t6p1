# Reproducibility

## Environment

Recommended Python version:

```text
Python 3.11
```

Install:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .[pt,dev,notebooks]
```

## Artifacts

Expected local artifact paths:

```text
models/behavior/pig_behavior_sequence.pt
models/detector/pig_detector_yolo.pt
data/videos/pigs101219_full.mp4
```

Validate size and SHA256 against:

```text
artifacts/manifest.yaml
```

## Runtime Roles

- Detector: `models/detector/pig_detector_yolo.pt`
- Behavior sequence classifier: `models/behavior/pig_behavior_sequence.pt`
- Demo video: `data/videos/pigs101219_full.mp4`

## Demo Command

```bash
set PIG_BEHAVIOR_MODEL_BACKEND=pt
set PIG_BEHAVIOR_PT_MODEL_PATH=models\behavior\pig_behavior_sequence.pt
set PIG_BEHAVIOR_DETECT_MODEL_PATH=models\detector\pig_detector_yolo.pt
set PIG_BEHAVIOR_VIDEO_PATH=data\videos\pigs101219_full.mp4
python -m uvicorn pig_behavior.api:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/dashboard
```

## Validation

```bash
python -m compileall src main.py
ruff check src main.py tools tests
pytest -q
python tools/clean_notebooks.py --check notebooks
```

## Release Checklist

- Update external artifact URLs in `artifacts/manifest.yaml`.
- Record commit hash, hardware, Python version, package versions, and random
  seeds.
- Report detector metrics separately from behavior-classifier metrics.
- Confirm redistribution rights for videos, images, and model weights.
