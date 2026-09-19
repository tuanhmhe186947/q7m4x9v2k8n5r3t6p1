# Data Directory

This directory separates publishable metadata from large local research data.

- `data/`: native CVAT task exports used to build classification CSVs.
- `raw/`: extracted images consumed by training. Ignored by Git.
- `processed/`: generated tabular datasets grouped by workflow and run time.
- `annotations/`: source annotation assets grouped by purpose.
- `videos/`: local demo or research videos. Ignored by Git except README files.

The maintained classification flow writes:

```text
processed/classification/<YYYYMMDD_HHMMSS>/behavior_clean_merged.csv
processed/classification/<YYYYMMDD_HHMMSS>/behavior_with_feats_rectROI.csv
```

`pig_behavior.config.TrainConfig` resolves the newest
`behavior_with_feats_rectROI.csv` run by default. Matching image files live
under `raw/images_clean/`.

Render the tracked annotation preview:

```powershell
$env:PYTHONPATH = "$PWD\src"
$pythonCode = @'
import json
from pathlib import Path
from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.visualization import render_annotation_video

video = Path(r"data/videos/video_test_tracking/Pigs011219_0448.mp4")
root = Path(
    r"outputs/id_tracking/video_test_tracking/Pigs011219_0448/"
    r"hybrid_bytetrack/Pigs011219_0448"
)
shapes = json.loads(
    (root / "annotations_cvat_shapes.json").read_text(encoding="utf-8")
)[0]["shapes"]
output = root / "tracked_preview.mp4"
config = TrackingConfig(
    video_path=video,
    output_video=output,
    output_dir=root,
    mask_path=Path(r"data/annotations/scene/mask.png"),
    mode="hybrid_bytetrack",
    output_fps=30.0,
)
print(f"Rendered {render_annotation_video(video, output, shapes, config)} frames -> {output}")
'@
python -c $pythonCode
```

## Dataset Availability and Reproduction Notice

Due to privacy, commercial farm agreements, and file size constraints, raw video
recordings and full CVAT annotation XML bundles are not distributed directly within
the Git repository.

1. **Placing Local Video Assets**: Place raw evaluation videos under `data/videos/`
   matching the filenames defined in `docs/paper/dataset_role_table.csv`.
2. **Placing Annotations**: Ground-truth CVAT XML files should be placed under
   `data/annotations/tracking/` or `data/annotations/classification/`.
3. **Schemas and Masks**: Scene polygon masks (`data/annotations/scene/mask.png`)
   and ROI annotations (`data/annotations/roi/ROI_annotations.coco.json`) are
   tracked in the repository to support immediate feature extraction and testing.
