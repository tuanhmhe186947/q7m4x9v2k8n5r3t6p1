# Notebook Index

The notebooks are archived experiment workflows or thin wrappers. Reusable
logic belongs under `src/pig_behavior`; new automation should call the package
entry points instead of importing notebook files.

## Data Preparation

- `01_data_preparation/video_to_frame_phase_1.ipynb`
- `01_data_preparation/video_to_frame_phase_2.ipynb`
- `01_data_preparation/video_to_frame_annotate.ipynb`
- `01_data_preparation/clean_cvat_json_to_csv.ipynb`
- `01_data_preparation/clean_cvat_json_to_csv.py` wraps
  `pig_behavior.data_preparation.classification_dataset`.
- `01_data_preparation/track_video_ids_for_annotation.py` wraps
  `pig_behavior.data_preparation.tracking_annotation`.
- `01_data_preparation/update_ids_for_annotation.ipynb`
- `01_data_preparation/export_behavior_extra_features.ipynb`

## Training

- `02_training/yolov8_train.ipynb`
- `02_training/train_group_behavior_model.ipynb`
- `02_training/train_behavior_classifier_kaggle.ipynb`
- `02_training/train_behavior_classifier_experiment.ipynb`

## Detection And Tracking

- `03_detection_tracking/annotate_image.ipynb`
- `03_detection_tracking/video_bbox_tracking.ipynb`

## Evaluation

- `04_evaluation/pig_tracking_for_mota.ipynb`
- `04_evaluation/pig_tracking_for_mota.py` wraps
  `pig_behavior.evaluation.tracking_metrics`.

## Maintained Commands

```bash
pig-build-classification-data
pig-track-for-annotation --video data/videos/Pigs281119_000085_30fps.mp4
pig-tracking-eval --run-missing-tracker
```
