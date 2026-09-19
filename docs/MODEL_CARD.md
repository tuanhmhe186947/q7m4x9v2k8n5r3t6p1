# Model Card

## Models

This project uses two separate runtime models:

- `models/detector/pig_detector_yolo.pt`: YOLO detector/tracker weights. This
  model detects pigs and provides bounding boxes and track IDs.
- `models/behavior/pig_behavior_sequence.pt`: PyTorch behavior sequence
  classifier. This model classifies behavior from six pig crops and tabular
  context features.

The detector output feeds the behavior classifier. The behavior classifier is
not used for detection or tracking.

## Behavior Classifier Input

One prediction uses a temporal window of six cropped pig frames:

```text
offsets = [-3, -2, -1, 0, 1, 2] * behavior_stride_frames
default behavior_stride_frames = 3
```

Each crop is resized to `224x224` RGB. Each timestep also includes:

```text
cx_n, cy_n, bw_n, bh_n, speed_feat,
min_dist_other, num_close_other, in_feeder, in_drinker, in_toy
```

## Behavior Classifier Output

Softmax probabilities over:

```text
drink, eat, fight, social-nose, explore,
lying, stand, move, sitting, playwithtoy
```

## Detector Input and Output

Input is a video frame. Output is one or more pig detections with confidence,
bounding box coordinates, class ID, and track ID when tracking is active.

## Evaluation

Report detector precision/recall or MOT metrics separately from behavior
classification metrics. For behavior classification, report accuracy, macro F1,
per-class F1, confusion matrix, split seed, artifact checksum, and exact commit.

## Deployment

The FastAPI dashboard loads both `.pt` artifacts. Exported TFLite models under
`outputs/export/` are still supported for the older single-crop classifier API.

## Limitations

- Behavior predictions depend on detector quality and stable track IDs.
- The sequence classifier waits for future frames, so live labels are delayed.
- Farm layout, camera angle, lighting, occlusion, and animal age may cause
  distribution shift.
- Do not use this system for animal welfare decisions without validation by
  domain experts.
