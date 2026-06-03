# Models

Model weights are local runtime artifacts and are ignored by Git. Keep the
tracked paths below empty until you download or copy the files described in
`artifacts/manifest.yaml`.

```text
models/behavior/pig_behavior_sequence.pt
models/detector/pig_detector_yolo.pt
```

Roles:

- `models/behavior/pig_behavior_sequence.pt` is the six-frame behavior sequence
  classifier.
- `models/detector/pig_detector_yolo.pt` is the YOLO detector/tracker weight
  file used to produce pig bounding boxes and track IDs.

Publish release weights through an external registry such as Zenodo, Hugging
Face, Kaggle, OSF, or GitHub Releases, then update `artifacts/manifest.yaml`
with stable URLs and checksums.
