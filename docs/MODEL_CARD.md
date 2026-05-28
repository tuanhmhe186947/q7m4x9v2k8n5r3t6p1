# Model Card

## Model

The default classifier uses a MobileNetV3Small image backbone. It can run as:

- image-only classifier
- hybrid classifier with image crops and six tabular context features

## Inputs

- RGB pig crop resized to `224x224`
- Optional tabular features:
  - `in_feeder`
  - `in_drinker`
  - `in_toy`
  - `speed_feat`
  - `min_dist_other`
  - `num_close_other`

## Outputs

Softmax probabilities over either fine or coarse behavior labels.

## Evaluation

The training pipeline writes metrics and confusion matrices under
`outputs/logs/`. Report test accuracy, macro F1, per-class F1, and the exact
data split seed when publishing results.

## Deployment

The export path produces FP32 and optional quantized TFLite models under
`outputs/export/`.

## Limitations

- Performance depends on detector and bounding box quality.
- The model should be validated on farms, cameras, lighting conditions, and pig
  ages not seen during training before operational use.
