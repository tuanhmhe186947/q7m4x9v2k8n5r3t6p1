# Dataset Card

## Dataset Summary

The project uses extracted pig video frames, bounding boxes, behavior labels,
scene ROI annotations, and engineered temporal/context features.

## Local Paths

```text
data/raw/images_clean/
data/processed/behavior_with_feats_rectROI.csv
data/annotations/
data/videos/pigs101219_full.mp4
```

Raw images and videos are local research artifacts and are ignored by Git.

## Labels

Behavior sequence labels:

```text
drink, eat, fight, social-nose, explore,
lying, stand, move, sitting, playwithtoy
```

Legacy single-crop classifier labels:

```text
lying, eat, drink, explore, sitting, stand, social-nose, playwithtoy
```

Coarse labels:

```text
resting, feeding, locomotion, social
```

## Split Strategy

Use grouped splits by `group_id` to reduce leakage from adjacent frames or the
same burst. Record the split seed, source videos, and filtered rows.

## Redistribution

Treat raw frames, videos, and trained model artifacts as research-use only until
collection rights, facility privacy constraints, and redistribution permissions
are confirmed. The repository stores metadata and checksums, not the raw
artifacts.

## Known Limitations

- The dataset may be camera-specific and farm-specific.
- Occlusion and dense group interactions can reduce label quality.
- ROI features depend on scene annotation quality.
