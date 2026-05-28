# Dataset Card

## Dataset Summary

The dataset contains extracted pig video frames, pig bounding boxes, fine and
coarse behavior labels, and tabular context features used for behavior
classification.

## Local Paths

- Processed CSV: `data/processed/behavior_with_feats_rectROI.csv`
- Raw images: `data/raw/images_clean/`
- Scene annotations: `data/annotations/`

## Labels

Fine labels:

```text
lying, eat, drink, explore, sitting, stand, social-nose, playwithtoy
```

Coarse labels:

```text
resting, feeding, locomotion, social
```

## Split Strategy

Training, validation, and test sets are split by `group_id` so frames from the
same burst do not cross splits.

## Known Limitations

- Raw image data is not committed to Git.
- License and collection protocol must be documented before public release.
- Any animal welfare, farm identity, or private facility constraints should be
  reviewed before redistributing images.
