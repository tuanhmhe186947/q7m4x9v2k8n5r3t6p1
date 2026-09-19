"""Data loading and preprocessing for pig behavior classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

from pig_behavior.config import TABULAR_FEATURES, TrainConfig

REQUIRED_COLUMNS = {
    "img_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "behavior",
    "behavior_coarse",
    "hidden",
    "group_id",
    *TABULAR_FEATURES,
}


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Fail early when the CSV does not match the expected schema."""
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")


def _encode_labels(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    """Map configured string labels to integer class indices."""
    label_to_idx = {label: idx for idx, label in enumerate(cfg.labels)}
    encoded = df.copy()
    encoded["label_idx"] = encoded[cfg.label_column].map(label_to_idx)
    encoded = encoded.dropna(subset=["label_idx"])
    encoded["label_idx"] = encoded["label_idx"].astype(np.int32)
    return encoded


def _split_by_group(
    df: pd.DataFrame,
    cfg: TrainConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split train/validation/test sets without leaking a group across splits."""
    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg.test_split,
        random_state=cfg.random_seed,
    )
    trainval_idx, test_idx = next(gss_test.split(df, groups=df["group_id"]))
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg.val_split,
        random_state=cfg.random_seed,
    )
    train_idx, val_idx = next(
        gss_val.split(trainval_df, groups=trainval_df["group_id"])
    )
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def _load_and_crop(
    img_path: tf.Tensor,
    bbox: tf.Tensor,
    img_size: tuple[int, int],
) -> tf.Tensor:
    """Read an image, crop it to a bounding box, and resize it."""
    raw = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.cast(image, tf.float32)

    height = tf.shape(image)[0]
    width = tf.shape(image)[1]

    x1 = tf.cast(tf.round(bbox[0]), tf.int32)
    y1 = tf.cast(tf.round(bbox[1]), tf.int32)
    x2 = tf.cast(tf.round(bbox[2]), tf.int32)
    y2 = tf.cast(tf.round(bbox[3]), tf.int32)

    x1 = tf.clip_by_value(x1, 0, width - 1)
    y1 = tf.clip_by_value(y1, 0, height - 1)
    x2 = tf.clip_by_value(tf.maximum(x2, x1 + 1), x1 + 1, width)
    y2 = tf.clip_by_value(tf.maximum(y2, y1 + 1), y1 + 1, height)

    crop = image[y1:y2, x1:x2, :]
    crop = tf.image.resize(crop, img_size)
    return crop / 255.0


def _augment(image: tf.Tensor, cfg: TrainConfig) -> tf.Tensor:
    """Apply lightweight augmentations during training."""
    if cfg.horizontal_flip:
        image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(
        image,
        lower=cfg.contrast_range[0],
        upper=cfg.contrast_range[1],
    )
    return tf.clip_by_value(image, 0.0, 1.0)


def _make_dataset(
    df: pd.DataFrame,
    cfg: TrainConfig,
    *,
    training: bool = False,
    images_dir: Path | None = None,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from a dataframe split."""
    images_dir = images_dir or cfg.images_dir

    img_paths = [str(images_dir / name) for name in df["img_name"].to_numpy()]
    bboxes = df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
    labels = df["label_idx"].to_numpy(dtype=np.int32)

    ds_paths = tf.data.Dataset.from_tensor_slices(img_paths)
    ds_bboxes = tf.data.Dataset.from_tensor_slices(bboxes)
    ds_labels = tf.data.Dataset.from_tensor_slices(labels)

    def process_image(path: tf.Tensor, bbox: tf.Tensor, label: tf.Tensor) -> Any:
        image = _load_and_crop(path, bbox, cfg.image_size)
        if training:
            image = _augment(image, cfg)
        return image, label

    if cfg.use_hybrid:
        tabular = df[TABULAR_FEATURES].to_numpy(dtype=np.float32)
        ds_tabular = tf.data.Dataset.from_tensor_slices(tabular)

        def process_hybrid(
            path: tf.Tensor,
            bbox: tf.Tensor,
            features: tf.Tensor,
            label: tf.Tensor,
        ) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
            image = _load_and_crop(path, bbox, cfg.image_size)
            if training:
                image = _augment(image, cfg)
            return {"image_input": image, "tabular_input": features}, label

        dataset = tf.data.Dataset.zip((ds_paths, ds_bboxes, ds_tabular, ds_labels))
        dataset = dataset.map(process_hybrid, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = tf.data.Dataset.zip((ds_paths, ds_bboxes, ds_labels))
        dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(df), 10_000),
            seed=cfg.random_seed,
            reshuffle_each_iteration=True,
        )

    return dataset.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)


def load_dataframes(
    cfg: TrainConfig,
    csv_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the CSV, clean labels, and return train/val/test dataframes."""
    csv_path = csv_path or cfg.csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    _validate_dataframe(df)

    df = df[df["hidden"].astype(str).str.lower() != "yes"].reset_index(drop=True)
    df = _encode_labels(df, cfg)

    if cfg.dry_run:
        df = df.head(500)

    if df.empty:
        raise ValueError("No rows remain after filtering hidden pigs and labels.")

    return _split_by_group(df, cfg)


def build_datasets(
    cfg: TrainConfig,
    csv_path: Path | None = None,
    images_dir: Path | None = None,
) -> dict[str, tf.data.Dataset]:
    """Build train, validation, and test TensorFlow datasets."""
    train_df, val_df, test_df = load_dataframes(cfg, csv_path)
    images_dir = images_dir or cfg.images_dir
    _validate_image_paths(train_df, images_dir)
    _validate_image_paths(val_df, images_dir)
    _validate_image_paths(test_df, images_dir)

    print(f"[data] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("[data] Training labels:")
    print(train_df[cfg.label_column].value_counts().to_string())
    print()

    return {
        "train": _make_dataset(train_df, cfg, training=True, images_dir=images_dir),
        "val": _make_dataset(val_df, cfg, images_dir=images_dir),
        "test": _make_dataset(test_df, cfg, images_dir=images_dir),
    }


def _validate_image_paths(
    df: pd.DataFrame,
    images_dir: Path,
    *,
    max_examples: int = 10,
) -> None:
    """Fail before TensorFlow starts if referenced images are missing."""
    missing = [
        str(images_dir / name)
        for name in df["img_name"].drop_duplicates()
        if not (images_dir / name).exists()
    ]
    if missing:
        examples = "\n".join(missing[:max_examples])
        raise FileNotFoundError(
            "CSV references images that were not found under "
            f"{images_dir}. Examples:\n{examples}"
        )
