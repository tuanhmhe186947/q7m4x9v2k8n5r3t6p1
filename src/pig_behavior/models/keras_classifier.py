"""Keras model definitions for pig behavior classification."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers

from pig_behavior.config import TABULAR_FEATURES, TrainConfig


def _build_backbone(cfg: TrainConfig) -> tf.keras.Model:
    """Create the frozen MobileNetV3Small image backbone."""
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(*cfg.image_size, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=False,
    )
    base.trainable = False
    return base


def build_image_model(cfg: TrainConfig) -> Model:
    """Build the image-only classifier."""
    base = _build_backbone(cfg)

    image_input = layers.Input(shape=(*cfg.image_size, 3), name="image_input")
    x = base(image_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(cfg.dropout_rate)(x)
    x = layers.Dense(cfg.dense_units, activation="relu")(x)
    x = layers.Dropout(cfg.dropout_rate / 2)(x)
    output = layers.Dense(cfg.num_classes, activation="softmax", name="output")(x)

    return Model(image_input, output, name="pig_image_model")


def build_hybrid_model(cfg: TrainConfig) -> Model:
    """Build the image plus tabular feature classifier."""
    base = _build_backbone(cfg)

    image_input = layers.Input(shape=(*cfg.image_size, 3), name="image_input")
    x_image = base(image_input, training=False)
    x_image = layers.GlobalAveragePooling2D()(x_image)
    x_image = layers.Dense(64, activation="relu", name="image_dense")(x_image)

    tabular_input = layers.Input(
        shape=(len(TABULAR_FEATURES),),
        name="tabular_input",
    )
    x_tabular = layers.Dense(
        cfg.tabular_dense_units,
        activation="relu",
        name="tabular_dense_1",
    )(tabular_input)
    x_tabular = layers.Dense(16, activation="relu", name="tabular_dense_2")(x_tabular)

    x = layers.Concatenate(name="feature_concat")([x_image, x_tabular])
    x = layers.Dropout(cfg.dropout_rate)(x)
    x = layers.Dense(64, activation="relu", name="merged_dense")(x)
    x = layers.BatchNormalization()(x)
    output = layers.Dense(cfg.num_classes, activation="softmax", name="output")(x)

    return Model(
        inputs=[image_input, tabular_input],
        outputs=output,
        name="pig_hybrid_model",
    )


def build_model(cfg: TrainConfig) -> Model:
    """Build the configured model variant."""
    return build_hybrid_model(cfg) if cfg.use_hybrid else build_image_model(cfg)


def prepare_for_fine_tuning(model: Model, cfg: TrainConfig) -> None:
    """Unfreeze the last configured backbone layers for phase 2 training."""
    backbone = next(
        (layer for layer in model.layers if isinstance(layer, tf.keras.Model)),
        None,
    )
    if backbone is None:
        print("[model] WARNING: could not locate backbone for fine-tuning.")
        return

    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) + cfg.fine_tune_at_layer)
    for layer in backbone.layers[:cutoff]:
        layer.trainable = False

    trainable = sum(1 for layer in backbone.layers if layer.trainable)
    print(
        "[model] Fine-tuning: "
        f"{trainable}/{len(backbone.layers)} backbone layers trainable."
    )


def compile_model(
    model: Model,
    cfg: TrainConfig,
    *,
    fine_tuning: bool = False,
) -> None:
    """Compile a model with the configured optimizer and loss."""
    learning_rate = cfg.fine_tune_lr if fine_tuning else cfg.learning_rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
