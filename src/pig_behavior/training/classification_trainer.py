"""Training workflow for pig behavior classification."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from pig_behavior.config import CHECKPOINT_DIR, LOG_DIR, TrainConfig, ensure_output_dirs
from pig_behavior.data.tf_dataset import build_datasets
from pig_behavior.models.keras_classifier import (
    build_model,
    compile_model,
    prepare_for_fine_tuning,
)


def _get_callbacks(phase: str, cfg: TrainConfig) -> list[tf.keras.callbacks.Callback]:
    """Build Keras callbacks for one training phase."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, cfg.patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_DIR / f"best_{phase}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(LOG_DIR / phase),
            histogram_freq=0,
        ),
    ]


def train(
    cfg: TrainConfig,
    datasets: dict[str, tf.data.Dataset] | None = None,
) -> dict[str, Any]:
    """Run the two-phase training pipeline."""
    ensure_output_dirs()

    if datasets is None:
        datasets = build_datasets(cfg)

    train_ds = datasets["train"]
    val_ds = datasets["val"]
    test_ds = datasets["test"]

    model = build_model(cfg)
    compile_model(model, cfg)
    model.summary()

    results: dict[str, Any] = {}

    phase1_epochs = 1 if cfg.dry_run else cfg.phase1_epochs
    print()
    print("=" * 72)
    print(f"Phase 1: training classifier head ({phase1_epochs} epochs)")
    print("=" * 72)
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=_get_callbacks("phase1", cfg),
    )
    results["phase1_history"] = _serialise_history(history_1.history)

    phase2_epochs = 1 if cfg.dry_run else cfg.phase2_epochs
    print()
    print("=" * 72)
    print(f"Phase 2: fine-tuning backbone ({phase2_epochs} epochs)")
    print("=" * 72)
    prepare_for_fine_tuning(model, cfg)
    compile_model(model, cfg, fine_tuning=True)

    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase2_epochs,
        callbacks=_get_callbacks("phase2", cfg),
    )
    results["phase2_history"] = _serialise_history(history_2.history)

    print()
    print("=" * 72)
    print("Evaluation")
    print("=" * 72)
    test_loss, test_accuracy = model.evaluate(test_ds)
    results["test_loss"] = float(test_loss)
    results["test_accuracy"] = float(test_accuracy)
    print(f"Test accuracy: {test_accuracy:.4f} | Test loss: {test_loss:.4f}")

    y_true, y_pred = _predict_all(model, test_ds)
    report = classification_report(
        y_true,
        y_pred,
        target_names=cfg.labels,
        output_dict=True,
        zero_division=0,
    )
    results["classification_report"] = report
    print()
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=cfg.labels,
            zero_division=0,
        )
    )

    results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    saved_model_path = CHECKPOINT_DIR / "final_model.keras"
    model.save(str(saved_model_path))
    print(f"[train] Saved Keras model to {saved_model_path}")

    results_path = LOG_DIR / "results.json"
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"[train] Saved results to {results_path}")

    return results


def _serialise_history(history: dict[str, list[float]]) -> dict[str, list[float]]:
    """Convert Keras history values to JSON-serialisable floats."""
    return {
        key: [float(value) for value in values]
        for key, values in history.items()
    }


def _predict_all(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect ground-truth and predicted labels from a dataset."""
    y_true_batches: list[np.ndarray] = []
    y_pred_batches: list[np.ndarray] = []

    for inputs, labels in dataset:
        predictions = model.predict(inputs, verbose=0)
        y_pred_batches.append(np.argmax(predictions, axis=-1))
        y_true_batches.append(labels.numpy())

    return np.concatenate(y_true_batches), np.concatenate(y_pred_batches)
