from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .augment import augment_training_set
from .config import (
    AugmentationConfig,
    PreprocessingConfig,
    ScenarioConfig,
    SplitConfig,
    TrainingConfig,
)
from .data import DatasetSplit, DiscoveredDataset, discover_dataset, stratified_split
from .evaluate import artifact_directory, safe_artifact_path
from .preprocess import FaceDetector, HaarFaceDetector, preprocess_path


@dataclass(slots=True)
class TrainingResult:
    scenario: ScenarioConfig
    class_names: tuple[str, ...]
    model: Any
    test_images: np.ndarray
    test_labels: np.ndarray
    history: dict[str, dict[str, list[float]]]
    artifact_dir: Path
    epochs_run: int
    fine_tuned: bool
    run_config: dict[str, Any] = field(default_factory=dict)


def _load_images(
    records: tuple,
    preprocessing: PreprocessingConfig,
    detector: FaceDetector | None,
) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for record in records:
        try:
            image = preprocess_path(
                record.path,
                detector=detector,
                margin=preprocessing.margin,
                output_size=preprocessing.image_size,
            )
        except ValueError as exc:
            raise ValueError(f"could not preprocess {record.path}: {exc}") from exc
        images.append(image)
        labels.append(record.class_index)
    if not images:
        raise ValueError("dataset split contains no images")
    return np.stack(images, axis=0), np.asarray(labels, dtype=np.int64)


def _callbacks(tf: Any, config: TrainingConfig) -> list[Any]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config.early_stopping_patience,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.reduce_lr_patience,
            min_lr=1e-7,
        ),
    ]


def _history_values(history: Any) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in history.history.items()}


def _write_training_metadata(
    directory: Path,
    scenario: ScenarioConfig,
    training: TrainingConfig,
    preprocessing: PreprocessingConfig,
    augmentation: AugmentationConfig,
    split: SplitConfig,
    class_names: tuple[str, ...],
    history: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    run_config = {
        "scenario": scenario.model_dump(mode="json"),
        "training": training.model_dump(mode="json"),
        "preprocessing": preprocessing.model_dump(mode="json"),
        "augmentation": augmentation.model_dump(mode="json"),
        "split": split.model_dump(mode="json"),
        "class_names": list(class_names),
    }
    safe_artifact_path(directory, "training-history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    safe_artifact_path(directory, "run-config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_config


def _arrays_for_split(
    split: DatasetSplit,
    preprocessing: PreprocessingConfig,
    detector: FaceDetector | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images, train_labels = _load_images(split.train, preprocessing, detector)
    validation_images, validation_labels = _load_images(split.validation, preprocessing, detector)
    test_images, test_labels = _load_images(split.test, preprocessing, detector)
    return (
        train_images,
        train_labels,
        validation_images,
        validation_labels,
        test_images,
        test_labels,
    )


def train_scenario(
    dataset_root: str | Path,
    scenario: ScenarioConfig,
    *,
    training: TrainingConfig | None = None,
    preprocessing: PreprocessingConfig | None = None,
    augmentation: AugmentationConfig | None = None,
    split: SplitConfig | None = None,
    output_root: str | Path = "artifacts",
    detector: FaceDetector | None = None,
    save_model: bool = False,
    verbose: int = 1,
) -> TrainingResult:
    training_config = training or TrainingConfig()
    preprocessing_config = (preprocessing or PreprocessingConfig()).model_copy(
        update={"image_size": training_config.image_size, "margin": scenario.margin}
    )
    augmentation_config = augmentation or AugmentationConfig(brightness=scenario.brightness)
    split_config = split or SplitConfig(seed=training_config.seed)
    dataset: DiscoveredDataset = discover_dataset(dataset_root)
    dataset_split = stratified_split(dataset, split_config)
    active_detector = detector or HaarFaceDetector(
        scale_factor=preprocessing_config.scale_factor,
        min_neighbors=preprocessing_config.min_neighbors,
        min_face_size=preprocessing_config.min_face_size,
    )
    (
        source_train,
        source_train_labels,
        validation_images,
        validation_labels,
        test_images,
        test_labels,
    ) = _arrays_for_split(dataset_split, preprocessing_config, active_detector)

    augmentation_config = augmentation_config.model_copy(update={"brightness": scenario.brightness})
    train_images, train_labels = augment_training_set(
        source_train,
        source_train_labels,
        augmentation_config,
        seed=training_config.seed,
    )
    evaluation_images, evaluation_labels = augment_training_set(
        test_images,
        test_labels,
        augmentation_config,
        seed=training_config.seed,
    )

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for training; install the package dependencies first"
        ) from exc

    tf.keras.utils.set_random_seed(training_config.seed)
    from .models import build_classifier, enable_fine_tuning

    model = build_classifier(
        scenario.backbone,
        len(dataset.classes),
        image_size=training_config.image_size,
        dropout=training_config.dropout,
        dense_units=training_config.dense_units,
        learning_rate=training_config.learning_rate,
        normalization=scenario.normalization,
    )
    callbacks = _callbacks(tf, training_config)
    histories: dict[str, dict[str, list[float]]] = {}
    baseline_history = model.fit(
        train_images,
        train_labels,
        validation_data=(validation_images, validation_labels),
        batch_size=training_config.batch_size,
        epochs=training_config.epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
    histories["baseline"] = _history_values(baseline_history)
    epochs_run = len(histories["baseline"].get("loss", []))

    if training_config.fine_tune:
        enable_fine_tuning(
            model,
            scenario.backbone,
            layers=training_config.fine_tune_layers,
            learning_rate=training_config.fine_tune_learning_rate,
        )
        fine_tune_history = model.fit(
            train_images,
            train_labels,
            validation_data=(validation_images, validation_labels),
            batch_size=training_config.batch_size,
            epochs=training_config.epochs,
            callbacks=_callbacks(tf, training_config),
            verbose=verbose,
        )
        histories["fine_tune"] = _history_values(fine_tune_history)
        epochs_run += len(histories["fine_tune"].get("loss", []))

    directory = artifact_directory(output_root, scenario.id)
    run_config = _write_training_metadata(
        directory,
        scenario,
        training_config,
        preprocessing_config,
        augmentation_config,
        split_config,
        dataset.classes,
        histories,
    )
    if save_model:
        model_path = safe_artifact_path(directory, "model.keras")
        model.save(model_path)

    return TrainingResult(
        scenario=scenario,
        class_names=dataset.classes,
        model=model,
        test_images=evaluation_images,
        test_labels=evaluation_labels,
        history=histories,
        artifact_dir=directory,
        epochs_run=epochs_run,
        fine_tuned=training_config.fine_tune,
        run_config=run_config,
    )
