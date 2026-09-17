from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from face_recognition.augment import augment_training_set
from face_recognition.config import AugmentationConfig, load_config
from face_recognition.evaluate import EvaluationError, safe_artifact_path

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "scenarios.yaml"


def test_scenario_matrix_has_sixteen_unique_combinations() -> None:
    scenarios = load_config(CONFIG_PATH).scenarios

    assert len(scenarios) == 16
    assert len({scenario.id for scenario in scenarios}) == 16
    assert (
        len(
            {
                (
                    scenario.margin,
                    scenario.normalization,
                    scenario.brightness,
                    scenario.backbone,
                )
                for scenario in scenarios
            }
        )
        == 16
    )


def test_augmentation_keeps_original_and_adds_five_seeded_copies() -> None:
    images = np.arange(2 * 8 * 8 * 3, dtype=np.uint8).reshape(2, 8, 8, 3)
    labels = np.array([4, 7])
    config = AugmentationConfig(copies_per_image=5, grayscale=False)

    first_images, first_labels = augment_training_set(images, labels, config, seed=42)
    second_images, second_labels = augment_training_set(images, labels, config, seed=42)

    assert first_images.shape == (12, 8, 8, 3)
    assert first_labels.tolist() == [4] * 6 + [7] * 6
    assert np.array_equal(first_images[0], images[0])
    assert np.array_equal(first_images[6], images[1])
    assert np.array_equal(first_images, second_images)
    assert np.array_equal(first_labels, second_labels)


def test_artifact_paths_cannot_escape_output_root(tmp_path: Path) -> None:
    output_root = tmp_path / "artifacts"

    assert safe_artifact_path(output_root, "scenario-01", "metrics.json").is_relative_to(
        output_root
    )
    with pytest.raises(EvaluationError, match="escapes output root"):
        safe_artifact_path(output_root, "..", "outside.json")
