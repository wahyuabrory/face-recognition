from __future__ import annotations

from pathlib import Path

import pytest

from face_recognition.data import DatasetError, discover_dataset, stratified_split


def _make_dataset(root: Path, class_count: int, images_per_class: int) -> Path:
    for class_index in range(class_count):
        class_directory = root / f"person_{class_index + 1:03d}"
        class_directory.mkdir(parents=True)
        for image_index in range(images_per_class):
            (class_directory / f"image_{image_index + 1:03d}.jpg").touch()
    return root


def test_split_is_stratified_and_approximately_70_20_10(tmp_path: Path) -> None:
    dataset = discover_dataset(_make_dataset(tmp_path / "dataset", 3, 20))

    split = stratified_split(dataset)

    assert (len(split.train), len(split.validation), len(split.test)) == (42, 12, 6)
    for class_index in range(3):
        assert sum(record.class_index == class_index for record in split.train) == 14
        assert sum(record.class_index == class_index for record in split.validation) == 4
        assert sum(record.class_index == class_index for record in split.test) == 2


def test_split_is_deterministic_with_seed_42(tmp_path: Path) -> None:
    dataset = discover_dataset(_make_dataset(tmp_path / "dataset", 2, 10))

    first = stratified_split(dataset)
    second = stratified_split(dataset)

    assert first == second


def test_split_rejects_insufficient_class_samples(tmp_path: Path) -> None:
    dataset = discover_dataset(_make_dataset(tmp_path / "dataset", 2, 2))

    with pytest.raises(DatasetError, match="at least three"):
        stratified_split(dataset)
