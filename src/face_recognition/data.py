from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import SplitConfig


class DatasetError(ValueError):
    pass


IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True, slots=True)
class ImageRecord:
    path: Path
    class_name: str
    class_index: int


@dataclass(frozen=True, slots=True)
class DiscoveredDataset:
    root: Path
    classes: tuple[str, ...]
    records: tuple[ImageRecord, ...]


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: tuple[ImageRecord, ...]
    validation: tuple[ImageRecord, ...]
    test: tuple[ImageRecord, ...]

    @property
    def all_records(self) -> tuple[ImageRecord, ...]:
        return self.train + self.validation + self.test


def _sorted_directories(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: path.name.casefold(),
    )


def discover_dataset(root: str | Path) -> DiscoveredDataset:
    dataset_root = Path(root).expanduser()
    if not dataset_root.is_dir():
        raise DatasetError(f"dataset directory does not exist: {dataset_root}")

    class_directories = _sorted_directories(dataset_root)
    if len(class_directories) < 2:
        raise DatasetError("dataset must contain at least two class directories")

    folded_names = [directory.name.casefold() for directory in class_directories]
    if len(folded_names) != len(set(folded_names)):
        raise DatasetError("class directory names must be unique without case differences")

    records: list[ImageRecord] = []
    for class_index, class_directory in enumerate(class_directories):
        image_paths = sorted(
            (
                path
                for path in class_directory.iterdir()
                if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.casefold(),
        )
        if not image_paths:
            raise DatasetError(
                f"class directory contains no supported images: {class_directory.name}"
            )
        records.extend(
            ImageRecord(path=path, class_name=class_directory.name, class_index=class_index)
            for path in image_paths
        )

    return DiscoveredDataset(
        root=dataset_root,
        classes=tuple(directory.name for directory in class_directories),
        records=tuple(records),
    )


def _allocate_counts(sample_count: int, config: SplitConfig) -> tuple[int, int, int]:
    if sample_count < 3:
        raise DatasetError(
            "each class needs at least three images for a stratified train/validation/test split"
        )

    ratios = (config.train_ratio, config.validation_ratio, config.test_ratio)
    desired = [sample_count * ratio for ratio in ratios]
    counts = [max(1, int(value)) for value in desired]

    while sum(counts) > sample_count:
        removable = [index for index, count in enumerate(counts) if count > 1]
        if not removable:
            raise DatasetError("class does not have enough images for three non-empty partitions")
        index = max(removable, key=lambda item: counts[item] - desired[item])
        counts[index] -= 1

    while sum(counts) < sample_count:
        index = max(range(3), key=lambda item: desired[item] - counts[item])
        counts[index] += 1

    return counts[0], counts[1], counts[2]


def stratified_split(
    dataset: DiscoveredDataset | str | Path,
    config: SplitConfig | None = None,
) -> DatasetSplit:
    if not isinstance(dataset, DiscoveredDataset):
        dataset = discover_dataset(dataset)
    split_config = config or SplitConfig()

    by_class: dict[int, list[ImageRecord]] = {index: [] for index in range(len(dataset.classes))}
    for record in dataset.records:
        by_class[record.class_index].append(record)

    rng = random.Random(split_config.seed)
    train: list[ImageRecord] = []
    validation: list[ImageRecord] = []
    test: list[ImageRecord] = []

    for class_index in range(len(dataset.classes)):
        records = sorted(by_class[class_index], key=lambda record: record.path.name.casefold())
        if len(records) < 3:
            raise DatasetError(
                f"class {dataset.classes[class_index]!r} has {len(records)} images; "
                "at least three are required"
            )
        rng.shuffle(records)
        train_count, validation_count, test_count = _allocate_counts(len(records), split_config)
        train.extend(records[:train_count])
        validation.extend(records[train_count : train_count + validation_count])
        test.extend(
            records[train_count + validation_count : train_count + validation_count + test_count]
        )

    return DatasetSplit(tuple(train), tuple(validation), tuple(test))


def split_dataset(
    dataset: DiscoveredDataset | str | Path,
    config: SplitConfig | None = None,
) -> DatasetSplit:
    return stratified_split(dataset, config)


def labels_for(records: Iterable[ImageRecord]) -> list[int]:
    return [record.class_index for record in records]
