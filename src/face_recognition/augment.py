"""Deterministic, in-memory image augmentation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import AugmentationConfig


def _validate_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("augmentation expects an RGB uint8 image with three channels")
    return np.ascontiguousarray(array)


def _grayscale_rgb(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _brightness(image: np.ndarray, limit: float, rng: np.random.Generator) -> np.ndarray:
    shift = float(rng.uniform(-limit, limit) * 255.0)
    return np.clip(image.astype(np.float32) + shift, 0, 255).astype(np.uint8)


def augment_image(
    image: np.ndarray,
    config: AugmentationConfig | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create one augmented copy without writing it to disk."""

    augmentation = config or AugmentationConfig()
    generator = rng if rng is not None else np.random.default_rng()
    result = _validate_image(image).copy()

    if augmentation.rotation_degrees:
        result = _rotate(
            result,
            float(
                generator.uniform(
                    -augmentation.rotation_degrees,
                    augmentation.rotation_degrees,
                )
            ),
        )
    if augmentation.horizontal_flip:
        result = cv2.flip(result, 1)
    if augmentation.brightness:
        result = _brightness(result, augmentation.brightness, generator)
    if augmentation.grayscale:
        result = _grayscale_rgb(result)
    return np.ascontiguousarray(result, dtype=np.uint8)


def augment_training_images(
    images: Iterable[np.ndarray],
    config: AugmentationConfig | None = None,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Return each input plus exactly ``copies_per_image`` augmented copies."""

    augmentation = config or AugmentationConfig()
    generator = np.random.default_rng(seed)
    augmented: list[np.ndarray] = []
    for image in images:
        original = _validate_image(image)
        augmented.append(original.copy())
        augmented.extend(
            augment_image(original, augmentation, rng=generator)
            for _ in range(augmentation.copies_per_image)
        )
    if not augmented:
        raise ValueError("at least one image is required for augmentation")
    return np.stack(augmented, axis=0)


def augment_training_set(
    images: Iterable[np.ndarray],
    labels: Iterable[int],
    config: AugmentationConfig | None = None,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment images and repeat each label for the original plus its copies."""

    augmentation = config or AugmentationConfig()
    image_list = list(images)
    label_list = list(labels)
    if len(image_list) != len(label_list):
        raise ValueError("images and labels must have the same length")
    augmented = augment_training_images(image_list, augmentation, seed=seed)
    copies_per_source = 1 + augmentation.copies_per_image
    repeated_labels = np.repeat(label_list, copies_per_source).astype(np.int64)
    return augmented, repeated_labels


def write_augmented_images(
    images: Iterable[np.ndarray], output_directory: str | Path
) -> list[Path]:
    """Write images only when the caller explicitly requests an output directory."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, image in enumerate(images, start=1):
        path = directory / f"augmented-{index:06d}.png"
        bgr = cv2.cvtColor(_validate_image(image), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr):
            raise OSError(f"could not write augmented image: {path}")
        paths.append(path)
    return paths
