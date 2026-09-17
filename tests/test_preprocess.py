from __future__ import annotations

import numpy as np
import pytest

from face_recognition.preprocess import BoundingBox, PreprocessingError, preprocess_image


def test_preprocess_returns_expected_shape_dtype_and_rgb_order() -> None:
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    result = preprocess_image(
        image,
        detector=lambda _: [BoundingBox(5, 5, 20, 20)],
        output_size=(224, 224),
        input_color="bgr",
    )

    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8
    assert tuple(result[0, 0]) == (0, 0, 255)


def test_preprocess_uses_largest_face_and_clamps_crop_bounds() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:15, :15] = (10, 20, 30)

    result = preprocess_image(
        image,
        detector=lambda _: [BoundingBox(10, 10, 3, 3), BoundingBox(-2, -3, 17, 18)],
        margin=0.3,
        output_size=(16, 12),
        input_color="rgb",
    )

    assert result.shape == (16, 12, 3)
    assert result.dtype == np.uint8


def test_preprocess_fails_when_no_face_is_found() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    with pytest.raises(PreprocessingError, match="no face"):
        preprocess_image(image, detector=lambda _: [], input_color="rgb")
