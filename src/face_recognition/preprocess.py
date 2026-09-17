from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import cv2
import numpy as np


class PreprocessingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


class FaceDetector(Protocol):
    def detect(self, grayscale_image: np.ndarray) -> Sequence[BoundingBox]: ...


DetectorLike = FaceDetector | Callable[[np.ndarray], Sequence[BoundingBox]]


class HaarFaceDetector:
    def __init__(
        self,
        cascade_path: str | Path | None = None,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: tuple[int, int] = (30, 30),
    ) -> None:
        default_cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        path = str(cascade_path or default_cascade)
        self._classifier = cv2.CascadeClassifier(path)
        if self._classifier.empty():
            raise PreprocessingError(f"could not load Haar cascade: {path}")
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_face_size = min_face_size

    def detect(self, grayscale_image: np.ndarray) -> tuple[BoundingBox, ...]:
        detections = self._classifier.detectMultiScale(
            grayscale_image,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_face_size,
        )
        return tuple(
            BoundingBox(int(x), int(y), int(width), int(height))
            for x, y, width, height in detections
        )


def _as_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in (2, 3) or array.size == 0:
        raise PreprocessingError("image must be a non-empty 2D or 3D array")
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _grayscale(image: np.ndarray, input_color: str) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] != 3:
        raise PreprocessingError("color images must have exactly three channels")
    if input_color == "bgr":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if input_color == "rgb":
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raise PreprocessingError("input_color must be 'bgr' or 'rgb'")


def _coerce_box(value: BoundingBox | Sequence[int]) -> BoundingBox:
    if isinstance(value, BoundingBox):
        return value
    if len(value) != 4:
        raise PreprocessingError("face detector boxes must contain x, y, width, and height")
    return BoundingBox(*(int(part) for part in value))


def _largest_box(detections: Sequence[BoundingBox | Sequence[int]]) -> BoundingBox:
    boxes = [_coerce_box(detection) for detection in detections]
    boxes = [box for box in boxes if box.width > 0 and box.height > 0]
    if not boxes:
        raise PreprocessingError("no face was detected in the image")
    return max(boxes, key=lambda box: box.area)


def preprocess_image(
    image: np.ndarray,
    *,
    detector: DetectorLike | None = None,
    margin: float = 0.1,
    output_size: tuple[int, int] = (224, 224),
    input_color: str = "bgr",
) -> np.ndarray:
    if not 0 <= margin < 1:
        raise PreprocessingError("margin must be between 0 and 1")
    if len(output_size) != 2 or any(dimension <= 0 for dimension in output_size):
        raise PreprocessingError("output_size must contain two positive dimensions")

    source = _as_uint8(image)
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        input_color = "bgr"
    height, width = source.shape[:2]
    detector = detector or HaarFaceDetector()
    grayscale = _grayscale(source, input_color)
    detections = detector(grayscale) if callable(detector) else detector.detect(grayscale)
    box = _largest_box(detections)

    x_margin = int(box.width * margin)
    y_margin = int(box.height * margin)
    left = max(0, box.x - x_margin)
    top = max(0, box.y - y_margin)
    right = min(width, box.x + box.width + x_margin)
    bottom = min(height, box.y + box.height + y_margin)
    if right <= left or bottom <= top:
        raise PreprocessingError("detected face crop is outside image bounds")

    crop = source[top:bottom, left:right]
    if input_color == "bgr":
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    elif input_color != "rgb":
        raise PreprocessingError("input_color must be 'bgr' or 'rgb'")

    target_width, target_height = output_size[1], output_size[0]
    resized = cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized, dtype=np.uint8)


def preprocess_path(
    path: str | Path,
    *,
    detector: DetectorLike | None = None,
    margin: float = 0.1,
    output_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise PreprocessingError(f"could not read image: {image_path}")
    return preprocess_image(
        image,
        detector=detector,
        margin=margin,
        output_size=output_size,
        input_color="bgr",
    )
