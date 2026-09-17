from __future__ import annotations

from typing import Any

from .config import Backbone, Normalization


class ModelError(RuntimeError):
    pass


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ModelError(
            "TensorFlow is required for model construction; install the package dependencies first"
        ) from exc
    return tf


def _normalization_layer(tf: Any, normalization: str | Normalization) -> Any:
    value = normalization.value if isinstance(normalization, Normalization) else normalization
    if value == Normalization.ZERO_ONE.value:
        return tf.keras.layers.Rescaling(1.0 / 255.0, name="scale_zero_one")
    if value == Normalization.MINUS_ONE_ONE.value:
        return tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="scale_minus_one_one")
    raise ModelError(f"unsupported normalization: {normalization}")


def _build_backbone(
    tf: Any,
    backbone: str | Backbone,
    image_size: tuple[int, int],
    weights: str | None,
) -> Any:
    value = backbone.value if isinstance(backbone, Backbone) else backbone
    kwargs = {
        "include_top": False,
        "weights": weights,
        "input_shape": (*image_size, 3),
    }
    if value == Backbone.MOBILENETV2.value:
        return tf.keras.applications.MobileNetV2(name="mobilenetv2_backbone", **kwargs)
    if value == Backbone.EFFICIENTNETB0.value:
        kwargs["include_preprocessing"] = False
        try:
            return tf.keras.applications.EfficientNetB0(name="efficientnetb0_backbone", **kwargs)
        except TypeError:
            kwargs.pop("include_preprocessing")
            return tf.keras.applications.EfficientNetB0(name="efficientnetb0_backbone", **kwargs)
    raise ModelError(f"unsupported backbone: {backbone}")


def build_classifier(
    backbone: str | Backbone,
    num_classes: int,
    image_size: tuple[int, int] = (224, 224),
    dropout: float = 0.3,
    dense_units: int = 128,
    learning_rate: float = 0.0001,
    *,
    normalization: str | Normalization = Normalization.ZERO_ONE,
    weights: str | None = "imagenet",
) -> Any:
    if num_classes < 2:
        raise ModelError("num_classes must be at least two")
    if len(image_size) != 2 or any(dimension <= 0 for dimension in image_size):
        raise ModelError("image_size must contain two positive dimensions")
    if not 0 <= dropout < 1:
        raise ModelError("dropout must be between 0 and 1")
    if dense_units <= 0 or learning_rate <= 0:
        raise ModelError("dense_units and learning_rate must be positive")

    tf = _tensorflow()
    inputs = tf.keras.Input(shape=(*image_size, 3), name="image")
    scaled = _normalization_layer(tf, normalization)(inputs)
    feature_extractor = _build_backbone(tf, backbone, image_size, weights)
    feature_extractor.trainable = False
    features = feature_extractor(scaled, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    dropped_before_dense = tf.keras.layers.Dropout(dropout, name="classifier_dropout_before_dense")(
        pooled
    )
    dense = tf.keras.layers.Dense(dense_units, activation="relu", name="classifier_dense")(
        dropped_before_dense
    )
    dropped_after_dense = tf.keras.layers.Dropout(dropout, name="classifier_dropout_after_dense")(
        dense
    )
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_output")(
        dropped_after_dense
    )
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="face_classifier")
    _compile(model, tf, learning_rate)
    return model


def _compile(model: Any, tf: Any, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def enable_fine_tuning(
    model: Any,
    backbone: str | Backbone,
    *,
    layers: int = 30,
    learning_rate: float = 0.00001,
) -> Any:
    if layers <= 0 or learning_rate <= 0:
        raise ModelError("fine-tune layers and learning_rate must be positive")
    tf = _tensorflow()
    backbone_value = backbone.value if isinstance(backbone, Backbone) else backbone
    backbone_name = f"{backbone_value}_backbone"
    try:
        feature_extractor = model.get_layer(backbone_name)
    except ValueError as exc:
        raise ModelError(f"model does not contain backbone layer {backbone_name}") from exc

    feature_extractor.trainable = True
    split_at = max(0, len(feature_extractor.layers) - layers)
    for index, layer in enumerate(feature_extractor.layers):
        layer.trainable = index >= split_at
    _compile(model, tf, learning_rate)
    return model
