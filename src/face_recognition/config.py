from __future__ import annotations

from enum import StrEnum
from itertools import product
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ConfigurationError(ValueError):
    pass


class Normalization(StrEnum):
    ZERO_ONE = "0_1"
    MINUS_ONE_ONE = "minus1_1"


class Backbone(StrEnum):
    MOBILENETV2 = "mobilenetv2"
    EFFICIENTNETB0 = "efficientnetb0"


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=32, gt=0)
    epochs: int = Field(default=20, gt=0)
    learning_rate: float = Field(default=0.0001, gt=0)
    dropout: float = Field(default=0.3, ge=0, lt=1)
    dense_units: int = Field(default=128, gt=0)
    seed: int = 42
    early_stopping_patience: int = Field(default=5, ge=0)
    reduce_lr_patience: int = Field(default=3, ge=0)
    fine_tune: bool = False
    fine_tune_layers: int = Field(default=30, gt=0)
    fine_tune_learning_rate: float = Field(default=0.00001, gt=0)


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_ratio: float = Field(default=0.7, gt=0, lt=1)
    validation_ratio: float = Field(default=0.2, gt=0, lt=1)
    test_ratio: float = Field(default=0.1, gt=0, lt=1)
    seed: int = 42

    @model_validator(mode="after")
    def validate_ratios(self) -> SplitConfig:
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError("train_ratio, validation_ratio, and test_ratio must sum to 1")
        return self


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_size: tuple[int, int] = (224, 224)
    margin: float = Field(default=0.1, ge=0, lt=1)
    scale_factor: float = Field(default=1.1, gt=1)
    min_neighbors: int = Field(default=5, ge=0)
    min_face_size: tuple[int, int] = (30, 30)

    @field_validator("image_size", "min_face_size")
    @classmethod
    def validate_dimensions(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2 or any(dimension <= 0 for dimension in value):
            raise ValueError("image dimensions must contain two positive values")
        return value


class AugmentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    copies_per_image: int = Field(default=5, gt=0)
    rotation_degrees: float = Field(default=20.0, ge=0, le=180)
    horizontal_flip: bool = True
    brightness: float = Field(default=0.1, ge=0, le=1)
    grayscale: bool = False


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(gt=0)
    margin: float
    normalization: Normalization
    brightness: float
    backbone: Backbone

    @field_validator("margin")
    @classmethod
    def validate_margin(cls, value: float) -> float:
        if value not in (0.1, 0.3):
            raise ValueError("margin must be 0.1 or 0.3")
        return value

    @field_validator("brightness")
    @classmethod
    def validate_brightness(cls, value: float) -> float:
        if value not in (0.1, 0.4):
            raise ValueError("brightness must be 0.1 or 0.4")
        return value


class ScenarioFileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defaults: TrainingConfig = Field(default_factory=TrainingConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    save_model: bool = False
    scenarios: list[ScenarioConfig]

    @model_validator(mode="after")
    def validate_scenarios(self) -> ScenarioFileConfig:
        if len(self.scenarios) != 16:
            raise ValueError("scenarios must contain exactly 16 entries")

        ids = [scenario.id for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario IDs must be unique")

        expected = {
            (margin, normalization, brightness, backbone)
            for margin, normalization, brightness, backbone in product(
                (0.1, 0.3),
                (Normalization.ZERO_ONE, Normalization.MINUS_ONE_ONE),
                (0.1, 0.4),
                (Backbone.MOBILENETV2, Backbone.EFFICIENTNETB0),
            )
        }
        actual = {
            (scenario.margin, scenario.normalization, scenario.brightness, scenario.backbone)
            for scenario in self.scenarios
        }
        if actual != expected:
            raise ValueError(
                "scenarios must be the complete Cartesian product of margin, normalization, "
                "brightness, and backbone values"
            )
        return self

    def scenario(self, scenario_id: int) -> ScenarioConfig:
        for scenario in self.scenarios:
            if scenario.id == scenario_id:
                return scenario
        raise ConfigurationError(f"unknown scenario ID: {scenario_id}")


def load_config(path: str | Path) -> ScenarioFileConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw: Any = yaml.safe_load(stream)
    except OSError as exc:
        raise ConfigurationError(f"could not read configuration file: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in configuration file: {config_path}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError("configuration root must be a YAML mapping")

    try:
        return ScenarioFileConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(str(exc)) from exc
