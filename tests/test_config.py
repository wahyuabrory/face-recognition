from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from face_recognition.config import (
    Backbone,
    ConfigurationError,
    Normalization,
    ScenarioConfig,
    load_config,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "scenarios.yaml"


def test_config_loads_without_tensorflow_import() -> None:
    config = load_config(CONFIG_PATH)

    assert len(config.scenarios) == 16
    assert config.defaults.image_size == (224, 224)
    assert config.defaults.seed == 42


@pytest.mark.parametrize(
    ("field", "value"),
    [("normalization", "unknown"), ("backbone", "resnet50")],
)
def test_scenario_rejects_unsupported_enum_values(field: str, value: str) -> None:
    data = {
        "id": 1,
        "margin": 0.1,
        "normalization": Normalization.ZERO_ONE,
        "brightness": 0.1,
        "backbone": Backbone.MOBILENETV2,
    }
    data[field] = value

    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)


def test_invalid_configuration_has_clear_error(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("scenarios: []\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="exactly 16"):
        load_config(config_path)
