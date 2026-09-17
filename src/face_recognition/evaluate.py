from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import ScenarioConfig


class EvaluationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    scenario: ScenarioConfig | None
    seed: int
    test_loss: float
    test_accuracy: float
    classification_report: dict[str, Any]
    confusion_matrix: list[list[int]]
    class_names: tuple[str, ...] = ()

    @property
    def precision(self) -> float:
        return float(self.classification_report["macro avg"]["precision"])

    @property
    def recall(self) -> float:
        return float(self.classification_report["macro avg"]["recall"])

    @property
    def f1_score(self) -> float:
        return float(self.classification_report["macro avg"]["f1-score"])

    @property
    def support(self) -> int:
        return int(self.classification_report["macro avg"]["support"])


def _metric_value(metrics: Any, name: str, index: int) -> float:
    if isinstance(metrics, dict):
        value = metrics.get(name)
        if value is None and name == "accuracy":
            value = metrics.get("sparse_categorical_accuracy")
        if value is None:
            raise EvaluationError(f"model evaluation did not return {name}")
        return float(value)
    values = list(metrics) if isinstance(metrics, (list, tuple, np.ndarray)) else [metrics]
    if len(values) <= index:
        raise EvaluationError(f"model evaluation did not return {name}")
    return float(values[index])


def _predicted_labels(predictions: Any) -> np.ndarray:
    values = np.asarray(predictions)
    if values.ndim == 2:
        return np.argmax(values, axis=1).astype(np.int64)
    if values.ndim == 1:
        return np.rint(values).astype(np.int64)
    raise EvaluationError("model predictions must be a one- or two-dimensional array")


def _classification_report(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: Sequence[str],
) -> tuple[dict[str, Any], np.ndarray]:
    class_count = len(class_names)
    if class_count < 2:
        raise EvaluationError("at least two class names are required")
    if true_labels.shape != predicted_labels.shape:
        raise EvaluationError("model predictions and labels must have the same length")
    if np.any(true_labels < 0) or np.any(true_labels >= class_count):
        raise EvaluationError("true labels contain a class outside class_names")
    if np.any(predicted_labels < 0) or np.any(predicted_labels >= class_count):
        raise EvaluationError("predicted labels contain a class outside class_names")

    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for actual, predicted in zip(true_labels, predicted_labels, strict=True):
        confusion[int(actual), int(predicted)] += 1

    report: dict[str, Any] = {}
    values: list[tuple[float, float, float, int]] = []
    for index, class_name in enumerate(class_names):
        true_positive = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted_count = int(confusion[:, index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values.append((precision, recall, f1, support))
        report[str(class_name)] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }

    total_support = int(confusion.sum())
    macro = [sum(value[index] for value in values) / class_count for index in range(3)]
    weighted = [
        sum(value[index] * value[3] for value in values) / total_support if total_support else 0.0
        for index in range(3)
    ]
    report["macro avg"] = {
        "precision": macro[0],
        "recall": macro[1],
        "f1-score": macro[2],
        "support": total_support,
    }
    report["weighted avg"] = {
        "precision": weighted[0],
        "recall": weighted[1],
        "f1-score": weighted[2],
        "support": total_support,
    }
    return report, confusion


def evaluate_model(
    model: Any,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    class_names: Sequence[str],
    *,
    scenario: ScenarioConfig | None = None,
    seed: int = 42,
    output_root: str | Path | None = None,
    history: dict[str, Any] | None = None,
    run_config: dict[str, Any] | None = None,
    verbose: int = 0,
) -> EvaluationResult:
    labels = np.asarray(test_labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0:
        raise EvaluationError("test_labels must be a non-empty one-dimensional array")
    metrics = model.evaluate(test_images, labels, verbose=verbose, return_dict=True)
    predictions = model.predict(test_images, verbose=verbose)
    predicted_labels = _predicted_labels(predictions)
    report, confusion = _classification_report(labels, predicted_labels, class_names)
    result = EvaluationResult(
        scenario=scenario,
        seed=seed,
        test_loss=_metric_value(metrics, "loss", 0),
        test_accuracy=_metric_value(metrics, "accuracy", 1),
        classification_report=report,
        confusion_matrix=confusion.tolist(),
        class_names=tuple(str(name) for name in class_names),
    )
    if output_root is not None:
        write_artifacts(result, output_root, history=history, run_config=run_config)
    return result


def artifact_directory(output_root: str | Path, scenario_id: int) -> Path:
    root = Path(output_root).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise EvaluationError(f"output root is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"scenario-{scenario_id:02d}"
    try:
        directory.resolve().relative_to(root)
    except ValueError as exc:
        raise EvaluationError("scenario artifact directory escapes output root") from exc
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_artifact_path(output_root: str | Path, *parts: str) -> Path:
    root = Path(output_root).expanduser().resolve()
    candidate = root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise EvaluationError("artifact path escapes output root") from exc
    return candidate


def _scenario_data(result: EvaluationResult) -> dict[str, Any] | None:
    if result.scenario is None:
        return None
    return result.scenario.model_dump(mode="json")


def write_artifacts(
    result: EvaluationResult,
    output_root: str | Path,
    *,
    history: dict[str, Any] | None = None,
    run_config: dict[str, Any] | None = None,
) -> Path:
    if result.scenario is None:
        raise EvaluationError("a scenario is required when writing artifacts")
    directory = artifact_directory(output_root, result.scenario.id)
    metrics = {
        "scenario": _scenario_data(result),
        "seed": result.seed,
        "test_loss": result.test_loss,
        "test_accuracy": result.test_accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "support": result.support,
    }
    _write_json(safe_artifact_path(directory, "metrics.json"), metrics)
    _write_json(
        safe_artifact_path(directory, "classification-report.json"),
        result.classification_report,
    )
    _write_confusion_matrix(
        safe_artifact_path(directory, "confusion-matrix.csv"),
        result.confusion_matrix,
        list(result.class_names) or list(result.classification_report)[:-2],
    )
    _write_json(safe_artifact_path(directory, "training-history.json"), history or {})
    _write_json(
        safe_artifact_path(directory, "run-config.json"),
        run_config or {"scenario": _scenario_data(result), "seed": result.seed},
    )
    return directory


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_confusion_matrix(path: Path, matrix: list[list[int]], labels: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])
