"""Command-line interface with configuration validation before TensorFlow import."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import ConfigurationError, ScenarioFileConfig, TrainingConfig, load_config
from .evaluate import EvaluationError, evaluate_model

DEFAULT_CONFIG = Path("configs/scenarios.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face-recognition",
        description="Train and evaluate fixed-class face-recognition classifiers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-scenarios", help="list the validated scenarios")
    list_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    train_parser = subparsers.add_parser("train", help="train and evaluate one scenario")
    _add_run_arguments(train_parser, scenario_required=True)

    all_parser = subparsers.add_parser("run-all", help="train and evaluate all scenarios")
    _add_run_arguments(all_parser, scenario_required=False)
    return parser


def _add_run_arguments(parser: argparse.ArgumentParser, *, scenario_required: bool) -> None:
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    if scenario_required:
        parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--quiet", action="store_true")


def _validate_run_inputs(args: argparse.Namespace) -> ScenarioFileConfig:
    if not args.dataset.is_dir():
        raise ConfigurationError(f"dataset directory does not exist: {args.dataset}")
    if args.output.exists() and not args.output.is_dir():
        raise ConfigurationError(f"output path is not a directory: {args.output}")
    return load_config(args.config)


def _training_config(config: ScenarioFileConfig, fine_tune: bool) -> TrainingConfig:
    if not fine_tune:
        return config.defaults
    return config.defaults.model_copy(update={"fine_tune": True})


def _run_one(
    args: argparse.Namespace,
    config: ScenarioFileConfig,
    scenario_id: int,
) -> dict[str, object]:
    from .train import train_scenario

    scenario = config.scenario(scenario_id)
    result = train_scenario(
        args.dataset,
        scenario,
        training=_training_config(config, args.fine_tune),
        preprocessing=config.preprocessing,
        augmentation=config.augmentation,
        split=config.split,
        output_root=args.output,
        save_model=args.save_model or config.output.save_model,
        verbose=0 if args.quiet else 1,
    )
    evaluation = evaluate_model(
        result.model,
        result.test_images,
        result.test_labels,
        result.class_names,
        scenario=scenario,
        seed=config.defaults.seed,
        output_root=args.output,
        history=result.history,
        run_config=result.run_config,
        verbose=0,
    )
    return {
        "scenario": scenario.id,
        "test_accuracy": evaluation.test_accuracy,
        "test_loss": evaluation.test_loss,
        "precision": evaluation.precision,
        "recall": evaluation.recall,
        "f1_score": evaluation.f1_score,
        "support": evaluation.support,
    }


def _list_scenarios(config_path: Path) -> int:
    config = load_config(config_path)
    for scenario in sorted(config.scenarios, key=lambda item: item.id):
        print(
            json.dumps(
                {
                    "id": scenario.id,
                    "margin": scenario.margin,
                    "normalization": scenario.normalization.value,
                    "brightness": scenario.brightness,
                    "backbone": scenario.backbone.value,
                },
                sort_keys=True,
            )
        )
    return 0


def _run_command(args: argparse.Namespace) -> int:
    config = _validate_run_inputs(args)
    if args.command == "train":
        summary = _run_one(args, config, args.scenario)
        print(json.dumps(summary, sort_keys=True))
        return 0

    summaries = [_run_one(args, config, scenario.id) for scenario in config.scenarios]
    summaries.sort(key=lambda item: int(item["scenario"]))
    ranked = sorted(
        summaries,
        key=lambda item: (
            -float(item["test_accuracy"]),
            float(item["test_loss"]),
            int(item["scenario"]),
        ),
    )
    print(json.dumps({"runs": summaries, "ranking": ranked}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process status."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list-scenarios":
            return _list_scenarios(args.config)
        return _run_command(args)
    except (ConfigurationError, EvaluationError, ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
