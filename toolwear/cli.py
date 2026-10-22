"""Command-line interface: ``python -m toolwear {tune,train,predict}``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from .config import (  # noqa: E402
    BASE_SEED, CutWeighting, ModelConfig, ObjectiveConfig, TerminationConfig,
)
from .data import initial_measurements, load_labels  # noqa: E402
from .model import ToolWearModel  # noqa: E402
from .reporting import evaluate_predictions, plot_predictions  # noqa: E402
from .tuning import SearchSpace, Tuner  # noqa: E402
from .utils import configure_determinism  # noqa: E402

logger = logging.getLogger("toolwear")


def _tune(args: argparse.Namespace) -> None:
    tuner = Tuner(
        args.sensor_template, load_labels(args.labels), pd.read_csv(args.controller_features),
        args.output_dir,
        search=SearchSpace(n_trials=args.n_trials, n_inner_trials=args.n_inner_trials,
                           k_options=tuple(args.k_options)),
        objective=ObjectiveConfig(scale=args.objective_scale, teacher_forcing=args.teacher_forcing,
                                  metric=args.metric),
        cut_weighting=CutWeighting(shape=args.cut_weighting, end=args.cut_weight_end),
        termination=TerminationConfig(
            enabled=not args.no_early_stop, min_trials=args.stop_min_trials,
            threshold=args.stop_threshold,
        ),
        target_mode=args.target_mode, controller_method=args.controller_method,
        study_name=args.study_name, seed=args.seed, device=args.device,
    )
    tuner.run()
    logger.info("Best configuration and model written to %s", args.output_dir)


def _train(args: argparse.Namespace) -> None:
    config = ModelConfig.load(args.config)
    logger.info("Training with K=%d (make sure --sensor-features uses the same K).", config.K)
    model = ToolWearModel(config, args.device).fit(
        pd.read_csv(args.sensor_features), load_labels(args.labels),
        pd.read_csv(args.controller_features),
    )
    model.save(args.output_dir)


def _predict(args: argparse.Namespace) -> None:
    model = ToolWearModel.load(args.model_dir, args.device)
    labels = load_labels(args.labels) if args.labels else None
    predictions = model.predict(
        pd.read_csv(args.sensor_features), pd.read_csv(args.controller_features),
        initial_wear=initial_measurements(labels) if labels is not None else None,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    logger.info("Predictions written to %s", out_dir / "predictions.csv")

    if labels is not None and (labels["cut"] != 1).any():
        merged, metrics = evaluate_predictions(predictions, labels)
        merged.to_csv(out_dir / "predictions_vs_measured.csv", index=False)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        plot_predictions(merged, f"RMSE={metrics['rmse']:.3f}  MAPE={metrics['mape']:.2f}%  "
                                 f"R2={metrics['r2']:.3f}", out_dir / "predictions.png")
        logger.info("Metrics: %s", metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolwear", description="Tool-wear prediction with a self-supervised TCN and an SVR.",
    )
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available).")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Global random seed.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only log warnings and errors.")
    sub = parser.add_subparsers(dest="command", required=True)
    formatter = argparse.ArgumentDefaultsHelpFormatter

    tune = sub.add_parser("tune", formatter_class=formatter,
                          help="Nested hyperparameter search; saves best config and model.")
    tune.add_argument("--sensor-template", required=True,
                      help="Training sensor features path with a '{k}' placeholder.")
    tune.add_argument("--controller-features", type=Path, required=True)
    tune.add_argument("--labels", type=Path, required=True, help="Training wear labels CSV.")
    tune.add_argument("--output-dir", type=Path, required=True)
    tune.add_argument("--n-trials", type=int, default=50,
                      help="Maximum outer search trials (the search may stop earlier).")
    tune.add_argument("--n-inner-trials", type=int, default=25, help="Encoder search trials per (K, lambda).")
    tune.add_argument("--k-options", type=int, nargs="+", default=[1, 5, 10, 20])
    tune.add_argument("--target-mode", choices=["delta", "cumulative"], default="delta")
    tune.add_argument("--controller-method", choices=["mutual_info", "f_regression", "pca"],
                      default="mutual_info")
    tune.add_argument("--objective-scale", choices=["delta", "cumulative"], default="cumulative")
    tune.add_argument("--teacher-forcing", choices=["full", "partial"], default="partial")
    tune.add_argument("--metric", choices=["mse", "mae"], default="mse")
    tune.add_argument("--cut-weighting", default="sigmoid",
                      choices=["uniform", "linear", "quadratic", "exponential", "sigmoid"])
    tune.add_argument("--cut-weight-end", type=float, default=2.0, help="Weight of the last cut.")
    tune.add_argument("--no-early-stop", action="store_true",
                      help="Disable automatic termination and run all --n-trials.")
    tune.add_argument("--stop-min-trials", type=int, default=20,
                      help="Completed trials before the termination criterion is checked.")
    tune.add_argument("--stop-threshold", type=float, default=None,
                      help="Fixed regret threshold in objective units "
                           "(default: CV error of the best trial's robust score).")
    tune.add_argument("--study-name", default="toolwear")
    tune.set_defaults(func=_tune)

    train = sub.add_parser("train", formatter_class=formatter,
                           help="Fit a model from a saved configuration (no search).")
    train.add_argument("--config", type=Path, required=True, help="best_config.json from 'tune'.")
    train.add_argument("--sensor-features", type=Path, required=True)
    train.add_argument("--controller-features", type=Path, required=True)
    train.add_argument("--labels", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.set_defaults(func=_train)

    predict = sub.add_parser("predict", formatter_class=formatter,
                             help="Predict wear for new data with a saved model.")
    predict.add_argument("--model-dir", type=Path, required=True)
    predict.add_argument("--sensor-features", type=Path, required=True)
    predict.add_argument("--controller-features", type=Path, required=True)
    predict.add_argument("--labels", type=Path, default=None,
                         help="Optional labels: cut-1 wear is used as anchor; other cuts are scored.")
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.set_defaults(func=_predict)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    configure_determinism(args.seed)
    args.func(args)


if __name__ == "__main__":
    main()