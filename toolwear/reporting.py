"""Metrics, plots and hyperparameter-search diagnostics."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

CONVERGENCE_TAIL_FRACTION = 0.2
CONVERGENCE_PLATEAU_THRESHOLD = 0.05
TRIAL_TICK_STEP = 5


# --------------------------------------------------------------------------- #
# Prediction metrics
# --------------------------------------------------------------------------- #

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_predictions(
    predictions: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Join predictions with measurements and score them (cut 1 is excluded,
    since it is the given anchor)."""
    merged = labels.merge(predictions, on=["dataset", "cut"], how="left")
    merged["error"] = merged["prediction"] - merged["measurement"]
    scored = merged[(merged["cut"] != 1) & merged["prediction"].notna()]
    if len(scored) < len(merged[merged["cut"] != 1]):
        logger.warning("Some measured cuts have no prediction and are not scored.")
    metrics = regression_metrics(scored["measurement"].to_numpy(), scored["prediction"].to_numpy())
    return merged.sort_values(["dataset", "cut"]).reset_index(drop=True), metrics


def plot_predictions(merged: pd.DataFrame, title: str, path: str | Path) -> None:
    """One panel per dataset with measured and predicted wear."""
    datasets = sorted(merged["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4),
                             sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = merged[merged["dataset"] == ds].sort_values("cut")
        ax.plot(sub["cut"], sub["measurement"], "o-", color="tab:blue", label="measured")
        ax.plot(sub["cut"], sub["prediction"], "s--", color="tab:red", label="predicted")
        ax.set_title(f"Dataset {ds}")
        ax.set_xlabel("Cut")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Wear")
    axes[0][0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Optuna diagnostics
# --------------------------------------------------------------------------- #

def completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]


def running_best(study: optuna.Study) -> pd.DataFrame:
    """``trial_number, value, running_best`` for completed trials (minimisation)."""
    trials = sorted(completed_trials(study), key=lambda t: t.number)
    df = pd.DataFrame({"trial_number": [t.number for t in trials],
                       "value": [t.value for t in trials]})
    df["running_best"] = df["value"].cummin()
    return df


def diagnose_plateau(
    df: pd.DataFrame,
    tail_fraction: float = CONVERGENCE_TAIL_FRACTION,
    threshold: float = CONVERGENCE_PLATEAU_THRESHOLD,
) -> dict:
    """Share of the total improvement achieved in the last ``tail_fraction`` of trials.

    A small share suggests the search budget was sufficient.
    """
    n = len(df)
    if n < 5:
        return {"status": "too_few_trials", "n_trials": n}
    best = df["running_best"].to_numpy()
    n_tail = max(1, round(n * tail_fraction))
    total = float(best[0] - best[-1])
    tail = float(best[max(0, n - n_tail - 1)] - best[-1])
    share = 0.0 if total <= 0 else max(0.0, tail / total)
    return {
        "status": "converged" if share < threshold else "still_improving",
        "total_improvement": total, "tail_improvement": tail,
        "tail_share_of_improvement": share, "n_trials": n, "n_tail": n_tail,
    }


def save_convergence_report(study: optuna.Study, title: str, out_dir: str | Path, prefix: str) -> None:
    """Write ``<prefix>_running_best.csv``, ``<prefix>_convergence.png`` and ``.json``."""
    out_dir = Path(out_dir)
    df = running_best(study)
    if df.empty:
        logger.warning("No completed trials; skipping convergence report for %s.", title)
        return
    df.to_csv(out_dir / f"{prefix}_running_best.csv", index=False)

    trial = df["trial_number"] + 1  # Optuna numbers trials from 0; show them from 1
    search = "Outer" if prefix == "outer" else "Inner"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(trial, df["value"], "o", color="tab:gray", alpha=0.4, markersize=4,
            label="Current-trial objective")
    ax.step(trial, df["running_best"], where="post", color="tab:blue", linewidth=2,
            label="Best-trial objective")
    ax.set(xlabel="Trial number", ylabel=f"{search}-search objective", title=f"Convergence: {title}")
    _set_trial_axis(ax, int(trial.max()))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_convergence.png", dpi=150)
    plt.close(fig)

    diagnosis = diagnose_plateau(df)
    (out_dir / f"{prefix}_convergence.json").write_text(json.dumps(diagnosis, indent=2))
    logger.info("%s: %s", title, diagnosis["status"])


def save_trials_summary(study: optuna.Study, path: str | Path) -> pd.DataFrame:
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    df = df[df["state"] == "COMPLETE"].sort_values("value").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def save_param_plots(study: optuna.Study, params: Sequence[str], out_dir: str | Path) -> None:
    """Optuna slice, rank and parallel-coordinate plots for each single parameter."""
    import optuna.visualization.matplotlib as optuna_plt

    plots = {"slice": optuna_plt.plot_slice, "rank": optuna_plt.plot_rank,
             "parallel_coordinate": optuna_plt.plot_parallel_coordinate}
    for param in params:
        for name, plot_fn in plots.items():
            path = Path(out_dir) / f"optuna_{name}_{param}.png"
            try:
                fig = plot_fn(study, params=[param]).get_figure()
                fig.tight_layout()
                fig.savefig(path, dpi=150)
                plt.close(fig)
            except Exception as exc:  # plotting must never abort a finished search
                logger.warning("Could not create %s: %s", path.name, exc)


def save_termination_report(history: Sequence[dict], out_dir: str | Path) -> None:
    """Write ``termination_history.csv`` and ``termination.png``.

    The plot spans the trials at which the criterion was checked (from
    ``--stop-min-trials`` to the last trial).
    """
    if not history:
        return
    out_dir = Path(out_dir)
    df = pd.DataFrame(history)
    df.to_csv(out_dir / "termination_history.csv", index=False)

    trial = df["trial_number"] + 1  # Optuna numbers trials from 0; show them from 1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(trial, df["regret_bound"], color="tab:blue", linewidth=2,
            label="Estimated potential improvement")
    ax.plot(trial, df["threshold"], color="tab:orange", linewidth=2,
            label="Stopping threshold")
    ax.set(xlabel="Trial number", ylabel="Outer-search objective units",
           title="Automatic termination")
    _set_trial_axis(ax, int(trial.max()), first_trial=int(trial.min()))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "termination.png", dpi=150)
    plt.close(fig)


def _set_trial_axis(ax: Axes, last_trial: int, first_trial: int = 1,
                    step: int = TRIAL_TICK_STEP) -> None:
    """x-axis from ``first_trial`` to ``last_trial`` with integer ticks at the
    first trial and at every multiple of ``step`` (e.g. 1, 5, 10, ... or 20, 25, ...)."""
    first_multiple = -(-first_trial // step) * step
    ticks = sorted({first_trial, *range(first_multiple, last_trial + 1, step)})
    ax.set_xticks(ticks)
    ax.set_xlim(first_trial - 0.5, last_trial + 0.5)  # half-trial margin at both edges