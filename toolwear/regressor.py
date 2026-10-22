"""Autoregressive SVR wear regressor and its leave-one-dataset-out evaluation.

The regressor predicts, for each measured cut, either the wear increment since
the previous measured cut (``target_mode="delta"``) or the absolute wear
(``target_mode="cumulative"``). Its inputs are the frozen sensor embedding,
controller features, anchor features (cut number, initial wear) and one
autoregressive feature (previous increment or previous absolute wear).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .config import BASE_SEED, CutWeighting, ObjectiveConfig, RegressorConfig

AUTOREG_COLUMNS = {"delta": "prev_delta", "cumulative": "prev_cumulative"}
TARGET_COLUMNS = {"delta": "delta_measurement", "cumulative": "measurement_cur"}
ANCHOR_COLS = ["cut_number", "initial_measurement"]

_SIGMOID_MIDPOINT = 0.85
_SIGMOID_STEEPNESS = 25.0


def embedding_columns(embed_dim: int) -> list[str]:
    return [f"emb_{i}" for i in range(embed_dim)]


# --------------------------------------------------------------------------- #
# Cut weights
# --------------------------------------------------------------------------- #

def cut_weights(cuts: Sequence[float], weighting: CutWeighting) -> np.ndarray:
    """Weight per row: 1.0 at the earliest cut, ``weighting.end`` at the latest."""
    cuts = np.asarray(cuts, dtype=np.float64)
    if weighting.shape == "uniform" or cuts.size == 0:
        return np.ones_like(cuts)

    x = (cuts - cuts.min()) / max(cuts.max() - cuts.min(), 1e-9)
    end = weighting.end
    if weighting.shape == "linear":
        ramp = x
    elif weighting.shape == "quadratic":
        ramp = x ** 2
    elif weighting.shape == "exponential":
        ramp = x if end == 1.0 else (end ** x - 1.0) / (end - 1.0)
    elif weighting.shape == "sigmoid":
        logistic = lambda z: 1.0 / (1.0 + np.exp(-_SIGMOID_STEEPNESS * (z - _SIGMOID_MIDPOINT)))  # noqa: E731
        ramp = (logistic(x) - logistic(0.0)) / (logistic(1.0) - logistic(0.0))
    else:
        raise ValueError(f"Unknown cut weighting shape: {weighting.shape!r}")
    return 1.0 + (end - 1.0) * ramp


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def resolve_regressor_config(
    cfg: RegressorConfig, embed_dim: int, n_controller_features: int, n_samples: int
) -> RegressorConfig:
    """Clip PCA components and controller_k to values valid for the data size."""
    def clip(value, n_features):
        if value is None:
            return None
        max_valid = min(n_features - 1, n_samples - 2)
        return min(value, max_valid) if max_valid >= 1 else None

    return replace(cfg,
                   pca_n_components=clip(cfg.pca_n_components, embed_dim),
                   controller_k=clip(cfg.controller_k, n_controller_features))


def _controller_reducer(method: str, k: int, seed: int):
    if method == "mutual_info":
        return SelectKBest(functools.partial(mutual_info_regression, random_state=seed), k=k)
    if method == "f_regression":
        return SelectKBest(f_regression, k=k)
    if method == "pca":
        return PCA(n_components=k, random_state=seed)
    raise ValueError(f"Unknown controller feature method: {method!r}")


def build_svr_pipeline(
    cfg: RegressorConfig,
    embed_cols: Sequence[str],
    controller_cols: Sequence[str],
    seed: int = BASE_SEED,
) -> Pipeline:
    """Feature blocks (embedding, controller, anchor, autoregressive) + RBF SVR."""
    embed_step = (PCA(n_components=cfg.pca_n_components, random_state=seed)
                  if cfg.pca_n_components else "passthrough")
    transformers = [("embed", embed_step, list(embed_cols))]
    if cfg.controller_k and controller_cols:
        transformers.append(("ctrl", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("reduce", _controller_reducer(cfg.controller_method, cfg.controller_k, seed)),
        ]), list(controller_cols)))
    transformers += [
        ("anchor", StandardScaler(), ANCHOR_COLS),
        ("autoreg", StandardScaler(), [AUTOREG_COLUMNS[cfg.autoreg_mode]]),
    ]
    return Pipeline([
        ("features", ColumnTransformer(transformers)),
        ("model", SVR(kernel="rbf", C=cfg.svr_C, epsilon=cfg.svr_epsilon, gamma=cfg.svr_gamma)),
    ])


def fit_regressor(
    cfg: RegressorConfig,
    X_static: pd.DataFrame,
    transitions: pd.DataFrame,
    embed_cols: Sequence[str],
    controller_cols: Sequence[str],
    weighting: CutWeighting,
    seed: int = BASE_SEED,
) -> Pipeline:
    """Fit on all transitions with teacher forcing and per-cut sample weights."""
    autoreg_col = AUTOREG_COLUMNS[cfg.autoreg_mode]
    X = X_static.reset_index(drop=True).assign(**{autoreg_col: transitions[autoreg_col].to_numpy()})
    y = transitions[TARGET_COLUMNS[cfg.target_mode]].to_numpy(dtype=np.float64)
    estimator = build_svr_pipeline(cfg, embed_cols, controller_cols, seed)
    estimator.fit(X, y, model__sample_weight=cut_weights(transitions["cut"], weighting))
    return estimator


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

def predict_chained(
    estimator: Pipeline,
    X_static: pd.DataFrame,
    cfg: RegressorConfig,
    start_value: float,
    start_prev_delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict rows in order, feeding each prediction into the next input.

    Returns ``(pred_target, pred_cumulative)``: the prediction in the target's
    own scale and the reconstructed absolute wear.
    """
    autoreg_col = AUTOREG_COLUMNS[cfg.autoreg_mode]
    current, prev_delta = float(start_value), float(start_prev_delta)
    preds, cumulative = [], []
    for i in range(len(X_static)):
        autoreg_value = prev_delta if autoreg_col == "prev_delta" else current
        row = X_static.iloc[[i]].assign(**{autoreg_col: autoreg_value})
        pred = float(estimator.predict(row)[0])
        if cfg.target_mode == "delta":
            pred = max(pred, cfg.clip_delta_min)
            prev_delta, current = pred, current + pred
        else:
            prev_delta, current = pred - current, pred
        preds.append(pred)
        cumulative.append(current)
    return np.asarray(preds), np.asarray(cumulative)


def predict_teacher_forced(
    estimator: Pipeline, X: pd.DataFrame, cfg: RegressorConfig, start_value: float
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised prediction using the true autoregressive input already in ``X``."""
    preds = np.asarray(estimator.predict(X), dtype=np.float64)
    if cfg.target_mode == "delta":
        preds = np.maximum(preds, cfg.clip_delta_min)
        return preds, start_value + np.cumsum(preds)
    return preds, preds.copy()


# --------------------------------------------------------------------------- #
# Leave-one-dataset-out cross-validation
# --------------------------------------------------------------------------- #

@dataclass
class CVResult:
    score: float
    mean: float
    std: float
    pooled: float
    fold_scores: list[float]
    fold_datasets: list[int]


def _weighted_error(y_true, y_pred, weights, metric: str) -> float:
    err = np.abs(y_true - y_pred) if metric == "mae" else (y_true - y_pred) ** 2
    return float(np.average(err, weights=weights))


def lodo_cv_score(
    pipeline: Pipeline,
    X_static: pd.DataFrame,
    transitions: pd.DataFrame,
    cfg: RegressorConfig,
    objective: ObjectiveConfig,
    weights: np.ndarray,
) -> CVResult:
    """Robust, cut-weighted leave-one-dataset-out score.

    Each fold is fitted with teacher forcing (true autoregressive inputs, no
    sample weights). The held-out dataset is predicted according to
    ``objective.teacher_forcing`` and its error is weighted per cut. The
    returned score is ``mean + std`` over folds, which penalises
    configurations that only work for favourable folds.
    """
    if objective.scale == "delta" and cfg.target_mode != "delta":
        raise ValueError("objective.scale='delta' requires target_mode='delta'.")

    transitions = transitions.reset_index(drop=True)
    X_static = X_static.reset_index(drop=True)
    autoreg_col = AUTOREG_COLUMNS[cfg.autoreg_mode]
    X = X_static.assign(**{autoreg_col: transitions[autoreg_col].to_numpy()})
    y_fit = transitions[TARGET_COLUMNS[cfg.target_mode]].to_numpy(dtype=np.float64)
    y_eval = transitions[TARGET_COLUMNS[objective.scale]].to_numpy(dtype=np.float64)
    groups = transitions["dataset"].to_numpy()
    cuts = transitions["cut"].to_numpy()

    oof = np.full(len(transitions), np.nan)
    fold_scores, fold_datasets = [], []
    for dataset in np.unique(groups):
        in_fold = groups == dataset
        estimator = clone(pipeline).fit(X[~in_fold], y_fit[~in_fold])

        idx = np.flatnonzero(in_fold)
        idx = idx[np.argsort(cuts[idx], kind="stable")]
        first = transitions.iloc[idx[0]]
        if objective.teacher_forcing == "full":
            pred_t, pred_c = predict_teacher_forced(estimator, X.iloc[idx], cfg, first["prev_cumulative"])
        else:
            pred_t, pred_c = predict_chained(estimator, X_static.iloc[idx], cfg,
                                             first["prev_cumulative"], first["prev_delta"])
        pred = pred_t if objective.scale == "delta" else pred_c

        oof[idx] = pred
        fold_scores.append(_weighted_error(y_eval[idx], pred, weights[idx], objective.metric))
        fold_datasets.append(int(dataset))

    scores = np.asarray(fold_scores)
    mean = float(scores.mean())
    std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
    return CVResult(
        score=mean + std, mean=mean, std=std,
        pooled=_weighted_error(y_eval, oof, weights, objective.metric),
        fold_scores=fold_scores, fold_datasets=fold_datasets,
    )