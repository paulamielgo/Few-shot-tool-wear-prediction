"""End-to-end tool-wear model: SSL encoder + autoregressive SVR regressor."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .config import PREDICT_CHECKPOINTS, ModelConfig
from .data import (
    anchor_matrix, attach_positions, build_prediction_grid, build_sequences,
    build_transitions, controller_feature_columns, controller_matrix,
    initial_measurements, label_position, prune_correlated_features,
)
from .encoder import TCNEncoder
from .regressor import embedding_columns, fit_regressor, predict_chained, resolve_regressor_config
from .utils import resolve_device

logger = logging.getLogger(__name__)


def build_static_features(
    frame: pd.DataFrame,
    embeddings: Mapping[int, np.ndarray],
    embed_dim: int,
    controller_df: pd.DataFrame,
    controller_cols: Sequence[str],
    anchors: Mapping[int, float],
    anchor_fallback: float,
) -> pd.DataFrame:
    """Non-autoregressive regressor inputs for each ``(dataset, cut, position)`` row."""
    emb = pd.DataFrame(
        [embeddings[int(ds)][int(pos)] for ds, pos in zip(frame["dataset"], frame["position"])],
        columns=embedding_columns(embed_dim),
    )
    parts = [emb]
    if controller_cols:
        parts.append(controller_matrix(frame, controller_df, controller_cols))
    parts.append(anchor_matrix(frame, anchors, anchor_fallback))
    return pd.concat(parts, axis=1)


class ToolWearModel:
    """Predicts cumulative tool wear at the checkpoint cuts of each dataset.

    Example:
        >>> model = ToolWearModel(ModelConfig.load("best_config.json"))
        >>> model.fit(train_sensor_df, train_labels, train_controller_df)
        >>> model.save("artifacts/model")
        >>> ToolWearModel.load("artifacts/model").predict(sensor_df, controller_df, {1: 0.0})
    """

    ENCODER_DIR = "encoder"
    REGRESSOR_FILE = "regressor.joblib"
    METADATA_FILE = "model.json"

    def __init__(self, config: ModelConfig, device: str | torch.device | None = None) -> None:
        self.config = config
        self.device = resolve_device(device)
        self.encoder_: TCNEncoder | None = None
        self.regressor_ = None
        self.feature_cols_: list[str] = []
        self.controller_cols_: list[str] = []
        self.anchor_fallback_: float = float("nan")

    # ------------------------------------------------------------------ #
    def fit(
        self,
        sensor_features: pd.DataFrame,
        labels: pd.DataFrame,
        controller_features: pd.DataFrame,
        encoder: TCNEncoder | None = None,
    ) -> ToolWearModel:
        """Fit the encoder (unless a fitted one is given) and the regressor.

        Args:
            sensor_features: Windowed sensor features for ``config.K`` windows per cut.
            labels: ``[dataset, cut, measurement]`` (see :func:`toolwear.data.load_labels`).
            controller_features: Cumulative controller features per ``(dataset, cut)``.
            encoder: Optional pre-trained encoder (e.g. from the tuner cache).
        """
        cfg = self.config
        sequences, metas, self.feature_cols_ = build_sequences(sensor_features)

        if encoder is None:
            logger.info("Training SSL encoder for %d epochs...", cfg.encoder.n_epochs)
            encoder = TCNEncoder(cfg.encoder, seed=cfg.seed, device=self.device).fit(sequences.values())
        self.encoder_ = encoder

        labels = attach_positions(labels, metas)
        transitions = build_transitions(labels)
        anchors = initial_measurements(labels)
        self.anchor_fallback_ = float(np.median(list(anchors.values())))

        pruned = prune_correlated_features(controller_features,
                                           controller_feature_columns(controller_features),
                                           cfg.controller_corr_threshold)
        regressor_cfg = resolve_regressor_config(cfg.regressor, encoder.embed_dim,
                                                 len(pruned), len(transitions))
        self.config = replace(cfg, encoder=encoder.config, regressor=regressor_cfg)
        self.controller_cols_ = pruned if regressor_cfg.controller_k else []

        X_static = build_static_features(
            transitions, encoder.embed(sequences), encoder.embed_dim, controller_features,
            self.controller_cols_, anchors, self.anchor_fallback_,
        )
        self.regressor_ = fit_regressor(
            regressor_cfg, X_static, transitions, embedding_columns(encoder.embed_dim),
            self.controller_cols_, cfg.cut_weighting, cfg.seed,
        )
        return self

    # ------------------------------------------------------------------ #
    def predict(
        self,
        sensor_features: pd.DataFrame,
        controller_features: pd.DataFrame,
        initial_wear: Mapping[int, float] | None = None,
        checkpoints: Iterable[int] = PREDICT_CHECKPOINTS,
    ) -> pd.DataFrame:
        """Predict cumulative wear at cut 1 and at every checkpoint.

        Args:
            sensor_features: Windowed sensor features of the new datasets.
            controller_features: Controller features of the new datasets.
            initial_wear: Measured wear at cut 1 per dataset. As in training,
                a measurement is only used if the dataset has sensor data up to
                cut 1; otherwise, and for datasets not included, the training
                median is used.
            checkpoints: Cuts at which to predict.

        Returns:
            DataFrame ``[dataset, cut, prediction]``. Checkpoints without sensor
            data carry the previous prediction forward.
        """
        if self.regressor_ is None:
            raise RuntimeError("Model is not fitted.")
        checkpoints = list(checkpoints)
        sequences, metas, _ = build_sequences(sensor_features, self.feature_cols_)

        anchors = {}
        for ds, wear in (initial_wear or {}).items():
            if label_position(metas.get(ds, []), 1)[0] is None:
                logger.warning("No sensor data for dataset %d, cut 1; its cut-1 measurement "
                               "is not used.", ds)
            else:
                anchors[ds] = wear
        grid = build_prediction_grid(metas, checkpoints)
        X_static = build_static_features(
            grid, self.encoder_.embed(sequences), self.encoder_.embed_dim, controller_features,
            self.controller_cols_, anchors, self.anchor_fallback_,
        )

        rows = []
        for ds in sorted(metas):
            start = anchors.get(ds, self.anchor_fallback_)
            idx = np.flatnonzero(grid["dataset"].to_numpy() == ds)
            _, cumulative = predict_chained(self.regressor_, X_static.iloc[idx],
                                            self.config.regressor, start)
            by_cut = dict(zip(grid["cut"].iloc[idx], cumulative))
            current = start
            rows.append((ds, 1, current))
            for cut in checkpoints:
                current = by_cut.get(cut, current)
                rows.append((ds, cut, current))
        return pd.DataFrame(rows, columns=["dataset", "cut", "prediction"])

    # ------------------------------------------------------------------ #
    def save(self, directory: str | Path) -> None:
        if self.regressor_ is None:
            raise RuntimeError("Model is not fitted.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder_.save(directory / self.ENCODER_DIR)
        joblib.dump(self.regressor_, directory / self.REGRESSOR_FILE)
        metadata = {
            "config": self.config.to_dict(),
            "feature_cols": self.feature_cols_,
            "controller_cols": self.controller_cols_,
            "anchor_fallback": self.anchor_fallback_,
        }
        (directory / self.METADATA_FILE).write_text(json.dumps(metadata, indent=2))
        logger.info("Model saved to %s", directory)

    @classmethod
    def load(cls, directory: str | Path, device: str | torch.device | None = None) -> ToolWearModel:
        directory = Path(directory)
        metadata = json.loads((directory / cls.METADATA_FILE).read_text())
        model = cls(ModelConfig.from_dict(metadata["config"]), device)
        model.encoder_ = TCNEncoder.load(directory / cls.ENCODER_DIR, model.device)
        model.regressor_ = joblib.load(directory / cls.REGRESSOR_FILE)
        model.feature_cols_ = metadata["feature_cols"]
        model.controller_cols_ = metadata["controller_cols"]
        model.anchor_fallback_ = metadata["anchor_fallback"]
        return model