"""Nested Optuna hyperparameter search.

* Outer search: chooses K (windows per cut), the encoder loss weight
  ``lambda_loss`` and every regressor hyperparameter, scored with a robust
  leave-one-dataset-out CV of the regressor.
* Inner search: for each new ``(K, lambda_loss)`` pair, chooses the TCN
  hyperparameters with a leave-one-dataset-out CV of the self-supervised loss,
  then trains the encoder on all training data. Encoders are cached on disk
  and reused by later trials and later runs.

The outer search stops automatically (Makarova et al., 2022) once a
high-probability bound on its remaining regret falls below the CV estimation
error of the best trial; ``n_trials`` then acts as a maximum budget.

Both searches are persisted in an SQLite storage and can be resumed.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.terminator import report_cross_validation_scores

from .config import (
    BASE_SEED, CutWeighting, EncoderConfig, ModelConfig, ObjectiveConfig,
    RegressorConfig, TCNParams, TerminationConfig,
)
from .data import (
    attach_positions, build_sequences, build_transitions, controller_feature_columns,
    initial_measurements, prune_correlated_features,
)
from .encoder import SequencePreprocessor, TCNEncoder, sequence_variance, train_with_early_stopping
from .model import ToolWearModel, build_static_features
from .regressor import (
    build_svr_pipeline, cut_weights, embedding_columns, lodo_cv_score, resolve_regressor_config,
)
from .reporting import (
    completed_trials, save_convergence_report, save_termination_report, save_trials_summary,
)
from .termination import RegretBoundStopper
from .utils import resolve_device

logger = logging.getLogger(__name__)


@dataclass
class SearchSpace:
    """Budgets and candidate values of both searches.

    ``n_trials`` is the maximum budget of the outer search; it may stop earlier
    if automatic termination is enabled.
    """

    n_trials: int = 50
    n_inner_trials: int = 25
    inner_max_epochs: int = 2000
    inner_patience: int = 10
    k_options: tuple[int, ...] = (1, 5, 10, 20)
    lambda_options: tuple[float, ...] = tuple(round(i / 10, 1) for i in range(11))
    channels: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 384)
    kernel_sizes: tuple[int, ...] = (2, 3, 5, 7)
    batch_sizes: tuple[int, ...] = (1, 2, 3)
    pca_components: tuple[int, ...] = (3, 5, 10, 15, 20)
    controller_k: tuple[int, ...] = (3, 5, 10, 15, 20, 30)
    autoreg_modes: tuple[str, ...] = ("delta", "cumulative")


@dataclass
class _KData:
    sequences: dict[int, np.ndarray]
    transitions: pd.DataFrame
    anchors: dict[int, float]
    anchor_fallback: float


def _n_complete(study: optuna.Study) -> int:
    return len(completed_trials(study))


def suggest_tcn_params(trial: optuna.Trial, space: SearchSpace) -> TCNParams:
    return TCNParams(
        channels=trial.suggest_categorical("channels", list(space.channels)),
        num_layers=trial.suggest_int("num_layers", 2, 8),
        kernel_size=trial.suggest_categorical("kernel_size", list(space.kernel_sizes)),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        mask_ratio=trial.suggest_float("mask_ratio", 0.10, 0.35),
        mean_span_length=trial.suggest_int("mean_span_length", 2, 8),
        margin=trial.suggest_float("margin", 0.1, 1.0),
        batch_size=trial.suggest_categorical("batch_size", list(space.batch_sizes)),
    )


class Tuner:
    """Runs the nested search and fits the best model.

    Args:
        sensor_template: Path of the training sensor features with a ``{k}``
            placeholder, e.g. ``"features/features_sensor_train_K{k}.csv"``.
        labels: Training wear labels (``[dataset, cut, measurement]``).
        controller_features: Training controller features.
        output_dir: Directory for the study storage, encoder cache and reports.
    """

    def __init__(
        self,
        sensor_template: str,
        labels: pd.DataFrame,
        controller_features: pd.DataFrame,
        output_dir: str | Path,
        *,
        search: SearchSpace | None = None,
        objective: ObjectiveConfig | None = None,
        cut_weighting: CutWeighting | None = None,
        termination: TerminationConfig | None = None,
        target_mode: str = "delta",
        controller_method: str = "mutual_info",
        corr_threshold: float = 0.95,
        study_name: str = "toolwear",
        storage: str | None = None,
        seed: int = BASE_SEED,
        device: str | torch.device | None = None,
    ) -> None:
        if "{k}" not in sensor_template:
            raise ValueError("sensor_template must contain a '{k}' placeholder.")
        self.search = search or SearchSpace()
        self.objective = objective or ObjectiveConfig()
        if self.objective.scale == "delta" and target_mode != "delta":
            raise ValueError("objective scale 'delta' requires target_mode 'delta'.")

        self.sensor_template = sensor_template
        self.labels = labels
        self.controller_features = controller_features
        self.cut_weighting = cut_weighting or CutWeighting()
        self.termination = termination or TerminationConfig()
        self.target_mode = target_mode
        self.controller_method = controller_method
        self.corr_threshold = corr_threshold
        self.study_name = study_name
        self.seed = seed
        self.device = resolve_device(device)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.storage = storage or f"sqlite:///{(self.output_dir / 'optuna.db').resolve()}"

        self.controller_cols = prune_correlated_features(
            controller_features, controller_feature_columns(controller_features), corr_threshold)
        self._kdata: dict[int, _KData] = {}
        self._folds: dict[int, list] = {}
        self._features: dict[tuple[int, float], tuple[int, pd.DataFrame]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> ToolWearModel:
        """Run (or resume) the outer search, write reports and fit the best model.

        Outputs in ``output_dir``: ``trials_summary.csv``, the outer convergence
        and termination reports, ``best_config.json`` and the fitted model in
        ``model/``.
        """
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.seed),
            storage=self.storage, study_name=self.study_name, load_if_exists=True,
        )
        callbacks = []
        if self.termination.enabled:
            callbacks.append(RegretBoundStopper(self.termination, self.seed))

        remaining = self.search.n_trials - _n_complete(study)
        if self.termination.enabled and RegretBoundStopper.already_stopped(study):
            logger.info("Outer search '%s' was already terminated automatically.", self.study_name)
        elif remaining > 0:
            logger.info("Outer search '%s': up to %d trials to run.", self.study_name, remaining)
            study.optimize(self._objective, n_trials=remaining, callbacks=callbacks)

        save_trials_summary(study, self.output_dir / "trials_summary.csv")
        save_convergence_report(study, "outer search", self.output_dir, "outer")
        save_termination_report(RegretBoundStopper.history(study), self.output_dir)

        best = study.best_trial
        logger.info("Best trial %d: score=%.4f, K=%s, lambda=%s",
                    best.number, best.value, best.params["K"], best.params["lambda_loss"])
        return self.fit_trial(best)

    def fit_trial(self, trial: optuna.trial.FrozenTrial) -> ToolWearModel:
        """Fit and save the model defined by a completed outer trial."""
        K, lambda_loss = trial.params["K"], trial.params["lambda_loss"]
        encoder = self._get_encoder(K, lambda_loss)
        config = ModelConfig(
            K=K, encoder=encoder.config,
            regressor=RegressorConfig(**trial.user_attrs["regressor_config"]),
            cut_weighting=self.cut_weighting, controller_corr_threshold=self.corr_threshold,
            seed=self.seed,
        )
        config.save(self.output_dir / "best_config.json")

        model = ToolWearModel(config, self.device).fit(
            pd.read_csv(self.sensor_template.format(k=K)), self.labels,
            self.controller_features, encoder=encoder,
        )
        model.save(self.output_dir / "model")
        return model

    # ------------------------------------------------------------------ #
    # Outer search
    # ------------------------------------------------------------------ #
    def _objective(self, trial: optuna.Trial) -> float:
        space = self.search
        K = trial.suggest_categorical("K", list(space.k_options))
        lambda_loss = trial.suggest_categorical("lambda_loss", list(space.lambda_options))
        embed_dim, X_static = self._static_features(K, lambda_loss)
        kdata = self._k_data(K)

        use_pca = trial.suggest_categorical("use_pca", [False, True])
        pca = trial.suggest_categorical("pca_n_components", list(space.pca_components)) if use_pca else None
        controller_k = trial.suggest_categorical("controller_k", list(space.controller_k))
        autoreg_mode = trial.suggest_categorical("autoreg_mode", list(space.autoreg_modes))
        cfg = RegressorConfig(
            svr_C=trial.suggest_float("svr_C", 0.1, 100.0, log=True),
            svr_epsilon=trial.suggest_float("svr_epsilon", 0.1, 5.0, log=True),
            svr_gamma=trial.suggest_categorical("svr_gamma", ["scale", "auto"]),
            pca_n_components=pca, controller_k=controller_k,
            controller_method=self.controller_method,
            autoreg_mode=autoreg_mode, target_mode=self.target_mode,
        )
        cfg = resolve_regressor_config(cfg, embed_dim, len(self.controller_cols), len(kdata.transitions))
        trial.set_user_attr("regressor_config", asdict(cfg))

        controller_cols = self.controller_cols if cfg.controller_k else []
        pipeline = build_svr_pipeline(cfg, embedding_columns(embed_dim), controller_cols, self.seed)
        weights = cut_weights(kdata.transitions["cut"], self.cut_weighting)
        try:
            result = lodo_cv_score(pipeline, X_static, kdata.transitions, cfg, self.objective, weights)
        except ValueError as exc:
            raise optuna.TrialPruned(f"CV failed: {exc}") from exc
        if not np.isfinite(result.score):
            raise optuna.TrialPruned("Non-finite CV score.")

        trial.set_user_attr("cv_mean", result.mean)
        trial.set_user_attr("cv_std", result.std)
        trial.set_user_attr("cv_pooled", result.pooled)
        trial.set_user_attr("cv_fold_scores", result.fold_scores)
        trial.set_user_attr("cv_fold_datasets", result.fold_datasets)
        report_cross_validation_scores(trial, result.fold_scores)
        return result.score

    # ------------------------------------------------------------------ #
    # Per-K data and encoder cache
    # ------------------------------------------------------------------ #
    def _k_data(self, K: int) -> _KData:
        if K not in self._kdata:
            sequences, metas, _ = build_sequences(pd.read_csv(self.sensor_template.format(k=K)))
            labels = attach_positions(self.labels, metas)
            anchors = initial_measurements(labels)
            self._kdata[K] = _KData(sequences, build_transitions(labels), anchors,
                                    float(np.median(list(anchors.values()))))
        return self._kdata[K]

    def _encoder_dir(self, K: int, lambda_loss: float) -> Path:
        return self.output_dir / "encoders" / f"K{K}" / f"lambda_{lambda_loss:.1f}"

    def _get_encoder(self, K: int, lambda_loss: float) -> TCNEncoder:
        directory = self._encoder_dir(K, lambda_loss)
        if (directory / TCNEncoder.WEIGHTS_FILE).exists():
            logger.info("Loading cached encoder K=%d, lambda=%.1f", K, lambda_loss)
            return TCNEncoder.load(directory, self.device)

        logger.info("Training encoder K=%d, lambda=%.1f from scratch", K, lambda_loss)
        config = self._inner_search(K, lambda_loss, directory)
        encoder = TCNEncoder(config, seed=self.seed, device=self.device)
        encoder.fit(self._k_data(K).sequences.values())
        encoder.save(directory)
        return encoder

    def _static_features(self, K: int, lambda_loss: float) -> tuple[int, pd.DataFrame]:
        """Cached regressor inputs for a ``(K, lambda)`` pair (encoder is not kept in memory)."""
        key = (K, lambda_loss)
        if key not in self._features:
            encoder = self._get_encoder(K, lambda_loss)
            kdata = self._k_data(K)
            X_static = build_static_features(
                kdata.transitions, encoder.embed(kdata.sequences), encoder.embed_dim,
                self.controller_features, self.controller_cols, kdata.anchors, kdata.anchor_fallback,
            )
            self._features[key] = (encoder.embed_dim, X_static)
            del encoder
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return self._features[key]

    # ------------------------------------------------------------------ #
    # Inner search (encoder hyperparameters)
    # ------------------------------------------------------------------ #
    def _cv_folds(self, K: int) -> list[tuple[list[np.ndarray], np.ndarray, float]]:
        """Leave-one-dataset-out folds, preprocessed with statistics of the training part."""
        if K not in self._folds:
            sequences = self._k_data(K).sequences
            folds = []
            for held_out in sorted(sequences):
                train = [sequences[d] for d in sorted(sequences) if d != held_out]
                preprocessor = SequencePreprocessor().fit(train)
                train_t = [preprocessor.transform(s) for s in train]
                folds.append((train_t, preprocessor.transform(sequences[held_out]),
                              sequence_variance(train_t)))
            self._folds[K] = folds
        return self._folds[K]

    def _inner_search(self, K: int, lambda_loss: float, out_dir: Path) -> EncoderConfig:
        folds = self._cv_folds(K)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed + K * 1000 + round(lambda_loss * 10)),
            storage=self.storage, study_name=f"ssl_K{K}_lambda{lambda_loss:.2f}", load_if_exists=True,
        )
        remaining = self.search.n_inner_trials - _n_complete(study)
        if remaining > 0:
            study.optimize(lambda t: self._inner_objective(t, folds, lambda_loss), n_trials=remaining)

        out_dir.mkdir(parents=True, exist_ok=True)

        best = study.best_trial
        n_epochs = max(1, round(float(np.mean(best.user_attrs["fold_best_epochs"]))))
        return EncoderConfig(tcn=TCNParams(**best.params), lambda_loss=lambda_loss, n_epochs=n_epochs)

    def _inner_objective(self, trial: optuna.Trial, folds, lambda_loss: float) -> float:
        params = suggest_tcn_params(trial, self.search)
        losses, best_epochs = [], []
        for i, (train_t, val_t, var_target) in enumerate(folds):
            loss, epoch = train_with_early_stopping(
                train_t, val_t, params, lambda_loss, var_target,
                device=self.device, max_epochs=self.search.inner_max_epochs,
                patience=self.search.inner_patience, seed=self.seed + trial.number * 1000 + i,
            )
            losses.append(loss)
            best_epochs.append(epoch)
        trial.set_user_attr("fold_best_epochs", best_epochs)
        return float(np.mean(losses))