"""Configuration objects shared across the package.

A :class:`ModelConfig` fully describes a trained model (encoder + regressor), so
the best configuration found by the tuner can be saved as JSON and used later
to retrain the model without repeating the hyperparameter search.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

BASE_SEED = 42
ID_COLS = ("dataset", "cut", "window", "step")
PREDICT_CHECKPOINTS = (6, 11, 16, 21, 26)


@dataclass
class TCNParams:
    """Architecture and self-supervised training hyperparameters of the TCN."""

    channels: int = 64
    num_layers: int = 4
    kernel_size: int = 3
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    mask_ratio: float = 0.2
    mean_span_length: int = 3
    margin: float = 0.5
    batch_size: int = 2
    n_triplets: int = 20


@dataclass
class EncoderConfig:
    """Self-supervised encoder configuration.

    Attributes:
        tcn: Network and optimisation hyperparameters.
        lambda_loss: Weight of the reconstruction loss; ``1 - lambda_loss`` is
            the weight of the triplet loss.
        n_epochs: Number of epochs of the final fit (no early stopping).
        max_dilation: Upper bound for the dilation ``2**i`` of layer ``i``.
    """

    tcn: TCNParams = field(default_factory=TCNParams)
    lambda_loss: float = 0.5
    n_epochs: int = 100
    max_dilation: int = 32

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EncoderConfig:
        data = dict(data)
        return cls(tcn=TCNParams(**data.pop("tcn", {})), **data)


@dataclass
class RegressorConfig:
    """SVR regressor and feature-block configuration.

    Attributes:
        pca_n_components: PCA components for the embedding block
            (``None`` keeps the raw embedding).
        controller_k: Number of controller features kept
            (``None`` disables the controller block).
        controller_method: ``"mutual_info"``, ``"f_regression"`` or ``"pca"``.
        autoreg_mode: Autoregressive input: ``"delta"`` (previous increment)
            or ``"cumulative"`` (previous absolute wear).
        target_mode: Regression target: ``"delta"`` (wear increment) or
            ``"cumulative"`` (absolute wear).
        clip_delta_min: Lower bound for predicted increments.
    """

    svr_C: float = 1.0
    svr_epsilon: float = 0.5
    svr_gamma: str = "scale"
    pca_n_components: Optional[int] = None
    controller_k: Optional[int] = 10
    controller_method: str = "mutual_info"
    autoreg_mode: str = "delta"
    target_mode: str = "delta"
    clip_delta_min: float = 0.0


@dataclass
class CutWeighting:
    """Per-cut weights that grow from 1.0 at the first cut to ``end`` at the last.

    ``shape`` is one of ``"uniform"``, ``"linear"``, ``"quadratic"``,
    ``"exponential"`` or ``"sigmoid"``.
    """

    shape: str = "sigmoid"
    end: float = 2.0


@dataclass
class ObjectiveConfig:
    """Cross-validation objective minimised by the hyperparameter search.

    Attributes:
        scale: Error scale, ``"delta"`` or ``"cumulative"`` (reconstructed wear).
        teacher_forcing: ``"full"`` uses the true autoregressive input when
            validating; ``"partial"`` chains the model's own predictions, as
            at inference time.
        metric: ``"mse"`` or ``"mae"``.

    The score is always the robust ``mean(fold_errors) + std(fold_errors)``.
    """

    scale: str = "cumulative"
    teacher_forcing: str = "partial"
    metric: str = "mse"


@dataclass
class TerminationConfig:
    """Automatic termination of the outer search (Makarova et al., AutoML 2022).

    The search stops once Optuna's ``RegretBoundEvaluator`` bound on the
    remaining regret falls below the statistical error of the robust CV score
    (``mean + std``) of the best trial.

    Attributes:
        enabled: Whether to use the criterion (``n_trials`` becomes a maximum).
        min_trials: Completed trials before the criterion is checked; also the
            minimum number of top trials used by the GP.
        top_trials_ratio: Fraction of best trials used by the GP.
        threshold: Fixed regret threshold in objective units; ``None`` uses the
            CV error of the best trial.
    """

    enabled: bool = True
    min_trials: int = 20
    top_trials_ratio: float = 0.5
    threshold: Optional[float] = None


@dataclass
class ModelConfig:
    """Complete description of a tool-wear model."""

    K: int = 5
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    regressor: RegressorConfig = field(default_factory=RegressorConfig)
    cut_weighting: CutWeighting = field(default_factory=CutWeighting)
    controller_corr_threshold: float = 0.95
    seed: int = BASE_SEED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        data = dict(data)
        return cls(
            encoder=EncoderConfig.from_dict(data.pop("encoder", {})),
            regressor=RegressorConfig(**data.pop("regressor", {})),
            cut_weighting=CutWeighting(**data.pop("cut_weighting", {})),
            **data,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))