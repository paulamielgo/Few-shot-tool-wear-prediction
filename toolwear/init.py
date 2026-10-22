"""Tool-wear prediction from CNC sensor and controller data.

Pipeline: a self-supervised causal TCN encodes windowed sensor features into
embeddings, and an autoregressive SVR predicts wear from those embeddings plus
controller features.
"""

from .config import (
    CutWeighting, EncoderConfig, ModelConfig, ObjectiveConfig, RegressorConfig, TCNParams,
)
from .encoder import TCNEncoder
from .model import ToolWearModel
from .tuning import SearchSpace, Tuner

__version__ = "0.1.0"

__all__ = [
    "CutWeighting", "EncoderConfig", "ModelConfig", "ObjectiveConfig", "RegressorConfig",
    "SearchSpace", "TCNEncoder", "TCNParams", "ToolWearModel", "Tuner",
]