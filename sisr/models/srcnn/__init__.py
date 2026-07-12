"""SRCNN architecture and paper-faithful config defaults."""

from .config import SRCNNEvalConfig, SRCNNTrainingConfig
from .model import SRCNN

__all__ = ["SRCNN", "SRCNNTrainingConfig", "SRCNNEvalConfig"]
