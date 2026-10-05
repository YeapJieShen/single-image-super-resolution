"""EDSR architecture and paper defaults."""

from .config import EDSREvalConfig, EDSRTrainingConfig
from .model import EDSR

__all__ = ["EDSR", "EDSRTrainingConfig", "EDSREvalConfig"]
