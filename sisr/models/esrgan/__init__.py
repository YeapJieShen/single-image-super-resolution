"""ESRGAN: the RRDB generator, the VGG-style discriminator and their configs."""

from .config import (
    ESRGANEvalConfig,
    ESRGANTrainingConfig,
    RRDBNetEvalConfig,
    RRDBNetTrainingConfig,
)
from .discriminator import VGGStyleDiscriminator
from .model import RRDBNet

__all__ = [
    "RRDBNet",
    "VGGStyleDiscriminator",
    "RRDBNetTrainingConfig",
    "RRDBNetEvalConfig",
    "ESRGANTrainingConfig",
    "ESRGANEvalConfig",
]
