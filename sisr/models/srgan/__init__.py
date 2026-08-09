"""SRGAN — discriminator and paper-faithful configs. The generator is SRResNet."""

from .config import SRGANEvalConfig, SRGANTrainingConfig
from .discriminator import SRDiscriminator

__all__ = ["SRDiscriminator", "SRGANEvalConfig", "SRGANTrainingConfig"]
