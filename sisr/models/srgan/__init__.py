"""SRGAN — discriminator and paper-faithful configs. The generator is SRResNet."""

from .base import AdversarialDiscriminator
from .config import SRGANEvalConfig, SRGANTrainingConfig
from .discriminator import SRDiscriminator

__all__ = ["AdversarialDiscriminator", "SRDiscriminator", "SRGANEvalConfig", "SRGANTrainingConfig"]
