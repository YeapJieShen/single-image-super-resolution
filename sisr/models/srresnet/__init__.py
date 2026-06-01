"""SRResNet architecture (the residual generator from Ledig et al., 2017)."""
from .model import SRResidualBlock, SRResNet, SRUpsampleBlock

__all__ = ["SRResNet", "SRResidualBlock", "SRUpsampleBlock"]
