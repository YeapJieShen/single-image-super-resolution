"""Pluggable loss functions, wired via YAML ``model.criterion``."""

from .base import SRLoss
from .composite import WeightedSumLoss
from .pixel import CharbonnierLoss, TotalVariationLoss
from .vgg import VGG16FeatureLoss, VGG19FeatureLoss

__all__ = [
    "SRLoss",
    "CharbonnierLoss",
    "TotalVariationLoss",
    "VGG16FeatureLoss",
    "VGG19FeatureLoss",
    "WeightedSumLoss",
]
