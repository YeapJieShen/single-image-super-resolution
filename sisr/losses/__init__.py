"""Pluggable loss functions, wired via YAML ``model.criterion``."""

from .adversarial import AdversarialLoss
from .base import SRLoss
from .composite import WeightedSumLoss
from .pixel import CharbonnierLoss, TotalVariationLoss
from .vgg import VGG16FeatureLoss, VGG19FeatureLoss

__all__ = [
    "SRLoss",
    "AdversarialLoss",
    "CharbonnierLoss",
    "TotalVariationLoss",
    "VGG16FeatureLoss",
    "VGG19FeatureLoss",
    "WeightedSumLoss",
]
