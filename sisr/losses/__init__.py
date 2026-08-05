"""Pluggable loss functions, wired via YAML ``model.criterion``."""

from .base import SRLoss
from .pixel import CharbonnierLoss, TotalVariationLoss

__all__ = ["SRLoss", "CharbonnierLoss", "TotalVariationLoss"]
