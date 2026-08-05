"""Pluggable loss functions, wired via YAML ``model.criterion``."""

from .base import SRLoss

__all__ = ["SRLoss"]
