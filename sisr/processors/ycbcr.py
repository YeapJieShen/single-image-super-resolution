"""Full YCbCr processor: convert LR to YCbCr in, convert SR back to RGB out."""

from typing import Literal

import torch

from sisr.colorspace import rgb_to_ycbcr, ycbcr_to_rgb

from .base import SRProcessor


class YCbCrProcessor(SRProcessor):
    """Full YCbCr processor: model trains on YCbCr, output converted to RGB."""

    def extract(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        """Convert lr_rgb to YCbCr."""
        return rgb_to_ycbcr(lr_rgb)

    def reconstruct(self, sr_ycbcr: torch.Tensor, lr_rgb: torch.Tensor) -> torch.Tensor:
        """Convert the model's YCbCr output back to RGB."""
        return ycbcr_to_rgb(sr_ycbcr)

    @property
    def model_channels(self) -> int:
        """Number of model IO channels — 3 (YCbCr)."""
        return 3

    @property
    def output_range(self) -> tuple[float, float]:
        """Model output range — ``(0.0, 1.0)``, unscaled YCbCr."""
        return (0.0, 1.0)

    @property
    def output_colorspace(self) -> Literal["RGB", "YCbCr", "Y"]:
        """Model output colorspace — YCbCr."""
        return "YCbCr"
