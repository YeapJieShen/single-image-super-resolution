"""Full YCbCr processor: convert LR to YCbCr in, convert SR back to RGB out."""
import torch

from sisr.utils import rgb_to_ycbcr, ycbcr_to_rgb

from .base import SRProcessor


class YCbCrProcessor(SRProcessor):
    """Full YCbCr processor: model trains on YCbCr, output converted to RGB."""

    def extract(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        return rgb_to_ycbcr(lr_rgb)

    def reconstruct(
        self, sr_ycbcr: torch.Tensor, lr_rgb: torch.Tensor
    ) -> torch.Tensor:
        return ycbcr_to_rgb(sr_ycbcr)
