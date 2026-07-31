"""No-op pass-through processor — model trains and emits RGB directly."""

import torch

from .base import SRProcessor


class RGBProcessor(SRProcessor):
    """No-op processor: model trains and emits RGB directly."""

    def extract(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        return lr_rgb

    def reconstruct(self, sr_model_out: torch.Tensor, lr_rgb: torch.Tensor) -> torch.Tensor:
        return sr_model_out

    @property
    def model_channels(self) -> int:
        return 3
