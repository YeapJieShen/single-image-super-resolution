"""Studio-range Y processor: the model sees MATLAB's ``rgb2ycbcr`` luma (16-235)."""

from typing import Literal

import torch

from sisr.colorspace import rgb_to_y_studio, studio_y_to_full_y

from .base import SRProcessor
from .y_channel import stitch_y


class StudioYChannelProcessor(SRProcessor):
    """Y-channel processor over studio-range luma, for nets trained on MATLAB's Y (VDSR, DRCN).

    A separate class from :class:`~sisr.processors.YChannelProcessor`, not a flag: full-range Y
    is SRCNN's frozen training space. ``extract`` (and the default ``extract_target``) give the
    studio Y; ``reconstruct`` maps the output back to full-range Y and stitches bicubic LR
    Cb/Cr, so the RGB's studio luma is exactly the model's output (metrics read that).
    """

    def extract(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        """Return the studio-range Y of ``lr_rgb``, ``(B, 1, H, W)``."""
        return rgb_to_y_studio(lr_rgb)

    def reconstruct(self, sr_y: torch.Tensor, lr_rgb: torch.Tensor) -> torch.Tensor:
        """Studio ``sr_y`` -> full-range Y, stitched with bicubic-upsampled LR Cb/Cr, as RGB."""
        return stitch_y(studio_y_to_full_y(sr_y), lr_rgb)

    @property
    def model_channels(self) -> int:
        """Number of model IO channels -- 1 (Y)."""
        return 1

    @property
    def output_range(self) -> tuple[float, float]:
        """Model output range -- studio luma, ``(16/255, 235/255)``."""
        return (16 / 255, 235 / 255)

    @property
    def output_colorspace(self) -> Literal["RGB", "YCbCr", "Y"]:
        """Model output colorspace -- Y."""
        return "Y"
