"""Y-channel processor: extract Y from LR, stitch SR-Y with bicubic LR Cb/Cr."""

import torch
import torch.nn.functional as F

from sisr.utils import rgb_to_ycbcr, ycbcr_to_rgb

from .base import SRProcessor


class YChannelProcessor(SRProcessor):
    """Y-channel processor.

    ``extract`` returns the Y channel of the LR (converted to YCbCr).
    ``reconstruct`` stitches the SR-Y output with bicubic-upsampled LR
    Cb/Cr (target size taken from ``sr_y.shape[-2:]``, so the same code
    works whether the model is same-size (SRCNN) or upscaling (SRResNet)).
    """

    def extract(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        return rgb_to_ycbcr(lr_rgb)[:, 0:1]

    def reconstruct(self, sr_y: torch.Tensor, lr_rgb: torch.Tensor) -> torch.Tensor:
        lr_ycbcr = rgb_to_ycbcr(lr_rgb)
        cbcr = F.interpolate(
            lr_ycbcr[:, 1:],
            size=sr_y.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        return ycbcr_to_rgb(torch.cat([sr_y, cbcr], dim=1))
