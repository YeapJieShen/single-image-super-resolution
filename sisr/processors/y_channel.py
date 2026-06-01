"""Y-channel processor: extract Y from LR, stitch SR-Y with bicubic LR Cb/Cr."""
import torch
import torch.nn.functional as F

from sisr.utils import rgb_to_ycbcr, ycbcr_to_rgb

from .base import SRProcessor

# BT.601 full-range luma/chroma coefficients (mirrors sisr.utils).
_CB_COEFF = 1.772   # B = Y + _CB_COEFF * (Cb - 0.5)
_CR_COEFF = 1.402   # R = Y + _CR_COEFF * (Cr - 0.5)
_G_CB = -0.344      # G = Y + _G_CB*(Cb-0.5) + _G_CR*(Cr-0.5)
_G_CR = -0.714


class YChannelProcessor(SRProcessor):
    """Y-channel processor.

    ``extract`` returns the Y channel of the LR (converted to YCbCr).
    ``reconstruct`` stitches the SR-Y output with bicubic-upsampled LR
    Cb/Cr (target size taken from ``sr_y.shape[-2:]``, so the same code
    works whether the model is same-size (SRCNN) or upscaling (SRResNet)).

    The bicubic Cb/Cr are gamut-scaled toward neutral (0.5, 0.5) to
    ensure the composed YCbCr produces valid RGB without clamping,
    preserving the SR-Y channel through the roundtrip.
    """

    def extract(self, lr_rgb: torch.Tensor) -> torch.Tensor:
        return rgb_to_ycbcr(lr_rgb)[:, 0:1]

    def reconstruct(
        self, sr_y: torch.Tensor, lr_rgb: torch.Tensor
    ) -> torch.Tensor:
        lr_ycbcr = rgb_to_ycbcr(lr_rgb)
        cbcr = F.interpolate(
            lr_ycbcr[:, 1:],
            size=sr_y.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        # Chroma deviations from neutral gray.
        cb_dev = cbcr[:, 0:1] - 0.5
        cr_dev = cbcr[:, 1:2] - 0.5
        y = sr_y

        # Scale the chroma vector (cb_dev, cr_dev) toward (0, 0) — i.e. toward
        # neutral gray — until R, G, B all land in [0, 1].  This is equivalent
        # to a ray-cast from the bicubic point to the neutral point and finding
        # the first intersection with the gamut boundary.  The valid interval for
        # each channel gives an upper bound on the scale factor t ∈ [0, 1]:
        #   R = y + _CR_COEFF * t * cr_dev  ∈ [0, 1]
        #   B = y + _CB_COEFF * t * cb_dev  ∈ [0, 1]
        #   G = y + (_G_CB*cb_dev + _G_CR*cr_dev) * t  ∈ [0, 1]
        def _max_t(signal: torch.Tensor) -> torch.Tensor:
            """Largest t ∈ [0,1] such that y + signal*t ∈ [0, 1]."""
            # t <= (1 - y) / signal  when signal > 0
            # t <= -y / signal        when signal < 0
            # t unconstrained         when signal == 0
            eps = 1e-8
            hi = torch.where(signal > eps, (1.0 - y) / signal.clamp(min=eps),
                 torch.where(signal < -eps, -y / signal.clamp(max=-eps),
                             torch.ones_like(signal)))
            return hi.clamp(0.0, 1.0)

        t = torch.min(
            torch.min(_max_t(_CR_COEFF * cr_dev), _max_t(_CB_COEFF * cb_dev)),
            _max_t(_G_CB * cb_dev + _G_CR * cr_dev),
        )

        cb = t * cb_dev + 0.5
        cr = t * cr_dev + 0.5
        return ycbcr_to_rgb(torch.cat([sr_y, cb, cr], dim=1))
