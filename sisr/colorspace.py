"""BT.601 full-range RGB <-> YCbCr conversion.

BT.601 full-range YCbCr is the project's working chroma space. These pure
tensor functions live on their own so the
:class:`~sisr.processors.SRProcessor` subclasses and any external callers can
convert colorspaces without depending on Lightning or model code.
"""

import torch

# BT.601 full-range RGB <-> YCbCr conversion (ITU-R Rec. BT.601-7).
# Both colorspaces are normalised to [0, 1]. Cb and Cr have a +0.5 offset on
# their stored representation so that signed chroma values in [-0.5, +0.5]
# map onto unsigned [0, 1]. This is the *full-range* (a.k.a. "JPEG") variant
# of BT.601 — distinct from the studio-range variant which scales Y to
# [16/255, 235/255] and Cb/Cr to [16/255, 240/255].

_RGB_TO_Y = (0.299, 0.587, 0.114)
_RGB_TO_CB = (-0.169, -0.331, 0.500)  # output offset +0.5
_RGB_TO_CR = (0.500, -0.419, -0.081)  # output offset +0.5
_YCBCR_TO_R = 1.402  # cr coefficient
_YCBCR_TO_G_CB = -0.344  # cb coefficient
_YCBCR_TO_G_CR = -0.714  # cr coefficient
_YCBCR_TO_B = 1.772  # cb coefficient


def rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    """Convert a normalised RGB tensor to YCbCr (BT.601 full-range).

    Args:
        img (torch.Tensor): RGB tensor of shape ``(B, 3, H, W)`` with
            values in ``[0, 1]``.

    Returns:
        torch.Tensor: YCbCr tensor of shape ``(B, 3, H, W)`` in
        ``[0, 1]`` with Cb/Cr offset by +0.5.
    """
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = _RGB_TO_Y[0] * r + _RGB_TO_Y[1] * g + _RGB_TO_Y[2] * b
    cb = _RGB_TO_CB[0] * r + _RGB_TO_CB[1] * g + _RGB_TO_CB[2] * b + 0.5
    cr = _RGB_TO_CR[0] * r + _RGB_TO_CR[1] * g + _RGB_TO_CR[2] * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def ycbcr_to_rgb(img: torch.Tensor) -> torch.Tensor:
    """Convert a YCbCr tensor to RGB (BT.601 full-range, clamped to [0, 1]).

    Args:
        img (torch.Tensor): YCbCr tensor of shape ``(B, 3, H, W)`` with
            Cb/Cr offset by +0.5.

    Returns:
        torch.Tensor: RGB tensor of shape ``(B, 3, H, W)`` clamped to
        ``[0, 1]``.
    """
    y, cb, cr = img[:, 0:1], img[:, 1:2] - 0.5, img[:, 2:3] - 0.5
    r = y + _YCBCR_TO_R * cr
    g = y + _YCBCR_TO_G_CB * cb + _YCBCR_TO_G_CR * cr
    b = y + _YCBCR_TO_B * cb
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)
