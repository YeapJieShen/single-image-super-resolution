"""BT.601 RGB <-> YCbCr conversion, full-range and studio-range.

Two distinct ranges live here, deliberately kept apart:

* **Full-range** (:func:`rgb_to_ycbcr` / :func:`ycbcr_to_rgb`) is the
  project's *training* chroma space — what :class:`~sisr.processors.SRProcessor`
  subclasses (``YChannelProcessor``, ``YCbCrProcessor``) feed the model.
  Changing these would silently renumber a trained model's inputs and
  invalidate existing checkpoints, so they are frozen.
* **Studio-range** (:func:`rgb_to_ycbcr_studio`) is the *metric* colorspace —
  what MATLAB's ``rgb2ycbcr``, BasicSR's ``bgr2ycbcr(y_only=...)``, and every
  published SR benchmark actually score against. It exists solely for
  :class:`~sisr.training.lightning_module.SRLightning`'s PSNR/SSIM
  computation and must never be used as a processor's colorspace.

Do not "unify" these two — that is the one mistake this split guards against.
These pure tensor functions live on their own so callers can convert
colorspaces without depending on Lightning or model code.
"""

import torch

# BT.601 full-range RGB <-> YCbCr conversion (ITU-R Rec. BT.601-7).
# Both colorspaces are normalised to [0, 1]. Cb and Cr have a +0.5 offset on
# their stored representation so that signed chroma values in [-0.5, +0.5]
# map onto unsigned [0, 1]. This is the *full-range* (a.k.a. "JPEG") variant
# of BT.601 — distinct from the studio-range variant below, which scales Y to
# [16/255, 235/255] and Cb/Cr to [16/255, 240/255].

_RGB_TO_Y = (0.299, 0.587, 0.114)
_RGB_TO_CB = (-0.169, -0.331, 0.500)  # output offset +0.5
_RGB_TO_CR = (0.500, -0.419, -0.081)  # output offset +0.5
_YCBCR_TO_R = 1.402  # cr coefficient
_YCBCR_TO_G_CB = -0.344  # cb coefficient
_YCBCR_TO_G_CR = -0.714  # cr coefficient
_YCBCR_TO_B = 1.772  # cb coefficient

# BT.601 studio-range ("limited"/"TV" range) rescale, applied on top of the
# full-range YCbCr above. Luma occupies 219 of 255 levels ([16, 235]); chroma
# occupies 224 of 255 levels ([16, 240], centered on 128) — the two scales
# differ, so collapsing them to one constant would itself be a fidelity bug.
# A full-range diff scales by exactly one of these two factors on conversion,
# so PSNR computed in studio range sits a *constant*, algebraically exact
# 20*log10(255/219) dB above full-range for Y (and 20*log10(255/224) dB for
# Cb/Cr) — see tests/test_colorspace.py for the identity this predicts.
_STUDIO_Y_SCALE = 219 / 255
_STUDIO_Y_OFFSET = 16 / 255
_STUDIO_C_SCALE = 224 / 255
_STUDIO_C_OFFSET = 128 / 255


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


def rgb_to_ycbcr_studio(img: torch.Tensor) -> torch.Tensor:
    """Convert a normalised RGB tensor to YCbCr (BT.601 studio-range) — metric only.

    Rescales :func:`rgb_to_ycbcr`'s full-range output onto the studio
    ("limited"/"TV") range MATLAB's ``rgb2ycbcr`` and the SR literature
    report PSNR/SSIM against. This is a one-way conversion for scoring —
    there is no ``ycbcr_to_rgb_studio``, since nothing reconstructs an
    image from it. Do not feed this to a model; every :class:`SRProcessor`
    trains in full-range (see module docstring).

    Args:
        img (torch.Tensor): RGB tensor of shape ``(B, 3, H, W)`` with
            values in ``[0, 1]``.

    Returns:
        torch.Tensor: YCbCr tensor of shape ``(B, 3, H, W)``, Y in
        ``[16/255, 235/255]``, Cb/Cr in ``[16/255, 240/255]``.
    """
    full = rgb_to_ycbcr(img)
    y, cb, cr = full[:, 0:1], full[:, 1:2], full[:, 2:3]
    y = _STUDIO_Y_OFFSET + _STUDIO_Y_SCALE * y
    cb = _STUDIO_C_OFFSET + _STUDIO_C_SCALE * (cb - 0.5)
    cr = _STUDIO_C_OFFSET + _STUDIO_C_SCALE * (cr - 0.5)
    return torch.cat([y, cb, cr], dim=1)
