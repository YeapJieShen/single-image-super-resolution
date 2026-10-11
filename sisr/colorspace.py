"""BT.601 RGB <-> YCbCr conversion, full-range and studio-range.

**Two ranges, never one.** Unifying them is the mistake this split guards against.

* **Full-range** (:func:`rgb_to_ycbcr` / :func:`ycbcr_to_rgb`) is the *training*
  space every :class:`~sisr.processors.SRProcessor` feeds the model. Frozen:
  changing it renumbers a trained model's inputs and invalidates checkpoints.
* **Studio-range** (:func:`rgb_to_ycbcr_studio`) is the *metric* space MATLAB's
  ``rgb2ycbcr``, BasicSR and every published SR benchmark score against. The three-channel
  conversion is scoring only. A *separate* processor,
  :class:`~sisr.processors.StudioYChannelProcessor`, feeds a MATLAB-trained net (VDSR, DRCN)
  the studio **Y plane** via :func:`rgb_to_y_studio` / :func:`studio_y_to_full_y`; the frozen
  full-range processors are unchanged.

Pure tensor functions, so callers convert without importing Lightning or models.
"""

import torch

# BT.601-7 full-range ("JPEG") variant. Both spaces are [0, 1]; Cb/Cr carry a
# +0.5 offset so signed chroma in [-0.5, +0.5] stores as unsigned [0, 1].

_RGB_TO_Y = (0.299, 0.587, 0.114)
# MATLAB's published rgb2ycbcr matrix over the 224-level chroma range. Left as
# literal ratios, not decimals: the source constants stay visible, so a
# transcription slip cannot hide.
_RGB_TO_CB = (-37.797 / 224, -74.203 / 224, 112 / 224)  # output offset +0.5
_RGB_TO_CR = (112 / 224, -93.786 / 224, -18.214 / 224)  # output offset +0.5
_YCBCR_TO_R = 1.402  # cr coefficient
_YCBCR_TO_G_CB = -0.344136  # cb coefficient
_YCBCR_TO_G_CR = -0.714136  # cr coefficient
_YCBCR_TO_B = 1.772  # cb coefficient

# Studio ("limited"/"TV") rescale over the full-range values above. Luma spans
# 219 of 255 levels, chroma 224 -- two different scales, so collapsing them to
# one constant is itself a fidelity bug. The consequence: studio-range PSNR sits
# a constant 20*log10(255/219) dB above full-range for Y, 20*log10(255/224) for
# Cb/Cr. tests/test_colorspace.py asserts that identity.
_STUDIO_Y_SCALE = 219 / 255
_STUDIO_Y_OFFSET = 16 / 255
_STUDIO_C_SCALE = 224 / 255
_STUDIO_C_OFFSET = 128 / 255


def rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    """Convert a normalised RGB tensor to YCbCr (BT.601 full-range).

    Args:
        img: RGB ``(B, 3, H, W)`` in ``[0, 1]``.

    Returns:
        YCbCr ``(B, 3, H, W)`` in ``[0, 1]``, Cb/Cr offset by +0.5.
    """
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = _RGB_TO_Y[0] * r + _RGB_TO_Y[1] * g + _RGB_TO_Y[2] * b
    cb = _RGB_TO_CB[0] * r + _RGB_TO_CB[1] * g + _RGB_TO_CB[2] * b + 0.5
    cr = _RGB_TO_CR[0] * r + _RGB_TO_CR[1] * g + _RGB_TO_CR[2] * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def ycbcr_to_rgb(img: torch.Tensor) -> torch.Tensor:
    """Convert a YCbCr tensor to RGB (BT.601 full-range, clamped to [0, 1]).

    Args:
        img: YCbCr ``(B, 3, H, W)``, Cb/Cr offset by +0.5.

    Returns:
        RGB ``(B, 3, H, W)`` clamped to ``[0, 1]``.
    """
    y, cb, cr = img[:, 0:1], img[:, 1:2] - 0.5, img[:, 2:3] - 0.5
    r = y + _YCBCR_TO_R * cr
    g = y + _YCBCR_TO_G_CB * cb + _YCBCR_TO_G_CR * cr
    b = y + _YCBCR_TO_B * cb
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)


def rgb_to_ycbcr_studio(img: torch.Tensor) -> torch.Tensor:
    """Convert a normalised RGB tensor to YCbCr (BT.601 studio-range) — metric only.

    Rescales :func:`rgb_to_ycbcr`'s output onto the range MATLAB's
    ``rgb2ycbcr`` and the SR literature report PSNR/SSIM against. One-way:
    nothing reconstructs an image from it, so there is no studio inverse.
    **Never feed this to a model** -- every full-range processor trains full-range; the
    one model-facing use is the Y plane via :func:`rgb_to_y_studio`.

    Args:
        img: RGB ``(B, 3, H, W)`` in ``[0, 1]``.

    Returns:
        YCbCr ``(B, 3, H, W)``: Y in ``[16/255, 235/255]``, Cb/Cr in
        ``[16/255, 240/255]``.
    """
    full = rgb_to_ycbcr(img)
    y, cb, cr = full[:, 0:1], full[:, 1:2], full[:, 2:3]
    y = _STUDIO_Y_OFFSET + _STUDIO_Y_SCALE * y
    cb = _STUDIO_C_OFFSET + _STUDIO_C_SCALE * (cb - 0.5)
    cr = _STUDIO_C_OFFSET + _STUDIO_C_SCALE * (cr - 0.5)
    return torch.cat([y, cb, cr], dim=1)


def rgb_to_y_studio(img: torch.Tensor) -> torch.Tensor:
    """Studio-range luma plane (MATLAB ``rgb2ycbcr`` Y) of a normalised RGB tensor.

    Args:
        img: RGB ``(B, 3, H, W)`` in ``[0, 1]``.

    Returns:
        ``(B, 1, H, W)``, black 16/255 and white 235/255.
    """
    return _STUDIO_Y_OFFSET + _STUDIO_Y_SCALE * rgb_to_ycbcr(img)[:, 0:1]


def studio_y_to_full_y(y: torch.Tensor) -> torch.Tensor:
    """Exact inverse of the luma rescale: studio Y -> the full-range Y :func:`ycbcr_to_rgb` takes.

    Args:
        y: Studio-range luma, any shape.

    Returns:
        Full-range luma, same shape (outside ``[0, 1]`` if ``y`` left 16/255-235/255).
    """
    return (y - _STUDIO_Y_OFFSET) / _STUDIO_Y_SCALE
