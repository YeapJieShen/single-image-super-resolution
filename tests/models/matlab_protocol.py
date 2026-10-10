"""The authors' MATLAB evaluation protocol (VDSR.m, DRCN.m, compute_diff.m), test-side only.

Shared by the DRCN, VDSR, EDSR and MDSR parity tests (the last two use ``psnr_matlab``, scoring
a float pipeline output the way the paper's MATLAB evaluation would).
DRCN and VDSR evaluate: ``modcrop``;
``rgb2ycbcr`` for a colour image but the **raw plane for a single-channel one** (Set14's
``bridge``: scored through RGB it gains 20*log10(255/219) = 1.3219 dB); bicubic down and up
on ``single`` 0-1 planes with no uint8 in between; the network input's Y clamped to
[16/255, 235/255] for colour; and a PNG round trip before ``compute_diff`` re-reads it
(Y as uint8, ``SF`` pixels shaved, PSNR on doubles). It is a protocol, not a pipeline
feature: nothing here is imported by ``sisr``.
"""

import numpy as np
import torch
from PIL import Image

from sisr.colorspace import rgb_to_ycbcr_studio
from sisr.utils.imresize import _contributions


def rgb2ycbcr(rgb):
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    ycc = np.stack(
        [
            16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255,
            128 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255,
            128 + (112.0 * r - 93.786 * g - 18.214 * b) / 255,
        ],
        -1,
    )
    return np.floor(np.clip(ycc, 0, 255) + 0.5).astype(np.uint8)


def ycbcr2rgb(ycc):
    y, cb, cr = (ycc[..., i].astype(np.float64) for i in range(3))
    y = 1.164384 * (y - 16)
    rgb = np.stack(
        [
            y + 1.596027 * (cr - 128),
            y - 0.391762 * (cb - 128) - 0.812968 * (cr - 128),
            y + 2.017232 * (cb - 128),
        ],
        -1,
    )
    return np.floor(np.clip(rgb, 0, 255) + 0.5).astype(np.uint8)


def resize(plane, shape):
    """MATLAB imresize bicubic on a float plane: no uint8 between passes, as for ``single``."""
    wh, ih = _contributions(plane.shape[0], shape[0], shape[0] / plane.shape[0])
    ww, iw = _contributions(plane.shape[1], shape[1], shape[1] / plane.shape[1])
    plane = np.einsum("ij,ijk->ik", wh, plane[ih])
    return np.einsum("jk,ijk->ij", ww, plane[:, iw])


def prepare(path, sf):
    """(ground-truth plane uint8, network input, Cb, Cr or None) for one image at scale ``sf``."""
    img = Image.open(path)
    grey = img.mode == "L"
    rgb = np.asarray(img if grey else img.convert("RGB"))
    h, w = rgb.shape[0] - rgb.shape[0] % sf, rgb.shape[1] - rgb.shape[1] % sf
    if grey:
        gt = rgb[:h, :w]
        up = resize(resize(gt / 255.0, (h // sf, w // sf)), (h, w))
        return gt, up, None, None
    ycc = rgb2ycbcr(rgb[:h, :w])
    up = [resize(resize(ycc[..., i] / 255.0, (h // sf, w // sf)), (h, w)) for i in range(3)]
    return ycc[..., 0], np.clip(up[0], 16 / 255, 235 / 255), up[1], up[2]


def u8(x):
    return np.floor(np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def psnr(a, b, sf):
    a, b = a[sf:-sf, sf:-sf].astype(np.float64), b[sf:-sf, sf:-sf].astype(np.float64)
    return float(20 * np.log10(255 / np.sqrt(np.mean((a - b) ** 2))))


def score_image(path, sf, predict):
    """PSNR-Y of one image under the authors' protocol, ``VDSR.m`` through ``compute_diff.m``.

    Args:
        path: The HR image file.
        sf: Scale factor.
        predict: Maps the network-input Y plane (float, 0-1) to the network output plane.

    Returns:
        The PSNR the authors' evaluation would print. A colour image goes through
        ``ycbcr2rgb(uint8(...))`` and back through ``rgb2ycbcr`` as their PNG write/read
        does; a single-channel one is written and scored as the raw plane.
    """
    gt, y, cb, cr = prepare(path, sf)
    pred = predict(y)
    if cb is None:
        return psnr(gt, u8(pred), sf)
    png = ycbcr2rgb(u8(np.stack([pred, cb, cr], -1)))
    return psnr(gt, rgb2ycbcr(png)[..., 0], sf)


# --- scoring a float pipeline output (EDSR / MDSR) ---------------------------------------


def psnr_float_y(sr, hr, scale):
    """This pipeline's own metric: studio Y of the float RGB, border = scale."""
    a, b = (rgb_to_ycbcr_studio(t)[:, :1, scale:-scale, scale:-scale] * 255 for t in (sr, hr))
    return float(10 * torch.log10(255.0**2 / ((a - b) ** 2).mean()))


def psnr_matlab(sr, hr, scale):
    """The paper's protocol (sec 4.5: "We used MATLAB functions for evaluation").

    SR is clamped and rounded to 8-bit RGB, MATLAB's ``rgb2ycbcr`` on uint8 returns uint8 Y (so
    both luma planes are rounded), ``scale`` pixels are shaved, and PSNR is taken on doubles.
    """

    def luma(t):
        rgb = (t[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
        return rgb2ycbcr(rgb)[..., 0]

    return psnr(luma(hr), luma(sr), scale)


def psnr_matlab_eval(sr, hr, scale, grey):
    """NTIRE2017 ``demo/evaluation.m``: a single-channel target is scored on its raw plane.

    Colour targets go through ``psnr_matlab``. For a grey one (Set14's ``bridge``) the raw 8-bit
    plane is compared with channel 1 of the 8-bit output, with no ``rgb2ycbcr`` on either side,
    so the 1.3219 dB studio-range gain of the RGB route does not apply.
    """
    if not grey:
        return psnr_matlab(sr, hr, scale)

    def plane(t):
        return (t[0, 0].clamp(0, 1) * 255).round().byte().cpu().numpy()

    return psnr(plane(hr), plane(sr), scale)
