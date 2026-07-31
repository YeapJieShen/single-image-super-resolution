"""MATLAB-compatible bicubic image resizing (INIT.10 / INIT.11).

Vendored -- not a dependency -- from ``fatheral/matlab_imresize``
(https://github.com/fatheral/matlab_imresize/blob/master/imresize.py), MIT
License, Copyright (c) 2020 Alex; full license text below. That file is
itself a line-for-line Python port of MATLAB's own
``toolbox/images/images/imresize.m``. The BasicSR / SwinIR / Real-ESRGAN
lineage carries the same algorithm (Apache-2.0 / BSD-3-Clause respectively,
and in BasicSR's case a ``torch.FloatTensor`` i.e. float32 port that rounds
only once at the very end); this module follows the original MIT source
instead, since it already computes in double and rounds every pass -- see
below for why both of those turned out load-bearing, not incidental.

Two corrections were needed on top of a literal port, both because the
upstream file does not itself achieve MATLAB parity here:

* **Round half away from zero** on every uint8 cast. MATLAB's ``round``
  ties away from zero; NumPy's ``np.round``/``np.around`` ties to even
  (banker's rounding) -- the upstream file uses ``np.around`` and inherits
  this off-by-one, the classic reason a "faithful" port fails byte-equality.
* **Re-quantize to uint8 after *each* 1-D pass, not only at the end.**
  MATLAB's ``imresizemex`` re-quantizes to the image's class between the
  row-pass and the column-pass. Skipping that (as e.g. BasicSR's float32
  port does, rounding only once at the very end) is a second, independent
  source of drift from real MATLAB -- confirmed against the unmodified
  upstream file, which does the same double-rounding (see
  ``tests/test_imresize.py``).

All arithmetic runs in float64 throughout (MATLAB computes ``imresize`` in
double; a float32 port drifts by fractions of a level -- invisible until a
uint8 cast turns it into scattered +/-1 errors).

The two real deltas this closes vs. ``cv2.INTER_CUBIC``: MATLAB widens the
kernel by ``1/scale`` when downscaling (antialiasing -- cv2 does not
antialias at all, the dominant term) and the bicubic coefficient differs
(MATLAB a=-0.5, OpenCV a=-0.75).

------------------------------------------------------------------------------
MIT License

Copyright (c) 2020 Alex

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------
"""

from math import ceil
from typing import Literal

import cv2
import numpy as np

_KERNEL_WIDTH = 4.0  # bicubic support width; MATLAB's a=-0.5 coefficient (OpenCV uses a=-0.75)

ResizeBackend = Literal["matlab", "cv2"]


def _cubic(x: np.ndarray) -> np.ndarray:
    """MATLAB's bicubic convolution kernel (Keys' family, a=-0.5)."""
    absx = np.abs(x)
    absx2 = absx * absx
    absx3 = absx2 * absx
    return np.where(
        absx <= 1.0,
        1.5 * absx3 - 2.5 * absx2 + 1.0,
        np.where(absx <= 2.0, -0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0, 0.0),
    )


def _round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    """MATLAB's ``round`` (ties away from zero) -- NumPy's ties to even."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _contributions(in_length: int, out_length: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-pixel interpolation weights and source indices for one axis.

    Downscaling (``scale < 1``) widens the kernel by ``1/scale`` and
    pre-multiplies it by ``scale`` -- this widening *is* MATLAB's
    antialiasing low-pass, not a separate blur step. Out-of-range source
    indices are folded back via symmetric (mirror) padding, matching
    MATLAB's boundary handling.

    Args:
        in_length: Input size along this axis.
        out_length: Output size along this axis.
        scale: ``out_length / in_length``.

    Returns:
        A ``(weights, indices)`` pair, each ``(out_length, num_taps)``:
        row *i* lists the source-pixel weights/indices contributing to
        output pixel *i*.
    """
    if scale < 1.0:
        kernel_width = _KERNEL_WIDTH / scale

        def kernel(x: np.ndarray) -> np.ndarray:
            return scale * _cubic(scale * x)
    else:
        kernel_width = _KERNEL_WIDTH
        kernel = _cubic

    x = np.arange(1, out_length + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)
    left = np.floor(u - kernel_width / 2.0)
    num_taps = int(ceil(kernel_width)) + 2

    indices = left[:, None] + np.arange(num_taps)[None, :] - 1.0
    weights = kernel(u[:, None] - indices - 1.0)
    weights = weights / weights.sum(axis=1, keepdims=True)

    mirror = np.concatenate([np.arange(in_length), np.arange(in_length - 1, -1, -1)])
    indices = mirror[np.mod(indices.astype(np.int64), mirror.size)]

    keep = np.any(weights != 0.0, axis=0)
    return weights[:, keep], indices[:, keep]


def _resize_axis(
    image: np.ndarray, axis: int, weights: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """One 1-D weighted-interpolation pass along *axis*, re-quantized to uint8.

    Args:
        image: uint8 array, ``(H, W, C)``.
        axis: ``0`` to resize height, ``1`` to resize width.
        weights: ``(out_length, num_taps)`` from :func:`_contributions`.
        indices: ``(out_length, num_taps)`` from :func:`_contributions`.

    Returns:
        uint8 array with *axis* resized to ``out_length``.
    """
    if axis == 0:
        gathered = image[indices].astype(np.float64)  # (out_h, num_taps, W, C)
        out = np.sum(weights[:, :, None, None] * gathered, axis=1)
    else:
        gathered = image[:, indices].astype(np.float64)  # (H, out_w, num_taps, C)
        out = np.sum(weights[None, :, :, None] * gathered, axis=2)
    return _round_half_away_from_zero(np.clip(out, 0.0, 255.0)).astype(np.uint8)


def matlab_imresize(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """MATLAB-``imresize``-compatible bicubic resize of a uint8 image.

    Args:
        image: uint8 array, ``(H, W)`` or ``(H, W, C)``.
        output_shape: Target ``(height, width)``.

    Returns:
        uint8 array of shape ``output_shape`` (with trailing ``C`` if
        *image* had one).
    """
    in_h, in_w = image.shape[0], image.shape[1]
    out_h, out_w = output_shape
    scale_h = out_h / in_h
    scale_w = out_w / in_w

    flag2d = image.ndim == 2
    arr = image[:, :, None] if flag2d else image

    weights_h, indices_h = _contributions(in_h, out_h, scale_h)
    weights_w, indices_w = _contributions(in_w, out_w, scale_w)

    # MATLAB's imresize.m resizes the smaller-scale-factor axis first. Because
    # of the intermediate uint8 rounding in _resize_axis this is not a mere
    # performance choice -- swapping the order changes the exact output -- so
    # it must mirror the reference exactly, not just be "mathematically free".
    passes = [(scale_h, 0, weights_h, indices_h), (scale_w, 1, weights_w, indices_w)]
    passes.sort(key=lambda p: p[0])
    for _, axis, weights, indices in passes:
        arr = _resize_axis(arr, axis, weights, indices)

    return arr[:, :, 0] if flag2d else arr


def resize(
    image: np.ndarray, size: tuple[int, int], backend: ResizeBackend = "matlab"
) -> np.ndarray:
    """Resizes a uint8 image to ``(height, width) = size``.

    Args:
        image: uint8 array, ``(H, W)`` or ``(H, W, C)``.
        size: Target ``(height, width)``.
        backend: ``'matlab'`` (default) -- antialiased bicubic, a=-0.5,
            matching MATLAB's ``imresize`` so benchmark numbers are
            comparable to published papers. ``'cv2'`` -- ``cv2.INTER_CUBIC``
            (a=-0.75, no antialiasing); kept only so LMDB caches built
            before this module existed stay reproducible.

    Returns:
        Resized uint8 array.

    Raises:
        ValueError: If *backend* is not ``'matlab'`` or ``'cv2'``.
    """
    if backend == "matlab":
        return matlab_imresize(image, size)
    if backend == "cv2":
        height, width = size
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    raise ValueError(f"Unknown resize backend {backend!r}; expected 'matlab' or 'cv2'.")
