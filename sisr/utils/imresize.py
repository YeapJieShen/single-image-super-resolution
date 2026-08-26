"""MATLAB-compatible bicubic image resizing.

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
  ``tests/utils/test_imresize.py``).

All arithmetic runs in float64 throughout (MATLAB computes ``imresize`` in
double; a float32 port drifts by fractions of a level -- invisible until a
uint8 cast turns it into scattered +/-1 errors).

The two real deltas this closes vs. a naive OpenCV-style bicubic resize:
MATLAB widens the kernel by ``1/scale`` when downscaling (antialiasing --
OpenCV does not antialias at all, the dominant term) and the bicubic
coefficient differs (MATLAB a=-0.5, OpenCV a=-0.75).

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

import functools
from math import ceil

import numpy as np

_KERNEL_WIDTH = 4.0  # bicubic support width; MATLAB's a=-0.5 coefficient (OpenCV uses a=-0.75)

# Cap on one gather's float64 footprint in _resize_axis -- see its docstring. Sized
# well above any training crop (a 96x96 crop at scale 4 gathers 0.84 MiB, so the
# training path never splits) and well below a full-image validation gather (286.88
# MiB for a 1536x2040 DIV2K image at the same scale).
_GATHER_CHUNK_BYTES = 64 << 20


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


@functools.lru_cache(maxsize=32)
def _contributions(in_length: int, out_length: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-pixel interpolation weights and source indices for one axis.

    Downscaling (``scale < 1``) widens the kernel by ``1/scale`` and
    pre-multiplies it by ``scale`` -- this widening *is* MATLAB's
    antialiasing low-pass, not a separate blur step. Out-of-range source
    indices are folded back via symmetric (mirror) padding, matching
    MATLAB's boundary handling.

    A pure function of its arguments and constant for any one configured
    dataset (fixed crop/scale), so it is memoized -- measured 2.21x/1.41x
    (SRCNN/SRResNet) per-sample speedups with byte-identical output. The
    returned arrays are marked read-only (``setflags(write=False)``) since
    they are shared, not copied, across every call with the same arguments;
    a caller mutating them would corrupt every other cache consumer.

    Args:
        in_length: Input size along this axis.
        out_length: Output size along this axis.
        scale: ``out_length / in_length``.

    Returns:
        A ``(weights, indices)`` pair, each ``(out_length, num_taps)``,
        both read-only: row *i* lists the source-pixel weights/indices
        contributing to output pixel *i*.
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
    weights, indices = weights[:, keep], indices[:, keep]
    weights.setflags(write=False)
    indices.setflags(write=False)
    return weights, indices


def _resize_axis(
    image: np.ndarray, axis: int, weights: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """One 1-D weighted-interpolation pass along *axis*, re-quantized to uint8.

    The weights/gathered-taps contraction runs through a single
    ``np.einsum`` call rather than broadcasting ``weights * gathered`` into a
    second full-size float64 temporary before ``np.sum`` -- on a ~1356x2040x3
    DIV2K image that temporary was >530 MB transient per call; einsum fuses
    the multiply and reduction, measured 1.77x per axis pass with
    bit-identical output.

    ``gathered`` itself is then built in slices of the *free* axis (the one not
    being resized) rather than whole. Whole, it is
    ``(out_length, num_taps, free_length, C)`` float64 -- 286.88 MiB for a
    1536x2040 DIV2K image at scale 4, per DataLoader worker per image, which is
    what exhausted host RAM at the first full-image validation of a run using the
    shipped worker counts. The contraction runs over the tap axis alone, so every
    output element's summation is unchanged by where the free axis is cut: the
    result is byte-identical, not merely close, and is asserted that way in
    ``tests/utils/test_imresize.py``.

    Args:
        image: uint8 array, ``(H, W, C)``.
        axis: ``0`` to resize height, ``1`` to resize width.
        weights: ``(out_length, num_taps)`` from :func:`_contributions`.
        indices: ``(out_length, num_taps)`` from :func:`_contributions`.

    Returns:
        uint8 array with *axis* resized to ``out_length``.
    """
    out_length, num_taps = weights.shape
    channels = image.shape[2]
    free_length = image.shape[1 if axis == 0 else 0]
    step = max(1, _GATHER_CHUNK_BYTES // (out_length * num_taps * channels * 8))

    def contract(start: int, stop: int) -> np.ndarray:
        if axis == 0:
            gathered = image[indices, start:stop].astype(np.float64)  # (out_h, taps, w, C)
            return np.einsum("ij,ijkl->ikl", weights, gathered, optimize=True)
        gathered = image[start:stop][:, indices].astype(np.float64)  # (h, out_w, taps, C)
        return np.einsum("jk,ijkl->ijl", weights, gathered, optimize=True)

    if step >= free_length:
        # One slice covers the image, so skip the output buffer and the copy into
        # it -- neither is free. Every training crop lands here (0.84 MiB gathered
        # at 96x96, scale 4), and this branch is what keeps the cap off the hot
        # path: without it the crop case measured +10.6% for a split it never does.
        out = contract(0, free_length)
    else:
        if axis == 0:
            shape = (out_length, free_length, channels)
        else:
            shape = (free_length, out_length, channels)
        out = np.empty(shape, dtype=np.float64)
        for start in range(0, free_length, step):
            stop = min(start + step, free_length)
            if axis == 0:
                out[:, start:stop] = contract(start, stop)
            else:
                out[start:stop] = contract(start, stop)

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

    # MATLAB's imresize.m resizes the smaller-scale-factor axis first. Mirrored
    # here because the intermediate uint8 rounding in _resize_axis makes pass
    # order observable in principle. Note it is currently unexercised: every
    # call site passes one integer `scale`, so scale_h == scale_w and this sort
    # is a no-op (reversing it reproduces the reference byte-for-byte on all
    # Set5/Set14 pairs). It would start to matter under an anisotropic resize.
    passes = [(scale_h, 0, weights_h, indices_h), (scale_w, 1, weights_w, indices_w)]
    passes.sort(key=lambda p: p[0])
    for _, axis, weights, indices in passes:
        arr = _resize_axis(arr, axis, weights, indices)

    return arr[:, :, 0] if flag2d else arr


def resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resizes a uint8 image to ``(height, width) = size``.

    MATLAB-``imresize``-compatible antialiased bicubic (a=-0.5), so benchmark
    numbers stay comparable to published papers. Thin wrapper over
    :func:`matlab_imresize`, kept as the stable import site for callers.

    Args:
        image: uint8 array, ``(H, W)`` or ``(H, W, C)``.
        size: Target ``(height, width)``.

    Returns:
        Resized uint8 array.
    """
    return matlab_imresize(image, size)
