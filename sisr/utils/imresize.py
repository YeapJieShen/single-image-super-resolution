"""MATLAB-compatible bicubic image resizing.

Vendored -- not a dependency -- from ``fatheral/matlab_imresize`` (MIT, (c) 2020
Alex; licence verbatim below), itself a line-for-line port of MATLAB's
``toolbox/images/images/imresize.m``. The BasicSR / SwinIR / Real-ESRGAN lineage
carries the same algorithm, but BasicSR's float32 port rounds only once at the
end; this follows the MIT original, which computes in double and rounds every
pass -- both of which are load-bearing, not incidental.

**Three details are load-bearing for byte-equality and must not be
"simplified".** Each was confirmed by building the broken variant:

* **float64 throughout.** MATLAB computes in double; a float32 port drifts by
  fractions of a level, invisible until a uint8 cast scatters it into +/-1
  errors.
* **Round half away from zero on every uint8 cast.** MATLAB's ``round`` ties
  away from zero, ``np.round`` ties to even. The upstream file uses
  ``np.around`` and inherits the off-by-one -- the classic reason a "faithful"
  port fails byte-equality. Banker's rounding scores 19/57.
* **Re-quantize to uint8 after *each* 1-D pass, not only at the end.** MATLAB's
  ``imresizemex`` re-quantizes between the row and column passes. It looks
  redundant; skipping it is an independent source of drift. Rounding once at the
  end scores 0/57.

Two real deltas this closes against a naive OpenCV-style bicubic: MATLAB widens
the kernel by ``1/scale`` when downscaling (antialiasing, which OpenCV does not
do at all -- the dominant term), and the bicubic coefficient differs
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


def _order_free_factor(in_length: int, out_length: int) -> int | None:
    """The integer upscale factor ``out_length / in_length`` if any summation order is exact.

    For an integer upscale factor *s*, every bicubic weight is an exact rational:
    the kernel is sampled at multiples of ``1/(2s)``, its rows sum to exactly 1,
    and the polynomial's halves cancel against the sample's denominator. When *s*
    is odd, every weight is a multiple of ``1/s**3``, so a pass over integer pixels
    lands on a multiple of ``1/s**3`` too -- never within ``1/(2*s**3)`` of a
    rounding tie, against a float error below ``1e-11``. When *s* is a power of
    two, every weight, product and sum is exactly representable, so there is no
    error at all. Either way, every uint8 a pass produces is independent of the
    order its terms are summed in, which is what lets :func:`resize_window` use a
    different (and faster) contraction from :func:`matlab_imresize` and still
    match it byte for byte. Other factors (6, non-integer, downscales) can land
    on a tie that only the exact summation order resolves, and return ``None``.
    """
    factor, rem = divmod(out_length, in_length)
    if rem or factor < 1 or (factor % 2 == 0 and factor & (factor - 1)):
        return None
    return factor


@functools.cache
def _upscale_taps(factor: int, phase: int, length: int) -> tuple[np.ndarray, int]:
    """Dense weights for *length* output pixels starting at output pixel *phase*.

    Output pixel ``i``'s centre sits at source coordinate ``(2i + 1 - factor) /
    (2 * factor)``, and its four taps start one pixel before that centre's floor
    (a tap two pixels away has weight exactly 0, so four cover the support). The
    centre advances one source pixel per *factor* output pixels, so a window at
    output pixel ``q * factor + phase`` is this window shifted by ``q`` source
    pixels -- the table depends only on ``(factor, phase, length)``, a handful of
    entries per dataset.

    Returns:
        ``(dense, first)``: a read-only ``(length, n)`` matrix whose row ``r``
        holds output pixel ``phase + r``'s weights over the source pixels
        ``first .. first + n - 1`` (unfolded -- may run past either edge).
    """
    num = 2 * np.arange(phase, phase + length) + 1 - factor
    taps = (num // (2 * factor) - 1)[:, None] + np.arange(4)
    weights = _cubic(num[:, None] / (2 * factor) - taps)
    first = int(taps[0, 0])
    dense = np.zeros((length, int(taps[-1, -1]) - first + 1))
    dense[np.arange(length)[:, None], taps - first] = weights
    dense.setflags(write=False)
    return dense, first


def _upscale_source(
    factor: int, in_length: int, start: int, length: int
) -> tuple[np.ndarray, slice | np.ndarray]:
    """One axis' dense weights for an output window, and the source pixels they act on.

    The source selector is a plain slice when the window's taps stay inside the
    image; otherwise its unfolded indices are mirror-folded exactly as
    :func:`_contributions` folds them, so a border tap reads the same pixel.
    """
    q, phase = divmod(start, factor)
    dense, first = _upscale_taps(factor, phase, length)
    first += q
    stop = first + dense.shape[1]
    if first >= 0 and stop <= in_length:
        return dense, slice(first, stop)
    folded = np.mod(np.arange(first, stop), 2 * in_length)
    return dense, np.where(folded < in_length, folded, 2 * in_length - 1 - folded)


def _upscale_pass(dense: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """``dense`` applied along *arr*'s first axis, rounded MATLAB's way.

    *arr* holds integer values, so the result is too: ``floor(clip(x) + 0.5)``
    is round-half-away-from-zero for the non-negative values clipping leaves,
    folded into one clip at ``255.5`` (identical for every input).
    """
    out = dense @ arr.reshape(arr.shape[0], -1)
    out += 0.5
    np.clip(out, 0.0, 255.5, out=out)
    np.floor(out, out=out)
    return out.reshape(dense.shape[0], *arr.shape[1:])


def resize_window(
    image: np.ndarray,
    output_shape: tuple[int, int],
    row_start: int,
    row_len: int,
    col_start: int,
    col_len: int,
) -> np.ndarray:
    """One window of an integer-factor upscale, without computing the rest.

    Byte-identical to ``matlab_imresize(image, output_shape)[row_start :
    row_start + row_len, col_start : col_start + col_len]``. Each output pixel
    depends only on its own few source taps, so only the source pixels the
    window's taps reach are read, and each pass is one small dense matrix
    product. That contraction sums in a different order from
    :func:`matlab_imresize`'s, which is exact only for the factors
    :func:`_order_free_factor` admits -- odd factors and powers of two; any
    other factor raises rather than risk a tie resolved the other way. Passes
    run smaller-factor axis first, as in :func:`matlab_imresize`.

    Args:
        image: uint8 array, ``(H, W)`` or ``(H, W, C)`` -- the whole source
            (e.g. a cached LR plane), not a pre-cropped window.
        output_shape: The full ``(height, width)`` the whole upscale would
            produce; it fixes each axis' factor.
        row_start: First output row of the window.
        row_len: Number of output rows to return.
        col_start: First output column of the window.
        col_len: Number of output columns to return.

    Returns:
        uint8 array of shape ``(row_len, col_len)`` or ``(row_len, col_len, C)``.

    Raises:
        ValueError: If the window does not fit inside *output_shape*, or an
            axis' factor is not an odd or power-of-two integer upscale.
    """
    in_h, in_w = image.shape[0], image.shape[1]
    out_h, out_w = output_shape
    if not (0 <= row_start and row_start + row_len <= out_h):
        msg = f"Window [{row_start}, {row_start + row_len}) exceeds output height {out_h}"
        raise ValueError(msg)
    if not (0 <= col_start and col_start + col_len <= out_w):
        msg = f"Window [{col_start}, {col_start + col_len}) exceeds output width {out_w}"
        raise ValueError(msg)
    factor_h = _order_free_factor(in_h, out_h)
    factor_w = _order_free_factor(in_w, out_w)
    if factor_h is None or factor_w is None:
        msg = (
            f"resize_window needs an odd or power-of-two integer upscale per axis; "
            f"got {in_h}x{in_w} -> {out_h}x{out_w}"
        )
        raise ValueError(msg)

    dense_h, rows = _upscale_source(factor_h, in_h, row_start, row_len)
    dense_w, cols = _upscale_source(factor_w, in_w, col_start, col_len)
    arr = image[:, :, None] if image.ndim == 2 else image
    arr = arr[rows][:, cols].astype(np.float64)
    if factor_h <= factor_w:
        arr = _upscale_pass(dense_h, arr)
        arr = _upscale_pass(dense_w, arr.transpose(1, 0, 2)).transpose(1, 0, 2)
    else:
        arr = _upscale_pass(dense_w, arr.transpose(1, 0, 2)).transpose(1, 0, 2)
        arr = _upscale_pass(dense_h, arr)
    out = arr.astype(np.uint8, order="C")
    return out[:, :, 0] if image.ndim == 2 else out


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
