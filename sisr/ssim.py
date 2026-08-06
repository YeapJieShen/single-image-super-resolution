"""SSIM by daala's methodology — the convention Ledig et al. actually used.

Ledig et al. (arXiv:1609.04802) state their PSNR/SSIM were "calculated on the
y-channel of center-cropped, removal of a 4-pixel wide strip from each border,
images using the **daala** package". daala's SSIM is not Wang's: its gaussian
sigma **scales with frame height** (``_h*(1.5/256)``), so it equals Wang's 1.5
only at H=256 and is wider for anything taller -- which reads systematically
higher. Measured on this project's data, that single difference accounts for
the whole SSIM gap against the paper, while PSNR already matched.

Four further differences from ``torchmetrics``, all load-bearing:

* The kernel is **integer-quantised**, summing to exactly ``256``, with the
  centre tap absorbing all rounding error (``256 - 2*sum(sides)``).
* Borders **truncate and renormalise** the kernel -- every pixel position is
  scored, with a smaller effective weight near edges -- rather than being
  dropped by a valid convolution.
* Pooling is a **weight-weighted** mean over those positions.
* Input is **8-bit**; ``samplemax`` is 255.

Vectorisation is exact, not an approximation: daala clips the kernel at edges
and accumulates only the weights it used, so a zero-padded separable
convolution produces identical moment sums (missing taps contribute zero), and
a zero-padded ones-mask convolution produces exactly its ``m.w`` -- horizontal
truncation depends only on x and vertical only on y, so the 2-D weight
factorises into what the separable mask convolution computes.

All arithmetic runs in float64. daala accumulates moments in ``int64`` with
integer weights; the largest quantity involved is ~2.8e14, well under 2**53, so
float64 reproduces its integer accumulation **exactly** and the per-position
divide runs in the same double arithmetic as the C. Only the final pooling
*summation order* differs (C: sequential row-major; torch: tree reduction), so
parity against ``tests/reference/daala_ssim.c`` is asserted at ``rel=1e-12``
rather than bit-equality.

------------------------------------------------------------------------------
Algorithm ported from daala's ``tools/dump_ssim.c``.

Copyright 2001-2012 Xiph.Org and contributors

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE FOUNDATION OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import math

import torch  # noqa: F401 -- consumed by the daala_ssim metric added in a later task.
import torch.nn.functional as F  # noqa: F401 -- ditto.

_KERNEL_WEIGHT = 1 << 8  # daala's KERNEL_SHIFT is 8; the kernel sums to this.
_SSIM_K1 = 0.01 * 0.01
_SSIM_K2 = 0.03 * 0.03
_SAMPLEMAX = 255  # daala's (1 << depth) - 1, pinned to 8-bit input.
_SIGMA_PER_ROW = 1.5 / 256  # sigma = height * this. The whole story, one line.


def _gaussian_kernel_int(sigma: float, max_len: int) -> list[int]:
    """Build daala's integer gaussian kernel.

    Literal transcription of ``gaussian_filter_init``. The kernel length is
    chosen so the first truncated coefficient errs by at most
    ``0.5 * KERNEL_WEIGHT``, then capped at ``max_len - 1``.

    Args:
        sigma: Gaussian standard deviation, in samples.
        max_len: Upper bound on the half-length, i.e. ``min(width, height)``.

    Returns:
        Odd-length list of non-negative integer taps summing to exactly 256.
    """
    scale = 1 / (math.sqrt(2 * math.pi) * sigma)
    nhisigma2 = -0.5 / (sigma * sigma)
    s = math.sqrt(0.5 * math.pi) * sigma * (1.0 / _KERNEL_WEIGHT)
    length = 0.0 if s >= 1 else math.floor(sigma * math.sqrt(-2 * math.log(s)))
    kernel_len = max_len - 1 if length >= max_len else int(length)

    kernel = [0] * (kernel_len * 2 + 1)
    side_sum = 0
    for ci in range(kernel_len, 0, -1):
        # int(v + 0.5), NOT round(): the C casts to unsigned, truncating toward
        # zero, i.e. round-half-UP. Python's round() ties to even -- the classic
        # reason a "faithful" port misses parity (see sisr/imresize.py).
        tap = int(_KERNEL_WEIGHT * scale * math.exp(nhisigma2 * ci * ci) + 0.5)
        kernel[kernel_len - ci] = kernel[kernel_len + ci] = tap
        side_sum += tap
    # The centre absorbs all quantisation error, so the sum is exactly 256.
    # Never renormalise this kernel.
    kernel[kernel_len] = _KERNEL_WEIGHT - 2 * side_sum
    return kernel
