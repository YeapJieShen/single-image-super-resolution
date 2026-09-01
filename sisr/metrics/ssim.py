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

**Vectorisation is exact, not an approximation.** daala clips the kernel at
edges and accumulates only the weights it used, so a zero-padded separable
convolution gives identical moment sums (missing taps contribute zero) and a
zero-padded ones-mask convolution gives exactly its ``m.w`` -- horizontal
truncation depends only on x, vertical only on y, so the 2-D weight factorises.

**float64 throughout, and it is exact.** daala accumulates in ``int64``; the
largest quantity is ~2.8e14, well under 2**53, so float64 reproduces its integer
accumulation exactly. Only the pooling *summation order* differs (C row-major vs
torch tree reduction), so parity against ``tests/reference/daala_ssim.c`` is
asserted at ``rel=1e-9``, not bit-equality. **That bound is measured, not
padding**: divergence grows as ~``eps*n/4``, so a correct implementation already
exceeds ``1e-12`` on flat 256x256 content, while the faintest real defect seen
(a float32 leak) reads ``5e-8``. Read the parity test's bracket before
tightening it.

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

import torch
import torch.nn.functional as F

_KERNEL_WEIGHT = 1 << 8  # daala's KERNEL_SHIFT is 8; the kernel sums to this.
_SSIM_K1 = 0.01 * 0.01
_SSIM_K2 = 0.03 * 0.03
_SAMPLEMAX = 255  # daala's (1 << depth) - 1, pinned to 8-bit input.
_SIGMA_PER_ROW = 1.5 / 256  # sigma = height * this. The whole story, one line.


def _gaussian_kernel_int(sigma: float, max_len: int) -> list[int]:
    """Build daala's integer gaussian kernel.

    Literal transcription of ``gaussian_filter_init``. The kernel length is
    chosen so the first truncated tap would quantise to zero -- its value is
    at most ``0.5`` in the integer tap units where the whole kernel sums to
    256 -- then capped at ``max_len - 1``. (daala's own C comment phrases this
    bound more loosely, as an error of ``0.5 * KERNEL_WEIGHT``.)

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
        # reason a "faithful" port misses parity (see sisr/utils/imresize.py).
        tap = int(_KERNEL_WEIGHT * scale * math.exp(nhisigma2 * ci * ci) + 0.5)
        kernel[kernel_len - ci] = kernel[kernel_len + ci] = tap
        side_sum += tap
    # The centre absorbs all quantisation error, so the sum is exactly 256.
    # Never renormalise this kernel.
    kernel[kernel_len] = _KERNEL_WEIGHT - 2 * side_sum
    return kernel


def quantize_u8(t: torch.Tensor) -> torch.Tensor:
    """Clamp to ``[0, 1]`` and quantise to integer 8-bit levels, in float64.

    daala reads 8-bit planes, so the daala path scores what an 8-bit image
    would hold. Rounds half up via ``floor(x + 0.5)`` — equivalent to
    half-away-from-zero here only because the preceding clamp to ``[0, 1]``
    means ``x`` is never negative; the two conventions disagree for negative
    ties. Matches :mod:`sisr.utils.imresize`'s convention rather than
    ``torch.round``'s ties-to-even. The Wang path is deliberately *not*
    quantised — changing it would renumber every SSIM this project has ever
    logged.

    Args:
        t: Float tensor in ``[0, 1]`` (values outside are clamped).

    Returns:
        ``float64`` tensor of integer values in ``[0, 255]``.
    """
    return torch.floor(t.to(torch.float64).clamp(0.0, 1.0) * 255.0 + 0.5)


def _blur(t: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Zero-padded separable correlation of ``(N, 1, H, W)`` with *kernel*.

    Vertical then horizontal; the pass order is irrelevant because every
    accumulation is an exact integer in float64. ``conv2d`` is a correlation,
    not a convolution, but the kernel is symmetric so it makes no difference.
    """
    n = kernel.numel() // 2
    t = F.conv2d(F.pad(t, (0, 0, n, n)), kernel.view(1, 1, -1, 1))
    return F.conv2d(F.pad(t, (n, n, 0, 0)), kernel.view(1, 1, 1, -1))


def daala_ssim(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    """SSIM computed by daala's methodology.

    Sigma is ``height * 1.5/256`` on **both** axes — daala derives the
    horizontal sigma from the height too, dividing by the pixel aspect ratio,
    which is 1 for the square-pixel images this project scores. The height used
    is that of the tensor passed in, i.e. already border-cropped by the caller.

    Multi-channel inputs are scored per plane and averaged, matching how the
    Wang path reduces channels; daala's own luma/chroma ``cweight`` is not used,
    since it weights chroma against luma for a full-colour score rather than
    the per-colourspace aggregate an ``ssim/.../RGB`` tag means here.

    Args:
        sr: Reconstruction, ``(B, C, H, W)`` float in ``[0, 1]``.
        hr: Reference, same shape.

    Returns:
        0-dim tensor: per-sample SSIM (itself a mean over planes), meaned over
        the batch — the same reduction the Wang path applies.

    Raises:
        ValueError: If either tensor is not 4-D, or if the two differ in shape.
    """
    if sr.dim() != 4 or hr.dim() != 4:
        raise ValueError(
            f"sr and hr must be 4-D (B, C, H, W); got shapes {tuple(sr.shape)} and "
            f"{tuple(hr.shape)}"
        )
    if sr.shape != hr.shape:
        raise ValueError(
            f"sr and hr must have the same shape; got {tuple(sr.shape)} vs {tuple(hr.shape)}"
        )

    b, c, h, w = sr.shape
    x = quantize_u8(sr).reshape(b * c, 1, h, w)
    y = quantize_u8(hr).reshape(b * c, 1, h, w)

    kernel = torch.tensor(
        _gaussian_kernel_int(h * _SIGMA_PER_ROW, min(w, h)),
        dtype=torch.float64,
        device=sr.device,
    )

    mux = _blur(x, kernel)
    muy = _blur(y, kernel)
    x2 = _blur(x * x, kernel)
    xy = _blur(x * y, kernel)
    y2 = _blur(y * y, kernel)
    # The ones-mask convolution reproduces daala's m.w exactly: near a border it
    # is the sum of the kernel taps that actually landed inside the image. `mw`
    # depends only on (h, w, kernel), not on plane content, so compute it once
    # on a single plane and let broadcasting apply it across all B*C planes —
    # float64 conv is otherwise the same cost as the five moment sums above,
    # paid B*C times for an identical result.
    mw = _blur(torch.ones((1, 1, h, w), dtype=x.dtype, device=x.device), kernel)

    # Operand grouping mirrors the C expression by expression. float64
    # multiply/add rounding is order-dependent and these products exceed 2**53,
    # so a regrouping is not bit-identical to this one -- measured at ~1e-16
    # relative after pooling, for each of four algebraically-equivalent
    # regroupings tried. That's far below any tolerance this module asserts
    # (all four still passed at rel=1e-12), so no test guards this grouping:
    # it stays exactly as written below by discipline, not by a check.
    smax2 = _SAMPLEMAX * _SAMPLEMAX
    c1 = ((smax2 * _SSIM_K1) * mw) * mw
    c2 = ((smax2 * _SSIM_K2) * mw) * mw
    mx2 = mux * mux
    mxy = mux * muy
    my2 = muy * muy
    numerator = (mw * (2 * mxy + c1)) * (c2 + 2 * (xy * mw - mxy))
    denominator = (mx2 + my2 + c1) * ((((x2 * mw - mx2) + y2 * mw) - my2) + c2)

    per_plane = (numerator / denominator).sum(dim=(1, 2, 3)) / mw.sum(dim=(1, 2, 3))
    return per_plane.view(b, c).mean(dim=1).mean()
