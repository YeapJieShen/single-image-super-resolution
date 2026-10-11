"""NIQE -- a native port of Mittal et al.'s MATLAB ``niqe_release``.

A no-reference quality score: the distance between a multivariate Gaussian fitted
to an image's natural-scene statistics and one fitted to pristine images (lower is
better). Ported line for line from the authors' release (``computequality.m``,
``computefeature.m``, ``estimateaggdparam.m``) rather than taken from pyiqa, whose
NIQE misses the MATLAB release by ~0.1 on one of its own calibration images (I03,
where this port is 0.011 off).

The details that make it the release's number and not a lookalike:

* **8-bit input.** The release is run on ``imread`` output, so the image is
  quantised to uint8 first, and the luma is MATLAB's (``rgb2gray``, or the Y of
  ``rgb2ycbcr`` for PIRM's perceptual index) rounded back to uint8 -- MATLAB's
  coefficients, not the rounded 0.299/0.587/0.114.
* **The second scale is MATLAB ``imresize(im, 0.5)`` on doubles**: antialiased
  bicubic with no re-quantisation, built on the same contributions as the
  byte-exact :mod:`sisr.utils.imresize`.
* **Every 96x96 block is scored.** The 0.75 sharpness threshold in the release
  belongs to model *training* (``estimatemodelparam.m``); ``computequality.m``
  computes the sharpness and never uses it.
* **Pair products shift within the block** (``circshift`` inside ``blkproc``), and a
  block whose statistics are undefined yields NaN features that ``nanmean`` /
  ``nancov`` drop. An image with no measurable block at all (a constant one) scores
  NaN; the release raises there instead.

Verified against MATLAB (``tests/metrics/test_niqe_parity.py``): within 0.05 of the
release on 4 of IQA-PyTorch's 5 calibration images (I04 and I06 to 2e-4), and 4e-4 from
BasicSR's MATLAB R2021a figure for its baboon.png. The fifth, I19, is recorded as 8.635
where this port and every independent one give ~9.1, and is treated as a suspect
reference rather than tuned towards.

The pristine model (``modelparameters.mat``) is not redistributed here. It is
fetched on first use from the authors' release archive, checked against a pinned
SHA-256, and cached under ``torch.hub.get_dir()``.

------------------------------------------------------------------------------
-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
Copyright (c) 2011 The University of Texas at Austin
All rights reserved.

Permission is hereby granted, without written agreement and without license or
royalty fees, to use, copy, modify, and distribute this code (the source files) and
its documentation for any purpose, provided that the copyright notice in its
entirety appear in all copies of this code, and the original source of this code,
Laboratory for Image and Video Engineering (LIVE, http://live.ece.utexas.edu) and
Center for Perceptual Systems (CPS, http://www.cps.utexas.edu) at the University of
Texas at Austin (UT Austin, http://www.utexas.edu), is acknowledged in any
publication that reports research using this code. The research is to be cited in
the bibliography as:

1) A. Mittal, R. Soundararajan and A. C. Bovik, "NIQE Software Release",
URL: http://live.ece.utexas.edu/research/quality/niqe.zip, 2012.

2) A. Mittal, R. Soundararajan and A. C. Bovik, "Making a Completely Blind Image
Quality Analyzer", submitted to IEEE Signal Processing Letters, 2012.

IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR
DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF TEXAS AT
AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE
UNIVERSITY OF TEXAS AT AUSTIN HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------
------------------------------------------------------------------------------
"""

import functools
import hashlib
import io
import math
import urllib.request
import zipfile
from pathlib import Path
from typing import Literal

import torch

from sisr.utils.imresize import _contributions

#: The authors' release archive and the model file inside it, both pinned.
RELEASE_URL = "http://live.ece.utexas.edu/research/quality/niqe_release.zip"
RELEASE_SHA256 = "3d613b077a1b08273c79999f2e1643d5b2d9b2d0d3f3aa622236fc9592ea857c"
_MODEL_MEMBER = "niqe_release/modelparameters.mat"
MODEL_SHA256 = "fa2aa4d12dd9595c3fb6bf8b4e1a15c79d5d1469238d0901c88dead195be578d"

BLOCK_SIZE = 96
_FEATURES_PER_SCALE = 18

# MATLAB rgb2gray: the first row of inv(YIQ-from-RGB), i.e. not 0.299/0.587/0.114.
_RGB2GRAY = (0.298936021293775, 0.587043074451121, 0.114020904255103)
# MATLAB rgb2ycbcr's Y row for [0, 255] input, and its offset.
_RGB2Y = (65.481 / 255.0, 128.553 / 255.0, 24.966 / 255.0)
_Y_OFFSET = 16.0

Luma = Literal["gray", "ycbcr"]
PristineModel = tuple[torch.Tensor, torch.Tensor]


def _round_half_away(x: torch.Tensor) -> torch.Tensor:
    """MATLAB's ``round`` -- ties away from zero, where ``torch.round`` ties to even."""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def to_uint8_luma(img: torch.Tensor, luma: Luma = "gray") -> torch.Tensor:
    """8-bit luma exactly as MATLAB derives it from an ``imread`` RGB image.

    Args:
        img: ``(B, 3, H, W)`` RGB in ``[0, 1]``, quantised to uint8 levels first.
        luma: ``'gray'`` for ``rgb2gray`` (the NIQE release's own conversion) or
            ``'ycbcr'`` for the Y of ``rgb2ycbcr`` (what PIRM's perceptual index
            feeds NIQE).

    Returns:
        ``(B, H, W)`` float64 holding integer values in ``[0, 255]``.
    """
    rgb = _round_half_away(img.detach().to(torch.float64).clamp(0.0, 1.0) * 255.0)
    if luma == "gray":
        coef, offset = _RGB2GRAY, 0.0
    elif luma == "ycbcr":
        coef, offset = _RGB2Y, _Y_OFFSET
    else:
        raise ValueError(f"luma must be 'gray' or 'ycbcr'; got {luma!r}")
    y = coef[0] * rgb[:, 0] + coef[1] * rgb[:, 1] + coef[2] * rgb[:, 2] + offset
    return _round_half_away(y).clamp(0.0, 255.0)


@functools.cache
def _gaussian_window() -> torch.Tensor:
    """``fspecial('gaussian', 7, 7/6)``, normalised to unit sum as the release does."""
    ax = torch.arange(-3, 4, dtype=torch.float64)
    sq = ax[:, None] ** 2 + ax[None, :] ** 2
    h = torch.exp(-sq / (2.0 * (7.0 / 6.0) ** 2))
    h = torch.where(h < torch.finfo(torch.float64).eps * h.max(), 0.0, h)
    return h / h.sum()


def _imfilter_replicate(x: torch.Tensor) -> torch.Tensor:
    """``imfilter(x, window, 'replicate')`` on one ``(H, W)`` plane (correlation, 'same')."""
    padded = torch.nn.functional.pad(x[None, None], (3, 3, 3, 3), mode="replicate")
    return torch.nn.functional.conv2d(padded, _gaussian_window()[None, None])[0, 0]


def _mscn(im: torch.Tensor) -> torch.Tensor:
    """Mean-subtracted contrast-normalised coefficients (``structdis``)."""
    mu = _imfilter_replicate(im)
    sigma = torch.sqrt((_imfilter_replicate(im * im) - mu * mu).abs())
    return (im - mu) / (sigma + 1.0)


@functools.cache
def _aggd_grid() -> tuple[torch.Tensor, torch.Tensor]:
    """``gam = 0.2:0.001:10`` and its ``r_gam``, built the way MATLAB builds them.

    The colon range is formed symmetrically from both ends, as MATLAB does, and the
    ratio uses :func:`math.gamma` rather than ``exp(lgamma)`` so each grid value is
    as close to MATLAB's ``gamma`` as double allows.
    """
    n = round((10.0 - 0.2) / 0.001) + 1
    half = n // 2
    gam = [0.2 + k * 0.001 if k < half else 10.0 - (n - 1 - k) * 0.001 for k in range(n)]
    r_gam = [math.gamma(2.0 / g) ** 2 / (math.gamma(1.0 / g) * math.gamma(3.0 / g)) for g in gam]
    return torch.tensor(gam, dtype=torch.float64), torch.tensor(r_gam, dtype=torch.float64)


def _aggd(vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``estimateaggdparam`` on each row of ``vec`` (``(N, P)``) -> alpha, beta_l, beta_r.

    An empty side (no negative, or no positive, values) gives MATLAB's
    ``sqrt(mean([]))`` = NaN, which propagates into that row's features.
    """
    neg, pos = vec < 0, vec > 0
    left = torch.sqrt((vec * vec * neg).sum(1) / neg.sum(1))
    right = torch.sqrt((vec * vec * pos).sum(1) / pos.sum(1))
    gammahat = left / right
    rhat = vec.abs().mean(1) ** 2 / (vec * vec).mean(1)
    rhatnorm = rhat * (gammahat**3 + 1) * (gammahat + 1) / (gammahat**2 + 1) ** 2
    gam, r_gam = _aggd_grid()
    alpha = gam[((r_gam[None, :] - rhatnorm[:, None]) ** 2).argmin(1)]
    scale = torch.sqrt(torch.exp(torch.lgamma(1 / alpha) - torch.lgamma(3 / alpha)))
    return alpha, left * scale, right * scale


def _block_features(structdis: torch.Tensor, block: int) -> torch.Tensor:
    """``computefeature`` on every ``block x block`` tile -> ``(num_blocks, 18)``."""
    h, w = structdis.shape
    tiles = (
        structdis.reshape(h // block, block, w // block, block)
        .permute(0, 2, 1, 3)
        .reshape(-1, block, block)
    )
    flat = tiles.reshape(tiles.shape[0], -1)
    alpha, beta_l, beta_r = _aggd(flat)
    feats = [alpha, (beta_l + beta_r) / 2]
    for shift in ((0, 1), (1, 0), (1, 1), (1, -1)):
        pair = flat * torch.roll(tiles, shift, dims=(1, 2)).reshape(flat.shape)
        alpha, beta_l, beta_r = _aggd(pair)
        mean = (beta_r - beta_l) * torch.exp(torch.lgamma(2 / alpha) - torch.lgamma(1 / alpha))
        feats += [alpha, mean, beta_l, beta_r]
    return torch.stack(feats, dim=1)


def _imresize_half(im: torch.Tensor) -> torch.Tensor:
    """MATLAB ``imresize(im, 0.5)`` on a double ``(H, W)`` plane: no rounding, no clip."""
    out = im
    for axis in (0, 1):
        length = out.shape[axis]
        weights, indices = _contributions(length, math.ceil(length * 0.5), 0.5)
        w = torch.from_numpy(weights.copy())
        idx = torch.from_numpy(indices.copy())
        if axis == 0:
            out = torch.einsum("ot,otw->ow", w, out[idx])
        else:
            out = torch.einsum("ot,hot->ho", w, out[:, idx])
    return out


def niqe_features(luma: torch.Tensor) -> torch.Tensor:
    """The release's 36 features for every 96x96 block of one luma plane.

    Args:
        luma: ``(H, W)`` float64 on the 0-255 scale; cropped here to whole blocks.

    Returns:
        ``(num_blocks, 36)`` float64 -- 18 features at full scale, then the same 18
        at half scale for the co-located 48x48 block.
    """
    rows, cols = luma.shape[0] // BLOCK_SIZE, luma.shape[1] // BLOCK_SIZE
    im = luma[: rows * BLOCK_SIZE, : cols * BLOCK_SIZE]
    feats = []
    for scale in (1, 2):
        feats.append(_block_features(_mscn(im), BLOCK_SIZE // scale))
        im = _imresize_half(im)
    return torch.cat(feats, dim=1)


def _mvg_distance(features: torch.Tensor, model: PristineModel) -> torch.Tensor:
    """``sqrt(d' pinv((cov_p + cov_d) / 2) d)`` with MATLAB's ``nanmean``/``nancov``.

    ``nancov`` drops every row holding a NaN and, like MATLAB's ``cov``, normalises a
    single remaining row by 1 (a zero covariance). With no complete row MATLAB's
    ``pinv`` raises on the NaN matrix; this returns NaN instead, so an image with no
    measurable block logs NaN rather than aborting the run.
    """
    mu_p, cov_p = model
    complete = features[~features.isnan().any(dim=1)]
    if complete.shape[0] == 0:
        return torch.tensor(math.nan, dtype=torch.float64)
    mu_d = features.nanmean(dim=0)
    cov_d = torch.cov(complete.T, correction=1 if complete.shape[0] > 1 else 0)
    d = (mu_p - mu_d)[None, :]
    return torch.sqrt(d @ torch.linalg.pinv((cov_p + cov_d) / 2) @ d.T)[0, 0]


def niqe(
    img: torch.Tensor,
    *,
    luma: Luma = "gray",
    shave: int = 0,
    model: PristineModel | None = None,
) -> torch.Tensor:
    """NIQE per image, as ``computequality(imread(...), 96, 96, 0, 0, mu, cov)`` gives it.

    Args:
        img: ``(B, 3, H, W)`` RGB in ``[0, 1]``. Needs at least one 96x96 block
            after ``shave``.
        luma: Which MATLAB luma to score -- ``'gray'`` (the release's own
            ``rgb2gray``) or ``'ycbcr'`` (PIRM's Y). See :func:`to_uint8_luma`.
        shave: Pixels dropped from every edge of the luma before scoring (PIRM's
            ``shave_width``).
        model: ``(mu, cov)`` of the pristine Gaussian, ``(36,)`` and ``(36, 36)``.
            Defaults to the release's own model (:func:`pristine_model`).

    Returns:
        ``(B,)`` float64 scores on ``img``'s device; lower is better.

    Raises:
        ValueError: If the image has no whole 96x96 block after ``shave``.
    """
    mu, cov = pristine_model() if model is None else model
    mu, cov = mu.to(torch.float64).reshape(-1).cpu(), cov.to(torch.float64).cpu()
    planes = to_uint8_luma(img.cpu(), luma)
    if shave:
        planes = planes[:, shave:-shave, shave:-shave]
    if planes.shape[-2] < BLOCK_SIZE or planes.shape[-1] < BLOCK_SIZE:
        raise ValueError(
            f"NIQE needs at least one {BLOCK_SIZE}x{BLOCK_SIZE} block; got a "
            f"{tuple(planes.shape[-2:])} luma after shave={shave}"
        )
    scores = [_mvg_distance(niqe_features(p), (mu, cov)) for p in planes]
    return torch.stack(scores).to(img.device)


def _cache_path() -> Path:
    return Path(torch.hub.get_dir()) / "sisr" / "niqe" / "modelparameters.mat"


def _download_model(dest: Path) -> None:
    """Fetch the release archive, verify it, and write only the model file to ``dest``."""
    with urllib.request.urlopen(RELEASE_URL) as response:  # noqa: S310 -- pinned URL + hash
        archive = response.read()
    digest = hashlib.sha256(archive).hexdigest()
    if digest != RELEASE_SHA256:
        raise RuntimeError(
            f"NIQE release archive checksum mismatch: got {digest}, expected {RELEASE_SHA256}"
        )
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        payload = zf.read(_MODEL_MEMBER)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".partial")
    tmp.write_bytes(payload)
    tmp.replace(dest)


@functools.cache
def pristine_model() -> PristineModel:
    """The release's pristine Gaussian ``(mu (36,), cov (36, 36))``, float64.

    Downloaded once from :data:`RELEASE_URL` into ``torch.hub.get_dir()`` and
    verified against :data:`MODEL_SHA256` on every load.

    Raises:
        ImportError: If SciPy (needed to read the ``.mat``) is not installed.
        RuntimeError: If a checksum does not match.
    """
    try:
        from scipy.io import loadmat
    except ImportError as err:
        raise ImportError(
            "NIQE reads its pristine model with SciPy, which is not installed. Install the "
            "extra: pip install '.[perceptual]'."
        ) from err
    path = _cache_path()
    if not path.exists():
        _download_model(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(
            f"{path} checksum mismatch: got {digest}, expected {MODEL_SHA256}; delete it to "
            "re-download."
        )
    mat = loadmat(path)
    return (
        torch.from_numpy(mat["mu_prisparam"]).to(torch.float64).reshape(-1),
        torch.from_numpy(mat["cov_prisparam"]).to(torch.float64),
    )
