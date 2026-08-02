"""Tests for the vendored MATLAB-compatible :mod:`sisr.imresize` (INIT.11).

Two layers of evidence:

1. Structural tests (always run) exercise the specific properties that make
   this a MATLAB-parity port rather than "generic bicubic": the a=-0.5
   kernel coefficient, antialiasing kernel widening on downscale, symmetric
   (mirror) border padding, and round-half-away-from-zero on the uint8 cast.
2. A byte-equality test against real MATLAB-generated reference data —
   the strongest available claim, since without MATLAB itself this project
   cannot prove byte-identity from first principles.

Reference data (not committed — ``data/`` is gitignored):

    Source: https://cv.snu.ac.kr/research/EDSR/benchmark.tar
    (Lim et al., "Enhanced Deep Residual Networks for Single Image
    Super-Resolution", CVPRW 2017 — the EDSR authors' own benchmark
    distribution; Set5/Set14/B100 HR + MATLAB-``imresize``-generated
    ``LR_bicubic`` X2/X3/X4 pairs, the same lineage BasicSR/EDSR-derived
    repos distribute.)
    SHA-256 of the full 250112000-byte tar: 80c21c333bbf6ceb5308b7243761f82
    84478274413a97b96f1d63e9045fd93e8

To enable the byte-equality test, download that URL, verify the checksum,
then extract only the ``Set5``, ``Set14`` and ``B100`` subtrees into::

    data/reference/Set5/{HR,LR_bicubic/{X2,X3,X4}}
    data/reference/Set14/{HR,LR_bicubic/{X2,X3,X4}}
    data/reference/B100/{HR,LR_bicubic/{X2,X3,X4}}

e.g. (Windows ``tar`` needs ``--force-local`` or a ``C:`` path is parsed as
a remote host, and ``--wildcards`` for the subtree globs to expand)::

    tar --force-local -xf benchmark.tar --wildcards \
        'benchmark/Set5/*' 'benchmark/Set14/*' 'benchmark/B100/*'

then move all three dirs under ``data/reference/``. The test skips cleanly
(not an error, not a silent no-op) when that directory is absent, so CI
stays hermetic and contributors without the archive get a green suite.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sisr.imresize import (
    _contributions,
    _cubic,
    _round_half_away_from_zero,
    matlab_imresize,
    resize,
)

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------


def test_cubic_kernel_uses_matlab_a_minus_half_coefficient():
    """MATLAB's bicubic a=-0.5 gives cubic(0.5) = 0.5625; OpenCV's a=-0.75
    would give 0.59375 for the same input -- this pins which one we ported."""
    assert _cubic(np.array([0.5]))[0] == pytest.approx(0.5625)
    assert _cubic(np.array([0.0]))[0] == pytest.approx(1.0)
    assert _cubic(np.array([1.0]))[0] == pytest.approx(0.0)
    assert _cubic(np.array([2.0]))[0] == pytest.approx(0.0)
    assert _cubic(np.array([3.0]))[0] == pytest.approx(0.0)  # outside support


def test_round_half_away_from_zero_differs_from_numpy_banker_rounding():
    x = np.array([-2.5, -0.5, 0.5, 1.5, 2.5])
    assert list(_round_half_away_from_zero(x)) == [-3.0, -1.0, 1.0, 2.0, 3.0]
    # np.round ties to even -- the exact bug this module avoids.
    assert list(np.round(x)) == [-2.0, 0.0, 0.0, 2.0, 2.0]


def test_contributions_widens_kernel_when_downscaling():
    """Antialiasing = kernel widening by 1/scale on downscale only; the
    upscale/identity case keeps the fixed a=-0.5 support width."""
    _, indices_down = _contributions(100, 25, 0.25)  # 4x downscale
    _, indices_up = _contributions(25, 100, 4.0)  # 4x upscale
    assert indices_down.shape[1] > indices_up.shape[1]


def test_contributions_folds_out_of_range_indices_symmetrically():
    """Border output pixels must reference in-range source indices via
    mirror (reflect) padding, not clamping to the edge or wrapping around."""
    # A small in_length with output pixels near both edges forces taps to
    # fall outside [0, in_length) for a wide (downscale-widened) kernel.
    _, indices = _contributions(6, 3, 0.5)
    assert indices.min() >= 0
    assert indices.max() < 6
    # Mirror padding revisits near-edge source pixels rather than saturating
    # at a single edge value -- i.e. more than one distinct low index is used.
    assert len(set(indices[0].tolist())) > 1


def test_matlab_imresize_constant_image_is_invariant():
    """Weights always sum to 1, so resizing a constant-value image at any
    scale must reproduce that exact constant everywhere (up or down)."""
    const = np.full((20, 20, 3), 137, dtype=np.uint8)
    for out_shape in [(5, 5), (7, 13), (40, 40), (20, 20)]:
        out = matlab_imresize(const, out_shape)
        assert out.shape == (*out_shape, 3)
        assert np.all(out == 137)


def test_matlab_imresize_supports_2d_grayscale():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(20, 20), dtype=np.uint8)
    out = matlab_imresize(img, (10, 10))
    assert out.shape == (10, 10)
    assert out.dtype == np.uint8


def test_matlab_imresize_output_is_uint8_and_clipped():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(30, 30, 3), dtype=np.uint8)
    out = matlab_imresize(img, (11, 17))
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


# ---------------------------------------------------------------------------
# resize() backend dispatcher
# ---------------------------------------------------------------------------


def test_resize_matlab_backend_matches_matlab_imresize():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    assert np.array_equal(resize(img, (8, 8), backend="matlab"), matlab_imresize(img, (8, 8)))


def test_resize_cv2_backend_matches_cv2_directly():
    import cv2

    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    expected = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    assert np.array_equal(resize(img, (8, 8), backend="cv2"), expected)


def test_resize_defaults_to_matlab_backend():
    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    assert np.array_equal(resize(img, (8, 8)), matlab_imresize(img, (8, 8)))


def test_resize_unknown_backend_raises():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="pillow"):
        resize(img, (2, 2), backend="pillow")


# ---------------------------------------------------------------------------
# Byte-equality against real MATLAB-generated reference data
# ---------------------------------------------------------------------------


def _reference_cases() -> list[tuple[str, Path, Path, int]]:
    cases = []
    for dataset in ("Set5", "Set14", "B100"):
        hr_dir = REFERENCE_DIR / dataset / "HR"
        if not hr_dir.is_dir():
            continue
        for scale in (2, 3, 4):
            lr_dir = REFERENCE_DIR / dataset / "LR_bicubic" / f"X{scale}"
            for hr_path in sorted(hr_dir.glob("*.png")):
                lr_path = lr_dir / f"{hr_path.stem}x{scale}.png"
                if lr_path.exists():
                    cases.append((dataset, hr_path, lr_path, scale))
    return cases


def test_matlab_imresize_matches_real_matlab_reference_data():
    """Byte-equality against MATLAB-generated Set5/Set14 LR_bicubic pairs.

    Generation convention (reverse-engineered and confirmed exact): crop HR
    to a multiple of `scale` (mod-crop), then bicubic-downsample the cropped
    array directly to the exact `(h_crop // scale, w_crop // scale)` target
    -- matching how this project's own datasets derive LR (crop first, then
    resize the cropped array). Resizing the *uncropped* image straight to
    the floor-sized target does NOT match (confirmed during development:
    max|diff| up to 99/255 on the mismatching approach) -- the mod-crop step
    is load-bearing, not cosmetic.
    """
    if not REFERENCE_DIR.is_dir():
        pytest.skip(
            f"{REFERENCE_DIR} not present -- fetch per this file's module docstring to enable "
            "the byte-equality check against real MATLAB-generated reference data."
        )
    cases = _reference_cases()
    assert cases, f"{REFERENCE_DIR} exists but no HR/LR_bicubic.../X{{2,3,4}} pairs found under it"

    mismatches = []
    for dataset, hr_path, lr_path, scale in cases:
        hr = np.array(Image.open(hr_path).convert("RGB"))
        lr_real = np.array(Image.open(lr_path).convert("RGB"))
        h, w = hr.shape[:2]
        h_crop, w_crop = h - h % scale, w - w % scale
        cropped = hr[:h_crop, :w_crop, :]
        lr_mine = matlab_imresize(cropped, (h_crop // scale, w_crop // scale))
        if lr_mine.shape != lr_real.shape or not np.array_equal(lr_mine, lr_real):
            mismatches.append(f"{dataset}/{hr_path.stem} x{scale}")

    assert not mismatches, (
        f"{len(mismatches)}/{len(cases)} reference images did not reproduce byte-exactly: "
        f"{mismatches[:10]}"
    )
