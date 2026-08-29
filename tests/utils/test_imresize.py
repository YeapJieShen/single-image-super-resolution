"""Tests for the vendored MATLAB-compatible :mod:`sisr.utils.imresize`.

Two layers of evidence:

1. Structural tests (always run) exercise the specific properties that make
   this a MATLAB-parity port rather than "generic bicubic": the a=-0.5
   kernel coefficient, antialiasing kernel widening on downscale, symmetric
   (mirror) border padding, and round-half-away-from-zero on the uint8 cast.
2. Byte-equality tests against real MATLAB-generated reference data —
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

To enable the downscale byte-equality test, download that URL, verify the
checksum, then extract only the ``Set5``, ``Set14`` and ``B100`` subtrees
into::

    data/reference/Set5/{HR,LR_bicubic/{X2,X3,X4}}
    data/reference/Set14/{HR,LR_bicubic/{X2,X3,X4}}
    data/reference/B100/{HR,LR_bicubic/{X2,X3,X4}}

e.g. (Windows ``tar`` needs ``--force-local`` or a ``C:`` path is parsed as
a remote host, and ``--wildcards`` for the subtree globs to expand)::

    tar --force-local -xf benchmark.tar --wildcards \
        'benchmark/Set5/*' 'benchmark/Set14/*' 'benchmark/B100/*'

then move all three dirs under ``data/reference/``.

SRCNN's degradation is bicubic-down *then* bicubic-up (see
:func:`sisr.datasets.srcnn._degrade`) -- the benchmark distribution above
only covers the downscale leg. To also cover the upscale leg, generate, in
MATLAB, for every ``LR_bicubic/X{s}/<stem>x{s}.png`` produced above::

    Bicubic_up/X{s}/<stem>x{s}.png = imresize(lr, [h_crop w_crop], 'bicubic')

where ``lr`` is that ``LR_bicubic`` image and ``h_crop, w_crop`` are the
source HR image's dimensions mod-cropped to a multiple of ``s``
(equivalently ``s ×`` ``lr``'s own dimensions), and drop the results
under::

    data/reference/Set5/Bicubic_up/{X2,X3,X4}
    data/reference/Set14/Bicubic_up/{X2,X3,X4}
    data/reference/B100/Bicubic_up/{X2,X3,X4}

Both legs now have byte-equality evidence -- the ``Bicubic_up`` data above was
generated and is in place, so the upscale test below runs rather than skipping.
It stays a conditional skip so a contributor without the archive still gets a
green suite; **a skip here means the leg was not exercised, never that it
passed**, so check the skip count, not just the exit code.

Both byte-equality tests skip cleanly (not an error, not a silent no-op)
when their respective reference data is absent, so CI stays hermetic and
contributors without the archive get a green suite.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sisr.utils.imresize import (
    _contributions,
    _cubic,
    _resize_axis,
    _round_half_away_from_zero,
    matlab_imresize,
    resize,
)

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"


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


##########################################################################
# _contributions memoization (functools.lru_cache)
##########################################################################


def test_contributions_repeat_call_returns_identical_cached_arrays():
    """Same args must return the exact same (cached) array objects, not
    merely equal ones -- proves the lru_cache is actually hit, not just
    present and bypassed."""
    weights1, indices1 = _contributions(50, 20, 0.4)
    weights2, indices2 = _contributions(50, 20, 0.4)
    assert weights1 is weights2
    assert indices1 is indices2


def test_contributions_cached_arrays_are_read_only():
    """Cached arrays are shared across every call with the same args; a
    mutating caller would corrupt every other consumer, so both arrays must
    refuse in-place writes."""
    weights, indices = _contributions(50, 20, 0.4)
    assert weights.flags.writeable is False
    assert indices.flags.writeable is False
    with pytest.raises(ValueError):
        weights[0, 0] = 999.0
    with pytest.raises(ValueError):
        indices[0, 0] = 999


def test_contributions_cache_matches_uncached_computation():
    """The memoized wrapper must compute the exact same weights/indices as
    the undecorated function -- reached via __wrapped__, lru_cache's
    standard escape hatch to the raw callable, so this holds regardless of
    what else has populated the cache."""
    for in_length, out_length, scale in [(100, 25, 0.25), (25, 100, 4.0), (24, 8, 1 / 3)]:
        cached_weights, cached_indices = _contributions(in_length, out_length, scale)
        raw_weights, raw_indices = _contributions.__wrapped__(in_length, out_length, scale)
        assert np.array_equal(cached_weights, raw_weights)
        assert np.array_equal(cached_indices, raw_indices)


def test_matlab_imresize_output_unchanged_by_contributions_cache():
    """End-to-end resize output must be byte-identical whether _contributions
    is served from cache or recomputed via __wrapped__ -- the strongest
    available proof that memoizing it didn't change matlab_imresize's
    numerics."""
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    out_shape = (8, 8)

    cached_out = matlab_imresize(img, out_shape)

    in_h, in_w = img.shape[:2]
    out_h, out_w = out_shape
    scale = out_h / in_h  # square image, square target -> one scale for both axes
    weights_h, indices_h = _contributions.__wrapped__(in_h, out_h, scale)
    weights_w, indices_w = _contributions.__wrapped__(in_w, out_w, scale)
    arr = _resize_axis(img, 0, weights_h, indices_h)
    arr = _resize_axis(arr, 1, weights_w, indices_w)

    assert np.array_equal(cached_out, arr)


##########################################################################
# _resize_axis single-pass einsum contraction
##########################################################################


def _naive_resize_axis(image: np.ndarray, axis: int, weights: np.ndarray, indices: np.ndarray):
    """Pre-optimization reference: materializes the full weights*gathered
    temporary before np.sum -- the exact computation _resize_axis's einsum
    now replaces. Kept local to this test so the einsum path has an
    independent, un-optimized oracle to compare against."""
    if axis == 0:
        gathered = image[indices].astype(np.float64)
        out = np.sum(weights[:, :, None, None] * gathered, axis=1)
    else:
        gathered = image[:, indices].astype(np.float64)
        out = np.sum(weights[None, :, :, None] * gathered, axis=2)
    return _round_half_away_from_zero(np.clip(out, 0.0, 255.0)).astype(np.uint8)


@pytest.mark.parametrize(
    "in_h,in_w,out_h,out_w,channels",
    [
        (30, 45, 10, 15, 3),  # downscale, non-square
        (10, 15, 30, 45, 3),  # upscale, non-square
        (17, 23, 9, 8, 1),  # downscale, 1-channel, odd sizes
        (9, 8, 17, 23, 1),  # upscale, 1-channel
        (24, 24, 24, 24, 3),  # identity scale
    ],
)
def test_resize_axis_einsum_matches_naive_multiply_then_sum(in_h, in_w, out_h, out_w, channels):
    """The einsum contraction in _resize_axis must be bit-identical to the
    multiply-then-np.sum it replaces, for both axis passes, non-square
    shapes, 1- and 3-channel images, and both up- and down-scaling."""
    rng = np.random.default_rng(hash((in_h, in_w, out_h, out_w, channels)) % (2**32))
    img = rng.integers(0, 256, size=(in_h, in_w, channels), dtype=np.uint8)

    weights_h, indices_h = _contributions(in_h, out_h, out_h / in_h)
    weights_w, indices_w = _contributions(in_w, out_w, out_w / in_w)

    assert np.array_equal(
        _resize_axis(img, 0, weights_h, indices_h),
        _naive_resize_axis(img, 0, weights_h, indices_h),
    )
    assert np.array_equal(
        _resize_axis(img, 1, weights_w, indices_w),
        _naive_resize_axis(img, 1, weights_w, indices_w),
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_resize_axis_gather_chunking_is_byte_identical_and_really_splits(axis, monkeypatch):
    """Slicing the gather along the free axis must not move a single byte.

    ``_resize_axis`` caps one gather's float64 footprint at
    ``_GATHER_CHUNK_BYTES`` and walks the free axis in slices, because the
    un-split gather is 286.88 MiB for a full-size DIV2K validation image and that
    allocation is what exhausted host RAM at a run's first validation. The
    contraction runs over the tap axis alone, so the split is arithmetically
    inert.

    Both halves are load-bearing. The equality alone would pass vacuously under a
    cap that never splits — which is exactly the configuration the training path
    runs in — so the observed gather sizes are asserted too: more than one, and
    every one within the cap.
    """
    import sisr.utils.imresize as imresize

    rng = np.random.default_rng(20260809 + axis)
    img = rng.integers(0, 256, size=(40, 53, 3), dtype=np.uint8)
    in_length = img.shape[axis]
    out_length = in_length // 3
    free_length = img.shape[1 if axis == 0 else 0]
    weights, indices = _contributions(in_length, out_length, out_length / in_length)
    bytes_per_column = out_length * weights.shape[1] * img.shape[2] * 8

    sizes: list[int] = []
    real_einsum = np.einsum

    def spy_einsum(subscripts, *operands, **kwargs):
        sizes.append(operands[1].nbytes)  # the gathered taps
        return real_einsum(subscripts, *operands, **kwargs)

    monkeypatch.setattr(imresize.np, "einsum", spy_einsum)

    monkeypatch.setattr(imresize, "_GATHER_CHUNK_BYTES", free_length * bytes_per_column)
    whole = _resize_axis(img, axis, weights, indices)
    assert len(sizes) == 1, "the unsplit reference itself split -- cap chosen too small"

    for columns in (3, 1):
        sizes.clear()
        cap = columns * bytes_per_column
        monkeypatch.setattr(imresize, "_GATHER_CHUNK_BYTES", cap)
        split = _resize_axis(img, axis, weights, indices)
        assert len(sizes) == -(-free_length // columns), "the cap did not split the gather"
        assert max(sizes) <= cap
        assert np.array_equal(whole, split)


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
# resize()
# ---------------------------------------------------------------------------


def test_resize_matches_matlab_imresize():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    assert np.array_equal(resize(img, (8, 8)), matlab_imresize(img, (8, 8)))


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


# ---------------------------------------------------------------------------
# Byte-equality for the upscale leg (bicubic-up) against MATLAB reference data
# ---------------------------------------------------------------------------


def _upscale_cases(base_dir: Path = REFERENCE_DIR) -> list[tuple[str, Path, Path, int]]:
    """Pairs each ``LR_bicubic`` reference image with its ``Bicubic_up`` counterpart.

    Takes *base_dir* as a parameter (rather than hardcoding
    :data:`REFERENCE_DIR`) so the mismatch-detection sanity check can point
    it at a synthetic tmp-dir fixture instead of real reference data.
    """
    cases = []
    for dataset in ("Set5", "Set14", "B100"):
        for scale in (2, 3, 4):
            lr_dir = base_dir / dataset / "LR_bicubic" / f"X{scale}"
            up_dir = base_dir / dataset / "Bicubic_up" / f"X{scale}"
            if not lr_dir.is_dir():
                continue
            for lr_path in sorted(lr_dir.glob("*.png")):
                up_path = up_dir / lr_path.name
                if up_path.exists():
                    cases.append((dataset, lr_path, up_path, scale))
    return cases


def test_matlab_imresize_upscale_matches_real_matlab_reference_data():
    """Byte-equality against MATLAB-generated ``Bicubic_up`` pairs.

    Covers SRCNN's second (upscale) degradation step -- see
    :func:`sisr.datasets.srcnn._degrade` -- which the downscale test above
    does not exercise at all: that test only proves HR-to-LR byte-equality,
    but ``_degrade`` also resizes the LR back up to the HR's own size, and
    until now nothing checked that leg against real MATLAB output. No
    recomputation from HR is needed here: since the referenced LR image is
    itself already mod-cropped (see the downscale test's docstring), the
    upscale target is simply ``scale x lr.shape``.
    """
    cases = _upscale_cases()
    if not cases:
        pytest.skip(
            f"No Bicubic_up/ reference data under {REFERENCE_DIR} -- generate it per this "
            "file's module docstring to enable the upscale byte-equality check."
        )

    mismatches = []
    for dataset, lr_path, up_path, scale in cases:
        lr = np.array(Image.open(lr_path).convert("RGB"))
        up_real = np.array(Image.open(up_path).convert("RGB"))
        h_lr, w_lr = lr.shape[:2]
        up_mine = matlab_imresize(lr, (h_lr * scale, w_lr * scale))
        if up_mine.shape != up_real.shape or not np.array_equal(up_mine, up_real):
            mismatches.append(f"{dataset}/{lr_path.stem} x{scale}")

    assert not mismatches, (
        f"{len(mismatches)}/{len(cases)} reference images did not reproduce byte-exactly: "
        f"{mismatches[:10]}"
    )
