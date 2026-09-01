"""The whole-image derived-plane cache.

Its whole reason to exist is that the raw-HR cache's checksum deliberately
ignores every derivation parameter, so anything scale-dependent stored there
would never be invalidated. These tests hold that separation.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sisr.datasets import derived_cache as dc
from sisr.utils.imresize import resize


@pytest.fixture
def image_paths(tmp_path: Path) -> list[Path]:
    """Two images, one of them not a multiple of 3 on either axis."""
    rng = np.random.default_rng(seed=3)
    paths = []
    for i, (h, w) in enumerate([(36, 36), (321, 481)]):
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


@pytest.mark.parametrize("scale", [2, 3, 4])
@pytest.mark.parametrize("kind", ["lr", "bicubic"])
def test_derive_matches_modcrop_then_resize_computed_independently(kind: str, scale: int):
    """Byte-exact against the authors' order, spelled out here rather than reused."""
    rng = np.random.default_rng(seed=11)
    arr = rng.integers(0, 256, size=(321, 481, 3), dtype=np.uint8)

    h, w = 321 - 321 % scale, 481 - 481 % scale
    gnd = np.ascontiguousarray(arr[:h, :w])
    expected = resize(gnd, (h // scale, w // scale))
    if kind == "bicubic":
        expected = resize(np.ascontiguousarray(expected), (h, w))

    assert np.array_equal(dc.derive(arr, kind, scale), expected)


@pytest.mark.parametrize("scale", [2, 3, 4])
@pytest.mark.parametrize("kind", ["lr", "bicubic"])
def test_derived_shape_predicts_the_real_output_shape(kind: str, scale: int):
    """``estimate_map_size`` sizes the database from this without deriving anything,
    so a disagreement is a MapFullError mid-build."""
    rng = np.random.default_rng(seed=12)
    arr = rng.integers(0, 256, size=(321, 481, 3), dtype=np.uint8)
    assert dc.derived_shape(321, 481, kind, scale) == dc.derive(arr, kind, scale).shape[:2]


def test_derive_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        dc.derive(np.zeros((8, 8, 3), np.uint8), "bilinear", 2)


def test_cache_name_separates_kind_scale_and_imresize_version():
    """A stale plane must be unreachable, not silently wrong: the database name
    itself carries every parameter that changes the pixels."""
    names = {
        dc.cache_name("lr", 2),
        dc.cache_name("lr", 4),
        dc.cache_name("bicubic", 2),
        dc.cache_name("bicubic", 4),
    }
    assert len(names) == 4
    assert all(f"v{dc.IMRESIZE_VERSION}" in n for n in names)


def test_checksum_changes_with_kind_scale_and_imresize_version(image_paths, monkeypatch):
    """Contrast with the raw-HR checksum, which deliberately ignores all three."""
    base = dc.compute_checksum(image_paths, "lr", 4)
    assert dc.compute_checksum(image_paths, "bicubic", 4) != base
    assert dc.compute_checksum(image_paths, "lr", 3) != base

    monkeypatch.setattr(dc, "IMRESIZE_VERSION", dc.IMRESIZE_VERSION + 1)
    assert dc.compute_checksum(image_paths, "lr", 4) != base


def test_checksum_ignores_content_at_equal_name_and_size(image_paths):
    """Same accepted property as the raw-HR cache, stated here so a future edit
    does not silently make one stricter than the other."""
    before = dc.compute_checksum(image_paths, "lr", 4)
    target = image_paths[0]
    size = target.stat().st_size
    rng = np.random.default_rng(seed=99)
    while True:  # write a different image of identical byte length
        arr = rng.integers(0, 256, size=(36, 36, 3), dtype=np.uint8)
        Image.fromarray(arr).save(target)
        if target.stat().st_size == size:
            break
    assert dc.compute_checksum(image_paths, "lr", 4) == before


def test_estimate_map_size_is_floored_at_64_mib():
    assert dc.estimate_map_size([(8, 8)], "lr", 4) == 64 * 1024 * 1024


def test_estimate_map_size_covers_the_real_payload(image_paths):
    sizes = [(321, 481)] * 400
    est = dc.estimate_map_size(sizes, "bicubic", 3)
    payload = sum(
        dc.HEADER.size + h * w * 3 for h, w in (dc.derived_shape(*s, "bicubic", 3) for s in sizes)
    )
    assert est >= payload


def test_module_imports_no_torch():
    """``process_derived_image`` is what gets pickled to build workers, and a
    spawned worker re-imports *this* module — so torch must not ride along."""
    probe = "import sisr.datasets.derived_cache, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_process_derived_image_round_trips_through_its_header(image_paths):
    """The entry must be self-describing: shape recoverable from the value alone."""
    [(key, value)] = dc.process_derived_image(image_paths[1], 7, "lr", 3)
    assert key == "lr_00000007"
    h, w = dc.HEADER.unpack_from(value, 0)
    plane = np.frombuffer(value, np.uint8, offset=dc.HEADER.size).reshape(h, w, 3)
    expected = dc.derive(np.array(Image.open(image_paths[1]).convert("RGB")), "lr", 3)
    assert np.array_equal(plane, expected)


# ---------------------------------------------------------------------------
# Byte-equality against externally generated reference data
#
# `derive` is now the single owner of the degradation the *training* path
# applies, so the convention it implements is checkable against data this
# project did not produce. Both checks skip cleanly when their data is absent —
# **a skip means the leg was not exercised, never that it passed.**
# ---------------------------------------------------------------------------

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"
DIV2K_HR_DIR = Path(__file__).resolve().parents[2] / "data" / "DIV2K_train_HR"
DIV2K_LR_DIR = Path(__file__).resolve().parents[2] / "data" / "DIV2K_train_LR_bicubic" / "X4"


def test_derive_matches_real_matlab_reference_lr_byte_for_byte():
    """The training path's degradation, against MATLAB-generated benchmark LR.

    ``tests/utils/test_imresize.py`` already pins the *kernel* against this
    data. This pins the **order** — modcrop the whole image, then downsample it
    — which is the axis the kernel test cannot see, because it was written when
    the training path degraded an extracted patch instead.
    """
    if not REFERENCE_DIR.is_dir():
        pytest.skip(
            f"{REFERENCE_DIR} not present -- see tests/utils/test_imresize.py's module "
            "docstring for how to fetch it."
        )
    cases = []
    for dataset in ("Set5", "Set14", "B100"):
        hr_dir = REFERENCE_DIR / dataset / "HR"
        if not hr_dir.is_dir():
            continue
        for scale in (2, 3, 4):
            for hr_path in sorted(hr_dir.glob("*.png")):
                lr_path = (
                    REFERENCE_DIR
                    / dataset
                    / "LR_bicubic"
                    / f"X{scale}"
                    / (f"{hr_path.stem}x{scale}.png")
                )
                if lr_path.exists():
                    cases.append((hr_path, lr_path, scale))
    assert cases, f"{REFERENCE_DIR} exists but holds no HR/LR_bicubic pairs"

    mismatches = []
    for hr_path, lr_path, scale in cases:
        hr = np.array(Image.open(hr_path).convert("RGB"))
        real = np.array(Image.open(lr_path).convert("RGB"))
        got = dc.derive(hr, "lr", scale)
        if got.shape != real.shape or not np.array_equal(got, real):
            mismatches.append((hr_path.name, scale))
    assert not mismatches, f"{len(mismatches)}/{len(cases)} differ: {mismatches[:5]}"


def test_derive_matches_div2ks_own_distributed_lr_byte_for_byte():
    """The same check on the actual training set, at the actual training scale.

    DIV2K distributes its bicubic LR as full images; every reference pipeline
    surveyed trains from those. If ours is byte-identical to them, then the
    order this project adopted is not merely *like* the field's — it reproduces
    the field's own artifact exactly.
    """
    if not DIV2K_LR_DIR.is_dir():
        pytest.skip(
            f"{DIV2K_LR_DIR} not present -- fetch DIV2K_train_LR_bicubic_X4.zip from the "
            "DIV2K dataset page and unzip it under data/ to enable this check."
        )
    hr_paths = sorted(DIV2K_HR_DIR.glob("*.png"))
    assert hr_paths, f"{DIV2K_HR_DIR} holds no images"

    mismatches, checked = [], 0
    for hr_path in hr_paths:
        lr_path = DIV2K_LR_DIR / f"{hr_path.stem}x4.png"
        if not lr_path.exists():
            continue
        hr = np.array(Image.open(hr_path).convert("RGB"))
        real = np.array(Image.open(lr_path).convert("RGB"))
        got = dc.derive(hr, "lr", 4)
        checked += 1
        if got.shape != real.shape or not np.array_equal(got, real):
            mismatches.append(hr_path.name)
    assert checked, "no HR/LR filename pairs matched -- check the x4 naming convention"
    assert not mismatches, f"{len(mismatches)}/{checked} differ: {mismatches[:5]}"
